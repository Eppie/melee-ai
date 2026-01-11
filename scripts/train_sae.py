#!/usr/bin/env python3
"""
Train Sparse Autoencoders (SAEs) on transformer residual stream activations.

Usage:
    # Train SAE on layer 4 (middle layer)
    python scripts/train_sae.py --checkpoint checkpoints/model.pt --layer 4

    # Train on all layers
    python scripts/train_sae.py --checkpoint checkpoints/model.pt --layer all

    # Train on MLP activations (4x dimension)
    python scripts/train_sae.py --checkpoint checkpoints/model.pt --hook-type mlp_post_act --layer 3

    # Custom hyperparameters
    python scripts/train_sae.py --checkpoint checkpoints/model.pt \
        --expansion-factor 16 --k 64 --training-steps 50000

    # With stratified sampling (oversample high-loss frames)
    python scripts/train_sae.py --checkpoint checkpoints/model.pt --stratified
"""

from __future__ import annotations

# Suppress warnings that clutter output
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", message=".*pin_memory.*MPS.*")

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import torch

from column_map import ColumnMap
from config import Config, get_config, init_config_from_checkpoint
from interp import (
    HookPoint,
    HookPointType,
    SAEConfig,
    TopKSparseAutoencoder,
    train_sae,
)
from model.nano_gpt import GPT
from train.checkpoint import load_config_from_checkpoint
from utils import _resolve_device, match_state_dict_keys
from window_dataset import make_dataloader


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Sparse Autoencoders on model activations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint (.pt file)",
    )

    # Hook point configuration
    parser.add_argument(
        "--layer",
        type=str,
        default="middle",
        help="Layer index to train SAE on. Options: integer, 'all', 'middle', 'first', 'last' (default: middle)",
    )
    parser.add_argument(
        "--hook-type",
        type=str,
        default="block_output",
        choices=["block_output", "mlp_post_act", "final_norm"],
        help="Type of activations to capture (default: block_output)",
    )

    # SAE architecture
    parser.add_argument(
        "--expansion-factor",
        type=int,
        default=8,
        help="Ratio of SAE features to input dimension (default: 8)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=32,
        help="Number of active features per input (TopK sparsity, default: 32)",
    )

    # Training
    parser.add_argument(
        "--training-steps",
        type=int,
        default=20000,
        help="Number of training steps (default: 20000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Batch size for SAE training (default: 4096)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--max-activation-samples",
        type=int,
        default=200000,
        help="Maximum activation samples to cache (default: 200000)",
    )

    # Sampling strategy
    parser.add_argument(
        "--stratified",
        action="store_true",
        help="Use loss-stratified sampling (oversample uncertain frames)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sae_checkpoints"),
        help="Directory to save trained SAEs (default: sae_checkpoints/)",
    )

    # Data configuration
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override data directory (default: use checkpoint config)",
    )
    parser.add_argument(
        "--dataloader-batch-size",
        type=int,
        default=32,
        help="Batch size for data loading (default: 32)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers (default: 4)",
    )

    return parser.parse_args()


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    overrides: Optional[dict] = None,
) -> tuple[GPT, Config]:
    """Load model and config from checkpoint."""
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Initialize config from checkpoint with optional overrides
    config = init_config_from_checkpoint(checkpoint_path, overrides=overrides)

    # Create model
    model = GPT(config)

    # Load model weights
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model_state = ckpt.get("model")
    if model_state is None:
        print("Error: Checkpoint does not contain model state.")
        sys.exit(1)

    # Handle torch.compile prefix mismatch
    model_state = match_state_dict_keys(model_state, model)
    model.load_state_dict(model_state)

    model = model.to(device)
    model.eval()

    return model, config


def resolve_layers(layer_arg: str, n_layers: int) -> List[int]:
    """Resolve layer argument to list of layer indices."""
    if layer_arg == "all":
        return list(range(n_layers))
    elif layer_arg == "middle":
        return [n_layers // 2]
    elif layer_arg == "first":
        return [0]
    elif layer_arg == "last":
        return [n_layers - 1]
    else:
        try:
            layer_idx = int(layer_arg)
            if layer_idx < 0 or layer_idx >= n_layers:
                print(f"Error: Layer {layer_idx} out of range [0, {n_layers - 1}]")
                sys.exit(1)
            return [layer_idx]
        except ValueError:
            print(f"Error: Invalid layer argument: {layer_arg}")
            print("Expected: integer, 'all', 'middle', 'first', or 'last'")
            sys.exit(1)


def get_hook_point(hook_type: str, layer_idx: Optional[int]) -> HookPoint:
    """Create a HookPoint from type string and layer index."""
    type_map = {
        "block_output": HookPointType.BLOCK_OUTPUT,
        "mlp_post_act": HookPointType.MLP_POST_ACT,
        "final_norm": HookPointType.FINAL_NORM,
    }

    hook_point_type = type_map[hook_type]

    if hook_type == "final_norm":
        return HookPoint(hook_point_type, layer_idx=None)
    else:
        if layer_idx is None:
            print(f"Error: Hook type '{hook_type}' requires a layer index")
            sys.exit(1)
        return HookPoint(hook_point_type, layer_idx=layer_idx)


def train_single_sae(
    model: GPT,
    dataloader,
    colmap: ColumnMap,
    hook_point: HookPoint,
    config: SAEConfig,
    device: torch.device,
    output_dir: Path,
    max_activation_samples: int,
    stratified: bool,
) -> tuple[TopKSparseAutoencoder, dict]:
    """Train a single SAE and save it."""
    print(f"\n{'=' * 60}")
    print(f"Training SAE at {hook_point}")
    print(f"{'=' * 60}")

    # Create checkpoint directory for this SAE
    checkpoint_dir = output_dir / str(hook_point).replace("/", "_")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Train SAE
    sae, history = train_sae(
        model=model,
        dataloader=dataloader,
        colmap=colmap,
        hook_point=hook_point,
        config=config,
        device=device,
        max_activation_samples=max_activation_samples,
        stratified=stratified,
        show_progress=True,
        checkpoint_dir=checkpoint_dir,
    )

    # Save metadata
    metadata = {
        "hook_point": str(hook_point),
        "hook_type": hook_point.hook_type.value,
        "layer_idx": hook_point.layer_idx,
        "input_dim": sae.input_dim,
        "hidden_dim": sae.hidden_dim,
        "expansion_factor": config.expansion_factor,
        "k": config.k,
        "training_steps": config.training_steps,
        "batch_size": config.batch_size,
        "lr": config.lr,
        "stratified_sampling": stratified,
        "max_activation_samples": max_activation_samples,
        "final_loss": history.losses[-1] if history.losses else None,
        "num_dead_features": len(history.final_dead_features),
        "dead_feature_indices": history.final_dead_features,
    }

    metadata_path = checkpoint_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSAE saved to: {checkpoint_dir}")
    print(f"  Final loss: {metadata['final_loss']:.6f}")
    print(f"  Dead features: {metadata['num_dead_features']} / {sae.hidden_dim}")

    return sae, metadata


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Resolve device
    device = _resolve_device(None)
    print(f"Using device: {device}")

    # Build config overrides
    overrides = {
        "train.batch_size": str(args.dataloader_batch_size),
        "train.num_workers": str(args.num_workers),
    }
    if args.data_dir is not None:
        # Note: ZarrConfig has a validator that appends _{episode_count} to out_root
        # We need to set the path WITHOUT the suffix and set episode_count to match
        import re
        data_dir_str = str(args.data_dir)
        # Extract episode count from path if present (e.g., "processed_data_1" -> 1)
        match = re.search(r"_(\d+)$", data_dir_str)
        if match:
            episode_count = int(match.group(1))
            base_path = data_dir_str[: match.start()]  # Remove the _N suffix
            overrides["zarr.out_root"] = base_path
            overrides["zarr.episode_count"] = str(episode_count)
        else:
            overrides["zarr.out_root"] = data_dir_str

    # Load model with all overrides
    print(f"\nLoading model from: {args.checkpoint}")
    model, config = load_model_from_checkpoint(
        args.checkpoint,
        device,
        overrides=overrides,
    )
    n_layers = config.model.n_layer
    print(f"Model loaded: {n_layers} layers, {config.model.n_embd} embedding dim")

    # Resolve layers to train
    layers = resolve_layers(args.layer, n_layers)
    print(f"Training SAEs on layers: {layers}")

    # Create SAE config
    sae_config = SAEConfig(
        expansion_factor=args.expansion_factor,
        k=args.k,
        training_steps=args.training_steps,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    print(f"\nSAE config:")
    print(f"  Expansion factor: {sae_config.expansion_factor}")
    print(f"  TopK: {sae_config.k}")
    print(f"  Training steps: {sae_config.training_steps}")
    print(f"  Batch size: {sae_config.batch_size}")
    print(f"  Learning rate: {sae_config.lr}")

    # Create dataloader
    print(f"\nCreating dataloader from: {config.zarr.out_root}")
    loader, ds, _ = make_dataloader(config)
    colmap = ColumnMap.from_dataset(ds)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Train SAEs
    all_metadata = {}
    for layer_idx in layers:
        hook_point = get_hook_point(args.hook_type, layer_idx)

        sae, metadata = train_single_sae(
            model=model,
            dataloader=loader,
            colmap=colmap,
            hook_point=hook_point,
            config=sae_config,
            device=device,
            output_dir=args.output_dir,
            max_activation_samples=args.max_activation_samples,
            stratified=args.stratified,
        )
        all_metadata[str(hook_point)] = metadata

    # Save summary
    summary_path = args.output_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "checkpoint": str(args.checkpoint),
                "hook_type": args.hook_type,
                "layers": layers,
                "sae_config": sae_config.model_dump(),
                "stratified": args.stratified,
                "saes": all_metadata,
            },
            f,
            indent=2,
        )

    print(f"\n{'=' * 60}")
    print(f"Training complete!")
    print(f"{'=' * 60}")
    print(f"Output directory: {args.output_dir}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
