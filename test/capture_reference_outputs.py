"""
Script to capture exact model outputs for test assertions.

This generates the reference outputs that we'll use to verify
the KV cache produces identical results.
"""
from pathlib import Path
import torch
import numpy as np

from config import get_config, init_config, reset_config
from model.nano_gpt import GPT
from schema import get_feature_names, get_raw_target_names
from train.batch_utils import build_model_inputs
from column_map import ColumnMap
from utils import _resolve_device, match_state_dict_keys
from zarr_storage import Schema, _process_episode_task


def main():
    # Initialize config
    reset_config()
    init_config(freeze=False)

    # Paths
    test_slp_path = Path(__file__).with_name("test.slp")
    checkpoint_path = Path(__file__).parent.parent / "checkpoints" / "model_ep013_000001.pt"

    # Load test data
    print("Loading test data from test.slp...")
    schema = Schema(features=get_feature_names(), targets=get_raw_target_names())
    episodes = _process_episode_task(str(test_slp_path), schema)
    episode = episodes[0]  # First episode

    features = torch.from_numpy(episode.features).float()
    targets = torch.from_numpy(episode.targets).float()
    feature_names = episode.feature_names
    target_names = episode.target_names

    print(f"Loaded {features.shape[0]} frames")

    # Load model
    print(f"Loading model from {checkpoint_path}...")
    device = _resolve_device()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    config = get_config()
    ckpt_cfg = ckpt.get("config")
    if isinstance(ckpt_cfg, dict):
        model_cfg = ckpt_cfg.get("model", {})
        for field, value in model_cfg.items():
            setattr(config.model, field, value)

    model = GPT(config).to(device)
    model_state = match_state_dict_keys(ckpt["model"], model)
    model.load_state_dict(model_state, strict=False)
    model.eval()

    print(f"Model loaded. Config: use_alibi={config.model.use_alibi}, head_flow={config.model.head_flow}")

    # Create column map
    colmap = ColumnMap(feature_names, target_names)

    # Test 1: Single forward pass
    print("\n" + "="*80)
    print("TEST 1: Single forward pass (64 frames)")
    print("="*80)
    seq_len = 64
    batch_X = features[:seq_len].unsqueeze(0)
    horizon_feature = torch.full((1, seq_len, 1), 30.0 / 60.0, dtype=batch_X.dtype)
    batch_X_with_horizon = torch.cat([batch_X, horizon_feature], dim=-1).to(device)
    inputs = build_model_inputs(batch_X_with_horizon, colmap)

    with torch.inference_mode():
        outputs, _ = model(inputs, use_cache=False)

    # Print last frame outputs
    print("\nOutputs for last frame (index -1):")
    for key in ["main_stick", "c_stick", "buttons", "shoulder", "value"]:
        output = outputs[key][0, -1].cpu().numpy()
        print(f"\n{key}:")
        print(f"  shape: {output.shape}")
        print(f"  first 5 values: {output[:5]}")
        print(f"  last 5 values: {output[-5:]}")
        print(f"  mean: {output.mean():.6f}, std: {output.std():.6f}")
        print(f"  min: {output.min():.6f}, max: {output.max():.6f}")

    # Test 2: Autoregressive generation (3 steps to keep it manageable)
    print("\n" + "="*80)
    print("TEST 2: Autoregressive generation (3 steps)")
    print("="*80)

    context_len = 32
    num_steps = 3

    for step in range(num_steps):
        seq_len = context_len + step
        batch_X = features[:seq_len].unsqueeze(0)
        horizon_feature = torch.full((1, seq_len, 1), 30.0 / 60.0, dtype=batch_X.dtype)
        batch_X_with_horizon = torch.cat([batch_X, horizon_feature], dim=-1).to(device)
        inputs = build_model_inputs(batch_X_with_horizon, colmap)

        with torch.inference_mode():
            outputs, _ = model(inputs, use_cache=False)

        print(f"\nStep {step} (seq_len={seq_len}):")
        for key in ["main_stick", "c_stick", "buttons", "shoulder", "value"]:
            output = outputs[key][0, -1].cpu().numpy()
            print(f"  {key}: first 3={output[:3]}, argmax={output.argmax() if len(output) > 1 else 'N/A'}")

    # Test 3: Save exact values for a few key frames for assertion
    print("\n" + "="*80)
    print("TEST 3: Exact values for test assertions")
    print("="*80)

    # Run with 64 frames
    seq_len = 64
    batch_X = features[:seq_len].unsqueeze(0)
    horizon_feature = torch.full((1, seq_len, 1), 30.0 / 60.0, dtype=batch_X.dtype)
    batch_X_with_horizon = torch.cat([batch_X, horizon_feature], dim=-1).to(device)
    inputs = build_model_inputs(batch_X_with_horizon, colmap)

    with torch.inference_mode():
        outputs, _ = model(inputs, use_cache=False)

    # Save numpy arrays for specific frames
    test_frames = [0, 31, 63]  # first, middle, last

    print("\nPython code for test assertions:")
    print("-" * 80)

    for frame_idx in test_frames:
        print(f"\n# Frame {frame_idx}")
        for key in ["main_stick", "c_stick", "buttons", "shoulder", "value"]:
            output = outputs[key][0, frame_idx].cpu().numpy()
            # Convert to Python list for easy copy-paste
            values_list = output.tolist()
            print(f"EXPECTED_{key.upper()}_FRAME_{frame_idx} = np.array({values_list})")

    print("\n" + "="*80)
    print("Done! Copy the arrays above into your test file.")
    print("="*80)


if __name__ == "__main__":
    main()
