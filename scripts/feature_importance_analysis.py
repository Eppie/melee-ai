"""Feature importance analysis using real validation data"""
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

from model.nano_gpt import GPT
from config import Config
from column_map import ColumnMap
from schema import get_feature_names, get_target_names
from window_dataset import WindowDataset
from train.batch_utils import build_model_inputs


def load_model(checkpoint_path: str | Path):
    """Load model from checkpoint, handling compiled model prefix."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = Config.model_validate(checkpoint["config"])
    model = GPT(config)

    # Strip _orig_mod. prefix from state dict (saved from compiled model)
    state_dict = checkpoint["model"]
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            cleaned_state_dict[k[len("_orig_mod."):]] = v
        else:
            cleaned_state_dict[k] = v

    model.load_state_dict(cleaned_state_dict)
    model.eval()
    model.to("cpu")  # Ensure CPU for gradient analysis
    return model, config


def categorize_feature(name: str) -> str:
    """Categorize a feature name into semantic groups."""
    if name == "stage":
        return "stage"
    if "_character" in name:
        return "character"
    if "_action" in name:
        return "action"
    if "position_x" in name or "position_y" in name:
        return "position"
    if "speed" in name:
        return "speed"
    if "percent" in name:
        return "percent"
    if "facing" in name:
        return "facing"
    if "on_ground" in name:
        return "on_ground"
    if "jumps" in name:
        return "jumps"
    if "is_" in name:
        return "boolean_flags"
    if "main_stick" in name or "c_stick" in name:
        return "stick_inputs"
    if "button" in name:
        return "buttons"
    if "shoulder" in name:
        return "shoulder"
    return "other"


def main():
    # Load specific checkpoint
    checkpoint_path = Path("checkpoints/model_ep029_005001.pt")
    print(f"Loading checkpoint: {checkpoint_path}")
    model, config = load_model(checkpoint_path)
    device = torch.device("cpu")

    # Get feature names from dataset
    # Use local path (checkpoint may have Linux paths)
    val_root = "validation_set"
    val_dataset = WindowDataset(val_root)

    feature_names = val_dataset.index.feature_names
    target_names = val_dataset.index.target_names

    # Create column map from feature/target names
    column_map = ColumnMap(feature_names, target_names)

    print(f"Validation dataset has {len(val_dataset)} windows")
    print(f"Feature names: {len(feature_names)} total")
    print(f"Gamestate features: {len(column_map.gamestate_idxs)}")
    print(f"Controller features: {len(column_map.controller_idxs)}")

    # Sample multiple windows and compute gradient importance
    n_samples = 100
    importance_accum = defaultdict(list)

    print(f"\nAnalyzing {n_samples} samples for gradient-based importance...")

    for i in range(min(n_samples, len(val_dataset))):
        try:
            # Sample spread throughout dataset
            idx = (i * 100) % len(val_dataset)
            sample = val_dataset[idx]
            inputs = sample["X"].unsqueeze(0)  # [1, seq, features]
            inputs.requires_grad_(True)

            # Build model inputs
            inputs_td = build_model_inputs(inputs, column_map)

            # Forward pass
            outputs = model(inputs_td)

            # Compute gradient w.r.t. main stick prediction
            main_logits = outputs["main_stick"]  # Key is "main_stick", not "main_stick_logits"
            probs = torch.softmax(main_logits, dim=-1)

            # Use sum of max probs across sequence as objective
            max_probs = probs.max(dim=-1).values.sum()
            max_probs.backward()

            # Extract gradients (average over sequence)
            if inputs.grad is not None:
                grads = inputs.grad[0].abs().mean(dim=0)  # [features]

                for j, name in enumerate(feature_names):
                    if j < len(grads):
                        importance_accum[name].append(grads[j].item())

            # Clean up
            inputs.grad = None
            model.zero_grad()

            if (i + 1) % 20 == 0:
                print(f"  Processed {i+1}/{n_samples} samples")

        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            continue

    # Compute average importance per feature
    avg_importance = {}
    for name, values in importance_accum.items():
        avg_importance[name] = np.mean(values)

    # Sort by importance
    sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)

    print("\n" + "="*70)
    print("GRADIENT-BASED FEATURE IMPORTANCE (averaged over validation samples)")
    print("="*70)

    print("\nTOP 30 MOST IMPORTANT FEATURES:")
    for i, (name, imp) in enumerate(sorted_features[:30]):
        print(f"  {i+1:2d}. {name:45s}: {imp:.6f}")

    print("\nBOTTOM 20 LEAST IMPORTANT FEATURES:")
    for i, (name, imp) in enumerate(sorted_features[-20:]):
        print(f"  {i+1:2d}. {name:45s}: {imp:.6f}")

    # Group by category
    print("\n" + "="*70)
    print("IMPORTANCE BY FEATURE CATEGORY")
    print("="*70)

    category_importance = defaultdict(list)
    for name, imp in avg_importance.items():
        cat = categorize_feature(name)
        category_importance[cat].append((name, imp))

    # Print category totals
    print("\nCategory importance (total gradient magnitude):")
    cat_totals = [(cat, sum(imp for _, imp in feats), feats) for cat, feats in category_importance.items()]
    cat_totals.sort(key=lambda x: x[1], reverse=True)

    for cat, total, feats in cat_totals:
        n_feats = len(feats)
        avg = total / n_feats if n_feats > 0 else 0
        print(f"\n  {cat.upper()} (n={n_feats}, total={total:.5f}, avg={avg:.5f}):")
        # Show top features in this category
        feats_sorted = sorted(feats, key=lambda x: x[1], reverse=True)[:5]
        for name, imp in feats_sorted:
            print(f"    - {name}: {imp:.6f}")


if __name__ == "__main__":
    main()
