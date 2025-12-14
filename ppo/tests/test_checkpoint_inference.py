"""Test that PPO checkpoints produce valid outputs for inference."""

import sys
from pathlib import Path

import numpy as np
import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import Config
from controller_utils import (
    CONTROL_STICK_QUANTIZED,
    C_STICK_QUANTIZED,
    SHOULDER_QUANTIZED,
)
from model.nano_gpt import GPT
from schema import get_feature_names, get_target_names
from column_map import ColumnMap
from train.batch_utils import build_model_inputs


def load_checkpoint(checkpoint_path: Path):
    """Load checkpoint and return model."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Load config
    if "config" in ckpt:
        config_dict = ckpt["config"]
        config = Config.model_validate(config_dict)
    else:
        config = Config()

    # Create model
    model = GPT(config)

    # Load weights
    if "model" in ckpt:
        state_dict = ckpt["model"]
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model, config


def test_output_sanity(checkpoint_path: Path):
    """Test that checkpoint produces sane outputs."""
    print(f"\n{'='*80}")
    print(f"Testing checkpoint: {checkpoint_path.name}")
    print(f"{'='*80}")

    model, config = load_checkpoint(checkpoint_path)

    # Create dummy input (batch=1, seq_len=256, features)
    feature_names = get_feature_names()
    target_names = get_target_names()
    n_features = len(feature_names)
    seq_len = 256

    # Create column map to properly structure inputs
    colmap = ColumnMap(feature_names, target_names)

    # Random features in reasonable range [-1, 1]
    features = torch.randn(1, seq_len, n_features) * 0.5

    # Set categorical features to valid integer values
    features[..., colmap.stage_idx] = 0  # FD
    features[..., colmap.ego_char_idx] = 1  # Fox
    features[..., colmap.opp_char_idx] = 1  # Fox
    features[..., colmap.ego_action_idx] = 0  # Some action
    features[..., colmap.opp_action_idx] = 0  # Some action

    # Append horizon feature (model was trained with this)
    horizon = torch.full((1, seq_len, 1), 0.5)
    features_with_horizon = torch.cat([features, horizon], dim=-1)

    # Build proper model inputs
    inputs = build_model_inputs(features_with_horizon, colmap)

    # Run forward pass
    with torch.inference_mode():
        outputs = model(inputs)

    print(f"\n📊 Model Architecture:")
    print(f"   Input features: {n_features}")
    print(f"   Sequence length: {seq_len}")
    print(f"   Output heads: {list(outputs.keys())}")

    # Check each output head
    print(f"\n📈 Output Statistics:")

    # Main stick
    main_logits = outputs["main_stick"][0, -1]  # Last timestep
    print(f"\n  Main Stick (64 positions):")
    print(f"    Shape: {main_logits.shape}")
    print(f"    Min/Max: {main_logits.min():.3f} / {main_logits.max():.3f}")
    print(f"    Mean/Std: {main_logits.mean():.3f} / {main_logits.std():.3f}")
    print(f"    Has NaN: {torch.isnan(main_logits).any()}")
    print(f"    Has Inf: {torch.isinf(main_logits).any()}")

    # Check if distribution is reasonable
    probs = torch.softmax(main_logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-10)).sum()
    max_prob_idx = torch.argmax(probs)
    max_prob = probs[max_prob_idx]
    print(f"    Entropy: {entropy:.3f} (max: {np.log(64):.3f})")
    print(f"    Top prediction: idx={max_prob_idx} ({max_prob:.1%})")
    print(f"    Top 5 probs: {probs.topk(5).values.tolist()}")

    # Check if argmax is in valid range
    main_idx = int(torch.argmax(main_logits).item())
    if not (0 <= main_idx < 64):
        print(f"    ❌ ERROR: Invalid main_idx={main_idx} (should be 0-63)")
    else:
        main_xy = CONTROL_STICK_QUANTIZED[main_idx]
        print(
            f"    ✓ Decoded position: idx={main_idx} → ({main_xy[0]:.3f}, {main_xy[1]:.3f})"
        )

    # C-stick
    c_logits = outputs["c_stick"][0, -1]
    print(f"\n  C-Stick (9 positions):")
    print(f"    Shape: {c_logits.shape}")
    print(f"    Min/Max: {c_logits.min():.3f} / {c_logits.max():.3f}")
    print(f"    Mean/Std: {c_logits.mean():.3f} / {c_logits.std():.3f}")
    print(f"    Has NaN: {torch.isnan(c_logits).any()}")

    c_idx = int(torch.argmax(c_logits).item())
    if not (0 <= c_idx < 9):
        print(f"    ❌ ERROR: Invalid c_idx={c_idx} (should be 0-8)")
    else:
        c_xy = C_STICK_QUANTIZED[c_idx]
        print(f"    ✓ Decoded position: idx={c_idx} → ({c_xy[0]:.3f}, {c_xy[1]:.3f})")

    # Shoulder
    shoulder_logits = outputs["shoulder"][0, -1]
    print(f"\n  Shoulder (5 levels):")
    print(f"    Shape: {shoulder_logits.shape}")
    print(f"    Min/Max: {shoulder_logits.min():.3f} / {shoulder_logits.max():.3f}")
    print(f"    Mean/Std: {shoulder_logits.mean():.3f} / {shoulder_logits.std():.3f}")
    print(f"    Has NaN: {torch.isnan(shoulder_logits).any()}")

    s_idx = int(torch.argmax(shoulder_logits).item())
    if not (0 <= s_idx < 5):
        print(f"    ❌ ERROR: Invalid shoulder_idx={s_idx} (should be 0-4)")
    else:
        s_val = SHOULDER_QUANTIZED[s_idx]
        print(f"    ✓ Decoded value: idx={s_idx} → {s_val:.3f}")

    # Buttons
    button_logits = outputs["buttons"][0, -1]
    print(f"\n  Buttons (5 buttons):")
    print(f"    Shape: {button_logits.shape}")
    print(f"    Logits: {button_logits.tolist()}")
    print(f"    Has NaN: {torch.isnan(button_logits).any()}")

    button_probs = torch.sigmoid(button_logits)
    print(f"    Probabilities: {button_probs.tolist()}")
    print(f"    Predictions: {(button_probs > 0.5).tolist()}")

    # Check for pathological cases
    print(f"\n🔍 Pathological Checks:")

    issues = []

    # Check for NaN/Inf anywhere
    for name, tensor in outputs.items():
        if torch.isnan(tensor).any():
            issues.append(f"NaN in {name}")
        if torch.isinf(tensor).any():
            issues.append(f"Inf in {name}")

    # Check if all outputs are same (dead network)
    if torch.allclose(main_logits, main_logits[0]):
        issues.append("All main_stick logits are identical (dead network?)")

    # Check if probabilities are degenerate
    if max_prob > 0.99:
        issues.append(f"Overconfident: main stick prob={max_prob:.1%}")
    elif max_prob < 0.05:
        issues.append(f"Underconfident: main stick prob={max_prob:.1%}")

    # Check button probabilities
    if (button_probs > 0.9).sum() >= 4:
        issues.append(f"Spamming buttons: {(button_probs > 0.9).sum()}/5 buttons > 90%")

    if issues:
        print(f"  ❌ Found {len(issues)} issues:")
        for issue in issues:
            print(f"     - {issue}")
    else:
        print(f"  ✓ No pathological behavior detected")

    print(f"\n{'='*80}\n")

    return len(issues) == 0


def compare_checkpoints(imitation_ckpt: Path, ppo_ckpt: Path):
    """Compare outputs from imitation vs PPO checkpoint."""
    print(f"\n🔬 Comparing Imitation vs PPO Checkpoints")
    print(f"   Imitation: {imitation_ckpt}")
    print(f"   PPO:       {ppo_ckpt}")

    model_im, _ = load_checkpoint(imitation_ckpt)
    model_ppo, _ = load_checkpoint(ppo_ckpt)

    # Same random input
    torch.manual_seed(42)
    feature_names = get_feature_names()
    target_names = get_target_names()
    colmap = ColumnMap(feature_names, target_names)

    features = torch.randn(1, 256, len(feature_names)) * 0.5
    features[..., colmap.stage_idx] = 0
    features[..., colmap.ego_char_idx] = 1
    features[..., colmap.opp_char_idx] = 1
    features[..., colmap.ego_action_idx] = 0
    features[..., colmap.opp_action_idx] = 0

    # Append horizon
    horizon = torch.full((1, 256, 1), 0.5)
    features_with_horizon = torch.cat([features, horizon], dim=-1)

    inputs = build_model_inputs(features_with_horizon, colmap)

    with torch.inference_mode():
        out_im = model_im(inputs)
        out_ppo = model_ppo(inputs)

    print(f"\n📊 Output Comparison:")

    for key in ["main_stick", "c_stick", "shoulder", "buttons"]:
        diff = (out_im[key][0, -1] - out_ppo[key][0, -1]).abs()
        print(f"\n  {key}:")
        print(f"    Mean diff: {diff.mean():.4f}")
        print(f"    Max diff:  {diff.max():.4f}")
        print(f"    Imitation argmax: {torch.argmax(out_im[key][0, -1]).item()}")
        print(f"    PPO argmax:       {torch.argmax(out_ppo[key][0, -1]).item()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ppo-checkpoint", type=Path, required=True)
    parser.add_argument("--imitation-checkpoint", type=Path, default=None)
    args = parser.parse_args()

    # Test PPO checkpoint
    is_sane = test_output_sanity(args.ppo_checkpoint)

    # Compare if imitation checkpoint provided
    if args.imitation_checkpoint:
        print("\n" + "=" * 80)
        test_output_sanity(args.imitation_checkpoint)
        compare_checkpoints(args.imitation_checkpoint, args.ppo_checkpoint)

    sys.exit(0 if is_sane else 1)
