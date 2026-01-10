#!/usr/bin/env python3
"""Persona vectors for behavioral steering of the Melee AI.

Implements contrastive activation steering:
1. Label windows as "aggressive" vs "passive" using game state features
2. Collect residual stream activations from each transformer layer
3. Compute steering vector: v_l = mean(h_l | aggressive) - mean(h_l | passive)
4. At inference, inject: hidden_states[:, -1, :] += alpha * v_l

Usage:
    python scripts/persona_vectors.py                     # Full pipeline
    python scripts/persona_vectors.py --collect-only      # Just collect and compute vectors
    python scripts/persona_vectors.py --max-batches 500   # Limit collection batches
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from itertools import islice
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from column_map import ColumnMap
from config import get_config, init_config
from model.nano_gpt import GPT, BUTTON_NAMES
from model.norm import norm
from train import find_latest_checkpoint
from train.batch_utils import build_model_inputs
from train.validation import _get_validation_loader
from utils import _resolve_device, match_state_dict_keys


# -----------------------------------------------------------------------------
# 1. Activation Collection
# -----------------------------------------------------------------------------


class ActivationCollector:
    """Hooks into model blocks to collect residual stream activations."""

    def __init__(self, model: GPT):
        self.model = model
        self.activations: Dict[int, List[Tensor]] = defaultdict(list)
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

    def register_hooks(self) -> None:
        """Register forward hooks on all transformer blocks."""
        for layer_idx, block in enumerate(self.model.blocks):
            hook = block.register_forward_hook(self._make_hook(layer_idx))
            self.hooks.append(hook)

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            # output: [B, L, 512] - take last position for decision point
            self.activations[layer_idx].append(output[:, -1, :].detach().cpu())

        return hook_fn

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def clear(self) -> None:
        """Clear collected activations."""
        self.activations.clear()


# -----------------------------------------------------------------------------
# 2. Window Labeling - Improved with action states and sequence patterns
# -----------------------------------------------------------------------------

# Action state categories (values from melee/enums.py)
# High commitment attacks - these are proactive, aggressive decisions
HARD_COMMITS = {
    0x3C, 0x3D, 0x3E,  # FSMASH variants
    0x3F,              # UPSMASH
    0x40,              # DOWNSMASH
    0xD4,              # GRAB (standing)
    0xD6,              # GRAB_RUNNING
    0x45,              # DAIR (spikes are committal)
    0x164,             # FIREFOX_AIR (recovery commitment)
    0x163,             # FIREFOX_GROUND
    0x170,             # UP_B_AIR (recovery/attack)
    0x16F,             # UP_B_GROUND
}

# Medium commitment - still aggressive but safer
MEDIUM_COMMITS = {
    0x41, 0x42, 0x43, 0x44,  # NAIR, FAIR, BAIR, UAIR
    0x38,              # UPTILT
    0x39,              # DOWNTILT
    0x33, 0x34, 0x35, 0x36, 0x37,  # FTILT variants
    0x32,              # DASH_ATTACK
    0x169,             # DOWN_B_GROUND (shine)
    0x16E,             # DOWN_B_AIR (shine)
    0x15E, 0x15F, 0x160,  # FOX_ILLUSION variants
    0x2C, 0x2D, 0x2E,  # JAB 1, 2, 3
}

# Low commitment - probing/positioning
SOFT_COMMITS = {
    0x14,              # DASHING
    0x15,              # RUNNING
    0x0F, 0x10, 0x11,  # WALK variants
    0x19, 0x1A,        # JUMPING_FORWARD, JUMPING_BACKWARD
    0x1B, 0x1C,        # JUMPING_ARIAL_FORWARD, JUMPING_ARIAL_BACKWARD
    0x1D, 0x1E, 0x1F, 0x20, 0x21, 0x22,  # FALLING variants
}

# Neutral/waiting - not committing
NO_COMMIT = {
    0x0E,              # STANDING
    0x27, 0x28, 0x29,  # CROUCH variants
    0x12,              # TURNING
    0x18,              # KNEE_BEND (pre-jump)
    0x2A,              # LANDING
}

# Defensive actions
DEFENSIVE_ACTIONS = {
    0xB2,              # SHIELD_START
    0xB3,              # SHIELD
    0xB4,              # SHIELD_RELEASE
    0xE9,              # ROLL_FORWARD
    0xEA,              # ROLL_BACKWARD
    0xEB,              # SPOTDODGE
    0xEC,              # AIRDODGE
    0xFD,              # EDGE_HANGING
    0xFC,              # EDGE_CATCHING
}

# Hitstun/tumble states (being hit = passive)
HITSTUN_ACTIONS = {
    0x4B, 0x4C, 0x4D,  # DAMAGE_HIGH variants
    0x4E, 0x4F, 0x50,  # DAMAGE_NEUTRAL variants
    0x51, 0x52, 0x53,  # DAMAGE_LOW variants
    0x54, 0x55, 0x56,  # DAMAGE_AIR variants
    0x57, 0x58, 0x59, 0x5A, 0x5B,  # DAMAGE_FLY (tumble) variants
    0x26,              # TUMBLING
}

# Aerial states (for classifying aerial intent)
AERIAL_ATTACKS = {0x41, 0x42, 0x43, 0x44, 0x45}  # NAIR, FAIR, BAIR, UAIR, DAIR


def build_feature_index(colmap: ColumnMap) -> Dict[str, int]:
    """Build a name -> index mapping for all features."""
    return {name: i for i, name in enumerate(colmap.feat_names)}


def classify_aerial_intent(
    X: Tensor,
    frame_idx: int,
    feature_idx: Dict[str, int],
    batch_idx: int,
) -> str:
    """
    Classify whether an aerial is approaching (aggressive) or fading (defensive).

    Looks at drift direction relative to opponent position.
    """
    if frame_idx < 3:
        return "neutral"

    # Get current and previous positions
    curr_p1_x = X[batch_idx, frame_idx, feature_idx["p1_position_x"]].item()
    prev_p1_x = X[batch_idx, frame_idx - 3, feature_idx["p1_position_x"]].item()
    opp_x = X[batch_idx, frame_idx, feature_idx["p2_position_x"]].item()

    # Compute drift direction
    drift = curr_p1_x - prev_p1_x
    direction_to_opp = opp_x - curr_p1_x

    # Same sign = approaching, opposite = fading
    if abs(drift) < 0.5:
        return "neutral"  # Not really drifting
    elif (drift > 0 and direction_to_opp > 0) or (drift < 0 and direction_to_opp < 0):
        return "approaching"
    else:
        return "fading"


def detect_dash_dance(
    X: Tensor,
    feature_idx: Dict[str, int],
    batch_idx: int,
    lookback: int = 30,
) -> bool:
    """
    Detect dash dance pattern: rapid DASHING/TURNING sequence.

    Returns True if player is dash dancing (passive micro-spacing).
    """
    seq_len = X.shape[1]
    start_frame = max(0, seq_len - lookback)

    turn_count = 0
    last_was_turn = False

    for f in range(start_frame, seq_len):
        action = int(X[batch_idx, f, feature_idx["p1_action"]].item())
        is_turn = action == 0x12  # TURNING
        is_dash = action == 0x14  # DASHING

        if is_turn and not last_was_turn:
            turn_count += 1
        last_was_turn = is_turn or is_dash

    # 3+ direction changes in window = dash dancing
    return turn_count >= 3


def detect_empty_hop(
    X: Tensor,
    feature_idx: Dict[str, int],
    batch_idx: int,
    lookback: int = 30,
) -> bool:
    """
    Detect empty hop: jump that lands without attacking.

    Pattern: KNEE_BEND -> JUMP -> FALL -> LANDING without any aerial attack.
    Empty hops are passive/spacing moves.
    """
    seq_len = X.shape[1]
    start_frame = max(0, seq_len - lookback)

    in_jump = False
    saw_attack = False

    JUMP_ACTIONS = {0x18, 0x19, 0x1A, 0x1B, 0x1C}  # KNEE_BEND, JUMP variants
    FALL_ACTIONS = {0x1D, 0x1E, 0x1F, 0x20, 0x21, 0x22}  # FALLING variants

    for f in range(start_frame, seq_len):
        action = int(X[batch_idx, f, feature_idx["p1_action"]].item())

        if action in JUMP_ACTIONS:
            in_jump = True
            saw_attack = False
        elif action in AERIAL_ATTACKS:
            saw_attack = True
        elif action == 0x2A and in_jump:  # LANDING
            if not saw_attack:
                return True  # Empty hop detected
            in_jump = False

    return False


def compute_initiative(
    X: Tensor,
    feature_idx: Dict[str, int],
    batch_idx: int,
    lookback: int = 60,
) -> float:
    """
    Who commits first after neutral reset?

    Returns positive value if P1 initiated, negative if P2.
    Looks for first medium/hard commit from neutral.
    """
    seq_len = X.shape[1]
    start_frame = max(0, seq_len - lookback)

    p1_first_commit_frame = None
    p2_first_commit_frame = None

    for f in range(start_frame, seq_len):
        p1_action = int(X[batch_idx, f, feature_idx["p1_action"]].item())
        p2_action = int(X[batch_idx, f, feature_idx["p2_action"]].item())

        # P1 commits
        if p1_first_commit_frame is None:
            if p1_action in HARD_COMMITS or p1_action in MEDIUM_COMMITS:
                p1_first_commit_frame = f

        # P2 commits
        if p2_first_commit_frame is None:
            if p2_action in HARD_COMMITS or p2_action in MEDIUM_COMMITS:
                p2_first_commit_frame = f

        if p1_first_commit_frame is not None and p2_first_commit_frame is not None:
            break

    # Return initiative score
    if p1_first_commit_frame is not None and p2_first_commit_frame is None:
        return 1.0  # P1 initiated
    elif p2_first_commit_frame is not None and p1_first_commit_frame is None:
        return -1.0  # P2 initiated
    elif p1_first_commit_frame is not None and p2_first_commit_frame is not None:
        frame_diff = p2_first_commit_frame - p1_first_commit_frame
        return np.clip(frame_diff / 10.0, -1.0, 1.0)  # Normalize
    else:
        return 0.0  # Neither committed


def label_windows(
    X: Tensor,
    feature_idx: Dict[str, int],
    aggressive_threshold: float = 0.3,
    passive_threshold: float = -0.3,
) -> List[str]:
    """
    Label each window in a batch as 'aggressive', 'passive', or 'neutral'.

    Uses multi-factor scoring:
    - 40% Action + intent (what are you doing?)
    - 25% Patterns (dash dance, empty hop, etc.)
    - 20% Context (positioning, facing)
    - 15% Initiative (who commits first)

    Args:
        X: Feature tensor [B, L, F]
        feature_idx: Feature name to index mapping
        aggressive_threshold: Score threshold for aggressive label
        passive_threshold: Score threshold for passive label

    Returns:
        List of labels, one per batch element
    """
    batch_size = X.shape[0]
    seq_len = X.shape[1]
    last_frame = seq_len - 1

    labels = []

    for b in range(batch_size):
        # Extract last frame features
        p1_action = int(X[b, last_frame, feature_idx["p1_action"]].item())
        p1_x = X[b, last_frame, feature_idx["p1_position_x"]].item()
        p2_x = X[b, last_frame, feature_idx["p2_position_x"]].item()
        p1_y = X[b, last_frame, feature_idx["p1_position_y"]].item()
        p2_y = X[b, last_frame, feature_idx["p2_position_y"]].item()
        facing = X[b, last_frame, feature_idx["p1_facing"]].item()
        is_hitting = X[b, last_frame, feature_idx["p1_is_in_hitlag"]].item()

        distance = ((p1_x - p2_x) ** 2 + (p1_y - p2_y) ** 2) ** 0.5

        # ========== 1. Action + Intent Score (40% weight) ==========
        action_score = 0.0

        if p1_action in HARD_COMMITS:
            action_score = 1.0
        elif p1_action in MEDIUM_COMMITS:
            # Check aerial intent if applicable
            if p1_action in AERIAL_ATTACKS:
                intent = classify_aerial_intent(X, last_frame, feature_idx, b)
                if intent == "approaching":
                    action_score = 0.8
                elif intent == "fading":
                    action_score = -0.3  # Fading aerial is defensive
                else:
                    action_score = 0.4  # Neutral drift
            else:
                action_score = 0.7
        elif p1_action in DEFENSIVE_ACTIONS:
            action_score = -0.8
        elif p1_action in HITSTUN_ACTIONS:
            action_score = -1.0  # Being hit = very passive
        elif p1_action in SOFT_COMMITS:
            # Soft commit direction matters
            direction_to_opp = p2_x - p1_x
            if (facing > 0 and direction_to_opp > 0) or (facing < 0 and direction_to_opp < 0):
                action_score = 0.3  # Moving toward opponent
            else:
                action_score = -0.2  # Moving away
        elif p1_action in NO_COMMIT:
            action_score = 0.0

        # Bonus for actively hitting
        if is_hitting > 0:
            action_score = min(action_score + 0.3, 1.0)

        # ========== 2. Pattern Score (25% weight) ==========
        pattern_score = 0.0

        # Dash dancing = passive micro-spacing
        if detect_dash_dance(X, feature_idx, b):
            pattern_score -= 0.6

        # Empty hop = passive spacing
        if detect_empty_hop(X, feature_idx, b):
            pattern_score -= 0.4

        # ========== 3. Context Score (20% weight) ==========
        context_score = 0.0

        # Distance-based: close = aggressive, far = passive
        if distance < 30:
            context_score = 0.5
        elif distance < 60:
            context_score = 0.2
        elif distance > 120:
            context_score = -0.5
        elif distance > 80:
            context_score = -0.2

        # Facing toward opponent = aggressive intent
        direction_to_opp = p2_x - p1_x
        facing_toward = (facing > 0 and direction_to_opp > 0) or (facing < 0 and direction_to_opp < 0)
        if facing_toward:
            context_score += 0.2
        else:
            context_score -= 0.2

        # ========== 4. Initiative Score (15% weight) ==========
        initiative_score = compute_initiative(X, feature_idx, b)

        # ========== Composite Score ==========
        composite = (
            0.40 * action_score +
            0.25 * pattern_score +
            0.20 * context_score +
            0.15 * initiative_score
        )

        # Assign label
        if composite >= aggressive_threshold:
            labels.append("aggressive")
        elif composite <= passive_threshold:
            labels.append("passive")
        else:
            labels.append("neutral")

    return labels


# -----------------------------------------------------------------------------
# 3. Steering Vector Computation
# -----------------------------------------------------------------------------


def compute_steering_vectors(
    aggressive_acts: Dict[int, Tensor],
    passive_acts: Dict[int, Tensor],
) -> Dict[int, Tensor]:
    """
    Compute steering vector per layer: v_l = mean(aggressive) - mean(passive).

    Args:
        aggressive_acts: layer_idx -> [N_agg, 512] activations
        passive_acts: layer_idx -> [N_pas, 512] activations

    Returns:
        layer_idx -> [512] steering vector
    """
    vectors = {}
    for layer_idx in aggressive_acts.keys():
        agg_mean = aggressive_acts[layer_idx].mean(dim=0)
        pas_mean = passive_acts[layer_idx].mean(dim=0)
        vectors[layer_idx] = agg_mean - pas_mean
    return vectors


# -----------------------------------------------------------------------------
# 4. Steerable Forward Pass
# -----------------------------------------------------------------------------


def steered_forward(
    model: GPT,
    inputs: TensorDict,
    steering_vector: Tensor,
    layer_idx: int,
    scale: float,
) -> TensorDict:
    """
    Forward pass with steering injection at specified layer.

    Args:
        model: The GPT model
        inputs: TensorDict with model inputs
        steering_vector: [512] steering direction
        layer_idx: Which layer to inject at (0-indexed)
        scale: Steering magnitude (positive = toward aggressive)

    Returns:
        TensorDict with model outputs
    """
    device = next(model.parameters()).device
    steering_vector = steering_vector.to(device)

    # Embed inputs
    combined = model._embed_inputs(inputs)
    hidden_states = model.projection_down(combined)
    hidden_states = model.dropout(hidden_states)

    batch_size, seq_len, _ = hidden_states.shape

    cos = model.cos[:, :seq_len]
    sin = model.sin[:, :seq_len]

    # Forward through blocks with steering injection
    for i, block in enumerate(model.blocks):
        hidden_states = block(hidden_states, cos, sin)
        if i == layer_idx:
            # Inject steering at last position only
            hidden_states = hidden_states.clone()
            hidden_states[:, -1, :] = hidden_states[:, -1, :] + scale * steering_vector

    hidden_states = norm(hidden_states)

    # Output heads (copied from GPT.forward)
    button_hidden_concat = None
    if model.separate_button_heads:
        button_logit_list = []
        button_hidden_list = []
        button_input = hidden_states
        for name in BUTTON_NAMES:
            if model.pass_button_hidden_to_stick:
                logit, h = model.button_heads[name].forward_with_hidden(button_input)
                button_hidden_list.append(h)
            else:
                logit = model.button_heads[name](button_input)
            button_logit_list.append(logit)
            button_input = torch.cat(
                [hidden_states] + [l.detach() for l in button_logit_list], dim=-1
            )
        button_logits = torch.cat(button_logit_list, dim=-1)
        if model.pass_button_hidden_to_stick:
            button_hidden_concat = torch.cat(button_hidden_list, dim=-1)
    else:
        button_logits = model.button_head(hidden_states)

    # Main stick
    if model.pass_button_hidden_to_stick and button_hidden_concat is not None:
        main_stick_input = torch.cat(
            (hidden_states, button_logits.detach(), button_hidden_concat.detach()),
            dim=-1,
        )
    else:
        main_stick_input = torch.cat((hidden_states, button_logits.detach()), dim=-1)
    main_stick = model.main_stick_head(main_stick_input)

    # C-stick
    c_stick = model.c_stick_head(
        torch.cat(
            (hidden_states, button_logits.detach(), main_stick.detach()),
            dim=-1,
        )
    )

    # Shoulder
    shoulder = model.shoulder_head(
        torch.cat(
            (
                hidden_states,
                button_logits.detach(),
                main_stick.detach(),
                c_stick.detach(),
            ),
            dim=-1,
        )
    )

    # Value
    value = model.value_head(hidden_states)

    outputs = TensorDict(
        {
            "buttons": button_logits,
            "main_stick": main_stick,
            "c_stick": c_stick,
            "shoulder": shoulder,
            "value": value,
        },
        batch_size=(batch_size, seq_len),
    )

    return outputs


# -----------------------------------------------------------------------------
# 5. Evaluation
# -----------------------------------------------------------------------------


def evaluate_steering(
    model: GPT,
    loader: DataLoader,
    colmap: ColumnMap,
    steering_vectors: Dict[int, Tensor],
    device: torch.device,
    exclude_p1_controller: bool = False,
    layers: List[int] = [3, 4, 5],
    scales: List[float] = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
    max_batches: int = 100,
) -> Dict[Tuple[int, float], Dict[str, float]]:
    """
    Evaluate steering effect across layers and scales.

    Measures:
    - Value head delta (capability preservation)
    - Button A probability shift (attack intent)
    - Main stick entropy change (decisiveness)

    Args:
        model: The GPT model
        loader: Data loader
        colmap: Column mapping
        steering_vectors: layer_idx -> [512] vectors
        device: Compute device
        layers: Which layers to test
        scales: Which scales to test
        max_batches: Maximum batches to evaluate

    Returns:
        Dict mapping (layer, scale) -> metrics dict
    """
    results = {}

    model.eval()
    with torch.inference_mode():
        for layer_idx in layers:
            if layer_idx not in steering_vectors:
                continue

            for scale in scales:
                value_deltas = []
                button_a_deltas = []
                main_entropy_deltas = []

                for batch in tqdm(
                    islice(loader, max_batches),
                    total=max_batches,
                    desc=f"Layer {layer_idx}, scale {scale:+.1f}",
                    leave=False,
                ):
                    X = batch["X"].to(device)
                    inputs = build_model_inputs(
                        X, colmap, training=False,
                        exclude_p1_controller=exclude_p1_controller,
                    )

                    # Baseline forward
                    baseline = model(inputs)

                    # Steered forward
                    steered = steered_forward(
                        model, inputs, steering_vectors[layer_idx], layer_idx, scale
                    )

                    # Value delta (should be small for good steering)
                    value_delta = (
                        (steered["value"] - baseline["value"]).mean().item()
                    )
                    value_deltas.append(value_delta)

                    # Button A probability shift
                    baseline_a_prob = torch.sigmoid(baseline["buttons"][:, -1, 0]).mean()
                    steered_a_prob = torch.sigmoid(steered["buttons"][:, -1, 0]).mean()
                    button_a_deltas.append((steered_a_prob - baseline_a_prob).item())

                    # Main stick entropy change
                    baseline_entropy = -(
                        F.softmax(baseline["main_stick"][:, -1, :], dim=-1)
                        * F.log_softmax(baseline["main_stick"][:, -1, :], dim=-1)
                    ).sum(-1).mean()
                    steered_entropy = -(
                        F.softmax(steered["main_stick"][:, -1, :], dim=-1)
                        * F.log_softmax(steered["main_stick"][:, -1, :], dim=-1)
                    ).sum(-1).mean()
                    main_entropy_deltas.append(
                        (steered_entropy - baseline_entropy).item()
                    )

                results[(layer_idx, scale)] = {
                    "value_delta_mean": np.mean(value_deltas),
                    "value_delta_std": np.std(value_deltas),
                    "button_a_delta": np.mean(button_a_deltas),
                    "main_entropy_delta": np.mean(main_entropy_deltas),
                }

    return results


def print_results(results: Dict[Tuple[int, float], Dict[str, float]]) -> None:
    """Print evaluation results in a nice table."""
    print("\n" + "=" * 80)
    print("STEERING EVALUATION RESULTS")
    print("=" * 80)
    print(
        f"{'Layer':>6} {'Scale':>8} {'Value Δ':>12} {'Button A Δ':>12} {'Entropy Δ':>12}"
    )
    print("-" * 80)

    # Group by layer
    layers = sorted(set(l for l, s in results.keys()))
    for layer in layers:
        for scale in sorted(s for l, s in results.keys() if l == layer):
            r = results[(layer, scale)]
            value_str = f"{r['value_delta_mean']:+.4f}"
            button_str = f"{r['button_a_delta']:+.4f}"
            entropy_str = f"{r['main_entropy_delta']:+.4f}"

            # Highlight good results
            if abs(r["value_delta_mean"]) < 0.1 and r["button_a_delta"] > 0.01:
                marker = " <-- good"
            elif abs(r["value_delta_mean"]) > 0.3:
                marker = " (degraded)"
            else:
                marker = ""

            print(
                f"{layer:>6} {scale:>+8.1f} {value_str:>12} {button_str:>12} {entropy_str:>12}{marker}"
            )
        print()


# -----------------------------------------------------------------------------
# 6. Main
# -----------------------------------------------------------------------------


def load_model(config, device: torch.device) -> GPT:
    """Load model from latest checkpoint."""
    ckpt_path = find_latest_checkpoint(Path(config.train.out_dir))
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint found in {config.train.out_dir}")

    print(f"Loading checkpoint: {ckpt_path}")
    model = GPT(config).to(device)

    ckpt_state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"]
    model_state = match_state_dict_keys(ckpt_state, model)
    model.load_state_dict(model_state)
    model.eval()

    return model


def main():
    parser = argparse.ArgumentParser(description="Compute and evaluate persona vectors")
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Only collect activations and compute vectors (skip evaluation)",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=500,
        help="Maximum batches for activation collection",
    )
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=100,
        help="Maximum batches for evaluation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for data loading",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("steering_vectors_aggression.pt"),
        help="Output path for steering vectors",
    )
    args = parser.parse_args()

    # Initialize
    init_config()
    config = get_config()
    device = _resolve_device()
    print(f"Using device: {device}")

    # Load model and data
    model = load_model(config, device)
    print(f"Model: {config.model.n_layer} layers, {config.model.n_embd} dim")

    loader, dataset, colmap, _ = _get_validation_loader(
        config,
        batch_size=args.batch_size,
        window_stride=256,
    )
    feature_idx = build_feature_index(colmap)
    print(f"Dataset: {len(dataset)} windows")

    # Collect activations with labels
    collector = ActivationCollector(model)
    collector.register_hooks()

    aggressive_acts: Dict[int, List[Tensor]] = defaultdict(list)
    passive_acts: Dict[int, List[Tensor]] = defaultdict(list)
    label_counts = {"aggressive": 0, "passive": 0, "neutral": 0}

    print(f"\nCollecting activations (max {args.max_batches} batches)...")
    with torch.inference_mode():
        for batch in tqdm(islice(loader, args.max_batches), total=args.max_batches):
            X = batch["X"].to(device)
            labels = label_windows(X, feature_idx)

            # Forward pass triggers hooks
            inputs = build_model_inputs(
                X, colmap, training=False,
                exclude_p1_controller=config.model.exclude_p1_controller,
            )
            _ = model(inputs)

            # Collect activations by label
            for i, label in enumerate(labels):
                label_counts[label] += 1
                if label == "aggressive":
                    for layer_idx, acts_list in collector.activations.items():
                        aggressive_acts[layer_idx].append(acts_list[-1][i : i + 1])
                elif label == "passive":
                    for layer_idx, acts_list in collector.activations.items():
                        passive_acts[layer_idx].append(acts_list[-1][i : i + 1])

            collector.clear()

    collector.remove_hooks()

    print(f"\nLabel distribution:")
    print(f"  Aggressive: {label_counts['aggressive']:,}")
    print(f"  Passive:    {label_counts['passive']:,}")
    print(f"  Neutral:    {label_counts['neutral']:,}")

    if label_counts["aggressive"] < 100 or label_counts["passive"] < 100:
        print("\nWARNING: Low sample count. Consider adjusting thresholds.")

    # Compute steering vectors
    print("\nComputing steering vectors...")
    steering_vectors = compute_steering_vectors(
        {k: torch.cat(v) for k, v in aggressive_acts.items()},
        {k: torch.cat(v) for k, v in passive_acts.items()},
    )

    print("\nSteering vector norms per layer:")
    for layer_idx in sorted(steering_vectors.keys()):
        vec = steering_vectors[layer_idx]
        print(f"  Layer {layer_idx}: norm={vec.norm().item():.4f}")

    # Save vectors
    torch.save(steering_vectors, args.output)
    print(f"\nSaved steering vectors to: {args.output}")

    if args.collect_only:
        print("\nSkipping evaluation (--collect-only)")
        return

    # Evaluate steering
    print(f"\nEvaluating steering effects (max {args.eval_batches} batches)...")

    # Reload loader to reset iterator
    loader, _, _, _ = _get_validation_loader(
        config,
        batch_size=args.batch_size,
        window_stride=256,
    )

    results = evaluate_steering(
        model,
        loader,
        colmap,
        steering_vectors,
        device,
        exclude_p1_controller=config.model.exclude_p1_controller,
        layers=[3, 4, 5, 6],  # Mid-to-late layers
        scales=[-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
        max_batches=args.eval_batches,
    )

    print_results(results)

    # Find best layer/scale
    best_key = None
    best_score = -float("inf")
    for key, r in results.items():
        # Score: want high button_a_delta with low value degradation
        if abs(r["value_delta_mean"]) < 0.15:
            score = r["button_a_delta"]
            if score > best_score:
                best_score = score
                best_key = key

    if best_key:
        print(f"\nRecommended: Layer {best_key[0]}, scale {best_key[1]:+.1f}")
        print(f"  Button A shift: {results[best_key]['button_a_delta']:+.4f}")
        print(f"  Value delta: {results[best_key]['value_delta_mean']:+.4f}")


if __name__ == "__main__":
    main()
