#!/usr/bin/env python
"""
Targeted interpretability script to analyze the model's Tech DIRECTION decision making.

Goals:
1. Find instances of Neutral Tech, Forward Tech, and Backward Tech.
2. Analyze the Main Stick output probabilities for these events.
3. Perform counterfactual ablations to see what drives the direction choice:
   - Opponent position (Fear factor)
   - Stage position (Stage control)
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import init_config, get_config
from model.nano_gpt import GPT
from train import find_latest_checkpoint
from utils import _resolve_device, match_state_dict_keys
from column_map import ColumnMap
from validation import PreloadedWindowDataset, _palette_on_device, _decode_stick_coords
from train.batch_utils import build_model_inputs
from libmelee.melee.enums import Action
from constants import CONTROLLER_KEY_GROUPS

# Tech Action IDs
TECH_NEUTRAL = {
    Action.NEUTRAL_TECH.value,
    Action.WALL_TECH.value,
    Action.CEILING_TECH.value,
}
TECH_FORWARD = {Action.FORWARD_TECH.value}  # "Forward" implies facing direction
TECH_BACKWARD = {Action.BACKWARD_TECH.value}


def find_tech_direction_events(
    ds: PreloadedWindowDataset, colmap: ColumnMap, limit_per_type: int = 3
) -> Dict[str, List[Dict]]:
    """
    Finds examples of each tech type.
    """
    events = {"neutral": [], "forward": [], "backward": []}
    print(f"Scanning {len(ds)} windows for tech direction events...")

    act_idx = colmap.ego_action_idx

    for ep_idx, cached_ep in enumerate(ds._episode_cache):
        features = cached_ep.features
        actions = features[:, act_idx]

        # Scan for transitions into tech states
        for t in range(1, len(actions) - 5):
            prev_act = actions[t - 1]
            curr_act = actions[t]

            # Skip if we were already teching (not a fresh start)
            if prev_act == curr_act:
                continue

            etype = None
            if curr_act in TECH_NEUTRAL:
                etype = "neutral"
            elif curr_act in TECH_FORWARD:
                etype = "forward"
            elif curr_act in TECH_BACKWARD:
                etype = "backward"

            if etype and len(events[etype]) < limit_per_type:
                # Found one! Grab context
                end_frame = t + 5
                start_frame = end_frame - ds.seq_len
                if start_frame < 0:
                    continue

                X = torch.from_numpy(features[start_frame:end_frame]).unsqueeze(0)

                events[etype].append(
                    {
                        "X": X,
                        "tech_frame_rel": t - start_frame,
                        "description": f"Ep {ep_idx} Frame {t} ({etype.upper()})",
                        "type": etype,
                    }
                )

        if all(len(v) >= limit_per_type for v in events.values()):
            break

    return events


def get_predicted_stick_direction(
    model: GPT, colmap: ColumnMap, X_in: torch.Tensor, tech_idx: int
) -> str:
    """
    Returns the predicted stick direction (Neutral, Left, Right) at the tech frame.
    """
    with torch.no_grad():
        inputs = build_model_inputs(X_in, colmap)
        outputs = model(inputs)
        logits = outputs["main_stick"]  # [1, L, 64]

        # Get distribution at tech frame (or slightly before)
        # Tech direction inputs are buffered. Let's look at tech_idx - 1
        frame_idx = max(0, tech_idx - 1)

        # Softmax to get probs
        probs = torch.softmax(logits[0, frame_idx], dim=0)  # [64]

        # We need the stick palette to map indices to coordinates
        # This is a bit hacky, ideally we'd use the shared palette cache
        # For now, we just rely on the fact that index 0 is usually neutral
        # and other indices map to coordinates.
        # Let's use the helper from validation.py if possible or rebuild

        # Recalculate weighted average X position
        # This requires the palette.
        # Let's just assume standard quantization for a quick check:
        # If argmax is 0 -> Neutral.
        # If coordinate x > 0.3 -> Right
        # If coordinate x < -0.3 -> Left

        best_idx = probs.argmax().item()

        # We need the palette to know what best_idx means.
        # validation.py has _MAIN_PALETTE_T. Let's assume we can get it.
        # For this script, let's assume the model is imported and we can't easily grab the palette
        # without importing 'controller_utils'.
        from controller_utils import CONTROL_STICK_QUANTIZED

        x, y = CONTROL_STICK_QUANTIZED[best_idx]

        if abs(x) < 0.2875:
            return "Neutral"
        elif x > 0:
            return "Right"
        else:
            return "Left"


def perform_ablation(model: GPT, colmap: ColumnMap, event: Dict, device: torch.device):
    X_orig = event["X"].to(device)
    tech_idx = event["tech_frame_rel"]

    def get_idx(name: str) -> int:
        return colmap.feat_names.index(name)

    base_dir = get_predicted_stick_direction(model, colmap, X_orig, tech_idx)

    # Ablation 1: Opponent on Left (-10 X)
    X_opp_left = X_orig.clone()
    opp_x_idx = get_idx("p2_position_x")
    # Move opponent to -20 (far left) for last 30 frames
    X_opp_left[0, tech_idx - 30 : tech_idx + 5, opp_x_idx] = -20.0
    dir_opp_left = get_predicted_stick_direction(model, colmap, X_opp_left, tech_idx)

    # Ablation 2: Opponent on Right (+10 X)
    X_opp_right = X_orig.clone()
    X_opp_right[0, tech_idx - 30 : tech_idx + 5, opp_x_idx] = 20.0
    dir_opp_right = get_predicted_stick_direction(model, colmap, X_opp_right, tech_idx)

    # Ablation 3: Center Stage Control (Self at -20, trying to get to 0)
    X_edge_left = X_orig.clone()
    ego_x_idx = get_idx("p1_position_x")
    X_edge_left[0, tech_idx - 30 : tech_idx + 5, ego_x_idx] = -20.0
    # Also move opponent to center to motivate rolling IN
    X_edge_left[0, tech_idx - 30 : tech_idx + 5, opp_x_idx] = 0.0
    dir_edge_left = get_predicted_stick_direction(model, colmap, X_edge_left, tech_idx)

    print(f"\nEvent: {event['description']}")
    print(f"  Baseline: {base_dir}")
    print(f"  Opponent Far Left:  {dir_opp_left} (Expected: Right/Away)")
    print(f"  Opponent Far Right: {dir_opp_right} (Expected: Left/Away)")
    print(f"  Self Far Left:      {dir_edge_left} (Expected: Right/Center)")


def main():
    init_config()
    config = get_config()
    device = _resolve_device()

    data_root = Path("validation_set")
    if not data_root.exists():
        data_root = Path(config.zarr.out_root)

    ds = PreloadedWindowDataset(str(data_root), progress=False)
    colmap = ColumnMap.from_dataset(ds)

    ckpt_path = Path("checkpoints/model_ep021_005001.pt")
    if not ckpt_path.exists():
        ckpt_path = find_latest_checkpoint(Path(config.train.out_dir))

    model = GPT(config).to(device)
    ckpt_state = torch.load(ckpt_path, map_location="cpu")["model"]
    model_state = match_state_dict_keys(ckpt_state, model)
    model.load_state_dict(model_state)
    model.eval()

    events_dict = find_tech_direction_events(ds, colmap)

    all_events = (
        events_dict["neutral"] + events_dict["forward"] + events_dict["backward"]
    )
    if not all_events:
        print("No tech events found.")
        return

    print("\nAnalyzing Tech Direction Choices...")
    for event in all_events:
        perform_ablation(model, colmap, event, device)


if __name__ == "__main__":
    main()
