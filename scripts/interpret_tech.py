#!/usr/bin/env python
"""
Targeted interpretability script to analyze the model's "Tech" decision making.

Goals:
1. Find specific instances in the validation set where the model (or player) performs a tech.
2. Perform counterfactual ablations on these instances to see what triggers the decision.
   - Does it stop teching if we move it away from the ground?
   - Does it stop teching if we remove the hitstun state?
   - Does it stop teching if we inject a "lockout" (press L 30 frames ago)?
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
from validation import PreloadedWindowDataset
from train.batch_utils import build_model_inputs
from libmelee.melee.enums import Action
from constants import CONTROLLER_KEY_GROUPS

# Action IDs for Teching
TECH_ACTIONS = {
    Action.NEUTRAL_TECH.value,
    Action.FORWARD_TECH.value,
    Action.BACKWARD_TECH.value,
    Action.WALL_TECH.value,
    Action.WALL_TECH_JUMP.value,
    Action.CEILING_TECH.value,
}

# Actions that imply we are in hitstun/tumble and COULD tech
HITSTUN_ACTIONS = {
    Action.DAMAGE_FLY_HIGH.value,
    Action.DAMAGE_FLY_NEUTRAL.value,
    Action.DAMAGE_FLY_LOW.value,
    Action.DAMAGE_FLY_TOP.value,
    Action.DAMAGE_FLY_ROLL.value,
    Action.TUMBLING.value,
}

def find_tech_events(
    ds: PreloadedWindowDataset, colmap: ColumnMap, limit: int = 5
) -> List[Dict[str, torch.Tensor]]:
    """
    Scans the dataset for windows where a tech occurs near the end of the sequence.
    """
    events = []
    print(f"Scanning {len(ds)} windows for tech events...")
    
    # We iterate with a stride to cover ground faster
    stride = 50
    
    # Get feature indices
    def get_idx(name: str) -> int:
        return colmap.feat_names.index(name)

    act_idx = colmap.ego_action_idx
    btn_lr_idx = get_idx("p1_button_lr")
    pos_y_idx = get_idx("p1_position_y")
    
    # We can access the cached episode data directly for speed
    for ep_idx, cached_ep in enumerate(ds._episode_cache):
        features = cached_ep.features # [T, F]
        
        # Find frames where action transitions TO a tech state
        actions = features[:, act_idx]
        
        # Create a boolean mask for tech states
        is_tech = np.isin(actions, list(TECH_ACTIONS))
        
        # Find transitions (0 -> 1)
        tech_starts = np.where((~is_tech[:-1]) & (is_tech[1:]))[0] + 1
        
        for t in tech_starts:
            # We want the window to END shortly after the tech start
            # to see the decision process leading up to it.
            # Let's say the window ends 5 frames after the tech start.
            end_frame = t + 5
            start_frame = end_frame - ds.seq_len
            
            if start_frame < 0:
                continue
                
            # Check if we actually pressed L/R in the few frames before the tech
            # (Sometimes tech happens automatically or buffered, but we want active presses)
            # Look at 20 frames before tech
            window_slice = features[start_frame:end_frame]
            
            # Grab the window as tensors
            X = torch.from_numpy(window_slice).unsqueeze(0) # [1, L, F]
            
            # Verify it's a valid event
            events.append({
                "X": X,
                "tech_frame_rel": t - start_frame, # Index in the window where tech happens
                "description": f"Ep {ep_idx} Frame {t} (Action {actions[t]})"
            })
            
            if len(events) >= limit:
                return events
                
    print(f"Found {len(events)} tech events.")
    return events

def perform_ablation(
    model: GPT, 
    colmap: ColumnMap, 
    event: Dict[str, torch.Tensor],
    device: torch.device
):
    """
    Runs the model on the event window with various ablations.
    """
    X_orig = event["X"].to(device)
    tech_idx = event["tech_frame_rel"]
    
    def get_idx(name: str) -> int:
        return colmap.feat_names.index(name)
    
    # Identify output head indices
    # In the output, "buttons" is [B, L, 5] usually.
    # We need to know which bit is L/R.
    # validation.py says: _BUTTON_NAME_TO_INDEX["button_lr"]
    # Let's hardcode or find it dynamically
    btn_names = CONTROLLER_KEY_GROUPS["buttons"]
    lr_output_idx = btn_names.index("button_lr")
    
    # --- Helper to run model and get prob of L/R press near tech time ---
    def get_tech_prob(X_in: torch.Tensor) -> float:
        with torch.no_grad():
            inputs = build_model_inputs(X_in, colmap)
            outputs = model(inputs)
            logits = outputs["buttons"] # [1, L, 5]
            probs = torch.sigmoid(logits)
            
            # Look at the probability of L/R press in the 5 frames LEADING UP to the tech
            # We take the max probability in that small window
            window_start = max(0, tech_idx - 10)
            window_end = tech_idx + 1
            
            lr_probs = probs[0, window_start:window_end, lr_output_idx]
            return float(lr_probs.max().item())

    # 1. Baseline
    baseline_prob = get_tech_prob(X_orig)
    
    # 2. Ablation: Position Y (Sky High)
    # Move player high up for the last 20 frames
    X_sky = X_orig.clone()
    pos_y_feat = get_idx("p1_position_y")
    X_sky[0, tech_idx-20:tech_idx+5, pos_y_feat] = 100.0
    prob_sky = get_tech_prob(X_sky)
    
    # 3. Ablation: Action State (Not in Hitstun)
    # Change action to FALLING (0x1D) for the last 20 frames
    X_safe = X_orig.clone()
    act_feat = colmap.ego_action_idx
    X_safe[0, tech_idx-20:tech_idx+5, act_feat] = float(Action.FALLING.value)
    prob_safe = get_tech_prob(X_safe)
    
    # 4. Ablation: Lockout (Simulate Press 30 frames ago)
    # Set p1_button_lr to 1.0 at t-30
    X_lockout = X_orig.clone()
    # button_lr is a feature too (input history)
    btn_lr_feat = get_idx("p1_button_lr")
    lockout_time = tech_idx - 30
    if lockout_time >= 0:
        X_lockout[0, lockout_time:lockout_time+5, btn_lr_feat] = 1.0 # Hold for 5 frames
        prob_lockout = get_tech_prob(X_lockout)
    else:
        prob_lockout = -1.0 # Window too short
        
    # 5. Ablation: Hitstun + Ground (The perfect storm)
    # Just to verify, what if we remove BOTH cues?
    X_chill = X_sky.clone()
    X_chill[0, tech_idx-20:tech_idx+5, act_feat] = float(Action.STANDING.value)
    prob_chill = get_tech_prob(X_chill)

    print(f"\nEvent: {event['description']}")
    print(f"  Baseline Tech Prob: {baseline_prob:.4f}")
    print(f"  Ablation [Sky High]: {prob_sky:.4f}  (Delta: {prob_sky - baseline_prob:+.4f})")
    print(f"  Ablation [Safe Act]: {prob_safe:.4f}  (Delta: {prob_safe - baseline_prob:+.4f})")
    if prob_lockout != -1:
        print(f"  Ablation [Lockout]:  {prob_lockout:.4f}  (Delta: {prob_lockout - baseline_prob:+.4f})")
    else:
        print(f"  Ablation [Lockout]:  N/A (Window start)")
    print(f"  Ablation [Chill]:    {prob_chill:.4f}  (Delta: {prob_chill - baseline_prob:+.4f})")

def main():
    init_config()
    config = get_config()
    device = _resolve_device()
    
    print("Loading Dataset...")
    # We use validation root
    data_root = Path("validation_set")
    if not data_root.exists():
        # Fallback for local testing if validation_set isn't there
        data_root = Path(config.zarr.out_root)
        
    ds = PreloadedWindowDataset(str(data_root), progress=False)
    colmap = ColumnMap.from_dataset(ds)
    
    print("Loading Model...")
    # Use the specific checkpoint we've been analyzing
    ckpt_path = Path("checkpoints/model_ep021_005001.pt")
    if not ckpt_path.exists():
        ckpt_path = find_latest_checkpoint(Path(config.train.out_dir))
        
    print(f"Using checkpoint: {ckpt_path}")
    
    model = GPT(config).to(device)
    ckpt_state = torch.load(ckpt_path, map_location="cpu")["model"]
    model_state = match_state_dict_keys(ckpt_state, model)
    model.load_state_dict(model_state)
    model.eval()
    
    print("Finding Tech Events...")
    events = find_tech_events(ds, colmap, limit=10)
    
    if not events:
        print("No tech events found in the first few episodes.")
        return

    print("\nPerforming Counterfactual Analysis...")
    print("We check the probability of an L/R press in the frames immediately preceding the tech.")
    
    for event in events:
        perform_ablation(model, colmap, event, device)

if __name__ == "__main__":
    main()
