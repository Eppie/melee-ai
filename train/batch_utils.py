"""Batch processing and preparation utilities."""
from __future__ import annotations

from typing import Dict

import torch
from tensordict import TensorDict

from column_map import ColumnMap
from controller_quantization import quantize_targets


def build_model_inputs(batch_X: torch.FloatTensor, colmap: ColumnMap) -> TensorDict:
    """Build TensorDict inputs for the model from batch features.
    
    Args:
        batch_X: [B, L, F] float32 features of current frame
        colmap: Column mapping for feature indices
        
    Returns:
        TensorDict with keys the model expects (stage, characters, actions, gamestate, controller)
    """
    B, L, _ = batch_X.shape

    # Categoricals back to long indices
    stage = batch_X[..., colmap.stage_idx].to(torch.long).unsqueeze(-1)  # [B,L,1]
    ego_character = batch_X[..., colmap.ego_char_idx].to(torch.long).unsqueeze(-1)
    opp_character = batch_X[..., colmap.opp_char_idx].to(torch.long).unsqueeze(-1)
    ego_action = batch_X[..., colmap.ego_action_idx].to(torch.long).unsqueeze(-1)
    opp_action = batch_X[..., colmap.opp_action_idx].to(torch.long).unsqueeze(-1)

    gamestate = batch_X[..., colmap.gamestate_idxs]  # [B,L,Gg]
    controller = batch_X[..., colmap.controller_idxs]  # [B,L,Gc]

    return TensorDict(
        {
            "stage": stage,
            "ego_character": ego_character,
            "opponent_character": opp_character,
            "ego_action": ego_action,
            "opponent_action": opp_action,
            "gamestate": gamestate,
            "controller": controller,
        },
        batch_size=(B, L),
    )


def quantize_controller_targets(
        batch_Y: torch.Tensor,
        colmap: ColumnMap,
        input_domain: str = "unit11"
) -> Dict[str, torch.Tensor]:
    """Quantize controller targets for loss computation.
    
    Wrapper around controller_quantization.quantize_targets for consistency.
    
    Args:
        batch_Y: [B, L, Y] target controller values
        colmap: Column mapping for target indices
        input_domain: Domain of input values ("unit11" or "unit01")
        
    Returns:
        Dictionary with quantized targets and metadata
    """
    return quantize_targets(batch_Y, colmap, input_domain=input_domain)


def compute_sample_weights(
        Y: torch.Tensor,
        B: int,
        L: int,
        device: torch.device,
        **kwargs
) -> torch.Tensor:
    """Compute per-sample weights for loss reweighting.
    
    Args:
        Y: [B, L, Y] target tensor
        B: Batch size
        L: Sequence length
        device: Device for output tensor
        **kwargs: Mode-specific parameters (e.g., ratio for change_boost)
        
    Returns:
        [B, L] weight tensor
    """
    ratio = kwargs.get("ratio", 10.0)

    # Identify "change" frames
    change_mask = torch.zeros((B, L), device=device, dtype=torch.bool)
    if L > 1:
        state_changed = torch.any(Y[:, 1:] != Y[:, :-1], dim=-1)
        change_mask[:, 1:] = state_changed

    # Raw weights: 1x for hold, ratio for change
    w = torch.ones((B, L), device=device)
    w[change_mask] = ratio

    # Normalize to remove batch composition effects (keep ratio intact)
    w = w / (w.mean() + 1e-12)
    return w