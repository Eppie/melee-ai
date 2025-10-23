"""Batch processing and preparation utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence

import torch
from tensordict import TensorDict
from torch import Tensor

from column_map import CONTROLLER_KEY_GROUPS
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


@dataclass(frozen=True)
class SampleWeightRatios:
    """How much to upweight 'change' frames vs 'hold' frames, per component."""
    main_change: float = 8.0
    c_change: float = 10.0
    shoulder_change: float = 10.0
    buttons_change_default: float = 10.0
    buttons_change_per_key: Dict[str, float] = field(default_factory=dict)
    hold_base: float = 1.0
    value_change: Optional[float] = None  # None => reuse "main" weights


def _normalize(w: Tensor) -> Tensor:
    return w / (w.mean() + 1e-12)


def compute_component_sample_weights(
        target_info: Mapping[str, Tensor],
        device: torch.device,
        *,
        ratios: Optional[SampleWeightRatios] = None,
        button_names: Optional[Sequence[str]] = None,
) -> Dict[str, Tensor]:
    """
    Build per-component loss weights:
      - 'main':    [B, L]
      - 'c':       [B, L]
      - 'shoulder':[B, L] (if present, else ones)
      - 'buttons': [B, L, K]
      - 'global':  [B, L] (union-of-changes; can be useful for value head)
    """
    r = ratios or SampleWeightRatios()
    B, L = target_info["main_idx"].shape

    # --- MAIN ---
    main_idx = target_info["main_idx"]  # [B, L]
    main_change = torch.zeros((B, L), device=device, dtype=torch.bool)
    if L > 1:
        main_change[:, 1:] = (main_idx[:, 1:] != main_idx[:, :-1])
    w_main = torch.where(
        main_change, torch.as_tensor(r.main_change, device=device), torch.as_tensor(r.hold_base, device=device)
    ).to(torch.float32)
    w_main = _normalize(w_main)

    # --- C-STICK ---
    c_idx = target_info["c_idx"]
    c_change = torch.zeros((B, L), device=device, dtype=torch.bool)
    if L > 1:
        c_change[:, 1:] = (c_idx[:, 1:] != c_idx[:, :-1])
    w_c = torch.where(
        c_change, torch.as_tensor(r.c_change, device=device), torch.as_tensor(r.hold_base, device=device)
    ).to(torch.float32)
    w_c = _normalize(w_c)

    # --- SHOULDER (optional) ---
    w_shoulder = torch.ones((B, L), device=device, dtype=torch.float32)
    sh_idx = target_info.get("shoulder_idx")
    if sh_idx is not None and sh_idx.numel() > 0:
        sh_change = torch.zeros((B, L), device=device, dtype=torch.bool)
        if L > 1:
            sh_change[:, 1:] = (sh_idx[:, 1:] != sh_idx[:, :-1])
        w_shoulder = torch.where(
            sh_change, torch.as_tensor(r.shoulder_change, device=device), torch.as_tensor(r.hold_base, device=device)
        ).to(torch.float32)
        w_shoulder = _normalize(w_shoulder)

    # --- BUTTONS (per-button) ---
    btn_t = target_info["buttons"].to(torch.float32)  # [B, L, K], {0,1}
    K = btn_t.shape[-1]
    if button_names is None:
        button_names = CONTROLLER_KEY_GROUPS["buttons"]

    # Change mask per button at frame t>0
    btn_change = torch.zeros((B, L, K), device=device, dtype=torch.bool)
    if L > 1:
        btn_change[:, 1:, :] = (btn_t[:, 1:, :] != btn_t[:, :-1, :])

    # Build per-button change ratios
    per_button_ratio = torch.full((K,), float(r.buttons_change_default), device=device, dtype=torch.float32)
    for k, name in enumerate(button_names):
        if name in r.buttons_change_per_key:
            per_button_ratio[k] = float(r.buttons_change_per_key[name])

    # weights = hold_base on holds; ratio_k on changes of button k
    w_buttons = torch.where(
        btn_change,
        per_button_ratio.view(1, 1, K),
        torch.as_tensor(r.hold_base, device=device).view(1, 1, 1),
    ).to(torch.float32)
    w_buttons = _normalize(w_buttons)

    # --- GLOBAL (union-of-changes) ---
    union_change = main_change | c_change
    if sh_idx is not None and sh_idx.numel() > 0:
        union_change = union_change | sh_change
    union_change = union_change | btn_change.any(dim=-1)

    value_ratio = r.value_change if (r.value_change is not None) else r.main_change
    w_global = torch.where(
        union_change, torch.as_tensor(value_ratio, device=device), torch.as_tensor(r.hold_base, device=device)
    ).to(torch.float32)
    w_global = _normalize(w_global)

    return {
        "main": w_main,  # [B, L]
        "c": w_c,  # [B, L]
        "shoulder": w_shoulder,  # [B, L]
        "buttons": w_buttons,  # [B, L, K]
        "global": w_global,  # [B, L]
    }
