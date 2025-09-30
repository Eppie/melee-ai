from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch

from melee_ai.config import FOX_STICK_64, C_STICK_XY_CLUSTER_CENTERS_V0_1


def sticks01_to_unit11(xy01: torch.Tensor) -> torch.Tensor:
    """Map controller coordinates from [0,1] to [-1,1] and clamp to the unit circle."""
    xy11 = torch.clamp(xy01 * 2.0 - 1.0, -1.0, 1.0)
    radius = torch.linalg.norm(xy11, dim=-1, keepdim=True)
    scale = torch.clamp(radius, min=1.0)
    return xy11 / scale.clamp_min(1e-12)


def quantize_targets(
    batch_Y: torch.FloatTensor,
    colmap,
    shoulder_centers: Optional[Sequence[float]] = None,
) -> Dict[str, torch.Tensor]:
    """Convert raw controller targets in [0,1] space to palette indices."""
    B, L, _ = batch_Y.shape
    device = batch_Y.device

    # Main stick palette lookup
    main_xy01 = batch_Y[..., list(colmap.y_main)]
    main_xy11 = sticks01_to_unit11(main_xy01)
    P_main = torch.tensor(np.asarray(FOX_STICK_64, dtype=np.float32), device=device)
    V_main = main_xy11.reshape(-1, 2)
    main_norm = V_main.pow(2).sum(dim=1, keepdim=True)
    palette_norm = P_main.pow(2).sum(dim=1).unsqueeze(0)
    dot = V_main @ P_main.t()
    d2 = main_norm - 2.0 * dot + palette_norm
    y_main_idx = torch.argmin(d2, dim=1).view(B, L)

    # C-stick palette lookup
    c_xy01 = batch_Y[..., list(colmap.y_c)]
    c_xy11 = sticks01_to_unit11(c_xy01)
    P_c = torch.tensor(np.asarray(C_STICK_XY_CLUSTER_CENTERS_V0_1, dtype=np.float32), device=device)
    V_c = c_xy11.reshape(-1, 2)
    c_norm = V_c.pow(2).sum(dim=1, keepdim=True)
    palette_c_norm = P_c.pow(2).sum(dim=1).unsqueeze(0)
    dot_c = V_c @ P_c.t()
    d2c = c_norm - 2.0 * dot_c + palette_c_norm
    y_c_idx = torch.argmin(d2c, dim=1).view(B, L)

    # Buttons remain probabilistic targets
    btn_cols = colmap.y_buttons
    y_buttons = batch_Y[..., btn_cols].to(torch.float32)
    y_buttons = torch.clamp(y_buttons, 0.0, 1.0)

    # Optional shoulder binning
    y_shoulder_idx = None
    if colmap.y_shoulder is not None and shoulder_centers is not None:
        centers = torch.tensor(np.asarray(shoulder_centers, dtype=np.float32), device=device)
        s = batch_Y[..., colmap.y_shoulder].unsqueeze(-1)
        d2s = (s - centers) ** 2
        y_shoulder_idx = torch.argmin(d2s, dim=-1)

    return {
        "main_idx": y_main_idx,
        "c_idx": y_c_idx,
        "buttons": y_buttons,
        "shoulder_idx": y_shoulder_idx,
        "main_K": P_main.shape[0],
        "c_K": P_c.shape[0],
        "buttons_K": y_buttons.shape[-1],
        "shoulder_K": (
            len(shoulder_centers) if (colmap.y_shoulder is not None and shoulder_centers is not None) else 0
        ),
    }
