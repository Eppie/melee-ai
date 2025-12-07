from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _sticks01_to_unit11(arr: Any):
    """Convert [0,1] stick coords to [-1,1] and clamp to unit circle."""
    if torch.is_tensor(arr):
        xy11 = arr * 2.0 - 1.0
        norm = torch.linalg.norm(xy11, dim=-1, keepdim=True)
        over = norm > 1.0
        if over.any():
            scale = torch.where(over, norm, torch.ones_like(norm))
            xy11 = xy11 / scale
        return xy11
    xy11 = arr * 2.0 - 1.0
    norms = np.linalg.norm(xy11, axis=-1, keepdims=True)
    over = norms > 1.0
    if np.any(over):
        scale = np.where(over, norms, 1.0)
        xy11 = xy11 / scale
    return xy11


def _clamp_unit_circle(arr: Any):
    """Clamp vectors in [-1,1] domain to unit circle."""
    if torch.is_tensor(arr):
        norm = torch.linalg.norm(arr, dim=-1, keepdim=True)
        over = norm > 1.0
        if over.any():
            scale = torch.where(over, norm, torch.ones_like(norm))
            arr = arr / scale
        return arr
    norm = np.linalg.norm(arr, axis=-1, keepdims=True)
    over = norm > 1.0
    if np.any(over):
        scale = np.where(over, norm, 1.0)
        arr = arr / scale
    return arr


def quantize_stick_indices(xy, palette, palette_norm_sq, input_domain: str = "auto"):
    """
    Snap stick coordinates to nearest palette entry using shared geometry.

    Supports both numpy arrays and torch tensors. Returns indices shaped like xy[...,0].
    """
    is_tensor = torch.is_tensor(xy)

    if is_tensor:
        if input_domain == "unit11":
            xy11 = _clamp_unit_circle(torch.clamp(xy, -1.0, 1.0))
        elif input_domain == "unit01":
            xy11 = _sticks01_to_unit11(xy)
        else:
            needs_clamp = torch.any(xy < 0.0) or torch.any(xy > 1.0)
            xy11 = (
                _clamp_unit_circle(torch.clamp(xy, -1.0, 1.0))
                if needs_clamp
                else _sticks01_to_unit11(xy)
            )

        V = xy11.reshape(-1, 2)
        v_norm_sq = (V * V).sum(dim=1, keepdim=True)
        dot = V @ palette.t()
        palette_ns = palette_norm_sq.view(1, -1)
        d2 = v_norm_sq - 2.0 * dot + palette_ns
        flat_idx = torch.argmin(d2, dim=1)
        return flat_idx.view(*xy.shape[:-1])

    # numpy branch
    arr = np.asarray(xy, dtype=np.float32)
    palette_np = np.asarray(palette, dtype=np.float32)
    palette_norm = np.asarray(palette_norm_sq, dtype=np.float32).reshape(-1)

    if input_domain == "unit11":
        xy11 = _clamp_unit_circle(np.clip(arr, -1.0, 1.0))
    elif input_domain == "unit01":
        xy11 = _sticks01_to_unit11(arr)
    else:
        needs_clamp = np.any(arr < 0.0) or np.any(arr > 1.0)
        if needs_clamp:
            xy11 = _clamp_unit_circle(np.clip(arr, -1.0, 1.0))
        else:
            xy11 = _sticks01_to_unit11(arr)

    V = xy11.reshape(-1, 2)
    dot = V @ palette_np.T
    norm = np.sum(V * V, axis=1, keepdims=True)
    d2 = norm - 2.0 * dot + palette_norm.reshape(1, -1)
    idx = np.argmin(d2, axis=1)
    return idx.reshape(*xy.shape[:-1])
