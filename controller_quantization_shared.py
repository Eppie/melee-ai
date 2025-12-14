from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch

InputDomain = Literal["unit01", "unit11"]
_EPS = 1e-7


def _assert_range_unit01(arr: Any) -> None:
    """Ensure inputs meant to be in [0,1] are not drifting."""
    if torch.is_tensor(arr):
        min_val = torch.min(arr).item()
        max_val = torch.max(arr).item()
    else:
        arr = np.asarray(arr)
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
    if min_val < -_EPS or max_val > 1.0 + _EPS:
        raise ValueError(f"unit01 input out of range: min={min_val}, max={max_val}")


def _assert_range_unit11(arr: Any) -> None:
    """Ensure inputs meant to be in [-1,1] are not drifting."""
    if torch.is_tensor(arr):
        min_val = torch.min(arr).item()
        max_val = torch.max(arr).item()
    else:
        arr = np.asarray(arr)
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
    if min_val < -1.0 - _EPS or max_val > 1.0 + _EPS:
        raise ValueError(f"unit11 input out of range: min={min_val}, max={max_val}")


def clamp_unit_circle(arr: Any):
    """Clamp vectors in [-1,1] domain to the unit circle."""
    if torch.is_tensor(arr):
        _assert_range_unit11(arr)
        radius_sq = (arr * arr).sum(dim=-1, keepdim=True)
        scale = torch.where(
            radius_sq > 1.0,
            torch.sqrt(radius_sq.clamp_min(1e-12)),
            torch.ones_like(radius_sq),
        )
        return arr / scale
    arr_np = np.asarray(arr, dtype=np.float32)
    _assert_range_unit11(arr_np)
    radius_sq = np.sum(arr_np * arr_np, axis=-1, keepdims=True)
    scale = np.where(radius_sq > 1.0, np.sqrt(radius_sq), 1.0)
    return arr_np / scale


def sticks01_to_unit11(arr: Any):
    """Convert [0,1] stick coords to [-1,1] and clamp to unit circle."""
    if torch.is_tensor(arr):
        _assert_range_unit01(arr)
        xy01 = torch.clamp(arr, 0.0, 1.0)
        xy11 = xy01 * 2.0 - 1.0
        return clamp_unit_circle(xy11)
    arr_np = np.asarray(arr, dtype=np.float32)
    _assert_range_unit01(arr_np)
    xy01 = np.clip(arr_np, 0.0, 1.0)
    xy11 = xy01 * 2.0 - 1.0
    return clamp_unit_circle(xy11)


def _quantize_from_unit11(xy11: Any, palette: Any, palette_norm_sq: Any):
    """Shared quantization assuming inputs are already in [-1, 1] space."""
    is_tensor = torch.is_tensor(xy11)

    if is_tensor:
        V = xy11.reshape(-1, 2)
        v_norm_sq = (V * V).sum(dim=1, keepdim=True)
        dot = V @ palette.t()
        palette_ns = palette_norm_sq.view(1, -1)
        d2 = v_norm_sq - 2.0 * dot + palette_ns
        flat_idx = torch.argmin(d2, dim=1)
        return flat_idx.view(*xy11.shape[:-1])

    V = xy11.reshape(-1, 2)
    palette_np = np.asarray(palette, dtype=np.float32)
    palette_norm = np.asarray(palette_norm_sq, dtype=np.float32).reshape(-1)
    dot = V @ palette_np.T
    norm = np.sum(V * V, axis=1, keepdims=True)
    d2 = norm - 2.0 * dot + palette_norm.reshape(1, -1)
    idx = np.argmin(d2, axis=1)
    return idx.reshape(*xy11.shape[:-1])


def quantize_stick_indices_unit01(xy, palette, palette_norm_sq):
    """Quantize stick coordinates provided in [0,1] space."""
    xy11 = sticks01_to_unit11(xy)
    return _quantize_from_unit11(xy11, palette, palette_norm_sq)


def quantize_stick_indices_unit11(xy, palette, palette_norm_sq):
    """Quantize stick coordinates provided in [-1,1] space."""
    xy = clamp_unit_circle(xy)
    return _quantize_from_unit11(xy, palette, palette_norm_sq)


__all__ = [
    "InputDomain",
    "clamp_unit_circle",
    "sticks01_to_unit11",
    "quantize_stick_indices_unit01",
    "quantize_stick_indices_unit11",
]
