from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from controller_utils import CONTROL_STICK_QUANTIZED, C_STICK_QUANTIZED, SHOULDER_QUANTIZED


_MAIN_STICK_PALETTE_CPU = torch.as_tensor(
    np.asarray(CONTROL_STICK_QUANTIZED, dtype=np.float32)
)
_C_STICK_PALETTE_CPU = torch.as_tensor(
    np.asarray(C_STICK_QUANTIZED, dtype=np.float32)
)
_SHOULDER_PALETTE_CPU = torch.as_tensor(
    np.asarray(SHOULDER_QUANTIZED, dtype=np.float32)
) if SHOULDER_QUANTIZED else None

# Precompute palette norm squared for faster distance calculations
_MAIN_STICK_NORM_SQ_CPU = (_MAIN_STICK_PALETTE_CPU * _MAIN_STICK_PALETTE_CPU).sum(dim=1)
_C_STICK_NORM_SQ_CPU = (_C_STICK_PALETTE_CPU * _C_STICK_PALETTE_CPU).sum(dim=1)

_MAIN_STICK_CACHE: Dict[Tuple[str, Optional[int]], torch.Tensor] = {}
_C_STICK_CACHE: Dict[Tuple[str, Optional[int]], torch.Tensor] = {}
_SHOULDER_CACHE: Dict[Tuple[str, Optional[int]], torch.Tensor] = {}
_MAIN_STICK_NORM_CACHE: Dict[Tuple[str, Optional[int]], torch.Tensor] = {}
_C_STICK_NORM_CACHE: Dict[Tuple[str, Optional[int]], torch.Tensor] = {}


def sticks01_to_unit11(xy01: torch.Tensor) -> torch.Tensor:
    """Map controller coordinates from [0,1] to [-1,1] and clamp to the unit circle."""
    xy11 = torch.clamp(xy01 * 2.0 - 1.0, -1.0, 1.0)
    return _clamp_unit_circle(xy11)


def _clamp_unit_circle(xy11: torch.Tensor) -> torch.Tensor:
    """Clamp arbitrary stick coordinates in [-1,1] to the unit circle."""
    # Optimized: use squared norm to avoid sqrt, then only normalize if needed
    radius_sq = (xy11 * xy11).sum(dim=-1, keepdim=True)
    # Only normalize if radius > 1
    scale = torch.where(radius_sq > 1.0, torch.sqrt(radius_sq.clamp_min(1e-12)), torch.ones_like(radius_sq))
    return xy11 / scale


def _device_cache_key(device: torch.device) -> Tuple[str, Optional[int]]:
    """Canonicalize device key to support caching with and without explicit indices."""
    if device.type == "cuda":
        index = device.index
        if index is None and torch.cuda.is_available():
            index = torch.cuda.current_device()
        return device.type, index
    return device.type, device.index


def _palette_for_device(
    cpu_palette: torch.Tensor,
    cache: Dict[Tuple[str, Optional[int]], torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Return palette tensor on the requested device, caching to avoid repeated copies."""
    if device.type == "cpu":
        return cpu_palette
    key = _device_cache_key(device)
    cached = cache.get(key)
    if cached is None or cached.device != device:
        cached = cpu_palette.to(device=device)
        cache[key] = cached
    return cached


def _quantize_stick(xy: torch.Tensor, palette: torch.Tensor, palette_norm_sq: torch.Tensor, 
                    input_domain: str, B: int, L: int) -> torch.Tensor:
    """Helper to quantize a stick to palette indices. Optimized to reduce redundant code."""
    if input_domain == "unit11":
        xy11 = _clamp_unit_circle(torch.clamp(xy, -1.0, 1.0))
    elif input_domain == "unit01":
        xy11 = sticks01_to_unit11(xy)
    else:  # auto
        needs_clamp = torch.any(xy < 0.0) or torch.any(xy > 1.0)
        xy11 = _clamp_unit_circle(torch.clamp(xy, -1.0, 1.0)) if needs_clamp else sticks01_to_unit11(xy)
    
    # Quantize using squared distance (avoids redundant pow/sum calls)
    V = xy11.reshape(-1, 2)
    # Compute ||V - P||^2 = ||V||^2 - 2*V·P + ||P||^2 (palette norm is precomputed)
    v_norm_sq = (V * V).sum(dim=1, keepdim=True)
    dot = V @ palette.t()
    d2 = v_norm_sq - 2.0 * dot + palette_norm_sq.unsqueeze(0)
    return torch.argmin(d2, dim=1).view(B, L)


def quantize_targets(
        batch_Y: torch.FloatTensor,
        colmap,
        *,
        input_domain: str = "auto",
) -> Dict[str, torch.Tensor]:
    """Convert controller targets to palette indices.

    Parameters
    ----------
    batch_Y:
        Target tensor [..., features].
    colmap:
        Column map describing which slices correspond to controller fields.
    input_domain:
        "unit01" if the incoming stick coordinates are in [0,1],
        "unit11" if they are already in [-1,1],
        "auto" (default) to inspect the data and choose automatically.
    """
    B, L, _ = batch_Y.shape
    device = batch_Y.device

    # Main stick quantization
    main_xy = batch_Y[..., list(colmap.y_main)]
    P_main = _palette_for_device(_MAIN_STICK_PALETTE_CPU, _MAIN_STICK_CACHE, device)
    P_main_norm_sq = _palette_for_device(_MAIN_STICK_NORM_SQ_CPU, _MAIN_STICK_NORM_CACHE, device)
    y_main_idx = _quantize_stick(main_xy, P_main, P_main_norm_sq, input_domain, B, L)

    # C-stick quantization
    c_xy = batch_Y[..., list(colmap.y_c)]
    P_c = _palette_for_device(_C_STICK_PALETTE_CPU, _C_STICK_CACHE, device)
    P_c_norm_sq = _palette_for_device(_C_STICK_NORM_SQ_CPU, _C_STICK_NORM_CACHE, device)
    y_c_idx = _quantize_stick(c_xy, P_c, P_c_norm_sq, input_domain, B, L)

    # Buttons remain probabilistic targets
    btn_cols = colmap.y_buttons
    y_buttons = batch_Y[..., btn_cols].to(torch.float32)
    y_buttons = torch.clamp(y_buttons, 0.0, 1.0)

    y_shoulder_idx = None
    shoulder_K = 0
    if getattr(colmap, "y_shoulder", None) is not None:
        if _SHOULDER_PALETTE_CPU is None:
            raise RuntimeError("Shoulder quantization palette requested but not defined.")
        centers = _palette_for_device(_SHOULDER_PALETTE_CPU, _SHOULDER_CACHE, device)
        s = batch_Y[..., colmap.y_shoulder].unsqueeze(-1)
        d2s = (s - centers) ** 2
        y_shoulder_idx = torch.argmin(d2s, dim=-1)
        shoulder_K = len(SHOULDER_QUANTIZED)

    return {
        "main_idx": y_main_idx,
        "c_idx": y_c_idx,
        "buttons": y_buttons,
        "shoulder_idx": y_shoulder_idx,
        "main_K": P_main.shape[0],
        "c_K": P_c.shape[0],
        "buttons_K": y_buttons.shape[-1],
        "shoulder_K": shoulder_K,
    }
