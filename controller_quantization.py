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

_MAIN_STICK_CACHE: Dict[Tuple[str, Optional[int]], torch.Tensor] = {}
_C_STICK_CACHE: Dict[Tuple[str, Optional[int]], torch.Tensor] = {}
_SHOULDER_CACHE: Dict[Tuple[str, Optional[int]], torch.Tensor] = {}


def sticks01_to_unit11(xy01: torch.Tensor) -> torch.Tensor:
    """Map controller coordinates from [0,1] to [-1,1] and clamp to the unit circle."""
    xy11 = torch.clamp(xy01 * 2.0 - 1.0, -1.0, 1.0)
    return _clamp_unit_circle(xy11)


def _clamp_unit_circle(xy11: torch.Tensor) -> torch.Tensor:
    """Clamp arbitrary stick coordinates in [-1,1] to the unit circle."""
    radius = torch.linalg.norm(xy11, dim=-1, keepdim=True)
    scale = torch.clamp(radius, min=1.0)
    return xy11 / scale.clamp_min(1e-12)


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

    # Main stick palette lookup
    main_xy = batch_Y[..., list(colmap.y_main)]
    if input_domain == "unit11":
        main_xy11 = _clamp_unit_circle(torch.clamp(main_xy, -1.0, 1.0))
    elif input_domain == "unit01":
        main_xy11 = sticks01_to_unit11(main_xy)
    else:  # auto
        if torch.any(main_xy < 0.0) or torch.any(main_xy > 1.0):
            main_xy11 = _clamp_unit_circle(torch.clamp(main_xy, -1.0, 1.0))
        else:
            main_xy11 = sticks01_to_unit11(main_xy)
    P_main = _palette_for_device(_MAIN_STICK_PALETTE_CPU, _MAIN_STICK_CACHE, device)
    V_main = main_xy11.reshape(-1, 2)
    main_norm = V_main.pow(2).sum(dim=1, keepdim=True)
    palette_norm = P_main.pow(2).sum(dim=1).unsqueeze(0)
    dot = V_main @ P_main.t()
    d2 = main_norm - 2.0 * dot + palette_norm
    y_main_idx = torch.argmin(d2, dim=1).view(B, L)

    # C-stick palette lookup
    c_xy = batch_Y[..., list(colmap.y_c)]
    if input_domain == "unit11":
        c_xy11 = _clamp_unit_circle(torch.clamp(c_xy, -1.0, 1.0))
    elif input_domain == "unit01":
        c_xy11 = sticks01_to_unit11(c_xy)
    else:
        if torch.any(c_xy < 0.0) or torch.any(c_xy > 1.0):
            c_xy11 = _clamp_unit_circle(torch.clamp(c_xy, -1.0, 1.0))
        else:
            c_xy11 = sticks01_to_unit11(c_xy)
    P_c = _palette_for_device(_C_STICK_PALETTE_CPU, _C_STICK_CACHE, device)
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
