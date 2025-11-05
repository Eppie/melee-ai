from __future__ import annotations

from typing import Tuple

from torch import Tensor


def sticks01_to_unit11_torch(xy01: Tensor) -> Tensor:
    """[0,1] -> [-1,1] (vectorized, PyTorch)."""
    return xy01.mul(2.0).sub(1.0)


def unit11_to_sticks01_torch(xy11: Tensor) -> Tensor:
    """[-1,1] -> [0,1] (vectorized, PyTorch)."""
    return xy11.add(1.0).mul(0.5)


def quantize_unit11_to_palette_torch(
        xy11: Tensor, palette11: Tensor, *, return_index: bool = False
) -> Tensor | Tuple[Tensor, Tensor]:
    """
    Quantize inputs in [-1,1]^D to nearest palette vector (PyTorch).
    Shapes:
      xy11: (..., D)
      palette11: (K, D)
    """
    assert xy11.shape[-1] == palette11.shape[-1], "Dimensionality mismatch"
    assert xy11.dtype == palette11.dtype, "dtype mismatch"
    assert xy11.device == palette11.device, "device mismatch"

    orig = xy11.shape
    d = orig[-1]
    x = xy11.reshape(-1, d)  # (N, D)
    p = palette11  # (K, D)

    x_norm = (x * x).sum(dim=1, keepdim=True)  # (N, 1)
    p_norm = (p * p).sum(dim=1).unsqueeze(0)  # (1, K)
    dist2 = x_norm + p_norm - 2.0 * (x @ p.T)  # (N, K)

    idx = dist2.argmin(dim=1)  # (N,)
    quant = p.index_select(0, idx).reshape(orig)  # (..., D)

    if return_index:
        return quant, idx.reshape(orig[:-1])
    return quant


def scale_torch(x: Tensor, factor: float) -> Tensor:
    """Elementwise: return x * factor (PyTorch). Works for any shape/dtype/device."""
    return x.mul(factor)


def bit01_to_sign11_torch(x: Tensor) -> Tensor:
    """Elementwise map: 0.0 --> -1.0, 1.0 --> 1.0"""
    return x.mul(2.0).sub(1.0)
