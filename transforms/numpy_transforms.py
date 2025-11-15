import numpy as np
from numpy.typing import NDArray


def sticks01_to_unit11_np(xy01: NDArray[np.floating]) -> NDArray[np.floating]:
    """[0,1] -> [-1,1] (vectorized, NumPy)."""
    return xy01 * 2.0 - 1.0


def unit11_to_sticks01_np(xy11: NDArray[np.floating]) -> NDArray[np.floating]:
    """[-1,1] -> [0,1] (vectorized, NumPy)."""
    return (xy11 + 1.0) * 0.5


def quantize_unit11_to_palette_np(
    xy11: NDArray[np.floating],
    palette11: NDArray[np.floating],
    *,
    return_index: bool = False,
) -> NDArray[np.floating] | tuple[NDArray[np.floating], NDArray[np.intp]]:
    """
    Quantize inputs in [-1,1]^D to nearest palette vector (NumPy).
    Shapes:
      xy11: (..., D)
      palette11: (K, D)
    """
    if xy11.shape[-1] != palette11.shape[-1]:
        raise ValueError("Dimensionality mismatch")

    orig = xy11.shape
    d = orig[-1]
    x = xy11.reshape(-1, d)  # (N, D)
    p = palette11  # (K, D)

    x_norm = (x * x).sum(axis=1, keepdims=True)  # (N, 1)
    p_norm = (p * p).sum(axis=1)[np.newaxis, :]  # (1, K)
    dist2 = x_norm + p_norm - 2.0 * (x @ p.T)  # (N, K)

    idx = dist2.argmin(axis=1).astype(np.intp)  # (N,)
    quant = p[idx].reshape(orig)  # (..., D)

    if return_index:
        return quant, idx.reshape(orig[:-1])
    return quant


def scale_np(x: np.ndarray, factor: float) -> np.ndarray:
    """Elementwise: return x * factor (NumPy). Works for any shape/dtype."""
    return x * factor


def bit01_to_sign11_np(x: NDArray[np.floating]) -> NDArray[np.floating]:
    """Elementwise map: 0.0 --> -1.0, 1.0 --> 1.0"""
    return x * 2.0 - 1.0
