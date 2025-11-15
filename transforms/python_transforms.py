from typing import Sequence


def sticks01_to_unit11_py(x: float) -> float:
    """[0,1] -> [-1,1] for a single scalar (pure Python)."""
    return x * 2.0 - 1.0


def unit11_to_sticks01_py(x: float) -> float:
    """[-1,1] -> [0,1] for a single scalar (pure Python)."""
    return (x + 1.0) * 0.5


def quantize_unit11_to_palette_py(
    x: Sequence[float],
    palette11: Sequence[Sequence[float]],
    *,
    return_index: bool = False,
) -> list[float] | tuple[list[float], int]:
    """
    Quantize a single point in [-1,1]^D to the nearest palette entry (pure Python).

    Args:
        x: D-length sequence (e.g., [x, y] for sticks).
        palette11: sequence of K entries, each a D-length sequence.

    Returns:
        The nearest palette vector as a list[float], and optionally the index.
    """
    best_i = -1
    best_d2 = float("inf")
    for i, p in enumerate(palette11):
        # squared L2 distance
        d2 = 0.0
        for xi, pi in zip(x, p):
            diff = xi - pi
            d2 += diff * diff
        if d2 < best_d2:
            best_d2 = d2
            best_i = i

    q = list(palette11[best_i])
    if return_index:
        return q, best_i
    return q


def scale(x: float, factor: float) -> float:
    return x * factor


def bit01_to_sign11(x: float) -> float:
    """Map 0.0 → -1.0 and 1.0 → 1.0. Assumes x ∈ {0.0, 1.0}."""
    return x * 2.0 - 1.0
