"""Compute ground truth future position deltas.

For each frame, compute the actual position deltas at each horizon.
"""

import numpy as np
from typing import List, Tuple


def compute_future_deltas(
    positions: np.ndarray,
    horizons: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute future position deltas for all frames.

    Args:
        positions: [n_frames, 4] array (p1_x, p1_y, p2_x, p2_y)
        horizons: List of frame offsets to predict (e.g., [5, 10, 20, ...])

    Returns:
        deltas: [n_frames, n_horizons, 4] position deltas
        valid: [n_frames, n_horizons] bool mask (False for end-of-episode)

    Notes:
        - delta[t, h] = position[t + h] - position[t]
        - Mark invalid if t + h >= n_frames (episode ends)
    """
    n_frames = len(positions)
    n_horizons = len(horizons)

    deltas = np.zeros((n_frames, n_horizons, 4), dtype=np.float32)
    valid = np.zeros((n_frames, n_horizons), dtype=bool)

    for t in range(n_frames):
        for i, h in enumerate(horizons):
            if t + h < n_frames:
                deltas[t, i] = positions[t + h] - positions[t]
                valid[t, i] = True
            else:
                # Deliberately leave deltas as zeros for invalid, as per the comment,
                # and rely on the valid mask to ignore these entries.
                pass 
    return deltas, valid


def normalize_deltas(
    deltas: np.ndarray,
    stage_half_width: float,
    stage_half_height: float,
) -> np.ndarray:
    """Normalize position deltas to standard scale.

    Args:
        deltas: [n_frames, n_horizons, 4] position deltas
        stage_half_width: Normalization constant for X
        stage_half_height: Normalization constant for Y

    Returns:
        Normalized deltas where X deltas / stage_half_width, Y deltas / stage_half_height
    """
    deltas_norm = deltas.copy()
    deltas_norm[..., [0, 2]] /= stage_half_width   # p1_x, p2_x deltas
    deltas_norm[..., [1, 3]] /= stage_half_height  # p1_y, p2_y deltas
    return deltas_norm


def denormalize_deltas(
    deltas_norm: np.ndarray,
    stage_half_width: float,
    stage_half_height: float,
) -> np.ndarray:
    """Convert normalized deltas back to game units.

    Args:
        deltas_norm: [n_frames, n_horizons, 4] normalized deltas
        stage_half_width: Normalization constant for X
        stage_half_height: Normalization constant for Y

    Returns:
        Deltas in game units
    """
    deltas = deltas_norm.copy()
    deltas[..., [0, 2]] *= stage_half_width   # p1_x, p2_x deltas
    deltas[..., [1, 3]] *= stage_half_height  # p1_y, p2_y deltas
    return deltas
