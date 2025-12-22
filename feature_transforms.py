"""Hardcoded feature transforms for dataset preprocessing and inference."""

from __future__ import annotations

from typing import Dict, List, Sequence, TYPE_CHECKING

import numpy as np

from controller_utils import C_STICK_QUANTIZED, CONTROL_STICK_QUANTIZED
from controller_quantization_shared import (
    quantize_stick_indices_unit01,
    sticks01_to_unit11,
)

_sticks01_to_unit11 = sticks01_to_unit11

if TYPE_CHECKING:
    from column_map import ColumnMap

# Stick palettes for quantization
MAIN_PALETTE = np.asarray(CONTROL_STICK_QUANTIZED, dtype=np.float32)
C_PALETTE = np.asarray(C_STICK_QUANTIZED, dtype=np.float32)
_MAIN_PALETTE_NORM = np.sum(MAIN_PALETTE**2, axis=1, keepdims=True)
_C_PALETTE_NORM = np.sum(C_PALETTE**2, axis=1, keepdims=True)

# Transform constants
_SCALE_FACTORS: Dict[str, float] = {
    "percent": 1 / 100.0,
    "shield_strength": 1.0 / 60.0,
    "position_x": 1 / 20.0,
    "position_y": 1 / 20.0,
    "jumps_left": 1 / 6.0,
}


def _quantize_stick(
    block: np.ndarray,
    palette: np.ndarray,
    palette_norm: np.ndarray,
) -> np.ndarray:
    """Snap (x, y) stick pairs to the nearest palette entry."""
    idx = quantize_stick_indices_unit01(block, palette, palette_norm)
    return palette[idx]


def _get_column_indices(
    feature_names: Sequence[str], base_names: Sequence[str]
) -> List[List[int]]:
    """Find column indices for base feature names across player prefixes."""
    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}

    # Try exact match first
    if all(name in name_to_idx for name in base_names):
        return [[name_to_idx[name] for name in base_names]]

    # Find player prefixes (p1_, p2_, etc.)
    prefixes = set()
    for name in feature_names:
        head, _, tail = name.partition("_")
        if tail and head.startswith("p") and head[1:].isdigit():
            prefixes.add(head)

    groups = []
    for prefix in sorted(prefixes):
        indices = []
        found_all = True
        for base_name in base_names:
            col = f"{prefix}_{base_name}"
            if col not in name_to_idx:
                found_all = False
                break
            indices.append(name_to_idx[col])
        if found_all and indices:
            groups.append(indices)
    return groups


def apply_feature_transforms(
    features: np.ndarray,
    feature_names: Sequence[str],
) -> np.ndarray:
    """Apply all hardcoded feature transforms in-place.

    Transforms applied:
    - Main stick (x, y): quantize to 64-position palette
    - C-stick (x, y): quantize to 9-position palette
    - facing: scale by 2.0, offset by -1.0
    - percent: scale by 1/100
    - shield_strength: scale by 1/60
    - position_x, position_y: scale by 1/20
    - jumps_left: scale by 1/6
    """
    out = features

    # Main stick palette quantization
    for indices in _get_column_indices(feature_names, ["main_stick_x", "main_stick_y"]):
        idx_arr = np.array(indices, dtype=np.int32)
        block = out[:, idx_arr].copy()
        out[:, idx_arr] = _quantize_stick(block, MAIN_PALETTE, _MAIN_PALETTE_NORM)

    # C-stick palette quantization
    for indices in _get_column_indices(feature_names, ["c_stick_x", "c_stick_y"]):
        idx_arr = np.array(indices, dtype=np.int32)
        block = out[:, idx_arr].copy()
        out[:, idx_arr] = _quantize_stick(block, C_PALETTE, _C_PALETTE_NORM)

    # Facing: scale by 2.0 then offset by -1.0
    for indices in _get_column_indices(feature_names, ["facing"]):
        idx = indices[0]
        out[:, idx] = out[:, idx] * 2.0 - 1.0

    # Apply scale factors
    for base_name, factor in _SCALE_FACTORS.items():
        for indices in _get_column_indices(feature_names, [base_name]):
            idx = indices[0]
            out[:, idx] *= factor

    return out


def apply_feature_transforms_dict(features: Dict[str, float]) -> Dict[str, float]:
    """Apply transforms to a single-frame feature dictionary (for inference).

    This provides the same transforms as apply_feature_transforms but operates
    on a Dict[str, float] instead of a numpy array.
    """
    out = dict(features)
    keys = set(out.keys())

    # Find player prefixes
    prefixes = set()
    for key in keys:
        head, _, tail = key.partition("_")
        if tail and head.startswith("p") and head[1:].isdigit():
            prefixes.add(head)

    def _get_keys(base_names: Sequence[str]) -> List[List[str]]:
        """Get key groups for base names across prefixes."""
        if all(name in keys for name in base_names):
            return [list(base_names)]
        groups = []
        for prefix in sorted(prefixes):
            group = [f"{prefix}_{name}" for name in base_names]
            if all(k in keys for k in group):
                groups.append(group)
        return groups

    # Main stick palette quantization
    for key_group in _get_keys(["main_stick_x", "main_stick_y"]):
        block = np.array([[out[key_group[0]], out[key_group[1]]]], dtype=np.float32)
        result = _quantize_stick(block, MAIN_PALETTE, _MAIN_PALETTE_NORM)
        out[key_group[0]] = float(result[0, 0])
        out[key_group[1]] = float(result[0, 1])

    # C-stick palette quantization
    for key_group in _get_keys(["c_stick_x", "c_stick_y"]):
        block = np.array([[out[key_group[0]], out[key_group[1]]]], dtype=np.float32)
        result = _quantize_stick(block, C_PALETTE, _C_PALETTE_NORM)
        out[key_group[0]] = float(result[0, 0])
        out[key_group[1]] = float(result[0, 1])

    # Facing: scale by 2.0 then offset by -1.0
    for key_group in _get_keys(["facing"]):
        key = key_group[0]
        out[key] = out[key] * 2.0 - 1.0

    # Apply scale factors
    for base_name, factor in _SCALE_FACTORS.items():
        for key_group in _get_keys([base_name]):
            key = key_group[0]
            out[key] = out[key] * factor

    return out


__all__ = [
    "MAIN_PALETTE",
    "C_PALETTE",
    "apply_feature_transforms",
    "apply_feature_transforms_dict",
    "_quantize_stick",
    "_sticks01_to_unit11",
]
