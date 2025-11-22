from __future__ import annotations

import numpy as np
import pytest

from config import FeatureConfig
from controller_utils import C_STICK_QUANTIZED
from feature_transforms import (
    _normalize_features,
    _sticks01_to_unit11_np,
    build_transform_spec,
    feature_spec_from_config,
)


def _nearest_palette_rows(points: np.ndarray, palette: np.ndarray) -> np.ndarray:
    dists = ((palette[None, ...] - points[:, None, :]) ** 2).sum(axis=2)
    idx = np.argmin(dists, axis=1)
    return palette[idx]


def test_feature_spec_from_config_matches_default_transform_list():
    cfg = FeatureConfig()
    spec = feature_spec_from_config(cfg)

    assert spec is not None
    assert len(spec.steps) == len(cfg.transforms)
    assert spec.steps[0].transform == "stick_palette"
    assert spec.steps[1].transform == "stick_palette"


def test_build_transform_spec_supports_palette_alias_and_applies_fn():
    transforms = [
        {
            "transform": "stick_palette",
            "features": ["c_stick_x", "c_stick_y"],
            "palette": "c",  # alias for c_stick palette
        }
    ]
    spec = build_transform_spec(transforms)
    assert spec is not None
    step = spec.steps[0]

    block = np.array([[0.2, 0.2], [0.9, 0.9]], dtype=np.float32)
    out = step.fn(block.copy())
    block_unit = _sticks01_to_unit11_np(block.copy())
    expected = _nearest_palette_rows(
        block_unit, np.asarray(C_STICK_QUANTIZED, dtype=np.float32)
    )
    assert np.allclose(out, expected)


def test_build_transform_spec_rejects_unknown_transform():
    with pytest.raises(KeyError):
        build_transform_spec([{"transform": "bogus", "features": "foo"}])


def test_normalize_features_rejects_invalid_inputs():
    with pytest.raises(TypeError):
        _normalize_features(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        _normalize_features([])
