"""Shared feature transform helpers for dataset preprocessing and inference."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TYPE_CHECKING,
)

import numpy as np

from controller_utils import C_STICK_QUANTIZED, CONTROL_STICK_QUANTIZED

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from config import FeatureConfig

FeatureFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class FeatureTransformStep:
    """Declarative description of a single transform application."""

    transform: str
    features: Tuple[str, ...]
    fn: FeatureFn


@dataclass(frozen=True)
class FeatureTransformSpec:
    """A sequence of transform steps that should run in order."""

    steps: Tuple[FeatureTransformStep, ...]

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return bool(self.steps)


def _transform_scale(column: np.ndarray, *, factor: float) -> np.ndarray:
    column *= factor
    return column


def _transform_offset(column: np.ndarray, *, delta: float) -> np.ndarray:
    column += delta
    return column


TransformFactory = Callable[[Mapping[str, Any]], FeatureFn]


def _sticks01_to_unit11_np(xy01: np.ndarray) -> np.ndarray:
    xy01_clipped = np.clip(xy01, 0.0, 1.0)
    xy11 = xy01_clipped * 2.0 - 1.0
    norms = np.linalg.norm(xy11, axis=1, keepdims=True)
    mask = norms > 1.0
    if np.any(mask):
        xy11[mask] /= norms[mask]
    return xy11


_MAIN_PALETTE = np.asarray(CONTROL_STICK_QUANTIZED, dtype=np.float32)
_C_PALETTE = np.asarray(C_STICK_QUANTIZED, dtype=np.float32)

_PALETTES: Dict[str, np.ndarray] = {
    "fox_main": _MAIN_PALETTE,
    "main": _MAIN_PALETTE,
    "c_stick": _C_PALETTE,
    "c": _C_PALETTE,
}


def _stick_palette_apply(
    block: np.ndarray,
    *,
    palette: np.ndarray,
    palette_norm: np.ndarray,
) -> np.ndarray:
    if block.shape[1] != 2:
        raise ValueError(
            "stick_palette transform expects exactly two feature columns (x, y)."
        )
    values = block.astype(np.float32, copy=False)
    if np.any(values < 0.0) or np.any(values > 1.0):
        xy11 = np.clip(values, -1.0, 1.0).copy()
        norms = np.linalg.norm(xy11, axis=1, keepdims=True)
        mask = norms > 1.0
        if np.any(mask):
            xy11[mask] /= norms[mask]
    else:
        xy01 = np.clip(values, 0.0, 1.0)
        xy11 = _sticks01_to_unit11_np(xy01.copy())
    dot = xy11 @ palette.T
    norm = np.sum(xy11**2, axis=1, keepdims=True)
    d2 = norm - 2.0 * dot + palette_norm.T
    idx = np.argmin(d2, axis=1)
    quantized11 = palette[idx]
    block[...] = quantized11
    return block


def _factory_stick_palette(params: Mapping[str, Any]) -> FeatureFn:
    palette_key = str(params.get("palette", "fox_main")).lower()
    palette = _PALETTES.get(palette_key)
    if palette is None:
        raise ValueError(f"Unknown stick palette '{palette_key}'.")

    palette = palette.astype(np.float32, copy=False)
    palette_norm = np.sum(palette**2, axis=1, keepdims=True)

    return partial(_stick_palette_apply, palette=palette, palette_norm=palette_norm)


def _factory_scale(params: Mapping[str, Any]) -> FeatureFn:
    raw = params.get("factor", params.get("scale", 1.0))
    try:
        factor = float(raw)
    except (TypeError, ValueError) as exc:
        raise TypeError("Scale transform requires a numeric 'factor'.") from exc
    return partial(_transform_scale, factor=factor)


def _factory_offset(params: Mapping[str, Any]) -> FeatureFn:
    raw = params.get("delta", params.get("offset", params.get("value")))
    if raw is None:
        raise TypeError("Offset transform requires a 'delta'.")
    try:
        delta = float(raw)
    except (TypeError, ValueError) as exc:
        raise TypeError("Offset transform requires a numeric 'delta'.") from exc
    return partial(_transform_offset, delta=delta)


_TRANSFORM_FACTORIES: Dict[str, TransformFactory] = {
    "scale": _factory_scale,
    "offset": _factory_offset,
    "stick_palette": _factory_stick_palette,
}


def _resolve_registered_transform(name: str, params: Mapping[str, Any]) -> FeatureFn:
    factory = _TRANSFORM_FACTORIES.get(name)
    if factory is None:
        raise ValueError(f"Unknown feature transform '{name}'.")
    return factory(params)


def _normalize_features(raw: Any) -> Tuple[str, ...]:
    if isinstance(raw, str):
        raw_features: Iterable[str] = (raw,)
    elif isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray)):
        raw_features = raw
    else:
        raise TypeError("'features' must be a string or sequence of strings.")
    out = tuple(str(f).strip() for f in raw_features if str(f).strip())
    if not out:
        raise ValueError("At least one feature name must be provided.")
    return out


def _extract_params(
    step: Mapping[str, Any], base_keys: Sequence[str]
) -> Mapping[str, Any]:
    params = step.get("params")
    if params is None:
        params = step.get("parameters")
    if params is None:
        params = {k: v for k, v in step.items() if k not in base_keys}
    if not isinstance(params, Mapping):
        raise TypeError("Transform 'params' must be a mapping.")
    return params


def _parse_step_mapping(step: Mapping[str, Any], idx: int) -> FeatureTransformStep:
    transform_name = step.get("transform") or step.get("name")
    if not transform_name:
        raise KeyError(f"Transform spec at index {idx} missing 'transform'.")
    features_raw = step.get("features")
    if features_raw is None:
        features_raw = step.get("feature")
    if features_raw is None:
        raise KeyError(f"Transform spec at index {idx} missing 'features'.")
    features = _normalize_features(features_raw)
    base_keys = {"transform", "name", "features", "feature", "params", "parameters"}
    params = _extract_params(step, base_keys)
    fn = _resolve_registered_transform(str(transform_name).lower(), params)
    return FeatureTransformStep(str(transform_name).lower(), features, fn)


def _parse_step(raw: Any, idx: int) -> FeatureTransformStep:
    if isinstance(raw, Mapping):
        return _parse_step_mapping(raw, idx)
    raise TypeError("Each transform spec must be a mapping.")


def build_transform_spec(transforms: Any) -> Optional[FeatureTransformSpec]:
    """Normalize raw configuration into a FeatureTransformSpec."""

    if not transforms:
        return None
    if isinstance(transforms, FeatureTransformSpec):
        return transforms
    if isinstance(transforms, Sequence) and not isinstance(
        transforms, (str, bytes, bytearray)
    ):
        steps = tuple(_parse_step(step, idx) for idx, step in enumerate(transforms))
        return FeatureTransformSpec(steps)
    raise TypeError("Feature transforms must be provided as a sequence of mappings.")


def feature_spec_from_config(
    feature_cfg: "FeatureConfig",
) -> Optional[FeatureTransformSpec]:
    transforms = getattr(feature_cfg, "transforms", None)
    return build_transform_spec(transforms)


__all__ = [
    "FeatureFn",
    "FeatureTransformSpec",
    "FeatureTransformStep",
    "build_transform_spec",
    "feature_spec_from_config",
]
