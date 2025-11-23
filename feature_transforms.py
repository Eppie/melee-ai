"""Shared feature transform helpers for dataset preprocessing and inference."""

# TODO: This might be a bit over-engineered
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
    """Scale a feature column in-place with a concrete walkthrough.

    Example
    -------
    ``column`` is ``array([1.0, -2.0, 0.5], dtype=float32)`` and ``factor=0.1``.
    We multiply elementwise, producing ``[0.1, -0.2, 0.05]`` which replaces the
    original array contents. The updated array is returned so chained transforms
    can continue using the same buffer.
    """
    column *= factor
    return column


def _transform_offset(column: np.ndarray, *, delta: float) -> np.ndarray:
    """Shift a feature column in-place and illustrate the effect.

    Example
    -------
    With ``column = array([-1.0, 0.0, 2.0])`` and ``delta=5`` we perform
    ``column += 5`` so the buffer becomes ``[4.0, 5.0, 7.0]``. The returned view
    allows downstream code to observe the shifted values without additional
    copies.
    """
    column += delta
    return column


TransformFactory = Callable[[Mapping[str, Any]], FeatureFn]


# TODO: might not need mask or clip
# TODO: This is duplicated elsewhere
def _sticks01_to_unit11_np(xy01: np.ndarray) -> np.ndarray:
    """Rescale ``[0, 1]`` stick coordinates to ``[-1, 1]`` with unit-circle clamping.

    Example
    -------
    Given ``xy01 = array([[0.0, 1.0], [0.75, 0.75]])``:

    1. Clip values to ``[0, 1]`` (already satisfied).
    2. Multiply by ``2`` and subtract ``1`` giving ``[[-1.0, 1.0], [0.5, 0.5]]``.
    3. Compute norms ``[√2, √0.5]`` and divide the overflowing ``[-1, 1]`` vector
       by ``√2`` so it lands on ``[-0.7071, 0.7071]`` while the second row stays
       untouched.

    The returned array is therefore suitable for palette quantization in the
    exact format produced by training.
    """
    xy01_clipped = np.clip(xy01, 0.0, 1.0)
    xy11 = xy01_clipped * 2.0 - 1.0
    norms = np.linalg.norm(xy11, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        xy11[mask] /= norms[mask, np.newaxis]
    return xy11


# TODO: Are these duplicated elsewhere? better place to put these?
_MAIN_PALETTE = np.asarray(CONTROL_STICK_QUANTIZED, dtype=np.float32)
_C_PALETTE = np.asarray(C_STICK_QUANTIZED, dtype=np.float32)

_PALETTES: Dict[str, np.ndarray] = {
    "fox_main": _MAIN_PALETTE,
    "main": _MAIN_PALETTE,
    "c_stick": _C_PALETTE,
    "c": _C_PALETTE,
}


# TODO: do we need both paths? should KNOW if [-1,1] or [0,1]
def _stick_palette_apply(
    block: np.ndarray,
    *,
    palette: np.ndarray,
    palette_norm: np.ndarray,
) -> np.ndarray:
    """Quantize ``(x, y)`` stick pairs to the nearest entry in ``palette``.

    Example
    -------
    Suppose ``block`` contains two rows ``[[0.1, 0.9], [1.2, -0.4]]`` and the
    palette has the four cardinal directions ``[[1, 0], [0, 1], [-1, 0], [0, -1]]``.

    1. Because values exceed ``1`` we clamp to ``[-1, 1]`` and renormalize, giving
       ``[[0.1, 0.9], [0.9487, -0.3162]]``.
    2. Compute dot products against the palette and the precomputed norms to build
       squared distances. The first vector is closest to ``[0, 1]`` (index ``1``)
       while the second lands near ``[1, 0]`` (index ``0``).
    3. Replace each row with the winning palette vector so ``block`` becomes
       ``[[0, 1], [1, 0]]`` before being returned.

    This in-place workflow mirrors how sequences are quantized during dataset
    creation and training.
    """
    if block.shape[1] != 2:
        raise ValueError(
            "stick_palette transform expects exactly two feature columns (x, y)."
        )
    values = block.astype(np.float32, copy=False)
    if np.any(values < 0.0) or np.any(values > 1.0):
        xy11 = np.clip(values, -1.0, 1.0).copy()
        norms = np.linalg.norm(xy11, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            xy11[mask] /= norms[mask, np.newaxis]
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
    """Build a stick palette transform and show how configuration is resolved.

    Example
    -------
    When ``params={'palette': 'main'}`` the factory selects the Fox main-stick
    palette, precomputes its squared norms, and returns a function equivalent to
    ``lambda block: _stick_palette_apply(block, palette=_MAIN_PALETTE, ...)``.
    Applying this function to a ``(N, 2)`` array therefore snaps each row to the
    nearest discrete stick direction consistent with quantization.
    """
    palette_key = str(params.get("palette", "fox_main")).lower()
    palette = _PALETTES.get(palette_key)
    if palette is None:
        raise ValueError(f"Unknown stick palette '{palette_key}'.")

    palette = palette.astype(np.float32, copy=False)
    palette_norm = np.sum(palette**2, axis=1, keepdims=True)

    return partial(_stick_palette_apply, palette=palette, palette_norm=palette_norm)


def _factory_scale(params: Mapping[str, Any]) -> FeatureFn:
    """Create a multiplicative transform from configuration parameters.

    Example
    -------
    Given ``params={'factor': 0.5}`` the factory returns a function that, when
    passed ``array([2.0, -4.0])``, multiplies the data by ``0.5`` to produce
    ``[1.0, -2.0]``. This mirrors how YAML/JSON transform specs reduce analog
    ranges before batching.
    """
    raw = params.get("factor", params.get("scale", 1.0))
    try:
        factor = float(raw)
    except (TypeError, ValueError) as exc:
        raise TypeError("Scale transform requires a numeric 'factor'.") from exc
    return partial(_transform_scale, factor=factor)


def _factory_offset(params: Mapping[str, Any]) -> FeatureFn:
    """Create an additive transform and illustrate its effect.

    Example
    -------
    ``params={'delta': -3}`` produces a callable that subtracts ``3`` from any
    supplied column. Applying it to ``array([5., 6.])`` yields ``[2., 3.]`` which
    is returned to the caller, matching the shift semantics used during
    preprocessing.
    """
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
    """Lookup a registered transform factory and demonstrate a lookup failure.

    Example
    -------
    * ``_resolve_registered_transform('scale', {'factor': 2})`` returns the scale
      function described above.
    * Passing ``'unknown'`` raises ``ValueError("Unknown feature transform 'unknown'.")``,
      so configuration errors are surfaced immediately instead of at runtime.
    """
    factory = _TRANSFORM_FACTORIES[name]
    return factory(params)


def _normalize_features(raw: Any) -> Tuple[str, ...]:
    """Normalize transform feature declarations with a concrete case analysis.

    Example
    -------
    * ``raw='p1_main_stick_x'`` becomes ``('p1_main_stick_x',)``.
    * ``raw=['p1_main_stick_x', 'p1_main_stick_y']`` preserves order while
      stripping whitespace.
    * An empty list triggers ``ValueError`` showing how the helper guards against
      silent misconfiguration.
    """
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
    """Derive transform parameters from multiple configuration styles.

    Example
    -------
    For a YAML entry ``{'transform': 'scale', 'features': 'foo', 'factor': 0.1}``
    the helper returns ``{'factor': 0.1}`` because ``'factor'`` is not in
    ``base_keys``. If the mapping already has ``'params': {'factor': 0.1}`` it is
    returned verbatim, demonstrating how both explicit and implicit parameter
    encodings converge to the same dictionary.
    """
    params = step.get("params")
    if params is None:
        params = step.get("parameters")
    if params is None:
        params = {k: v for k, v in step.items() if k not in base_keys}
    if not isinstance(params, Mapping):
        raise TypeError("Transform 'params' must be a mapping.")
    return params


def _parse_step_mapping(step: Mapping[str, Any], idx: int) -> FeatureTransformStep:
    """Parse a single mapping into :class:`FeatureTransformStep` with validation.

    Example
    -------
    Given ``step={'transform': 'scale', 'features': ['foo'], 'factor': 2}`` the
    helper:

    1. Extracts ``'scale'`` as the transform name and normalizes ``['foo']`` to a
       tuple.
    2. Calls :func:`_extract_params` to build ``{'factor': 2}``.
    3. Resolves the factory and returns ``FeatureTransformStep('scale', ('foo',), fn)``.

    If the mapping omitted ``'features'`` a ``KeyError`` is raised with the index
    ``idx`` embedded so configuration errors are easy to locate.
    """
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
    """Dispatch to :func:`_parse_step_mapping` while guarding the input type.

    Example
    -------
    * Passing a mapping proxies to :func:`_parse_step_mapping`.
    * Passing a list triggers ``TypeError("Each transform spec must be a mapping.")``
      so malformed configuration nodes are rejected early in the load process.
    """
    if isinstance(raw, Mapping):
        return _parse_step_mapping(raw, idx)
    raise TypeError("Each transform spec must be a mapping.")


def build_transform_spec(transforms: Any) -> Optional[FeatureTransformSpec]:
    """Normalize raw configuration into a :class:`FeatureTransformSpec`.

    Example
    -------
    If ``transforms`` is::

        [
            {'transform': 'scale', 'features': 'foo', 'factor': 0.5},
            {'transform': 'offset', 'features': ['foo', 'bar'], 'delta': 1},
        ]

    the function returns a spec with two ordered steps whose callables first
    multiply column ``foo`` by ``0.5`` then add ``1`` to both ``foo`` and
    ``bar``. Passing ``None`` yields ``None``, mirroring how ``FeatureConfig``
    omits transformations by default.
    """

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
    """Extract the transform spec from a :class:`FeatureConfig` instance.

    Example
    -------
    When ``feature_cfg.transforms`` already contains a parsed
    :class:`FeatureTransformSpec`, the function returns it unchanged. If the field
    stores raw configuration such as ``[{'transform': 'scale', ...}]`` the helper
    passes that list to :func:`build_transform_spec` so callers always receive a
    normalized object or ``None``.
    """
    transforms = getattr(feature_cfg, "transforms", None)
    return build_transform_spec(transforms)


__all__ = [
    "FeatureFn",
    "FeatureTransformSpec",
    "FeatureTransformStep",
    "build_transform_spec",
    "feature_spec_from_config",
]
