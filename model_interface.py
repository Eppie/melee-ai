from __future__ import annotations

import dataclasses
import json
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import torch
from tensordict import TensorDict

import constants
from column_map import ColumnMap
from config import FeatureConfig, get_config
from constants import CONTROLLER_KEY_GROUPS, _FEATURE_TRANSFORMS_SPEC
from controller_utils import (
    CONTROL_STICK_QUANTIZED,
    C_STICK_QUANTIZED,
    SHOULDER_QUANTIZED,
)
from feature_transforms import build_transform_spec
from libmelee.melee import enums
from libmelee.melee.controller import Controller
from libmelee.melee.gamestate import GameState
from model.nano_gpt import GPT
from schema import (
    PLAYER_SPEC,
    extract_common_fields,
    extract_player_fields,
    get_feature_names, get_target_names,
)
from train import build_model_inputs
from utils import _resolve_device

_DEFAULT_FEATURE_NAMES = get_feature_names()
_DEFAULT_TARGET_NAMES = get_target_names()

_PRINT_CACHE: dict[str, int] = {}

def print_cached(s: str) -> None:
    if s not in _PRINT_CACHE:
        _PRINT_CACHE[s] = 1
    else:
        _PRINT_CACHE[s] += 1
    if _PRINT_CACHE[s] % 100 == 0:
        print(s, _PRINT_CACHE[s])

@dataclasses.dataclass
class ControllerState:
    """Convenience container for controller outputs."""

    main_stick_x: float
    main_stick_y: float
    c_stick_x: float
    c_stick_y: float
    shoulder_analog: float
    button_a: bool
    button_b: bool
    button_xy: bool
    button_lr: bool
    button_z: bool

    @staticmethod
    def neutral() -> "ControllerState":
        """Return a neutral controller state with a concrete illustration.

        Example
        -------
        Calling ``ControllerState.neutral()`` yields a state where the analog
        sticks point to ``0.5`` (the Dolphin neutral value), the shoulder analog
        is ``0.0``, and all digital buttons are ``False``. Passing this object to
        :func:`apply_model_outputs_to_game` leaves the character idle, which is
        exactly what the inference engine emits during warm-up.
        """
        print_cached("[TRACE:ControllerState.neutral] Creating neutral controller state")
        return ControllerState(
            main_stick_x=0.5,
            main_stick_y=0.5,
            c_stick_x=0.5,
            c_stick_y=0.5,
            shoulder_analog=0.0,
            button_a=False,
            button_b=False,
            button_xy=False,
            button_lr=False,
            button_z=False,
        )


@dataclasses.dataclass
class FrameRecord:
    """Sliding window entry for inference logging."""

    raw_features: Dict[str, float]
    transformed_features: Dict[str, float]
    targets: Dict[str, float]
    logits: Optional[Dict[str, Any]] = None


@dataclasses.dataclass
class ModelFrameInputs(Mapping[str, float]):
    """Container bundling raw + transformed feature dicts."""

    raw: Dict[str, float]
    transformed: Dict[str, float]

    def __getitem__(self, key: str) -> float:
        """Return the transformed value for ``key`` mirroring dict access.

        Example
        -------
        With ``inputs = ModelFrameInputs({'foo': 1.0}, {'foo': 3.14})`` the call
        ``inputs['foo']`` returns ``3.14``. Attempting to access a missing key
        raises ``KeyError`` the same way as a standard dictionary.
        """
        return self.transformed[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over transformed feature names just like a dictionary.

        Example
        -------
        ``list(inputs)`` yields ``['foo']`` for the example above, allowing the
        object to integrate with APIs that expect a mapping view of features.
        """
        return iter(self.transformed)

    def __len__(self) -> int:
        """Expose the number of transformed features in the mapping.

        Example
        -------
        ``len(inputs)`` evaluates to ``1`` for the example, which matches the
        count of keys returned by ``__iter__``.
        """
        return len(self.transformed)

    def as_dict(self) -> Dict[str, float]:
        """Copy the transformed mapping into a standalone ``dict``.

        Example
        -------
        ``inputs.as_dict()`` returns ``{'foo': 3.14}``. Mutating this dictionary
        does not alter the underlying ``ModelFrameInputs`` object, making it safe
        to hand to logging or serialization routines.
        """
        return dict(self.transformed)


# TODO: We probably have other "safe" functions, maybe we can put them all in one place
# TODO: Why do we even need this, can't we just always know our types?
def _safe_float(value: object) -> float:
    """Convert ``value`` to ``float`` while logging failures with an example.

    Example
    -------
    ``_safe_float(np.int32(7))`` returns ``7.0``. If ``value='abc'`` the helper
    prints a trace line and re-raises the ``ValueError`` so callers see which
    feature failed to coerce during inference.
    """
    try:
        return float(value)
    except Exception as e:
        print_cached(
            f"[TRACE:_safe_float] Failed to convert {value} of type {type(value)} to float: {e}"
        )
        raise


# TODO: called 63 times per frame, there has to be a better way
def _coerce_scalar(value: object) -> float | int | bool:
    """Convert arbitrary scalars to Python ``bool``/``int``/``float``.

    Example
    -------
    ``_coerce_scalar(np.bool_(True))`` returns ``True`` while
    ``_coerce_scalar(np.float32(1.5))`` returns ``1.5``. Passing an object such as
    ``{'not': 'a scalar'}`` raises ``TypeError("Unable to coerce value ...")`` so
    malformed schema values surface immediately.
    """
    # print_cached(f"[TRACE:_coerce_scalar] Coercing value of type {type(value)}")
    if isinstance(value, (bool, int, float)):
        # print_cached("[TRACE:_coerce_scalar] Value is already bool/int/float")
        return value
    if isinstance(value, np.generic):
        # print_cached("[TRACE:_coerce_scalar] Value is numpy generic, calling item()")
        return value.item()
    try:
        result = float(value)
        # print_cached("[TRACE:_coerce_scalar] Successfully converted to float")
        return result
    except Exception as exc:
        # print_cached(f"[TRACE:_coerce_scalar] Failed to coerce value {value!r}")
        raise TypeError(
            f"Unable to coerce value {value!r} ({type(value)}) to scalar"
        ) from exc


def set_feature_transforms(transforms: Optional[Any]) -> None:
    """Configure per-feature transforms used at inference time with a demo.

    Example
    -------
    Passing ``[{'transform': 'scale', 'features': 'foo', 'factor': 0.5}]`` builds a
    spec via :func:`build_transform_spec` and stores it in the module-level cache.
    Subsequent calls to :func:`_apply_transforms_to_features` will then halve the
    ``foo`` feature before inference. Passing ``None`` clears the spec, restoring
    identity transforms.
    """
    print_cached(
        f"[TRACE:set_feature_transforms] Called with transforms={transforms is not None}"
    )

    if not transforms:
        print_cached("[TRACE:set_feature_transforms] No transforms provided, setting to None")
        constants._FEATURE_TRANSFORMS_SPEC = None
        return

    print_cached("[TRACE:set_feature_transforms] Building transform spec")
    constants._FEATURE_TRANSFORMS_SPEC = build_transform_spec(transforms)


set_feature_transforms(FeatureConfig().transforms)


# TODO: overly generic, maybe can cache some?
def _apply_transforms_to_features(features: Dict[str, float]) -> Dict[str, float]:
    """Apply the configured transform spec to the ``features`` mapping.

    Example
    -------
    Suppose ``features={'foo': 4.0, 'bar': 1.0}`` and the global spec contains two
    steps: scale ``foo`` by ``0.5`` and add ``1`` to both ``foo`` and ``bar``. The
    helper copies the mapping, resolves ``foo`` groups, and produces
    ``{'foo': 3.0, 'bar': 2.0}``, illustrating the same pipeline that runs before
    every inference call.
    """
    # print_cached(f"[TRACE:_apply_transforms_to_features] Called with {len(features)} features")
    spec = _FEATURE_TRANSFORMS_SPEC
    if not spec or not spec.steps:
        # print_cached("[TRACE:_apply_transforms_to_features] No transforms to apply, returning original")
        return features

    # print_cached(f"[TRACE:_apply_transforms_to_features] Applying {len(spec.steps)} transform steps")
    out = dict(features)
    keys = list(out.keys())
    key_set = set(keys)

    prefixes: set[str] = set()
    for key in keys:
        head, _, tail = key.partition("_")
        if tail and head.startswith("p") and head[1:].isdigit():
            prefixes.add(head)
    # print_cached(f"[TRACE:_apply_transforms_to_features] Found {len(prefixes)} player prefixes")

    # TODO: Surely we can cache this, or maybe even remove the need for it?
    def _resolve_groups(requested: Sequence[str]) -> List[Tuple[str, ...]]:
        """Expand ``requested`` feature names to actual keys with prefix handling.

        Example
        -------
        With ``requested=('main_stick_x', 'main_stick_y')`` and prefixed keys
        available, this helper returns groups like
        ``('p1_main_stick_x', 'p1_main_stick_y')`` and
        ``('p2_main_stick_x', 'p2_main_stick_y')`` so transforms execute for both
        players independently.
        """
        # print_cached(f"[TRACE:_resolve_groups] Resolving {len(requested)} requested features")
        if all(name in key_set for name in requested):
            # print_cached("[TRACE:_resolve_groups] All features found in key_set, returning single group")
            return [tuple(requested)]
        # print_cached("[TRACE:_resolve_groups] Looking for prefixed groups")
        groups: List[Tuple[str, ...]] = []
        for prefix in sorted(prefixes):
            group: List[str] = []
            for feature in requested:
                key = f"{prefix}_{feature}"
                if key not in key_set:
                    # print_cached(f"[TRACE:_resolve_groups] Key {key} not found, breaking")
                    break
                group.append(key)
            else:
                if group:
                    # print_cached(f"[TRACE:_resolve_groups] Found complete group with prefix {prefix}")
                    groups.append(tuple(group))
        # print_cached(f"[TRACE:_resolve_groups] Returning {len(groups)} groups")
        return groups

    for step_idx, step in enumerate(spec.steps):
        print_cached(
            f"[TRACE:_apply_transforms_to_features] Processing step {step_idx}: {step.transform}"
        )
        groups = _resolve_groups(step.features)
        if not groups:
            print_cached(
                f"[TRACE:_apply_transforms_to_features] No groups found for step {step_idx}, continuing"
            )
            continue
        for group_idx, group in enumerate(groups):
            if len(group) == 1:
                print_cached(
                    f"[TRACE:_apply_transforms_to_features] Step {step_idx} group {group_idx}: single feature"
                )
                key = group[0]
                value = np.array(out[key], dtype=np.float32)
                result = step.fn(value.copy())
                if result is None:
                    print_cached(
                        f"[TRACE:_apply_transforms_to_features] Transform returned None, using original"
                    )
                    result = value
                if result.shape != value.shape:
                    print_cached(
                        f"[TRACE:_apply_transforms_to_features] ERROR: Shape mismatch {value.shape} vs {result.shape}"
                    )
                    raise ValueError(
                        f"Transform '{step.transform}' expected output shape {value.shape}, got {result.shape}."
                    )
                out[key] = float(result)
            else:
                print_cached(
                    f"[TRACE:_apply_transforms_to_features] Step {step_idx} group {group_idx}: {len(group)} features"
                )
                block = np.array([[float(out[k]) for k in group]], dtype=np.float32)
                result = step.fn(block.copy())
                if result is None:
                    print_cached(
                        f"[TRACE:_apply_transforms_to_features] Transform returned None, using original"
                    )
                    result = block
                if result.shape != block.shape:
                    print_cached(
                        f"[TRACE:_apply_transforms_to_features] ERROR: Shape mismatch {block.shape} vs {result.shape}"
                    )
                    raise ValueError(
                        f"Transform '{step.transform}' expected output shape {block.shape}, got {result.shape}."
                    )
                for idx, key in enumerate(group):
                    out[key] = float(result[0, idx])

    print_cached("[TRACE:_apply_transforms_to_features] All transforms applied successfully")
    return out


def _controller_state_to_features(
    prefix: str, state: ControllerState
) -> Dict[str, float]:
    """Convert a :class:`ControllerState` into prefixed feature values.

    Example
    -------
    ``_controller_state_to_features('p1', ControllerState.neutral())`` produces a
    dictionary where ``'p1_main_stick_x'`` equals ``0.5`` and all button keys are
    ``0.0``. The inference engine caches this mapping to seed controller history
    before live predictions.
    """
    print_cached(
        f"[TRACE:_controller_state_to_features] Converting controller state with prefix={prefix}"
    )
    return {
        f"{prefix}_button_a": float(state.button_a),
        f"{prefix}_button_b": float(state.button_b),
        f"{prefix}_button_xy": float(state.button_xy),
        f"{prefix}_button_lr": float(state.button_lr),
        f"{prefix}_button_z": float(state.button_z),
        f"{prefix}_main_stick_x": float(state.main_stick_x),
        f"{prefix}_main_stick_y": float(state.main_stick_y),
        f"{prefix}_c_stick_x": float(state.c_stick_x),
        f"{prefix}_c_stick_y": float(state.c_stick_y),
        f"{prefix}_shoulder_analog": float(state.shoulder_analog),
    }


def _zero_player_fields(prefix: str) -> Dict[str, float]:
    """Return zero-valued player features for ``prefix``.

    Example
    -------
    ``_zero_player_fields('p2')`` yields entries like ``{'p2_stock': 0}`` and
    ``{'p2_main_stick_x': 0.0}``, matching the neutral placeholders used when an
    opponent is absent from the gamestate.
    """
    print_cached(f"[TRACE:_zero_player_fields] Creating zero fields for prefix={prefix}")
    return {f"{prefix}_{name}": dtype(0) for name, dtype in PLAYER_SPEC}


def _prefixed_player_fields(player, prefix: str) -> Dict[str, float]:
    """Extract prefixed player fields or fall back to zeros.

    Example
    -------
    If ``player`` has valid stats, the helper returns values such as
    ``{'p1_percent': 23.0}``. When ``player`` is ``None`` the output equals
    :func:`_zero_player_fields(prefix)`, ensuring callers receive a complete feature
    dictionary regardless of the gamestate.
    """
    print_cached(f"[TRACE:_prefixed_player_fields] Extracting fields for prefix={prefix}")
    if player is None or getattr(player, "controller_state", None) is None:
        print_cached(
            "[TRACE:_prefixed_player_fields] Player is None or has no controller_state, returning zeros"
        )
        return _zero_player_fields(prefix)

    try:
        extracted = extract_player_fields(player)
        print_cached(
            f"[TRACE:_prefixed_player_fields] Successfully extracted {len(extracted)} fields"
        )
    except ValueError:
        print_cached(
            "[TRACE:_prefixed_player_fields] ValueError during extraction, returning zeros"
        )
        return _zero_player_fields(prefix)

    return {f"{prefix}_{name}": value for name, value in extracted.items()}


def model_to_dolphin01(
    model_out: np.ndarray,
    palette11: np.ndarray | None = None,
) -> np.ndarray:
    """Convert model outputs to Dolphin's ``[0, 1]`` coordinate space.

    Example
    -------
    * ``model_out = [[0.0, 1.0]]`` (float inputs) is clamped to the unit circle and
      mapped to ``[[0.5, 1.0]]`` after scaling to ``[0, 1]``.
    * ``model_out = [[2]]`` with ``palette11`` equal to the Fox stick palette
      selects the third palette vector, clamps it if needed, and outputs the
      corresponding ``[0, 1]`` coordinates.
    """
    print_cached(
        f"[TRACE:model_to_dolphin01] Converting model output, palette provided={palette11 is not None}"
    )
    arr = np.asarray(model_out)

    if np.issubdtype(arr.dtype, np.integer):
        print_cached("[TRACE:model_to_dolphin01] Array is integer type, using palette")
        if palette11 is None:
            print_cached(
                "[TRACE:model_to_dolphin01] ERROR: palette11 is None for integer array"
            )
            raise ValueError("palette11 must be provided when converting indices")
        P = np.asarray(palette11, dtype=np.float32)
        coords11 = P[arr]
    else:
        print_cached("[TRACE:model_to_dolphin01] Array is float type, using directly")
        coords11 = np.asarray(model_out, dtype=np.float32)
        if coords11.shape[-1] != 2:
            print_cached(
                f"[TRACE:model_to_dolphin01] ERROR: Last dimension is {coords11.shape[-1]}, expected 2"
            )
            raise ValueError("model_out must have last dimension size 2")

    # Clamp to unit circle to be safe.
    r2 = np.einsum("...i,...i->...", coords11, coords11)
    over = r2 > 1.0
    if np.any(over):
        print_cached(
            f"[TRACE:model_to_dolphin01] Clamping {np.sum(over)} values outside unit circle"
        )
        coords11 = coords11.copy()
        coords11[over] /= np.sqrt(r2[over])[..., None]
    else:
        print_cached("[TRACE:model_to_dolphin01] All values within unit circle")

    # Map [-1,1] -> [0,1]
    xy01 = np.clip(coords11 * 0.5 + 0.5, 0.0, 1.0).astype(np.float32)
    return xy01


# TODO: maybe overengineered, maybe can collapse some. just be careful to keep in sync with what was done in training.
def collect_raw_inputs_from_gamestate(
    gamestate: GameState,
    bot_port: int,
    opp_port: int,
) -> ModelFrameInputs:
    """Extract both raw and transformed feature dictionaries with an example.

    Example
    -------
    Given ``bot_port=1`` and ``opp_port=2`` the helper:

    1. Extracts stage-level stats via :func:`extract_common_fields`.
    2. Gathers player-specific values for both ports and prefixes them with
       ``p1_``/``p2_``.
    3. Coerces scalars, applies configured transforms, and validates that every
       expected feature is present.

    The returned :class:`ModelFrameInputs` therefore contains the raw mapping used
    for logging and the transformed mapping used for inference.
    """
    print_cached(
        f"[TRACE:collect_raw_inputs_from_gamestate] bot_port={bot_port}, opp_port={opp_port}"
    )

    ego_player = gamestate.players.get(bot_port)
    opp_player = gamestate.players.get(opp_port)
    print_cached(
        f"[TRACE:collect_raw_inputs_from_gamestate] ego_player found={ego_player is not None}, opp_player found={opp_player is not None}"
    )

    values: Dict[str, float] = extract_common_fields(gamestate)
    prefixed_common = {name: val for name, val in values.items()}

    player_values: Dict[str, float] = {}
    player_values.update(_prefixed_player_fields(ego_player, "p1"))
    player_values.update(_prefixed_player_fields(opp_player, "p2"))

    combined = {**prefixed_common, **player_values}
    print_cached(
        f"[TRACE:collect_raw_inputs_from_gamestate] Combined {len(combined)} raw fields"
    )
    combined_raw = {name: _coerce_scalar(val) for name, val in combined.items()}
    transformed = _apply_transforms_to_features(dict(combined_raw))

    feature_names = get_feature_names()

    final: Dict[str, float] = {}
    for name in feature_names:
        if name not in transformed:
            print_cached(
                f"[TRACE:collect_raw_inputs_from_gamestate] ERROR: Feature {name} missing"
            )
            raise KeyError(f"Feature '{name}' missing from collected inputs.")
        final[name] = _safe_float(transformed[name])

    print_cached(
        f"[TRACE:collect_raw_inputs_from_gamestate] Returning {len(final)} final features"
    )
    return ModelFrameInputs(raw=combined_raw, transformed=final)


class GPTInferenceEngine:
    """Online inference helper around the GPT controller model."""

    def __init__(
        self,
        checkpoint_path: str | Path,
    ) -> None:
        """Load the GPT checkpoint and initialize inference buffers.

        Example
        -------
        Constructing ``GPTInferenceEngine('checkpoint.pt')`` performs:

        1. ``torch.load`` to retrieve the saved model state and training config.
        2. Metadata loading from ``meta.json`` (if present) to recover
           ``feature_names``, ``target_names``, and ``seq_len``.
        3. Model instantiation on the resolved device and buffer setup so
           :meth:`predict_from_raw` can immediately begin warm-up with neutral
           controller states.
        """
        print_cached(
            f"[TRACE:GPTInferenceEngine.__init__] Loading checkpoint from {checkpoint_path}"
        )
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        train_cfg = ckpt.get("config", {})

        # TODO: train_cfg might actually be full config
        # TODO: Don't load this from meta.json, load it from the checkpoint itself.
        data_root = Path(train_cfg.get("data_root", "dataset_FOX_vs_FOX"))
        meta_path = data_root / "meta.json"
        transforms_spec: Optional[Any] = None
        if meta_path.exists():
            print_cached(f"[TRACE:GPTInferenceEngine.__init__] Loading meta from {meta_path}")
            with meta_path.open("r") as f:
                meta = json.load(f)

            feature_names = meta["schema"]["features"]
            target_names = meta["schema"]["targets"]
            self.seq_len = int(meta.get("seq_len", 256))
            build_cfg = meta.get("build_config")
            if isinstance(build_cfg, Mapping):
                print_cached("[TRACE:GPTInferenceEngine.__init__] build_config found in meta")
                features_cfg = build_cfg.get("features")
                if isinstance(features_cfg, Mapping):
                    print_cached("[TRACE:GPTInferenceEngine.__init__] features config found")
                    transforms_cfg = features_cfg.get("transforms")
                    if transforms_cfg:
                        print_cached(
                            "[TRACE:GPTInferenceEngine.__init__] transforms config found in meta"
                        )
                        transforms_spec = transforms_cfg
        else:
            print_cached(
                "[TRACE:GPTInferenceEngine.__init__] No meta.json found, using defaults"
            )
            feature_names = list(_DEFAULT_FEATURE_NAMES)
            target_names = list(_DEFAULT_TARGET_NAMES)
            self.seq_len = 256

        self.device = _resolve_device()
        print_cached(f"[TRACE:GPTInferenceEngine.__init__] Device: {self.device}")
        self.shoulder_centers = SHOULDER_QUANTIZED
        self.warmup_frames = 256
        self._main_stick_palette = np.asarray(CONTROL_STICK_QUANTIZED, dtype=np.float32)
        # TODO: Isn't this already an ndarray?
        self._c_stick_palette = np.asarray(C_STICK_QUANTIZED, dtype=np.float32)

        if transforms_spec is None and isinstance(train_cfg, Mapping):
            print_cached(
                "[TRACE:GPTInferenceEngine.__init__] Checking train_cfg for transforms"
            )
            features_cfg = train_cfg.get("features")
            if isinstance(features_cfg, Mapping):
                print_cached(
                    "[TRACE:GPTInferenceEngine.__init__] features config found in train_cfg"
                )
                transforms_cfg = features_cfg.get("transforms")
                if transforms_cfg:
                    print_cached(
                        "[TRACE:GPTInferenceEngine.__init__] transforms config found in train_cfg"
                    )
                    transforms_spec = transforms_cfg

        if transforms_spec is not None:
            print_cached("[TRACE:GPTInferenceEngine.__init__] Setting feature transforms")
            set_feature_transforms(transforms_spec)
        else:
            print_cached("[TRACE:GPTInferenceEngine.__init__] No transforms spec found")

        print_cached("[TRACE:GPTInferenceEngine.__init__] Creating model")
        self.model = GPT(get_config()).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        self.feature_names = list(feature_names)
        self.target_names = list(target_names)
        self.colmap = ColumnMap(feature_names, target_names)
        self.feature_dim = len(feature_names)
        self.buffer: deque[torch.Tensor] = deque(maxlen=self.seq_len)
        self.frame_history: deque[FrameRecord] = deque(maxlen=self.seq_len)

        controller_feature_keys: list[str] = []
        for group in CONTROLLER_KEY_GROUPS.values():
            for name in group:
                key = f"p1_{name}"
                if key in self.feature_names and key not in controller_feature_keys:
                    controller_feature_keys.append(key)
        shoulder_key = "p1_shoulder_analog"
        # TODO: We will always have shoulder, what is this check actually for?
        if (
            shoulder_key in self.feature_names
            and shoulder_key not in controller_feature_keys
        ):
            print_cached(
                "[TRACE:GPTInferenceEngine.__init__] Adding shoulder_analog to controller keys"
            )
            controller_feature_keys.append(shoulder_key)
        self._controller_feature_keys: Tuple[str, ...] = tuple(controller_feature_keys)
        print_cached(
            f"[TRACE:GPTInferenceEngine.__init__] Tracking {len(self._controller_feature_keys)} controller keys"
        )
        self._prev_controller_features: Dict[str, float] = {}
        self._update_prev_controller_features(ControllerState.neutral())
        self._frames_seen = 0
        self._death_log_dir = Path.cwd() / "death_logs"
        self._death_counter = 0
        self._prev_stock: Optional[int] = None
        print_cached("[TRACE:GPTInferenceEngine.__init__] Initialization complete")

    # TODO: python loop - maybe vectorize?
    # TODO: Good candidate for a microbenchmark
    def _frame_to_tensor(self, raw_inputs: Dict[str, float]) -> torch.Tensor:
        """Convert ``raw_inputs`` into an ordered tensor of feature values.

        Example
        -------
        With ``self.feature_names = ['foo', 'bar']`` and ``raw_inputs`` containing
        those keys, the returned tensor is ``tensor([foo_value, bar_value])``.
        Missing keys raise ``KeyError`` through :func:`_safe_float`, making schema
        issues visible during inference.
        """
        print_cached(
            f"[TRACE:GPTInferenceEngine._frame_to_tensor] Converting {len(raw_inputs)} inputs"
        )
        frame = torch.zeros(self.feature_dim, dtype=torch.float32)
        for idx, name in enumerate(self.feature_names):
            frame[idx] = _safe_float(raw_inputs.get(name))
        return frame

    def _stack_frames(self) -> Optional[torch.Tensor]:
        """Stack buffered frames into a batch tensor when the buffer is non-empty.

        Example
        -------
        If ``self.buffer`` holds three ``(F,)`` tensors, the method returns a
        ``(1, 3, F)`` tensor ready for the model. When the buffer is empty (e.g.,
        before warm-up completes) the method returns ``None`` to signal that
        inference should keep emitting neutral actions.
        """
        print_cached(
            f"[TRACE:GPTInferenceEngine._stack_frames] Buffer size: {len(self.buffer)}"
        )
        if not self.buffer:
            print_cached(
                "[TRACE:GPTInferenceEngine._stack_frames] Buffer empty, returning None"
            )
            return None
        frames = list(self.buffer)
        stacked = torch.stack(frames, dim=0)
        print_cached(
            f"[TRACE:GPTInferenceEngine._stack_frames] Stacked shape: {stacked.shape}"
        )
        return stacked.unsqueeze(0).to(self.device)

    def _build_inputs(self, batch_X: torch.Tensor) -> TensorDict:
        """Wrap ``batch_X`` in the structured :class:`TensorDict` used by the model.

        Example
        -------
        ``batch_X`` with shape ``(1, 256, F)`` is forwarded to
        :func:`build_model_inputs`, which returns a dictionary containing the
        original tensor plus positional encodings. This mirrors the exact input
        contract used during training.
        """
        print_cached(
            f"[TRACE:GPTInferenceEngine._build_inputs] Building inputs from batch shape {batch_X.shape}"
        )
        return build_model_inputs(batch_X, self.colmap)

    def _override_controller_features(
        self, features: Mapping[str, float]
    ) -> Dict[str, float]:
        """Replace controller-related keys with cached values from prior outputs.

        Example
        -------
        If ``features['p1_main_stick_x']`` equals ``0.2`` but the cached previous
        value is ``0.5``, the returned dictionary substitutes ``0.5`` so the model
        sees consistent controller history until a fresh prediction is available.
        """
        print_cached(
            f"[TRACE:GPTInferenceEngine._override_controller_features] Overriding {len(self._controller_feature_keys)} controller features"
        )
        updated = dict(features)
        for key in self._controller_feature_keys:
            if key in self._prev_controller_features and key in updated:
                updated[key] = self._prev_controller_features[key]
        return updated

    def _update_prev_controller_features(self, state: ControllerState) -> None:
        """Cache the transformed controller state for future overrides.

        Example
        -------
        After decoding ``state`` with main stick ``(0.7, 0.4)``, the method expands
        it to feature keys, applies transforms if configured, and stores the result
        so :meth:`_override_controller_features` can reuse them on the next frame.
        """
        print_cached(
            "[TRACE:GPTInferenceEngine._update_prev_controller_features] Updating previous controller features"
        )
        values = _controller_state_to_features("p1", state)
        if _FEATURE_TRANSFORMS_SPEC and _FEATURE_TRANSFORMS_SPEC.steps:
            print_cached(
                "[TRACE:GPTInferenceEngine._update_prev_controller_features] Applying transforms"
            )
            transformed = _apply_transforms_to_features(dict(values))
            self._prev_controller_features = {k: float(transformed[k]) for k in values}
        else:
            print_cached(
                "[TRACE:GPTInferenceEngine._update_prev_controller_features] No transforms to apply"
            )
            self._prev_controller_features = {k: float(v) for k, v in values.items()}

    def _snapshot_features(self, features: Mapping[str, float]) -> Dict[str, float]:
        """Capture transformed feature values in model order.

        Example
        -------
        For ``self.feature_names = ['foo', 'bar']`` and ``features`` containing
        those keys, the returned dictionary is
        ``{'foo': float(features['foo']), 'bar': float(features['bar'])}``.
        Missing keys raise ``KeyError`` so logging never silently omits features.
        """
        print_cached(
            f"[TRACE:GPTInferenceEngine._snapshot_features] Snapshotting {len(self.feature_names)} features"
        )
        snapshot: Dict[str, float] = {}
        for name in self.feature_names:
            if name not in features:
                print_cached(
                    f"[TRACE:GPTInferenceEngine._snapshot_features] ERROR: Feature {name} missing"
                )
                raise KeyError(f"Feature '{name}' missing from snapshot inputs.")
            snapshot[name] = float(_safe_float(features[name]))
        return snapshot

    def _snapshot_raw(self, raw_inputs: Mapping[str, float]) -> Dict[str, float]:
        """Normalize raw mapping values into plain Python scalars.

        Example
        -------
        ``_snapshot_raw({'foo': np.float32(1.23)})`` yields ``{'foo': 1.23}``,
        making the snapshot JSON-serializable for death logs.
        """
        print_cached(
            f"[TRACE:GPTInferenceEngine._snapshot_raw] Snapshotting {len(raw_inputs)} raw inputs"
        )
        return {name: _coerce_scalar(value) for name, value in raw_inputs.items()}

    def _snapshot_targets(self, raw_inputs: Mapping[str, float]) -> Dict[str, float]:
        """Extract controller targets (sticks/buttons) from ``raw_inputs``.

        Example
        -------
        If ``raw_inputs`` provides ``'p1_button_a': 1`` and ``'p1_main_stick_x': 0.6``
        they appear in the returned dictionary. Missing keys trigger ``KeyError``
        so controller logs remain aligned with the model schema.
        """
        print_cached(
            f"[TRACE:GPTInferenceEngine._snapshot_targets] Snapshotting {len(self.target_names)} targets"
        )
        targets: Dict[str, float] = {}
        for name in self.target_names:
            if name not in raw_inputs:
                print_cached(
                    f"[TRACE:GPTInferenceEngine._snapshot_targets] ERROR: Target {name} missing"
                )
                raise KeyError(
                    f"Target '{name}' missing from raw inputs during logging."
                )
            targets[name] = float(_safe_float(raw_inputs[name]))
        return targets

    def _record_frame(
        self, model_features: Mapping[str, float], raw_inputs: Mapping[str, float]
    ) -> FrameRecord:
        """Append the current frame's raw, transformed, and target data to history.

        Example
        -------
        The method constructs a :class:`FrameRecord` using
        :meth:`_snapshot_raw`, :meth:`_snapshot_features`, and
        :meth:`_snapshot_targets`, appends it to ``self.frame_history``, and returns
        the record so callers can store logits alongside it.
        """
        print_cached("[TRACE:GPTInferenceEngine._record_frame] Recording frame")
        record = FrameRecord(
            raw_features=self._snapshot_raw(raw_inputs),
            transformed_features=self._snapshot_features(model_features),
            targets=self._snapshot_targets(raw_inputs),
        )
        self.frame_history.append(record)
        print_cached(
            f"[TRACE:GPTInferenceEngine._record_frame] Frame history size: {len(self.frame_history)}"
        )
        return record

    def _capture_logits(self, outputs: TensorDict) -> Dict[str, Any]:
        """Convert model logits into CPU lists for logging.

        Example
        -------
        When ``outputs['buttons']`` is available, the method stores
        ``outputs['buttons'][0, -1].detach().cpu().tolist()`` under the ``'buttons'``
        key so the JSON death log records the exact logits that led to an action.
        """
        logits: Dict[str, Any] = {"main_stick": outputs["main_stick"][0, -1].detach().cpu().tolist(),
                                  "c_stick": outputs["c_stick"][0, -1].detach().cpu().tolist(),
                                  "buttons": outputs["buttons"][0, -1].detach().cpu().tolist(),
                                  "shoulder": outputs["shoulder"][0, -1].detach().cpu().tolist(),
                                  "value": outputs["value"][0, -1].detach().cpu().tolist()
                                  }
        return logits

    def _persist_death_record(self, stock_after: int) -> None:
        """Write the current frame history to disk after a stock loss.

        Example
        -------
        If the player goes from three to two stocks, the method dumps
        ``self.frame_history`` into ``death_logs/death_000001.json`` including raw
        features, transformed features, targets, and logits for every recorded
        frame.
        """
        frames = list(self.frame_history)
        if not frames:
            return
        payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "frames_recorded": len(frames),
            "seq_len": self.seq_len,
            "warmup_frames": self.warmup_frames,
            "frames_seen": self._frames_seen,
            "stock_after_death": stock_after,
            "feature_names": self.feature_names,
            "target_names": self.target_names,
            "frames": [
                {
                    "raw_features": record.raw_features,
                    "transformed_features": record.transformed_features,
                    "targets": record.targets,
                    "logits": record.logits,
                }
                for record in frames
            ],
        }
        try:
            self._death_log_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            return
        filename = f"death_{self._death_counter:04d}_frame{self._frames_seen}.json"
        output_path = self._death_log_dir / filename
        try:
            with output_path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh)
        except Exception:
            return
        self._death_counter += 1

    def _maybe_log_death(self, current_stock: Optional[float]) -> None:
        """Detect stock drops and trigger :meth:`_persist_death_record`.

        Example
        -------
        When ``self._prev_stock`` is ``3`` and ``current_stock`` equals ``2``, the
        method saves the death log, increments counters, and resets tracking so the
        next stock loss generates a fresh log.
        """
        if current_stock is None:
            return
        stock_value = int(current_stock)
        if self._prev_stock is not None and stock_value < self._prev_stock:
            self._persist_death_record(stock_value)
        self._prev_stock = stock_value

    def _decode_stick(
        self, logits: torch.Tensor, palette: np.ndarray, stick_name: str
    ) -> np.ndarray:
        """Convert stick logits into palette coordinates with an example.

        Example
        -------
        If ``stick_name='main_stick'`` and ``logits=[2.0, 0.0, -1.0]`` over a
        palette of three vectors, the method selects the ``argmax`` index ``0`` and
        looks up the corresponding palette coordinate. When ``stick_name`` equals
        ``'c_stick'`` it additionally computes ``softmax`` probabilities for
        logging, mirroring the exact decoding performed during inference.
        """
        # Decode sticks by selecting the most likely quantized bin (argmax).
        idx = torch.argmax(logits.detach(), dim=-1)
        idx_np = idx.detach().cpu().numpy().astype(np.int32)
        xy01 = model_to_dolphin01(idx_np, palette11=palette)
        return xy01.reshape(-1)

    def _decode_buttons(self, probs: torch.Tensor) -> List[bool]:
        """Sample button activations from probabilities using Bernoulli draws.

        Example
        -------
        ``_decode_buttons(torch.tensor([[0.7, 0.2, 0.9, 0.1, 0.5]]))`` first clamps
        probabilities to ``[eps, 1-eps]`` then samples a Bernoulli outcome for each
        column, returning booleans such as ``[True, False, True, False, True]``.
        Re-running with the same tensor and manual seed reproduces the same sample,
        matching the stochastic decoding strategy used during evaluation.
        """
        # Interpret button activations probabilistically, sampling directly from the model probabilities.
        eps = torch.finfo(probs.dtype).eps
        clamped_probs = torch.clamp(probs.detach(), eps, 1 - eps)
        samples = torch.bernoulli(clamped_probs).bool().cpu()
        result = samples.tolist()
        return result

    # TODO: overly safe. decide on button logits or probs.
    def _decode_outputs(self, outputs: TensorDict) -> ControllerState:
        """Assemble decoded sticks, buttons, and shoulder into controller output.

        Example
        -------
        The method decodes main and C-stick logits via :meth:`_decode_stick`,
        samples button booleans through :meth:`_decode_buttons`, and chooses the
        highest-probability shoulder bin. If the shoulder palette is
        ``[0.0, 0.5, 1.0]`` and logits favor index ``1``, the resulting
        :class:`ControllerState` has ``shoulder_analog=0.5`` while the sticks hold
        their decoded coordinates.
        """
        print_cached("[TRACE:GPTInferenceEngine._decode_outputs] Decoding model outputs")
        main_logits = outputs["main_stick"][0, -1]
        c_logits = outputs["c_stick"][0, -1]
        button_probs = outputs.get("buttons_probs")
        if button_probs is None:
            print_cached(
                "[TRACE:GPTInferenceEngine._decode_outputs] buttons_probs not found, using sigmoid of buttons"
            )
            button_probs = torch.sigmoid(outputs["buttons"])
        else:
            print_cached(
                "[TRACE:GPTInferenceEngine._decode_outputs] Using buttons_probs from outputs"
            )
        button_probs = button_probs[0, -1]

        shoulder_logits = outputs.get("shoulder")

        main_xy = self._decode_stick(
            main_logits, self._main_stick_palette, "main_stick"
        )
        c_xy = self._decode_stick(c_logits, self._c_stick_palette, "c_stick")
        buttons_bool = self._decode_buttons(button_probs)

        if shoulder_logits is not None:
            print_cached(
                "[TRACE:GPTInferenceEngine._decode_outputs] Decoding shoulder from logits"
            )
            s_idx = int(torch.argmax(shoulder_logits[0, -1]).item())
            s_idx = max(0, min(s_idx, len(self.shoulder_centers) - 1))
            shoulder_val = float(self.shoulder_centers[s_idx])
            print_cached(
                f"[TRACE:GPTInferenceEngine._decode_outputs] Shoulder index={s_idx}, value={shoulder_val}"
            )
        else:
            print_cached(
                "[TRACE:GPTInferenceEngine._decode_outputs] No shoulder logits, using 0.0"
            )
            shoulder_val = 0.0

        print_cached(
            f"[TRACE:GPTInferenceEngine._decode_outputs] main_stick=({main_xy[0]:.3f}, {main_xy[1]:.3f})"
        )
        print_cached(
            f"[TRACE:GPTInferenceEngine._decode_outputs] c_stick=({c_xy[0]:.3f}, {c_xy[1]:.3f})"
        )
        return ControllerState(
            main_stick_x=float(main_xy[0]),
            main_stick_y=float(main_xy[1]),
            c_stick_x=float(c_xy[0]),
            c_stick_y=float(c_xy[1]),
            shoulder_analog=shoulder_val,
            button_a=bool(buttons_bool[0]),
            button_b=bool(buttons_bool[1]),
            button_xy=bool(buttons_bool[2]),
            button_z=bool(buttons_bool[3]),
            button_lr=bool(buttons_bool[4]),
        )

    def prepare_inputs(self, raw_inputs: Mapping[str, float]) -> Optional[TensorDict]:
        """Update the frame buffer and return model inputs when enough frames exist.

        Example
        -------
        Feeding a new frame appends its tensor to the buffer. Before
        ``self.warmup_frames`` are collected the method returns ``None``. Once the
        buffer reaches that length, it stacks the frames and returns the
        :class:`TensorDict` from :meth:`_build_inputs`, mirroring the workflow used
        inside :meth:`predict_from_raw`.
        """
        print_cached("[TRACE:GPTInferenceEngine.prepare_inputs] Preparing inputs")
        frame = self._frame_to_tensor(raw_inputs)
        self.buffer.append(frame)
        print_cached(
            f"[TRACE:GPTInferenceEngine.prepare_inputs] Buffer now has {len(self.buffer)} frames"
        )
        batch = self._stack_frames()
        if batch is None:
            print_cached(
                "[TRACE:GPTInferenceEngine.prepare_inputs] Batch is None, returning None"
            )
            return None
        return self._build_inputs(batch)

    def predict_from_raw(
        self, frame_inputs: Mapping[str, float] | ModelFrameInputs
    ) -> ControllerState:
        """Run inference on raw or preprocessed frame inputs with a full walkthrough.

        Example
        -------
        During warm-up, repeated calls buffer frames and return
        :func:`ControllerState.neutral`. After ``warmup_frames`` are reached, the
        method constructs inputs (or uses ``frame_inputs.transformed``), executes
        the GPT model under ``torch.inference_mode()``, logs logits via
        :meth:`_capture_logits`, decodes the outputs to a controller state, and
        updates the override cache so the next call sees the latest controls.
        """
        print_cached(f"[TRACE:GPTInferenceEngine.predict_from_raw] Frame {self._frames_seen}")
        if isinstance(frame_inputs, ModelFrameInputs):
            print_cached(
                "[TRACE:GPTInferenceEngine.predict_from_raw] Input is ModelFrameInputs"
            )
            raw_snapshot = dict(frame_inputs.raw)
            transformed = dict(frame_inputs.transformed)
        else:
            print_cached(
                "[TRACE:GPTInferenceEngine.predict_from_raw] Input is mapping, coercing"
            )
            raw_snapshot = {k: _coerce_scalar(v) for k, v in frame_inputs.items()}
            transformed = {k: float(_safe_float(v)) for k, v in frame_inputs.items()}

        features = self._override_controller_features(transformed)
        record = self._record_frame(features, raw_snapshot)
        inputs_td = self.prepare_inputs(features)
        self._frames_seen += 1
        self._maybe_log_death(raw_snapshot.get("p1_stock"))
        if inputs_td is None or len(self.buffer) < self.warmup_frames:
            print_cached(
                f"[TRACE:GPTInferenceEngine.predict_from_raw] Warmup phase (buffer={len(self.buffer)}, warmup={self.warmup_frames}), returning neutral"
            )
            controller = ControllerState.neutral()
            self._update_prev_controller_features(controller)
            return controller
        print_cached("[TRACE:GPTInferenceEngine.predict_from_raw] Running model inference")
        with torch.inference_mode():
            outputs = self.model(inputs_td)
        print_cached("[TRACE:GPTInferenceEngine.predict_from_raw] Model inference complete")
        record.logits = self._capture_logits(outputs)
        controller = self._decode_outputs(outputs)
        self._update_prev_controller_features(controller)
        print_cached("[TRACE:GPTInferenceEngine.predict_from_raw] Prediction complete")
        return controller


_ACTIVE_ENGINE: Optional[GPTInferenceEngine] = None


def apply_model_outputs_to_game(
    controller: Controller, model_outputs: ControllerState
) -> None:
    """Apply ``model_outputs`` to the Dolphin controller with an example.

    Example
    -------
    If ``model_outputs`` has ``button_a=True`` and ``main_stick=(0.8, 0.3)``, the
    helper presses button A, tilts the main analog stick to ``(0.8, 0.3)``, and
    releases any buttons marked ``False``. The shoulder analog is set via
    ``press_shoulder`` using ``model_outputs.shoulder_analog``, reproducing the
    exact physical controller state encoded in the :class:`ControllerState`.
    """
    controller.release_all()
    if model_outputs.button_a:
        controller.press_button(enums.Button.BUTTON_A)
    else:
        controller.release_button(enums.Button.BUTTON_A)

    if model_outputs.button_b:
        controller.press_button(enums.Button.BUTTON_B)
    else:
        controller.release_button(enums.Button.BUTTON_B)

    if model_outputs.button_xy:
        controller.press_button(enums.Button.BUTTON_X)
    else:
        controller.release_button(enums.Button.BUTTON_X)

    if model_outputs.button_lr:
        controller.press_button(enums.Button.BUTTON_L)
    else:
        controller.release_button(enums.Button.BUTTON_L)

    if model_outputs.button_z:
        controller.press_button(enums.Button.BUTTON_Z)
    else:
        controller.release_button(enums.Button.BUTTON_Z)

    controller.tilt_analog(
        enums.Button.BUTTON_MAIN, model_outputs.main_stick_x, model_outputs.main_stick_y
    )
    controller.tilt_analog(
        enums.Button.BUTTON_C, model_outputs.c_stick_x, model_outputs.c_stick_y
    )

    controller.press_shoulder(enums.Button.BUTTON_R, 0.0)
    controller.press_shoulder(enums.Button.BUTTON_L, model_outputs.shoulder_analog)
