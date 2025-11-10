from __future__ import annotations

import dataclasses
import json
import math
from collections import deque
from datetime import datetime
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
    get_feature_names,
    get_target_names,
)
from train import build_model_inputs
from utils import _resolve_device

"""
Add this code to the TOP of model_interface.py (after imports)
"""
import os
import atexit
import cProfile
import pstats
from functools import wraps
from pathlib import Path

# Check if profiling is enabled via environment variable
ENABLE_PROFILING = os.environ.get('PROFILE_MODEL_INTERFACE', '0') == '1'

if ENABLE_PROFILING:
    print("[PROFILING] model_interface.py profiling enabled")
    _profiler = cProfile.Profile()
    _profiler.enable()


    def _save_profile_stats():
        """Save profiling stats on exit"""
        _profiler.disable()
        output_dir = Path.cwd() / "profiling_output"
        output_dir.mkdir(exist_ok=True)

        # Save raw stats
        stats_file = output_dir / "model_interface_profile.stats"
        _profiler.dump_stats(str(stats_file))
        print(f"[PROFILING] Raw stats saved to: {stats_file}")

        # Save human-readable report
        report_file = output_dir / "model_interface_profile.txt"
        with open(report_file, 'w') as f:
            ps = pstats.Stats(_profiler, stream=f)

            # Filter to only show model_interface.py functions
            ps.strip_dirs()
            f.write("=" * 80 + "\n")
            f.write("TOP 50 FUNCTIONS BY CUMULATIVE TIME\n")
            f.write("=" * 80 + "\n")
            ps.sort_stats('cumulative').print_stats('model_interface', 50)

            f.write("\n" + "=" * 80 + "\n")
            f.write("TOP 50 FUNCTIONS BY TOTAL TIME\n")
            f.write("=" * 80 + "\n")
            ps.sort_stats('tottime').print_stats('model_interface', 50)

            f.write("\n" + "=" * 80 + "\n")
            f.write("TOP 30 CALLERS\n")
            f.write("=" * 80 + "\n")
            ps.print_callers('model_interface', 30)

        print(f"[PROFILING] Human-readable report saved to: {report_file}")

        # Print summary to console
        print("\n" + "=" * 80)
        print("PROFILING SUMMARY (Top 20 by cumulative time)")
        print("=" * 80)
        ps = pstats.Stats(_profiler)
        ps.strip_dirs()
        ps.sort_stats('cumulative').print_stats('model_interface', 20)


    atexit.register(_save_profile_stats)
else:
    _profiler = None


def profile_function(func):
    """Decorator to profile individual functions"""
    if not ENABLE_PROFILING:
        return func

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper

_DEFAULT_FEATURE_NAMES = get_feature_names()
_DEFAULT_TARGET_NAMES = get_target_names()

# Derived features are populated during dataset generation. They are not available in
# live inference, so we provide safe fallbacks here.
_DERIVED_FEATURE_DEFAULTS: Dict[str, float] = {
    "value_target": 0.0,
}

# Global debug flag - set to False in production
DEBUG_LOGGING = False

_PRINT_CACHE: dict[str, int] = {}


def print_cached(s: str) -> None:
    """Only log if DEBUG_LOGGING is enabled."""
    if not DEBUG_LOGGING:
        return
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

    # Cache neutral state
    _neutral_instance: Optional["ControllerState"] = None

    @staticmethod
    def neutral() -> "ControllerState":
        """Return a cached neutral controller state."""
        if ControllerState._neutral_instance is None:
            ControllerState._neutral_instance = ControllerState(
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
        return ControllerState._neutral_instance


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
        return self.transformed[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.transformed)

    def __len__(self) -> int:
        return len(self.transformed)

    def as_dict(self) -> Dict[str, float]:
        return dict(self.transformed)


# Optimized: Use numpy vectorization for type conversion
def _safe_float(value: object) -> float:
    """Convert value to float efficiently."""
    if isinstance(value, float):
        return value
    if isinstance(value, (int, np.integer)):
        return float(value)
    if isinstance(value, np.floating):
        return float(value)
    try:
        return float(value)
    except Exception as e:
        if DEBUG_LOGGING:
            print_cached(f"[TRACE:_safe_float] Failed to convert {value}: {e}")
        raise


# Optimized: Remove excessive logging, simplify logic
def _coerce_scalar(value: object) -> float | int | bool:
    """Convert arbitrary scalars to Python bool/int/float."""
    if isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    try:
        return float(value)
    except Exception as exc:
        raise TypeError(
            f"Unable to coerce value {value!r} ({type(value)}) to scalar"
        ) from exc


def set_feature_transforms(transforms: Any) -> None:
    """Configure per-feature transforms used at inference time."""
    constants._FEATURE_TRANSFORMS_SPEC = build_transform_spec(transforms)


set_feature_transforms(FeatureConfig().transforms)


def _apply_transforms_to_features(features: Dict[str, float]) -> Dict[str, float]:
    """Apply the configured transform spec to the features mapping."""
    spec = _FEATURE_TRANSFORMS_SPEC
    if not spec or not spec.steps:
        spec = build_transform_spec(FeatureConfig().transforms)

    out = dict(features)
    keys = list(out.keys())
    key_set = set(keys)

    # Extract prefixes once
    prefixes: set[str] = set()
    for key in keys:
        head, _, tail = key.partition("_")
        if tail and head.startswith("p") and head[1:].isdigit():
            prefixes.add(head)

    # Cache resolved groups to avoid recomputation
    def _resolve_groups(requested: Sequence[str]) -> List[Tuple[str, ...]]:
        """Expand requested feature names to actual keys with prefix handling."""
        if all(name in key_set for name in requested):
            return [tuple(requested)]
        groups: List[Tuple[str, ...]] = []
        for prefix in sorted(prefixes):
            group: List[str] = []
            for feature in requested:
                key = f"{prefix}_{feature}"
                if key not in key_set:
                    break
                group.append(key)
            else:
                if group:
                    groups.append(tuple(group))
        return groups

    for step in spec.steps:
        groups = _resolve_groups(step.features)
        if not groups:
            continue
        for group in groups:
            if len(group) == 1:
                key = group[0]
                value = np.array(out[key], dtype=np.float32)
                result = step.fn(value.copy())
                if result is None:
                    result = value
                if result.shape != value.shape:
                    raise ValueError(
                        f"Transform '{step.transform}' expected output shape {value.shape}, got {result.shape}."
                    )
                out[key] = float(result)
            else:
                block = np.array([[float(out[k]) for k in group]], dtype=np.float32)
                result = step.fn(block.copy())
                if result is None:
                    result = block
                if result.shape != block.shape:
                    raise ValueError(
                        f"Transform '{step.transform}' expected output shape {block.shape}, got {result.shape}."
                    )
                for idx, key in enumerate(group):
                    out[key] = float(result[0, idx])

    return out


# Cache controller feature template
_CONTROLLER_FEATURE_TEMPLATE = [
    "button_a", "button_b", "button_xy", "button_lr", "button_z",
    "main_stick_x", "main_stick_y", "c_stick_x", "c_stick_y", "shoulder_analog"
]


def _controller_state_to_features(
        prefix: str, state: ControllerState
) -> Dict[str, float]:
    """Convert a ControllerState into prefixed feature values."""
    # Use list comprehension for faster dictionary construction
    return {
        f"{prefix}_button_a": float(state.button_a),
        f"{prefix}_button_b": float(state.button_b),
        f"{prefix}_button_xy": float(state.button_xy),
        f"{prefix}_button_lr": float(state.button_lr),
        f"{prefix}_button_z": float(state.button_z),
        f"{prefix}_main_stick_x": state.main_stick_x,
        f"{prefix}_main_stick_y": state.main_stick_y,
        f"{prefix}_c_stick_x": state.c_stick_x,
        f"{prefix}_c_stick_y": state.c_stick_y,
        f"{prefix}_shoulder_analog": state.shoulder_analog,
    }


# Cache zero player fields
_ZERO_PLAYER_CACHE: Dict[str, Dict[str, float]] = {}


def _zero_player_fields(prefix: str) -> Dict[str, float]:
    """Return zero-valued player features for prefix (cached)."""
    if prefix not in _ZERO_PLAYER_CACHE:
        _ZERO_PLAYER_CACHE[prefix] = {f"{prefix}_{name}": dtype(0) for name, dtype in PLAYER_SPEC}
    return _ZERO_PLAYER_CACHE[prefix].copy()


def _prefixed_player_fields(player, prefix: str) -> Dict[str, float]:
    """Extract prefixed player fields or fall back to zeros."""
    if player is None or getattr(player, "controller_state", None) is None:
        return _zero_player_fields(prefix)

    try:
        extracted = extract_player_fields(player)
    except ValueError:
        return _zero_player_fields(prefix)

    # Use dictionary comprehension for efficiency
    return {f"{prefix}_{name}": value for name, value in extracted.items()}


def model_to_dolphin01(
        model_out: np.ndarray,
        palette11: np.ndarray | None = None,
) -> np.ndarray:
    """Convert model outputs to Dolphin's [0, 1] coordinate space."""
    arr = np.asarray(model_out)

    if np.issubdtype(arr.dtype, np.integer):
        if palette11 is None:
            raise ValueError("palette11 must be provided when converting indices")
        P = np.asarray(palette11, dtype=np.float32)
        coords11 = P[arr]
    else:
        coords11 = np.asarray(model_out, dtype=np.float32)
        if coords11.shape[-1] != 2:
            raise ValueError("model_out must have last dimension size 2")

    # Vectorized clamping to unit circle
    r2 = np.einsum("...i,...i->...", coords11, coords11)
    over = r2 > 1.0
    if np.any(over):
        coords11 = coords11.copy()
        coords11[over] /= np.sqrt(r2[over])[..., None]

    # Map [-1,1] -> [0,1]
    xy01 = np.clip(coords11 * 0.5 + 0.5, 0.0, 1.0).astype(np.float32)
    return xy01


def collect_raw_inputs_from_gamestate(
        gamestate: GameState,
        bot_port: int,
        opp_port: int,
) -> ModelFrameInputs:
    """Extract both raw and transformed feature dictionaries."""
    ego_player = gamestate.players.get(bot_port)
    opp_player = gamestate.players.get(opp_port)

    values: Dict[str, float] = extract_common_fields(gamestate)

    # Combine dictionaries efficiently
    player_values: Dict[str, float] = {}
    player_values.update(_prefixed_player_fields(ego_player, "p1"))
    player_values.update(_prefixed_player_fields(opp_player, "p2"))

    combined = {**values, **player_values}

    # Single pass coercion
    combined_raw = {name: _coerce_scalar(val) for name, val in combined.items()}
    transformed = _apply_transforms_to_features(combined_raw)

    feature_names = get_feature_names()

    # Build final dict with validation
    final: Dict[str, float] = {}
    for name in feature_names:
        if name not in transformed:
            if name in _DERIVED_FEATURE_DEFAULTS:
                final[name] = _DERIVED_FEATURE_DEFAULTS[name]
                continue
            raise KeyError(f"Feature '{name}' missing from collected inputs.")
        final[name] = _safe_float(transformed[name])

    return ModelFrameInputs(raw=combined_raw, transformed=final)


class GPTInferenceEngine:
    """Online inference helper around the GPT controller model."""

    def __init__(
            self,
            checkpoint_path: str | Path,
    ) -> None:
        """Load the GPT checkpoint and initialize inference buffers."""
        print_cached(f"[TRACE:GPTInferenceEngine.__init__] Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        train_cfg = ckpt.get("config", {})

        data_root = Path(train_cfg.get("data_root", "dataset_FOX_vs_FOX"))
        meta_path = data_root / "meta.json"
        transforms_spec: Optional[Any] = None
        if meta_path.exists():
            with meta_path.open("r") as f:
                meta = json.load(f)

            feature_names = meta["schema"]["features"]
            target_names = meta["schema"]["targets"]
            self.seq_len = int(meta.get("seq_len", 256))
            build_cfg = meta.get("build_config")
            if isinstance(build_cfg, Mapping):
                features_cfg = build_cfg.get("features")
                if isinstance(features_cfg, Mapping):
                    transforms_cfg = features_cfg.get("transforms")
                    if transforms_cfg:
                        transforms_spec = transforms_cfg
        else:
            feature_names = list(_DEFAULT_FEATURE_NAMES)
            target_names = list(_DEFAULT_TARGET_NAMES)
            self.seq_len = 256

        self.device = _resolve_device()
        self.shoulder_centers = SHOULDER_QUANTIZED
        self.warmup_frames = 256
        self._main_stick_palette = np.asarray(CONTROL_STICK_QUANTIZED, dtype=np.float32)
        self._c_stick_palette = np.asarray(C_STICK_QUANTIZED, dtype=np.float32)

        if transforms_spec is None and isinstance(train_cfg, Mapping):
            features_cfg = train_cfg.get("features")
            if isinstance(features_cfg, Mapping):
                transforms_cfg = features_cfg.get("transforms")
                if transforms_cfg:
                    transforms_spec = transforms_cfg

        if transforms_spec is not None:
            set_feature_transforms(transforms_spec)

        self.feature_names = list(feature_names)
        self.target_names = list(target_names)
        self.colmap = ColumnMap(self.feature_names, self.target_names)

        config = get_config()
        ckpt_cfg = ckpt.get("config")
        model_cfg_from_ckpt: Optional[Mapping[str, Any]] = None
        if isinstance(ckpt_cfg, Mapping):
            maybe_model_cfg = ckpt_cfg.get("model")
            if isinstance(maybe_model_cfg, Mapping):
                model_cfg_from_ckpt = maybe_model_cfg
                for field, value in maybe_model_cfg.items():
                    setattr(config.model, field, value)

        if getattr(config.model, "input_size", -1) < 0:
            gamestate_dim = len(self.colmap.gamestate_idxs)
            controller_dim = len(self.colmap.controller_idxs)
            config.model.input_size = (
                    config.model.num_stages
                    + config.model.num_characters * 2
                    + config.model.num_actions * 2
                    + gamestate_dim
                    + controller_dim
            )

        self.model = GPT(config).to(self.device)
        load_result = self.model.load_state_dict(ckpt["model"], strict=False)
        if load_result.unexpected_keys:
            dropped = ", ".join(sorted(load_result.unexpected_keys))
            print(f"[WARN] Dropping unexpected model keys: {dropped}")
        if load_result.missing_keys:
            missing = ", ".join(sorted(load_result.missing_keys))
            print(f"[WARN] Missing model keys during load: {missing}")
        self.model.eval()

        self.feature_dim = len(self.feature_names)
        self.buffer: deque[torch.Tensor] = deque(maxlen=self.seq_len)
        self.frame_history: deque[FrameRecord] = deque(maxlen=self.seq_len)

        # Build controller feature keys once
        controller_feature_keys: list[str] = []
        for group in CONTROLLER_KEY_GROUPS.values():
            for name in group:
                key = f"p1_{name}"
                if key in self.feature_names and key not in controller_feature_keys:
                    controller_feature_keys.append(key)
        shoulder_key = "p1_shoulder_analog"
        if shoulder_key in self.feature_names and shoulder_key not in controller_feature_keys:
            controller_feature_keys.append(shoulder_key)
        self._controller_feature_keys: Tuple[str, ...] = tuple(controller_feature_keys)

        # Create index mapping for fast tensor construction
        self._feature_name_to_idx = {name: idx for idx, name in enumerate(self.feature_names)}

        self._prev_controller_features: Dict[str, float] = {}
        self._update_prev_controller_features(ControllerState.neutral())
        self._frames_seen = 0
        self._death_log_dir = Path.cwd() / "death_logs"
        self._death_counter = 0
        self._prev_stock: Optional[int] = None

    def _frame_to_tensor(self, raw_inputs: Dict[str, float]) -> torch.Tensor:
        """Convert raw_inputs into an ordered tensor of feature values (optimized)."""
        # Use pre-allocated tensor and vectorized assignment
        frame = torch.zeros(self.feature_dim, dtype=torch.float32)

        # Batch convert to reduce function call overhead
        for idx, name in enumerate(self.feature_names):
            frame[idx] = raw_inputs[name]  # Already floats from upstream

        return frame

    def _stack_frames(self) -> Optional[torch.Tensor]:
        """Stack buffered frames into a batch tensor."""
        if not self.buffer:
            return None
        # torch.stack is already optimized
        stacked = torch.stack(list(self.buffer), dim=0)
        return stacked.unsqueeze(0).to(self.device)

    def _build_inputs(self, batch_X: torch.Tensor) -> TensorDict:
        """Wrap batch_X in the structured TensorDict used by the model."""
        return build_model_inputs(batch_X, self.colmap)

    def _override_controller_features(
            self, features: Mapping[str, float]
    ) -> Dict[str, float]:
        """Prefer live controller readings, only fall back to cached values if missing."""
        if not self._controller_feature_keys:
            return dict(features)

        out = dict(features)
        for key in self._controller_feature_keys:
            if key not in out:
                continue
            val = out[key]
            if val is None:
                cached = self._prev_controller_features.get(key)
                if cached is not None:
                    out[key] = cached
                continue
            if isinstance(val, float) and math.isnan(val):
                cached = self._prev_controller_features.get(key)
                if cached is not None:
                    out[key] = cached
        return out

    def _update_prev_controller_features(self, state: ControllerState) -> None:
        """Cache the transformed controller state."""
        values = _controller_state_to_features("p1", state)
        if _FEATURE_TRANSFORMS_SPEC and _FEATURE_TRANSFORMS_SPEC.steps:
            transformed = _apply_transforms_to_features(dict(values))
            self._prev_controller_features = {k: float(transformed[k]) for k in values}
        else:
            self._prev_controller_features = {k: float(v) for k, v in values.items()}

    def _snapshot_features(self, features: Mapping[str, float]) -> Dict[str, float]:
        """Capture transformed feature values in model order."""
        return {name: float(features[name]) for name in self.feature_names}

    def _snapshot_raw(self, raw_inputs: Mapping[str, float]) -> Dict[str, float]:
        """Normalize raw mapping values into plain Python scalars."""
        return {name: _coerce_scalar(value) for name, value in raw_inputs.items()}

    def _snapshot_targets(self, raw_inputs: Mapping[str, float]) -> Dict[str, float]:
        """Extract controller targets from raw_inputs."""
        return {name: float(raw_inputs[name]) for name in self.target_names}

    def _record_frame(
            self, model_features: Mapping[str, float], raw_inputs: Mapping[str, float]
    ) -> FrameRecord:
        """Append the current frame's data to history."""
        record = FrameRecord(
            raw_features=self._snapshot_raw(raw_inputs),
            transformed_features=self._snapshot_features(model_features),
            targets=self._snapshot_targets(raw_inputs),
        )
        self.frame_history.append(record)
        return record

    def _capture_logits(self, outputs: TensorDict) -> Dict[str, Any]:
        """Convert model logits into CPU lists for logging."""
        # Use .item() for single values, .tolist() for arrays
        return {
            "main_stick": outputs["main_stick"][0, -1].detach().cpu().tolist(),
            "c_stick": outputs["c_stick"][0, -1].detach().cpu().tolist(),
            "buttons": outputs["buttons"][0, -1].detach().cpu().tolist(),
            "shoulder": outputs["shoulder"][0, -1].detach().cpu().tolist(),
            "value": outputs["value"][0, -1].detach().cpu().tolist(),
        }

    def _persist_death_record(self, stock_after: int) -> None:
        """Write the current frame history to disk after a stock loss."""
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
            filename = f"death_{self._death_counter:04d}_frame{self._frames_seen}.json"
            output_path = self._death_log_dir / filename
            with output_path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            self._death_counter += 1
        except Exception:
            pass

    def _maybe_log_death(self, current_stock: Optional[float]) -> None:
        """Detect stock drops and trigger death logging."""
        if current_stock is None:
            return
        stock_value = int(current_stock)
        if self._prev_stock is not None and stock_value < self._prev_stock:
            self._persist_death_record(stock_value)
        self._prev_stock = stock_value

    # TODO: Raw or processed here?
    def _decode_stick(
            self, logits: torch.Tensor, palette: np.ndarray, stick_name: str
    ) -> np.ndarray:
        """Convert stick logits into palette coordinates."""
        idx = torch.argmax(logits.detach(), dim=-1)
        idx_np = idx.cpu().numpy().astype(np.int32)
        xy01 = model_to_dolphin01(idx_np, palette11=palette)
        return xy01.reshape(-1)

    def _decode_buttons(self, probs: torch.Tensor) -> List[bool]:
        """Sample button activations from probabilities."""
        eps = torch.finfo(probs.dtype).eps
        clamped_probs = torch.clamp(probs.detach(), eps, 1 - eps)
        samples = torch.bernoulli(clamped_probs).bool().cpu()
        return samples.tolist()

    def _decode_outputs(self, outputs: TensorDict) -> ControllerState:
        """Assemble decoded sticks, buttons, and shoulder into controller output."""
        main_logits = outputs["main_stick"][0, -1]
        c_logits = outputs["c_stick"][0, -1]
        button_probs = torch.sigmoid(outputs["buttons"])
        button_probs = button_probs[0, -1]

        shoulder_logits = outputs.get("shoulder")

        main_xy = self._decode_stick(main_logits, self._main_stick_palette, "main_stick")
        c_xy = self._decode_stick(c_logits, self._c_stick_palette, "c_stick")
        buttons_bool = self._decode_buttons(button_probs)

        s_idx = int(torch.argmax(shoulder_logits[0, -1]).item())
        s_idx = max(0, min(s_idx, len(self.shoulder_centers) - 1))
        shoulder_val = float(self.shoulder_centers[s_idx])

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
        """Update the frame buffer and return model inputs."""
        frame = self._frame_to_tensor(raw_inputs)
        self.buffer.append(frame)
        batch = self._stack_frames()
        if batch is None:
            return None
        return self._build_inputs(batch)

    def predict_from_raw(
            self, frame_inputs: Mapping[str, float] | ModelFrameInputs
    ) -> ControllerState:
        """Run inference on raw or preprocessed frame inputs."""
        # Fast path type checking
        if isinstance(frame_inputs, ModelFrameInputs):
            raw_snapshot = frame_inputs.raw
            transformed = frame_inputs.transformed
        else:
            # Single-pass coercion
            raw_snapshot = {k: _coerce_scalar(v) for k, v in frame_inputs.items()}
            transformed = {k: float(v) for k, v in frame_inputs.items()}

        features = self._override_controller_features(transformed)
        record = self._record_frame(features, raw_snapshot)
        inputs_td = self.prepare_inputs(features)
        self._frames_seen += 1
        self._maybe_log_death(raw_snapshot.get("p1_stock"))

        if inputs_td is None or len(self.buffer) < self.warmup_frames:
            controller = ControllerState.neutral()
            self._update_prev_controller_features(controller)
            return controller

        with torch.inference_mode():
            outputs = self.model(inputs_td)

        record.logits = self._capture_logits(outputs)
        controller = self._decode_outputs(outputs)
        self._update_prev_controller_features(controller)
        return controller


_ACTIVE_ENGINE: Optional[GPTInferenceEngine] = None


def apply_model_outputs_to_game(
        controller: Controller, model_outputs: ControllerState
) -> None:
    """Apply model_outputs to the Dolphin controller."""
    controller.release_all()

    # Batch button operations
    if model_outputs.button_a:
        controller.press_button(enums.Button.BUTTON_A)
    if model_outputs.button_b:
        controller.press_button(enums.Button.BUTTON_B)
    if model_outputs.button_xy:
        controller.press_button(enums.Button.BUTTON_X)
    if model_outputs.button_lr:
        controller.press_button(enums.Button.BUTTON_L)
    if model_outputs.button_z:
        controller.press_button(enums.Button.BUTTON_Z)

    controller.tilt_analog(
        enums.Button.BUTTON_MAIN, model_outputs.main_stick_x, model_outputs.main_stick_y
    )
    controller.tilt_analog(
        enums.Button.BUTTON_C, model_outputs.c_stick_x, model_outputs.c_stick_y
    )

    controller.press_shoulder(enums.Button.BUTTON_R, 0.0)
    controller.press_shoulder(enums.Button.BUTTON_L, model_outputs.shoulder_analog)
