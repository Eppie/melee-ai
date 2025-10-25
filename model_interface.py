from __future__ import annotations

import dataclasses
import json
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
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

from column_map import ColumnMap, CONTROLLER_KEY_GROUPS
from config import FeatureConfig, get_config
from controller_utils import (
    CONTROL_STICK_QUANTIZED,
    C_STICK_QUANTIZED,
    SHOULDER_QUANTIZED,
)
from feature_transforms import FeatureTransformSpec, build_transform_spec
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
from train.display import _format_action, _print_table_block
from utils import _resolve_device

_DEFAULT_FEATURE_NAMES = get_feature_names()

_DEFAULT_TARGET_NAMES = get_target_names()

_FEATURE_TRANSFORMS_SPEC: Optional[FeatureTransformSpec] = None


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
        return self.transformed[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.transformed)

    def __len__(self) -> int:
        return len(self.transformed)

    def as_dict(self) -> Dict[str, float]:
        return dict(self.transformed)


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception as e:
        print(f"Failed to convert `{value}`, of type `{type(value)}` to float: {e}")
        raise


def _coerce_scalar(value: object) -> float | int | bool:
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


def set_feature_transforms(transforms: Optional[Any]) -> None:
    """Configure per-feature transforms used at inference time."""

    global _FEATURE_TRANSFORMS_SPEC
    if not transforms:
        _FEATURE_TRANSFORMS_SPEC = None
        return

    _FEATURE_TRANSFORMS_SPEC = build_transform_spec(transforms)


set_feature_transforms(FeatureConfig().transforms)


def _apply_transforms_to_features(features: Dict[str, float]) -> Dict[str, float]:
    spec = _FEATURE_TRANSFORMS_SPEC
    if not spec or not spec.steps:
        return features

    out = dict(features)
    keys = list(out.keys())
    key_set = set(keys)

    prefixes: set[str] = set()
    for key in keys:
        head, _, tail = key.partition("_")
        if tail and head.startswith("p") and head[1:].isdigit():
            prefixes.add(head)

    def _resolve_groups(requested: Sequence[str]) -> List[Tuple[str, ...]]:
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


def _controller_state_to_features(
    prefix: str, state: ControllerState
) -> Dict[str, float]:
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
    return {f"{prefix}_{name}": dtype(0) for name, dtype in PLAYER_SPEC}


def _prefixed_player_fields(player, prefix: str) -> Dict[str, float]:
    if player is None or getattr(player, "controller_state", None) is None:
        return _zero_player_fields(prefix)

    try:
        extracted = extract_player_fields(player)
    except ValueError:
        return _zero_player_fields(prefix)

    return {f"{prefix}_{name}": value for name, value in extracted.items()}


def model_to_dolphin01(
    model_out: np.ndarray,
    palette11: np.ndarray | None = None,
) -> np.ndarray:
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

    # Clamp to unit circle to be safe.
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
    """Extract both raw + transformed feature dictionaries expected by the model."""

    ego_player = gamestate.players.get(bot_port)
    opp_player = gamestate.players.get(opp_port)

    values: Dict[str, float] = extract_common_fields(gamestate)
    prefixed_common = {name: val for name, val in values.items()}

    player_values: Dict[str, float] = {}
    player_values.update(_prefixed_player_fields(ego_player, "p1"))
    player_values.update(_prefixed_player_fields(opp_player, "p2"))

    combined = {**prefixed_common, **player_values}
    combined_raw = {name: _coerce_scalar(val) for name, val in combined.items()}
    transformed = _apply_transforms_to_features(dict(combined_raw))

    feature_names = get_feature_names()

    final: Dict[str, float] = {}
    for name in feature_names:
        if name not in transformed:
            raise KeyError(f"Feature '{name}' missing from collected inputs.")
        final[name] = _safe_float(transformed[name])

    return ModelFrameInputs(raw=combined_raw, transformed=final)


class GPTInferenceEngine:
    """Online inference helper around the GPT controller model."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        data_root: Optional[str | Path] = None,
        history: Optional[int] = None,
        warmup_frames: int = 128,
    ) -> None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        train_cfg = ckpt.get("config", {})

        data_root = Path(data_root or train_cfg.get("data_root", "dataset_FOX_vs_FOX"))
        meta_path = data_root / "meta.json"
        transforms_spec: Optional[Any] = None
        if meta_path.exists():
            with meta_path.open("r") as f:
                meta = json.load(f)

            feature_names = meta["schema"]["features"]
            target_names = meta["schema"]["targets"]
            self.seq_len = history or int(meta.get("seq_len", 256))
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
            self.seq_len = history or 256
        self.device = _resolve_device()
        self.shoulder_centers = SHOULDER_QUANTIZED
        self.warmup_frames = warmup_frames
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
        if (
            shoulder_key in self.feature_names
            and shoulder_key not in controller_feature_keys
        ):
            controller_feature_keys.append(shoulder_key)
        self._controller_feature_keys: Tuple[str, ...] = tuple(controller_feature_keys)
        self._prev_controller_features: Dict[str, float] = {}
        self._update_prev_controller_features(ControllerState.neutral())
        self._frames_seen = 0
        self._death_log_dir = Path.cwd() / "death_logs"
        self._death_counter = 0
        self._prev_stock: Optional[int] = None

    def _frame_to_tensor(self, raw_inputs: Dict[str, float]) -> torch.Tensor:
        frame = torch.zeros(self.feature_dim, dtype=torch.float32)
        for idx, name in enumerate(self.feature_names):
            frame[idx] = _safe_float(raw_inputs.get(name))
        return frame

    def _stack_frames(self) -> Optional[torch.Tensor]:
        if not self.buffer:
            return None
        frames = list(self.buffer)
        stacked = torch.stack(frames, dim=0)
        return stacked.unsqueeze(0).to(self.device)

    def _build_inputs(self, batch_X: torch.Tensor) -> TensorDict:
        return build_model_inputs(batch_X, self.colmap)

    def _preview_recent_frames(self) -> None:
        if not self.buffer:
            return
        frames = list(self.buffer)[-10:]
        data = torch.stack(frames, dim=0).cpu().numpy()
        feature_names = list(self.feature_names)
        formatters: Dict[str, Callable[[object], str]] = {}
        for key in feature_names:
            if key.endswith("_action"):
                formatters[key] = _format_action

        # print("=== Inference preview (most recent frames) ===")
        #_print_table_block(
        #    "Features",
        #    feature_names,
        #    data,
        #    formatters=formatters,
        #    max_columns=10,
        #)

    def _override_controller_features(
        self, features: Mapping[str, float]
    ) -> Dict[str, float]:
        updated = dict(features)
        for key in self._controller_feature_keys:
            if key in self._prev_controller_features and key in updated:
                updated[key] = self._prev_controller_features[key]
        return updated

    def _update_prev_controller_features(self, state: ControllerState) -> None:
        values = _controller_state_to_features("p1", state)
        if _FEATURE_TRANSFORMS_SPEC and _FEATURE_TRANSFORMS_SPEC.steps:
            transformed = _apply_transforms_to_features(dict(values))
            self._prev_controller_features = {k: float(transformed[k]) for k in values}
        else:
            self._prev_controller_features = {k: float(v) for k, v in values.items()}

    def _snapshot_features(self, features: Mapping[str, float]) -> Dict[str, float]:
        snapshot: Dict[str, float] = {}
        for name in self.feature_names:
            if name not in features:
                raise KeyError(f"Feature '{name}' missing from snapshot inputs.")
            snapshot[name] = float(_safe_float(features[name]))
        return snapshot

    def _snapshot_raw(self, raw_inputs: Mapping[str, float]) -> Dict[str, float]:
        return {name: _coerce_scalar(value) for name, value in raw_inputs.items()}

    def _snapshot_targets(self, raw_inputs: Mapping[str, float]) -> Dict[str, float]:
        targets: Dict[str, float] = {}
        for name in self.target_names:
            if name not in raw_inputs:
                raise KeyError(
                    f"Target '{name}' missing from raw inputs during logging."
                )
            targets[name] = float(_safe_float(raw_inputs[name]))
        return targets

    def _record_frame(
        self, model_features: Mapping[str, float], raw_inputs: Mapping[str, float]
    ) -> FrameRecord:
        record = FrameRecord(
            raw_features=self._snapshot_raw(raw_inputs),
            transformed_features=self._snapshot_features(model_features),
            targets=self._snapshot_targets(raw_inputs),
        )
        self.frame_history.append(record)
        return record

    def _capture_logits(self, outputs: TensorDict) -> Dict[str, Any]:
        logits: Dict[str, Any] = {}
        if "main_stick" in outputs:
            logits["main_stick"] = outputs["main_stick"][0, -1].detach().cpu().tolist()
        if "c_stick" in outputs:
            logits["c_stick"] = outputs["c_stick"][0, -1].detach().cpu().tolist()
        if "buttons" in outputs:
            logits["buttons"] = outputs["buttons"][0, -1].detach().cpu().tolist()
        if "shoulder" in outputs:
            logits["shoulder"] = outputs["shoulder"][0, -1].detach().cpu().tolist()
        for key in ("value", "value_head"):
            if key in outputs:
                logits[key] = outputs[key][0, -1].detach().cpu().tolist()
        return logits

    def _persist_death_record(self, stock_after: int) -> None:
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
        except Exception as exc:
            print(
                f"Failed to create death log directory '{self._death_log_dir}': {exc}"
            )
            return
        filename = f"death_{self._death_counter:04d}_frame{self._frames_seen}.json"
        output_path = self._death_log_dir / filename
        try:
            with output_path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            print(f"Logged death to {output_path}")
        except Exception as exc:
            print(f"Failed to write death log '{output_path}': {exc}")
            return
        self._death_counter += 1

    def _maybe_log_death(self, current_stock: Optional[float]) -> None:
        if current_stock is None:
            return
        stock_value = int(current_stock)
        if self._prev_stock is not None and stock_value < self._prev_stock:
            if len(self.frame_history) < self.seq_len:
                print(
                    "Death detected with only "
                    f"{len(self.frame_history)} frames buffered; logging truncated history."
                )
            self._persist_death_record(stock_value)
        self._prev_stock = stock_value

    def _decode_stick(
        self, logits: torch.Tensor, palette: np.ndarray, stick_name: str
    ) -> np.ndarray:
        # Decode sticks by selecting the most likely quantized bin (argmax).
        idx = torch.argmax(logits.detach(), dim=-1)
        if stick_name == "c_stick":
            probs = torch.softmax(logits.detach(), dim=-1).cpu().numpy()
        idx_np = idx.detach().cpu().numpy().astype(np.int32)
        xy01 = model_to_dolphin01(idx_np, palette11=palette)
        return xy01.reshape(-1)

    def _decode_buttons(self, probs: torch.Tensor) -> List[bool]:
        # Interpret button activations probabilistically, sampling directly from the model probabilities.
        eps = torch.finfo(probs.dtype).eps
        clamped_probs = torch.clamp(probs.detach(), eps, 1 - eps)
        samples = torch.bernoulli(clamped_probs).bool().cpu()
        return samples.tolist()

    def _decode_outputs(self, outputs: TensorDict) -> ControllerState:
        main_logits = outputs["main_stick"][0, -1]
        c_logits = outputs["c_stick"][0, -1]
        button_probs = outputs.get("buttons_probs")
        if button_probs is None:
            button_probs = torch.sigmoid(outputs["buttons"])
        button_probs = button_probs[0, -1]

        shoulder_logits = outputs.get("shoulder")

        main_xy = self._decode_stick(
            main_logits, self._main_stick_palette, "main_stick"
        )
        c_xy = self._decode_stick(c_logits, self._c_stick_palette, "c_stick")
        buttons_bool = self._decode_buttons(button_probs)

        if shoulder_logits is not None:
            s_idx = int(torch.argmax(shoulder_logits[0, -1]).item())
            s_idx = max(0, min(s_idx, len(self.shoulder_centers) - 1))
            shoulder_val = float(self.shoulder_centers[s_idx])
        else:
            shoulder_val = 0.0

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
        frame = self._frame_to_tensor(raw_inputs)
        self.buffer.append(frame)
        batch = self._stack_frames()
        if batch is None:
            return None
        return self._build_inputs(batch)

    def predict_from_raw(
        self, frame_inputs: Mapping[str, float] | ModelFrameInputs
    ) -> ControllerState:
        if isinstance(frame_inputs, ModelFrameInputs):
            raw_snapshot = dict(frame_inputs.raw)
            transformed = dict(frame_inputs.transformed)
        else:
            raw_snapshot = {k: _coerce_scalar(v) for k, v in frame_inputs.items()}
            transformed = {k: float(_safe_float(v)) for k, v in frame_inputs.items()}

        features = self._override_controller_features(transformed)
        record = self._record_frame(features, raw_snapshot)
        inputs_td = self.prepare_inputs(features)
        self._frames_seen += 1
        self._maybe_log_death(raw_snapshot.get("p1_stock"))
        if (
            self.buffer
            and len(self.buffer) >= self.warmup_frames
            and self._frames_seen % 1000 == 0
        ):
            self._preview_recent_frames()
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
