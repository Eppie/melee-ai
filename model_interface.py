from __future__ import annotations

import dataclasses
import json
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from tensordict import TensorDict

from gpt import GPTv7
from libmelee.melee import enums
from libmelee.melee.controller import Controller
from libmelee.melee.gamestate import GameState
from preprocess import C_STICK_XY_CLUSTER_CENTERS_V0_1, FOX_STICK_64, model_to_dolphin01
from train import build_inputs_for_gptv7

# Keep the feature ordering in-sync with training.
_FEATURE_CONTROLLER_KEYS = {
    "main": ("main_stick_x", "main_stick_y"),
    "c": ("c_stick_x", "c_stick_y"),
    "buttons": ("button_a", "button_b", "button_xy", "button_z", "button_lr"),
    "shoulder": ("shoulder_analog",),
}
_BUTTON_TARGETS = [
    "p1_button_a",
    "p1_button_b",
    "p1_button_xy",
    "p1_button_z",
    "p1_button_lr",
]

_DEFAULT_FEATURE_NAMES = [
    "stage",
    "p1_action",
    "p1_character",
    "p1_position_x",
    "p1_position_y",
    "p1_percent",
    "p1_stock",
    "p1_facing",
    "p1_on_ground",
    "p1_button_a",
    "p1_button_b",
    "p1_button_xy",
    "p1_button_z",
    "p1_button_lr",
    "p1_main_stick_x",
    "p1_main_stick_y",
    "p1_c_stick_x",
    "p1_c_stick_y",
    "p1_shoulder_analog",
    "p1_shield_strength",
    "p1_is_invulnerable",
    "p1_jumps_left",
    "p2_action",
    "p2_character",
    "p2_position_x",
    "p2_position_y",
    "p2_percent",
    "p2_stock",
    "p2_facing",
    "p2_on_ground",
    "p2_button_a",
    "p2_button_b",
    "p2_button_xy",
    "p2_button_z",
    "p2_button_lr",
    "p2_main_stick_x",
    "p2_main_stick_y",
    "p2_c_stick_x",
    "p2_c_stick_y",
    "p2_shoulder_analog",
    "p2_shield_strength",
    "p2_is_invulnerable",
    "p2_jumps_left",
]

_DEFAULT_TARGET_NAMES = _BUTTON_TARGETS + [
    "p1_main_stick_x",
    "p1_main_stick_y",
    "p1_c_stick_x",
    "p1_c_stick_y",
    "p1_shoulder_analog",
]

_DEFAULT_BUTTON_THRESHOLD = 0.5
_DEFAULT_SHOULDER_CENTERS = (0.0, 0.7, 0.85)


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


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _bool_to_float(value: bool) -> float:
    return 1.0 if value else 0.0


def _player_fields(player, prefix: str) -> Dict[str, float]:
    if player is None:
        zeros: Dict[str, float] = {}
        for key in [
            "action",
            "character",
            "position_x",
            "position_y",
            "percent",
            "stock",
            "facing",
            "on_ground",
            "button_a",
            "button_b",
            "button_xy",
            "button_z",
            "button_lr",
            "main_stick_x",
            "main_stick_y",
            "c_stick_x",
            "c_stick_y",
            "shoulder_analog",
            "shield_strength",
            "is_invulnerable",
            "jumps_left",
        ]:
            zeros[f"{prefix}_{key}"] = 0.0
        return zeros

    controller_state = player.controller_state
    button = controller_state.button

    button_xy = bool(button[enums.Button.BUTTON_X]) or bool(button[enums.Button.BUTTON_Y])
    button_lr = bool(button[enums.Button.BUTTON_L]) or bool(button[enums.Button.BUTTON_R])

    return {
        f"{prefix}_action": float(player.action.value),
        f"{prefix}_character": float(player.character.value),
        f"{prefix}_position_x": float(player.position.x),
        f"{prefix}_position_y": float(player.position.y),
        f"{prefix}_percent": float(player.percent),
        f"{prefix}_stock": float(player.stock),
        f"{prefix}_facing": _bool_to_float(bool(player.facing)),
        f"{prefix}_on_ground": _bool_to_float(bool(player.on_ground)),
        f"{prefix}_button_a": _bool_to_float(bool(button[enums.Button.BUTTON_A])),
        f"{prefix}_button_b": _bool_to_float(bool(button[enums.Button.BUTTON_B])),
        f"{prefix}_button_xy": _bool_to_float(button_xy),
        f"{prefix}_button_z": _bool_to_float(bool(button[enums.Button.BUTTON_Z])),
        f"{prefix}_button_lr": _bool_to_float(button_lr),
        f"{prefix}_main_stick_x": float(controller_state.main_stick[0]),
        f"{prefix}_main_stick_y": float(controller_state.main_stick[1]),
        f"{prefix}_c_stick_x": float(controller_state.c_stick[0]),
        f"{prefix}_c_stick_y": float(controller_state.c_stick[1]),
        f"{prefix}_shoulder_analog": float(controller_state.l_shoulder),
        f"{prefix}_shield_strength": float(getattr(player, "shield_strength", 0.0)),
        f"{prefix}_is_invulnerable": _bool_to_float(bool(getattr(player, "invulnerable", False))),
        f"{prefix}_jumps_left": float(getattr(player, "jumps_left", 0)),
    }


def collect_raw_inputs_from_gamestate(
        gamestate: GameState,
        bot_port: int,
        opp_port: int,
) -> Dict[str, float]:
    """Extract the raw feature dictionary expected by the transformer."""

    ego_player = gamestate.players.get(bot_port)
    opp_player = gamestate.players.get(opp_port)

    features: Dict[str, float] = {
        "stage": float(gamestate.stage.value),
    }
    features.update(_player_fields(ego_player, "p1"))
    features.update(_player_fields(opp_player, "p2"))
    return features


class InferenceColumnMap:
    """Minimal ColumnMap used at inference time."""

    def __init__(self, feature_names: Sequence[str], target_names: Sequence[str]) -> None:
        self.feat_names = list(feature_names)
        self.targ_names = list(target_names)

        name2idx = {n: i for i, n in enumerate(self.feat_names)}
        self.stage_idx = name2idx["stage"]
        self.ego_char_idx = name2idx["p1_character"]
        self.opp_char_idx = name2idx["p2_character"]
        self.ego_action_idx = name2idx["p1_action"]
        self.opp_action_idx = name2idx["p2_action"]

        self.controller_idxs: List[int] = []
        for prefix in ("p1_", "p2_"):
            for key_group in ("main", "c", "buttons", "shoulder"):
                for key in _FEATURE_CONTROLLER_KEYS[key_group]:
                    full = f"{prefix}{key}"
                    if full in name2idx:
                        self.controller_idxs.append(name2idx[full])

        excluded = {
            self.stage_idx,
            self.ego_char_idx,
            self.opp_char_idx,
            self.ego_action_idx,
            self.opp_action_idx,
            *self.controller_idxs,
        }
        self.gamestate_idxs = [i for i in range(len(self.feat_names)) if i not in excluded]

        targ2idx = {n: i for i, n in enumerate(self.targ_names)}

        def _ti(name: str) -> int:
            if name not in targ2idx:
                raise KeyError(f"Target '{name}' not present in checkpoint schema.")
            return targ2idx[name]

        self.y_main = (_ti("p1_main_stick_x"), _ti("p1_main_stick_y"))
        self.y_c = (_ti("p1_c_stick_x"), _ti("p1_c_stick_y"))
        self.y_buttons = [_ti(name) for name in _BUTTON_TARGETS]
        self.y_shoulder = targ2idx.get("p1_shoulder_analog")


def _resolve_device(preferred: Optional[str]) -> torch.device:
    if preferred is None or preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preferred)


class GPTInferenceEngine:
    """Online inference helper around the GPT controller model."""

    def __init__(
            self,
            checkpoint_path: str | Path,
            *,
            device: Optional[str] = None,
            data_root: Optional[str | Path] = None,
            button_threshold: float | Sequence[float] = _DEFAULT_BUTTON_THRESHOLD,
            shoulder_centers: Optional[Sequence[float]] = None,
            history: Optional[int] = None,
            warmup_frames: int = 128,
    ) -> None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        train_cfg = ckpt.get("config", {})

        data_root = Path(data_root or train_cfg.get("data_root", "dataset_FOX_vs_FOX"))
        meta_path = data_root / "meta.json"
        if meta_path.exists():
            with meta_path.open("r") as f:
                meta = json.load(f)

            feature_names = meta["schema"]["features"]
            target_names = meta["schema"]["targets"]
            self.seq_len = history or int(meta.get("seq_len", 256))
        else:
            feature_names = list(_DEFAULT_FEATURE_NAMES)
            target_names = list(_DEFAULT_TARGET_NAMES)
            self.seq_len = history or 256
        self.device = _resolve_device(device)
        self.shoulder_centers = tuple(
            shoulder_centers or train_cfg.get("shoulder_centers", _DEFAULT_SHOULDER_CENTERS)
        )
        self.warmup_frames = warmup_frames

        self.model = GPTv7().to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        self.feature_names = feature_names
        self.colmap = InferenceColumnMap(feature_names, target_names)
        self.feature_dim = len(feature_names)
        self.buffer: deque[torch.Tensor] = deque(maxlen=self.seq_len)

        if isinstance(button_threshold, Iterable) and not isinstance(button_threshold, (str, bytes)):
            thresholds = list(button_threshold)
            if len(thresholds) != len(_BUTTON_TARGETS):
                raise ValueError(
                    f"Expected {len(_BUTTON_TARGETS)} button thresholds, got {len(thresholds)}"
                )
            self.button_thresholds = torch.tensor(thresholds, dtype=torch.float32)
        else:
            self.button_thresholds = torch.full(
                (len(_BUTTON_TARGETS),), float(button_threshold), dtype=torch.float32
            )

        self.button_thresholds = torch.clamp(self.button_thresholds, 0.0, 1.0)
        self.button_thresholds = self.button_thresholds.to(torch.float32)

    # ------------------------------------------------------------------
    # Frame preparation utilities
    # ------------------------------------------------------------------
    def _frame_to_tensor(self, raw_inputs: Dict[str, float]) -> torch.Tensor:
        frame = torch.zeros(self.feature_dim, dtype=torch.float32)
        for idx, name in enumerate(self.feature_names):
            frame[idx] = _safe_float(raw_inputs.get(name, 0.0))
        return frame

    def _stack_frames(self) -> Optional[torch.Tensor]:
        if not self.buffer:
            return None
        frames = list(self.buffer)
        stacked = torch.stack(frames, dim=0)
        return stacked.unsqueeze(0).to(self.device)

    def _build_inputs(self, batch_X: torch.Tensor) -> TensorDict:
        return build_inputs_for_gptv7(batch_X, self.colmap)

    # ------------------------------------------------------------------
    # Decoding helpers
    # ------------------------------------------------------------------
    def _decode_stick(self, logits: torch.Tensor, palette: np.ndarray) -> np.ndarray:
        # print(logits)
        idx = torch.argmax(logits, dim=-1).detach().cpu().numpy().astype(np.int32)
        xy01 = model_to_dolphin01(idx, palette11=palette)
        return xy01.reshape(-1)

    def _decode_buttons(self, probs: torch.Tensor) -> List[bool]:
        probs_cpu = probs.detach().cpu()
        thresholds = self.button_thresholds.to(probs_cpu.device)
        return (probs_cpu >= thresholds).tolist()

    def _decode_outputs(self, outputs: TensorDict) -> ControllerState:
        main_logits = outputs["main_stick"][0, -1]
        c_logits = outputs["c_stick"][0, -1]
        button_probs = outputs.get("buttons_probs")
        if button_probs is None:
            button_probs = torch.sigmoid(outputs["buttons"])
        button_probs = button_probs[0, -1]

        shoulder_logits = outputs.get("shoulder")

        main_xy = self._decode_stick(main_logits, np.asarray(FOX_STICK_64, dtype=np.float32))
        c_xy = self._decode_stick(c_logits, np.asarray(C_STICK_XY_CLUSTER_CENTERS_V0_1, dtype=np.float32))
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def prepare_inputs(self, raw_inputs: Dict[str, float]) -> Optional[TensorDict]:
        frame = self._frame_to_tensor(raw_inputs)
        self.buffer.append(frame)
        batch = self._stack_frames()
        if batch is None:
            return None
        return self._build_inputs(batch)

    def predict_from_raw(self, raw_inputs: Dict[str, float]) -> ControllerState:
        inputs_td = self.prepare_inputs(raw_inputs)
        if inputs_td is None or len(self.buffer) < self.warmup_frames:
            print(inputs_td, len(self.buffer))
            return ControllerState.neutral()
        with torch.inference_mode():
            outputs = self.model(inputs_td)
        return self._decode_outputs(outputs)

    def act(self, gamestate: GameState, bot_port: int, opp_port: int) -> ControllerState:
        raw_inputs = collect_raw_inputs_from_gamestate(gamestate, bot_port, opp_port)
        return self.predict_from_raw(raw_inputs)


_ACTIVE_ENGINE: Optional[GPTInferenceEngine] = None


def set_active_engine(engine: Optional[GPTInferenceEngine]) -> None:
    global _ACTIVE_ENGINE
    _ACTIVE_ENGINE = engine


def get_active_engine() -> GPTInferenceEngine:
    if _ACTIVE_ENGINE is None:
        raise RuntimeError("Inference engine has not been initialised. Call set_active_engine().")
    return _ACTIVE_ENGINE


def transform_raw_inputs_for_model(raw_gamestate: Dict[str, float]) -> Optional[TensorDict]:
    engine = get_active_engine()
    return engine.prepare_inputs(raw_gamestate)


def transform_raw_outputs_for_game(raw_model_outputs: TensorDict) -> ControllerState:
    engine = get_active_engine()
    return engine._decode_outputs(raw_model_outputs)


def apply_model_outputs_to_game(controller: Controller, model_outputs: ControllerState) -> None:
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

    controller.tilt_analog(enums.Button.BUTTON_MAIN, model_outputs.main_stick_x, model_outputs.main_stick_y)
    controller.tilt_analog(enums.Button.BUTTON_C, model_outputs.c_stick_x, model_outputs.c_stick_y)

    controller.press_shoulder(enums.Button.BUTTON_R, 0.0)
    controller.press_shoulder(enums.Button.BUTTON_L, model_outputs.shoulder_analog)
