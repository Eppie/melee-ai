"""Utilities for converting game state to model inputs and applying model outputs to
controller actions."""

from __future__ import annotations

from collections import deque
from typing import Callable, Set
from typing import Deque
from typing import Sequence, Dict, Tuple, List, Optional

import numpy as np
import torch

from RecurrentNet import RecurrentNet
from config import FEATURE_COLUMNS, TARGET_COLUMNS, BCE_INDICES, SEQUENCE_LENGTH
from libmelee.melee import enums
from libmelee.melee.controller import Controller
from libmelee.melee.gamestate import GameState, PlayerState
from train import RECURRENT_CONFIG


def gamestate_to_input(gs: GameState, *, p1_port: int = 1, p2_port: int = 2) -> np.ndarray:
    """Convert the current :class:`~melee.gamestate.GameState` into a feature vector.

    Parameters
    ----------
    gs:
        Current game state object provided by ``libmelee``.
    p1_port:
        Controller port of the bot (the player we are predicting actions for).
    p2_port:
        Controller port of the opponent.

    Returns
    -------
    np.ndarray
        A 1-D float32 array with features ordered as ``FEATURE_COLUMNS``.
    """
    p1 = gs.players.get(p1_port)
    p2 = gs.players.get(p2_port)
    if p1 is None or p2 is None:
        raise ValueError("Both player ports must be present in the game state")

    def _flags(pl: PlayerState) -> list[float]:
        return [
            float(pl.invulnerable),
            float(pl.is_fastfalling),
            float(pl.is_defender_in_hitlag),
            float(pl.is_in_hitlag),
            float(pl.is_holding_character),
            float(pl.is_shield_active),
            float(pl.is_in_hitstun),
            float(pl.is_touching_shield),
            float(pl.is_powershield),
            float(pl.is_dead),
            float(pl.is_offscreen),
        ]

    p1_flags = _flags(p1)
    p2_flags = _flags(p2)

    p1_xy = float(
        bool(p1.controller_state.button[enums.Button.BUTTON_X])
        or bool(p1.controller_state.button[enums.Button.BUTTON_Y])
    )

    p1_lr = float(
        bool(p1.controller_state.button[enums.Button.BUTTON_L])
        or bool(p1.controller_state.button[enums.Button.BUTTON_R])
    )

    p2_xy = float(
        bool(p2.controller_state.button[enums.Button.BUTTON_X])
        or bool(p2.controller_state.button[enums.Button.BUTTON_Y])
    )

    p2_lr = float(
        bool(p2.controller_state.button[enums.Button.BUTTON_L])
        or bool(p2.controller_state.button[enums.Button.BUTTON_R])
    )

    feats = [
        float(p1.controller_state.button[enums.Button.BUTTON_A]),
        float(p1.controller_state.button[enums.Button.BUTTON_B]),
        float(p1.controller_state.button[enums.Button.BUTTON_Z]),
        p1_xy, p1_lr,
        float(p1.controller_state.main_stick[0]),
        float(p1.controller_state.main_stick[1]),
        float(p1.controller_state.c_stick[0]),
        float(p1.controller_state.c_stick[1]),
        # *p1_flags,
        float(p1.position.x),
        float(p1.position.y),
        # float(p1.action.value if hasattr(p1.action, "value") else p1.action),
        float(p1.percent),
        # float(p1.stock),
        float(p1.character.value if hasattr(p1.character, "value") else p1.character),
        # *p2_flags,
        float(p2.position.x),
        float(p2.position.y),
        # float(p2.action.value if hasattr(p2.action, "value") else p2.action),
        float(p2.percent),
        # float(p2.stock),
        float(p2.character.value if hasattr(p2.character, "value") else p2.character),
        # float(p2.controller_state.l_shoulder),
        # float(p2.controller_state.r_shoulder),
        float(p2.controller_state.button[enums.Button.BUTTON_A]),
        float(p2.controller_state.button[enums.Button.BUTTON_B]),
        float(p2.controller_state.button[enums.Button.BUTTON_Z]),
        p2_xy, p2_lr,
        float(p2.controller_state.main_stick[0]),
        float(p2.controller_state.main_stick[1]),
        float(p2.controller_state.c_stick[0]),
        float(p2.controller_state.c_stick[1]),
    ]

    return np.asarray(feats, dtype=np.float32)


_DIGITAL_TARGETS: Set[str] = {
    'p1_btn_a', 'p1_btn_b', 'p1_btn_z', 'p1_btn_xy', 'p1_btn_lr'
}
_BUTTON_SHORT: Dict[str, str] = {
    'p1_btn_a': 'A',
    'p1_btn_b': 'B',
    'p1_btn_z': 'Z',
    'p1_btn_xy': 'X/Y',
    'p1_btn_lr': 'L/R',
}
_ANALOG_LABEL: Dict[str, str] = {
    'p1_btn_l_analog': 'L_SHOULDER',
    'p1_btn_r_analog': 'R_SHOULDER',
    'p1_pre_joystick_x': 'MAIN_X',
    'p1_pre_joystick_y': 'MAIN_Y',
    'p1_pre_cstick_x': 'C_X',
    'p1_pre_cstick_y': 'C_Y',
}


def debug_dump_model_inputs(
        inputs: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
        *,
        writer: Optional[Callable[[str], None]] = None,
        show_window_stats: bool = False,
        which_frame: str = "last",  # "last" | "first" | "index"
        frame_index: Optional[int] = None,  # used when which_frame == "index"
) -> None:
    """
    Pretty-print the features that are fed to the model, aligned with FEATURE_COLUMNS.

    Accepts either:
      • A single frame vector of shape (F,), or
      • A window array/sequence of shape (T, F).

    Args:
        inputs: Single frame (F,) or window (T, F) of float features.
        writer: Optional sink for lines (defaults to print).
        show_window_stats: If True and inputs is a window (T, F), print mean/min/max over T.
        which_frame: Which frame to show from the window: "last", "first", or "index".
        frame_index: If which_frame == "index", the explicit index (0-based) into T.
    """
    sink: Callable[[str], None] = print if writer is None else writer

    arr = np.asarray(inputs, dtype=np.float32)
    if arr.ndim == 1:
        # Single frame (F,)
        if arr.shape[0] != len(FEATURE_COLUMNS):
            sink(f"[debug] Feature length mismatch: got {arr.shape[0]}, expected {len(FEATURE_COLUMNS)}")
            return
        _dump_feature_vector(arr, sink)
        return

    if arr.ndim != 2:
        sink(f"[debug] Expected 1-D or 2-D inputs, got shape {arr.shape}")
        return

    # Window (T, F)
    T, F = arr.shape
    if F != len(FEATURE_COLUMNS):
        sink(f"[debug] Feature width mismatch: got {F}, expected {len(FEATURE_COLUMNS)}")
        return

    if which_frame == "first":
        idx = 0
    elif which_frame == "last":
        idx = T - 1
    elif which_frame == "index":
        if frame_index is None or not (0 <= frame_index < T):
            sink(f"[debug] Invalid frame_index={frame_index}, window length T={T}")
            return
        idx = frame_index
    else:
        sink(f"[debug] Unknown which_frame='{which_frame}' (use 'first' | 'last' | 'index').")
        return

    sink(f"[debug] Window shape: T={T}, F={F}. Showing frame {idx} {'(last)' if idx == T - 1 else ''}")
    _dump_feature_vector(arr[idx], sink)

    if show_window_stats:
        _dump_window_stats(arr, sink)


def _dump_feature_vector(vec: np.ndarray, sink: Callable[[str], None]) -> None:
    header = f"{'Idx':>3}  {'Feature':<24} {'Value':>10}"
    sink(header)
    sink("-" * len(header))
    for i, name in enumerate(FEATURE_COLUMNS):
        val = float(vec[i])
        sink(f"{i:>3}  {name:<24} {val:>10.4f}")


def _dump_window_stats(win: np.ndarray, sink: Callable[[str], None]) -> None:
    # win: (T, F)
    means = win.mean(axis=0)
    mins = win.min(axis=0)
    maxs = win.max(axis=0)

    sink("\n[debug] Per-feature stats across window")
    header = f"{'Idx':>3}  {'Feature':<24} {'Mean':>10} {'Min':>10} {'Max':>10}"
    sink(header)
    sink("-" * len(header))
    for i, name in enumerate(FEATURE_COLUMNS):
        sink(f"{i:>3}  {name:<24} {means[i]:>10.4f} {mins[i]:>10.4f} {maxs[i]:>10.4f}")


def debug_dump_model_outputs(
        outputs_before: Sequence[float],
        outputs_after: Sequence[float],
        *,
        threshold: float = 0.5,
        writer: Optional[Callable[[str], None]] = None,
) -> None:
    """
    Print a table mapping each TARGET_COLUMNS entry to its value before/after processing.

    Args:
        outputs_before: Raw model outputs (e.g., logits for digital, unclamped floats for analog).
        outputs_after:  Post-processed outputs (e.g., sigmoid-applied probabilities and clamped analogs).
        threshold:      Threshold used to decide digital "press".
        writer:         Optional sink for lines (defaults to print). Useful to route into a logger.
    """
    sink: Callable[[str], None] = print if writer is None else writer

    n = len(TARGET_COLUMNS)
    if len(outputs_before) != n or len(outputs_after) != n:
        sink(f"[debug] Mismatched lengths: before={len(outputs_before)}, after={len(outputs_after)}, expected={n}")
        return

    header = f"{'Idx':>3}  {'Target':<20} {'Kind':<9} {'Raw':>9} {'After':>9}  {'Decision':<8}"
    sink(header)
    sink("-" * len(header))

    for i, name in enumerate(TARGET_COLUMNS):
        raw_val = float(outputs_before[i])
        post_val = float(outputs_after[i])

        if name in _DIGITAL_TARGETS:
            kind = "button"
            short = _BUTTON_SHORT.get(name, name)
            # decision = "PRESS" if post_val >= THRESH_ON_STR[short] else "release"
            decision = "PRESS" if post_val >= 0.5 else "release"
            label = f"{name}({short})"
        else:
            kind = "analog"
            decision = ""  # not applicable
            label = f"{name}({_ANALOG_LABEL.get(name, 'ANALOG')})"

        sink(f"{i:>3}  {label:<20} {kind:<9} {raw_val:>9.4f} {post_val:>9.4f}  {decision:<8}")


BTN_ORDER_STR: List[str] = ['A', 'B', 'Z', 'X/Y', 'L/R']
BTN_ORDER_ENUM: List[enums.Button] = [
    enums.Button.BUTTON_A,
    enums.Button.BUTTON_B,
    enums.Button.BUTTON_Z,
    enums.Button.BUTTON_X,
    enums.Button.BUTTON_L,
]


def thresholds_from_pos_weight(
        pos_weight: Sequence[float],
        *,
        alpha: Optional[float] = None,  # if you later introduce per-label alpha, pass a scalar for a warm start
) -> Tuple[Dict[enums.Button, float], Dict[str, float]]:
    """
    Convert class-imbalance weights into per-button probability thresholds.

    If alpha is None:
        tau = 1 / (w + 1)
    Else (scalar alpha):
        tau = (1 - alpha) / (w*alpha + (1 - alpha))   # warm start only if focal is on

    Returns:
        (THRESH_ON, THRESH_ON_STR)
    """
    if len(pos_weight) != len(BTN_ORDER_ENUM):
        raise ValueError(f"Expected {len(BTN_ORDER_ENUM)} weights, got {len(pos_weight)}")

    w: List[float] = [float(x) for x in pos_weight]

    if alpha is None:
        tau_list: List[float] = [1.0 / (wi + 1.0) if wi > 0.0 else 0.5 for wi in w]
    else:
        a = float(alpha)
        wneg = 1.0 - a
        tau_list = [(wneg / (wi * a + wneg)) if (wi * a + wneg) > 0.0 else 0.5 for wi in w]

    # Optionally round for readability; remove round(...) if you want full precision
    tau_list = [round(t, 6) for t in tau_list]

    thresh_on: Dict[enums.Button, float] = {btn: t for btn, t in zip(BTN_ORDER_ENUM, tau_list)}
    thresh_on_str: Dict[str, float] = {name: t for name, t in zip(BTN_ORDER_STR, tau_list)}
    return thresh_on, thresh_on_str


# --- Usage ---
pos_weight = [24.7068, 31.4086, 202.1463, 8.7873, 7.3041]
THRESH_ON, THRESH_ON_STR = thresholds_from_pos_weight(pos_weight)


def apply_model_output(controller: Controller, outputs: Sequence[float]) -> None:
    """Map model output values to controller actions.

    Parameters
    ----------
    controller:
        ``libmelee`` :class:`~melee.controller.Controller` instance to send inputs to.
    outputs:
        Sequence of predictions in the order given by ``TARGET_COLUMNS``.
    """
    if len(outputs) != len(TARGET_COLUMNS):
        raise ValueError(f"Expected {len(TARGET_COLUMNS)} outputs, got {len(outputs)}")

    # Digital button predictions – press if above threshold, otherwise release.
    button_probs = {
        enums.Button.BUTTON_A: float(outputs[0]),  # p1_btn_a
        enums.Button.BUTTON_B: float(outputs[1]),  # p1_btn_b
        enums.Button.BUTTON_Z: float(outputs[2]),  # p1_btn_z
        enums.Button.BUTTON_X: float(outputs[3]),  # p1_btn_xy
        enums.Button.BUTTON_L: float(outputs[4]),  # p1_btn_lr
    }

    for btn, p in button_probs.items():
        # if p >= THRESH_ON[btn]:
        if p >= 0.5:
            controller.press_button(btn)
        else:
            controller.release_button(btn)

    # Analog sticks
    controller.tilt_analog(enums.Button.BUTTON_MAIN, float(outputs[5]), float(outputs[6]))
    controller.tilt_analog(enums.Button.BUTTON_C, float(outputs[7]), float(outputs[8]))


CHECKPOINT_PATH: str = "/Users/eppie/melee-ai/trained_mlp_model.pt"

# Which target columns are digital vs analog (same as in training)

# For clipping analog ranges (should match your data conventions)
IDX_JS_X: int = TARGET_COLUMNS.index('p1_pre_joystick_x')  # [0, 1]
IDX_JS_Y: int = TARGET_COLUMNS.index('p1_pre_joystick_y')  # [0, 1]
IDX_CS_X: int = TARGET_COLUMNS.index('p1_pre_cstick_x')  # [0, 1]
IDX_CS_Y: int = TARGET_COLUMNS.index('p1_pre_cstick_y')  # [0, 1]


class InferenceEngine:
    """
    Maintains a sliding window of SEQUENCE_LENGTH frames, runs the model,
    and returns post-processed outputs in TARGET_COLUMNS order.
    """

    def __init__(self, checkpoint_path: str, threshold: float = 0.5) -> None:
        self.threshold = threshold

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda"
        if torch.cuda.is_available() else "cpu")

        self.model = RecurrentNet(RECURRENT_CONFIG, feature_dim=len(FEATURE_COLUMNS)).to(self.device)
        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        self.buffer: Deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)

    def push_frame(self, feature_vec: np.ndarray) -> None:
        """Append one (F,) feature vector for the current frame."""
        assert feature_vec.dtype == np.float32 and feature_vec.ndim == 1 and feature_vec.shape[0] == len(
            FEATURE_COLUMNS), f"{feature_vec.dtype}, {feature_vec.ndim}, {feature_vec.shape}, {len(FEATURE_COLUMNS)}"
        self.buffer.append(feature_vec)

    def ready(self) -> bool:
        """True if we have sequence_len frames buffered."""
        return len(self.buffer) == SEQUENCE_LENGTH

    @torch.no_grad()
    def infer(self) -> np.ndarray:
        """
        Run a forward pass on the most recent window.
        Returns a (C,) numpy array aligned to TARGET_COLUMNS with:
            - digital button indices passed through sigmoid (probabilities)
            - analog outputs clipped to expected ranges
        """
        if not self.ready():
            raise RuntimeError("Called infer() before buffer is full")

        seq = np.stack(self.buffer, axis=0)  # (T, F), float32
        x = torch.from_numpy(seq).unsqueeze(0).to(self.device)  # (1, T, F)

        logits = self.model(x).squeeze(0)  # (C,)
        preds_before = logits.clone()
        preds = logits.clone()

        # Sigmoid on digital button positions
        preds[BCE_INDICES] = torch.sigmoid(preds[BCE_INDICES])

        for idx in (IDX_JS_X, IDX_JS_Y, IDX_CS_X, IDX_CS_Y):
            preds[idx] = preds[idx].clamp_(0.0, 1.0)
        preds = preds.detach().cpu().numpy()
        debug_dump_model_outputs(preds_before, preds)
        return preds
