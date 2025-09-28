from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, Optional, Sequence

import numpy as np
import torch

from config import TARGET_COLUMNS, SEQUENCE_LENGTH
from libmelee.melee import Stage, stages, PlayerState, GameState, enums
from transformer import load_dt, _to_pm1

# ---- Mapping helpers (TARGET_COLUMNS name -> index) ----
_TIDX: Dict[str, int] = {name: i for i, name in enumerate(TARGET_COLUMNS)}

def _idx(name: str) -> int:
    if name not in _TIDX:
        raise KeyError(f"TARGET_COLUMNS missing '{name}'")
    return _TIDX[name]


@dataclass
class StateAdapter:
    """
    Adapts your per-frame FEATURE_COLUMNS vector (F,) into the DT state vector (D_state,).

    By default we pick a subset by indices. Replace with a custom callable if you trained
    the DT on a different schema.
    """
    indices: Optional[Sequence[int]] = None
    fn: Optional[Callable[[np.ndarray], np.ndarray]] = None  # maps (F,) -> (D_state,)

    def __call__(self, feat: np.ndarray) -> np.ndarray:
        if self.fn is not None:
            out = self.fn(feat)
            if out.ndim != 1:
                raise ValueError(f"state_adapter.fn must return 1-D, got {out.shape}")
            return out.astype(np.float32, copy=False)
        if self.indices is None:
            # Identity fallback (only works if F == D_state)
            return feat.astype(np.float32, copy=False)
        return feat[np.asarray(self.indices, dtype=np.int64)].astype(np.float32, copy=False)


def _safe_bool(x: object) -> float:
    return 1.0 if bool(x) else 0.0

def _safe_float(x: object) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0

def _facing_to_pm1(face_right: bool) -> float:
    return 1.0 if face_right else -1.0

def _blast_zones(stage: Stage) -> tuple[float, float, float, float]:
    """Return (left, right, bottom, top) to match training code."""
    left, right, upper, lower = stages.BLASTZONES[stage]
    return float(left), float(right), float(lower), float(upper)

def _distance_xy(x0: float, y0: float, x1: float, y1: float) -> float:
    dx, dy = (x1 - x0), (y1 - y0)
    return float((dx * dx + dy * dy) ** 0.5)

def _distance_to_blastzones_xy(x: float, y: float, stage: Stage) -> float:
    left, right, bottom, top = _blast_zones(stage)
    d_left = x - left
    d_right = right - x
    d_bottom = y - bottom
    d_top = top - y
    return float(min(d_left, d_right, d_bottom, d_top))

def _training_state_from_gs(gs: GameState, self_port: int, opp_port: int) -> np.ndarray:
    """
    Recreate the exact 16(self)+8(opp)+1(global)=25-dim vector you trained on.
    """
    me: PlayerState = gs.players[self_port]
    opp: PlayerState = gs.players[opp_port]
    stage_enum: Stage = Stage(int(gs.stage.value))  # gs.stage is an enum already; int() is safe

    # --- self subset (16 dims) ---
    me_percent = _safe_float(me.percent)
    # me_facing_pm1 = _facing_to_pm1(bool(me.facing))
    me_x = _safe_float(me.position.x)
    me_y = _safe_float(me.position.y)
    me_action = _safe_float(me.action.value if hasattr(me.action, "value") else me.action)
    me_invuln = _safe_bool(me.invulnerable)
    # me_char = _safe_float(me.character.value if hasattr(me.character, "value") else me.character)
    me_jumps = _safe_float(me.jumps_left)
    me_shield = _safe_float(me.shield_strength)
    me_on_ground = _safe_bool(me.on_ground)
    me_action_frame = _safe_float(me.action_frame)
    me_hitstun_left = _safe_float(getattr(me, "hitstun_frames_left", 0))
    me_off_stage = _safe_bool(getattr(me, "off_stage", False))

    # computed_features (we recompute the three that training used)
    dist_me_opp = _distance_xy(me_x, me_y, _safe_float(opp.position.x), _safe_float(opp.position.y))
    facing_opp = 1.0 if ((opp.position.x - me_x > 0 and me.facing) or (opp.position.x - me_x < 0 and not me.facing)) else 0.0
    dist_to_blast = _distance_to_blastzones_xy(me_x, me_y, stage_enum)

    s_self: list[float] = [
        me_percent,
        # me_facing_pm1,
        me_x,
        me_y,
        me_action,
        me_invuln,
        # me_char,
        me_jumps,
        me_shield,
        me_on_ground,
        me_action_frame,
        me_hitstun_left,
        me_off_stage,
        dist_me_opp,
        facing_opp,
        dist_to_blast,
        _safe_float(me.speed_air_x_self),
        _safe_float(me.speed_y_self),
        _safe_float(me.speed_x_attack),
        _safe_float(me.speed_y_attack),
        _safe_float(me.speed_ground_x_self),
    ]
    print(f"Distance me: {dist_me_opp}")

    # --- opponent subset (8 dims) ---
    opp_x = _safe_float(opp.position.x)
    opp_y = _safe_float(opp.position.y)
    opp_percent = _safe_float(opp.percent)
    opp_action = _safe_float(opp.action.value if hasattr(opp.action, "value") else opp.action)
    opp_on_ground = _safe_bool(opp.on_ground)
    opp_off_stage = _safe_bool(getattr(opp, "off_stage", False))
    opp_dist_me = _distance_xy(opp_x, opp_y, me_x, me_y)
    opp_dist_blast = _distance_to_blastzones_xy(opp_x, opp_y, stage_enum)

    s_opp: list[float] = [
        opp_x,
        opp_y,
        opp_percent,
        opp_action,
        opp_on_ground,
        opp_off_stage,
        opp_dist_me,
        opp_dist_blast,
        _safe_float(opp.speed_air_x_self),
        _safe_float(opp.speed_y_self),
        _safe_float(opp.speed_x_attack),
        _safe_float(opp.speed_y_attack),
        _safe_float(opp.speed_ground_x_self),
    ]

    # --- global (1 dim) ---
    # s_global: list[float] = [float(gs.stage.value)]

    vec = np.asarray(s_self + s_opp, dtype=np.float32)
    # vec = np.asarray(s_self + s_opp + s_global, dtype=np.float32)
    return vec

class DTInferenceEngine:
    """
    DecisionTransformer-backed drop-in replacement.

    Methods:
      - push_frame(feat: np.ndarray)
      - ready() -> bool
      - infer() -> np.ndarray   # aligned to TARGET_COLUMNS (same as your old engine)
    """

    def __init__(
        self,
        bundle_path: str,
        *,
        state_adapter: Optional[StateAdapter] = None,
        threshold: float = 0.5,
        verbose_debug: bool = False,
    ) -> None:
        self.threshold = float(threshold)
        self.verbose_debug = verbose_debug

        # Device
        self.device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load model + cfg (+ optional norm stats)
        self.model, self.cfg, self.norm = load_dt(bundle_path, device=str(self.device))
        self.model.eval()

        # Adapter from FEATURE_COLUMNS -> DT state vector
        self.state_adapter = state_adapter or StateAdapter()

        # Context length = what DT saw during training (cap by your SEQUENCE_LENGTH)
        self.ctx_len = min(int(SEQUENCE_LENGTH), int(self.cfg.seq_len))

        # Buffers
        self.buffer: Deque[np.ndarray] = deque(maxlen=self.ctx_len)

        # Previous action history (what DT conditions on)
        n_cont = self.model.aspace.n_cont
        n_bin = self.model.aspace.n_bin
        self.acont_hist: Deque[np.ndarray] = deque(maxlen=self.ctx_len)
        self.abin_hist: Deque[np.ndarray] = deque(maxlen=self.ctx_len)

        # Initialize with zeros so first few calls have something sensible
        self._zero_acont = np.zeros((n_cont,), dtype=np.float32)
        self._zero_abin = np.zeros((n_bin,), dtype=np.float32)

    def push_gamestate(self, gs: GameState, bot_port: int, opp_port: int) -> None:
        """
        Build the training-compatible DT state from the live GameState and push it.
        Also seeds prev-action histories from the bot's *current* controller state.
        """
        # 1) Build DT state (matches training exactly)
        state_vec = _training_state_from_gs(gs, bot_port, opp_port)
        if state_vec.dtype != np.float32 or state_vec.ndim != 1:
            raise ValueError(f"state vec must be (D_state,) float32, got {state_vec.dtype}, {state_vec.shape}")

        # If you saved normalization stats, DT will apply them in infer()

        # 2) Append to the state buffer (we treat buffer entries as already-in-DT-space)
        self.buffer.append(state_vec)

        # 3) Seed prev action from *live* controller state for this frame
        #    (helps cold-start instead of zeros)
        btn = gs.players[bot_port].controller_state.button
        acont = np.zeros((self.model.aspace.n_cont,), dtype=np.float32)
        abin = np.zeros((self.model.aspace.n_bin,), dtype=np.float32)

        # Continuous head convention: [main_x, main_y, c_x, c_y, (shoulder?) ...]
        if self.model.aspace.n_cont >= 4:
            acont[0] = _to_pm1(float(gs.players[bot_port].controller_state.main_stick[0]))
            acont[1] = _to_pm1(float(gs.players[bot_port].controller_state.main_stick[1]))
            acont[2] = _to_pm1(float(gs.players[bot_port].controller_state.c_stick[0]))
            acont[3] = _to_pm1(float(gs.players[bot_port].controller_state.c_stick[1]))
            acont[4] = _to_pm1(float(gs.players[bot_port].controller_state.l_shoulder))

        # Binary head convention: [A, B, X, Y, Z, L, R, D_UP] (adjust if you trained differently)
        def _pressed(b: enums.Button) -> float:
            return 1.0 if bool(btn[b]) else 0.0

        if self.model.aspace.n_bin >= 1: abin[0] = _pressed(enums.Button.BUTTON_A)
        if self.model.aspace.n_bin >= 2: abin[1] = _pressed(enums.Button.BUTTON_B)
        if self.model.aspace.n_bin >= 3: abin[2] = _pressed(enums.Button.BUTTON_X)
        if self.model.aspace.n_bin >= 4: abin[3] = _pressed(enums.Button.BUTTON_Y)
        if self.model.aspace.n_bin >= 5: abin[4] = _pressed(enums.Button.BUTTON_Z)
        if self.model.aspace.n_bin >= 6: abin[5] = _pressed(enums.Button.BUTTON_L)
        if self.model.aspace.n_bin >= 7: abin[6] = _pressed(enums.Button.BUTTON_R)

        # 4) Keep history sizes in sync with the state buffer
        self.acont_hist.append(acont)
        self.abin_hist.append(abin)
        while len(self.acont_hist) > len(self.buffer):
            self.acont_hist.popleft()
        while len(self.abin_hist) > len(self.buffer):
            self.abin_hist.popleft()

        # Cap to context length
        while len(self.buffer) > self.ctx_len:
            self.buffer.popleft()
            if self.acont_hist: self.acont_hist.popleft()
            if self.abin_hist:  self.abin_hist.popleft()


    def ready(self) -> bool:
        return len(self.buffer) == self.ctx_len

    @torch.no_grad()
    def infer(self) -> np.ndarray:
        """
        Produce a (C,) prediction aligned with TARGET_COLUMNS, ready for apply_model_output.
        """
        if not self.ready():
            raise RuntimeError("Called infer() before buffer is full")

        # --- Build state sequence (L, D_state)
        window = np.stack(self.buffer, axis=0)  # (L, F)
        states_np = np.stack([self.state_adapter(f) for f in window], axis=0)  # (L, D_state)

        # Optional normalization if you saved stats
        if "state_mean" in self.norm and "state_std" in self.norm:
            mean = self.norm["state_mean"].cpu().numpy()
            std = self.norm["state_std"].cpu().numpy()
            states_np = (states_np - mean) / (std + 1e-6)

        # --- Prev action sequences (L, n_cont / n_bin)
        acont_np = np.stack(self.acont_hist, axis=0)
        abin_np = np.stack(self.abin_hist, axis=0)

        # --- RTG (constant 0 for pure imitation) -> (L, 1)
        L = states_np.shape[0]
        rtg_np = np.zeros((L, 1), dtype=np.float32)

        # -> tensors (1, L, *)
        states = torch.from_numpy(states_np).unsqueeze(0).to(self.device)
        acont_in = torch.from_numpy(acont_np).unsqueeze(0).to(self.device)
        abin_in = torch.from_numpy(abin_np).unsqueeze(0).to(self.device)
        rtg = torch.from_numpy(rtg_np).unsqueeze(0).to(self.device)
        attn_mask = torch.ones(1, L, dtype=torch.bool, device=self.device)

        # --- DT forward
        out = self.model(rtg, states, acont_in, abin_in, attn_mask=attn_mask)
        # Take the last step
        a_cont = out["a_cont"][:, -1].squeeze(0)           # (n_cont,)
        a_bin_logits = out["a_bin_logits"][:, -1].squeeze(0)  # (n_bin,)

        # Sigmoid for binary probs
        a_bin_prob = torch.sigmoid(a_bin_logits)

        # Map DT continuous from [-1,1] -> [0,1] if model uses tanh head
        a_cont_01 = (a_cont + 1.0) * 0.5
        a_cont_01 = a_cont_01.clamp_(0.0, 1.0)
        # a_cont_01 = a_cont

        # --- Adapt DT outputs to your TARGET_COLUMNS layout ---
        # Expecting DT heads in this order (as per earlier setup):
        #   continuous: [main_x, main_y, c_x, c_y, (shoulder? optional at tail)]
        #   binary:     [A, B, X, Y, Z, L, R, D_UP]
        # If your trained aspace differs, adjust the slices below.
        n_cont = self.model.aspace.n_cont
        n_bin = self.model.aspace.n_bin
        if n_cont < 4 or n_bin < 5:
            raise RuntimeError(f"DT action heads too small: n_cont={n_cont}, n_bin={n_bin}")

        main_x = float(a_cont_01[0].item())
        main_y = float(a_cont_01[1].item())
        c_x    = float(a_cont_01[2].item())
        c_y    = float(a_cont_01[3].item())

        pA = float(a_bin_prob[0].item()) if n_bin >= 1 else 0.0
        pB = float(a_bin_prob[1].item()) if n_bin >= 2 else 0.0
        pX = float(a_bin_prob[2].item()) if n_bin >= 3 else 0.0
        pY = float(a_bin_prob[3].item()) if n_bin >= 4 else 0.0
        pZ = float(a_bin_prob[4].item()) if n_bin >= 5 else 0.0
        pL = float(a_bin_prob[5].item()) if n_bin >= 6 else 0.0
        pR = float(a_bin_prob[6].item()) if n_bin >= 7 else 0.0
        # pDUP = float(a_bin_prob[7].item()) if n_bin >= 8 else 0.0  # unused here

        pXY = max(pX, pY)
        pLR = max(pL, pR)

        # Fill outputs array aligned to TARGET_COLUMNS (same order/length as before)
        out_vec = np.zeros((len(TARGET_COLUMNS),), dtype=np.float32)
        out_vec[_idx("p1_btn_a")] = pA
        out_vec[_idx("p1_btn_b")] = pB
        out_vec[_idx("p1_btn_z")] = pZ
        out_vec[_idx("p1_btn_xy")] = pXY
        out_vec[_idx("p1_btn_lr")] = pLR
        out_vec[_idx("p1_pre_joystick_x")] = main_x
        out_vec[_idx("p1_pre_joystick_y")] = main_y
        out_vec[_idx("p1_pre_cstick_x")] = c_x
        out_vec[_idx("p1_pre_cstick_y")] = c_y

        # --- Update action history for next step (use hard decisions for buttons) ---
        # Buttons: replicate XY into both X and Y; LR into both L and R.
        abin_next = np.zeros((n_bin,), dtype=np.float32)
        abin_next[: n_bin] = 0.0
        abin_next[0] = 1.0 if pA >= self.threshold else 0.0
        abin_next[1] = 1.0 if pB >= self.threshold else 0.0
        if n_bin >= 3:
            abin_next[2] = 1.0 if pXY >= self.threshold else 0.0  # X
        if n_bin >= 4:
            abin_next[3] = 1.0 if pXY >= self.threshold else 0.0  # Y
        if n_bin >= 5:
            abin_next[4] = 1.0 if pZ >= self.threshold else 0.0
        if n_bin >= 6:
            abin_next[5] = 1.0 if pLR >= self.threshold else 0.0  # L
        if n_bin >= 7:
            abin_next[6] = 1.0 if pLR >= self.threshold else 0.0  # R
        # if n_bin >= 8: abin_next[7] = 0.0  # D_UP off

        acont_next = np.zeros((n_cont,), dtype=np.float32)
        # First 4 match your controller axes; keep shoulder (if present) at 0
        acont_next[:4] = np.array([main_x, main_y, c_x, c_y], dtype=np.float32)

        # Slide the histories (keep len == ctx_len)
        self.acont_hist.popleft()
        self.abin_hist.popleft()
        self.acont_hist.append(acont_next)
        self.abin_hist.append(abin_next)

        return out_vec