from __future__ import annotations

import json
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Callable
from typing import Optional

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from preprocess import NUM_USED_ACTIONS
from process_replays import process_one_replay
from schema import Row

BUTTON_LABELS: Tuple[str, ...] = ("A", "B", "Z", "JUMP", "SHIELD")
STICK_XY_CLUSTER_CENTERS_V2 = (
        np.array(
            [  # neutral
                [0.0, 0.0],
                # partial tilt
                [0.35, 0.0],
                [-0.35, 0.0],
                [0.0, 0.35],
                [0.0, -0.35],
                # tilt
                [0.675, 0.0],
                [-0.675, 0.0],
                [0.0, 0.675],
                [0.0, -0.675],
                # full press (dash / smash attack)
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
                [0.0, -1.0],
                # 17º / perfect wave/ledgedash
                [0.95, -0.3],
                [-0.95, -0.3],
                # 17º
                [0.95, 0.3],
                [-0.95, 0.3],
                # 30º / downward/up-angled f-smash
                [0.85, -0.5],
                [0.85, 0.5],
                [-0.85, -0.5],
                [-0.85, 0.5],
                # 45º + shield drops
                [0.7, -0.7],
                [-0.7, -0.7],
                [0.7, 0.7],
                [-0.7, 0.7],
                # [0.675, -0.675],
                # [-0.675, -0.675],
                # [0.675, 0.675],
                # [-0.675, 0.675],
                # up-/down-angled f-tilts
                [0.5, 0.5],
                [-0.5, 0.5],
                [0.5, -0.5],
                [-0.5, -0.5],
                # 60º
                [0.5, 0.85],
                [-0.5, 0.85],
                [0.5, -0.85],
                [-0.5, -0.85],
                # 72.5º
                [0.3, -0.95],
                [0.3, 0.95],
                [-0.3, -0.95],
                [-0.3, 0.95],
            ]
        )
        / 2
        + 0.5
)


def enum_to_int(x: object) -> int:
    # works for IntEnum or Enum
    try:
        return int(x)  # IntEnum
    except Exception:
        # type: ignore[attr-defined]
        return int(getattr(x, "value"))  # Enum.value


def one_hot(idx: Tensor, num_classes: int) -> Tensor:
    return torch.nn.functional.one_hot(idx.long(), num_classes=num_classes).float()


@dataclass
class RobustScaler:
    median: np.ndarray
    iqr: np.ndarray

    @staticmethod
    def fit(X: np.ndarray) -> "RobustScaler":
        q25 = np.percentile(X, 25, axis=0)
        q75 = np.percentile(X, 75, axis=0)
        med = np.percentile(X, 50, axis=0)
        iqr = q75 - q25
        iqr[iqr < 1e-6] = 1.0
        return RobustScaler(median=med.astype(np.float32), iqr=iqr.astype(np.float32))

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.median) / self.iqr).astype(np.float32)


# Signed-sqrt with symmetric winsorization
def winsor_signed_sqrt(x: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
    lo = np.percentile(x, p_lo)
    hi = np.percentile(x, p_hi)
    xw = np.clip(x, lo, hi)
    return np.sign(xw) * np.sqrt(np.abs(xw).astype(np.float32) + 1e-8)


@dataclass
class FeaturePack:
    X: np.ndarray  # shape (N, F)
    y_action: np.ndarray  # (N,) int in [0..K] (K = OTHER)
    y_stick: np.ndarray  # (N,) int in [0..8]
    y_buttons: np.ndarray  # (N,5) float32 0/1


@dataclass
class PreprocessArtifacts:
    scaler: RobustScaler
    vel_indices: Tuple[int, int, int]  # positions of vx_air, vy, vx_ground in feature columns
    feature_names: List[str]
    n_actions: int
    idx_to_action_name: List[str]


def make_action_label_fn(artifacts: PreprocessArtifacts) -> Callable[[int], str]:
    names = artifacts.idx_to_action_name

    def _label(i: int) -> str:
        if 0 <= i < len(names):
            return names[i]
        return str(i)

    return _label


def extract_arrays(rows: Sequence[Row]) -> Dict[str, np.ndarray]:
    action = np.asarray([r.p1_action for r in rows])
    msx = np.asarray([r.p1_main_stick_x for r in rows])
    msy = np.asarray([r.p1_main_stick_y for r in rows])
    # Buttons -> primitives
    A = np.asarray([r.p1_button_a for r in rows])
    B = np.asarray([r.p1_button_b for r in rows])
    Z = np.asarray([r.p1_button_z for r in rows])
    JUMP = np.asarray([r.p1_button_xy for r in rows])
    SHIELD = np.asarray([r.p1_button_lr for r in rows])

    facing = np.asarray([r.p1_facing for r in rows])  # 1=right, 0=left
    face_sign = (facing * 2.0 - 1.0)  # left=-1, right=+1

    pos_x = np.asarray([r.p1_pos_x for r in rows]) * face_sign
    pos_y = np.asarray([r.p1_pos_y for r in rows])
    distance = np.asarray([r.distance for r in rows])
    percent = np.log1p(np.asarray([float(r.p1_percent) for r in rows], dtype=np.float32))
    stock = np.asarray([float(r.p1_stock) for r in rows], dtype=np.float32) / 4.0
    on_ground = np.asarray([r.p1_on_ground for r in rows])
    invuln = np.asarray([r.p1_invulnerable for r in rows])
    offstage = np.asarray([r.p1_off_stage for r in rows])
    in_hitstun = np.asarray([r.p1_hitstun_frames_left > 0.0 for r in rows])

    vx_air = np.asarray([r.p1_speed_air_x_self for r in rows]) * face_sign
    vy = np.asarray([r.p1_speed_y_self for r in rows])
    vx_ground = np.asarray([r.p1_speed_ground_x_self for r in rows]) * face_sign

    # Lag-1 buttons as features
    def lag1(x: np.ndarray) -> np.ndarray:
        y = np.empty_like(x)
        y[0] = 0.0
        y[1:] = x[:-1]
        return y

    A_l1, B_l1, Z_l1, JUMP_l1, SHIELD_l1 = map(lag1, (A, B, Z, JUMP, SHIELD))

    feats = [
        ("pos_x_mir", pos_x),
        ("pos_y", pos_y),
        ("distance", distance),
        ("percent_log1p", percent),
        ("stock_div4", stock),
        ("on_ground", on_ground),
        ("off_stage", offstage),
        ("invulnerable", invuln),
        ("in_hitstun", in_hitstun),
        ("vx_air_mir", vx_air),
        ("vy", vy),
        ("vx_ground_mir", vx_ground),
        ("A_l1", A_l1), ("B_l1", B_l1), ("Z_l1", Z_l1), ("JUMP_l1", JUMP_l1), ("SHIELD_l1", SHIELD_l1),
    ]
    X = np.stack([v for _, v in feats], axis=1).astype(np.float32)  # (N, F)
    names = [n for n, _ in feats]

    # y at t+1 (aligned later via windowing)
    y_buttons = np.stack([A, B, Z, JUMP, SHIELD], axis=1).astype(np.float32)  # (N,5)
    y_stick = np.stack([msx, msy], axis=1).astype(np.float32)  # raw; will sectorize later

    return dict(
        X=X, names=np.array(names, dtype=object),
        y_action=action, y_buttons=y_buttons, y_stick_xy=y_stick
    )


def fit_preprocess(train_rows: Sequence[Row], all_rows: Sequence[Row],
                   ) -> Tuple[FeaturePack, FeaturePack, PreprocessArtifacts]:
    raw_all = extract_arrays(all_rows)
    raw_trn = extract_arrays(train_rows)

    # Velocity transforms (winsor + signed sqrt) fit on TRAIN only
    # indices in current feature layout:
    #  0:pos_x, 1:pos_y, 2:distance, 3:percent, 4:stock, 5:on_ground, 6:off_stage, 7:invuln, 8:in_hitstun,
    #  9:vx_air, 10:vy, 11:vx_ground, 12..16: lag buttons
    vel_idx = (9, 10, 11)
    X_trn = raw_trn["X"].copy()
    for j in vel_idx:
        X_trn[:, j] = winsor_signed_sqrt(X_trn[:, j], 1.0, 99.0)

    # Same transform for ALL rows using thresholds from TRAIN (approx: use train percentiles via function again on all)
    X_all = raw_all["X"].copy()
    for j in vel_idx:
        X_all[:, j] = winsor_signed_sqrt(X_all[:, j], 1.0, 99.0)

    # Robust scale all features (fit on TRAIN)
    scaler = RobustScaler.fit(X_trn)
    X_trn_s = scaler.transform(X_trn)
    X_all_s = scaler.transform(X_all)

    # Actions are already preprocessed to dense ids covering all used actions.
    y_action_all = raw_all["y_action"].astype(np.int64)  # (N,)
    y_action_trn = raw_trn["y_action"].astype(np.int64)
    n_actions = int(max(int(y_action_all.max()), int(y_action_trn.max())) + 1)
    idx_to_action_name: List[str] = [f"ACTION_{i}" for i in range(n_actions)]
    # Stick sector at t+1
    ms_xy = raw_all["y_stick_xy"]  # (N,2)
    ms_xy_t1 = ms_xy.copy()
    # sectorization done later in torch (same math), but we can do it here for numpy val metrics too:
    # We'll sectorize in torch path; here we keep raw.

    # Buttons (A,B,Z,JUMP,SHIELD) at t+1
    y_btn_all = raw_all["y_buttons"].astype(np.float32)  # (N,5)

    # Pack: we will window and align t->t+1 later in Dataset
    feats_all = FeaturePack(
        X=X_all_s, y_action=y_action_all, y_stick=ms_xy_t1[:, 0], y_buttons=y_btn_all
        # y_stick holds x, y kept separately via dataset
    )
    feats_trn = FeaturePack(
        X=X_trn_s, y_action=y_action_trn,
        y_stick=raw_trn["y_stick_xy"][:, 0], y_buttons=raw_trn["y_buttons"].astype(np.float32)
    )  # not actually used aside from scaler fitting

    artifacts = PreprocessArtifacts(
        scaler=scaler,
        vel_indices=vel_idx,
        feature_names=list(raw_all["names"]),
        idx_to_action_name=idx_to_action_name,
        n_actions=n_actions,
    )
    return feats_trn, feats_all, artifacts


@dataclass
class WindowConfig:
    T: int = 256  # frames per window (we will predict for T-1 positions: t->t+1)
    stride: int = 1


class ReplayWindowedDataset(Dataset):
    def __init__(self,
                 feats: FeaturePack,
                 stick_xy: np.ndarray,  # (N,2) in [0,1]
                 start_indices: List[int],
                 cfg: WindowConfig):
        super().__init__()
        self.cfg = cfg
        self.X = feats.X
        self.y_action = feats.y_action
        self.y_buttons = feats.y_buttons
        self.stick_xy = stick_xy
        self.starts = start_indices

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        s = self.starts[idx]
        T = self.cfg.T
        x = self.X[s: s + T - 1]  # (T-1, F) inputs at t

        # Targets at t+1:
        y_act = self.y_action[s + 1: s + T]  # (T-1,)
        y_btn = self.y_buttons[s + 1: s + T]  # (T-1,5)

        # Previous labels for inputs at t (observed at time t)
        prev_act = self.y_action[s: s + T - 1]  # (T-1,)

        ms_xy = self.stick_xy[s + 1: s + T]  # (T-1,2)
        prev_xy = self.stick_xy[s: s + T - 1]  # (T-1,2)

        def _nearest_idx_numpy(points: np.ndarray) -> np.ndarray:
            diffs = points[:, None, :] - STICK_XY_CLUSTER_CENTERS_V2[None, :, :]
            dist2 = np.sum(diffs * diffs, axis=2)
            return np.argmin(dist2, axis=1).astype(np.int64)

        stick_idx = _nearest_idx_numpy(ms_xy)
        prev_idx = _nearest_idx_numpy(prev_xy)
        out = {
            "x": torch.from_numpy(x).float(),
            "y_action": torch.from_numpy(y_act).long(),
            "y_buttons": torch.from_numpy(y_btn).float(),
            "y_stick_idx": torch.from_numpy(stick_idx).long(),
            "stick_xy": torch.from_numpy(ms_xy).float(),
            "prev_action": torch.from_numpy(prev_act).long(),
            "prev_stick_xy": torch.from_numpy(prev_xy).float(),
            "prev_stick_idx": torch.from_numpy(prev_idx).long(),
        }
        return out


@dataclass
class HeadDims:
    n_actions: int
    n_buttons: int = 5
    n_stick_bins: int = STICK_XY_CLUSTER_CENTERS_V2.shape[0]


def _stick_centers_tensor(device: torch.device, dtype: torch.dtype) -> Tensor:
    """Returns STICK_XY_CLUSTER_CENTERS_V2 as a torch tensor on the requested device/dtype."""
    return torch.tensor(STICK_XY_CLUSTER_CENTERS_V2, device=device, dtype=dtype)


def clusterize_tensor(ms_xy: Tensor, centers01: Tensor) -> Tensor:
    """Assign each (x,y) in [0,1] to nearest cluster center index (0..K-1)."""
    diff = ms_xy.unsqueeze(-2) - centers01  # (..., K, 2)
    dist2 = (diff * diff).sum(dim=-1)  # (..., K)
    return torch.argmin(dist2, dim=-1).long()


def soft_probs_from_xy_to_centers(v_xy_m11: Tensor, centers01: Tensor, beta: float = 50.0) -> Tensor:
    """Soft K-way distribution over centers from predicted (x,y) in [-1,1]."""
    centers_m11 = centers01 * 2.0 - 1.0  # (K,2) to [-1,1]
    diff = v_xy_m11.unsqueeze(-2) - centers_m11  # (B,T,K,2)
    d2 = (diff * diff).sum(dim=-1)  # (B,T,K)
    scores = -beta * d2
    return torch.softmax(scores, dim=-1).clamp_min(1e-8)


class TinyMeleeGRU(nn.Module):
    def __init__(self, f_in: int, hidden: int, heads: HeadDims) -> None:
        super().__init__()
        self.heads = heads

        # Prev-label embeddings (action idx, stick sector)
        self.prev_action_emb = nn.Embedding(heads.n_actions, 64)
        self.prev_stick_emb = nn.Embedding(self.heads.n_stick_bins, 16)

        f_in_total = f_in + 64 + 16
        self.enc = nn.Sequential(
            nn.Linear(f_in_total, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
        )
        # The GRU's hidden size is the most critical parameter.
        # It should be increased when calling this class, e.g., hidden=256 or hidden=512.
        self.gru = nn.GRU(input_size=128, hidden_size=hidden, batch_first=True)
        self.drop = nn.Dropout(0.1)

        self.stick_cls = nn.Sequential(
            nn.Linear(hidden, 256), nn.ReLU(inplace=True),
            nn.Linear(256, self.heads.n_stick_bins)
        )

        self.stick_reg = nn.Sequential(
            nn.Linear(hidden, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 2), nn.Tanh()
        )

        # Buttons: consume hidden + stick one-hot (teacher or predicted)
        self.button_head = nn.Sequential(
            nn.Linear(hidden + heads.n_stick_bins, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, heads.n_buttons)
        )

        self.action_head = nn.Sequential(
            nn.Linear(hidden, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 512), nn.ReLU(inplace=True),
            nn.Linear(512, heads.n_actions)
        )

    def forward(self, x: Tensor, *, prev_action_idx: Tensor, prev_stick_idx: Tensor,
                stick_teacher: Optional[Tensor] = None) -> Dict[str, Tensor]:
        a_emb = self.prev_action_emb(prev_action_idx)  # (B,T,64)
        s_emb = self.prev_stick_emb(prev_stick_idx)  # (B,T,16)
        xin = torch.cat([x, a_emb, s_emb], dim=-1)

        h = self.enc(xin)
        h, _ = self.gru(h)
        h = self.drop(h)

        stick_logits = self.stick_cls(h)  # (B,T,K)
        xy_pred = self.stick_reg(h)  # (B,T,2) in [-1,1]

        # Button conditioning uses either teacher or predicted cluster index
        if stick_teacher is None:
            stick_idx = torch.argmax(stick_logits, dim=-1)  # (B,T)
        else:
            stick_idx = stick_teacher
        stick_oh = one_hot(stick_idx, self.heads.n_stick_bins)

        btn_logits = self.button_head(torch.cat([h, stick_oh], dim=-1))
        action_logits = self.action_head(h)

        return {
            "stick_logits": stick_logits,  # (B,T,K) raw logits over centers
            "stick_xy_pred": xy_pred,  # (B,T,2) in [-1,1]
            "button_logits": btn_logits,  # (B,T,5)
            "action_logits": action_logits,  # (B,T,C)
        }


@dataclass
class LossWeights:
    w_action: float = 1.0
    w_buttons: float = 0.5
    w_stick_center: float = 0.5
    w_stick_dir: float = 5.0
    w_stick_reg: float = 2.0
    w_stick_kl: float = 0.1


def build_pos_weight(prevalences: List[float]) -> Tensor:
    return torch.tensor([(1.0 - p) / max(p, 1e-6) for p in prevalences], dtype=torch.float32)


def compute_button_prevalence(loader: DataLoader) -> List[float]:
    tot = 0
    pos = None
    for b in loader:
        y = b["y_buttons"]  # (B,T,5)
        if pos is None:
            pos = y.sum(dim=(0, 1))
        else:
            pos += y.sum(dim=(0, 1))
        tot += y.shape[0] * y.shape[1]
    assert pos is not None
    return (pos / float(tot)).tolist()  # type: ignore[return-value]


def loss_fn(out: Dict[str, Tensor],
            y_action: Tensor,
            y_stick_idx: Tensor,
            y_stick_xy: Tensor,
            y_buttons: Tensor,
            pos_weight_btn: Tensor,
            weights: LossWeights,
            stick_class_weights: Tensor
            ) -> Tuple[Tensor, Dict[str, float]]:
    ce = nn.CrossEntropyLoss()
    bce_btn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_btn)

    # Action CE
    la = ce(out["action_logits"].reshape(-1, out["action_logits"].size(-1)),
            y_action.reshape(-1))
    K = out["stick_logits"].size(-1)
    ce_stick = nn.CrossEntropyLoss(weight=stick_class_weights)
    lcls = ce_stick(out["stick_logits"].reshape(-1, K), y_stick_idx.reshape(-1))

    # (x,y) regression in [-1,1]
    target_xy = 2.0 * (y_stick_xy - 0.5)
    lreg = nn.SmoothL1Loss()(out["stick_xy_pred"], target_xy)

    # KL consistency between K-way classifier and soft assignment from regressed XY
    centers01 = _stick_centers_tensor(y_action.device, out["stick_logits"].dtype)
    logpK = torch.log_softmax(out["stick_logits"], dim=-1)  # (B,T,K)
    pK = torch.softmax(out["stick_logits"], dim=-1)  # (B,T,K)
    pregK = soft_probs_from_xy_to_centers(out["stick_xy_pred"], centers01)  # (B,T,K)
    lkl1 = torch.nn.functional.kl_div(logpK, pregK, reduction="batchmean")
    lkl2 = torch.nn.functional.kl_div(torch.log(pregK), pK.detach(), reduction="batchmean")
    lkl = 0.5 * (lkl1 + lkl2)

    # Buttons BCE (multi-label)
    lb = bce_btn(out["button_logits"], y_buttons)

    tot = (
            weights.w_action * la +
            weights.w_stick_dir * lcls +  # reuse w_stick_dir as the K-way class weight
            weights.w_stick_reg * lreg +
            weights.w_stick_kl * lkl +
            weights.w_buttons * lb
    )
    logs = {
        "L": float(tot.detach()),
        "La": float(la.detach()),
        "Ls": float((lcls + lreg + lkl).detach()),
        "Lb": float(lb.detach()),
        "Lstick_cls": float(lcls.detach()),
        "Lstick_reg": float(lreg.detach()),
        "Lstick_kl": float(lkl.detach()),
    }
    return tot, logs


def _format_action_confusions(sum_probs_by_true: torch.Tensor,
                              counts_by_true: torch.Tensor,
                              nll_true_sum: torch.Tensor,
                              top_classes: int = 8,
                              top_confusers: int = 5,
                              label_fn: Optional[Callable[[int], str]] = None) -> str:
    C = sum_probs_by_true.size(0)
    label = (label_fn if label_fn is not None else (lambda i: str(i)))

    # Mean pred distribution conditioned on each true class
    denom = counts_by_true.clamp_min(1.0).unsqueeze(1)  # (C,1)
    mean_probs = (sum_probs_by_true / denom)  # (C,C)

    # Per-class perplexity from per-class mean NLL
    per_class_ppl = torch.exp(nll_true_sum / counts_by_true.clamp_min(1.0))  # (C,)

    # Top-K most common true classes
    top_true = torch.topk(counts_by_true, k=min(top_classes, C)).indices.tolist()

    lines = []
    lines.append("Action soft confusions (top-{}/class):".format(top_confusers))
    for c in top_true:
        row = mean_probs[c].clone()
        # Exclude the true class when ranking confusers
        row[c] = -1.0
        vals, idx = torch.topk(row, k=min(top_confusers, C - 1))
        conf = ", ".join([f"{label(j)}={vals[i].item():.3f}" for i, j in enumerate(idx.tolist())])
        lines.append(
            f"  true={label(c):<16} "
            f"count={int(counts_by_true[c].item()):>7d}  "
            f"ppl={per_class_ppl[c].item():.2f}  "
            f"top_confusers: {conf}"
        )
    return "\n".join(lines)


# --- Stick soft confusion report (K-way, generic) ---
def _format_stick_confusions(sum_probs_by_true: torch.Tensor,
                             counts_by_true: torch.Tensor,
                             nll_true_sum: torch.Tensor,
                             top_classes: int = 8,
                             top_confusers: int = 5,
                             label_fn: Optional[Callable[[int], str]] = None) -> str:
    K = sum_probs_by_true.size(0)
    label = (label_fn if label_fn is not None else (lambda i: str(i)))

    # Mean pred distribution conditioned on each true class
    denom = counts_by_true.clamp_min(1.0).unsqueeze(1)  # (K,1)
    mean_probs = (sum_probs_by_true / denom)  # (K,K)

    # Per-class perplexity from per-class mean NLL
    per_class_ppl = torch.exp(nll_true_sum / counts_by_true.clamp_min(1.0))  # (K,)

    # Top-K most common true classes
    top_true = torch.topk(counts_by_true, k=min(top_classes, K)).indices.tolist()

    lines = []
    lines.append("Stick soft confusions (top-{}/class):".format(top_confusers))
    for c in top_true:
        row = mean_probs[c].clone()
        # Exclude the true class when ranking confusers
        row[c] = -1.0
        vals, idx = torch.topk(row, k=min(top_confusers, K - 1))
        conf = ", ".join([f"{label(j)}={vals[i].item():.3f}" for i, j in enumerate(idx.tolist())])
        lines.append(
            f"  true={label(c):<16} "
            f"count={int(counts_by_true[c].item()):>7d}  "
            f"ppl={per_class_ppl[c].item():.2f}  "
            f"top_confusers: {conf}"
        )
    return "\n".join(lines)


@torch.no_grad()
def evaluate(model: TinyMeleeGRU, loader: DataLoader, device: torch.device, *,
             confusion_top_classes: int = 0,  # 0 = don't print
             confusion_top_confusers: int = 5,
             action_label_fn: Optional[Callable[[int], str]] = None) -> Dict[str, float]:
    model.eval()
    centers01_eval = _stick_centers_tensor(device, torch.float32)
    K_centers = centers01_eval.size(0)
    centers_m11 = centers01_eval * 2.0 - 1.0
    r_centers = torch.linalg.norm(centers_m11, dim=-1)  # (K,)
    is_center_idx = (r_centers <= 0.15)  # neutral radius threshold

    n_tok = 0
    # --- Soft confusion accumulators for the action head ---
    sum_probs_by_true = None  # (C,C) sum of predicted prob mass per (true,pred)
    counts_by_true = None  # (C,) count per true class
    nll_true_sum = None  # (C,) sum of -log p(true|x) per true class

    # --- Soft confusion accumulators for the stick head ---
    sum_probs_by_true_stick = None  # (K,K) sum of predicted prob mass per (true,pred)
    counts_by_true_stick = None  # (K,) count per true class
    nll_true_sum_stick = None  # (K,) sum of -log p(true|x) per true class

    # Action / stick accuracies and NLL
    acc_action = 0.0
    acc_action_top5 = 0.0
    acc_stick = 0.0
    nll_action_sum = 0.0
    nll_stick_sum = 0.0

    n_actions: Optional[int] = None
    action_counts_true: Optional[torch.Tensor] = None  # (C,)
    stick_counts_true = torch.zeros(K_centers, device=device)

    # Lag-1 baselines
    lag1_correct_action = 0
    lag1_total_action = 0
    lag1_correct_stick = 0
    lag1_total_stick = 0

    # Center/neutral collapse checks
    center_true = 0
    center_pred = 0

    # Buttons: collect logits (closed-loop & teacher-forced) and labels to compute full metrics
    logits_btn_cl: List[torch.Tensor] = []
    logits_btn_tf: List[torch.Tensor] = []
    y_btn_all: List[torch.Tensor] = []

    # Micro counts at 0.5 (closed-loop & TF)
    tp_cl = torch.zeros(5, device=device)
    fp_cl = torch.zeros(5, device=device)
    fn_cl = torch.zeros(5, device=device)

    tp_tf = torch.zeros(5, device=device)
    fp_tf = torch.zeros(5, device=device)
    fn_tf = torch.zeros(5, device=device)

    # Shine probes (closed-loop & TF)
    down_bins = {5, 4, 6}  # d, dr, dl
    ce_action = nn.CrossEntropyLoss(reduction="none")
    ce_stick_eval = nn.CrossEntropyLoss(reduction="none")

    for b in loader:
        x = b["x"].to(device)  # (B,T,F)
        ya = b["y_action"].to(device)  # (B,T)
        yb = b["y_buttons"].to(device)  # (B,T,5)
        stick_idx_true = clusterize_tensor(b["stick_xy"].to(device), centers01_eval)  # (B,T)

        # Forward passes:
        pa = b["prev_action"].to(device)
        ps = b["prev_stick_idx"].to(device)

        out_cl = model(x, prev_action_idx=pa, prev_stick_idx=ps, stick_teacher=None)
        out_tf = model(x, prev_action_idx=pa, prev_stick_idx=ps, stick_teacher=stick_idx_true)

        # --------------------------
        # Action metrics
        # --------------------------
        action_logits = out_cl["action_logits"]  # (B,T,C)

        # Initialize dynamic dims once we see the first batch
        if n_actions is None:
            n_actions = action_logits.size(-1)
            action_counts_true = torch.zeros(n_actions, device=device)

        # Softmax probabilities for action confusion
        logits_flat = action_logits.reshape(-1, n_actions)
        ya_flat = ya.reshape(-1)

        probs = torch.softmax(logits_flat, dim=-1).detach()
        true_idx = ya_flat.detach()

        if sum_probs_by_true is None:
            sum_probs_by_true = torch.zeros(n_actions, n_actions, device=device)
            counts_by_true = torch.zeros(n_actions, device=device)
            nll_true_sum = torch.zeros(n_actions, device=device)

        # Row-wise accumulate probs into the row indexed by the true class
        sum_probs_by_true.index_add_(0, true_idx, probs)
        counts_by_true.index_add_(0, true_idx, torch.ones_like(true_idx, dtype=torch.float))

        # Per-class NLL sum: -log p(true)
        true_p = probs.gather(1, true_idx.unsqueeze(1)).squeeze(1).clamp_min(1e-12)
        nll_true_sum.index_add_(0, true_idx, -true_p.log())
        if n_actions is None:
            n_actions = action_logits.size(-1)
            action_counts_true = torch.zeros(n_actions, device=device)

        # Accumulate NLL
        nll_action_sum += ce_action(action_logits.reshape(-1, n_actions), ya.reshape(-1)).sum().item()

        # Top-1 / Top-5
        pred1 = action_logits.argmax(dim=-1)
        acc_action += (pred1 == ya).float().sum().item()
        top5_vals, top5_idx = torch.topk(action_logits, k=min(5, n_actions), dim=-1)
        acc_action_top5 += (top5_idx == ya.unsqueeze(-1)).any(dim=-1).float().sum().item()

        # True action distribution (for majority baseline)
        action_counts_true += torch.bincount(ya.reshape(-1), minlength=n_actions)

        ce_stick_eval = nn.CrossEntropyLoss(reduction="none")
        # ...
        stick_idx_true = clusterize_tensor(b["stick_xy"].to(device), centers01_eval)  # (B,T)
        # ...
        stick_logits = out_cl["stick_logits"]  # (B,T,K_centers)
        nll_stick_sum += ce_stick_eval(
            stick_logits.reshape(-1, K_centers),
            stick_idx_true.reshape(-1)
        ).sum().item()

        # Stick soft confusion accumulation
        probs_st = torch.softmax(stick_logits, dim=-1).detach()  # (B,T,K)
        true_idx_st = stick_idx_true.detach()  # (B,T)
        if sum_probs_by_true_stick is None:
            sum_probs_by_true_stick = torch.zeros(K_centers, K_centers, device=device)
            counts_by_true_stick = torch.zeros(K_centers, device=device)
            nll_true_sum_stick = torch.zeros(K_centers, device=device)

        # Flatten (B,T,*) -> (N, *)
        probs_st_flat = probs_st.reshape(-1, K_centers)  # (N,K)
        true_idx_st_flat = true_idx_st.reshape(-1)  # (N,)

        sum_probs_by_true_stick.index_add_(0, true_idx_st_flat, probs_st_flat)
        counts_by_true_stick.index_add_(0, true_idx_st_flat, torch.ones_like(true_idx_st_flat, dtype=torch.float))
        true_p_st = probs_st_flat.gather(1, true_idx_st_flat.unsqueeze(1)).squeeze(1).clamp_min(1e-12)
        nll_true_sum_stick.index_add_(0, true_idx_st_flat, -true_p_st.log())
        # true center rate from geometry (radius in [-1,1])
        r_true = torch.linalg.norm(2.0 * (b["stick_xy"].to(device) - 0.5), dim=-1)  # (B,T)
        center_true += (r_true <= 0.15).sum().item()

        # predicted center rate from predicted cluster index membership
        stick_pred_idx = stick_logits.argmax(dim=-1)  # (B,T)
        center_pred += is_center_idx[stick_pred_idx].sum().item()

        acc_stick += (stick_pred_idx == stick_idx_true).float().sum().item()
        stick_counts_true += torch.bincount(stick_idx_true.reshape(-1), minlength=K_centers)

        if ya.size(1) > 1:
            lag1_correct_action += (ya[:, 1:] == ya[:, :-1]).sum().item()
            lag1_total_action += ya[:, 1:].numel()

            lag1_correct_stick += (stick_idx_true[:, 1:] == stick_idx_true[:, :-1]).sum().item()
            lag1_total_stick += stick_idx_true[:, 1:].numel()

        logits_btn_cl.append(out_cl["button_logits"].detach().cpu())
        logits_btn_tf.append(out_tf["button_logits"].detach().cpu())
        y_btn_all.append(yb.detach().cpu())

        # Threshold 0.5 (closed-loop)
        pred_btn_cl = (torch.sigmoid(out_cl["button_logits"]) > 0.5).float()
        tp_cl += (pred_btn_cl * yb).sum(dim=(0, 1))
        fp_cl += (pred_btn_cl * (1.0 - yb)).sum(dim=(0, 1))
        fn_cl += ((1.0 - pred_btn_cl) * yb).sum(dim=(0, 1))

        # Threshold 0.5 (teacher-forced)
        pred_btn_tf = (torch.sigmoid(out_tf["button_logits"]) > 0.5).float()
        tp_tf += (pred_btn_tf * yb).sum(dim=(0, 1))
        fp_tf += (pred_btn_tf * (1.0 - yb)).sum(dim=(0, 1))
        fn_tf += ((1.0 - pred_btn_tf) * yb).sum(dim=(0, 1))

        # Token count
        n_tok += ya.numel()

    assert n_actions is not None and action_counts_true is not None

    # Majority baselines
    action_majority = int(torch.argmax(action_counts_true).item())
    stick_majority = int(torch.argmax(stick_counts_true).item())

    acc_action_majority = (action_counts_true.max() / action_counts_true.sum()).item()
    acc_stick_majority = (stick_counts_true.max() / stick_counts_true.sum()).item()
    acc_stick_center = (stick_counts_true[0] / stick_counts_true.sum()).item()  # always-predict-center baseline

    # Lag-1 baselines
    acc_action_lag1 = (lag1_correct_action / max(lag1_total_action, 1)) if lag1_total_action else 0.0
    acc_stick_lag1 = (lag1_correct_stick / max(lag1_total_stick, 1)) if lag1_total_stick else 0.0

    # NLL / perplexity
    nll_action = nll_action_sum / n_tok
    nll_stick = nll_stick_sum / n_tok
    ppl_action = float(math.exp(min(50.0, nll_action)))  # numeric safety
    ppl_stick = float(math.exp(min(50.0, nll_stick)))

    # Stick center rates
    stick_true_center_rate = center_true / n_tok
    stick_pred_center_rate = center_pred / n_tok

    # --------------------------
    # Buttons metrics
    # --------------------------
    # stack collected logits/labels to CPU
    logits_cl = torch.cat(logits_btn_cl, dim=0)  # (Nseq, T, 5) -> concatenate along batch dimension
    logits_tf = torch.cat(logits_btn_tf, dim=0)
    ybtn = torch.cat(y_btn_all, dim=0)

    def prf_from_counts(tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor) -> Tuple[
        float, float, float, torch.Tensor, torch.Tensor, torch.Tensor]:
        prec = tp / torch.clamp(tp + fp, min=1e-6)
        rec = tp / torch.clamp(tp + fn, min=1e-6)
        f1 = 2 * prec * rec / torch.clamp(prec + rec, min=1e-6)
        # micro
        tp_m = tp.sum().item()
        fp_m = fp.sum().item()
        fn_m = fn.sum().item()
        prec_m = tp_m / max(tp_m + fp_m, 1e-6)
        rec_m = tp_m / max(tp_m + fn_m, 1e-6)
        f1_m = 2 * prec_m * rec_m / max(prec_m + rec_m, 1e-6)
        return float(prec_m), float(rec_m), float(f1_m), prec, rec, f1

    # Closed-loop at 0.5
    prec_m_cl, rec_m_cl, f1_m_cl, prec_cl_vec, rec_cl_vec, f1_cl_vec = prf_from_counts(tp_cl, fp_cl, fn_cl)
    # Teacher-forced at 0.5
    prec_m_tf, rec_m_tf, f1_m_tf, prec_tf_vec, rec_tf_vec, f1_tf_vec = prf_from_counts(tp_tf, fp_tf, fn_tf)

    # Label prevalences (true) and predicted-positive-rates (closed-loop @0.5)
    total_btn = ybtn.shape[0] * ybtn.shape[1]
    prev_vec = (ybtn.sum(dim=(0, 1)) / total_btn)  # (5,)
    ppr_vec_cl = (torch.sigmoid(logits_cl) > 0.5).float().sum(dim=(0, 1)) / total_btn  # predicted positive rate

    # Tuned thresholds per label (maximize F1 on CLOSED-LOOP logits)
    # simple grid search 0.05..0.95
    grid = torch.linspace(0.05, 0.95, steps=19)
    tuned_thr = []
    tuned_f1 = []
    for k in range(5):
        scores = torch.sigmoid(logits_cl[..., k]).reshape(-1)
        yk = ybtn[..., k].reshape(-1) > 0.5
        best_f1 = 0.0
        best_t = 0.5
        for t in grid:
            pred = (scores >= t)
            tp = (pred & yk).sum().item()
            fp = (pred & (~yk)).sum().item()
            fn = ((~pred) & yk).sum().item()
            f1 = 2 * tp / max(2 * tp + fp + fn, 1e-6)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        tuned_thr.append(best_t)
        tuned_f1.append(best_f1)

    def average_precision(scores: torch.Tensor, labels: torch.Tensor) -> float:
        if labels.sum().item() == 0:
            return 0.0
        idx = torch.argsort(scores, descending=True)
        y = labels[idx].float()
        tp_cum = torch.cumsum(y, dim=0)
        denom = torch.arange(1, y.numel() + 1, dtype=torch.float32)
        prec = tp_cum / denom
        ap = (prec * y).sum().item() / max(1.0, y.sum().item())
        return ap

    ap_labels = []
    for k in range(5):
        scores = torch.sigmoid(logits_cl[..., k]).reshape(-1).cpu()
        yk = (ybtn[..., k].reshape(-1) > 0.5).cpu()
        ap_labels.append(average_precision(scores, yk))
    ap_macro = float(np.mean(ap_labels))
    scores_m = torch.sigmoid(logits_cl).reshape(-1).cpu()
    y_m = (ybtn.reshape(-1) > 0.5).cpu()
    ap_micro = average_precision(scores_m, y_m)

    out: Dict[str, float] = {
        # Tokens
        "n_tokens": float(n_tok),

        # Action
        "acc_action": acc_action / n_tok,
        "acc_action_top5": acc_action_top5 / n_tok,
        "acc_action_majority": acc_action_majority,
        "acc_action_lag1": acc_action_lag1,
        "nll_action": nll_action,
        "ppl_action": ppl_action,

        # Stick
        "acc_stick": acc_stick / n_tok,
        "acc_stick_majority": acc_stick_majority,
        "acc_stick_center_baseline": acc_stick_center,
        "acc_stick_lag1": acc_stick_lag1,
        "nll_stick": nll_stick,
        "ppl_stick": ppl_stick,
        "stick_true_center_rate": stick_true_center_rate,
        "stick_pred_center_rate": stick_pred_center_rate,
        "stick_majority_idx": float(stick_majority),

        # Buttons @0.5 thresholds
        "btn_prec_micro_cl_05": prec_m_cl,
        "btn_rec_micro_cl_05": rec_m_cl,
        "btn_f1_micro_cl_05": f1_m_cl,
        "btn_prec_micro_tf_05": prec_m_tf,
        "btn_rec_micro_tf_05": rec_m_tf,
        "btn_f1_micro_tf_05": f1_m_tf,

        # Per-label macro-ish (mean over labels) at 0.5
        "btn_f1_macro_cl_05": float(f1_cl_vec.mean().item()),
        "btn_f1_macro_tf_05": float(f1_tf_vec.mean().item()),

    }
    out["btn_ap_macro_cl"] = ap_macro
    out["btn_ap_micro_cl"] = ap_micro

    # Add per-label details: prevalence, predicted positive rate, tuned thresholds & tuned F1
    label_names = ["A", "B", "Z", "JUMP", "SHIELD"]
    for i, name in enumerate(label_names):
        out[f"btn_ap_cl_{name}"] = float(ap_labels[i])
        out[f"btn_prev_{name}"] = float(prev_vec[i].item())
        out[f"btn_ppr_cl_05_{name}"] = float(ppr_vec_cl[i].item())
        out[f"btn_f1_cl_05_{name}"] = float(f1_cl_vec[i].item())
        out[f"btn_f1_tf_05_{name}"] = float(f1_tf_vec[i].item())
        out[f"btn_thr_tuned_{name}"] = float(tuned_thr[i])
        out[f"btn_f1_cl_tuned_{name}"] = float(tuned_f1[i])
    if confusion_top_classes > 0 and sum_probs_by_true is not None:
        # move to CPU just for formatting
        rpt = _format_action_confusions(sum_probs_by_true.detach().cpu(),
                                        counts_by_true.detach().cpu(),
                                        nll_true_sum.detach().cpu(),
                                        top_classes=confusion_top_classes,
                                        top_confusers=confusion_top_confusers,
                                        label_fn=action_label_fn)
        out["action_confusions_report"] = rpt
    if confusion_top_classes > 0 and sum_probs_by_true_stick is not None:
        # Optional label: index + center coordinate
        def _stick_label(i: int) -> str:
            xy = centers01_eval[i]
            return f"{i}@({float(xy[0]):.2f},{float(xy[1]):.2f})"

        rpt_s = _format_stick_confusions(sum_probs_by_true_stick.detach().cpu(),
                                         counts_by_true_stick.detach().cpu(),
                                         nll_true_sum_stick.detach().cpu(),
                                         top_classes=min(confusion_top_classes, K_centers),
                                         top_confusers=confusion_top_confusers,
                                         label_fn=_stick_label)
        out["stick_confusions_report"] = rpt_s
    return out


# Device picker: CUDA > MPS > CPU
def get_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def find_optimal_thresholds(
        all_logits: torch.Tensor,
        all_labels: torch.Tensor
) -> Dict[str, float]:
    print("Finding optimal F1 thresholds...")
    # Detach and move to CPU just in case they are on the GPU
    all_logits = all_logits.detach().cpu()
    all_labels = all_labels.detach().cpu()

    optimal_thresholds = {}
    grid = torch.linspace(0.001, 0.999, steps=998)

    for k, label_name in enumerate(BUTTON_LABELS):
        scores = torch.sigmoid(all_logits[:, k])
        labels = (all_labels[:, k] > 0.5)
        best_f1 = -1.0
        best_t = 0.5

        for t in grid:
            preds = (scores >= t)
            tp = (preds & labels).sum().item()
            fp = (preds & (~labels)).sum().item()
            fn = ((~preds) & labels).sum().item()

            # F1 formula, with a safe guard against division by zero
            f1 = (2 * tp) / max(2 * tp + fp + fn, 1e-6)

            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)

        optimal_thresholds[label_name] = best_t
        print(f"  - Optimal threshold for '{label_name}': {best_t:.2f} (achieves F1={best_f1:.3f})")

    return optimal_thresholds


def _atomic_save(obj, path: Path) -> None:
    """Write to a temp file then atomically rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _list_epoch_checkpoints(ckpt_dir: Path):
    return sorted(ckpt_dir.glob("epoch_*.pt"))


def find_latest_checkpoint(ckpt_dir) -> Optional[Path]:
    """Return latest checkpoint path in a directory, or None."""
    d = Path(ckpt_dir)
    latest = d / "latest.pt"
    if latest.exists():
        return latest
    eps = _list_epoch_checkpoints(d)
    return eps[-1] if eps else None


def save_checkpoint(ckpt_dir,
                    epoch: int,
                    model: "TinyMeleeGRU",
                    optimizer: torch.optim.Optimizer,
                    *,
                    heads: "HeadDims",
                    artifacts: "PreprocessArtifacts",
                    win_cfg: "WindowConfig",
                    feature_dim: int,
                    device_str: str,
                    keep_last: int = 5) -> Path:
    """Save a training checkpoint containing everything needed to resume or run inference."""
    ckpt_dir = Path(ckpt_dir)
    payload = {
        "format": 1,
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": {
            "arch": type(model).__name__,
            "hidden": int(model.gru.hidden_size),
            "feature_dim": int(feature_dim),
        },
        "heads": {
            "n_actions": int(heads.n_actions),
            "n_buttons": int(heads.n_buttons),
            "n_stick_bins": int(heads.n_stick_bins),
        },
        "artifacts": artifacts,
        "win_cfg": {"T": int(win_cfg.T), "stride": int(win_cfg.stride)},
        "device_str": device_str,
        "rng": {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    ep_path = ckpt_dir / f"epoch_{epoch:03d}.pt"
    _atomic_save(payload, ep_path)
    _atomic_save(payload, ckpt_dir / "latest.pt")
    # Retain only last `keep_last` checkpoints
    eps = _list_epoch_checkpoints(ckpt_dir)
    if keep_last and len(eps) > keep_last:
        for p in eps[:-keep_last]:
            try:
                p.unlink()
            except Exception:
                pass
    return ep_path


def load_checkpoint(ckpt_path,
                    model: Optional["TinyMeleeGRU"] = None,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    map_location: str | torch.device = "cpu"):
    """Load a checkpoint. If `model`/`optimizer` are provided, their state_dicts are restored.
    Returns (payload, epoch).
    """
    payload = torch.load(str(ckpt_path), map_location=map_location, weights_only=False)
    if model is not None and "model_state" in payload:
        model.load_state_dict(payload["model_state"])
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload, int(payload.get("epoch", 0))


def load_model_for_inference(ckpt_path,
                             device_str: Optional[str] = None):
    """Construct a model from a checkpoint and load its weights. Returns (model, context_dict)."""
    payload = torch.load(str(ckpt_path), map_location="cpu")
    model_cfg = payload["model_config"]
    heads_cfg = payload["heads"]
    heads = HeadDims(n_actions=heads_cfg["n_actions"],
                     n_buttons=heads_cfg["n_buttons"],
                     n_stick_bins=heads_cfg["n_stick_bins"])
    model = TinyMeleeGRU(f_in=model_cfg["feature_dim"],
                         hidden=model_cfg["hidden"],
                         heads=heads)
    model.load_state_dict(payload["model_state"])
    dev = torch.device(device_str or get_default_device())
    model.to(dev).eval()
    ctx = {
        "artifacts": payload.get("artifacts"),
        "win_cfg": payload.get("win_cfg"),
        "device_str": device_str or payload.get("device_str", "cpu"),
        "model_config": model_cfg,
        "heads": heads_cfg,
    }
    return model, ctx


def _scaler_to_serializable(scaler) -> dict:
    """Best-effort conversion of our RobustScaler instance to plain Python.

    Supports either a custom scaler with attributes like `median_`/`iqr_` or
    `center_`/`scale_`, or any object exposing a `to_dict()` method. All arrays
    are converted to Python lists for JSON compatibility.
    """
    if hasattr(scaler, "to_dict") and callable(getattr(scaler, "to_dict")):
        d = scaler.to_dict()  # type: ignore[attr-defined]
    else:
        d = {}
        for name in ("center_", "scale_", "median_", "iqr_", "medians_", "iqrs_"):
            if hasattr(scaler, name):
                v = getattr(scaler, name)
                # numpy / torch to list
                try:
                    import torch  # local import safe here
                    if isinstance(v, torch.Tensor):
                        v = v.detach().cpu().tolist()
                except Exception:
                    pass
                try:
                    import numpy as _np
                    if isinstance(v, _np.ndarray):
                        v = v.tolist()
                except Exception:
                    pass
                # fall back to plain types
                if isinstance(v, (list, tuple)):
                    d[name] = list(v)
                else:
                    d[name] = v
        d["class"] = type(scaler).__name__
    return d


def save_runtime_json(path: str,
                      model: TinyMeleeGRU,
                      artifacts: "PreprocessArtifacts",
                      optimal_button_thresholds: dict,
                      *,
                      stick_r0: float = 0.15) -> None:
    # Infer a sensible cold-start previous action if we have a WAIT/IDLE class
    def _infer_start_action_idx() -> int:
        names = getattr(artifacts, "idx_to_action_name", []) or []
        for i, nm in enumerate(names):
            if nm.upper() in ("WAIT", "IDLE", "STAND", "STANDING", "NEUTRAL"):
                return i
        return 0

    # Serialize action maps (JSON keys are strings by spec)
    idx_to_action_name = [str(n) for n in getattr(artifacts, "idx_to_action_name", [])]

    # Scaler params
    scaler_dict = _scaler_to_serializable(artifacts.scaler)

    # Model/heads sizes
    heads = model.heads
    model_info = {
        "arch": type(model).__name__,
        "hidden": int(model.gru.hidden_size),
        "n_actions": int(heads.n_actions),
        "n_buttons": int(heads.n_buttons),
        "n_stick_bins": int(heads.n_stick_bins),
        "feature_dim": int(model.prev_action_emb.weight.size(1) +  # action emb dim
                           model.prev_stick_emb.weight.size(1))
    }

    payload = {
        "format_version": 1,
        "preprocessing": {
            "feature_names": list(artifacts.feature_names),
            "velocity_indices": list(artifacts.vel_indices),
            "winsor_percentiles": [1.0, 99.0],  # must match training
            "robust_scaler": scaler_dict,
        },
        "actions": {
            "n_classes": int(getattr(artifacts, "n_actions", len(idx_to_action_name))),
            "idx_to_action_name": idx_to_action_name,
        },
        "inference": {
            "button_thresholds": {str(k): float(v) for k, v in optimal_button_thresholds.items()},
            "stick_neutral_r0": float(stick_r0),
            "startup": {
                "prev_action_idx": _infer_start_action_idx(),
                "prev_stick_idx": 0  # center
            }
        },
        "model": model_info,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)


def train_gru_on_replay(rows: Sequence[Row],
                        T: int = 64,
                        stride: int = 32,
                        epochs: int = 5,
                        batch_size: int = 64,
                        lr: float = 3e-3,
                        device_str: Optional[str] = None,
                        export_json_path: Optional[str] = None,
                        ckpt_dir: Optional[str] = "./checkpoints/small_gru",
                        resume: bool = True,
                        keep_last: int = 5,
                        *,
                        # ReduceLROnPlateau controls
                        use_plateau_lr: bool = True,
                        plateau_mode: str = "min",
                        plateau_factor: float = 0.5,
                        plateau_patience: int = 2,
                        plateau_threshold: float = 1e-3,
                        plateau_threshold_mode: str = "rel",
                        plateau_cooldown: int = 0,
                        plateau_min_lr: float = 1e-6,
                        ) -> None:
    if device_str is None:
        device_str = get_default_device()
    device = torch.device(device_str)
    N = len(rows)
    assert N >= T + 1, "Replay too short for one window."

    # Temporal split
    split_idx = int(0.8 * N)
    train_rows = rows[:split_idx]
    # Fit preprocessing on TRAIN; produce scaled features for ALL
    _trn_pack, all_pack, art = fit_preprocess(train_rows, rows)

    # Build windowed datasets
    X_all = all_pack.X  # (N,F)
    # For sectorization we need (x,y) in [0,1]
    stick_xy = np.stack(
        [[float(r.p1_main_stick_x), float(r.p1_main_stick_y)] for r in rows],
        axis=0
    ).astype(np.float32)
    win_cfg = WindowConfig(T=T, stride=stride)

    print("Generating and shuffling all possible windows...")
    all_possible_starts = []
    for s in range(0, N - win_cfg.T + 1, win_cfg.stride):
        if s + win_cfg.T <= N:
            all_possible_starts.append(s)

    # Step 2: Shuffle this list of indices randomly.
    np.random.shuffle(all_possible_starts)

    # Step 3: Split the shuffled list of indices into 80% train and 20% validation.
    split_idx_random = int(0.8 * len(all_possible_starts))
    train_starts = all_possible_starts[:split_idx_random]
    val_starts = all_possible_starts[split_idx_random:]
    print(f"Total windows: {len(all_possible_starts)}. Split into {len(train_starts)} train and {len(val_starts)} val.")

    ds_tr = ReplayWindowedDataset(all_pack, stick_xy, train_starts, win_cfg)
    ds_va = ReplayWindowedDataset(all_pack, stick_xy, val_starts, win_cfg)

    def collate(batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        x = torch.stack([b["x"] for b in batch], dim=0)  # (B,T-1,F)
        ya = torch.stack([b["y_action"] for b in batch], dim=0)  # (B,T-1)
        yb = torch.stack([b["y_buttons"] for b in batch], dim=0)  # (B,T-1,5)
        ms = torch.stack([b["stick_xy"] for b in batch], dim=0)  # (B,T-1,2)

        pa = torch.stack([b["prev_action"] for b in batch], dim=0)  # (B,T-1)
        # ps_xy = torch.stack([b["prev_stick_xy"] for b in batch], dim=0)  # (B,T-1,2)
        # ps_idx = sectorize_tensor(ps_xy)  # (B,T-1)
        ystick_idx = torch.stack([b["y_stick_idx"] for b in batch], dim=0)  # (B,T-1)
        ps_idx = torch.stack([b["prev_stick_idx"] for b in batch], dim=0)  # (B,T-1)
        return {
            "x": x,
            "y_action": ya,
            "y_buttons": yb,
            "y_stick_idx": ystick_idx,  # NEW: cluster label for t+1
            "stick_xy": ms,  # keep the continuous [0,1] target
            "prev_action": pa,
            "prev_stick_idx": ps_idx,
        }

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collate)

    # Head dims
    heads = HeadDims(n_actions=NUM_USED_ACTIONS)

    model = TinyMeleeGRU(f_in=X_all.shape[1], hidden=512, heads=heads).to(device)

    # Button imbalance handling from TRAIN
    # Button & stick-center imbalance from TRAIN
    prevalences = compute_button_prevalence(dl_tr)
    pos_weight_btn = build_pos_weight(prevalences).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # --- ReduceLROnPlateau scheduler (step once per epoch after validation) ---
    plateau_scheduler = None
    if use_plateau_lr:
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode=plateau_mode,
            factor=plateau_factor,
            patience=plateau_patience,
            threshold=plateau_threshold,
            threshold_mode=plateau_threshold_mode,
            cooldown=plateau_cooldown,
            min_lr=plateau_min_lr,
        )
    print(f"Using device: {device}")

    print(f"Feature dims: {X_all.shape[1]}  |  Windows: train={len(ds_tr)} val={len(ds_va)}  |  Actions={NUM_USED_ACTIONS}")
    print(f"Per-label prevalences (A,B,Z,JUMP,SHIELD): {[round(p, 4) for p in prevalences]}")

    print("Computing stick cluster prevalences for weighting...")
    stick_counts = torch.zeros(heads.n_stick_bins)
    total_stick_samples = 0
    for b in dl_tr:
        # Use the cluster index you're already generating
        stick_idx = b["y_stick_idx"]  # (B, T-1)
        counts = torch.bincount(stick_idx.reshape(-1), minlength=heads.n_stick_bins)
        stick_counts += counts
        total_stick_samples += stick_idx.numel()

    stick_prevalences = stick_counts / total_stick_samples

    # Inverse frequency weighting
    stick_class_weights = (1.0 / stick_prevalences.clamp_min(1e-6)).to(device)
    # You might want to normalize them, e.g., by dividing by their sum
    stick_class_weights /= stick_class_weights.sum()
    stick_class_weights *= heads.n_stick_bins  # re-scale

    print(f"Stick class weights: {stick_class_weights.tolist()}")

    # Prepare checkpointing
    ckpt_path_obj = Path(ckpt_dir) if ckpt_dir else None
    if ckpt_path_obj:
        ckpt_path_obj.mkdir(parents=True, exist_ok=True)

    # Optionally resume training
    start_epoch = 1
    if resume and ckpt_path_obj:
        latest = find_latest_checkpoint(ckpt_path_obj)
        if latest is not None:
            try:
                payload, saved_epoch = load_checkpoint(latest, model=model, optimizer=opt, map_location="cpu")
                start_epoch = saved_epoch + 1
                start_epoch = saved_epoch + 1

                # Restore RNG states robustly
                rng = payload.get("rng", {})

                # Torch CPU RNG — must be a CPU ByteTensor
                if "torch" in rng and rng["torch"] is not None:
                    state = rng["torch"]
                    if isinstance(state, torch.Tensor):
                        state = state.detach().to(torch.device("cpu"), dtype=torch.uint8)
                    torch.set_rng_state(state)

                # NumPy & Python RNG
                if "numpy" in rng and rng["numpy"] is not None:
                    np.random.set_state(rng["numpy"])
                if "python" in rng and rng["python"] is not None:
                    random.setstate(rng["python"])

                # CUDA RNG (only if CUDA is available) — move per-device states back onto CUDA
                if torch.cuda.is_available() and rng.get("torch_cuda") is not None:
                    try:
                        states = rng["torch_cuda"]
                        if isinstance(states, (list, tuple)):
                            states = [
                                (s.detach().to(f"cuda:{i}", dtype=torch.uint8) if isinstance(s, torch.Tensor) else s)
                                for i, s in enumerate(states)
                            ]
                        torch.cuda.set_rng_state_all(states)
                    except Exception:
                        # Non-fatal; continue even if CUDA RNG restore fails
                        pass
                print(f"Resumed from checkpoint '{latest}' (epoch {saved_epoch}). Continuing at epoch {start_epoch}.")
            except Exception as e:
                print(f"Warning: failed to resume from checkpoint '{latest}': {e}")

    def _save_epoch_ckpt(ep: int):
        if ckpt_path_obj:
            path = save_checkpoint(ckpt_path_obj, ep, model, opt,
                                   heads=heads, artifacts=art, win_cfg=win_cfg,
                                   feature_dim=X_all.shape[1], device_str=device_str or get_default_device(),
                                   keep_last=keep_last)
            print(f"Saved checkpoint: {path}")

    try:
        for ep in range(start_epoch, epochs + 1):
            model.train()
            running: Dict[str, float] = {"L": 0.0, "La": 0.0, "Ls": 0.0, "Lb": 0.0}
            n_steps = 0
            for b in dl_tr:
                x = b["x"].to(device)
                ya = b["y_action"].to(device)
                yb = b["y_buttons"].to(device)
                ystick_idx = b["y_stick_idx"].to(device)  # (B,T-1), cluster indices
                ystick_xy = b["stick_xy"].to(device)  # (B,T-1,2) in [0,1]
                pa = b["prev_action"].to(device)
                ps = b["prev_stick_idx"].to(device)

                opt.zero_grad(set_to_none=True)
                out = model(x, prev_action_idx=pa, prev_stick_idx=ps,
                            stick_teacher=ystick_idx)  # teacher-forced for buttons
                loss, logs = loss_fn(out, ya, ystick_idx, ystick_xy, yb,
                                     pos_weight_btn, LossWeights(), stick_class_weights)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                for k in running:
                    running[k] += logs[k]
                n_steps += 1

            for k in running:
                running[k] /= max(n_steps, 1)

            eval_metrics = evaluate(
                model, dl_va, device,
                confusion_top_classes=8,
                confusion_top_confusers=5,
                action_label_fn=make_action_label_fn(art)  # ← fix: use art, not artifacts
            )
            print(pretty_eval(ep, running, eval_metrics))
            if "action_confusions_report" in eval_metrics:
                print(eval_metrics["action_confusions_report"])
            if "stick_confusions_report" in eval_metrics:
                print(eval_metrics["stick_confusions_report"])

            # Compose a validation objective to MINIMIZE (default: sum of NLLs)
            val_obj = float(
                eval_metrics.get("nll_action", 0.0)
                + eval_metrics.get("nll_stick", 0.0)
            )

            # Plateau step once per epoch AFTER validation
            lr_before = float(opt.param_groups[0]["lr"]) if opt.param_groups else float(lr)
            if plateau_scheduler is not None:
                plateau_scheduler.step(val_obj)
            lr_after = float(opt.param_groups[0]["lr"]) if opt.param_groups else float(lr)
            print(f"LR (epoch {ep:02d}): before={lr_before:.6g}  after={lr_after:.6g}  monitor=val_obj({val_obj:.6g})")

            print("\n" + "=" * 50)
            _save_epoch_ckpt(ep)
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected — saving 'latest' checkpoint for safe resume...")
        _save_epoch_ckpt(max(start_epoch, ep if 'ep' in locals() else 1))
        raise
    print("Training finished. Now finding optimal thresholds on validation set.")
    print("=" * 50)

    # Step 1: Ensure model is in evaluation mode
    model.eval()

    # Step 2: Collect all logits and labels from the validation set
    all_val_logits: List[Tensor] = []
    all_val_labels: List[Tensor] = []

    with torch.no_grad():
        for b in dl_va:
            # Get model predictions (we only need closed-loop button logits)
            out = model(
                x=b["x"].to(device),
                prev_action_idx=b["prev_action"].to(device),
                prev_stick_idx=b["prev_stick_idx"].to(device),
                stick_teacher=None  # Use closed-loop for real-world performance
            )
            # Flatten from (B, T, 5) to (B*T, 5) and append
            all_val_logits.append(out["button_logits"].reshape(-1, 5))
            all_val_labels.append(b["y_buttons"].reshape(-1, 5))

    # Concatenate all batches into single large tensors
    final_logits = torch.cat(all_val_logits, dim=0)
    final_labels = torch.cat(all_val_labels, dim=0)

    # Step 3: Call our new function to find the thresholds
    optimal_thresholds = find_optimal_thresholds(final_logits, final_labels)

    print("\nOptimal thresholds found:")
    print(optimal_thresholds)
    print("\nThese values can now be used in your deployed model to maximize F1 score.")
    print("=" * 50)

    # Optional: export a runtime JSON with everything needed for inference
    if export_json_path:
        try:
            save_runtime_json(export_json_path, model, art, optimal_thresholds, stick_r0=0.15)
            print(f"\nSaved runtime JSON to: {export_json_path}")
        except Exception as e:
            print(f"Failed to save runtime JSON to {export_json_path}: {e}")


def pretty_eval(ep: int, running: Dict[str, float], m: Dict[str, float]) -> str:
    lbls = ["A", "B", "Z", "JUMP", "SHIELD"]

    head = (
        f"[Epoch {ep:02d}] "
        f"L={running['L']:.4f} (A={running['La']:.4f}, S={running['Ls']:.4f}, B={running['Lb']:.4f})"
        f"  |  tokens={int(m['n_tokens'])}"
    )

    action = (
        f"Action: acc@1={m['acc_action']:.3f}  acc@5={m['acc_action_top5']:.3f}  "
        f"NLL={m['nll_action']:.3f}  ppl={m['ppl_action']:.2f}  "
        f"[baselines → maj={m['acc_action_majority']:.3f}, lag1={m['acc_action_lag1']:.3f}]"
    )

    stick = (
        f"Stick:  acc@1={m['acc_stick']:.3f}  NLL={m['nll_stick']:.3f}  ppl={m['ppl_stick']:.2f}  "
        f"[baselines → maj={m['acc_stick_majority']:.3f}, center={m['acc_stick_center_baseline']:.3f}, "
        f"lag1={m['acc_stick_lag1']:.3f}]  "
        f"center(true={m['stick_true_center_rate']:.3f}, pred={m['stick_pred_center_rate']:.3f})"
    )

    buttons = (
        "Buttons (closed-loop @0.5): "
        f"P={m['btn_prec_micro_cl_05']:.3f}  R={m['btn_rec_micro_cl_05']:.3f}  F1μ={m['btn_f1_micro_cl_05']:.3f}  "
        f"F1̄={m['btn_f1_macro_cl_05']:.3f}\n"
        "        (teacher-forced @0.5): "
        f"P={m['btn_prec_micro_tf_05']:.3f}  R={m['btn_rec_micro_tf_05']:.3f}  F1μ={m['btn_f1_micro_tf_05']:.3f}  "
        f"F1̄={m['btn_f1_macro_tf_05']:.3f}"
    )

    prauc = f"PR-AUC (closed-loop): micro={m['btn_ap_micro_cl']:.3f}  macro={m['btn_ap_macro_cl']:.3f}"

    # Per-label table
    header = "Label   prev   ppr@0.5   F1_cl@0.5   thr*   F1_cl@thr*   F1_tf@0.5   AP_cl"
    rows = []
    for name in lbls:
        rows.append(
            f"{name:<6} "
            f"{m[f'btn_prev_{name}']:.3f}   "
            f"{m[f'btn_ppr_cl_05_{name}']:.3f}     "
            f"{m[f'btn_f1_cl_05_{name}']:.3f}     "
            f"{m[f'btn_thr_tuned_{name}']:.2f}   "
            f"{m[f'btn_f1_cl_tuned_{name}']:.3f}      "
            f"{m[f'btn_f1_tf_05_{name}']:.3f}   "
            f"{m[f'btn_ap_cl_{name}']:.3f}"
        )
    table = "Per-label (closed-loop unless noted):\n" + header + "\n" + "\n".join(rows)

    return "\n".join([head, action, stick, buttons, prauc, table])


def choose_replays(dirpath: Path, k: int) -> list[Path]:
    """Return up to k .slp files from dirpath, sampled uniformly at random."""
    SEED = 100
    all_paths = [p for p in dirpath.glob("*.slp") if p.is_file()]
    if not all_paths:
        raise FileNotFoundError(f"No .slp files found in {dirpath}")
    if SEED is not None:
        random.seed(SEED)
    if k >= len(all_paths):
        return all_paths
    return random.sample(all_paths, k)


def process_path(path: Path) -> Sequence[object]:
    """Wrapper so the executor only receives picklable args."""
    return process_one_replay(str(path))


if __name__ == "__main__":
    base_dir = Path("/Users/eppie/Downloads/replays_sorted/FOX_vs_FOX")
    selected = choose_replays(base_dir, 1)
    print(f"Selected {len(selected)} replay(s) from {base_dir}")

    rows: list[Row] = []
    # For CPU-bound work, processes are preferable to threads.
    max_workers = os.cpu_count() or 4
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        future_to_path = {ex.submit(process_path, p): p for p in selected}
        for fut in as_completed(future_to_path):
            p = future_to_path[fut]
            try:
                result = fut.result()  # Sequence[object]
            except Exception as e:  # Keep going on errors
                print(f"ERROR processing {p.name}: {e!r}")
                continue
            rows.extend(result)
            print(f"Processed {p.name}, cumulative rows={len(rows)}")

    print(f"Done. Total rows: {len(rows)}")

    # rows = process_one_replay("/Users/eppie/PycharmProjects/new-melee-ai/test/test.slp")
    train_gru_on_replay(rows, T=256, stride=1, epochs=96, batch_size=512,
                        export_json_path="/Users/eppie/PycharmProjects/new-melee-ai/small_gru_on_replay.json")
