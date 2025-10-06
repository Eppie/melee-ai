from __future__ import annotations

import math
import time
from pathlib import Path
from textwrap import indent
from typing import Dict
from typing import List, Optional, Tuple

import numpy as np
import torch
from tensordict import TensorDict
from torch import GradScaler
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from config import get_config, init_config
from controller_quantization import quantize_targets
from gpt import GPTv7
from preprocess import (
    FOX_STICK_64,
    C_STICK_XY_CLUSTER_CENTERS_V0_1,
)
from window_dataset import WindowDataset, make_dataloader


def _bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024: return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def cosine_lr_schedule(step: int, total_steps: int, base_lr: float, warmup: int = 0) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# -----------------------------
# Column mapping / batch adapter
# -----------------------------

_CONTROLLER_KEYS = {
    "main": ("main_stick_x", "main_stick_y"),
    "c": ("c_stick_x", "c_stick_y"),
    "buttons": ("button_a", "button_b", "button_xy", "button_z", "button_lr"),
    "shoulder": ("shoulder_analog",),
}
_BUTTON_PRETTY = {
    "button_a": "A",
    "button_b": "B",
    "button_xy": "X/Y",
    "button_z": "Z",
    "button_lr": "L/R",
}
_CE_WEIGHT_CLAMP = 10.0
_CE_WEIGHT_MIN = 0.1
_POS_WEIGHT_CLAMP = 10.0


def _compute_ce_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    device = labels.device
    try:
        counts = torch.bincount(labels, minlength=num_classes)
    except RuntimeError:
        counts = torch.bincount(labels.cpu(), minlength=num_classes).to(device)
    counts = counts.float()
    counts = torch.clamp(counts, min=1.0)
    weights = counts.sum() / (counts * num_classes)
    weights = torch.clamp(weights, min=_CE_WEIGHT_MIN, max=_CE_WEIGHT_CLAMP)
    return weights


def _compute_pos_weights(targets: torch.Tensor) -> torch.Tensor:
    flat = targets.reshape(-1, targets.shape[-1])  # [N, K]
    pos = flat.sum(dim=0)
    total = flat.shape[0]
    neg = total - pos
    pos_weight = neg / torch.clamp(pos, min=1.0)
    pos_weight = torch.clamp(pos_weight, min=1.0, max=_POS_WEIGHT_CLAMP)
    return pos_weight.to(targets.device)


_CATEGORICAL = ("character", "action")


class ColumnMap:
    """
    Resolves feature/target column names -> indices once, then reuses.
    Produces:
      - stage_idx (int)
      - ego/opponent categorical idx map
      - gamestate_idxs: list[int] (all non-controller, non-stage numeric cols)
      - controller_idxs: list[int] (both players)
      - Y sub-slices for targets
    """

    def __init__(self, ds: WindowDataset):
        # feature names in the same order as X columns returned by Dataset
        self.feat_names: List[str] = getattr(ds, "_feature_names_sel", ds.index.feature_names)
        self.targ_names: List[str] = getattr(ds, "_target_names_sel", ds.index.target_names)

        name2idx = {n: i for i, n in enumerate(self.feat_names)}

        # singletons
        self.stage_idx = name2idx["stage"]

        # categorical (int indices)
        self.ego_char_idx = name2idx["p1_character"]
        self.opp_char_idx = name2idx["p2_character"]
        self.ego_action_idx = name2idx["p1_action"]
        self.opp_action_idx = name2idx["p2_action"]

        # controller numeric (both players)
        self.controller_idxs: List[int] = []
        for prefix in ("p1_", "p2_"):
            for k in _CONTROLLER_KEYS["main"] + _CONTROLLER_KEYS["c"] + _CONTROLLER_KEYS["buttons"] + _CONTROLLER_KEYS[
                "shoulder"]:
                self.controller_idxs.append(name2idx[f"{prefix}{k}"])

        # gamestate numeric = everything else (numeric) except stage + controller + categorical
        excluded = set([self.stage_idx, self.ego_char_idx, self.opp_char_idx,
                        self.ego_action_idx, self.opp_action_idx] + self.controller_idxs)
        self.gamestate_idxs = [i for i, n in enumerate(self.feat_names) if i not in excluded]

        # ---- target sub-slices (from Y) ----
        t2idx = {n: i for i, n in enumerate(self.targ_names)}

        def ti(name: str) -> int:
            if name not in t2idx:
                raise KeyError(f"Target '{name}' not found; Y columns: {self.targ_names}")
            return t2idx[name]

        # main stick (ego) next frame
        self.y_main = (ti("p1_main_stick_x"), ti("p1_main_stick_y"))
        # c-stick
        self.y_c = (ti("p1_c_stick_x"), ti("p1_c_stick_y"))
        # buttons (multi-label)
        self.y_buttons = [ti(f"p1_{k}") for k in _CONTROLLER_KEYS["buttons"]]
        # shoulder
        self.y_shoulder = ti("p1_shoulder_analog") if "p1_shoulder_analog" in t2idx else None


def build_inputs_for_gptv7(batch_X: torch.FloatTensor, colmap: ColumnMap) -> TensorDict:
    """
    batch_X: [B, L, F] float32 (features of current frame)
    Returns TensorDict with the keys GPTv7._embed_inputs expects.
    """
    B, L, _ = batch_X.shape

    # Categoricals back to long indices
    stage = batch_X[..., colmap.stage_idx].to(torch.long).unsqueeze(-1)  # [B,L,1]
    ego_character = batch_X[..., colmap.ego_char_idx].to(torch.long).unsqueeze(-1)
    opp_character = batch_X[..., colmap.opp_char_idx].to(torch.long).unsqueeze(-1)
    ego_action = batch_X[..., colmap.ego_action_idx].to(torch.long).unsqueeze(-1)
    opp_action = batch_X[..., colmap.opp_action_idx].to(torch.long).unsqueeze(-1)

    gamestate = batch_X[..., colmap.gamestate_idxs]  # [B,L,Gg]
    controller = batch_X[..., colmap.controller_idxs]  # [B,L,Gc]

    return TensorDict(
        {
            "stage": stage,
            "ego_character": ego_character,
            "opponent_character": opp_character,
            "ego_action": ego_action,
            "opponent_action": opp_action,
            "gamestate": gamestate,
            "controller": controller,
        },
        batch_size=(B, L),
    )


# -----------------------------
# Running metrics (per-epoch)
# -----------------------------

def _safe_div(n: float, d: float) -> float:
    return float(n) / float(d) if d else 0.0


def _safe_bincount(x: torch.Tensor, minlength: int, device: torch.device) -> torch.Tensor:
    if x.numel() == 0:
        return torch.zeros(minlength, device=device, dtype=torch.float32)
    if x.dtype != torch.long:
        x = x.to(torch.long)
    if x.device.type == device.type:
        try:
            counts = torch.bincount(x, minlength=minlength)
        except RuntimeError:
            counts = torch.bincount(x.cpu(), minlength=minlength).to(device)
    else:
        try:
            counts = torch.bincount(x.to(device), minlength=minlength)
        except RuntimeError:
            counts = torch.bincount(x.cpu(), minlength=minlength).to(device)
    return counts.to(device=device, dtype=torch.float32)


class RunningMetrics:
    """
    Tracks running metrics for:
      - main stick (classification)
      - c-stick (classification)
      - buttons (multi-label)
      - shoulder (classification)
    Includes baselines:
      - random
      - majority (running)
      - repeat-last (ignoring first timestep)
    Also accumulates confusion matrices for main and c.
    """

    def __init__(self, K_main: int, K_c: int, K_buttons: int, K_shoulder: int = 0,
                 device: torch.device | None = None) -> None:
        self.device = device or torch.device("cpu")
        dtype = torch.float32

        # Main
        self.K_main = K_main
        self.main_correct = torch.zeros((), device=self.device, dtype=dtype)
        self.main_total = torch.zeros((), device=self.device, dtype=dtype)
        self.main_confusion = torch.zeros((K_main, K_main), device=self.device, dtype=dtype)
        self.main_label_counts = torch.zeros(K_main, device=self.device, dtype=dtype)
        self.main_rand_correct = torch.zeros((), device=self.device, dtype=dtype)
        self.main_maj_correct = torch.zeros((), device=self.device, dtype=dtype)
        self.main_rep_correct = torch.zeros((), device=self.device, dtype=dtype)
        self.main_rep_total = torch.zeros((), device=self.device, dtype=dtype)

        # C-stick
        self.K_c = K_c
        self.c_correct = torch.zeros((), device=self.device, dtype=dtype)
        self.c_total = torch.zeros((), device=self.device, dtype=dtype)
        self.c_confusion = torch.zeros((K_c, K_c), device=self.device, dtype=dtype)
        self.c_label_counts = torch.zeros(K_c, device=self.device, dtype=dtype)
        self.c_rand_correct = torch.zeros((), device=self.device, dtype=dtype)
        self.c_maj_correct = torch.zeros((), device=self.device, dtype=dtype)
        self.c_rep_correct = torch.zeros((), device=self.device, dtype=dtype)
        self.c_rep_total = torch.zeros((), device=self.device, dtype=dtype)

        # Buttons (multi-label)
        self.K_buttons = K_buttons
        self.btn_tp = torch.zeros(K_buttons, device=self.device, dtype=dtype)
        self.btn_fp = torch.zeros(K_buttons, device=self.device, dtype=dtype)
        self.btn_fn = torch.zeros(K_buttons, device=self.device, dtype=dtype)
        self.btn_exact_match_correct = torch.zeros((), device=self.device, dtype=dtype)
        self.btn_total = torch.zeros((), device=self.device, dtype=dtype)
        self.btn_rand_tp = torch.zeros(K_buttons, device=self.device, dtype=dtype)
        self.btn_rand_fp = torch.zeros(K_buttons, device=self.device, dtype=dtype)
        self.btn_rand_fn = torch.zeros(K_buttons, device=self.device, dtype=dtype)
        self.btn_rand_exact = torch.zeros((), device=self.device, dtype=dtype)

        self.btn_maj_tp = torch.zeros(K_buttons, device=self.device, dtype=dtype)
        self.btn_maj_fp = torch.zeros(K_buttons, device=self.device, dtype=dtype)
        self.btn_maj_fn = torch.zeros(K_buttons, device=self.device, dtype=dtype)
        self.btn_maj_exact = torch.zeros((), device=self.device, dtype=dtype)
        self.btn_pos_counts = torch.zeros(K_buttons, device=self.device, dtype=dtype)

        self.btn_rep_tp = torch.zeros(K_buttons, device=self.device, dtype=dtype)
        self.btn_rep_fp = torch.zeros(K_buttons, device=self.device, dtype=dtype)
        self.btn_rep_fn = torch.zeros(K_buttons, device=self.device, dtype=dtype)
        self.btn_rep_exact = torch.zeros((), device=self.device, dtype=dtype)
        self.btn_rep_total = torch.zeros((), device=self.device, dtype=dtype)

        # Shoulder
        self.K_shoulder = K_shoulder
        self.shoulder_correct = torch.zeros((), device=self.device, dtype=dtype)
        self.shoulder_total = torch.zeros((), device=self.device, dtype=dtype)
        self.shoulder_rand_correct = torch.zeros((), device=self.device, dtype=dtype)
        self.shoulder_maj_correct = torch.zeros((), device=self.device, dtype=dtype)
        self.shoulder_label_counts = (
            torch.zeros(K_shoulder, device=self.device, dtype=dtype) if K_shoulder else None
        )

    # ------- helpers -------

    def _majority_label(self, counts: torch.Tensor) -> int:
        total = counts.sum()
        if total.detach().cpu().item() == 0:
            return 0
        return int(torch.argmax(counts).detach().cpu().item())

    # ------- main stick -------

    def update_main(
            self,
            pred_idx: torch.Tensor,  # [N]
            true_idx: torch.Tensor,  # [N]
            rand_idx: torch.Tensor,  # [N]
            maj_before: int,
            repeat_idx: torch.Tensor,  # [N]
            repeat_mask: torch.Tensor,  # [N] bool
    ) -> None:
        with torch.no_grad():
            self.main_total.add_(float(true_idx.numel()))

            self.main_correct.add_((pred_idx == true_idx).float().sum())

            K = self.K_main
            flat = true_idx * K + pred_idx
            cm = _safe_bincount(flat, K * K, self.device).view(K, K)
            self.main_confusion.add_(cm)

            self.main_rand_correct.add_((rand_idx == true_idx).float().sum())

            maj_tensor = torch.full_like(true_idx, maj_before)
            self.main_maj_correct.add_((true_idx == maj_tensor).float().sum())

            rep_correct = (repeat_idx[repeat_mask] == true_idx[repeat_mask]).float().sum()
            self.main_rep_correct.add_(rep_correct)
            self.main_rep_total.add_(repeat_mask.float().sum())

            self.main_label_counts.add_(_safe_bincount(true_idx, K, self.device))

    # ------- c-stick -------

    def update_c(
            self,
            pred_idx: torch.Tensor,
            true_idx: torch.Tensor,
            rand_idx: torch.Tensor,
            maj_before: int,
            repeat_idx: torch.Tensor,
            repeat_mask: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            self.c_total.add_(float(true_idx.numel()))

            self.c_correct.add_((pred_idx == true_idx).float().sum())

            K = self.K_c
            flat = true_idx * K + pred_idx
            cm = _safe_bincount(flat, K * K, self.device).view(K, K)
            self.c_confusion.add_(cm)

            self.c_rand_correct.add_((rand_idx == true_idx).float().sum())

            maj_tensor = torch.full_like(true_idx, maj_before)
            self.c_maj_correct.add_((true_idx == maj_tensor).float().sum())

            rep_correct = (repeat_idx[repeat_mask] == true_idx[repeat_mask]).float().sum()
            self.c_rep_correct.add_(rep_correct)
            self.c_rep_total.add_(repeat_mask.float().sum())

            self.c_label_counts.add_(_safe_bincount(true_idx, K, self.device))

    # ------- buttons (multi-label) -------

    def update_buttons(
            self,
            logits: torch.Tensor,  # [B,L,Kb]
            true: torch.Tensor,  # [B,L,Kb] in {0,1}
            probs: torch.Tensor | None = None,
    ) -> None:
        with torch.no_grad():
            B, L, Kb = true.shape
            if probs is None:
                probs = torch.sigmoid(logits)
            else:
                probs = probs.detach()
            pred = (probs > 0.5).to(true.dtype)

            self.btn_exact_match_correct.add_((pred == true).all(dim=-1).float().sum())
            self.btn_total.add_(float(B * L))

            pred_b = pred.bool()
            true_b = true.bool()
            tp = (pred_b & true_b).float().sum(dim=(0, 1))
            fp = (pred_b & (~true_b)).float().sum(dim=(0, 1))
            fn = ((~pred_b) & true_b).float().sum(dim=(0, 1))

            self.btn_tp.add_(tp)
            self.btn_fp.add_(fp)
            self.btn_fn.add_(fn)

            self.btn_pos_counts.add_(true.float().sum(dim=(0, 1)))

            rand = (torch.rand_like(true) > 0.5).to(true.dtype)
            rb = rand.bool()
            self.btn_rand_tp.add_((rb & true_b).float().sum(dim=(0, 1)))
            self.btn_rand_fp.add_((rb & (~true_b)).float().sum(dim=(0, 1)))
            self.btn_rand_fn.add_(((~rb) & true_b).float().sum(dim=(0, 1)))
            self.btn_rand_exact.add_((rand == true).all(dim=-1).float().sum())

            totals_seen = torch.clamp(self.btn_total - float(B * L), min=1.0)
            prev = self.btn_pos_counts
            maj_vec = (prev * 2.0 >= totals_seen).to(true.dtype)
            maj = maj_vec.view(1, 1, Kb).expand(B, L, Kb)

            mb = maj.bool()
            self.btn_maj_tp.add_((mb & true_b).float().sum(dim=(0, 1)))
            self.btn_maj_fp.add_((mb & (~true_b)).float().sum(dim=(0, 1)))
            self.btn_maj_fn.add_(((~mb) & true_b).float().sum(dim=(0, 1)))
            self.btn_maj_exact.add_((maj == true).all(dim=-1).float().sum())

            if L > 1:
                rep = torch.zeros_like(true)
                rep[:, 1:, :] = true[:, :-1, :]
                rb2 = rep.bool()
                mask = torch.zeros((B, L), dtype=torch.float32, device=true.device)
                mask[:, 1:] = 1.0
                m2 = mask.unsqueeze(-1).expand_as(true)
                mask_bool = m2.bool()
                self.btn_rep_tp.add_((rb2 & true_b & mask_bool).float().sum(dim=(0, 1)))
                self.btn_rep_fp.add_((rb2 & (~true_b) & mask_bool).float().sum(dim=(0, 1)))
                self.btn_rep_fn.add_(((~rb2) & true_b & mask_bool).float().sum(dim=(0, 1)))
                self.btn_rep_exact.add_(((rep == true) & mask_bool).view(B, L, -1).all(dim=-1).float().sum())
                self.btn_rep_total.add_(mask.sum())

    def update_shoulder(
            self,
            logits: Optional[torch.Tensor],  # [B,L,Ks] or None
            true_idx: Optional[torch.Tensor],  # [B,L] or None
            rand_idx: Optional[torch.Tensor],  # [B,L] or None
            maj_before: Optional[int],  # int or None
    ) -> None:
        if logits is None or true_idx is None or self.K_shoulder == 0:
            return
        with torch.no_grad():
            pred_idx = logits.argmax(dim=-1).reshape(-1)
            true_flat = true_idx.reshape(-1)

            self.shoulder_total.add_(float(true_flat.numel()))
            self.shoulder_correct.add_((pred_idx == true_flat).float().sum())
            if rand_idx is not None:
                rand_flat = rand_idx.reshape(-1)
                self.shoulder_rand_correct.add_((rand_flat == true_flat).float().sum())
            if maj_before is not None:
                maj_tensor = torch.full_like(true_flat, maj_before)
                self.shoulder_maj_correct.add_((true_flat == maj_tensor).float().sum())
            if self.shoulder_label_counts is not None:
                self.shoulder_label_counts.add_(_safe_bincount(true_flat, self.K_shoulder, self.device))

    # ------- summaries -------

    def _btn_prf(self, tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor) -> Tuple[float, float, float, float]:
        # micro
        TP = tp.sum().item()
        FP = fp.sum().item()
        FN = fn.sum().item()
        prec_micro = _safe_div(TP, TP + FP)
        rec_micro = _safe_div(TP, TP + FN)
        f1_micro = _safe_div(2 * prec_micro * rec_micro, prec_micro + rec_micro)
        # macro
        prec_c = (tp / (tp + fp).clamp_min(1)).float()
        rec_c = (tp / (tp + fn).clamp_min(1)).float()
        f1_c = (2 * prec_c * rec_c / (prec_c + rec_c).clamp_min(1e-12)).nan_to_num(0.0)
        f1_macro = float(f1_c.mean().item())
        return float(prec_micro), float(rec_micro), float(f1_micro), f1_macro

    def summary(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        # main/c acc
        out["acc_main"] = _safe_div(self.main_correct, self.main_total)
        out["acc_main_rand"] = _safe_div(self.main_rand_correct, self.main_total)
        out["acc_main_maj"] = _safe_div(self.main_maj_correct, self.main_total)
        out["acc_main_rep"] = _safe_div(self.main_rep_correct, self.main_rep_total)

        out["acc_c"] = _safe_div(self.c_correct, self.c_total)
        out["acc_c_rand"] = _safe_div(self.c_rand_correct, self.c_total)
        out["acc_c_maj"] = _safe_div(self.c_maj_correct, self.c_total)
        out["acc_c_rep"] = _safe_div(self.c_rep_correct, self.c_rep_total)

        # buttons
        pm, rm, f1m, f1macro = self._btn_prf(self.btn_tp, self.btn_fp, self.btn_fn)
        out["btn_em"] = _safe_div(self.btn_exact_match_correct, self.btn_total)
        out["btn_prec_micro"] = pm
        out["btn_rec_micro"] = rm
        out["btn_f1_micro"] = f1m
        out["btn_f1_macro"] = f1macro

        # buttons baselines
        pmr, rmr, f1mr, f1mjr = self._btn_prf(self.btn_rand_tp, self.btn_rand_fp, self.btn_rand_fn)
        out["btn_em_rand"] = _safe_div(self.btn_rand_exact, self.btn_total)
        out["btn_f1_micro_rand"] = pmr if pmr == pmr else 0.0  # NaN guard
        out["btn_f1_macro_rand"] = f1mr

        pmm, rmm, f1mm, f1mmm = self._btn_prf(self.btn_maj_tp, self.btn_maj_fp, self.btn_maj_fn)
        out["btn_em_maj"] = _safe_div(self.btn_maj_exact, self.btn_total)
        out["btn_f1_micro_maj"] = pmm if pmm == pmm else 0.0
        out["btn_f1_macro_maj"] = f1mm

        pmx, rmx, f1mx, f1mmx = self._btn_prf(self.btn_rep_tp, self.btn_rep_fp, self.btn_rep_fn)
        out["btn_em_rep"] = _safe_div(self.btn_rep_exact, self.btn_rep_total)
        out["btn_f1_micro_rep"] = f1mx
        out["btn_f1_macro_rep"] = f1mmx

        # shoulder
        if self.K_shoulder:
            out["acc_shoulder"] = _safe_div(self.shoulder_correct, self.shoulder_total)
            out["acc_shoulder_rand"] = _safe_div(self.shoulder_rand_correct, self.shoulder_total)
            out["acc_shoulder_maj"] = _safe_div(self.shoulder_maj_correct, self.shoulder_total)

        return out

    def short_str(self) -> str:
        s = self.summary()
        return (
                f" | main acc {s['acc_main']:.3f} (rand {s['acc_main_rand']:.3f}, maj {s['acc_main_maj']:.3f}, rep {s['acc_main_rep']:.3f})"
                f" | c acc {s['acc_c']:.3f} (rand {s['acc_c_rand']:.3f}, maj {s['acc_c_maj']:.3f}, rep {s['acc_c_rep']:.3f})"
                f" | btn EM {s['btn_em']:.3f} F1μ {s['btn_f1_micro']:.3f} (rand {s['btn_f1_micro_rand']:.3f}, maj {s['btn_f1_micro_maj']:.3f}, rep {s['btn_f1_micro_rep']:.3f})"
                + (f" | shoulder acc {s['acc_shoulder']:.3f}" if 'acc_shoulder' in s else "")
        )


# -----------------------------
# Pretty-print helpers for metrics (per-batch)
# -----------------------------

# MPS-safe majority helper (no torch.mode)
def _majority_flat(x: torch.Tensor) -> int:
    """Return majority label from a 1D integer tensor using CPU bincount (MPS-safe)."""
    if x.numel() == 0:
        return 0
    x_cpu = x.detach().to(torch.int64).cpu()
    counts = torch.bincount(x_cpu)
    return int(torch.argmax(counts).item())


def _confusion_from_flat(true_flat: torch.Tensor, pred_flat: torch.Tensor, K: int) -> torch.Tensor:
    """Return confusion matrix [K,K] on CPU from flattened integer labels."""
    tf = true_flat.to(torch.int64)
    pf = pred_flat.to(torch.int64)
    cm = torch.bincount(tf * K + pf, minlength=K * K).view(K, K)
    return cm.cpu()


def _top_confusions(cm: torch.Tensor, k: int = 8) -> List[Tuple[int, int, int, float]]:
    """Return top-k off-diagonal confusions as (true, pred, count, pct_of_offdiag)."""
    cm_np = cm.numpy()
    off = cm_np.copy()
    np.fill_diagonal(off, 0)
    total_off = off.sum()
    if total_off <= 0:
        return []
    flat = off.ravel()
    idx = np.argpartition(flat, -k)[-k:]
    idx = idx[np.argsort(flat[idx])[::-1]]
    out: List[Tuple[int, int, int, float]] = []
    W = cm_np.shape[1]
    for f in idx:
        cnt = int(flat[f])
        if cnt <= 0:
            continue
        i = int(f // W)
        j = int(f % W)
        pct = 100.0 * cnt / float(total_off)
        out.append((i, j, cnt, pct))
    return out


def _format_confusion_small(cm: torch.Tensor, max_size: int = 12, title: str | None = None) -> str:
    """
    If K <= max_size, render full matrix with row sums and diagonal percentages.
    Otherwise, print top off-diagonal confusions.
    """
    K = cm.shape[0]
    if title is None:
        title = "confusion"
    if K > max_size:
        tops = _top_confusions(cm, k=10)
        if not tops:
            return f"{title}: (no confusions)"
        lines = [f"{title}: top confusions (true->pred: count, %offdiag)"]
        lines += [f"  {t:02d}->{p:02d}: {c} ({pct:.1f}%)" for t, p, c, pct in tops]
        return "\n".join(lines)

    arr = cm.numpy()
    row_sums_1d = arr.sum(axis=1)  # shape (K,)
    diag_vals = np.diag(arr).astype(float)
    # Safe divide: out=0 where row sum == 0
    with np.errstate(divide='ignore', invalid='ignore'):
        diag_pct = np.divide(diag_vals * 100.0, row_sums_1d, out=np.zeros_like(diag_vals, dtype=float),
                             where=row_sums_1d > 0)

    header = "     " + " ".join([f"{j:>4d}" for j in range(K)]) + " | sum"
    lines = [f"{title}: full {K}x{K}", header]
    for i in range(K):
        row = " ".join([f"{int(v):>4d}" for v in arr[i]])
        lines.append(f"{i:>3d}: {row} | {int(row_sums_1d[i]):>4d}")
    lines.append("diag% per row: " + " ".join([f"{p:>5.1f}" for p in diag_pct]))
    return "\n".join(lines)


def _multilabel_prf(true: torch.Tensor, pred: torch.Tensor) -> Tuple[float, float, float, float, float]:
    """
    Compute (EM, precision_micro, recall_micro, f1_micro, f1_macro) for multi-label predictions.
    Accepts tensors of shape [B, L, K] *or* [N, K]; returns floats.
    """
    if true.dim() == 2:
        # treat as [N, K]
        true_ = true.unsqueeze(0)
        pred_ = pred.unsqueeze(0)
        B, L, K = 1, true.shape[0], true.shape[1]
    else:
        true_ = true
        pred_ = pred
        B, L, K = true.shape

    t = true_.bool()
    p = pred_.bool()
    tp = (t & p).sum().item()
    fp = ((~t) & p).sum().item()
    fn = (t & (~p)).sum().item()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    tp_c = (t & p).sum(dim=(0, 1)).cpu().numpy()
    fp_c = ((~t) & p).sum(dim=(0, 1)).cpu().numpy()
    fn_c = (t & (~p)).sum(dim=(0, 1)).cpu().numpy()
    f1_c = []
    for a, b, c in zip(tp_c, fp_c, fn_c):
        pr = a / (a + b) if (a + b) > 0 else 0.0
        rc = a / (a + c) if (a + c) > 0 else 0.0
        f1_c.append(2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.0)
    f1_macro = float(np.mean(f1_c) if len(f1_c) else 0.0)

    em = float((pred_ == true_).all(dim=-1).float().mean().item())
    return em, float(prec), float(rec), float(f1), f1_macro


# -----------------------------
# Training step / loop
# -----------------------------

def train_loop(
        model: GPTv7,
) -> None:
    device = torch.device("mps")
    model = model.to(device)
    config = get_config()
    # Build loader + sampler
    loader, ds, sampler = make_dataloader()

    # Column map built from dataset metadata (only once)
    colmap = ColumnMap(ds)

    # Optimizer & (optional) simple cosine LR
    opt = torch.optim.AdamW(model.parameters(), lr=config.train.lr, betas=config.train.betas,
                            weight_decay=config.train.weight_decay)
    scaler = GradScaler()

    steps_per_epoch = config.train.steps_per_epoch or math.ceil(len(loader))
    total_steps = config.train.max_steps or (config.train.epochs * steps_per_epoch)
    global_step = 0

    out_dir = Path(config.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Main epochs
    for epoch in range(config.train.epochs):
        # Important for samplers in distributed setups
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)  # ensures different shuffles per epoch in DDP
        model.train()

        metrics = RunningMetrics(
            K_main=len(FOX_STICK_64),
            K_c=len(C_STICK_XY_CLUSTER_CENTERS_V0_1),
            K_buttons=len(_CONTROLLER_KEYS["buttons"]),
            K_shoulder=(len(config.train.shoulder_centers) if config.train.shoulder_centers else 0),
            device=device,
        )

        epoch_loss = 0.0
        t0 = time.time()

        max_iters = config.train.steps_per_epoch
        for it, batch in enumerate(loader):
            if max_iters is not None and it >= max_iters:
                break
            if config.train.max_steps and global_step >= config.train.max_steps:
                break

            # Move to device
            X: torch.Tensor = batch["X"].to(device, non_blocking=True)  # [B,L,F]
            Y: torch.Tensor = batch["Y"].to(device, non_blocking=True)  # [B,L,Yd]

            # Build model inputs & target labels
            inputs_td = build_inputs_for_gptv7(X, colmap)
            target_info = quantize_targets(Y, colmap, config.train.shoulder_centers)

            pred: TensorDict = model(inputs_td)  # keys: buttons, main_stick, c_stick, (shoulder)
            B, L, _ = pred["main_stick"].shape

            loss = 0.0
            loss_dict = {
                "main": 0.0,
                "c": 0.0,
                "shoulder": 0.0,
                "buttons": 0.0,
                "moe_aux": 0.0,
            }
            loss_s: Optional[torch.Tensor] = None
            loss_aux: Optional[torch.Tensor] = None

            # --- main stick CE ---
            logits_main = pred["main_stick"].reshape(B * L, -1)
            target_main = target_info["main_idx"].reshape(B * L)
            main_weights = _compute_ce_weights(target_main, target_info["main_K"])
            loss_main = F.cross_entropy(
                logits_main,
                target_main,
                reduction="mean",
                label_smoothing=config.train.label_smoothing,
                weight=main_weights,
            )
            loss = loss + loss_main

            # --- c-stick CE ---
            logits_c = pred["c_stick"].reshape(B * L, -1)
            target_c = target_info["c_idx"].reshape(B * L)
            c_weights = _compute_ce_weights(target_c, target_info["c_K"])
            loss_c = F.cross_entropy(
                logits_c,
                target_c,
                reduction="mean",
                label_smoothing=config.train.label_smoothing,
                weight=c_weights,
            )
            loss = loss + loss_c

            # --- buttons (multi-label BCE-with-logits) ---
            logits_btn = pred["buttons"]  # [B,L,Kb]
            probs_btn = pred.get("buttons_probs", None)
            target_btn = target_info["buttons"]
            pos_weight = _compute_pos_weights(target_btn)
            loss_btn = F.binary_cross_entropy_with_logits(
                logits_btn,
                target_btn,
                reduction="mean",
                pos_weight=pos_weight,
            )
            loss = loss + loss_btn

            # --- shoulder (optional CE) ---
            if "shoulder" in pred.keys() and target_info["shoulder_K"] > 0 and target_info["shoulder_idx"] is not None:
                logits_s = pred["shoulder"].reshape(B * L, -1)
                target_s = target_info["shoulder_idx"].reshape(B * L)
                loss_s = F.cross_entropy(
                    logits_s,
                    target_s,
                    reduction="mean",
                    label_smoothing=config.train.label_smoothing,
                )
                loss = loss + loss_s

            # --- MoE auxiliary loss (optional) ---
            if config.model.use_moe and "moe_aux_loss" in pred.keys():
                moe_aux_tensor = pred["moe_aux_loss"]
                loss_aux = moe_aux_tensor.mean() * config.model.moe_aux_loss_weight
                loss = loss + loss_aux

            # LR schedule
            lr = cosine_lr_schedule(global_step, total_steps, config.train.lr, config.train.warmup_steps)
            for pg in opt.param_groups:
                pg["lr"] = lr

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            if config.train.grad_clip is not None and config.train.grad_clip > 0:
                scaler.unscale_(opt)
                clip_grad_norm_(model.parameters(), config.train.grad_clip)

            scaler.step(opt)
            scaler.update()

            epoch_loss += float(loss.detach().item())

            # ---- Running metrics & baselines ----
            # Shapes: [B,L,*]
            # Main stick predictions/labels
            pred_main_idx = logits_main.argmax(dim=-1).view(B, L)
            true_main_idx = target_main.view(B, L)
            # C-stick
            pred_c_idx = logits_c.argmax(dim=-1).view(B, L)
            true_c_idx = target_c.view(B, L)
            # Buttons
            btn_logits = logits_btn  # [B,L,Kb]
            btn_true = target_btn  # [B,L,Kb]
            btn_probs = probs_btn

            # Baselines for main/c:
            main_major = metrics._majority_label(metrics.main_label_counts)
            c_major = metrics._majority_label(metrics.c_label_counts)
            main_rand = torch.randint(high=target_info["main_K"], size=(B * L,), device=device)
            c_rand = torch.randint(high=target_info["c_K"], size=(B * L,), device=device)

            # repeat-last (ignore first timestep)
            rep_mask = torch.ones((B, L), dtype=torch.bool, device=device)
            rep_mask[:, 0] = False
            main_rep = torch.zeros_like(true_main_idx)
            c_rep = torch.zeros_like(true_c_idx)
            if L > 1:
                main_rep[:, 1:] = true_main_idx[:, :-1]
                c_rep[:, 1:] = true_c_idx[:, :-1]

            # Update metrics
            metrics.update_main(
                pred_main_idx.reshape(-1),
                true_main_idx.reshape(-1),
                main_rand,
                main_major,
                main_rep.reshape(-1),
                rep_mask.reshape(-1),
            )
            metrics.update_c(
                pred_c_idx.reshape(-1),
                true_c_idx.reshape(-1),
                c_rand,
                c_major,
                c_rep.reshape(-1),
                rep_mask.reshape(-1),
            )
            metrics.update_buttons(btn_logits, btn_true, btn_probs)

            # Optional shoulder metrics
            if "shoulder" in pred.keys() and target_info["shoulder_K"] > 0 and target_info["shoulder_idx"] is not None:
                B_, L_, _Ks = pred["shoulder"].shape
                shoulder_rand = torch.randint(high=target_info["shoulder_K"], size=(B_, L_), device=device)
                sh_major = metrics._majority_label(metrics.shoulder_label_counts) if metrics.K_shoulder else None
                metrics.update_shoulder(pred["shoulder"], target_info["shoulder_idx"], shoulder_rand, sh_major)

            # Prepare loss dict safely
            this_loss = {
                "main": float(loss_main.detach().item()),
                "c": float(loss_c.detach().item()),
                "shoulder": float(loss_s.detach().item()) if (loss_s is not None) else 0.0,
                "buttons": float(loss_btn.detach().item()),
                "moe_aux": float(loss_aux.detach().item()) if (loss_aux is not None) else 0.0,
            }

            global_step += 1

            # Throughput / logs (rank 0)
            if it % 100 == 0:
                ckpt = {
                    "model": model.state_dict() if not isinstance(model,
                                                                  torch.nn.parallel.DistributedDataParallel) else model.module.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scaler": scaler.state_dict(),
                    "config": config.train.__dict__,
                    "epoch": epoch + 1,
                    "global_step": global_step,
                }
                torch.save(ckpt, out_dir / f"model_ep{epoch + 1:03d}_{it:03d}.pt")
            if it % 30 == 0 or it == len(loader) - 1:

                dt = max(1e-9, time.time() - t0)
                tok_per_batch = X.numel()  # rough proxy (B*L*F numeric)
                ips = (it + 1) * tok_per_batch / dt

                # ---------- Per-batch metrics & confusions ----------
                # MAIN
                main_true_flat = true_main_idx.reshape(-1)
                main_pred_flat = pred_main_idx.reshape(-1)
                K_main = int(target_info["main_K"])
                cm_main_b = _confusion_from_flat(main_true_flat, main_pred_flat, K_main)
                acc_main_b = float((main_pred_flat == main_true_flat).float().mean().item())
                acc_main_rand_b = float((main_rand == main_true_flat).float().mean().item())
                main_major_lbl = _majority_flat(main_true_flat)
                acc_main_maj_b = float((main_true_flat == main_major_lbl).float().mean().item())
                acc_main_rep_b = float((main_rep.reshape(-1)[rep_mask.reshape(-1)] == main_true_flat[
                    rep_mask.reshape(-1)]).float().mean().item()) if rep_mask.any() else 0.0
                main_conf_str = _format_confusion_small(cm_main_b, max_size=10, title="MAIN confusion")

                # C-STICK
                c_true_flat = true_c_idx.reshape(-1)
                c_pred_flat = pred_c_idx.reshape(-1)
                K_c = int(target_info["c_K"])
                cm_c_b = _confusion_from_flat(c_true_flat, c_pred_flat, K_c)
                acc_c_b = float((c_pred_flat == c_true_flat).float().mean().item())
                acc_c_rand_b = float((c_rand == c_true_flat).float().mean().item())
                c_major_lbl = _majority_flat(c_true_flat)
                acc_c_maj_b = float((c_true_flat == c_major_lbl).float().mean().item())
                acc_c_rep_b = float((c_rep.reshape(-1)[rep_mask.reshape(-1)] == c_true_flat[
                    rep_mask.reshape(-1)]).float().mean().item()) if rep_mask.any() else 0.0
                c_conf_str = _format_confusion_small(cm_c_b, max_size=12, title="C-STICK confusion")

                # BUTTONS
                btn_probs = torch.sigmoid(btn_logits)
                btn_pred = (btn_probs > 0.5).to(btn_true.dtype)
                em_b, p_b, r_b, f1_b, f1_macro_b = _multilabel_prf(btn_true, btn_pred)

                btn_true_flat = btn_true.reshape(-1, btn_true.shape[-1]).float()
                btn_pred_flat = btn_pred.reshape(-1, btn_pred.shape[-1]).float()
                btn_match = (btn_true_flat == btn_pred_flat).float().mean(dim=0)
                btn_tp = (btn_true_flat * btn_pred_flat).sum(dim=0)
                btn_fp = ((1.0 - btn_true_flat) * btn_pred_flat).sum(dim=0)
                btn_fn = (btn_true_flat * (1.0 - btn_pred_flat)).sum(dim=0)
                eps = 1e-9
                btn_prec = btn_tp / (btn_tp + btn_fp + eps)
                btn_rec = btn_tp / (btn_tp + btn_fn + eps)
                btn_f1 = 2 * btn_prec * btn_rec / (btn_prec + btn_rec + eps)
                btn_rate = btn_true_flat.mean(dim=0)
                # baselines
                btn_rand_pred = (torch.rand_like(btn_true) > 0.5).to(btn_true.dtype)
                em_rand, p_rand, r_rand, f1_rand, f1_macro_rand = _multilabel_prf(btn_true, btn_rand_pred)
                pos_rate = btn_true.float().mean(dim=(0, 1), keepdim=True)  # [1,1,K]
                btn_maj_pred = (pos_rate >= 0.5).to(btn_true.dtype).expand_as(btn_true)
                em_maj, p_maj, r_maj, f1_maj, f1_macro_maj = _multilabel_prf(btn_true, btn_maj_pred)
                if L > 1:
                    btn_rep = torch.zeros_like(btn_true)
                    btn_rep[:, 1:, :] = btn_true[:, :-1, :]
                    mask_flat = rep_mask.view(B * L)
                    t_flat = btn_true.reshape(B * L, -1)[mask_flat]
                    p_flat = btn_rep.reshape(B * L, -1)[mask_flat]
                    em_rep, p_rep, r_rep, f1_rep, f1_macro_rep = _multilabel_prf(t_flat, p_flat)
                else:
                    em_rep = p_rep = r_rep = f1_rep = f1_macro_rep = 0.0

                # SHOULDER (optional)
                shoulder_present = ("shoulder" in pred.keys()) and (target_info["shoulder_K"] > 0) and (
                        target_info["shoulder_idx"] is not None)
                if shoulder_present:
                    sh_logits = pred["shoulder"]
                    sh_pred_idx = sh_logits.argmax(dim=-1)  # [B,L]
                    sh_true_idx = target_info["shoulder_idx"]
                    acc_sh = float((sh_pred_idx == sh_true_idx).float().mean().item())
                    sh_rand = torch.randint(high=target_info["shoulder_K"], size=(B, L), device=device)
                    acc_sh_rand = float((sh_rand == sh_true_idx).float().mean().item())
                    sh_major_lbl = _majority_flat(sh_true_idx.reshape(-1))
                    acc_sh_maj = float((sh_true_idx == sh_major_lbl).float().mean().item())
                    if L > 1:
                        sh_rep = torch.zeros_like(sh_true_idx)
                        sh_rep[:, 1:] = sh_true_idx[:, :-1]
                        acc_sh_rep = float((sh_rep[rep_mask] == sh_true_idx[rep_mask]).float().mean().item())
                    else:
                        acc_sh_rep = 0.0

                # ---------- Compose log ----------
                header = (
                    f"ep {epoch + 1}/{config.train.epochs} it {it + 1}/{len(loader)}\n"
                    f"  loss {epoch_loss / (it + 1):.4f} | lr {lr:.2e} | items/s {ips:,.0f} | {this_loss}"
                )
                main_line = (
                    f"  MAIN:     acc {acc_main_b:.3f} | rand {acc_main_rand_b:.3f} | maj {acc_main_maj_b:.3f} | rep {acc_main_rep_b:.3f}"
                )
                c_line = (
                    f"  C-STICK:  acc {acc_c_b:.3f} | rand {acc_c_rand_b:.3f} | maj {acc_c_maj_b:.3f} | rep {acc_c_rep_b:.3f}"
                )
                btn_line1 = (
                    f"  BUTTONS:  EM {em_b:.3f} | F1μ {f1_b:.3f} | F1macro {f1_macro_b:.3f}"
                )
                btn_line2 = (
                    f"            rand F1μ {f1_rand:.3f} | maj F1μ {f1_maj:.3f} | rep F1μ {f1_rep:.3f} | EM_rep {em_rep:.3f}"
                )
                per_button = []
                btn_names = _CONTROLLER_KEYS["buttons"]
                for idx, name in enumerate(btn_names):
                    label = _BUTTON_PRETTY.get(name, name)
                    per_button.append(
                        f"{label}: acc {btn_match[idx].item():.3f} F1 {btn_f1[idx].item():.3f} rate {btn_rate[idx].item():.3f}"
                    )
                btn_line3 = "            " + " | ".join(per_button)

                log_lines = [
                    header,
                    main_line,
                    indent(main_conf_str, "    "),
                    c_line,
                    indent(c_conf_str, "    "),
                    btn_line1,
                    btn_line2,
                    btn_line3,
                ]
                if shoulder_present:
                    log_lines.append(
                        f"  SHOULDER: acc {acc_sh:.3f} | rand {acc_sh_rand:.3f} | maj {acc_sh_maj:.3f} | rep {acc_sh_rep:.3f}"
                    )

                print("\n".join(log_lines))

        # End-of-epoch: save confusion matrices and print summary
        np.save(out_dir / f"confusion_main_ep{epoch + 1:03d}.npy", metrics.main_confusion.cpu().numpy())
        np.save(out_dir / f"confusion_c_ep{epoch + 1:03d}.npy", metrics.c_confusion.cpu().numpy())
        if (epoch + 1) % config.train.save_every_epochs == 0:
            print(f"[epoch {epoch + 1}] summary: {metrics.short_str()}")

        # Save checkpoint (rank 0)
        if (epoch + 1) % config.train.save_every_epochs == 0:
            ckpt = {
                "model": model.state_dict() if not isinstance(model,
                                                              torch.nn.parallel.DistributedDataParallel) else model.module.state_dict(),
                "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "config": config.train.__dict__,
                "epoch": epoch + 1,
                "global_step": global_step,
            }
            torch.save(ckpt, out_dir / f"model_ep{epoch + 1:03d}.pt")


if __name__ == "__main__":
    init_config()
    model = GPTv7()
    train_loop(model)
