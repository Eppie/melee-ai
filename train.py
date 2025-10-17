from __future__ import annotations

import math
import time
from dataclasses import dataclass
import json
import os
from pathlib import Path
from textwrap import indent
from typing import Callable, Dict
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from tensordict import TensorDict
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer

from column_map import ColumnMap, CONTROLLER_KEY_GROUPS
from config import get_config, init_config
from controller_quantization import quantize_targets
from controller_utils import CONTROL_STICK_QUANTIZED
from libmelee.melee.enums import Action
from loss import compute_loss_components
from model.gpt import GPTv7
from utils import print_model_diagram, _resolve_device

# Optional Weights & Biases logging
try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None  # type: ignore
from window_dataset import make_dataloader

_MAIN_STICK_LABELS: List[str] = [f"({x:.2f},{y:.2f})" for x, y in CONTROL_STICK_QUANTIZED]
_ACTION_VALUE_TO_NAME = {action.value: action.name for action in Action}


def _sorted_checkpoint_paths(directory: Path) -> List[Path]:
    checkpoints = [p for p in directory.glob("*.pt") if p.is_file()]
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return checkpoints


def _prune_checkpoints(directory: Path, keep: int = 10) -> None:
    if keep <= 0:
        return
    checkpoints = _sorted_checkpoint_paths(directory)
    for old_ckpt in checkpoints[keep:]:
        try:
            old_ckpt.unlink()
        except OSError as err:
            print(f"Warning: failed to remove old checkpoint {old_ckpt}: {err}")


def _move_optimizer_state_to_device(optimizer: Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _load_latest_checkpoint(
        directory: Path,
        model: torch.nn.Module,
        optimizer: Optimizer,
        scaler: GradScaler,
        device: torch.device,
) -> Tuple[int, int]:
    checkpoints = _sorted_checkpoint_paths(directory)
    if not checkpoints:
        return 0, 0

    latest = checkpoints[0]
    print(f"Resuming from checkpoint: {latest}")
    ckpt = torch.load(latest, map_location="cpu")

    model_state = ckpt.get("model")
    if model_state:
        model.load_state_dict(model_state)

    opt_state = ckpt.get("optimizer")
    if opt_state:
        optimizer.load_state_dict(opt_state)
        _move_optimizer_state_to_device(optimizer, device)

    scaler_state = ckpt.get("scaler")
    if scaler_state:
        scaler.load_state_dict(scaler_state)

    start_epoch = int(ckpt.get("epoch", 0))
    global_step = int(ckpt.get("global_step", 0))
    return max(start_epoch, 0), max(global_step, 0)


def _bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024: return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def _format_value(value: object) -> str:
    """Format numeric values with up to 6 significant figures."""
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def _format_action(value: object) -> str:
    try:
        idx = int(float(value))
    except (TypeError, ValueError):
        return str(value)
    return _ACTION_VALUE_TO_NAME.get(idx, str(idx))


def _print_table_block(
        title: str,
        headers: Sequence[str],
        data: np.ndarray,
        *,
        max_columns: int = 8,
        formatters: Optional[Dict[str, Callable[[object], str]]] = None,
) -> None:
    if data.size == 0 or not len(headers):
        print(f"{title}: <empty>")
        return

    total_cols = len(headers)
    num_rows = data.shape[0]
    frame_label = "frame"
    formatters = formatters or {}
    frame_width = max(len(frame_label), len(str(num_rows - 1)) if num_rows else len(frame_label))

    for start in range(0, total_cols, max_columns):
        cols = headers[start:start + max_columns]
        block = data[:, start:start + len(cols)]
        formatted_columns: List[List[str]] = []
        col_widths: List[int] = []
        for col_idx, col_name in enumerate(cols):
            formatter = formatters.get(col_name, _format_value)
            col_values: List[str] = []
            for row_idx in range(num_rows):
                value = block[row_idx, col_idx]
                col_values.append(str(formatter(value)))
            max_value_width = max((len(val) for val in col_values), default=0)
            col_width = max(len(col_name), max_value_width, 6)
            formatted_columns.append(col_values)
            col_widths.append(col_width)

        print(f"{title} (columns {start + 1}-{start + len(cols)} of {total_cols}):")
        widths = [frame_width + 2] + col_widths
        header_cells = [frame_label.rjust(widths[0])]
        header_cells.extend(col.rjust(width) for col, width in zip(cols, widths[1:]))
        print(" ".join(header_cells))

        for row_idx in range(num_rows):
            row_cells = [str(row_idx).rjust(widths[0])]
            for col_values, width in zip(formatted_columns, widths[1:]):
                row_cells.append(col_values[row_idx].rjust(width))
            print(" ".join(row_cells))
        print()


def preview_training_batch(
        batch: Dict[str, torch.Tensor],
        feature_names: Sequence[str],
        target_names: Sequence[str],
        *,
        max_frames: int = 10,
) -> None:
    """Pretty-print the first sequence from the first batch for manual inspection."""

    X = batch["X"].detach().cpu()
    Y = batch["Y"].detach().cpu() if batch["Y"].numel() else None

    first_seq = X[0]
    num_frames = min(max_frames, first_seq.shape[0])
    feat_slice = first_seq[:num_frames].numpy()

    print("=== First batch preview (sequence 0, first {num_frames} frames) ===".format(num_frames=num_frames))
    formatters: Dict[str, Callable[[object], str]] = {}
    for key in feature_names:
        if key.endswith("_action"):
            formatters[key] = _format_action

    _print_table_block(
        "Feature preview",
        feature_names,
        feat_slice,
        formatters=formatters,
    )

    if Y is not None and Y.shape[-1] > 0:
        target_slice = Y[0, :num_frames].numpy()
        _print_table_block(
            "Target preview",
            target_names,
            target_slice,
        )


def cosine_lr_schedule(step: int, total_steps: int, base_lr: float, warmup: int = 0) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def per_epoch_linear_decay_lr(step_in_epoch: int, steps_per_epoch: int, lr_max: float) -> float:
    """Per-epoch linear decay: start at lr_max then decrease to 0 by epoch end.

    - step_in_epoch: zero-based iteration index within current epoch
    - steps_per_epoch: total iterations planned in the epoch
    """
    if steps_per_epoch <= 1:
        return lr_max
    # progress in [0, 1] across the epoch
    t = float(step_in_epoch) / float(max(1, steps_per_epoch - 1))
    # linear decay from lr_max to 0
    return lr_max * max(0.0, 1.0 - t)


# -----------------------------
# Column mapping / batch adapter
# -----------------------------

_BUTTON_PRETTY = {
    "button_a": "A",
    "button_b": "B",
    "button_xy": "X/Y",
    "button_z": "Z",
    "button_lr": "L/R",
}


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
      - majority (running)
      - repeat-last (ignoring first timestep)
    Also accumulates confusion matrices for main and c.
    """

    def __init__(self, K_main: int, K_c: int, K_buttons: int, K_shoulder: int,
                 device: torch.device | None = None) -> None:
        self.device = device or torch.device("cpu")
        dtype = torch.float32

        # Main
        self.K_main = K_main
        self.main_correct = torch.zeros((), device=self.device, dtype=dtype)
        self.main_total = torch.zeros((), device=self.device, dtype=dtype)
        self.main_confusion = torch.zeros((K_main, K_main), device=self.device, dtype=dtype)
        self.main_label_counts = torch.zeros(K_main, device=self.device, dtype=dtype)
        self.main_maj_correct = torch.zeros((), device=self.device, dtype=dtype)
        self.main_rep_correct = torch.zeros((), device=self.device, dtype=dtype)
        self.main_rep_total = torch.zeros((), device=self.device, dtype=dtype)

        # C-stick
        self.K_c = K_c
        self.c_correct = torch.zeros((), device=self.device, dtype=dtype)
        self.c_total = torch.zeros((), device=self.device, dtype=dtype)
        self.c_confusion = torch.zeros((K_c, K_c), device=self.device, dtype=dtype)
        self.c_label_counts = torch.zeros(K_c, device=self.device, dtype=dtype)
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
        self.shoulder_maj_correct = torch.zeros((), device=self.device, dtype=dtype)
        self.shoulder_label_counts = torch.zeros(K_shoulder, device=self.device, dtype=dtype)

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
            maj_before: Optional[int],  # int or None
    ) -> None:
        if logits is None or true_idx is None:
            return
        with torch.no_grad():
            pred_idx = logits.argmax(dim=-1).reshape(-1)
            true_flat = true_idx.reshape(-1)

            self.shoulder_total.add_(float(true_flat.numel()))
            self.shoulder_correct.add_((pred_idx == true_flat).float().sum())
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
        out["acc_main_maj"] = _safe_div(self.main_maj_correct, self.main_total)
        out["acc_main_rep"] = _safe_div(self.main_rep_correct, self.main_rep_total)

        out["acc_c"] = _safe_div(self.c_correct, self.c_total)

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

        pmm, rmm, f1mm, f1mmm = self._btn_prf(self.btn_maj_tp, self.btn_maj_fp, self.btn_maj_fn)
        out["btn_em_maj"] = _safe_div(self.btn_maj_exact, self.btn_total)
        out["btn_f1_micro_maj"] = pmm if pmm == pmm else 0.0
        out["btn_f1_macro_maj"] = f1mm

        pmx, rmx, f1mx, f1mmx = self._btn_prf(self.btn_rep_tp, self.btn_rep_fp, self.btn_rep_fn)
        out["btn_em_rep"] = _safe_div(self.btn_rep_exact, self.btn_rep_total)
        out["btn_f1_micro_rep"] = f1mx
        out["btn_f1_macro_rep"] = f1mmx

        # shoulder
        out["acc_shoulder"] = _safe_div(self.shoulder_correct, self.shoulder_total)
        out["acc_shoulder_maj"] = _safe_div(self.shoulder_maj_correct, self.shoulder_total)

        return out

    def short_str(self) -> str:
        s = self.summary()
        return (
                f" | main acc {s['acc_main']:.3f} maj {s['acc_main_maj']:.3f}, rep {s['acc_main_rep']:.3f})"
                f" | c acc {s['acc_c']:.3f}, maj {s['acc_c_maj']:.3f}, rep {s['acc_c_rep']:.3f})"
                f" | btn EM {s['btn_em']:.3f} F1μ {s['btn_f1_micro']:.3f} , maj {s['btn_f1_micro_maj']:.3f}, rep {s['btn_f1_micro_rep']:.3f})"
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


def _format_confusion_small(
        cm: torch.Tensor,
        max_size: int = 12,
        title: str | None = None,
        labels: Optional[Sequence[str]] = None,
) -> str:
    """
    If K <= max_size, render full matrix with row sums and diagonal percentages.
    Otherwise, print top off-diagonal confusions.
    """
    K = cm.shape[0]
    if title is None:
        title = "confusion"

    label_list: Sequence[str]
    if labels is not None:
        if len(labels) != K:
            raise ValueError("labels length must match confusion matrix dimensions")
        label_list = [str(lbl) for lbl in labels]
    else:
        label_list = [f"{i:02d}" for i in range(K)]

    if K > max_size:
        tops = _top_confusions(cm, k=10)
        if not tops:
            return f"{title}: (no confusions)"
        lines = [f"{title}: top confusions (true->pred: count, %offdiag)"]
        lines += [
            f"  {label_list[t]}->{label_list[p]}: {c} ({pct:.1f}%)"
            for t, p, c, pct in tops
        ]
        return "\n".join(lines)

    arr = cm.numpy()
    row_sums_1d = arr.sum(axis=1)  # shape (K,)
    diag_vals = np.diag(arr).astype(float)
    # Safe divide: out=0 where row sum == 0
    with np.errstate(divide='ignore', invalid='ignore'):
        diag_pct = np.divide(diag_vals * 100.0, row_sums_1d, out=np.zeros_like(diag_vals, dtype=float),
                             where=row_sums_1d > 0)

    col_width = max(len(lbl) for lbl in label_list)
    cell_width = max(4, col_width)
    header = " " * (cell_width + 1) + " ".join([lbl.rjust(cell_width) for lbl in label_list]) + " | sum"
    lines = [f"{title}: full {K}x{K}", header]
    for i in range(K):
        row = " ".join([f"{int(v):>{cell_width}d}" for v in arr[i]])
        lines.append(f"{label_list[i].rjust(cell_width)}: {row} | {int(row_sums_1d[i]):>{cell_width}d}")
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


def change_boost(Y, B, L, device, epoch):
    change_mask = torch.zeros((B, L), device=device, dtype=torch.bool)
    # Compare from the second timestep onwards
    state_changed = torch.any(Y[:, 1:] != Y[:, :-1], dim=-1)
    change_mask[:, 1:] = state_changed

    # Define a weight for "change" frames vs "hold" frames
    # For example, make change frames 10x more important.
    change_weight = max(1, 10 - (epoch % 7))
    # Create a weight tensor for the loss function
    sample_weights = torch.ones((B, L), device=device)
    sample_weights[change_mask] = change_weight
    return sample_weights


@dataclass(frozen=True)
class RewardFeatureIdx:
    p1_stock: Optional[int] = None
    p2_stock: Optional[int] = None
    p1_percent: Optional[int] = None
    p2_percent: Optional[int] = None
    p1_in_hitlag: Optional[int] = None
    p2_in_hitlag: Optional[int] = None
    p1_in_defender_hitlag: Optional[int] = None
    p2_in_defender_hitlag: Optional[int] = None
    p1_shield_strength: Optional[int] = None


def build_reward_feature_index(colmap: ColumnMap) -> RewardFeatureIdx:
    """Resolve once and reuse; avoids per-call .index() overhead."""
    names = colmap.feat_names

    def idx(name: str) -> Optional[int]:
        return names.index(name) if name in names else None

    return RewardFeatureIdx(
        p1_stock=idx("p1_stock"),
        p2_stock=idx("p2_stock"),
        p1_percent=idx("p1_percent"),
        p2_percent=idx("p2_percent"),
        p1_in_hitlag=idx("p1_in_hitlag"),
        p2_in_hitlag=idx("p2_in_hitlag"),
        p1_in_defender_hitlag=idx("p1_in_defender_hitlag"),
        p2_in_defender_hitlag=idx("p2_in_defender_hitlag"),
        p1_shield_strength=idx("p1_shield_strength"),
    )


def compute_frame_rewards(
        X: torch.Tensor,
        colmap: ColumnMap,
        *,
        idx: Optional[RewardFeatureIdx] = None,
) -> torch.Tensor:
    """Vectorized & allocation-lean version."""
    B, L, _F = X.shape
    device = X.device
    dtype = X.dtype

    if idx is None:
        # Resolve on the fly (still cheap), or pass a cached `idx` from caller for max perf.
        idx = build_reward_feature_index(colmap)

    cfg = get_config().rl
    rw = torch.full((B, L), float(cfg.reward_per_frame), device=device, dtype=dtype)  # base per-frame reward

    if L > 1:
        # --- Damage deltas (vectorized with torch.diff) ---
        if (idx.p1_percent is not None) and (idx.p2_percent is not None):
            # deltas are current - previous over time dim
            d_p2 = torch.diff(X[:, :, idx.p2_percent], dim=1)  # [B, L-1]
            d_p1 = torch.diff(X[:, :, idx.p1_percent], dim=1)  # [B, L-1]
            # Scale percents from [0,1] back to 0..100 if that matches your preprocessing
            # Then apply weights, in-place add to t>=1 frames
            if float(cfg.reward_damage_dealt) != 0.0:
                rw[:, 1:].add_(d_p2.mul_(100.0).mul_(float(cfg.reward_damage_dealt)))
            if float(cfg.reward_damage_taken) != 0.0:
                rw[:, 1:].add_(d_p1.mul_(100.0).mul_(float(cfg.reward_damage_taken)))

        # --- Stock changes (vectorized) ---
        if (idx.p1_stock is not None) and (idx.p2_stock is not None):
            d_p2_stock = torch.diff(X[:, :, idx.p2_stock], dim=1)  # [B,L-1]
            d_p1_stock = torch.diff(X[:, :, idx.p1_stock], dim=1)  # [B,L-1]
            # Lost stock ⇒ delta = -1. Clamp the positive part of (-delta)
            stock_taken = (-d_p2_stock).clamp_min_(0)  # opponent lost stock
            stock_lost = (-d_p1_stock).clamp_min_(0)  # we lost stock
            if float(cfg.reward_stock_taken) != 0.0:
                rw[:, 1:].add_(stock_taken.mul_(float(cfg.reward_stock_taken)))
            if float(cfg.reward_stock_lost) != 0.0:
                rw[:, 1:].add_(stock_lost.mul_(float(cfg.reward_stock_lost)))

    # --- Hitlag rewards/penalties (no diffs, per-frame) ---
    if (
            (idx.p1_in_hitlag is not None) and (idx.p1_in_defender_hitlag is not None) and
            (idx.p2_in_hitlag is not None) and (idx.p2_in_defender_hitlag is not None)
    ):
        # metric: in_hitlag - in_defender_hitlag; reward when equals 1
        p1_metric = X[:, :, idx.p1_in_hitlag] - X[:, :, idx.p1_in_defender_hitlag]
        p2_metric = X[:, :, idx.p2_in_hitlag] - X[:, :, idx.p2_in_defender_hitlag]

        if float(cfg.reward_hitlag_self) != 0.0:
            rw.add_((p1_metric == 1).to(dtype).mul_(float(cfg.reward_hitlag_self)))
        if float(cfg.reward_hitlag_opponent) != 0.0:
            rw.add_((p2_metric == 1).to(dtype).mul_(float(cfg.reward_hitlag_opponent)))

    # --- Shield penalty (fused math; no mask tensor needed) ---
    if idx.p1_shield_strength is not None and float(cfg.reward_low_shield) != 0.0:
        s = X[:, :, idx.p1_shield_strength]  # [B, L] in [0,1]
        # penalty_multiplier = clamp(1 - 2*shield, 0, 1)
        pen = (1.0 - 2.0 * s).clamp_min_(0.0).clamp_max_(1.0)
        rw.add_(pen.mul_(float(cfg.reward_low_shield)))

    return rw


def compute_value_targets(
        X: torch.Tensor,
        colmap: ColumnMap,
        gamma: float = 0.99,
        *,
        reward_idx: Optional[RewardFeatureIdx] = None,
) -> torch.Tensor:
    """Compute discounted returns in O(B·L), vectorized across batch."""
    B, L, _ = X.shape
    device = X.device
    dtype = X.dtype

    # Per-frame rewards (already fast)
    r = compute_frame_rewards(X, colmap, idx=reward_idx)  # [B, L]

    # Backward discounted scan: G_t = r_t + gamma * G_{t+1}
    G = torch.empty((B, L), device=device, dtype=dtype)
    next_G = torch.zeros((B,), device=device, dtype=dtype)

    # reverse loop over time dimension (only L steps, not L^2)
    for t in range(L - 1, -1, -1):
        # G[:, t] = r[:, t] + gamma * next_G
        cur = r[:, t].add(next_G.mul(gamma))
        G[:, t] = cur
        next_G = cur  # reuse for next iteration

    # Add terminal “win” bonus (constant 1.0 at episode end, discounted by steps remaining)
    terminal = 1.0
    # vector of gamma^(L - t - 1) for t=0..L-1
    pow_vec = torch.pow(torch.tensor(gamma, device=device, dtype=dtype),
                        torch.arange(L - 1, -1, -1, device=device, dtype=dtype))
    # pow_vec[t] = gamma^(L-1-t) == gamma^(steps_to_end)
    G.add_(pow_vec.mul_(terminal))

    return G.unsqueeze(-1)  # [B, L, 1]


def train_loop(
        model: GPTv7,
) -> None:
    device = _resolve_device(None)
    model = model.to(device)
    config = get_config()
    
    # Verify PyTorch version and MPS support for AMP
    if config.train.use_amp:
        print(f"Using PyTorch {torch.__version__}")
        if device.type == 'mps':
            if not torch.backends.mps.is_available():
                raise RuntimeError("MPS backend not available, cannot use AMP on MPS")
            print(f"AMP enabled with {config.train.amp_dtype} on MPS backend")
        elif device.type == 'cuda':
            print(f"AMP enabled with {config.train.amp_dtype} on CUDA backend")
        else:
            print(f"Warning: AMP may not be optimized for device type '{device.type}'")
    
    # Build loader + sampler
    loader, ds, sampler = make_dataloader()

    # Column map built from dataset metadata (only once)
    colmap = ColumnMap.from_dataset(ds)
    reward_idx = build_reward_feature_index(colmap)

    # Optimizer & (optional) simple cosine LR
    opt = torch.optim.AdamW(model.parameters(), lr=config.train.lr, betas=config.train.betas,
                            weight_decay=config.train.weight_decay)
    
    # GradScaler for automatic mixed precision (no device arg in torch 2.1)
    scaler = GradScaler(enabled=config.train.use_amp)

    steps_per_epoch = config.train.steps_per_epoch or math.ceil(len(loader))
    total_steps = config.train.max_steps or (config.train.epochs * steps_per_epoch)
    global_step = 0
    start_epoch = 0

    out_dir = Path(config.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Persist most-recent global_step so wandb step stays monotonic between checkpoints
    last_step_file = out_dir / "last_step.txt"

    # Initialize Weights & Biases if available
    use_wandb = False
    if wandb is not None:
        try:
            wandb_settings = {}
            run_name = getattr(config.train, "run_name", None)
            if run_name:
                wandb_settings["name"] = run_name
            # Enable resuming to the SAME run after restarts by keeping a stable run id
            run_id_file = out_dir / "wandb_run_id.txt"
            # Allow explicit id via config if provided
            resume_run_id: Optional[str] = getattr(config.train, "wandb_run_id", None)
            # 1) Prefer a stored run id (created after first init)
            try:
                if not resume_run_id and run_id_file.exists():
                    resume_run_id = run_id_file.read_text().strip() or None
            except Exception:
                resume_run_id = resume_run_id or None
            # 2) Allow override via environment (e.g., WANDB_RUN_ID)
            if not resume_run_id:
                resume_run_id = os.environ.get("WANDB_RUN_ID") or os.environ.get("WANDB_RESUME_ID")
            # 3) As a fallback, try to discover the last run id from out_dir/wandb/latest-run
            if not resume_run_id:
                try:
                    latest = (out_dir / "wandb" / "latest-run").resolve(strict=True)
                    # Try metadata first
                    meta_path = latest / "files" / "wandb-metadata.json"
                    if meta_path.exists():
                        meta = json.loads(meta_path.read_text())
                        resume_run_id = meta.get("id") or meta.get("run_id")
                    # Fallback: parse directory name 'run-YYYYMMDD_HHMMSS-<id>'
                    if not resume_run_id:
                        base = latest.name
                        if "-" in base:
                            resume_run_id = base.split("-")[-1]
                except Exception:
                    pass
            if resume_run_id:
                # Ask wandb to resume, but allow fresh start if the id doesn't exist remotely
                wandb_settings["id"] = resume_run_id
                wandb_settings["resume"] = "allow"
            wandb.init(
                project=getattr(config.train, "wandb_project", "melee-ai"),
                config={
                    "train": dict(vars(config.train)),
                    "model": dict(vars(config.model)),
                    "seq_len": getattr(config, "seq_len", None),
                },
                dir=str(out_dir),
                **wandb_settings,
            )
            # Persist the effective run id so a subsequent restart can reuse it
            try:
                if wandb.run is not None and getattr(wandb.run, "id", None):
                    run_id_file.write_text(str(wandb.run.id))
            except Exception:
                pass
            try:
                wandb.watch(model, log="gradients", log_freq=100)
            except Exception:
                pass
            use_wandb = True
        except Exception as e:
            print(f"wandb init failed: {e}. Continuing without wandb.")

    start_epoch, global_step = _load_latest_checkpoint(out_dir, model, opt, scaler, device)
    # If we have a more recent persisted step, prefer it to keep wandb step increasing
    try:
        if last_step_file.exists():
            persisted = int(last_step_file.read_text().strip())
            global_step = max(global_step, persisted)
    except Exception:
        pass

    if start_epoch >= config.train.epochs:
        print(f"All requested epochs ({config.train.epochs}) already completed (start_epoch={start_epoch}); exiting.")
        return

    if config.train.max_steps and global_step >= config.train.max_steps:
        print(f"Global step {global_step} reached configured max_steps={config.train.max_steps}; exiting.")
        return

    preview_done = False

    # Main epochs
    for epoch in range(start_epoch, config.train.epochs):
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        model.train()

        metrics = RunningMetrics(
            K_main=config.model.target_shapes_by_head["main_stick"],
            K_c=config.model.target_shapes_by_head["c_stick"],
            K_buttons=config.model.target_shapes_by_head["buttons"],
            K_shoulder=config.model.target_shapes_by_head["shoulder"],
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

            if not preview_done:
                preview_training_batch(batch, colmap.feat_names, colmap.targ_names)
                preview_done = True

            # Move to device
            X: torch.Tensor = batch["X"].to(device, non_blocking=True)  # [B,L,F]
            Y: torch.Tensor = batch["Y"].to(device, non_blocking=True)  # [B,L,Yd]

            # Determine autocast device type and dtype
            autocast_device = 'cuda' if device.type in ('cuda', 'mps') else 'cpu'
            amp_dtype = torch.float16 if config.train.amp_dtype == "float16" else torch.bfloat16
            
            # Forward pass and loss computation with automatic mixed precision
            with autocast(device_type=autocast_device, dtype=amp_dtype, enabled=config.train.use_amp):
                # Build model inputs & target labels
                inputs_td = build_inputs_for_gptv7(X, colmap)
                target_info = quantize_targets(Y, colmap, input_domain="unit11")

                pred: TensorDict = model(inputs_td)  # keys: buttons, main_stick, c_stick, (shoulder), optionally value
                B, L, _ = pred["main_stick"].shape
                sample_weights = change_boost(Y, B, L, device, epoch)

                probs_btn = pred.get("buttons_probs", None)

                loss_components = compute_loss_components(
                    pred,
                    target_info,
                    label_smoothing=config.train.label_smoothing,
                    use_moe=config.model.use_moe,
                    moe_aux_loss_weight=config.model.moe_aux_loss_weight,
                    sample_weights=sample_weights,  # Pass the new weights
                )
                loss = loss_components["total"]
                loss_main = loss_components["main"]
                loss_c = loss_components["c"]
                loss_btn = loss_components["buttons"]
                loss_s = loss_components["shoulder"]
                loss_aux = loss_components["moe_aux"]

                # Value head loss (if enabled)
                loss_value = torch.tensor(0.0, device=device)
                if config.model.use_value_head and "value" in pred:
                    value_pred = pred["value"]  # [B, L, 1]
                    value_target = compute_value_targets(X, colmap, gamma=config.rl.gamma,
                                                         reward_idx=reward_idx)  # [B, L, 1]

                    # MSE loss for value prediction
                    value_loss_raw = torch.nn.functional.mse_loss(value_pred, value_target, reduction='none')  # [B, L, 1]

                    # Apply same sample weights as policy loss
                    weighted_value_loss = value_loss_raw.squeeze(-1) * sample_weights  # [B, L]
                    loss_value = weighted_value_loss.mean()

                    # Add to total loss with coefficient
                    loss = loss + config.rl.value_loss_coef * loss_value

            logits_main = pred["main_stick"].reshape(B * L, -1)
            target_main = target_info["main_idx"].reshape(B * L)
            logits_c = pred["c_stick"].reshape(B * L, -1)
            target_c = target_info["c_idx"].reshape(B * L)
            logits_btn = pred["buttons"]  # [B,L,Kb]
            target_btn = target_info["buttons"]

            # Per-epoch LR: start at max at epoch start, linearly decay to 0 by epoch end
            lr_max = getattr(config.train, "lr_max", None) or config.train.lr
            lr = per_epoch_linear_decay_lr(it, steps_per_epoch, lr_max)
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

            # Create masks for where the true action changes. Shape: [B, L]
            main_change_mask = torch.zeros_like(true_main_idx, dtype=torch.bool)
            main_change_mask[:, 1:] = (true_main_idx[:, 1:] != true_main_idx[:, :-1])
            main_hold_mask = ~main_change_mask
            # The first frame of a sequence can't be a "change", so it's always a "hold".
            main_hold_mask[:, 0] = True

            c_change_mask = torch.zeros_like(true_c_idx, dtype=torch.bool)
            c_change_mask[:, 1:] = (true_c_idx[:, 1:] != true_c_idx[:, :-1])
            c_hold_mask = ~c_change_mask
            c_hold_mask[:, 0] = True

            # For buttons, a change is any bit flip
            btn_change_mask = torch.zeros((B, L), device=device, dtype=torch.bool)
            btn_change_mask[:, 1:] = torch.any(btn_true[:, 1:] != btn_true[:, :-1], dim=-1)
            btn_hold_mask = ~btn_change_mask
            btn_hold_mask[:, 0] = True

            # Baselines for main/c:
            main_major = metrics._majority_label(metrics.main_label_counts)
            c_major = metrics._majority_label(metrics.c_label_counts)

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
                main_major,
                main_rep.reshape(-1),
                rep_mask.reshape(-1),
            )
            metrics.update_c(
                pred_c_idx.reshape(-1),
                true_c_idx.reshape(-1),
                c_major,
                c_rep.reshape(-1),
                rep_mask.reshape(-1),
            )
            metrics.update_buttons(btn_logits, btn_true, btn_probs)

            # shoulder metrics
            B_, L_, _Ks = pred["shoulder"].shape
            sh_major = metrics._majority_label(metrics.shoulder_label_counts) if metrics.K_shoulder else None
            metrics.update_shoulder(pred["shoulder"], target_info["shoulder_idx"], sh_major)

            # Prepare loss dict safely
            this_loss = {
                "main": float(loss_main.detach().item()),
                "c": float(loss_c.detach().item()),
                "shoulder": float(loss_s.detach().item()),
                "buttons": float(loss_btn.detach().item()),
                "moe_aux": float(loss_aux.detach().item()) if (loss_aux is not None) else 0.0,
                "value": float(loss_value.detach().item()) if config.model.use_value_head else 0.0,
            }

            global_step += 1

            # Throughput / logs (rank 0)
            if it % 5000 == 0:
                ckpt = {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scaler": scaler.state_dict(),
                    "config": config.train.__dict__,
                    "epoch": epoch + 1,
                    "global_step": global_step,
                }
                torch.save(ckpt, out_dir / f"model_ep{epoch + 1:03d}_{it:03d}.pt")
                _prune_checkpoints(out_dir, keep=10)
            if it % 100 == 0 or it == len(loader) - 1:

                dt = max(1e-9, time.time() - t0)
                B_cur, L_cur, F_cur = X.shape
                # frames/s: each frame is a token in [B,L]
                frames_per_batch = B_cur * L_cur
                frames_per_s = (it + 1) * frames_per_batch / dt

                # ---------- Per-batch metrics & confusions ----------
                # MAIN
                main_true_flat = true_main_idx.reshape(-1)
                main_pred_flat = pred_main_idx.reshape(-1)
                K_main = int(target_info["main_K"])
                cm_main_b = _confusion_from_flat(main_true_flat, main_pred_flat, K_main)
                acc_main_b = float((main_pred_flat == main_true_flat).float().mean().item())
                main_major_lbl = _majority_flat(main_true_flat)
                acc_main_rep_b = float((main_rep.reshape(-1)[rep_mask.reshape(-1)] == main_true_flat[
                    rep_mask.reshape(-1)]).float().mean().item()) if rep_mask.any() else 0.0
                main_conf_str = _format_confusion_small(
                    cm_main_b,
                    max_size=10,
                    title="MAIN confusion",
                    labels=_MAIN_STICK_LABELS[:K_main],
                )

                # C-STICK
                c_true_flat = true_c_idx.reshape(-1)
                c_pred_flat = pred_c_idx.reshape(-1)
                K_c = int(target_info["c_K"])
                cm_c_b = _confusion_from_flat(c_true_flat, c_pred_flat, K_c)
                acc_c_b = float((c_pred_flat == c_true_flat).float().mean().item())
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

                # SHOULDER
                sh_logits = pred["shoulder"]
                sh_pred_idx = sh_logits.argmax(dim=-1)  # [B,L]
                sh_true_idx = target_info["shoulder_idx"]
                acc_sh = float((sh_pred_idx == sh_true_idx).float().mean().item())
                sh_major_lbl = _majority_flat(sh_true_idx.reshape(-1))
                acc_sh_maj = float((sh_true_idx == sh_major_lbl).float().mean().item())
                if L > 1:
                    sh_rep = torch.zeros_like(sh_true_idx)
                    sh_rep[:, 1:] = sh_true_idx[:, :-1]
                    acc_sh_rep = float((sh_rep[rep_mask] == sh_true_idx[rep_mask]).float().mean().item())
                else:
                    acc_sh_rep = 0.0

                # --- MAIN STICK ---
                correct_main = (pred_main_idx == true_main_idx)
                # Calculate split accuracies
                acc_main_chg = correct_main[main_change_mask].float().mean().item() if main_change_mask.any() else 0.0
                acc_main_hold = correct_main[main_hold_mask].float().mean().item() if main_hold_mask.any() else 0.0

                # --- C-STICK ---
                correct_c = (pred_c_idx == true_c_idx)
                acc_c_chg = correct_c[c_change_mask].float().mean().item() if c_change_mask.any() else 0.0
                acc_c_hold = correct_c[c_hold_mask].float().mean().item() if c_hold_mask.any() else 0.0

                # --- BUTTONS (Exact Match Ratio) ---
                correct_btn_em = (btn_pred == btn_true).all(dim=-1)
                em_btn_chg = correct_btn_em[btn_change_mask].float().mean().item() if btn_change_mask.any() else 0.0
                em_btn_hold = correct_btn_em[btn_hold_mask].float().mean().item() if btn_hold_mask.any() else 0.0

                # --- Update the log strings ---

                # ---------- Compose log ----------
                header = (
                    f"ep {epoch + 1}/{config.train.epochs} it {it + 1}/{len(loader)}\n"
                    f"  loss {epoch_loss / (it + 1):.4f} | lr {lr:.2e} | frames/s {frames_per_s:,.0f} | {this_loss}"
                )
                main_line = (
                    f"  MAIN:     acc {acc_main_b:.3f} (chg: {acc_main_chg:.3f}, hold: {acc_main_hold:.3f}) | rep {acc_main_rep_b:.3f}"
                )
                c_line = (
                    f"  C-STICK:  acc {acc_c_b:.3f} (chg: {acc_c_chg:.3f}, hold: {acc_c_hold:.3f}) | rep {acc_c_rep_b:.3f}"
                )
                btn_line1 = (
                    f"  BUTTONS:  EM {em_b:.3f} (chg: {em_btn_chg:.3f}, hold: {em_btn_hold:.3f}) | F1μ {f1_b:.3f}"
                )
                btn_line2 = (
                    f"            maj F1μ {f1_maj:.3f} | rep F1μ {f1_rep:.3f} | EM_rep {em_rep:.3f}"
                )
                per_button = []
                btn_names = CONTROLLER_KEY_GROUPS["buttons"]
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
                log_lines.append(
                    f"  SHOULDER: acc {acc_sh:.3f} | maj {acc_sh_maj:.3f} | rep {acc_sh_rep:.3f}"
                )

                # VALUE HEAD (if enabled)
                if config.model.use_value_head and "value" in pred:
                    value_pred = pred["value"]  # [B, L, 1]
                    value_target = compute_value_targets(X, colmap, gamma=config.rl.gamma)

                    # Compute value prediction metrics
                    value_pred_mean = value_pred.mean().item()
                    value_target_mean = value_target.mean().item()
                    value_mse = ((value_pred - value_target) ** 2).mean().item()
                    value_mae = (value_pred - value_target).abs().mean().item()

                    # Correlation between predicted and target values
                    vp_flat = value_pred.reshape(-1)
                    vt_flat = value_target.reshape(-1)
                    vp_centered = vp_flat - vp_flat.mean()
                    vt_centered = vt_flat - vt_flat.mean()
                    correlation = (vp_centered * vt_centered).sum() / (
                            torch.sqrt((vp_centered ** 2).sum() * (vt_centered ** 2).sum()) + 1e-8
                    )

                    log_lines.append(
                        f"  VALUE:    pred {value_pred_mean:.3f} | targ {value_target_mean:.3f} | "
                        f"MSE {value_mse:.4f} | MAE {value_mae:.4f} | corr {correlation.item():.3f}"
                    )

                print("\n".join(log_lines))

                # Log to wandb (mirror console metrics)
                if use_wandb:
                    log_payload = {
                        "epoch": epoch + 1,
                        "iter": it + 1,
                        "global_step": global_step,
                        "lr": lr,
                        "loss/total": epoch_loss / (it + 1),
                        "loss/main": this_loss.get("main", 0.0),
                        "loss/c": this_loss.get("c", 0.0),
                        "loss/buttons": this_loss.get("buttons", 0.0),
                        "loss/shoulder": this_loss.get("shoulder", 0.0),
                        "loss/moe_aux": this_loss.get("moe_aux", 0.0),
                        "loss/value": this_loss.get("value", 0.0),
                        # main stick
                        "metrics/acc_main_batch": acc_main_b,
                        "metrics/acc_main_change": acc_main_chg,
                        "metrics/acc_main_hold": acc_main_hold,
                        "metrics/acc_main_rep": acc_main_rep_b,
                        # c-stick
                        "metrics/acc_c_batch": acc_c_b,
                        "metrics/acc_c_change": acc_c_chg,
                        "metrics/acc_c_hold": acc_c_hold,
                        "metrics/acc_c_rep": acc_c_rep_b,
                        # buttons
                        "metrics/buttons_em_batch": em_b,
                        "metrics/buttons_em_change": em_btn_chg,
                        "metrics/buttons_em_hold": em_btn_hold,
                        "metrics/buttons_f1_micro_batch": f1_b,
                        "metrics/buttons_f1_micro_maj": f1_maj,
                        "metrics/buttons_f1_micro_rep": f1_rep,
                        "metrics/buttons_em_rep": em_rep,
                        "throughput/frames_per_s": frames_per_s,
                    }
                    # Per-button metrics
                    try:
                        btn_names = CONTROLLER_KEY_GROUPS["buttons"]
                        for idx, name in enumerate(btn_names):
                            label = _BUTTON_PRETTY.get(name, name)
                            log_payload[f"buttons/{label}_acc"] = float(btn_match[idx].item())
                            log_payload[f"buttons/{label}_f1"] = float(btn_f1[idx].item())
                            log_payload[f"buttons/{label}_rate"] = float(btn_rate[idx].item())
                    except Exception:
                        pass
                    if 'value_pred_mean' in locals():
                        log_payload.update({
                            "value/pred_mean": value_pred_mean,
                            "value/target_mean": value_target_mean,
                            "value/mse": value_mse,
                            "value/mae": value_mae,
                            "value/corr": float(correlation.item()),
                        })
                    try:
                        wandb.log(log_payload, step=global_step)
                    except Exception:
                        pass
                    # persist latest step for robust resume
                    try:
                        last_step_file.write_text(str(global_step))
                    except Exception:
                        pass

        if (epoch + 1) % config.train.save_every_epochs == 0:
            print(f"[epoch {epoch + 1}] summary: {metrics.short_str()}")

        # Save checkpoint
        if (epoch + 1) % config.train.save_every_epochs == 0:
            ckpt = {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "config": config.train.__dict__,
                "epoch": epoch + 1,
                "global_step": global_step,
            }
            torch.save(ckpt, out_dir / f"model_ep{epoch + 1:03d}.pt")
            _prune_checkpoints(out_dir, keep=10)
            # persist latest step alongside checkpoint
            try:
                last_step_file.write_text(str(global_step))
            except Exception:
                pass

            # Log checkpoint as artifact
            if use_wandb:
                ckpt_path = out_dir / f"model_ep{epoch + 1:03d}.pt"
                try:
                    wandb.log({"checkpoint/epoch": epoch + 1})
                    art = wandb.Artifact("model", type="model")
                    art.add_file(str(ckpt_path))
                    wandb.log_artifact(art)
                except Exception:
                    pass

    # Finish wandb run
    if use_wandb:
        try:
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    init_config()
    model = GPTv7()
    print_model_diagram(model)
    train_loop(model)
