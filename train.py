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
) -> Tuple[int, int, int]:
    checkpoints = _sorted_checkpoint_paths(directory)
    if not checkpoints:
        return 0, 0, 0

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

    resume_epoch = ckpt.get("resume_epoch", None)
    if resume_epoch is None:
        raw_epoch = int(ckpt.get("epoch", 0))
        resume_epoch = raw_epoch
        resume_iter = ckpt.get("resume_iter", ckpt.get("iteration", 0))
        if "resume_iter" not in ckpt and "iteration" not in ckpt:
            # Legacy checkpoints stored the *next* epoch to run. Adjust so we resume from the
            # previous epoch and start at the beginning of that epoch.
            if raw_epoch > 0:
                resume_epoch = raw_epoch - 1
            resume_iter = 0
    else:
        resume_iter = ckpt.get("resume_iter", 0)

    start_epoch = int(resume_epoch)
    start_iter = max(int(resume_iter), 0)
    global_step = int(ckpt.get("global_step", 0))
    return max(start_epoch, 0), max(global_step, 0), start_iter


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
# Metrics utilities
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


# -----------------------------
# Metrics helpers for per-batch logging
# -----------------------------

def _majority_flat(x: torch.Tensor) -> int:
    """Return the majority label from a 1D tensor using CPU bincount (MPS-safe)."""
    if x.numel() == 0:
        return 0
    counts = torch.bincount(x.detach().to(torch.int64).cpu())
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
    """Render a confusion matrix or its top confusions in a compact string."""
    K = cm.shape[0]
    if title is None:
        title = "confusion"

    if labels is not None:
        if len(labels) != K:
            raise ValueError("labels length must match confusion matrix dimensions")
        label_list: Sequence[str] = [str(lbl) for lbl in labels]
    else:
        label_list = [f"{i:02d}" for i in range(K)]

    if K > max_size:
        tops = _top_confusions(cm, k=10)
        if not tops:
            return f"{title}: (no confusions)"
        lines = [f"{title}: top confusions (true->pred: count, %offdiag)"]
        lines += [f"  {label_list[t]}->{label_list[p]}: {c} ({pct:.1f}%)" for t, p, c, pct in tops]
        return "\n".join(lines)

    arr = cm.numpy()
    row_sums = arr.sum(axis=1)
    diag_vals = np.diag(arr).astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        diag_pct = np.divide(diag_vals * 100.0, row_sums, out=np.zeros_like(diag_vals, dtype=float), where=row_sums > 0)

    cell_width = max(4, max(len(lbl) for lbl in label_list))
    header = " " * (cell_width + 1) + " ".join(lbl.rjust(cell_width) for lbl in label_list) + " | sum"
    lines = [f"{title}: full {K}x{K}", header]
    for i in range(K):
        row = " ".join(f"{int(v):>{cell_width}d}" for v in arr[i])
        lines.append(f"{label_list[i].rjust(cell_width)}: {row} | {int(row_sums[i]):>{cell_width}d}")
    lines.append("diag% per row: " + " ".join(f"{p:>5.1f}" for p in diag_pct))
    return "\n".join(lines)


def _multilabel_prf(true: torch.Tensor, pred: torch.Tensor) -> Tuple[float, float, float, float, float]:
    """Return EM, precision, recall, F1 (micro), and F1 (macro) for multi-label predictions."""
    if true.dim() == 2:
        true_ = true.unsqueeze(0)
        pred_ = pred.unsqueeze(0)
    else:
        true_ = true
        pred_ = pred

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


def _collect_gradient_diagnostics(model: torch.nn.Module, *, eps: float = 1e-12) -> Dict[str, float]:
    """Aggregate gradient statistics for monitoring numerical stability."""
    total_sq = 0.0
    total_abs = 0.0
    total_sum = 0.0
    grad_elems = 0
    zero_elems = 0
    nan_elems = 0
    inf_elems = 0
    max_grad_abs = 0.0
    params_with_grad = 0

    total_param_sq = 0.0
    max_param_abs = 0.0
    ratio_sum = 0.0
    ratio_max = 0.0
    ratio_min = float("inf")
    ratio_count = 0

    for param in model.parameters():
        grad = param.grad
        if grad is None:
            continue
        params_with_grad += 1

        grad_data = grad.detach()
        grad_float = grad_data.float()

        sum_sq = grad_float.pow(2).sum().item()
        total_sq += sum_sq
        abs_sum = grad_float.abs().sum().item()
        total_abs += abs_sum
        total_sum += grad_float.sum().item()

        numel = grad_float.numel()
        grad_elems += numel
        zero_elems += int((grad_float == 0).sum().item())
        nan_elems += int(torch.isnan(grad_float).sum().item())
        inf_elems += int(torch.isinf(grad_float).sum().item())

        if numel:
            max_grad_abs = max(max_grad_abs, float(grad_float.abs().max().item()))

        param_data = param.detach().float()
        total_param_sq += param_data.pow(2).sum().item()
        if param_data.numel():
            max_param_abs = max(max_param_abs, float(param_data.abs().max().item()))

        param_abs_mean = float(param_data.abs().mean().item()) if param_data.numel() else 0.0
        grad_abs_mean = float(grad_float.abs().mean().item()) if numel else 0.0
        if param_abs_mean > eps and numel:
            ratio = grad_abs_mean / max(param_abs_mean, eps)
            ratio_sum += ratio
            ratio_count += 1
            ratio_max = max(ratio_max, ratio)
            ratio_min = min(ratio_min, ratio)

    total_norm = math.sqrt(total_sq) if total_sq > 0 else 0.0
    mean_abs = total_abs / max(1, grad_elems)
    mean_val = total_sum / max(1, grad_elems)
    mean_sq = total_sq / max(1, grad_elems)
    variance = max(mean_sq - mean_val ** 2, 0.0)
    std_val = math.sqrt(variance)
    zero_fraction = zero_elems / max(1, grad_elems)

    param_total_norm = math.sqrt(total_param_sq) if total_param_sq > 0 else 0.0
    ratio_avg = ratio_sum / ratio_count if ratio_count else 0.0
    ratio_min = ratio_min if ratio_count else 0.0
    grad_to_param_ratio = total_norm / max(param_total_norm, eps)

    return {
        "total_norm": float(total_norm),
        "mean_abs": float(mean_abs),
        "mean": float(mean_val),
        "std": float(std_val),
        "max_abs": float(max_grad_abs),
        "zero_fraction": float(zero_fraction),
        "num_elements": float(grad_elems),
        "zero_count": float(zero_elems),
        "nan_count": float(nan_elems),
        "inf_count": float(inf_elems),
        "nonfinite_count": float(nan_elems + inf_elems),
        "params_with_grad": float(params_with_grad),
        "param_total_norm": float(param_total_norm),
        "param_max_abs": float(max_param_abs),
        "grad_param_ratio_mean": float(ratio_avg),
        "grad_param_ratio_max": float(ratio_max if ratio_count else 0.0),
        "grad_param_ratio_min": float(ratio_min),
        "grad_to_param_norm_ratio": float(grad_to_param_ratio),
    }


def change_boost(Y: torch.Tensor, B: int, L: int, device: torch.device, epoch: int, *, ratio: float = 10, mode: str = "batch") -> torch.Tensor:
    # Identify "change" frames
    change_mask = torch.zeros((B, L), device=device, dtype=torch.bool)
    if L > 1:
        state_changed = torch.any(Y[:, 1:] != Y[:, :-1], dim=-1)
        change_mask[:, 1:] = state_changed

    # Raw weights: 1x for hold, 10x for change
    w = torch.ones((B, L), device=device)
    w[change_mask] = ratio

    # Normalize to remove batch composition effects (keep ratio intact)
    w = w / (w.mean() + 1e-12)
    return w

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


_GAMMA_POW_CACHE: Dict[Tuple[int, float, torch.dtype, str, int], torch.Tensor] = {}


def _gamma_cache_key(length: int, gamma: float, device: torch.device, dtype: torch.dtype) -> Tuple[int, float, torch.dtype, str, int]:
    dev = torch.device(device)
    return (
        int(length),
        float(gamma),
        dtype,
        dev.type,
        dev.index if dev.index is not None else -1,
    )


def _get_gamma_powers(length: int, gamma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if length <= 0:
        return torch.empty((0,), device=device, dtype=dtype)

    key = _gamma_cache_key(length, gamma, device, dtype)
    cached = _GAMMA_POW_CACHE.get(key)
    if cached is not None and cached.device == device and cached.dtype == dtype:
        return cached

    compute_dtype = dtype
    if device.type == "cpu" and dtype == torch.float16:
        compute_dtype = torch.float32

    arange = torch.arange(length, device=device, dtype=compute_dtype)
    gamma_scalar = torch.as_tensor(gamma, device=device, dtype=compute_dtype)
    powers = torch.pow(gamma_scalar, arange)
    if compute_dtype != dtype:
        powers = powers.to(dtype=dtype)
    _GAMMA_POW_CACHE[key] = powers
    return powers


def compute_value_targets(
        X: torch.Tensor,
        colmap: ColumnMap,
        gamma: float = 0.99,
        *,
        reward_idx: Optional[RewardFeatureIdx] = None,
) -> torch.Tensor:
    """Compute discounted returns in O(B·L) using cached gamma powers and fused scans."""
    B, L, _ = X.shape
    device = X.device
    dtype = X.dtype

    if L == 0:
        return torch.empty((B, 0, 1), device=device, dtype=dtype)

    rewards = compute_frame_rewards(X, colmap, idx=reward_idx)  # [B, L]
    gamma_val = float(gamma)
    gamma_powers = _get_gamma_powers(L, gamma_val, device, rewards.dtype)

    if abs(gamma_val) < 1e-12:
        returns = rewards.clone()
    else:
        weighted = rewards * gamma_powers  # broadcast multiply
        discounted = torch.cumsum(weighted.flip(1), dim=1).flip(1)
        returns = discounted / gamma_powers.clamp_min(1e-12)

    terminal_bonus = 1.0
    terminal_vec = gamma_powers.flip(0)
    returns = returns + terminal_bonus * terminal_vec

    return returns.unsqueeze(-1)  # [B, L, 1]


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

    start_epoch, global_step, start_iter = _load_latest_checkpoint(out_dir, model, opt, scaler, device)
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
    resume_epoch = start_epoch
    resume_iter = start_iter

    # Main epochs
    for epoch in range(start_epoch, config.train.epochs):
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        model.train()

        epoch_loss = 0.0
        t0 = time.time()
        skip_until = resume_iter if epoch == resume_epoch else 0
        applied_skip = skip_until if skip_until else 0
        skip_remaining = skip_until
        if skip_until and hasattr(sampler, "set_start_offset"):
            try:
                sampler.set_start_offset(skip_until)
                print(f"Resuming epoch {epoch + 1}: skipping first {skip_until} batches via sampler offset.")
                preview_done = True
                skip_remaining = 0
                skip_until = 0
            except Exception as exc:
                print(f"Sampler offset failed ({exc}); falling back to loading batches for skip.")
        elif skip_until:
            print(f"Resuming epoch {epoch + 1}: skipping first {skip_until} batches by consuming them (may take time).")
            preview_done = True

        max_iters = config.train.steps_per_epoch
        iters_processed = 0

        for it, batch in enumerate(loader):
            if skip_remaining:
                skip_remaining -= 1
                continue
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

            value_pred: Optional[torch.Tensor] = None
            value_target: Optional[torch.Tensor] = None
            loss_value = torch.tensor(0.0, device=device)

            # Forward pass and loss computation with automatic mixed precision
            with autocast(device_type=autocast_device, dtype=amp_dtype, enabled=config.train.use_amp):
                # Build model inputs & target labels
                inputs_td = build_inputs_for_gptv7(X, colmap)
                target_info = quantize_targets(Y, colmap, input_domain="unit11")

                pred: TensorDict = model(inputs_td)  # keys: buttons, main_stick, c_stick, (shoulder), optionally value
                B, L, _ = pred["main_stick"].shape
                sample_weights = change_boost(Y, B, L, device, epoch)

                value_pred = pred.get("value", None)
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
                if config.model.use_value_head and value_pred is not None:
                    value_target = compute_value_targets(
                        X,
                        colmap,
                        gamma=config.rl.gamma,
                        reward_idx=reward_idx,
                    )  # [B, L, 1]

                    # MSE loss for value prediction
                    value_loss_raw = torch.nn.functional.mse_loss(
                        value_pred,
                        value_target,
                        reduction='none',
                    )  # [B, L, 1]

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

            lr_max = getattr(config.train, "lr_max", None) or config.train.lr
            lr = cosine_lr_schedule(
                global_step,
                total_steps,
                lr_max,
                getattr(config.train, "warmup_steps", 0),
            )
            for pg in opt.param_groups:
                pg["lr"] = lr

            current_iter = applied_skip + iters_processed
            log_this_iter = (current_iter % 100 == 0)
            should_collect_grad_stats = use_wandb and log_this_iter
            grad_stats: Optional[Dict[str, float]] = None

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            if scaler.is_enabled():
                if should_collect_grad_stats or (config.train.grad_clip is not None and config.train.grad_clip > 0):
                    scaler.unscale_(opt)

            if should_collect_grad_stats:
                grad_stats = _collect_gradient_diagnostics(model)

            if config.train.grad_clip is not None and config.train.grad_clip > 0:
                pre_clip_norm_tensor = clip_grad_norm_(model.parameters(), config.train.grad_clip)
                pre_clip_norm = float(pre_clip_norm_tensor)
                if should_collect_grad_stats and grad_stats is not None:
                    grad_stats.setdefault("total_norm", float(grad_stats.get("total_norm", pre_clip_norm)))
                    grad_stats["total_norm_pre_clip"] = pre_clip_norm
                    grad_stats["total_norm_post_clip"] = float(min(pre_clip_norm, config.train.grad_clip))
                    if pre_clip_norm > config.train.grad_clip:
                        grad_stats["clip_coef"] = float(config.train.grad_clip / max(pre_clip_norm, 1e-12))
                        grad_stats["was_clipped"] = 1.0
                    else:
                        grad_stats["clip_coef"] = 1.0
                        grad_stats["was_clipped"] = 0.0
            elif should_collect_grad_stats and grad_stats is not None:
                grad_stats.setdefault("total_norm_pre_clip", grad_stats.get("total_norm", 0.0))
                grad_stats["total_norm_post_clip"] = grad_stats.get("total_norm", 0.0)
                grad_stats["clip_coef"] = 1.0
                grad_stats["was_clipped"] = 0.0

            scaler.step(opt)
            scaler.update()

            epoch_loss += float(loss.detach().item())

            # ---- Per-batch metrics (no running aggregation) ----
            pred_main_idx = logits_main.argmax(dim=-1).view(B, L)
            true_main_idx = target_main.view(B, L)
            pred_c_idx = logits_c.argmax(dim=-1).view(B, L)
            true_c_idx = target_c.view(B, L)
            btn_logits = logits_btn  # [B,L,Kb]
            btn_true = target_btn  # [B,L,Kb]
            btn_probs = probs_btn

            main_change_mask = torch.zeros_like(true_main_idx, dtype=torch.bool)
            main_change_mask[:, 1:] = (true_main_idx[:, 1:] != true_main_idx[:, :-1])
            main_hold_mask = ~main_change_mask
            main_hold_mask[:, 0] = True

            c_change_mask = torch.zeros_like(true_c_idx, dtype=torch.bool)
            c_change_mask[:, 1:] = (true_c_idx[:, 1:] != true_c_idx[:, :-1])
            c_hold_mask = ~c_change_mask
            c_hold_mask[:, 0] = True

            btn_change_mask = torch.zeros((B, L), device=device, dtype=torch.bool)
            btn_change_mask[:, 1:] = torch.any(btn_true[:, 1:] != btn_true[:, :-1], dim=-1)
            btn_hold_mask = ~btn_change_mask
            btn_hold_mask[:, 0] = True

            rep_mask = torch.ones((B, L), dtype=torch.bool, device=device)
            rep_mask[:, 0] = False
            main_rep = torch.zeros_like(true_main_idx)
            c_rep = torch.zeros_like(true_c_idx)
            if L > 1:
                main_rep[:, 1:] = true_main_idx[:, :-1]
                c_rep[:, 1:] = true_c_idx[:, :-1]

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
            iters_processed += 1
            completed_batches = applied_skip + iters_processed

            # Throughput / logs (rank 0)
            if it % 5000 == 0:
                ckpt = {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scaler": scaler.state_dict(),
                    "config": config.train.__dict__,
                    "epoch": epoch + 1,
                    "resume_epoch": epoch,
                    "resume_iter": completed_batches,
                    "iteration": completed_batches,
                    "global_step": global_step,
                }
                torch.save(ckpt, out_dir / f"model_ep{epoch + 1:03d}_{it:03d}.pt")
                _prune_checkpoints(out_dir, keep=10)
            if log_this_iter:

                dt = max(1e-9, time.time() - t0)
                B_cur, L_cur, F_cur = X.shape
                # frames/s: each frame is a token in [B,L]
                frames_per_batch = B_cur * L_cur
                frames_per_s = iters_processed * frames_per_batch / dt
                avg_loss_running = epoch_loss / max(1, iters_processed)

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
                    f"ep {epoch + 1}/{config.train.epochs} it {completed_batches}/{len(loader)}\n"
                    f"  loss {avg_loss_running:.4f} | lr {lr:.2e} | frames/s {frames_per_s:,.0f} | {this_loss}"
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
                if config.model.use_value_head and value_pred is not None:
                    if value_target is None:
                        value_target_eval = compute_value_targets(
                            X,
                            colmap,
                            gamma=config.rl.gamma,
                            reward_idx=reward_idx,
                        )
                    else:
                        value_target_eval = value_target
                    # Compute value prediction metrics
                    value_pred_mean = value_pred.mean().item()
                    value_target_mean = value_target_eval.mean().item()
                    value_mse = ((value_pred - value_target_eval) ** 2).mean().item()
                    value_mae = (value_pred - value_target_eval).abs().mean().item()

                    # Correlation between predicted and target values
                    vp_flat = value_pred.reshape(-1)
                    vt_flat = value_target_eval.reshape(-1)
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
                        "iter": completed_batches,
                        "global_step": global_step,
                        "lr": lr,
                        "loss/total": avg_loss_running,
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
                    if grad_stats is not None:
                        log_payload.update({
                            "gradients/total_norm": grad_stats.get("total_norm", 0.0),
                            "gradients/total_norm_pre_clip": grad_stats.get("total_norm_pre_clip", grad_stats.get("total_norm", 0.0)),
                            "gradients/total_norm_post_clip": grad_stats.get("total_norm_post_clip", grad_stats.get("total_norm", 0.0)),
                            "gradients/mean_abs": grad_stats.get("mean_abs", 0.0),
                            "gradients/mean": grad_stats.get("mean", 0.0),
                            "gradients/std": grad_stats.get("std", 0.0),
                            "gradients/max_abs": grad_stats.get("max_abs", 0.0),
                            "gradients/zero_fraction": grad_stats.get("zero_fraction", 0.0),
                            "gradients/grad_to_param_norm_ratio": grad_stats.get("grad_to_param_norm_ratio", 0.0),
                            "gradients/grad_param_ratio_mean": grad_stats.get("grad_param_ratio_mean", 0.0),
                            "gradients/grad_param_ratio_max": grad_stats.get("grad_param_ratio_max", 0.0),
                            "gradients/grad_param_ratio_min": grad_stats.get("grad_param_ratio_min", 0.0),
                            "gradients/nonfinite_count": grad_stats.get("nonfinite_count", 0.0),
                            "gradients/nan_count": grad_stats.get("nan_count", 0.0),
                            "gradients/inf_count": grad_stats.get("inf_count", 0.0),
                            "gradients/params_with_grad": grad_stats.get("params_with_grad", 0.0),
                            "gradients/clip_coef": grad_stats.get("clip_coef", 1.0),
                            "gradients/was_clipped": grad_stats.get("was_clipped", 0.0),
                            "gradients/num_elements": grad_stats.get("num_elements", 0.0),
                            "parameters/total_norm": grad_stats.get("param_total_norm", 0.0),
                            "parameters/max_abs": grad_stats.get("param_max_abs", 0.0),
                        })
                        grad_elems = grad_stats.get("num_elements", 0.0)
                        nonfinite = grad_stats.get("nonfinite_count", 0.0)
                        if grad_elems:
                            log_payload["gradients/nonfinite_fraction"] = float(nonfinite / max(grad_elems, 1.0))
                    try:
                        log_payload["optimizer/loss_scale"] = float(scaler.get_scale())
                    except Exception:
                        pass
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

        if epoch == resume_epoch:
            resume_iter = 0
            resume_epoch = -1

        if (epoch + 1) % config.train.save_every_epochs == 0 and iters_processed:
            avg_epoch_loss = epoch_loss / max(1, iters_processed)
            print(f"[epoch {epoch + 1}] avg_loss {avg_epoch_loss:.4f} ({iters_processed} iters)")

        # Save checkpoint
        if (epoch + 1) % config.train.save_every_epochs == 0:
            ckpt = {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "config": config.train.__dict__,
                "epoch": epoch + 1,
                "resume_epoch": epoch + 1,
                "resume_iter": 0,
                "iteration": 0,
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
