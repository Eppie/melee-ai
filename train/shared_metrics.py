"""Shared metric computation utilities for training and validation.

This module contains reusable functions for computing metrics that are used
in both training logging and validation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Tuple

import torch
from tensordict import TensorDict

from constants import CONTROLLER_KEY_GROUPS, _BUTTON_PRETTY

if TYPE_CHECKING:
    from model.nano_gpt import GPT


def compute_tensor_stats_batch(tensors: List[torch.Tensor]) -> torch.Tensor:
    """Compute min, max, mean, std for multiple tensors in a single GPU operation.

    Returns a tensor of shape [len(tensors), 4] containing [min, max, mean, std] for each input.
    """
    stats_list = []
    for tensor in tensors:
        flat = tensor.detach()
        if not torch.is_floating_point(flat):
            flat = flat.float()
        else:
            flat = flat.to(torch.float32)

        min_val = torch.amin(flat)
        max_val = torch.amax(flat)
        mean_val = flat.mean()
        std_val = (
            flat.std(unbiased=False)
            if flat.numel() > 1
            else torch.zeros((), dtype=flat.dtype, device=flat.device)
        )
        stats_list.append(torch.stack([min_val, max_val, mean_val, std_val]))

    return torch.stack(stats_list)


def gather_logit_metrics(pred: TensorDict) -> Dict[str, float]:
    """Gather logit statistics for all output heads.

    Returns metrics with keys like 'logits/main_min', 'logits/main_max', etc.
    """
    tensor_names = [
        "logits/main",
        "logits/c",
        "logits/buttons",
        "logits/shoulder",
    ]
    tensors = [
        pred["main_stick"],
        pred["c_stick"],
        pred["buttons"],
        pred["shoulder"],
    ]

    all_stats = compute_tensor_stats_batch(tensors).cpu().tolist()

    metrics: Dict[str, float] = {}
    stat_suffixes = ["_min", "_max", "_mean", "_std"]
    for i, name in enumerate(tensor_names):
        for j, suffix in enumerate(stat_suffixes):
            metrics[f"{name}{suffix}"] = all_stats[i][j]

    return metrics


def compute_per_button_metrics(
    target_btn: torch.Tensor,
    btn_pred: torch.Tensor,
) -> Dict[str, float]:
    """Compute per-button accuracy, F1, precision, recall, and rate.

    Args:
        target_btn: Ground truth button states [B, L, num_buttons]
        btn_pred: Predicted button states (after thresholding) [B, L, num_buttons]

    Returns:
        Dictionary with metrics for each button (A, B, X/Y, Z, L/R).
    """
    btn_true_flat = target_btn.reshape(-1, target_btn.shape[-1]).float()
    btn_pred_flat = btn_pred.reshape(-1, btn_pred.shape[-1]).float()

    # Per-button metrics
    btn_match = (btn_true_flat == btn_pred_flat).float().mean(dim=0)
    btn_tp = (btn_true_flat * btn_pred_flat).sum(dim=0)
    btn_fp = ((1.0 - btn_true_flat) * btn_pred_flat).sum(dim=0)
    btn_fn = (btn_true_flat * (1.0 - btn_pred_flat)).sum(dim=0)

    eps = 1e-9
    btn_prec = btn_tp / (btn_tp + btn_fp + eps)
    btn_rec = btn_tp / (btn_tp + btn_fn + eps)
    btn_f1 = 2 * btn_prec * btn_rec / (btn_prec + btn_rec + eps)
    btn_rate = btn_true_flat.mean(dim=0)

    # Stack for single GPU->CPU transfer
    all_btn_stats = torch.cat([btn_match, btn_f1, btn_prec, btn_rec, btn_rate])
    all_btn_stats_cpu = all_btn_stats.cpu().tolist()

    # Unpack
    num_buttons = len(CONTROLLER_KEY_GROUPS["buttons"])
    idx = 0
    btn_match_cpu = all_btn_stats_cpu[idx : idx + num_buttons]
    idx += num_buttons
    btn_f1_cpu = all_btn_stats_cpu[idx : idx + num_buttons]
    idx += num_buttons
    btn_prec_cpu = all_btn_stats_cpu[idx : idx + num_buttons]
    idx += num_buttons
    btn_rec_cpu = all_btn_stats_cpu[idx : idx + num_buttons]
    idx += num_buttons
    btn_rate_cpu = all_btn_stats_cpu[idx : idx + num_buttons]

    metrics: Dict[str, float] = {}
    for i, name in enumerate(CONTROLLER_KEY_GROUPS["buttons"]):
        label = _BUTTON_PRETTY.get(name, name)
        metrics[f"buttons/{label}_acc"] = btn_match_cpu[i]
        metrics[f"buttons/{label}_f1"] = btn_f1_cpu[i]
        metrics[f"buttons/{label}_precision"] = btn_prec_cpu[i]
        metrics[f"buttons/{label}_recall"] = btn_rec_cpu[i]
        metrics[f"buttons/{label}_rate"] = btn_rate_cpu[i]

    return metrics


def compute_value_head_metrics(
    value_pred: torch.Tensor,
    value_target: torch.Tensor,
) -> Dict[str, float]:
    """Compute value head metrics including correlation, MAE, MSE.

    Args:
        value_pred: Predicted values [B, L] or [B*L]
        value_target: Target values [B, L] or [B*L]

    Returns:
        Dictionary with value head metrics.
    """
    value_diff = value_pred - value_target
    vp_flat = value_pred.reshape(-1)
    vt_flat = value_target.reshape(-1)
    vp_centered = vp_flat - vp_flat.mean()
    vt_centered = vt_flat - vt_flat.mean()

    value_stats = torch.stack(
        [
            value_pred.mean(),
            value_target.mean(),
            (value_diff**2).mean(),  # MSE
            value_diff.abs().mean(),  # MAE
            # Correlation
            (vp_centered * vt_centered).sum()
            / (torch.sqrt((vp_centered**2).sum() * (vt_centered**2).sum()) + 1e-8),
        ]
    )

    values = value_stats.cpu().tolist()
    return {
        "value/pred_mean": values[0],
        "value/target_mean": values[1],
        "value/mse": values[2],
        "value/mae": values[3],
        "value/corr": values[4],
    }


def compute_hold_change_accuracy(
    pred_idx: torch.Tensor,
    target_idx: torch.Tensor,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Compute overall, hold, and change accuracy for stick predictions.

    Args:
        pred_idx: Predicted indices [B, L]
        target_idx: Target indices [B, L]
        device: Device for computation

    Returns:
        Tuple of (overall_acc, change_acc, hold_acc)
    """
    B, L = pred_idx.shape

    # Create change mask
    change_mask = torch.zeros((B, L), dtype=torch.bool, device=device)
    change_mask[:, 1:] = target_idx[:, 1:] != target_idx[:, :-1]
    hold_mask = ~change_mask
    hold_mask[:, 0] = True  # First frame is always "hold"

    correct = pred_idx == target_idx

    overall_acc = correct.float().mean().item()

    if change_mask.any():
        change_acc = correct[change_mask].float().mean().item()
    else:
        change_acc = 0.0

    if hold_mask.any():
        hold_acc = correct[hold_mask].float().mean().item()
    else:
        hold_acc = 0.0

    return overall_acc, change_acc, hold_acc


def compute_button_em_change_hold(
    target_btn: torch.Tensor,
    btn_pred: torch.Tensor,
    device: torch.device,
) -> Tuple[float, float]:
    """Compute button exact match accuracy for change and hold frames.

    Args:
        target_btn: Ground truth button states [B, L, num_buttons]
        btn_pred: Predicted button states [B, L, num_buttons]
        device: Device for computation

    Returns:
        Tuple of (change_em_acc, hold_em_acc)
    """
    B, L, _ = target_btn.shape

    # Button change mask
    btn_change_mask = torch.zeros((B, L), device=device, dtype=torch.bool)
    btn_change_mask[:, 1:] = torch.any(target_btn[:, 1:] != target_btn[:, :-1], dim=-1)
    btn_hold_mask = ~btn_change_mask
    btn_hold_mask[:, 0] = True

    # Exact match per frame
    correct_btn_em = (btn_pred == target_btn).all(dim=-1)

    if btn_change_mask.any():
        change_em = correct_btn_em[btn_change_mask].float().mean().item()
    else:
        change_em = 0.0

    if btn_hold_mask.any():
        hold_em = correct_btn_em[btn_hold_mask].float().mean().item()
    else:
        hold_em = 0.0

    return change_em, hold_em
