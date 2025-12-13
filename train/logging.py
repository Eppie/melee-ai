"""Logging and metric utilities for the training loop."""

from __future__ import annotations

import time
from textwrap import indent
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from constants import CONTROLLER_KEY_GROUPS, _BUTTON_PRETTY, _MAIN_STICK_LABELS
from model.nano_gpt import GPT
from train.components import (
    EpochContext,
    ForwardPassResult,
    LoggingBundle,
    TrainingComponents,
)
from train.display import format_confusion_matrix
from train.metrics import compute_binary_rates, compute_confusion_matrix, multilabel_prf


def _compute_tensor_stats_batch(tensors: List[torch.Tensor]) -> torch.Tensor:
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


def gather_logit_and_bias_metrics_batched(
    pred: TensorDict, model: GPT
) -> Dict[str, float]:
    """Gather all logit and bias metrics with a single GPU->CPU transfer."""

    def get_head_bias(module: nn.Module) -> torch.Tensor:
        # SimpleHead exposes its final projection as `fc2`
        if hasattr(module, "fc2") and isinstance(module.fc2, nn.Linear):
            return module.fc2.bias

        net = getattr(module, "net", None)
        if isinstance(net, (nn.Sequential, list, tuple)) and len(net) > 0:
            last = net[-1]
            bias = getattr(last, "bias", None)
            if bias is not None:
                return bias

        bias = getattr(module, "bias", None)
        if bias is not None:
            return bias

        raise AttributeError(
            f"{module.__class__.__name__} does not expose a bias parameter."
        )

    # Collect all tensors we need stats for
    tensor_names = [
        "logits/main",
        "logits/c",
        "logits/buttons",
        "logits/shoulder",
        "bias/input_projection",
        "bias/buttons_out",
        "bias/main_stick_out",
        "bias/c_stick_out",
        "bias/shoulder_out",
        "bias/value_out",
    ]
    tensors = [
        pred["main_stick"],
        pred["c_stick"],
        pred["buttons"],
        pred["shoulder"],
        model.projection_down.bias,
        get_head_bias(model.button_head),
        get_head_bias(model.main_stick_head),
        get_head_bias(model.c_stick_head),
        get_head_bias(model.shoulder_head),
        get_head_bias(model.value_head),
    ]

    # Compute all stats on GPU, then transfer once
    all_stats = _compute_tensor_stats_batch(tensors).cpu().tolist()

    metrics: Dict[str, float] = {}
    stat_suffixes = ["_min", "_max", "_mean", "_std"]
    for i, name in enumerate(tensor_names):
        for j, suffix in enumerate(stat_suffixes):
            metrics[f"{name}{suffix}"] = all_stats[i][j]

    return metrics


def extract_loss_breakdown(
    loss_components: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    keys = ["main", "c", "buttons", "shoulder", "value"]
    # Batch the loss component transfers
    available_keys = [k for k in keys if k in loss_components]
    loss_tensor = torch.stack([loss_components[k] for k in available_keys])
    loss_values = loss_tensor.cpu().tolist()
    result = {key: loss_values[i] for i, key in enumerate(available_keys)}
    return result


def _compute_masked_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute accuracy under a boolean mask, returning 0 if mask is empty."""
    if not mask.any():
        return torch.zeros((), device=pred.device, dtype=torch.float32)
    return (pred[mask] == target[mask]).float().mean()


def compute_confidence_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    head_name: str,
) -> Dict[str, float]:
    """Compute confidence and entropy metrics for a classification head.

    Args:
        logits: [B, L, K] logits
        targets: [B, L] target indices
        head_name: Name for metric keys (e.g., "main_stick")

    Returns:
        Dictionary with:
        - confidence/{head}/avg_maxprob: Average max probability
        - confidence/{head}/avg_maxprob_correct: Avg max prob when correct
        - entropy/{head}/mean: Average entropy in nats
    """
    B, L, K = logits.shape
    probs = F.softmax(logits, dim=-1)  # [B, L, K]

    # Max probabilities (confidence)
    max_probs = probs.max(dim=-1).values  # [B, L]

    # Check which predictions are correct
    pred_idx = logits.argmax(dim=-1)  # [B, L]
    correct_mask = pred_idx == targets  # [B, L]

    # Entropy: -sum(p * log(p))
    entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)  # [B, L]

    # Aggregate on GPU, then single transfer
    stats = torch.stack(
        [
            max_probs.mean(),
            (
                max_probs[correct_mask].mean()
                if correct_mask.any()
                else torch.tensor(0.0, device=logits.device)
            ),
            entropy.mean(),
        ]
    )

    stats_cpu = stats.cpu().tolist()

    return {
        f"confidence/{head_name}/avg_maxprob": stats_cpu[0],
        f"confidence/{head_name}/avg_maxprob_correct": stats_cpu[1],
        f"entropy/{head_name}/mean": stats_cpu[2],
    }


def compute_topk_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    head_name: str,
    k_values: List[int] = [1, 3, 5],
) -> Dict[str, float]:
    """Compute top-K accuracy for k in k_values.

    Args:
        logits: [B, L, K] logits
        targets: [B, L] target indices
        head_name: Name for metric keys
        k_values: List of K values to compute

    Returns:
        Dictionary with accuracy/{head}/topK for each K
    """
    B, L, K = logits.shape
    targets_flat = targets.reshape(-1)  # [B*L]
    logits_flat = logits.reshape(B * L, K)  # [B*L, K]

    # Filter k_values to only those <= K
    valid_k = [k for k in k_values if k <= K]
    if not valid_k:
        return {}

    max_k = max(valid_k)
    topk_indices = logits_flat.topk(max_k, dim=-1).indices  # [B*L, max_k]

    # Check if target is in top-k for each k
    targets_expanded = targets_flat.unsqueeze(-1)  # [B*L, 1]

    results = {}
    for k in valid_k:
        # Check if target in top k
        in_topk = (topk_indices[:, :k] == targets_expanded).any(dim=-1)
        acc = in_topk.float().mean()
        results[f"accuracy/{head_name}/top{k}"] = float(acc.cpu().item())

    return results


def compute_frequency_stats(
    predictions: torch.Tensor,
    num_classes: int,
    head_name: str,
    prefix: str = "freq",
) -> Dict[str, float]:
    """Compute prediction frequency statistics to detect mode collapse.

    Args:
        predictions: [B, L] predicted class indices
        num_classes: Total number of classes
        head_name: Name for metric keys
        prefix: "freq" for predictions, "tgt_freq" for targets

    Returns:
        Dictionary with:
        - {prefix}/{head}/top1_class: Most frequent class
        - {prefix}/{head}/top1_prop: Proportion of most frequent class
        - {prefix}/{head}/top5_class_{i}: i-th most frequent class
        - {prefix}/{head}/top5_prop_{i}: Proportion of i-th most frequent
        - {prefix}/{head}/diversity: Gini-Simpson diversity index
    """
    pred_flat = predictions.reshape(-1)

    # Compute class frequencies
    counts = torch.bincount(pred_flat, minlength=num_classes).float()
    props = counts / counts.sum()

    # Top-k most frequent
    topk_props, topk_classes = props.topk(min(5, num_classes))

    # Gini-Simpson diversity: 1 - sum(p_i^2)
    # Higher = more diverse, lower = mode collapse
    diversity = 1.0 - (props**2).sum()

    # Single GPU->CPU transfer
    topk_props_cpu = topk_props.cpu().tolist()
    topk_classes_cpu = topk_classes.cpu().tolist()
    diversity_cpu = float(diversity.cpu().item())

    results = {
        f"{prefix}/{head_name}/top1_class": topk_classes_cpu[0],
        f"{prefix}/{head_name}/top1_prop": topk_props_cpu[0],
        f"{prefix}/{head_name}/diversity": diversity_cpu,
    }

    # Add top-5 if we have at least 5 classes
    for i in range(min(5, len(topk_classes_cpu))):
        results[f"{prefix}/{head_name}/top5_class_{i}"] = topk_classes_cpu[i]
        results[f"{prefix}/{head_name}/top5_prop_{i}"] = topk_props_cpu[i]

    return results


def compute_temporal_consistency(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    head_name: str,
) -> Dict[str, float]:
    """Compute how often predictions/targets change vs stay constant.

    Args:
        predictions: [B, L] predicted indices
        targets: [B, L] target indices
        head_name: Name for metric keys

    Returns:
        Dictionary with:
        - consistency/{head}/pred_change_rate: How often predictions change
        - consistency/{head}/target_change_rate: How often targets change
        - consistency/{head}/change_rate_ratio: pred/target ratio
    """
    B, L = predictions.shape

    # Change masks (exclude first frame which has no previous)
    pred_changes = (predictions[:, 1:] != predictions[:, :-1]).float().mean()
    target_changes = (targets[:, 1:] != targets[:, :-1]).float().mean()

    # Ratio of pred changes to target changes
    ratio = pred_changes / (target_changes + 1e-9)

    # Single transfer
    stats = torch.stack([pred_changes, target_changes, ratio]).cpu().tolist()

    return {
        f"consistency/{head_name}/pred_change_rate": stats[0],
        f"consistency/{head_name}/target_change_rate": stats[1],
        f"consistency/{head_name}/change_rate_ratio": stats[2],
    }


def prepare_logging_bundle(
    components: TrainingComponents,
    forward_result: ForwardPassResult,
    epoch: int,
    completed_batches: int,
    total_batches: int,
    lr: float,
    frames_per_s: float,
    avg_loss_running: float,
    grad_stats: Dict[str, float],
    global_step: int,
    epoch_ctx: Optional[EpochContext] = None,
) -> LoggingBundle:
    pred = forward_result.pred
    target_info = forward_result.target_info
    config = components.config
    device = components.device
    batch_size, sequence_length, _ = pred["main_stick"].shape

    logits_main = pred["main_stick"].reshape(batch_size * sequence_length, -1)
    target_main = target_info["main_idx"].reshape(batch_size * sequence_length)
    logits_c = pred["c_stick"].reshape(batch_size * sequence_length, -1)
    target_c = target_info["c_idx"].reshape(batch_size * sequence_length)
    btn_logits = pred["buttons"]
    target_btn = target_info["buttons"]
    btn_probs = torch.sigmoid(btn_logits)

    # Compute masks on GPU
    target_main_2d = target_main.view(batch_size, sequence_length)
    target_c_2d = target_c.view(batch_size, sequence_length)

    main_change_mask = torch.zeros(
        (batch_size, sequence_length), dtype=torch.bool, device=device
    )
    main_change_mask[:, 1:] = target_main_2d[:, 1:] != target_main_2d[:, :-1]
    main_hold_mask = ~main_change_mask
    main_hold_mask[:, 0] = True

    c_change_mask = torch.zeros(
        (batch_size, sequence_length), dtype=torch.bool, device=device
    )
    c_change_mask[:, 1:] = target_c_2d[:, 1:] != target_c_2d[:, :-1]
    c_hold_mask = ~c_change_mask
    c_hold_mask[:, 0] = True

    btn_change_mask = torch.zeros(
        (batch_size, sequence_length), device=device, dtype=torch.bool
    )
    btn_change_mask[:, 1:] = torch.any(target_btn[:, 1:] != target_btn[:, :-1], dim=-1)
    btn_hold_mask = ~btn_change_mask
    btn_hold_mask[:, 0] = True

    rep_mask = torch.ones(
        (batch_size, sequence_length), dtype=torch.bool, device=device
    )
    rep_mask[:, 0] = False

    # Compute predictions
    main_pred_idx = logits_main.argmax(dim=-1)
    c_pred_idx = logits_c.argmax(dim=-1)
    main_pred = main_pred_idx.view(batch_size, sequence_length)
    c_pred = c_pred_idx.view(batch_size, sequence_length)

    # Repeat baselines

    # Button predictions
    btn_pred = (btn_probs >= 0.5).to(target_btn.dtype)
    correct_btn_em = (btn_pred == target_btn).all(dim=-1)

    # Shoulder predictions
    sh_logits = pred["shoulder"]
    sh_true_idx = target_info["shoulder_idx"]
    sh_pred_idx = sh_logits.argmax(dim=-1)

    # Compute all accuracy metrics on GPU and batch them
    acc_metrics = torch.stack(
        [
            # Main stick accuracies
            (main_pred == target_main_2d).float().mean(),
            _compute_masked_accuracy(main_pred, target_main_2d, main_change_mask),
            _compute_masked_accuracy(main_pred, target_main_2d, main_hold_mask),
            # C-stick accuracies
            (c_pred == target_c_2d).float().mean(),
            _compute_masked_accuracy(c_pred, target_c_2d, c_change_mask),
            _compute_masked_accuracy(c_pred, target_c_2d, c_hold_mask),
            # Button EM accuracies
            _compute_masked_accuracy(
                correct_btn_em.int(),
                torch.ones_like(correct_btn_em, dtype=torch.int32),
                btn_change_mask,
            ),
            _compute_masked_accuracy(
                correct_btn_em.int(),
                torch.ones_like(correct_btn_em, dtype=torch.int32),
                btn_hold_mask,
            ),
            # Shoulder accuracies
            (sh_pred_idx == sh_true_idx).float().mean(),
        ]
    )

    # Button per-class metrics (computed on GPU)
    btn_true_flat = target_btn.reshape(-1, target_btn.shape[-1]).float()
    btn_pred_flat = btn_pred.reshape(-1, btn_pred.shape[-1]).float()
    btn_match = (btn_true_flat == btn_pred_flat).float().mean(dim=0)
    btn_tp = (btn_true_flat * btn_pred_flat).sum(dim=0)
    btn_fp = ((1.0 - btn_true_flat) * btn_pred_flat).sum(dim=0)
    btn_fn = (btn_true_flat * (1.0 - btn_pred_flat)).sum(dim=0)
    eps = 1e-9
    btn_prec = btn_tp / (btn_tp + btn_fp + eps)
    btn_tpr, btn_tnr, btn_fpr, btn_fnr, _ = compute_binary_rates(
        true_positives=btn_tp,
        false_positives=btn_fp,
        false_negatives=btn_fn,
        total_count=float(btn_true_flat.shape[0]),
        eps=eps,
    )
    btn_rec = btn_tpr  # Recall equals TPR for binary button labels
    btn_f1 = 2 * btn_prec * btn_rec / (btn_prec + btn_rec + eps)
    btn_rate = btn_true_flat.mean(dim=0)

    # Multilabel PRF metrics (these return Python floats - computed on GPU then transferred)
    # Note: multilabel_prf does internal GPU->CPU transfers, but the main savings come from
    # batching all the other metrics
    em_b, _, _, f1_b, _ = multilabel_prf(target_btn, btn_pred)

    pos_rate = target_btn.float().mean(dim=(0, 1), keepdim=True)
    btn_maj_pred = (pos_rate >= 0.5).to(target_btn.dtype).expand_as(target_btn)
    _, _, _, f1_maj, _ = multilabel_prf(target_btn, btn_maj_pred)

    # Value head metrics (on GPU)
    value_target_eval = forward_result.value_target
    value_diff = forward_result.value_pred - value_target_eval
    vp_flat = forward_result.value_pred.reshape(-1)
    vt_flat = value_target_eval.reshape(-1)
    vp_centered = vp_flat - vp_flat.mean()
    vt_centered = vt_flat - vt_flat.mean()

    value_metrics = torch.stack(
        [
            forward_result.value_pred.mean(),
            value_target_eval.mean(),
            (value_diff**2).mean(),
            value_diff.abs().mean(),
            (vp_centered * vt_centered).sum()
            / (torch.sqrt((vp_centered**2).sum() * (vt_centered**2).sum()) + 1e-8),
        ]
    )

    # Shoulder majority label
    sh_flat = sh_true_idx.reshape(-1)
    sh_bincount = torch.bincount(sh_flat.to(torch.int64), minlength=sh_logits.shape[-1])
    sh_major_lbl = sh_bincount.argmax()
    acc_sh_maj = (sh_true_idx == sh_major_lbl).float().mean()

    # Batch all GPU tensor metrics for single GPU->CPU transfer
    all_scalars = torch.cat(
        [
            acc_metrics,
            value_metrics,
            acc_sh_maj.unsqueeze(0),
            btn_match,
            btn_f1,
            btn_prec,
            btn_rec,
            btn_rate,
            btn_tnr,
            btn_fpr,
            btn_fnr,
        ]
    )

    # Single GPU->CPU transfer for all tensor metrics
    all_scalars_cpu = all_scalars.cpu().tolist()

    # Unpack the values
    idx = 0
    acc_main_b, acc_main_chg, acc_main_hold = all_scalars_cpu[idx : idx + 3]
    idx += 3
    acc_c_b, acc_c_chg, acc_c_hold = all_scalars_cpu[idx : idx + 3]
    idx += 3
    em_btn_chg, em_btn_hold = all_scalars_cpu[idx : idx + 2]
    idx += 2
    acc_sh = all_scalars_cpu[idx]
    idx += 1
    (
        value_pred_mean,
        value_target_mean,
        value_mse,
        value_mae,
        correlation,
    ) = all_scalars_cpu[idx : idx + 5]
    idx += 5
    acc_sh_maj = all_scalars_cpu[idx]
    idx += 1
    num_buttons = len(CONTROLLER_KEY_GROUPS["buttons"])
    btn_match_cpu = all_scalars_cpu[idx : idx + num_buttons]
    idx += num_buttons
    btn_f1_cpu = all_scalars_cpu[idx : idx + num_buttons]
    idx += num_buttons
    btn_prec_cpu = all_scalars_cpu[idx : idx + num_buttons]
    idx += num_buttons
    btn_rec_cpu = all_scalars_cpu[idx : idx + num_buttons]
    idx += num_buttons
    btn_rate_cpu = all_scalars_cpu[idx : idx + num_buttons]
    idx += num_buttons
    btn_tnr_cpu = all_scalars_cpu[idx : idx + num_buttons]
    idx += num_buttons
    btn_fpr_cpu = all_scalars_cpu[idx : idx + num_buttons]
    idx += num_buttons
    btn_fnr_cpu = all_scalars_cpu[idx : idx + num_buttons]
    idx += num_buttons
    btn_tpr_cpu = btn_rec_cpu

    # Confusion matrix (requires CPU anyway)
    K_main = int(target_info.get("main_K", logits_main.shape[-1]))
    cm_main_b = compute_confusion_matrix(target_main, main_pred_idx, K_main)
    main_conf_str = format_confusion_matrix(
        cm_main_b,
        max_size=10,
        title="MAIN confusion",
        labels=_MAIN_STICK_LABELS[:K_main],
    )

    # Loss breakdown (single transfer)
    loss_summary = extract_loss_breakdown(forward_result.loss_components)

    # Build log lines
    log_lines: List[str] = [
        (
            f"ep {epoch + 1}/{config.train.epochs} it {completed_batches}/{total_batches}\n"
            f"  loss {avg_loss_running:.4f} | lr {lr:.2e} | frames/s {frames_per_s:,.0f} | "
            f"ls {forward_result.label_smoothing:.4f} | cw {forward_result.change_scale:.3f} | {loss_summary}"
        ),
        f"  MAIN:     acc {acc_main_b:.3f} (chg: {acc_main_chg:.3f}, hold: {acc_main_hold:.3f})",
        indent(main_conf_str, "    "),
        f"  C-STICK:  acc {acc_c_b:.3f} (chg: {acc_c_chg:.3f}, hold: {acc_c_hold:.3f})",
    ]

    btn_line1 = f"  BUTTONS:  EM {em_b:.3f} (chg: {em_btn_chg:.3f}, hold: {em_btn_hold:.3f}) | F1μ {f1_b:.3f}"
    btn_line2 = f"            maj F1μ {f1_maj:.3f}"

    per_button: List[str] = []
    per_button_rates: List[str] = []
    for i, name in enumerate(CONTROLLER_KEY_GROUPS["buttons"]):
        label = _BUTTON_PRETTY.get(name, name)
        per_button.append(
            f"{label}: acc {btn_match_cpu[i]:.3f} F1 {btn_f1_cpu[i]:.3f} rate {btn_rate_cpu[i]:.3f}"
        )
        per_button_rates.append(
            f"{label}: TPR {btn_tpr_cpu[i]:.3f} TNR {btn_tnr_cpu[i]:.3f} FPR {btn_fpr_cpu[i]:.3f} FNR {btn_fnr_cpu[i]:.3f}"
        )

    log_lines.append(btn_line1)
    log_lines.append(btn_line2)
    log_lines.append("            " + " | ".join(per_button))
    log_lines.append("            " + " | ".join(per_button_rates))
    log_lines.append(f"  SHOULDER: acc {acc_sh:.3f} | maj {acc_sh_maj:.3f}")
    log_lines.append(
        f"  VALUE:    pred {value_pred_mean:.3f} | targ {value_target_mean:.3f} | "
        f"MSE {value_mse:.4f} | MAE {value_mae:.4f} | corr {correlation:.3f}"
    )

    # Add advantage metrics if available (value_advantage strategy)
    if forward_result.advantages is not None:
        advantages = forward_result.advantages
        adv_mean = float(advantages.mean().cpu())
        adv_std = float(advantages.std().cpu())
        adv_max = float(advantages.max().cpu())
        adv_min = float(advantages.min().cpu())
        positive_mask = advantages > 0
        frac_positive = float(positive_mask.sum().cpu()) / advantages.numel()

        log_lines.append(
            f"  ADVANTAGE: μ {adv_mean:.4f} | σ {adv_std:.4f} | "
            f"range [{adv_min:.4f}, {adv_max:.4f}] | pos {frac_positive:.1%}"
        )

    # Build payload
    log_payload: Dict[str, float] = {
        "epoch": epoch + 1,
        "iter": completed_batches,
        "global_step": global_step,
        "lr": lr,
        "loss/total": avg_loss_running,
        "loss/main": loss_summary["main"],
        "loss/c": loss_summary["c"],
        "loss/buttons": loss_summary["buttons"],
        "loss/shoulder": loss_summary["shoulder"],
        "loss/value": loss_summary["value"],
        "metrics/acc_main_batch": acc_main_b,
        "metrics/acc_main_change": acc_main_chg,
        "metrics/acc_main_hold": acc_main_hold,
        "metrics/acc_c_batch": acc_c_b,
        "metrics/acc_c_change": acc_c_chg,
        "metrics/acc_c_hold": acc_c_hold,
        "metrics/buttons_em_batch": em_b,
        "metrics/buttons_em_change": em_btn_chg,
        "metrics/buttons_em_hold": em_btn_hold,
        "metrics/buttons_f1_micro_batch": f1_b,
        "metrics/buttons_f1_micro_maj": f1_maj,
        "throughput/frames_per_s": frames_per_s,
        "schedule/label_smoothing": float(forward_result.label_smoothing),
        "schedule/change_weight_scale": float(forward_result.change_scale),
    }

    # Logit and bias metrics (single transfer)
    log_payload.update(gather_logit_and_bias_metrics_batched(pred, components.model))

    # Per-head diagnostic metrics for instability detection
    if forward_result.head_diagnostics is not None:
        log_payload.update(forward_result.head_diagnostics)

    log_payload["optimizer/loss_scale"] = float(components.scaler.get_scale())

    for i, name in enumerate(CONTROLLER_KEY_GROUPS["buttons"]):
        label = _BUTTON_PRETTY.get(name, name)
        log_payload[f"buttons/{label}_acc"] = btn_match_cpu[i]
        log_payload[f"buttons/{label}_f1"] = btn_f1_cpu[i]
        log_payload[f"buttons/{label}_precision"] = btn_prec_cpu[i]
        log_payload[f"buttons/{label}_recall"] = btn_rec_cpu[i]
        log_payload[f"buttons/{label}_rate"] = btn_rate_cpu[i]
        log_payload[f"buttons/{label}_tpr"] = btn_tpr_cpu[i]
        log_payload[f"buttons/{label}_tnr"] = btn_tnr_cpu[i]
        log_payload[f"buttons/{label}_fpr"] = btn_fpr_cpu[i]
        log_payload[f"buttons/{label}_fnr"] = btn_fnr_cpu[i]

    log_payload.update(
        {
            "value/pred_mean": value_pred_mean,
            "value/target_mean": value_target_mean,
            "value/mse": value_mse,
            "value/mae": value_mae,
            "value/corr": correlation,
        }
    )

    # === NEW METRICS ===

    # 1. Confidence and entropy metrics for stick heads
    log_payload.update(
        compute_confidence_metrics(
            logits_main.reshape(batch_size, sequence_length, -1),
            target_main_2d,
            "main_stick",
        )
    )
    log_payload.update(
        compute_confidence_metrics(
            logits_c.reshape(batch_size, sequence_length, -1), target_c_2d, "c_stick"
        )
    )
    log_payload.update(compute_confidence_metrics(sh_logits, sh_true_idx, "shoulder"))

    # 2. Top-K accuracy for stick heads
    log_payload.update(
        compute_topk_accuracy(
            logits_main.reshape(batch_size, sequence_length, -1),
            target_main_2d,
            "main_stick",
            k_values=[1, 3, 5],
        )
    )
    log_payload.update(
        compute_topk_accuracy(
            logits_c.reshape(batch_size, sequence_length, -1),
            target_c_2d,
            "c_stick",
            k_values=[1, 3, 5],
        )
    )
    log_payload.update(
        compute_topk_accuracy(sh_logits, sh_true_idx, "shoulder", k_values=[1, 3, 5])
    )

    # 3. Frequency statistics (mode collapse detection)
    # For predictions
    log_payload.update(
        compute_frequency_stats(
            main_pred,
            int(target_info.get("main_K", logits_main.shape[-1])),
            "main_stick",
            prefix="freq",
        )
    )
    log_payload.update(
        compute_frequency_stats(
            c_pred,
            int(target_info.get("c_K", logits_c.shape[-1])),
            "c_stick",
            prefix="freq",
        )
    )
    log_payload.update(
        compute_frequency_stats(
            sh_pred_idx, sh_logits.shape[-1], "shoulder", prefix="freq"
        )
    )

    # For targets (ground truth distribution)
    log_payload.update(
        compute_frequency_stats(
            target_main_2d,
            int(target_info.get("main_K", logits_main.shape[-1])),
            "main_stick",
            prefix="tgt_freq",
        )
    )
    log_payload.update(
        compute_frequency_stats(
            target_c_2d,
            int(target_info.get("c_K", logits_c.shape[-1])),
            "c_stick",
            prefix="tgt_freq",
        )
    )
    log_payload.update(
        compute_frequency_stats(
            sh_true_idx, sh_logits.shape[-1], "shoulder", prefix="tgt_freq"
        )
    )

    # 4. Temporal consistency
    log_payload.update(
        compute_temporal_consistency(main_pred, target_main_2d, "main_stick")
    )
    log_payload.update(compute_temporal_consistency(c_pred, target_c_2d, "c_stick"))
    log_payload.update(
        compute_temporal_consistency(sh_pred_idx, sh_true_idx, "shoulder")
    )

    # Button temporal consistency (use exact match as binary signal)
    log_payload.update(
        compute_temporal_consistency(
            correct_btn_em.int(),
            torch.ones_like(correct_btn_em, dtype=torch.int32),
            "buttons",
        )
    )

    # 5. Sample weight statistics (imitation learning weights)
    if forward_result.imitation_weights is not None:
        weights = forward_result.imitation_weights  # [B, L]
        weights_sq = weights**2

        # Compute effective batch size: (sum w)^2 / sum(w^2)
        # This shows how many samples are effectively contributing
        eff_batch_size = (weights.sum() ** 2) / (weights_sq.sum() + 1e-9)

        weight_stats_tensor = torch.stack(
            [
                weights.mean(),
                weights.std(),
                weights.max(),
                weights.min(),
                torch.quantile(weights.flatten(), 0.95),
                torch.quantile(weights.flatten(), 0.05),
                eff_batch_size / weights.numel(),  # Normalized by actual batch size
            ]
        )
        weight_stats_cpu = weight_stats_tensor.cpu().tolist()

        log_payload.update(
            {
                "imitation/weight_mean": weight_stats_cpu[0],
                "imitation/weight_std": weight_stats_cpu[1],
                "imitation/weight_max": weight_stats_cpu[2],
                "imitation/weight_min": weight_stats_cpu[3],
                "imitation/weight_p95": weight_stats_cpu[4],
                "imitation/weight_p05": weight_stats_cpu[5],
                "imitation/effective_batch_fraction": weight_stats_cpu[6],
            }
        )

    # 5b. Advantage statistics (only for value_advantage strategy)
    if forward_result.advantages is not None:
        advantages = forward_result.advantages  # [B, L]

        # Count positive vs negative advantages
        positive_mask = advantages > 0
        negative_mask = advantages < 0
        num_positive = positive_mask.sum()
        num_negative = negative_mask.sum()
        total_nonzero = num_positive + num_negative

        # Separate stats for positive and negative advantages
        positive_advantages = advantages[positive_mask]
        negative_advantages = advantages[negative_mask]

        advantage_stats_tensor = torch.stack(
            [
                advantages.mean(),
                advantages.std(),
                advantages.max(),
                advantages.min(),
                torch.quantile(advantages.flatten(), 0.95),
                torch.quantile(advantages.flatten(), 0.05),
                num_positive.float() / (total_nonzero + 1e-9),  # Fraction positive
                (
                    positive_advantages.mean()
                    if len(positive_advantages) > 0
                    else torch.tensor(0.0, device=advantages.device)
                ),
                (
                    negative_advantages.mean()
                    if len(negative_advantages) > 0
                    else torch.tensor(0.0, device=advantages.device)
                ),
                torch.abs(advantages).mean(),  # Mean absolute advantage
            ]
        )
        advantage_stats_cpu = advantage_stats_tensor.cpu().tolist()

        log_payload.update(
            {
                "advantage/mean": advantage_stats_cpu[0],
                "advantage/std": advantage_stats_cpu[1],
                "advantage/max": advantage_stats_cpu[2],
                "advantage/min": advantage_stats_cpu[3],
                "advantage/p95": advantage_stats_cpu[4],
                "advantage/p05": advantage_stats_cpu[5],
                "advantage/frac_positive": advantage_stats_cpu[6],
                "advantage/mean_positive": advantage_stats_cpu[7],
                "advantage/mean_negative": advantage_stats_cpu[8],
                "advantage/mean_abs": advantage_stats_cpu[9],
            }
        )

    # 6. Loss variance (batch-to-batch stability)
    if epoch_ctx is not None:
        loss_std = epoch_ctx.get_loss_std()
        loss_cv = loss_std / (avg_loss_running + 1e-9)  # Coefficient of variation
        log_payload.update(
            {
                "loss/total_std": loss_std,
                "loss/total_cv": loss_cv,
            }
        )

    # 7. Gradient variance (gradient stability across batches)
    grad_variance_metrics = {
        "gradients/total_norm_variance": components.gradient_variance_tracker.get_variance(),
        "gradients/total_norm_std": components.gradient_variance_tracker.get_std(),
        "gradients/total_norm_cv": components.gradient_variance_tracker.get_cv(),
    }
    log_payload.update(grad_variance_metrics)

    # 8. Weight drift (parameter norm velocity)
    if grad_stats and "param_total_norm" in grad_stats:
        weight_drift_metrics = components.weight_drift_tracker.update(
            grad_stats["param_total_norm"], global_step
        )
        log_payload.update(weight_drift_metrics)

    # 9. Data loading metrics (chunk load time, GPU idle time, etc.)
    dataloader_metrics = components.dataloader_metrics.get_summary()
    log_payload.update(dataloader_metrics)

    return LoggingBundle(log_lines=log_lines, payload=log_payload)


from loguru import logger


def emit_logging(
    components: TrainingComponents,
    bundle: LoggingBundle,
    grad_stats: Dict[str, float],
    global_step: int,
    epoch_ctx: EpochContext,
) -> None:
    logger.info("\n" + "\n".join(bundle.log_lines))

    if components.logger.enabled:
        components.logger.log_gradients(grad_stats, step=global_step)
        components.logger.log_metrics(bundle.payload, step=global_step)
        try:
            components.last_step_file.write_text(str(global_step))
        except Exception:
            pass

    epoch_ctx.last_log_time = time.time()
    epoch_ctx.frames_since_last_log = 0.0
