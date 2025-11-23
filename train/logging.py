"""Logging and metric utilities for the training loop."""

from __future__ import annotations

import time
from textwrap import indent
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
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
from train.metrics import compute_confusion_matrix, multilabel_prf


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
        net = module.net
        if isinstance(net, (nn.Sequential, list, tuple)) and len(net) > 0:
            return net[-1].bias
        return module.bias

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
    loss_tensor = torch.stack([loss_components[k] for k in keys])
    loss_values = loss_tensor.cpu().tolist()
    return {key: loss_values[i] for i, key in enumerate(keys)}


def _compute_masked_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute accuracy under a boolean mask, returning 0 if mask is empty."""
    if not mask.any():
        return torch.zeros((), device=pred.device, dtype=torch.float32)
    return (pred[mask] == target[mask]).float().mean()


def prepare_logging_bundle(
    components: TrainingComponents,
    forward_result: ForwardPassResult,
    epoch: int,
    completed_batches: int,
    lr: float,
    frames_per_s: float,
    avg_loss_running: float,
    grad_stats: Dict[str, float],
    global_step: int,
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
    main_rep = torch.zeros_like(target_main_2d)
    c_rep = torch.zeros_like(target_c_2d)
    btn_rep = torch.zeros_like(target_btn)
    main_rep[:, 1:] = target_main_2d[:, :-1]
    c_rep[:, 1:] = target_c_2d[:, :-1]
    btn_rep[:, 1:, :] = target_btn[:, :-1, :]

    # Button predictions
    btn_pred = (btn_probs >= 0.5).to(target_btn.dtype)
    correct_btn_em = (btn_pred == target_btn).all(dim=-1)

    # Shoulder predictions
    sh_logits = pred["shoulder"]
    sh_true_idx = target_info["shoulder_idx"]
    sh_pred_idx = sh_logits.argmax(dim=-1)
    sh_rep = torch.zeros_like(sh_true_idx)
    sh_rep[:, 1:] = sh_true_idx[:, :-1]

    # Compute all accuracy metrics on GPU and batch them
    acc_metrics = torch.stack(
        [
            # Main stick accuracies
            (main_pred == target_main_2d).float().mean(),
            _compute_masked_accuracy(main_pred, target_main_2d, main_change_mask),
            _compute_masked_accuracy(main_pred, target_main_2d, main_hold_mask),
            _compute_masked_accuracy(main_rep, target_main_2d, rep_mask),
            # C-stick accuracies
            (c_pred == target_c_2d).float().mean(),
            _compute_masked_accuracy(c_pred, target_c_2d, c_change_mask),
            _compute_masked_accuracy(c_pred, target_c_2d, c_hold_mask),
            _compute_masked_accuracy(c_rep, target_c_2d, rep_mask),
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
            _compute_masked_accuracy(sh_rep, sh_true_idx, rep_mask),
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
    btn_rec = btn_tp / (btn_tp + btn_fn + eps)
    btn_f1 = 2 * btn_prec * btn_rec / (btn_prec + btn_rec + eps)
    btn_rate = btn_true_flat.mean(dim=0)

    # Multilabel PRF metrics (these return Python floats - computed on GPU then transferred)
    # Note: multilabel_prf does internal GPU->CPU transfers, but the main savings come from
    # batching all the other metrics
    em_b, _, _, f1_b, _ = multilabel_prf(target_btn, btn_pred)

    pos_rate = target_btn.float().mean(dim=(0, 1), keepdim=True)
    btn_maj_pred = (pos_rate >= 0.5).to(target_btn.dtype).expand_as(target_btn)
    _, _, _, f1_maj, _ = multilabel_prf(target_btn, btn_maj_pred)

    mask_flat = rep_mask.view(batch_size * sequence_length)
    t_flat = target_btn.reshape(batch_size * sequence_length, -1)[mask_flat]
    p_flat = btn_rep.reshape(batch_size * sequence_length, -1)[mask_flat]
    em_rep, _, _, f1_rep, _ = multilabel_prf(t_flat, p_flat)

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
        ]
    )

    # Single GPU->CPU transfer for all tensor metrics
    all_scalars_cpu = all_scalars.cpu().tolist()

    # Unpack the values
    idx = 0
    acc_main_b, acc_main_chg, acc_main_hold, acc_main_rep_b = all_scalars_cpu[
        idx : idx + 4
    ]
    idx += 4
    acc_c_b, acc_c_chg, acc_c_hold, acc_c_rep_b = all_scalars_cpu[idx : idx + 4]
    idx += 4
    em_btn_chg, em_btn_hold = all_scalars_cpu[idx : idx + 2]
    idx += 2
    acc_sh, acc_sh_rep = all_scalars_cpu[idx : idx + 2]
    idx += 2
    value_pred_mean, value_target_mean, value_mse, value_mae, correlation = (
        all_scalars_cpu[idx : idx + 5]
    )
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
            f"ep {epoch + 1}/{config.train.epochs} it {completed_batches}/{len(components.loader)}\n"
            f"  loss {avg_loss_running:.4f} | lr {lr:.2e} | frames/s {frames_per_s:,.0f} | "
            f"ls {forward_result.label_smoothing:.4f} | cw {forward_result.change_scale:.3f} | {loss_summary}"
        ),
        f"  MAIN:     acc {acc_main_b:.3f} (chg: {acc_main_chg:.3f}, hold: {acc_main_hold:.3f}) | rep {acc_main_rep_b:.3f}",
        indent(main_conf_str, "    "),
        f"  C-STICK:  acc {acc_c_b:.3f} (chg: {acc_c_chg:.3f}, hold: {acc_c_hold:.3f}) | rep {acc_c_rep_b:.3f}",
    ]

    btn_line1 = f"  BUTTONS:  EM {em_b:.3f} (chg: {em_btn_chg:.3f}, hold: {em_btn_hold:.3f}) | F1μ {f1_b:.3f}"
    btn_line2 = (
        f"            maj F1μ {f1_maj:.3f} | rep F1μ {f1_rep:.3f} | EM_rep {em_rep:.3f}"
    )

    per_button: List[str] = []
    for i, name in enumerate(CONTROLLER_KEY_GROUPS["buttons"]):
        label = _BUTTON_PRETTY.get(name, name)
        per_button.append(
            f"{label}: acc {btn_match_cpu[i]:.3f} F1 {btn_f1_cpu[i]:.3f} rate {btn_rate_cpu[i]:.3f}"
        )

    log_lines.append(btn_line1)
    log_lines.append(btn_line2)
    log_lines.append("            " + " | ".join(per_button))
    log_lines.append(
        f"  SHOULDER: acc {acc_sh:.3f} | maj {acc_sh_maj:.3f} | rep {acc_sh_rep:.3f}"
    )
    log_lines.append(
        f"  VALUE:    pred {value_pred_mean:.3f} | targ {value_target_mean:.3f} | "
        f"MSE {value_mse:.4f} | MAE {value_mae:.4f} | corr {correlation:.3f}"
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
        "metrics/acc_main_rep": acc_main_rep_b,
        "metrics/acc_c_batch": acc_c_b,
        "metrics/acc_c_change": acc_c_chg,
        "metrics/acc_c_hold": acc_c_hold,
        "metrics/acc_c_rep": acc_c_rep_b,
        "metrics/buttons_em_batch": em_b,
        "metrics/buttons_em_change": em_btn_chg,
        "metrics/buttons_em_hold": em_btn_hold,
        "metrics/buttons_f1_micro_batch": f1_b,
        "metrics/buttons_f1_micro_maj": f1_maj,
        "metrics/buttons_f1_micro_rep": f1_rep,
        "metrics/buttons_em_rep": em_rep,
        "throughput/frames_per_s": frames_per_s,
        "schedule/label_smoothing": float(forward_result.label_smoothing),
        "schedule/change_weight_scale": float(forward_result.change_scale),
    }

    # Logit and bias metrics (single transfer)
    log_payload.update(gather_logit_and_bias_metrics_batched(pred, components.model))

    log_payload["optimizer/loss_scale"] = float(components.scaler.get_scale())

    for i, name in enumerate(CONTROLLER_KEY_GROUPS["buttons"]):
        label = _BUTTON_PRETTY.get(name, name)
        log_payload[f"buttons/{label}_acc"] = btn_match_cpu[i]
        log_payload[f"buttons/{label}_f1"] = btn_f1_cpu[i]
        log_payload[f"buttons/{label}_precision"] = btn_prec_cpu[i]
        log_payload[f"buttons/{label}_recall"] = btn_rec_cpu[i]
        log_payload[f"buttons/{label}_rate"] = btn_rate_cpu[i]

    log_payload.update(
        {
            "value/pred_mean": value_pred_mean,
            "value/target_mean": value_target_mean,
            "value/mse": value_mse,
            "value/mae": value_mae,
            "value/corr": correlation,
        }
    )

    return LoggingBundle(log_lines=log_lines, payload=log_payload)


def emit_logging(
    components: TrainingComponents,
    bundle: LoggingBundle,
    grad_stats: Dict[str, float],
    global_step: int,
    epoch_ctx: EpochContext,
) -> None:
    print("\n".join(bundle.log_lines))

    if components.logger.enabled:
        components.logger.log_gradients(grad_stats, step=global_step)
        components.logger.log_metrics(bundle.payload, step=global_step)
        try:
            components.last_step_file.write_text(str(global_step))
        except Exception:
            pass

    epoch_ctx.last_log_time = time.time()
    epoch_ctx.frames_since_last_log = 0.0
