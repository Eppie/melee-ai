"""Logging and metric utilities for the training loop."""

from __future__ import annotations

import time
from textwrap import indent
from typing import Dict, List

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


def append_tensor_stats(
    prefix: str, tensor: torch.Tensor, out: Dict[str, float]
) -> None:
    flat = tensor.detach()
    if not torch.is_floating_point(flat):
        flat = flat.float()
    else:
        flat = flat.to(torch.float32)
    out[f"{prefix}_min"] = float(torch.amin(flat).item())
    out[f"{prefix}_max"] = float(torch.amax(flat).item())
    out[f"{prefix}_mean"] = float(flat.mean().item())
    if flat.numel() > 1:
        out[f"{prefix}_std"] = float(flat.std(unbiased=False).item())
    else:
        out[f"{prefix}_std"] = 0.0


def gather_logit_metrics(pred: TensorDict) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    append_tensor_stats("logits/main", pred["main_stick"], metrics)
    append_tensor_stats("logits/c", pred["c_stick"], metrics)
    append_tensor_stats("logits/buttons", pred["buttons"], metrics)
    append_tensor_stats("logits/shoulder", pred["shoulder"], metrics)
    return metrics


def get_head_bias(module: nn.Module) -> torch.Tensor:
    net = module.net
    if isinstance(net, (nn.Sequential, list, tuple)) and len(net) > 0:
        return net[-1].bias
    return module.bias


def gather_bias_metrics(model: GPT) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    append_tensor_stats("bias/input_projection", model.projection_down.bias, metrics)
    append_tensor_stats("bias/buttons_out", get_head_bias(model.button_head), metrics)
    append_tensor_stats(
        "bias/main_stick_out", get_head_bias(model.main_stick_head), metrics
    )
    append_tensor_stats("bias/c_stick_out", get_head_bias(model.c_stick_head), metrics)
    append_tensor_stats(
        "bias/shoulder_out", get_head_bias(model.shoulder_head), metrics
    )
    append_tensor_stats("bias/value_out", get_head_bias(model.value_head), metrics)
    return metrics


def extract_loss_breakdown(
    loss_components: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    keys = ["main", "c", "buttons", "shoulder", "value"]
    return {key: float(loss_components[key].item()) for key in keys}


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

    def _to_float(val) -> float:
        if isinstance(val, (float, int)):
            return float(val)
        if hasattr(val, "item"):
            return float(val.item())
        return float(val)

    logits_main = pred["main_stick"].reshape(batch_size * sequence_length, -1)
    target_main = target_info["main_idx"].reshape(batch_size * sequence_length)
    logits_c = pred["c_stick"].reshape(batch_size * sequence_length, -1)
    target_c = target_info["c_idx"].reshape(batch_size * sequence_length)
    btn_logits = pred["buttons"]
    target_btn = target_info["buttons"]
    btn_probs = torch.sigmoid(btn_logits)

    main_change_mask = torch.zeros((batch_size, sequence_length), dtype=torch.bool, device=device)
    main_change_mask[:, 1:] = (
        target_main.view(batch_size, sequence_length)[:, 1:] != target_main.view(batch_size, sequence_length)[:, :-1]
    )
    main_hold_mask = ~main_change_mask
    main_hold_mask[:, 0] = True

    c_change_mask = torch.zeros((batch_size, sequence_length), dtype=torch.bool, device=device)
    c_change_mask[:, 1:] = target_c.view(batch_size, sequence_length)[:, 1:] != target_c.view(batch_size, sequence_length)[:, :-1]
    c_hold_mask = ~c_change_mask
    c_hold_mask[:, 0] = True

    btn_change_mask = torch.zeros((batch_size, sequence_length), device=device, dtype=torch.bool)
    btn_change_mask[:, 1:] = torch.any(target_btn[:, 1:] != target_btn[:, :-1], dim=-1)
    btn_hold_mask = ~btn_change_mask
    btn_hold_mask[:, 0] = True

    rep_mask = torch.ones((batch_size, sequence_length), dtype=torch.bool, device=device)
    rep_mask[:, 0] = False
    main_rep = torch.zeros_like(target_main.view(batch_size, sequence_length))
    c_rep = torch.zeros_like(target_c.view(batch_size, sequence_length))
    btn_rep = torch.zeros_like(target_btn)

    main_rep[:, 1:] = target_main.view(batch_size, sequence_length)[:, :-1]
    c_rep[:, 1:] = target_c.view(batch_size, sequence_length)[:, :-1]
    btn_rep[:, 1:, :] = target_btn[:, :-1, :]

    logits_main_flat = logits_main
    target_main_flat = target_main
    logits_c_flat = logits_c
    main_pred_idx = logits_main_flat.argmax(dim=-1)
    c_pred_idx = logits_c_flat.argmax(dim=-1)

    K_main = int(target_info.get("main_K", logits_main.shape[-1]))
    main_true_flat = target_main_flat
    main_pred_flat = main_pred_idx
    cm_main_b = compute_confusion_matrix(main_true_flat, main_pred_flat, K_main)
    main_conf_str = format_confusion_matrix(
        cm_main_b,
        max_size=10,
        title="MAIN confusion",
        labels=_MAIN_STICK_LABELS[:K_main],
    )

    main_pred = main_pred_idx.view(batch_size, sequence_length)
    c_pred = c_pred_idx.view(batch_size, sequence_length)

    acc_main_b = float((main_pred == target_main.view(batch_size, sequence_length)).float().mean().item())
    acc_main_chg = (
        float(
            (main_pred[main_change_mask] == target_main.view(batch_size, sequence_length)[main_change_mask])
            .float()
            .mean()
            .item()
        )
        if main_change_mask.any()
        else 0.0
    )
    acc_main_hold = (
        float(
            (main_pred[main_hold_mask] == target_main.view(batch_size, sequence_length)[main_hold_mask])
            .float()
            .mean()
            .item()
        )
        if main_hold_mask.any()
        else 0.0
    )
    acc_main_rep_b = (
        float(
            (main_rep[rep_mask] == target_main.view(batch_size, sequence_length)[rep_mask])
            .float()
            .mean()
            .item()
        )
        if rep_mask.any()
        else 0.0
    )

    acc_c_b = float((c_pred == target_c.view(batch_size, sequence_length)).float().mean().item())
    acc_c_chg = (
        float(
            (c_pred[c_change_mask] == target_c.view(batch_size, sequence_length)[c_change_mask])
            .float()
            .mean()
            .item()
        )
        if c_change_mask.any()
        else 0.0
    )
    acc_c_hold = (
        float(
            (c_pred[c_hold_mask] == target_c.view(batch_size, sequence_length)[c_hold_mask])
            .float()
            .mean()
            .item()
        )
        if c_hold_mask.any()
        else 0.0
    )
    acc_c_rep_b = (
        float((c_rep[rep_mask] == target_c.view(batch_size, sequence_length)[rep_mask]).float().mean().item())
        if rep_mask.any()
        else 0.0
    )

    btn_pred = (btn_probs >= 0.5).to(target_btn.dtype)
    em_b, _, _, f1_b, _ = multilabel_prf(target_btn, btn_pred)
    em_b = _to_float(em_b)
    f1_b = _to_float(f1_b)
    correct_btn_em = (btn_pred == target_btn).all(dim=-1)
    em_btn_chg = (
        correct_btn_em[btn_change_mask].float().mean().item()
        if btn_change_mask.any()
        else 0.0
    )
    em_btn_hold = (
        correct_btn_em[btn_hold_mask].float().mean().item()
        if btn_hold_mask.any()
        else 0.0
    )

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

    pos_rate = target_btn.float().mean(dim=(0, 1), keepdim=True)
    btn_maj_pred = (pos_rate >= 0.5).to(target_btn.dtype).expand_as(target_btn)
    _, _, _, f1_maj, _ = multilabel_prf(target_btn, btn_maj_pred)
    f1_maj = _to_float(f1_maj)
    mask_flat = rep_mask.view(batch_size * sequence_length)
    t_flat = target_btn.reshape(batch_size * sequence_length, -1)[mask_flat]
    p_flat = btn_rep.reshape(batch_size * sequence_length, -1)[mask_flat]
    em_rep, _, _, f1_rep, _ = multilabel_prf(t_flat, p_flat)
    em_rep = _to_float(em_rep)
    f1_rep = _to_float(f1_rep)

    sh_logits = pred["shoulder"]
    sh_true_idx = target_info["shoulder_idx"]
    sh_pred_idx = sh_logits.argmax(dim=-1)
    sh_rep = torch.zeros_like(sh_true_idx)
    acc_sh = float((sh_pred_idx == sh_true_idx).float().mean().item())
    sh_flat = sh_true_idx.reshape(-1).cpu()
    sh_major_lbl = (
        int(torch.bincount(sh_flat).argmax().item()) if sh_flat.numel() else 0
    )
    acc_sh_maj = float((sh_true_idx == sh_major_lbl).float().mean().item())
    sh_rep[:, 1:] = sh_true_idx[:, :-1]
    acc_sh_rep = float(
        (sh_rep[rep_mask] == sh_true_idx[rep_mask]).float().mean().item()
    )

    loss_summary = extract_loss_breakdown(forward_result.loss_components)
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
    for idx, name in enumerate(CONTROLLER_KEY_GROUPS["buttons"]):
        label = _BUTTON_PRETTY.get(name, name)
        per_button.append(
            f"{label}: acc {btn_match[idx].item():.3f} F1 {btn_f1[idx].item():.3f} rate {btn_rate[idx].item():.3f}"
        )
    log_lines.append(btn_line1)
    log_lines.append(btn_line2)
    log_lines.append("            " + " | ".join(per_button))
    log_lines.append(
        f"  SHOULDER: acc {acc_sh:.3f} | maj {acc_sh_maj:.3f} | rep {acc_sh_rep:.3f}"
    )

    value_target_eval = forward_result.value_target
    value_pred_mean = forward_result.value_pred.mean().item()
    value_target_mean = value_target_eval.mean().item()
    value_mse = ((forward_result.value_pred - value_target_eval) ** 2).mean().item()
    value_mae = (forward_result.value_pred - value_target_eval).abs().mean().item()
    vp_flat = forward_result.value_pred.reshape(-1)
    vt_flat = value_target_eval.reshape(-1)
    vp_centered = vp_flat - vp_flat.mean()
    vt_centered = vt_flat - vt_flat.mean()
    correlation = (vp_centered * vt_centered).sum() / (
        torch.sqrt((vp_centered**2).sum() * (vt_centered**2).sum()) + 1e-8
    )
    log_lines.append(
        f"  VALUE:    pred {value_pred_mean:.3f} | targ {value_target_mean:.3f} | "
        f"MSE {value_mse:.4f} | MAE {value_mae:.4f} | corr {correlation.item():.3f}"
    )

    log_payload: Dict[str, float] = {
        "epoch": epoch + 1,
        "iter": completed_batches,
        "global_step": global_step,
        "lr": lr,
        "loss/total": avg_loss_running,
        "loss/main": float(forward_result.loss_components["main"].item()),
        "loss/c": float(forward_result.loss_components["c"].item()),
        "loss/buttons": float(forward_result.loss_components["buttons"].item()),
        "loss/shoulder": float(forward_result.loss_components["shoulder"].item()),
        "loss/value": float(forward_result.loss_components["value"].item()),
        "metrics/acc_main_batch": acc_main_b,
        "metrics/acc_main_change": acc_main_chg,
        "metrics/acc_main_hold": acc_main_hold,
        "metrics/acc_main_rep": acc_main_rep_b,
        "metrics/acc_c_batch": acc_c_b,
        "metrics/acc_c_change": acc_c_chg,
        "metrics/acc_c_hold": acc_c_hold,
        "metrics/acc_c_rep": acc_c_rep_b,
        "metrics/buttons_em_batch": em_b,
        "metrics/buttons_em_change": float(em_btn_chg),
        "metrics/buttons_em_hold": float(em_btn_hold),
        "metrics/buttons_f1_micro_batch": f1_b,
        "metrics/buttons_f1_micro_maj": _to_float(f1_maj),
        "metrics/buttons_f1_micro_rep": _to_float(f1_rep),
        "metrics/buttons_em_rep": _to_float(em_rep),
        "throughput/frames_per_s": frames_per_s,
        "schedule/label_smoothing": float(forward_result.label_smoothing),
        "schedule/change_weight_scale": float(forward_result.change_scale),
    }

    log_payload.update(gather_logit_metrics(pred))
    log_payload.update(gather_bias_metrics(components.model))

    log_payload["optimizer/loss_scale"] = float(components.scaler.get_scale())

    for idx, name in enumerate(CONTROLLER_KEY_GROUPS["buttons"]):
        label = _BUTTON_PRETTY.get(name, name)
        log_payload[f"buttons/{label}_acc"] = float(btn_match[idx].item())
        log_payload[f"buttons/{label}_f1"] = float(btn_f1[idx].item())
        log_payload[f"buttons/{label}_precision"] = float(btn_prec[idx].item())
        log_payload[f"buttons/{label}_recall"] = float(btn_rec[idx].item())
        log_payload[f"buttons/{label}_rate"] = float(btn_rate[idx].item())

    log_payload.update(
        {
            "value/pred_mean": value_pred_mean,
            "value/target_mean": value_target_mean,
            "value/mse": value_mse,
            "value/mae": value_mae,
            "value/corr": float(correlation.item()),
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
