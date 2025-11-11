"""Main training script for the Melee controller model.

Consolidates train.py and train_wandb.py with optional wandb logging.
Uses train/ module utilities for all common operations.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from pprint import pformat
from textwrap import indent
from typing import Dict, List, Optional
from typing import Sequence

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.amp import GradScaler, autocast
from torch.amp.autocast_mode import is_autocast_available
from torch.nn.utils import clip_grad_norm_

from column_map import ColumnMap
from config import get_config, init_config
from constants import CONTROLLER_KEY_GROUPS, _BUTTON_PRETTY, _MAIN_STICK_LABELS
from loss import compute_loss_components
from model.nano_gpt import GPT
from train.batch_utils import (
    SampleWeightRatios,
    build_model_inputs,
    compute_component_sample_weights,
    quantize_controller_targets,
)
# Train module utilities
from train.checkpoint import (
    save_checkpoint,
    _load_latest_checkpoint,
    _prune_checkpoints,
)
from train.components import AMPContext, TrainingComponents, ForwardPassResult, EpochContext, LoggingBundle, \
    TrainingState
from train.display import format_confusion_matrix
from train.gradients import collect_gradient_diagnostics
from train.lr_schedule import _update_learning_rate
from train.metrics import (
    compute_confusion_matrix,
    multilabel_prf,
)
from train.value_head import compute_value_targets
from train.wandb_utils import (
    WandbConfig,
    WandbLogger,
    init_wandb,
    finish_wandb,
)
from utils import print_model_diagram, _resolve_device
from window_dataset import make_dataloader


def parse_cli_overrides(argv: Sequence[str]) -> Dict[str, str]:
    """
    Parses CLI arguments for --set KEY=VALUE overrides.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ns, _ = p.parse_known_args(argv)

    overrides: Dict[str, str] = {}
    for item in ns.set:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}', expected KEY=VALUE.")
        k, v = item.split("=", 1)
        overrides[k.strip()] = v.strip()
    return overrides


def _configure_amp(config, device: torch.device) -> AMPContext:
    if device.type in ("cuda", "mps") and is_autocast_available(device.type):
        device_type = device.type
    else:
        device_type = "cpu"

    amp_enabled = config.train.use_amp
    amp_context = AMPContext(
        enabled=amp_enabled,
        device_type=device_type,
        dtype=torch.float16,
    )
    return amp_context


def _report_amp_configuration(amp: AMPContext, device: torch.device) -> None:
    print(f"Using PyTorch {torch.__version__}")
    if amp.enabled:
        backend_name = "CUDA" if amp.device_type == "cuda" else "MPS"
        print(f"AMP enabled with float16 on {backend_name} backend")
    else:
        print(
            f"AMP requested but disabled for device '{device.type}';"
            " falling back to full precision."
        )


def _build_optimizer(model: GPT, config) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.train.lr,
        betas=config.train.betas,
        weight_decay=config.train.weight_decay,
    )


def _initialize_training_components(
    model: GPT,
    loader,
    ds,
    sampler,
    debug: bool,
) -> tuple[TrainingComponents, int, int, int]:
    config = get_config()
    device = _resolve_device(None)
    model = model.to(device)

    amp= _configure_amp(config, device)
    _report_amp_configuration(amp, device)

    colmap = ColumnMap.from_dataset(ds)
    value_idx = colmap.value_idx
    if value_idx is None:
        raise RuntimeError(
            "Dataset is missing 'value_target'; preprocessing must store discounted returns before training."
        )
    lw_cfg = config.loss_weights
    button_overrides = {
        "button_z": lw_cfg.button_z,
        "button_b": lw_cfg.button_b,
        "button_a": lw_cfg.button_a,
        "button_xy": lw_cfg.button_xy,
        "button_lr": lw_cfg.button_lr,
    }
    ratios = SampleWeightRatios(
        main_change=lw_cfg.main_change,
        c_change=lw_cfg.c_change,
        shoulder_change=lw_cfg.shoulder_change,
        buttons_change_default=lw_cfg.buttons_change_default,
        buttons_change_per_key=button_overrides,
        hold_base=lw_cfg.hold_base,
        value_change=lw_cfg.value_change,
    )

    optimizer = _build_optimizer(model, config)
    scaler_device = amp.device_type if amp.enabled else "cpu"
    scaler = GradScaler(device=scaler_device, enabled=amp.enabled)

    steps_per_epoch = math.ceil(len(loader))
    total_steps = config.train.epochs * steps_per_epoch

    out_dir = Path(config.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    last_step_file = out_dir / "last_step.txt"

    wandb_run = None
    if not debug:
        wandb_cfg = WandbConfig(
            project=getattr(config.train, "wandb_project", "melee-ai"),
            name=getattr(config.train, "run_name", None),
            mode=getattr(config.train, "wandb_mode", "online"),
        )
        wandb_run = init_wandb(
            config=wandb_cfg,
            run_dir=out_dir,
            hyperparameters={
                "train": dict(vars(config.train)),
                "model": dict(vars(config.model)),
                "seq_len": getattr(config, "seq_len", None),
            },
        )
    logger = WandbLogger(
        wandb_run, enabled=not debug and wandb_run is not None
    )

    start_epoch, global_step, start_iter = _load_latest_checkpoint(
        out_dir, model, optimizer, scaler, device
    )
    try:
        if last_step_file.exists():
            persisted = int(last_step_file.read_text().strip())
            global_step = max(global_step, persisted)
    except Exception:
        pass

    components = TrainingComponents(
        config=config,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        logger=logger,
        device=device,
        amp=amp,
        ratios=ratios,
        colmap=colmap,
        value_idx=value_idx,
        loader=loader,
        sampler=sampler,
        total_steps=total_steps,
        out_dir=out_dir,
        last_step_file=last_step_file,
        debug=debug,
    )

    return components, start_epoch, global_step, start_iter


def _prepare_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "X": batch["X"].to(device, non_blocking=True),
        "Y": batch["Y"].to(device, non_blocking=True),
    }


def _forward_pass(
    components: TrainingComponents,
    batch_tensors: Dict[str, torch.Tensor],
    *,
    progress: float,
    in_warmup: bool,
) -> ForwardPassResult:
    X = batch_tensors["X"]
    Y = batch_tensors["Y"]
    config = components.config
    amp = components.amp

    with autocast(
        device_type=amp.device_type,
        dtype=amp.dtype,
        enabled=amp.enabled,
    ):
        inputs_td = build_model_inputs(X, components.colmap)
        target_info = quantize_controller_targets(
            Y, components.colmap, input_domain="unit01"
        )
        pred: TensorDict = components.model(inputs_td)
        base_smoothing = components.config.train.label_smoothing
        final_smoothing = 0.5 * base_smoothing
        if in_warmup:
            label_smoothing = base_smoothing
        else:
            label_smoothing = base_smoothing + (final_smoothing - base_smoothing) * progress
        label_smoothing = float(max(label_smoothing, 0.0))

        final_change_scale = 0.5
        if in_warmup:
            imbalance_scale = 1.0
        else:
            imbalance_scale = 1.0 - (1.0 - final_change_scale) * progress
        imbalance_scale = float(max(min(imbalance_scale, 1.0), final_change_scale))

        weights = compute_component_sample_weights(
            target_info,
            components.device,
            ratios=components.ratios,
            button_names=CONTROLLER_KEY_GROUPS["buttons"],
            change_scale=imbalance_scale,
        )

        policy_loss_components = compute_loss_components(
            pred,
            target_info,
            label_smoothing=label_smoothing,
            sample_weights=weights,
            loss_config=config.loss_weights,
            ce_weight_scale=imbalance_scale,
            pos_weight_scale=imbalance_scale,
        )
        loss = policy_loss_components["total"]
        loss_components = dict(policy_loss_components)

        value_pred = pred.get("value")
        if components.value_idx is None:
            raise RuntimeError(
                "Training components must include a value target index; ensure datasets encode 'value_target'."
            )
        value_target = compute_value_targets(
            X,
            components.colmap,
            gamma=config.rl.gamma,
            reward_idx=components.value_idx,
        )
        value_loss_raw = torch.nn.functional.mse_loss(
            value_pred, value_target, reduction="none"
        ).squeeze(-1)
        value_w = weights.get("global", weights["main"])
        loss_value = (
            value_loss_raw * value_w
        ).sum() / value_w.sum().clamp_min(1e-12)
        loss = loss + config.rl.value_loss_coef * loss_value
        loss_components["value"] = loss_value

    batch_targets = {
        "main": target_info["main_idx"],
        "c": target_info["c_idx"],
        "buttons": target_info["buttons"],
        "shoulder_idx": target_info.get("shoulder_idx"),
    }
    batch_inputs = {"X": X}

    return ForwardPassResult(
        pred=pred,
        target_info=target_info,
        weights=weights,
        loss=loss,
        loss_components=loss_components,
        value_pred=value_pred,
        value_target=value_target,
        batch_inputs=batch_inputs,
        batch_targets=batch_targets,
        label_smoothing=label_smoothing,
        change_scale=imbalance_scale,
    )

def _compute_training_progress(
    epoch: int,
    iteration: int,
    total_batches: int,
    total_epochs: int,
    warmup_epochs: int,
    cooldown_epochs: int,
) -> float:
    warmup_epochs = max(warmup_epochs, 0)
    cooldown_epochs = max(cooldown_epochs, 0)
    if total_epochs <= warmup_epochs + cooldown_epochs:
        return 1.0 if epoch >= warmup_epochs else 0.0

    if epoch < warmup_epochs:
        return 0.0
    if epoch >= total_epochs - cooldown_epochs:
        return 1.0

    effective_epochs = total_epochs - warmup_epochs - cooldown_epochs
    epoch_offset = epoch - warmup_epochs
    if total_batches <= 0:
        progress = (epoch_offset + 1) / float(effective_epochs)
    else:
        batch_fraction = (iteration + 1) / float(total_batches)
        progress = (epoch_offset + batch_fraction) / float(effective_epochs)
    return float(min(max(progress, 0.0), 1.0))


def _should_log(current_iter: int) -> bool:
    return current_iter % 100 == 0


def _backward_step(
    components: TrainingComponents,
    loss: torch.Tensor,
    collect_grad_stats: bool,
) -> Dict[str, float]:
    optimizer = components.optimizer
    scaler = components.scaler
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()

    if scaler.is_enabled():
        scaler.unscale_(optimizer)

    if collect_grad_stats:
        grad_stats =  collect_gradient_diagnostics(components.model)
        grad_clip = components.config.train.grad_clip
        pre_clip_norm = float(clip_grad_norm_(components.model.parameters(), grad_clip))
        grad_stats["total_norm_pre_clip"] = pre_clip_norm
        grad_stats["total_norm_post_clip"] = min(pre_clip_norm, grad_clip)
        grad_stats["was_clipped"] = float(pre_clip_norm > grad_clip)
        grad_stats["clip_coef"] = (
            grad_clip / max(pre_clip_norm, 1e-12)
            if pre_clip_norm > grad_clip
            else 1.0
        )
    else:
        grad_stats = {}

    scaler.step(optimizer)
    scaler.update()
    return grad_stats


def _update_epoch_statistics(epoch_ctx: EpochContext, forward_result: ForwardPassResult) -> None:
    epoch_ctx.epoch_loss += forward_result.loss.item()
    B, L, _ = forward_result.pred["main_stick"].shape
    epoch_ctx.frames_since_last_log += float(B * L)


def _maybe_checkpoint_batch(
    components: TrainingComponents,
    epoch: int,
    iteration_index: int,
    completed_batches: int,
    global_step: int,
) -> None:
    if iteration_index % 5000 != 0:
        return
    ckpt_path = (
        components.out_dir / f"model_ep{epoch + 1:03d}_{completed_batches:06d}.pt"
    )
    save_checkpoint(
        path=ckpt_path,
        model=components.model,
        optimizer=components.optimizer,
        scaler=components.scaler,
        epoch=epoch + 1,
        global_step=global_step,
        config=components.config.train.__dict__,
        resume_epoch=epoch,
        resume_iter=completed_batches,
        iteration=completed_batches,
    )
    _prune_checkpoints(components.out_dir, keep=10)


def _append_tensor_stats(prefix: str, tensor: Optional[torch.Tensor], out: Dict[str, float]) -> None:
    if tensor is None:
        return
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
    out[f"{prefix}_abs_max"] = float(flat.abs().max().item())


def _gather_logit_metrics(pred: TensorDict, model: GPT) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    logits_main = pred["main_stick"]
    logits_c = pred["c_stick"]
    logits_buttons = pred["buttons"]
    logits_shoulder = pred.get("shoulder")

    _append_tensor_stats("logits/main", logits_main, metrics)
    _append_tensor_stats("logits/c", logits_c, metrics)
    _append_tensor_stats("logits/buttons", logits_buttons, metrics)
    _append_tensor_stats("logits/shoulder", logits_shoulder, metrics)
    return metrics


def _get_head_bias(module: Optional[nn.Module]) -> Optional[torch.Tensor]:
    if module is None:
        return None
    net = getattr(module, "net", None)
    if isinstance(net, (nn.Sequential, list, tuple)) and len(net) > 0:
        return getattr(net[-1], "bias", None)
    return getattr(module, "bias", None)


def _gather_bias_metrics(model: GPT) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    _append_tensor_stats("bias/input_projection", getattr(model.projection_down, "bias", None), metrics)
    _append_tensor_stats("bias/buttons_out", _get_head_bias(model.button_head), metrics)
    _append_tensor_stats("bias/main_stick_out", _get_head_bias(model.main_stick_head), metrics)
    _append_tensor_stats("bias/c_stick_out", _get_head_bias(model.c_stick_head), metrics)
    _append_tensor_stats("bias/shoulder_out", _get_head_bias(model.shoulder_head), metrics)
    _append_tensor_stats("bias/value_out", _get_head_bias(getattr(model, "value_head", None)), metrics)
    return metrics


def _extract_loss_breakdown(loss_components: Dict[str, torch.Tensor]) -> Dict[str, float]:
    keys = ["main", "c", "buttons", "shoulder", "value"]
    summary = {}
    for key in keys:
        tensor = loss_components.get(key)
        if tensor is None:
            continue
        summary[key] = float(tensor.item())
    return summary


def _prepare_logging_bundle(
    components: TrainingComponents,
    forward_result: ForwardPassResult,
    epoch_ctx: EpochContext,
    epoch: int,
    completed_batches: int,
    lr: float,
    frames_per_s: float,
    avg_loss_running: float,
    grad_stats: Optional[Dict[str, float]],
    global_step: int,
) -> LoggingBundle:
    pred = forward_result.pred
    target_info = forward_result.target_info
    config = components.config
    device = components.device
    B, L, _ = pred["main_stick"].shape

    def _to_float(val) -> float:
        if isinstance(val, (float, int)):
            return float(val)
        if hasattr(val, "item"):
            return float(val.item())
        return float(val)

    logits_main = pred["main_stick"].reshape(B * L, -1)
    target_main = target_info["main_idx"].reshape(B * L)
    logits_c = pred["c_stick"].reshape(B * L, -1)
    target_c = target_info["c_idx"].reshape(B * L)
    btn_logits = pred["buttons"]
    target_btn = target_info["buttons"]
    btn_probs = torch.sigmoid(btn_logits)

    main_change_mask = torch.zeros((B, L), dtype=torch.bool, device=device)
    main_change_mask[:, 1:] = target_main.view(B, L)[:, 1:] != target_main.view(B, L)[
        :, :-1
    ]
    main_hold_mask = ~main_change_mask
    main_hold_mask[:, 0] = True

    c_change_mask = torch.zeros((B, L), dtype=torch.bool, device=device)
    c_change_mask[:, 1:] = target_c.view(B, L)[:, 1:] != target_c.view(B, L)[:, :-1]
    c_hold_mask = ~c_change_mask
    c_hold_mask[:, 0] = True

    btn_change_mask = torch.zeros((B, L), device=device, dtype=torch.bool)
    btn_change_mask[:, 1:] = torch.any(
        target_btn[:, 1:] != target_btn[:, :-1], dim=-1
    )
    btn_hold_mask = ~btn_change_mask
    btn_hold_mask[:, 0] = True

    rep_mask = torch.ones((B, L), dtype=torch.bool, device=device)
    rep_mask[:, 0] = False
    main_rep = torch.zeros_like(target_main.view(B, L))
    c_rep = torch.zeros_like(target_c.view(B, L))
    if L > 1:
        main_rep[:, 1:] = target_main.view(B, L)[:, :-1]
        c_rep[:, 1:] = target_c.view(B, L)[:, :-1]

    main_true_flat = target_main
    main_pred_flat = logits_main.argmax(dim=-1)
    K_main = int(target_info["main_K"])
    cm_main_b = compute_confusion_matrix(main_true_flat, main_pred_flat, K_main)
    acc_main_b = float((main_pred_flat == main_true_flat).float().mean().item())
    acc_main_rep_b = (
        float(
            (
                main_rep.reshape(-1)[rep_mask.reshape(-1)]
                == main_true_flat[rep_mask.reshape(-1)]
            )
            .float()
            .mean()
            .item()
        )
        if rep_mask.any()
        else 0.0
    )
    main_conf_str = format_confusion_matrix(
        cm_main_b,
        max_size=10,
        title="MAIN confusion",
        labels=_MAIN_STICK_LABELS[:K_main],
    )
    correct_main = main_pred_flat == main_true_flat
    acc_main_chg = (
        correct_main[main_change_mask.reshape(-1)].float().mean().item()
        if main_change_mask.any()
        else 0.0
    )
    acc_main_hold = (
        correct_main[main_hold_mask.reshape(-1)].float().mean().item()
        if main_hold_mask.any()
        else 0.0
    )

    c_true_flat = target_c
    c_pred_flat = logits_c.argmax(dim=-1)
    acc_c_b = float((c_pred_flat == c_true_flat).float().mean().item())
    c_major_lbl = (
        int(torch.bincount(c_true_flat.cpu()).argmax().item())
        if c_true_flat.numel()
        else 0
    )
    acc_c_rep_b = (
        float(
            (
                c_rep.reshape(-1)[rep_mask.reshape(-1)]
                == c_true_flat[rep_mask.reshape(-1)]
            )
            .float()
            .mean()
            .item()
        )
        if rep_mask.any()
        else 0.0
    )
    correct_c = c_pred_flat == c_true_flat
    acc_c_chg = (
        correct_c[c_change_mask.reshape(-1)].float().mean().item()
        if c_change_mask.any()
        else 0.0
    )
    acc_c_hold = (
        correct_c[c_hold_mask.reshape(-1)].float().mean().item()
        if c_hold_mask.any()
        else 0.0
    )

    btn_pred = (btn_probs > 0.5).to(target_btn.dtype)
    em_b, p_b, r_b, f1_b, f1_macro_b = multilabel_prf(target_btn, btn_pred)
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
    if L > 1:
        btn_rep = torch.zeros_like(target_btn)
        btn_rep[:, 1:, :] = target_btn[:, :-1, :]
        mask_flat = rep_mask.view(B * L)
        t_flat = target_btn.reshape(B * L, -1)[mask_flat]
        p_flat = btn_rep.reshape(B * L, -1)[mask_flat]
        em_rep, _, _, f1_rep, _ = multilabel_prf(t_flat, p_flat)
        em_rep = _to_float(em_rep)
        f1_rep = _to_float(f1_rep)
    else:
        em_rep = f1_rep = 0.0

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
    if L > 1:
        sh_rep[:, 1:] = sh_true_idx[:, :-1]
        acc_sh_rep = float(
            (sh_rep[rep_mask] == sh_true_idx[rep_mask]).float().mean().item()
        )
    else:
        acc_sh_rep = 0.0

    loss_summary = _extract_loss_breakdown(forward_result.loss_components)
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

    btn_line1 = (
        f"  BUTTONS:  EM {em_b:.3f} (chg: {em_btn_chg:.3f}, hold: {em_btn_hold:.3f}) | F1μ {f1_b:.3f}"
    )
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

    if forward_result.value_target is None:
        raise RuntimeError(
            "Value targets were not populated during the forward pass; dataset-stored targets are required."
        )
    value_target_eval = forward_result.value_target
    value_pred_mean = forward_result.value_pred.mean().item()
    value_target_mean = value_target_eval.mean().item()
    value_mse = ((forward_result.value_pred - value_target_eval) ** 2).mean().item()
    value_mae = (
        (forward_result.value_pred - value_target_eval).abs().mean().item()
    )
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

    log_payload: Dict[str, float] = {"epoch": epoch + 1, "iter": completed_batches, "global_step": global_step,
                                     "lr": lr, "loss/total": avg_loss_running,
                                     "loss/main": float(forward_result.loss_components["main"].item()),
                                     "loss/c": float(forward_result.loss_components["c"].item()),
                                     "loss/buttons": float(forward_result.loss_components["buttons"].item()),
                                     "loss/shoulder": float(forward_result.loss_components["shoulder"].item()),
                                     "loss/value": float(forward_result.loss_components["value"].item()),
                                     "metrics/acc_main_batch": acc_main_b, "metrics/acc_main_change": acc_main_chg,
                                     "metrics/acc_main_hold": acc_main_hold, "metrics/acc_main_rep": acc_main_rep_b,
                                     "metrics/acc_c_batch": acc_c_b, "metrics/acc_c_change": acc_c_chg,
                                     "metrics/acc_c_hold": acc_c_hold, "metrics/acc_c_rep": acc_c_rep_b,
                                     "metrics/buttons_em_batch": em_b, "metrics/buttons_em_change": float(em_btn_chg),
                                     "metrics/buttons_em_hold": float(em_btn_hold),
                                     "metrics/buttons_f1_micro_batch": f1_b,
                                     "metrics/buttons_f1_micro_maj": _to_float(f1_maj),
                                     "metrics/buttons_f1_micro_rep": _to_float(f1_rep),
                                     "metrics/buttons_em_rep": _to_float(em_rep),
                                     "throughput/frames_per_s": frames_per_s,
                                     "schedule/label_smoothing": float(forward_result.label_smoothing),
                                     "schedule/change_weight_scale": float(forward_result.change_scale)}

    log_payload.update(_gather_logit_metrics(pred, components.model))
    log_payload.update(_gather_bias_metrics(components.model))

    if grad_stats is not None:
        grad_elems = grad_stats.get("num_elements", 0.0)
        nonfinite = grad_stats.get("nonfinite_count", 0.0)
        if grad_elems:
            log_payload["gradients/nonfinite_fraction"] = float(
                nonfinite / max(grad_elems, 1.0)
            )

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


def _emit_logging(
    components: TrainingComponents,
    bundle: LoggingBundle,
    grad_stats: Optional[Dict[str, float]],
    global_step: int,
    epoch_ctx: EpochContext,
) -> None:
    print("\n".join(bundle.log_lines))

    if components.logger.enabled:
        if grad_stats is not None:
            components.logger.log_gradients(grad_stats, step=global_step)
        components.logger.log_metrics(bundle.payload, step=global_step)
        try:
            components.last_step_file.write_text(str(global_step))
        except Exception:
            pass

    epoch_ctx.last_log_time = time.time()
    epoch_ctx.frames_since_last_log = 0.0


def _finalize_training(components: TrainingComponents) -> None:
    if not components.debug:
        finish_wandb()


def _maybe_checkpoint_epoch(
    components: TrainingComponents,
    epoch: int,
    global_step: int,
    save_condition: bool,
) -> None:
    if not save_condition:
        return
    ckpt_path = components.out_dir / f"model_ep{epoch + 1:03d}_000000.pt"
    save_checkpoint(
        path=ckpt_path,
        model=components.model,
        optimizer=components.optimizer,
        scaler=components.scaler,
        epoch=epoch + 1,
        global_step=global_step,
        config=components.config.train.__dict__,
        resume_epoch=epoch + 1,
        resume_iter=0,
        iteration=0,
    )
    _prune_checkpoints(components.out_dir, keep=10)
    try:
        components.last_step_file.write_text(str(global_step))
    except Exception:
        pass


def _run_epoch(state: TrainingState, epoch: int) -> TrainingState:
    components = state.components
    config = components.config

    if hasattr(components.sampler, "set_epoch"):
        components.sampler.set_epoch(epoch)
    components.model.train()

    try:
        total_batches = len(components.loader)
    except TypeError:
        total_batches = 0
    total_batches = max(int(total_batches), 1)

    epoch_ctx = EpochContext()
    epoch_ctx.last_log_time = time.time()
    if epoch == state.resume_epoch:
        epoch_ctx.applied_skip = state.resume_iter
        epoch_ctx.skip_remaining = state.resume_iter
        if state.resume_iter and hasattr(components.sampler, "set_start_offset"):
            try:
                components.sampler.set_start_offset(state.resume_iter)
                print(
                    f"Resuming epoch {epoch + 1}: skipping first {state.resume_iter} batches via sampler offset."
                )
                epoch_ctx.skip_remaining = 0
            except Exception as exc:
                print(
                    f"Sampler offset failed ({exc}); falling back to loading batches for skip."
                )
        elif state.resume_iter:
            print(
                f"Resuming epoch {epoch + 1}: skipping first {state.resume_iter} batches by consuming them (may take time)."
            )

    for iteration, batch in enumerate(components.loader):
        if epoch_ctx.skip_remaining:
            epoch_ctx.skip_remaining -= 1
            continue

        batch_tensors = _prepare_batch(batch, components.device)
        progress = _compute_training_progress(
            epoch,
            iteration,
            total_batches,
            config.train.epochs,
            config.train.schedule_warmup_epochs,
            config.train.schedule_cooldown_epochs,
        )
        forward_result = _forward_pass(
            components,
            batch_tensors,
            progress=progress,
            in_warmup=epoch < config.train.schedule_warmup_epochs,
        )

        current_iter = epoch_ctx.applied_skip + epoch_ctx.iters_processed
        log_this_iter = _should_log(current_iter)
        lr = _update_learning_rate(components, state.global_step)
        grad_stats = _backward_step(
            components, forward_result.loss, components.logger.enabled and log_this_iter
        )

        epoch_ctx.iters_processed += 1
        _update_epoch_statistics(epoch_ctx, forward_result)
        state.global_step += 1

        completed_batches = epoch_ctx.applied_skip + epoch_ctx.iters_processed
        _maybe_checkpoint_batch(
            components,
            epoch,
            iteration,
            completed_batches,
            state.global_step,
        )

        if log_this_iter:
            now = time.time()
            dt = max(1e-9, now - epoch_ctx.last_log_time)
            frames_per_s = epoch_ctx.frames_since_last_log / dt
            avg_loss_running = epoch_ctx.epoch_loss / max(1, epoch_ctx.iters_processed)
            bundle = _prepare_logging_bundle(
                components=components,
                forward_result=forward_result,
                epoch_ctx=epoch_ctx,
                epoch=epoch,
                completed_batches=completed_batches,
                lr=lr,
                frames_per_s=frames_per_s,
                avg_loss_running=avg_loss_running,
                grad_stats=grad_stats,
                global_step=state.global_step,
            )
            _emit_logging(
                components=components,
                bundle=bundle,
                grad_stats=grad_stats,
                global_step=state.global_step,
                epoch_ctx=epoch_ctx,
            )

    if epoch == state.resume_epoch:
        state.resume_iter = 0
        state.resume_epoch = -1

    if epoch_ctx.iters_processed:
        avg_epoch_loss = epoch_ctx.epoch_loss / max(1, epoch_ctx.iters_processed)
        print(
            f"[epoch {epoch + 1}] avg_loss {avg_epoch_loss:.4f} ({epoch_ctx.iters_processed} iters)"
        )

    _maybe_checkpoint_epoch(components, epoch, state.global_step, True)

    return state


def _make_printable_config(value):
    """Recursively convert complex config values into printable representations."""
    if isinstance(value, dict):
        return {k: _make_printable_config(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_make_printable_config(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_make_printable_config(v) for v in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)



def train_loop(
    model: GPT,
    loader,
    ds,
    sampler,
    *,
    debug: bool = False,
) -> None:
    components, start_epoch, global_step, start_iter = _initialize_training_components(
        model, loader, ds, sampler, debug
    )
    config = components.config

    state = TrainingState(
        components=components,
        global_step=global_step,
        resume_epoch=start_epoch,
        resume_iter=start_iter,
    )

    if start_epoch >= config.train.epochs:
        print(
            f"All requested epochs ({config.train.epochs}) already completed (start_epoch={start_epoch}); exiting."
        )
        _finalize_training(components)
        return

    printable_config = _make_printable_config(config.model_dump(mode="python"))
    print("Resolved training configuration:")
    print(pformat(printable_config, indent=2, width=100))

    for epoch in range(start_epoch, config.train.epochs):
        state = _run_epoch(state, epoch)

    _finalize_training(components)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Disable wandb logging for local debugging runs.",
    )
    args, remaining = parser.parse_known_args()

    overrides = parse_cli_overrides(remaining)
    # Initialize a partial config first
    init_config(overrides=overrides)

    # Create dataset and dataloader to get feature dimensions
    loader, ds, sampler = make_dataloader(get_config())
    feature_names = getattr(ds, "_feature_names_sel", ds.index.feature_names)
    target_names = getattr(ds, "_target_names_sel", ds.index.target_names)
    colmap = ColumnMap(feature_names, target_names)
    gamestate_dim = len(colmap.gamestate_idxs)
    controller_dim = len(colmap.controller_idxs)

    # Update the config with the dynamic dimensions
    config = get_config()
    config.model.input_size = (
        config.model.num_stages
        + config.model.num_characters * 2
        + config.model.num_actions * 2
        + gamestate_dim
        + controller_dim
    )

    # Create model and train
    model = GPT(config)
    print_model_diagram(model)
    train_loop(model, loader, ds, sampler, debug=args.debug)
