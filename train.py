"""Main training script for the Melee controller model.

Consolidates train.py and train_wandb.py with optional wandb logging.
Uses train/ module utilities for all common operations.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from textwrap import indent
from typing import Dict, List, Optional

import torch
from tensordict import TensorDict
from torch.amp import autocast, GradScaler
from torch.amp.autocast_mode import is_autocast_available

from column_map import ColumnMap
from config import get_config, init_config
from constants import CONTROLLER_KEY_GROUPS, _MAIN_STICK_LABELS, _BUTTON_PRETTY
from loss import compute_loss_components
from model.nano_gpt import GPT
from train.batch_utils import (
    build_model_inputs,
    quantize_controller_targets,
    SampleWeightRatios,
    compute_component_sample_weights,
)

# Train module utilities
from train.checkpoint import (
    save_checkpoint,
    _load_latest_checkpoint,
    _prune_checkpoints,
)
from train.display import format_confusion_matrix
from train.gradients import collect_gradient_diagnostics
from train.lr_schedule import cosine_lr_schedule
from train.metrics import (
    compute_confusion_matrix,
    multilabel_prf,
)
from train.value_head import build_reward_feature_index, compute_value_targets
from train.wandb_utils import (
    WandbConfig,
    WandbLogger,
    init_wandb,
    finish_wandb,
    WANDB_AVAILABLE,
)
from utils import print_model_diagram, _resolve_device
from window_dataset import make_dataloader
from typing import Sequence  # Added for parse_cli_overrides


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


def train_loop(
    model: GPT,
    loader,
    ds,
    sampler,
    *,
    debug: bool = False,
) -> None:
    device = _resolve_device(None)
    model = model.to(device)
    config = get_config()

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
    # Configure AMP support dynamically for CUDA and MPS
    if device.type in ("cuda", "mps") and is_autocast_available(device.type):
        amp_device_type = device.type
    else:
        amp_device_type = "cpu"

    amp_enabled = bool(config.train.use_amp and amp_device_type != "cpu")
    amp_dtype_cfg = getattr(config.train, "amp_dtype", "float16").lower()
    # TODO: We never want to use bfloat16
    if amp_device_type == "mps":
        autocast_dtype = torch.float16
        if config.train.use_amp and amp_dtype_cfg != "float16":
            print(
                "Warning: MPS AMP only supports float16; overriding amp_dtype to 'float16'."
            )
    else:
        autocast_dtype = torch.float16 if amp_dtype_cfg == "float16" else torch.bfloat16
    if config.train.use_amp:
        print(f"Using PyTorch {torch.__version__}")
        if amp_enabled:
            backend_name = "CUDA" if amp_device_type == "cuda" else "MPS"
            dtype_name = "float16" if autocast_dtype == torch.float16 else "bfloat16"
            print(f"AMP enabled with {dtype_name} on {backend_name} backend")
        else:
            print(
                f"AMP requested but disabled for device '{device.type}';"
                " falling back to full precision."
            )

    # Column map built from dataset metadata (only once)
    colmap = ColumnMap.from_dataset(ds)
    reward_idx = build_reward_feature_index(colmap)

    # Optimizer & (optional) simple cosine LR
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.lr,
        betas=config.train.betas,
        weight_decay=config.train.weight_decay,
    )

    # GradScaler for automatic mixed precision (only active on supported backends)
    scaler_device = amp_device_type if amp_enabled else "cpu"
    scaler = GradScaler(device=scaler_device, enabled=amp_enabled)

    steps_per_epoch = math.ceil(len(loader))
    total_steps = config.train.max_steps or (config.train.epochs * steps_per_epoch)
    global_step = 0
    start_epoch = 0

    out_dir = Path(config.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Persist most-recent global_step so wandb step stays monotonic between checkpoints
    last_step_file = out_dir / "last_step.txt"

    # Initialize Weights & Biases if available
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
        wandb_run, enabled=not debug and WANDB_AVAILABLE and wandb_run is not None
    )

    start_epoch, global_step, start_iter = _load_latest_checkpoint(
        out_dir, model, opt, scaler, device
    )
    # If we have a more recent persisted step, prefer it to keep wandb step increasing
    try:
        if last_step_file.exists():
            persisted = int(last_step_file.read_text().strip())
            global_step = max(global_step, persisted)
    except Exception:
        pass

    if start_epoch >= config.train.epochs:
        print(
            f"All requested epochs ({config.train.epochs}) already completed (start_epoch={start_epoch}); exiting."
        )
        return

    if config.train.max_steps and global_step >= config.train.max_steps:
        print(
            f"Global step {global_step} reached configured max_steps={config.train.max_steps}; exiting."
        )
        return

    resume_epoch = start_epoch
    resume_iter = start_iter

    # Main epochs
    for epoch in range(start_epoch, config.train.epochs):
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        model.train()

        epoch_loss = 0.0
        t0 = time.time()
        last_log_time = t0
        frames_since_last_log = 0.0
        skip_until = resume_iter if epoch == resume_epoch else 0
        applied_skip = skip_until if skip_until else 0
        skip_remaining = skip_until
        if skip_until and hasattr(sampler, "set_start_offset"):
            try:
                sampler.set_start_offset(skip_until)
                print(
                    f"Resuming epoch {epoch + 1}: skipping first {skip_until} batches via sampler offset."
                )
                skip_remaining = 0
                skip_until = 0
            except Exception as exc:
                print(
                    f"Sampler offset failed ({exc}); falling back to loading batches for skip."
                )
        elif skip_until:
            print(
                f"Resuming epoch {epoch + 1}: skipping first {skip_until} batches by consuming them (may take time)."
            )

        iters_processed = 0

        for it, batch in enumerate(loader):
            if skip_remaining:
                skip_remaining -= 1
                continue
            if config.train.max_steps and global_step >= config.train.max_steps:
                break

            # Move to device
            X: torch.Tensor = batch["X"].to(device, non_blocking=True)  # [B,L,F]
            Y: torch.Tensor = batch["Y"].to(device, non_blocking=True)  # [B,L,Yd]

            # Determine autocast device type and dtype
            autocast_device = amp_device_type
            amp_dtype = autocast_dtype

            value_pred: Optional[torch.Tensor] = None
            value_target: Optional[torch.Tensor] = None
            loss_value = torch.tensor(0.0, device=device)

            # Forward pass and loss computation with automatic mixed precision
            with autocast(
                device_type=autocast_device,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                # Build model inputs & target labels
                inputs_td = build_model_inputs(X, colmap)
                target_info = quantize_controller_targets(
                    Y, colmap, input_domain="unit11"
                )

                pred: TensorDict = model(
                    inputs_td
                )  # keys: buttons, main_stick, c_stick, (shoulder), optionally value
                B, L, _ = pred["main_stick"].shape
                # TODO: Make doing this configurable
                weights = compute_component_sample_weights(
                    target_info,
                    device,
                    ratios=ratios,
                    # TODO: This doesn't need to be a param, just centralize it
                    button_names=CONTROLLER_KEY_GROUPS["buttons"],
                )

                value_pred = pred.get("value", None)

                policy_loss_components = compute_loss_components(
                    pred,
                    target_info,
                    label_smoothing=config.train.label_smoothing,
                    sample_weights=weights,  # <-- dict with per-component weights
                )
                loss = policy_loss_components["total"]
                loss_main = policy_loss_components["main"]
                loss_c = policy_loss_components["c"]
                loss_btn = policy_loss_components["buttons"]
                loss_s = policy_loss_components["shoulder"]
                # TODO: Make value_head mandatory
                # TODO: Move this computation to loss.py
                if config.model.use_value_head and value_pred is not None:
                    value_target = compute_value_targets(
                        X, colmap, gamma=config.rl.gamma, reward_idx=reward_idx
                    )  # [B, L, 1]
                    value_loss_raw = torch.nn.functional.mse_loss(
                        value_pred, value_target, reduction="none"
                    ).squeeze(
                        -1
                    )  # [B, L]

                    value_w = weights.get("global", weights["main"])  # pick your poison
                    loss_value = (
                        value_loss_raw * value_w
                    ).sum() / value_w.sum().clamp_min(1e-12)
                    loss = loss + config.rl.value_loss_coef * loss_value

            logits_main = pred["main_stick"].reshape(B * L, -1)
            target_main = target_info["main_idx"].reshape(B * L)
            logits_c = pred["c_stick"].reshape(B * L, -1)
            target_c = target_info["c_idx"].reshape(B * L)
            logits_btn = pred["buttons"]  # [B,L,Kb]
            target_btn = target_info["buttons"]
            frames_since_last_log += float(B * L)
            # TODO: Consolidate this, remove fallback, stop using getattr
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
            # TODO: Don't log on the very first iter
            # TODO: Move the `10` to new LoggingConfig
            log_this_iter = current_iter % 10 == 0
            should_collect_grad_stats = logger.enabled and log_this_iter
            grad_stats: Optional[Dict[str, float]] = None

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            # Unscale before collecting diagnostics or clipping
            if scaler.is_enabled():
                scaler.unscale_(opt)

            # Collect gradient diagnostics before clipping
            if should_collect_grad_stats:
                grad_stats = collect_gradient_diagnostics(model)

            # Gradient clipping
            # TODO: Make grad clip mandatory
            # TODO: Separate out stats collection
            if config.train.grad_clip is not None and config.train.grad_clip > 0:
                from torch.nn.utils import clip_grad_norm_

                pre_clip_norm = float(
                    clip_grad_norm_(model.parameters(), config.train.grad_clip)
                )
                if grad_stats is not None:
                    grad_stats["total_norm_pre_clip"] = pre_clip_norm
                    grad_stats["total_norm_post_clip"] = min(
                        pre_clip_norm, config.train.grad_clip
                    )
                    grad_stats["was_clipped"] = float(
                        pre_clip_norm > config.train.grad_clip
                    )
                    grad_stats["clip_coef"] = (
                        config.train.grad_clip / max(pre_clip_norm, 1e-12)
                        if pre_clip_norm > config.train.grad_clip
                        else 1.0
                    )

            scaler.step(opt)
            scaler.update()

            epoch_loss += loss.item()

            # Prepare loss dict safely
            this_loss: Dict[str, float] = {
                "main": loss_main.item(),
                "c": loss_c.item(),
                "shoulder": loss_s.item(),
                "buttons": loss_btn.item(),
                # TODO: We will always use value head
                "value": (loss_value.item() if config.model.use_value_head else 0.0),
            }

            global_step += 1
            iters_processed += 1
            completed_batches = applied_skip + iters_processed

            # Throughput / logs (rank 0)
            if it % 5000 == 0:
                ckpt_path = (
                    out_dir / f"model_ep{epoch + 1:03d}_{completed_batches:06d}.pt"
                )
                save_checkpoint(
                    path=ckpt_path,
                    model=model,
                    optimizer=opt,
                    scaler=scaler,
                    epoch=epoch + 1,
                    global_step=global_step,
                    config=config.train.__dict__,
                    resume_epoch=epoch,
                    resume_iter=completed_batches,
                    iteration=completed_batches,
                )
                _prune_checkpoints(out_dir, keep=10)
            if log_this_iter:
                now = time.time()
                dt = max(1e-9, now - last_log_time)
                frames_per_s = frames_since_last_log / dt
                avg_loss_running = epoch_loss / max(1, iters_processed)

                # ---- Per-batch metrics (no running aggregation) ----
                pred_main_idx = logits_main.argmax(dim=-1).view(B, L)
                true_main_idx = target_main.view(B, L)
                pred_c_idx = logits_c.argmax(dim=-1).view(B, L)
                true_c_idx = target_c.view(B, L)
                btn_logits = logits_btn  # [B,L,Kb]
                btn_true = target_btn  # [B,L,Kb]
                btn_probs = torch.sigmoid(btn_logits)

                main_change_mask = torch.zeros_like(true_main_idx, dtype=torch.bool)
                main_change_mask[:, 1:] = true_main_idx[:, 1:] != true_main_idx[:, :-1]
                main_hold_mask = ~main_change_mask
                main_hold_mask[:, 0] = True

                c_change_mask = torch.zeros_like(true_c_idx, dtype=torch.bool)
                c_change_mask[:, 1:] = true_c_idx[:, 1:] != true_c_idx[:, :-1]
                c_hold_mask = ~c_change_mask
                c_hold_mask[:, 0] = True

                btn_change_mask = torch.zeros((B, L), device=device, dtype=torch.bool)
                btn_change_mask[:, 1:] = torch.any(
                    btn_true[:, 1:] != btn_true[:, :-1], dim=-1
                )
                btn_hold_mask = ~btn_change_mask
                btn_hold_mask[:, 0] = True

                rep_mask = torch.ones((B, L), dtype=torch.bool, device=device)
                rep_mask[:, 0] = False
                main_rep = torch.zeros_like(true_main_idx)
                c_rep = torch.zeros_like(true_c_idx)
                if L > 1:
                    main_rep[:, 1:] = true_main_idx[:, :-1]
                    c_rep[:, 1:] = true_c_idx[:, :-1]

                # ---------- Per-batch metrics & confusions ----------
                # MAIN
                main_true_flat = true_main_idx.reshape(-1)
                main_pred_flat = pred_main_idx.reshape(-1)
                K_main = int(target_info["main_K"])
                cm_main_b = compute_confusion_matrix(
                    main_true_flat, main_pred_flat, K_main
                )
                acc_main_b = float(
                    (main_pred_flat == main_true_flat).float().mean().item()
                )
                main_major_lbl = (
                    int(torch.bincount(main_true_flat.cpu()).argmax().item())
                    if main_true_flat.numel()
                    else 0
                )
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

                # C-STICK
                c_true_flat = true_c_idx.reshape(-1)
                c_pred_flat = pred_c_idx.reshape(-1)
                acc_c_b = float((c_pred_flat == c_true_flat).float().mean().item())
                c_major_lbl = (
                    int(torch.bincount(c_true_flat.cpu()).argmax().item())
                    if c_true_flat.numel()
                    else 0
                )
                acc_c_maj_b = float((c_true_flat == c_major_lbl).float().mean().item())
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

                # BUTTONS
                btn_pred = (btn_probs > 0.5).to(btn_true.dtype)
                em_b, p_b, r_b, f1_b, f1_macro_b = multilabel_prf(btn_true, btn_pred)

                btn_rep = torch.zeros_like(btn_true)
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
                em_maj, p_maj, r_maj, f1_maj, f1_macro_maj = multilabel_prf(
                    btn_true, btn_maj_pred
                )
                if L > 1:
                    btn_rep[:, 1:, :] = btn_true[:, :-1, :]
                    mask_flat = rep_mask.view(B * L)
                    t_flat = btn_true.reshape(B * L, -1)[mask_flat]
                    p_flat = btn_rep.reshape(B * L, -1)[mask_flat]
                    em_rep, p_rep, r_rep, f1_rep, f1_macro_rep = multilabel_prf(
                        t_flat, p_flat
                    )
                else:
                    em_rep = p_rep = r_rep = f1_rep = f1_macro_rep = 0.0

                # SHOULDER
                sh_logits = pred["shoulder"]
                sh_pred_idx = sh_logits.argmax(dim=-1)  # [B,L]
                sh_true_idx = target_info["shoulder_idx"]
                sh_rep = torch.zeros_like(sh_true_idx)
                acc_sh = float((sh_pred_idx == sh_true_idx).float().mean().item())
                sh_flat = sh_true_idx.reshape(-1).cpu()
                sh_major_lbl = (
                    int(torch.bincount(sh_flat).argmax().item())
                    if sh_flat.numel()
                    else 0
                )
                acc_sh_maj = float((sh_true_idx == sh_major_lbl).float().mean().item())
                if L > 1:
                    sh_rep[:, 1:] = sh_true_idx[:, :-1]
                    acc_sh_rep = float(
                        (sh_rep[rep_mask] == sh_true_idx[rep_mask])
                        .float()
                        .mean()
                        .item()
                    )
                else:
                    acc_sh_rep = 0.0

                # --- MAIN STICK ---
                correct_main = pred_main_idx == true_main_idx
                # Calculate split accuracies
                acc_main_chg = (
                    correct_main[main_change_mask].float().mean().item()
                    if main_change_mask.any()
                    else 0.0
                )
                acc_main_hold = (
                    correct_main[main_hold_mask].float().mean().item()
                    if main_hold_mask.any()
                    else 0.0
                )

                # --- C-STICK ---
                correct_c = pred_c_idx == true_c_idx
                acc_c_chg = (
                    correct_c[c_change_mask].float().mean().item()
                    if c_change_mask.any()
                    else 0.0
                )
                acc_c_hold = (
                    correct_c[c_hold_mask].float().mean().item()
                    if c_hold_mask.any()
                    else 0.0
                )

                # --- BUTTONS (Exact Match Ratio) ---
                correct_btn_em = (btn_pred == btn_true).all(dim=-1)
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

                # --- Update the log strings ---

                # ---------- Compose log ----------
                header = (
                    f"ep {epoch + 1}/{config.train.epochs} it {completed_batches}/{len(loader)}\n"
                    f"  loss {avg_loss_running:.4f} | lr {lr:.2e} | frames/s {frames_per_s:,.0f} | {this_loss}"
                )
                main_line = f"  MAIN:     acc {acc_main_b:.3f} (chg: {acc_main_chg:.3f}, hold: {acc_main_hold:.3f}) | rep {acc_main_rep_b:.3f}"
                c_line = f"  C-STICK:  acc {acc_c_b:.3f} (chg: {acc_c_chg:.3f}, hold: {acc_c_hold:.3f}) | rep {acc_c_rep_b:.3f}"
                btn_line1 = f"  BUTTONS:  EM {em_b:.3f} (chg: {em_btn_chg:.3f}, hold: {em_btn_hold:.3f}) | F1μ {f1_b:.3f}"
                btn_line2 = f"            maj F1μ {f1_maj:.3f} | rep F1μ {f1_rep:.3f} | EM_rep {em_rep:.3f}"
                per_button = []
                for idx, name in enumerate(CONTROLLER_KEY_GROUPS["buttons"]):
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
                        torch.sqrt((vp_centered**2).sum() * (vt_centered**2).sum())
                        + 1e-8
                    )

                    log_lines.append(
                        f"  VALUE:    pred {value_pred_mean:.3f} | targ {value_target_mean:.3f} | "
                        f"MSE {value_mse:.4f} | MAE {value_mae:.4f} | corr {correlation.item():.3f}"
                    )


                print("\n".join(log_lines))

                # Log to wandb (mirror console metrics)
                if logger.enabled:
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
                        logger.log_gradients(grad_stats, step=global_step)
                        grad_elems = grad_stats.get("num_elements", 0.0)
                        nonfinite = grad_stats.get("nonfinite_count", 0.0)
                        if grad_elems:
                            log_payload["gradients/nonfinite_fraction"] = float(
                                nonfinite / max(grad_elems, 1.0)
                            )
                    try:
                        log_payload["optimizer/loss_scale"] = float(scaler.get_scale())
                    except Exception:
                        pass
                    # Per-button metrics
                    try:
                        for idx, name in enumerate(CONTROLLER_KEY_GROUPS["buttons"]):
                            label = _BUTTON_PRETTY.get(name, name)
                            log_payload[f"buttons/{label}_acc"] = float(
                                btn_match[idx].item()
                            )
                            log_payload[f"buttons/{label}_f1"] = float(
                                btn_f1[idx].item()
                            )
                            log_payload[f"buttons/{label}_precision"] = float(
                                btn_prec[idx].item()
                            )
                            log_payload[f"buttons/{label}_recall"] = float(
                                btn_rec[idx].item()
                            )
                            log_payload[f"buttons/{label}_rate"] = float(
                                btn_rate[idx].item()
                            )
                    except Exception:
                        pass
                    if "value_pred_mean" in locals():
                        log_payload.update(
                            {
                                "value/pred_mean": value_pred_mean,
                                "value/target_mean": value_target_mean,
                                "value/mse": value_mse,
                                "value/mae": value_mae,
                                "value/corr": float(correlation.item()),
                            }
                        )
                    logger.log_metrics(log_payload, step=global_step)
                    # persist latest step for robust resume
                    try:
                        last_step_file.write_text(str(global_step))
                    except Exception:
                        pass

                last_log_time = now
                frames_since_last_log = 0.0

        if epoch == resume_epoch:
            resume_iter = 0
            resume_epoch = -1

        if (epoch + 1) % config.train.save_every_epochs == 0 and iters_processed:
            avg_epoch_loss = epoch_loss / max(1, iters_processed)
            print(
                f"[epoch {epoch + 1}] avg_loss {avg_epoch_loss:.4f} ({iters_processed} iters)"
            )

        # Save checkpoint
        if (epoch + 1) % config.train.save_every_epochs == 0:
            ckpt_path = out_dir / f"model_ep{epoch + 1:03d}_000000.pt"
            save_checkpoint(
                path=ckpt_path,
                model=model,
                optimizer=opt,
                scaler=scaler,
                epoch=epoch + 1,
                global_step=global_step,
                config=config.train.__dict__,
                resume_epoch=epoch + 1,
                resume_iter=0,
                iteration=0,
            )
            _prune_checkpoints(out_dir, keep=10)
            # persist latest step alongside checkpoint
            try:
                last_step_file.write_text(str(global_step))
            except Exception:
                pass

    # Finish wandb run if logging was enabled
    if not debug:
        finish_wandb()


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
