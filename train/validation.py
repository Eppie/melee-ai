"""Validation utilities for computing metrics during training."""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from column_map import ColumnMap
from constants import CONTROLLER_KEY_GROUPS
from controller_quantization import quantize_targets
from loss import compute_loss_components
from train.batch_utils import (
    SampleWeightRatios,
    build_model_inputs,
    compute_component_sample_weights,
)
from train.metrics import multilabel_prf
from train.value_head import compute_value_targets
from window_dataset import RandomWindowSampler

if TYPE_CHECKING:
    from model.nano_gpt import GPT
    from train.components import TrainingComponents

# Global cache for validation dataset and dataloader
_VAL_CACHE: Dict[str, Any] = {}


def _build_sample_weight_ratios(loss_cfg) -> SampleWeightRatios:
    """Mirror the training-time ratios derived from LossConfig."""
    button_overrides = {
        "button_z": loss_cfg.button_z,
        "button_b": loss_cfg.button_b,
        "button_a": loss_cfg.button_a,
        "button_xy": loss_cfg.button_xy,
        "button_lr": loss_cfg.button_lr,
    }
    return SampleWeightRatios(
        main_change=loss_cfg.main_change,
        c_change=loss_cfg.c_change,
        shoulder_change=loss_cfg.shoulder_change,
        buttons_change_default=loss_cfg.buttons_change_default,
        buttons_change_per_key=button_overrides,
        hold_base=loss_cfg.hold_base,
        value_change=loss_cfg.value_change,
    )


def _get_validation_loader(
    config,
    batch_size: int,
    window_stride: int = 256,
) -> Tuple[DataLoader, Any, ColumnMap, Optional[int]]:
    """Get or create cached validation dataloader.

    Returns:
        Tuple of (DataLoader, dataset, column_map, value_idx)
    """
    # Lazy import to avoid circular import with validation.py
    from validation import PreloadedWindowDataset

    cache_key = f"{config.zarr.validation_root}_{batch_size}_{window_stride}"

    if cache_key in _VAL_CACHE:
        return _VAL_CACHE[cache_key]

    data_root = Path(config.zarr.validation_root).expanduser()
    if not data_root.is_absolute():
        project_root = Path(__file__).resolve().parent.parent
        data_root = (project_root / data_root).resolve()

    if not data_root.exists():
        raise FileNotFoundError(f"Validation data root not found: {data_root}")

    dataset = PreloadedWindowDataset(
        str(data_root),
        progress=True,
    )

    sampler = RandomWindowSampler(
        index=dataset.index,
        stride=window_stride,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,  # In-memory dataset, no workers needed
        pin_memory=False,
        drop_last=False,
    )

    colmap = ColumnMap.from_dataset(dataset)
    value_idx = colmap.value_idx

    _VAL_CACHE[cache_key] = (loader, dataset, colmap, value_idx)
    return loader, dataset, colmap, value_idx


def run_validation(
    model: "GPT",
    device: torch.device,
    config,
    max_batches: Optional[int] = None,
    batch_size: Optional[int] = None,
    imbalance_scale: float = 1.0,
) -> Dict[str, float]:
    """Run validation and return metrics dictionary suitable for wandb logging.

    Args:
        model: The model to evaluate.
        device: Device to run evaluation on.
        config: Training configuration.
        max_batches: Maximum number of batches to evaluate (None for all).
        batch_size: Override batch size (defaults to config.train.batch_size).
        imbalance_scale: Current imbalance scale from training (default 1.0).

    Returns:
        Dictionary of validation metrics with "val/" prefix.
    """
    if batch_size is None:
        batch_size = config.train.batch_size

    loader, dataset, colmap, value_idx = _get_validation_loader(
        config,
        batch_size=batch_size,
        window_stride=256,
    )

    ratios = _build_sample_weight_ratios(config.loss_weights)

    metrics = defaultdict(float)
    loss_sums: Dict[str, float] = {
        key: 0.0 for key in ("total", "main", "c", "buttons", "shoulder", "value")
    }

    total_frames = 0
    batches_processed = 0
    start_time = time.time()

    model.eval()
    total_batches = len(loader)

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader, start=1):
            if max_batches is not None and batch_idx > max_batches:
                break

            X: torch.Tensor = batch["X"].to(device, non_blocking=True)
            Y: torch.Tensor = batch["Y"].to(device, non_blocking=True)

            inputs_td = build_model_inputs(X, colmap)
            target_info = quantize_targets(Y, colmap, input_domain="unit01")
            weights = compute_component_sample_weights(
                target_info,
                device,
                ratios=ratios,
                button_names=CONTROLLER_KEY_GROUPS["buttons"],
                change_scale=imbalance_scale,
            )

            pred = model(inputs_td)

            logits_main = pred["main_stick"]
            logits_c = pred["c_stick"]
            logits_btn = pred["buttons"]

            # Match training's imbalance scale for comparable loss values
            loss_components = compute_loss_components(
                pred,
                target_info,
                label_smoothing=config.train.label_smoothing,
                sample_weights=weights,
                loss_config=config.loss_weights,
                ce_weight_scale=imbalance_scale,
                pos_weight_scale=imbalance_scale,
            )

            for key, value in loss_components.items():
                loss_sums[key] += value.item()

            # Compute value loss (same as training)
            value_pred = pred["value"]
            if value_idx is None:
                raise RuntimeError(
                    "value_idx is None during validation. Pre-computed value_target column "
                    "is required because X features are transformed. Ensure 'value_target' "
                    "is in your dataset schema and zarr files."
                )
            value_target = compute_value_targets(
                X,
                colmap,
                gamma=config.rl.gamma,
                reward_idx=value_idx,
            )
            # Unweighted loss to match training (value head learns all states equally)
            loss_value = torch.nn.functional.mse_loss(value_pred, value_target)
            loss_sums["value"] += loss_value.item()

            pred_main_idx = logits_main.argmax(dim=-1)
            pred_c_idx = logits_c.argmax(dim=-1)
            btn_probs = torch.sigmoid(logits_btn)
            btn_pred = (btn_probs > 0.5).to(target_info["buttons"].dtype)

            B, L = pred_main_idx.shape
            target_main = target_info["main_idx"].view(B, L)
            target_c = target_info["c_idx"].view(B, L)
            target_btn = target_info["buttons"]

            main_correct = (pred_main_idx == target_main).float().sum().item()
            c_correct = (pred_c_idx == target_c).float().sum().item()
            metrics["main_correct"] += main_correct
            metrics["c_correct"] += c_correct
            metrics["main_total"] += B * L
            metrics["c_total"] += B * L

            em_b, p_b, r_b, f1_b, f1_macro_b = multilabel_prf(target_btn, btn_pred)
            metrics["btn_em_correct"] += em_b * B * L
            metrics["btn_total"] += B * L
            metrics["btn_f1_micro_sum"] += f1_b * B * L
            metrics["btn_f1_macro_sum"] += f1_macro_b * B * L

            # Change accuracy for main stick
            main_change_mask = torch.zeros_like(target_main, dtype=torch.bool)
            main_change_mask[:, 1:] = target_main[:, 1:] != target_main[:, :-1]
            main_correct_mask = pred_main_idx == target_main

            if main_change_mask.any():
                metrics["main_change_correct"] += (
                    main_correct_mask[main_change_mask].float().sum().item()
                )
                metrics["main_change_total"] += main_change_mask.sum().item()

            total_frames += B * L
            batches_processed += 1

    elapsed = time.time() - start_time

    # Compute final metrics
    result: Dict[str, float] = {}

    # Loss metrics
    if batches_processed > 0:
        avg_policy_loss = loss_sums["total"] / batches_processed
        avg_value_loss = loss_sums["value"] / batches_processed
        # Combined loss matches training: policy_loss + value_loss_coef * value_loss
        result["val/loss"] = (
            avg_policy_loss + config.rl.value_loss_coef * avg_value_loss
        )
        result["val/loss_policy"] = avg_policy_loss
        result["val/loss_main"] = loss_sums["main"] / batches_processed
        result["val/loss_c"] = loss_sums["c"] / batches_processed
        result["val/loss_buttons"] = loss_sums["buttons"] / batches_processed
        result["val/loss_shoulder"] = loss_sums["shoulder"] / batches_processed
        result["val/loss_value"] = avg_value_loss

    # Accuracy metrics
    if metrics["main_total"] > 0:
        result["val/acc_main"] = metrics["main_correct"] / metrics["main_total"]
    if metrics["c_total"] > 0:
        result["val/acc_c"] = metrics["c_correct"] / metrics["c_total"]
    if metrics["btn_total"] > 0:
        result["val/btn_em"] = metrics["btn_em_correct"] / metrics["btn_total"]
        result["val/btn_f1_micro"] = metrics["btn_f1_micro_sum"] / metrics["btn_total"]
        result["val/btn_f1_macro"] = metrics["btn_f1_macro_sum"] / metrics["btn_total"]

    # Change accuracy
    if metrics.get("main_change_total", 0) > 0:
        result["val/acc_main_change"] = (
            metrics["main_change_correct"] / metrics["main_change_total"]
        )

    # Metadata
    result["val/batches"] = float(batches_processed)
    result["val/frames"] = float(total_frames)
    result["val/elapsed_s"] = elapsed

    return result


def maybe_run_validation(
    components: "TrainingComponents",
    global_step: int,
) -> None:
    """Run validation and log metrics to wandb if enabled.

    Args:
        components: Training components containing model, logger, etc.
        global_step: Current global training step.
    """
    if not components.logger.enabled:
        return

    print("[validation] Running validation...")
    start_time = time.time()

    # Compute current imbalance_scale based on training progress
    config = components.config
    progress = min(global_step / float(components.total_steps), 1.0)

    initial_scale = config.train.imbalance_scale_initial
    final_scale = config.train.imbalance_scale_final
    final_fraction = config.train.imbalance_scale_final_fraction

    # Match training's imbalance scale computation
    warmup_steps = config.train.schedule_warmup_epochs * (
        components.total_steps // config.train.epochs
    )
    in_warmup = global_step < warmup_steps

    if in_warmup:
        imbalance_scale = initial_scale
    elif progress >= (1.0 - final_fraction):
        imbalance_scale = final_scale
    else:
        ramp_progress = progress / (1.0 - final_fraction)
        imbalance_scale = initial_scale + (final_scale - initial_scale) * ramp_progress

    imbalance_scale = float(max(min(imbalance_scale, final_scale), initial_scale))

    try:
        val_metrics = run_validation(
            model=components.model,
            device=components.device,
            config=components.config,
            max_batches=None,  # Run on full validation set
            imbalance_scale=imbalance_scale,
        )

        # Log to wandb
        components.logger.log_metrics(val_metrics, step=global_step, commit=True)

        # Log to local file (training_metrics.jsonl)
        if components.local_logger.enabled:
            components.local_logger.log_metrics(val_metrics, step=global_step, log_type="val")

        elapsed = time.time() - start_time
        print(
            f"[validation] Completed in {elapsed:.1f}s: "
            f"loss={val_metrics.get('val/loss', 0):.4f} "
            f"main_acc={val_metrics.get('val/acc_main', 0):.3f} "
            f"c_acc={val_metrics.get('val/acc_c', 0):.3f} "
            f"btn_em={val_metrics.get('val/btn_em', 0):.3f}"
        )

    except FileNotFoundError as e:
        print(f"[validation] Skipped - {e}")
    except Exception as e:
        print(f"[validation] Error: {e}")

    # Restore model to training mode
    components.model.train()
