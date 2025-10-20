"""Shared training operations for model training and evaluation.

This module provides reusable components for training loops, including loss
computation, baseline generation, and metrics tracking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from tensordict import TensorDict

from column_map import ColumnMap
from loss import _compute_ce_weights, _compute_pos_weights
from train import build_inputs_for_gptv7


@dataclass
class LossResult:
    """Structured result from computing all training losses.

    Attributes:
        total: Combined total loss for backpropagation
        main: Main stick classification loss
        c: C-stick classification loss
        buttons: Button multi-label loss
        shoulder: Shoulder classification loss
        components: Dict mapping loss names to values for logging
    """
    total: torch.Tensor
    main: torch.Tensor
    c: torch.Tensor
    buttons: torch.Tensor
    shoulder: torch.Tensor
    components: Dict[str, float]


def compute_all_losses(
        pred: TensorDict,
        target_info: TargetInfo,
        *,
        label_smoothing: float = 0.0,
        sample_weights: Optional[torch.Tensor] = None,
) -> LossResult:
    """Compute all training losses in a unified way.

    This function computes losses for all output heads (main stick, c-stick,
    buttons, shoulder) with appropriate weighting schemes. It replaces the
    duplicated loss computation logic in train.py and run_training.py.

    Args:
        pred: Model predictions (TensorDict with keys: main_stick, c_stick, buttons, shoulder)
        target_info: Quantized target information from controller_quantization
        label_smoothing: Label smoothing factor for cross-entropy losses
        sample_weights: Optional per-sample weights [B, L] for weighted loss

    Returns:
        LossResult with all computed losses
    """
    logits_main = pred["main_stick"]
    B, L, _ = logits_main.shape
    device = logits_main.device

    # Flatten sample weights if provided
    weights_flat = sample_weights.view(B * L) if sample_weights is not None else None

    # Main stick loss
    main_targets = target_info["main_idx"].reshape(B * L)
    main_logits = logits_main.reshape(B * L, -1)
    main_weights = _compute_ce_weights(main_targets, int(target_info["main_K"]))
    loss_main = torch.nn.functional.cross_entropy(
        main_logits,
        main_targets,
        reduction='none',
        label_smoothing=label_smoothing,
        weight=main_weights,
    )
    if weights_flat is not None:
        loss_main = (loss_main * weights_flat).mean()
    else:
        loss_main = loss_main.mean()

    # C-stick loss
    logits_c = pred["c_stick"]
    c_targets = target_info["c_idx"].reshape(B * L)
    c_logits = logits_c.reshape(B * L, -1)
    c_weights = _compute_ce_weights(c_targets, int(target_info["c_K"]))
    loss_c = torch.nn.functional.cross_entropy(
        c_logits,
        c_targets,
        reduction="none",
        label_smoothing=label_smoothing,
        weight=c_weights,
    )
    if weights_flat is not None:
        loss_c = (loss_c * weights_flat).mean()
    else:
        loss_c = loss_c.mean()

    # Buttons loss
    logits_btn = pred["buttons"]
    target_btn = target_info["buttons"]
    pos_weight = _compute_pos_weights(target_btn)
    loss_buttons = torch.nn.functional.binary_cross_entropy_with_logits(
        logits_btn,
        target_btn,
        reduction='none',
        pos_weight=pos_weight,
    ).mean(dim=-1)  # Mean over button dimension

    if sample_weights is not None:
        loss_buttons = (loss_buttons * sample_weights).mean()
    else:
        loss_buttons = loss_buttons.mean()

    # Shoulder loss
    loss_shoulder = torch.zeros((), device=device)
    shoulder_logits = pred.get("shoulder")
    shoulder_idx = target_info.get("shoulder_idx")
    shoulder_K = int(target_info.get("shoulder_K", 0))
    if (
            shoulder_logits is not None
            and shoulder_idx is not None
            and shoulder_K > 0
    ):
        shoulder_logits_flat = shoulder_logits.reshape(B * L, -1)
        shoulder_targets_flat = shoulder_idx.reshape(B * L)
        loss_shoulder = torch.nn.functional.cross_entropy(
            shoulder_logits_flat,
            shoulder_targets_flat,
            reduction="none",
            label_smoothing=label_smoothing,
        )
        if weights_flat is not None:
            loss_shoulder = (loss_shoulder * weights_flat).mean()
        else:
            loss_shoulder = loss_shoulder.mean()

    # Total loss
    total_loss = loss_main + loss_c + loss_buttons + loss_shoulder

    # Components dict for logging
    components = {
        "main": float(loss_main.detach().item()),
        "c": float(loss_c.detach().item()),
        "buttons": float(loss_buttons.detach().item()),
        "shoulder": float(loss_shoulder.detach().item()),
    }

    return LossResult(
        total=total_loss,
        main=loss_main,
        c=loss_c,
        buttons=loss_buttons,
        shoulder=loss_shoulder,
        components=components,
    )


def apply_sample_weights(
        loss: torch.Tensor,
        sample_weights: torch.Tensor,
        reduction: str = "mean",
) -> torch.Tensor:
    """Apply per-sample weights to a loss tensor.

    Args:
        loss: Unreduced loss tensor [B, L, ...]
        sample_weights: Weight tensor [B, L]
        reduction: Reduction method ('mean', 'sum', or 'none')

    Returns:
        Weighted and reduced loss
    """
    weighted = loss * sample_weights

    if reduction == "mean":
        return weighted.mean()
    elif reduction == "sum":
        return weighted.sum()
    elif reduction == "none":
        return weighted
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


@dataclass
class BaselineInputs:
    """Inputs needed to compute all baselines for a batch.

    Attributes:
        pred_idx: Predicted class indices [B, L]
        true_idx: True class indices [B, L]
        K: Number of classes
        running_counts: Running label counts from previous batches (for majority)
        device: Device to run on
    """
    pred_idx: torch.Tensor
    true_idx: torch.Tensor
    K: int
    running_counts: torch.Tensor
    device: torch.device


@dataclass
class BaselineOutputs:
    """All baseline predictions and metadata.

    Attributes:
        majority_idx: Majority class baseline (scalar, same for all samples)
        repeat_idx: Repeat-last baseline [B*L]
        repeat_mask: Mask indicating which positions are valid for repeat [B*L]
    """
    majority_idx: int
    repeat_idx: torch.Tensor
    repeat_mask: torch.Tensor


def compute_classification_baselines(inputs: BaselineInputs) -> BaselineOutputs:
    """Compute all baseline predictions for a classification task.

    Generates majority-class and repeat-last baselines for comparison.
    The majority baseline uses running statistics from previous batches.
    The repeat-last baseline predicts each frame's label from the previous frame
    (masking out the first frame in each sequence).

    Args:
        inputs: Structured inputs containing predictions, targets, and metadata

    Returns:
        BaselineOutputs with all baseline predictions
    """
    B, L = inputs.true_idx.shape

    # Majority baseline - find the most common class from running counts
    total = inputs.running_counts.sum()
    if total.detach().cpu().item() == 0:
        majority_idx = 0
    else:
        majority_idx = int(torch.argmax(inputs.running_counts).detach().cpu().item())

    # Repeat-last baseline
    repeat_idx = torch.zeros_like(inputs.true_idx)
    repeat_mask = torch.ones((B, L), dtype=torch.bool, device=inputs.device)
    repeat_mask[:, 0] = False  # First frame can't repeat anything

    if L > 1:
        repeat_idx[:, 1:] = inputs.true_idx[:, :-1]

    return BaselineOutputs(
        majority_idx=majority_idx,
        repeat_idx=repeat_idx.reshape(-1),
        repeat_mask=repeat_mask.reshape(-1),
    )


def compute_button_baselines(
        true: torch.Tensor,
        running_pos_counts: torch.Tensor,
        running_total: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute baselines for multi-label button prediction.

    Args:
        true: True button states [B, L, K_buttons]
        running_pos_counts: Running positive counts per button [K_buttons]
        running_total: Total samples seen so far

    Returns:
        Tuple of (majority_pred, repeat_pred) all [B, L, K_buttons]
    """
    B, L, K = true.shape

    # Majority baseline - predict 1 if positive rate >= 0.5
    totals_seen = max(1.0, running_total)
    maj_vec = (running_pos_counts * 2.0 >= totals_seen).to(true.dtype)
    majority_pred = maj_vec.view(1, 1, K).expand(B, L, K)

    # Repeat-last baseline
    repeat_pred = torch.zeros_like(true)
    if L > 1:
        repeat_pred[:, 1:, :] = true[:, :-1, :]

    return majority_pred, repeat_pred


# ============================================================================
# Metrics Update Orchestration
# ============================================================================

def update_running_metrics_from_batch(
        metrics: "RunningMetrics",
        pred: TensorDict,
        target_info: TargetInfo,
        colmap: ColumnMap,
        device: torch.device,
) -> None:
    """Update RunningMetrics instance with predictions from a batch.

    This function orchestrates all the baseline generation and metrics updates
    for a single batch, handling main stick, c-stick, buttons, and shoulder.
    It eliminates the duplicated metrics update logic in both training scripts.

    Args:
        metrics: RunningMetrics instance to update (modified in-place)
        pred: Model predictions
        target_info: Quantized targets
        colmap: Column mapping for feature/target indices
        device: Device for tensor operations
    """
    B, L, _ = pred["main_stick"].shape

    # Main stick
    pred_main_idx = pred["main_stick"].argmax(dim=-1)  # [B, L]
    true_main_idx = target_info["main_idx"].reshape(B, L)

    main_baselines = compute_classification_baselines(
        BaselineInputs(
            pred_idx=pred_main_idx,
            true_idx=true_main_idx,
            K=int(target_info["main_K"]),
            running_counts=metrics.main_label_counts,
            device=device,
        )
    )

    metrics.update_main(
        pred_main_idx.reshape(-1),
        true_main_idx.reshape(-1),
        main_baselines.majority_idx,
        main_baselines.repeat_idx,
        main_baselines.repeat_mask,
    )

    # C-stick
    pred_c_idx = pred["c_stick"].argmax(dim=-1)  # [B, L]
    true_c_idx = target_info["c_idx"].reshape(B, L)

    c_baselines = compute_classification_baselines(
        BaselineInputs(
            pred_idx=pred_c_idx,
            true_idx=true_c_idx,
            K=int(target_info["c_K"]),
            running_counts=metrics.c_label_counts,
            device=device,
        )
    )

    metrics.update_c(
        pred_c_idx.reshape(-1),
        true_c_idx.reshape(-1),
        c_baselines.majority_idx,
        c_baselines.repeat_idx,
        c_baselines.repeat_mask,
    )

    # Buttons
    btn_logits = pred["buttons"]
    btn_true = target_info["buttons"]
    btn_probs = pred.get("buttons_probs")
    if btn_probs is None:
        btn_probs = torch.sigmoid(btn_logits)

    metrics.update_buttons(btn_logits, btn_true, btn_probs)

    # Shoulder
    if "shoulder" in pred and target_info.get("shoulder_idx") is not None:
        shoulder_major = metrics._majority_label(metrics.shoulder_label_counts) if metrics.K_shoulder else None
        metrics.update_shoulder(
            pred["shoulder"],
            target_info["shoulder_idx"],
            shoulder_major,
        )


# ============================================================================
# Training Step Coordination
# ============================================================================

@dataclass
class TrainingStepInputs:
    """All inputs needed for a single training step."""
    batch: Dict[str, torch.Tensor]
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    colmap: ColumnMap
    device: torch.device
    label_smoothing: float
    grad_clip: Optional[float]
    sample_weights: Optional[torch.Tensor] = None


@dataclass
class TrainingStepOutputs:
    """Results from a single training step.

    Attributes:
        loss_result: Structured losses
        pred: Model predictions
        target_info: Quantized targets
        tokens_processed: Number of tokens in this batch (B * L)
    """
    loss_result: LossResult
    pred: TensorDict
    target_info: TargetInfo
    tokens_processed: int


def perform_training_step(inputs: TrainingStepInputs) -> TrainingStepOutputs:
    """Execute a complete training step: forward, loss, backward, optimizer step.

    This function encapsulates the core training loop iteration that is duplicated
    between train.py and run_training.py. It does NOT handle:
    - Learning rate scheduling (different in each script)
    - Metrics tracking (caller's responsibility)
    - Logging (different in each script)
    - Early stopping (only in run_training.py)

    Args:
        inputs: All required inputs for the training step

    Returns:
        Outputs containing losses, predictions, and metadata
    """
    from controller_quantization import quantize_targets

    # Move batch to device
    X = inputs.batch["X"].to(inputs.device, non_blocking=True)
    Y = inputs.batch["Y"].to(inputs.device, non_blocking=True)

    # Build model inputs and quantize targets
    inputs_td = build_inputs_for_gptv7(X, inputs.colmap)
    target_info = quantize_targets(Y, inputs.colmap, input_domain="unit11")

    # Forward pass
    pred = inputs.model(inputs_td)
    B, L, _ = pred["main_stick"].shape
    tokens_processed = B * L

    # Compute losses
    loss_result = compute_all_losses(
        pred,
        target_info,
        label_smoothing=inputs.label_smoothing,
        sample_weights=inputs.sample_weights,
    )

    # Backward pass
    inputs.optimizer.zero_grad(set_to_none=True)
    loss_result.total.backward()

    # Gradient clipping
    if inputs.grad_clip is not None and inputs.grad_clip > 0:
        from torch.nn.utils import clip_grad_norm_
        clip_grad_norm_(inputs.model.parameters(), inputs.grad_clip)

    # Optimizer step
    inputs.optimizer.step()

    return TrainingStepOutputs(
        loss_result=loss_result,
        pred=pred,
        target_info=target_info,
        tokens_processed=tokens_processed,
    )


# ============================================================================
# Utilities
# ============================================================================

def compute_change_weights(
        Y: torch.Tensor,
        change_multiplier: float = 10.0,
) -> torch.Tensor:
    """Compute sample weights that boost frames where controller state changes.

    Args:
        Y: Target tensor [B, L, Y_dim]
        change_multiplier: Weight multiplier for change frames

    Returns:
        Sample weights [B, L]
    """
    B, L, _ = Y.shape
    device = Y.device

    # Create change mask by comparing each frame to the previous
    change_mask = torch.zeros((B, L), device=device, dtype=torch.bool)
    # Compare from the second timestep onwards
    state_changed = torch.any(Y[:, 1:] != Y[:, :-1], dim=-1)
    change_mask[:, 1:] = state_changed

    # Create weight tensor: 1.0 for hold frames, change_multiplier for change frames
    sample_weights = torch.ones((B, L), device=device)
    sample_weights[change_mask] = change_multiplier

    return sample_weights
