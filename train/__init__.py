"""Training utilities and operations.

This package provides modular, loosely-coupled utilities for training:
- checkpoint: Save/load/manage checkpoints
- metrics: Compute and accumulate training metrics
- lr_schedule: Learning rate schedules
- gradients: Gradient diagnostics and clipping
- batch_utils: Batch processing and preparation
- value_head: RL/value head utilities
- display: Logging and formatting
- wandb_utils: Optional wandb integration
"""

from __future__ import annotations

# Batch processing
from train.batch_utils import (
    build_model_inputs,
)

# Checkpoint management
from train.checkpoint import (
    _sorted_checkpoint_paths as sorted_checkpoint_paths,
    _latest_checkpoint as find_latest_checkpoint,
    _prune_checkpoints as prune_checkpoints,
    _load_latest_checkpoint as load_latest_checkpoint,
    save_checkpoint,
)

# Display and formatting
from train.display import (
    format_confusion_matrix,
    format_metrics_dict,
    format_loss_summary,
    format_training_progress,
    print_batch_preview,
)

# Gradient utilities
from train.gradients import (
    _move_optimizer_state_to_device as move_optimizer_state_to_device,
    collect_gradient_diagnostics,
)

# Learning rate schedules
from train.lr_schedule import (
    cosine_lr_schedule,
    linear_warmup,
)

# Metrics
from train.metrics import (
    MetricsAccumulator,
    compute_confusion_matrix,
    compute_change_hold_accuracy,
    multilabel_prf,
)

# Value head (RL)
from train.value_head import (
    RewardFeatureIdx,
    build_reward_feature_index,
    compute_frame_rewards,
    compute_value_targets,
)

# Wandb integration (optional)
from train.wandb_utils import (
    WandbConfig,
    WandbLogger,
    init_wandb,
    finish_wandb,
)

__all__ = [
    # Checkpoint
    "sorted_checkpoint_paths",
    "find_latest_checkpoint",
    "prune_checkpoints",
    "load_latest_checkpoint",
    "save_checkpoint",
    # Metrics
    "MetricsAccumulator",
    "compute_confusion_matrix",
    "compute_change_hold_accuracy",
    "multilabel_prf",
    # LR schedules
    "cosine_lr_schedule",
    "linear_warmup",
    # Gradients
    "move_optimizer_state_to_device",
    "collect_gradient_diagnostics",
    # Batch utils
    "build_model_inputs",
    # Value head
    "RewardFeatureIdx",
    "build_reward_feature_index",
    "compute_frame_rewards",
    "compute_value_targets",
    # Display
    "format_confusion_matrix",
    "format_metrics_dict",
    "format_loss_summary",
    "format_training_progress",
    "print_batch_preview",
    # Wandb
    "WandbConfig",
    "WandbLogger",
    "init_wandb",
    "finish_wandb",
]
