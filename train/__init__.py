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

from train.batch_utils import build_model_inputs
from train.checkpoint import (
    _sorted_checkpoint_paths as sorted_checkpoint_paths,
    _latest_checkpoint as find_latest_checkpoint,
    _prune_checkpoints as prune_checkpoints,
    _load_latest_checkpoint as load_latest_checkpoint,
    load_config_from_checkpoint,
    load_config_from_latest_checkpoint,
    save_checkpoint,
    maybe_checkpoint_batch,
    maybe_checkpoint_epoch,
)
from train.display import (
    format_confusion_matrix,
)
from train.gradients import (
    _move_optimizer_state_to_device as move_optimizer_state_to_device,
    collect_gradient_diagnostics,
)
from train.lr_schedule import (
    cosine_lr_schedule,
)
from train.metrics import (
    MetricsAccumulator,
    compute_confusion_matrix,
    multilabel_prf,
)
from train.value_head import (
    RewardFeatureIdx,
    build_reward_feature_index,
    compute_frame_rewards,
    compute_value_targets,
)
from train.setup import (
    build_optimizer,
    configure_amp,
    initialize_training_components,
    parse_cli_overrides,
    print_config,
)
from train.logging import (
    prepare_logging_bundle,
    emit_logging,
)
from train.step import (
    perform_forward_pass,
    perform_backward_pass,
)
from train.wandb_utils import (
    WandbConfig,
    WandbLogger,
    init_wandb,
    finish_wandb,
)
from train.validation import (
    run_validation,
    maybe_run_validation,
)

__all__ = [
    # Checkpoint
    "sorted_checkpoint_paths",
    "find_latest_checkpoint",
    "prune_checkpoints",
    "load_latest_checkpoint",
    "load_config_from_checkpoint",
    "load_config_from_latest_checkpoint",
    "save_checkpoint",
    "maybe_checkpoint_batch",
    "maybe_checkpoint_epoch",
    # Metrics
    "MetricsAccumulator",
    "compute_confusion_matrix",
    "multilabel_prf",
    # LR schedule
    "cosine_lr_schedule",
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
    # Setup helpers
    "parse_cli_overrides",
    "print_config",
    "configure_amp",
    "build_optimizer",
    "initialize_training_components",
    # Logging
    "prepare_logging_bundle",
    "emit_logging",
    # Step helpers
    "perform_forward_pass",
    "perform_backward_pass",
    # Wandb
    "WandbConfig",
    "WandbLogger",
    "init_wandb",
    "finish_wandb",
    # Validation
    "run_validation",
    "maybe_run_validation",
]
