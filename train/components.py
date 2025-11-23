from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tensordict import TensorDict
from torch.amp import GradScaler

from column_map import ColumnMap
from model.nano_gpt import GPT
from train.batch_utils import SampleWeightRatios
from train.wandb_utils import WandbLogger


@dataclass
class AMPContext:
    enabled: bool
    device_type: str
    dtype: torch.dtype


@dataclass
class TrainingComponents:
    config: Any
    model: GPT
    optimizer: torch.optim.Optimizer
    scaler: GradScaler
    logger: WandbLogger
    device: torch.device
    amp: AMPContext
    ratios: SampleWeightRatios
    column_map: ColumnMap
    value_idx: int
    loader: any
    sampler: any
    total_steps: int
    out_dir: Path
    last_step_file: Path
    debug: bool


@dataclass
class TrainingState:
    components: TrainingComponents
    global_step: int
    resume_epoch: int
    resume_iter: int
    stop_requested: bool = False


@dataclass
class EpochContext:
    epoch_loss_sum: Optional[torch.Tensor] = (
        None  # Accumulated on GPU, transferred only when needed
    )
    iters_processed: int = 0
    applied_skip: int = 0
    frames_since_last_log: float = 0.0
    last_log_time: float = field(default_factory=time.time)
    skip_remaining: int = 0
    progress_iter_base: int = 0

    def add_loss(self, loss: torch.Tensor) -> None:
        """Accumulate loss on GPU without transferring to CPU."""
        loss_detached = loss.detach()
        if self.epoch_loss_sum is None:
            self.epoch_loss_sum = loss_detached.clone()
        else:
            self.epoch_loss_sum.add_(loss_detached)

    def get_avg_loss(self) -> float:
        """Get average loss (transfers to CPU only when called)."""
        if self.epoch_loss_sum is None or self.iters_processed == 0:
            return 0.0
        return float(self.epoch_loss_sum.item()) / self.iters_processed


@dataclass
class ForwardPassResult:
    pred: TensorDict
    target_info: Dict[str, torch.Tensor]
    weights: Dict[str, torch.Tensor]
    loss: torch.Tensor
    loss_components: Dict[str, torch.Tensor]
    value_pred: torch.Tensor
    value_target: torch.Tensor
    batch_inputs: Dict[str, torch.Tensor]
    batch_targets: Dict[str, torch.Tensor]
    label_smoothing: float
    change_scale: float


@dataclass
class LoggingBundle:
    log_lines: List[str]
    payload: Dict[str, float]
