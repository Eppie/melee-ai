from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.amp import GradScaler
from tensordict import TensorDict

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
    colmap: ColumnMap
    value_idx: Optional[int]
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
    epoch_loss: float = 0.0
    iters_processed: int = 0
    applied_skip: int = 0
    frames_since_last_log: float = 0.0
    last_log_time: float = field(default_factory=time.time)
    skip_remaining: int = 0


@dataclass
class ForwardPassResult:
    pred: TensorDict
    target_info: Dict[str, torch.Tensor]
    weights: Dict[str, torch.Tensor]
    loss: torch.Tensor
    loss_components: Dict[str, torch.Tensor]
    value_pred: Optional[torch.Tensor]
    value_target: Optional[torch.Tensor]
    batch_inputs: Dict[str, torch.Tensor]
    batch_targets: Dict[str, torch.Tensor]
    label_smoothing: float
    change_scale: float


@dataclass
class LoggingBundle:
    log_lines: List[str]
    payload: Dict[str, float]
