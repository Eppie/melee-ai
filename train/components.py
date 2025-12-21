from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import torch
from tensordict import TensorDict
from torch.amp import GradScaler

from column_map import ColumnMap
from data_loading.instrumentation import DataLoadingMetrics
from model.nano_gpt import GPT
from train.batch_utils import SampleWeightRatios
from train.nvtx_utils import NVTXContext
from train.wandb_utils import WandbLogger
from utils import Profiler


@dataclass
class VarianceTracker:
    """Tracks running variance of scalar values using a sliding window.

    This version keeps values on GPU to avoid synchronization overhead.
    Only transfers to CPU when variance/std is actually computed.
    """

    window_size: int = 100
    values: Deque[torch.Tensor] = field(default_factory=lambda: deque(maxlen=100))
    _pinned_buffer: Optional[torch.Tensor] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Fix maxlen after dataclass initialization."""
        self.values = deque(maxlen=self.window_size)
        # Create pinned memory buffer for efficient CPU transfer
        self._pinned_buffer = torch.empty(self.window_size, dtype=torch.float32, pin_memory=True)

    def add(self, value: torch.Tensor | float) -> None:
        """Add a new value to the tracker.

        Args:
            value: Can be a GPU tensor (preferred) or float. If a tensor,
                   it will be detached and kept on GPU.
        """
        if isinstance(value, torch.Tensor):
            # Detach and keep on GPU (no sync!)
            self.values.append(value.detach().reshape(()))
        else:
            # Legacy support for float values
            device = self.values[0].device if self.values else torch.device('cpu')
            self.values.append(torch.tensor(value, device=device))

    def get_variance(self) -> float:
        """Compute variance of stored values.

        This WILL sync to transfer values from GPU to CPU, but only
        when variance is actually needed (typically during logging).
        """
        if len(self.values) < 2:
            return 0.0

        # Stack all GPU tensors and compute on GPU
        stacked = torch.stack(list(self.values))
        mean = stacked.mean()
        variance = stacked.var(unbiased=True)

        # Single sync for the final result
        return float(variance.item())

    def get_std(self) -> float:
        """Compute standard deviation of stored values."""
        return math.sqrt(self.get_variance())

    def get_cv(self) -> float:
        """Compute coefficient of variation (std / mean)."""
        if len(self.values) < 2:
            return 0.0

        # Stack and compute on GPU
        stacked = torch.stack(list(self.values))
        mean = stacked.mean()
        std = stacked.std(unbiased=True)

        # Single sync for both values
        mean_val = float(mean.item())
        std_val = float(std.item())

        if abs(mean_val) < 1e-9:
            return 0.0
        return std_val / abs(mean_val)


@dataclass
class WeightDriftTracker:
    """Tracks how parameter norms change over time."""

    last_param_norm: Optional[float] = None
    last_update_step: int = 0

    def update(self, param_norm: float, current_step: int) -> Dict[str, float]:
        """Update tracker and return drift metrics.

        Returns:
            Dictionary with:
            - params/total_norm_velocity: Change in norm per step
            - params/total_norm: Current norm (for reference)
        """
        metrics = {"params/total_norm": param_norm}

        if self.last_param_norm is not None and current_step > self.last_update_step:
            steps_elapsed = current_step - self.last_update_step
            velocity = (param_norm - self.last_param_norm) / steps_elapsed
            metrics["params/total_norm_velocity"] = velocity

        self.last_param_norm = param_norm
        self.last_update_step = current_step
        return metrics


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
    dataset: any
    loader: any
    sampler: any
    total_steps: int
    out_dir: Path
    last_step_file: Path
    debug: bool
    gradient_variance_tracker: VarianceTracker = field(
        default_factory=lambda: VarianceTracker(window_size=100)
    )
    loss_variance_tracker: VarianceTracker = field(
        default_factory=lambda: VarianceTracker(window_size=100)
    )
    weight_drift_tracker: WeightDriftTracker = field(default_factory=WeightDriftTracker)
    # Profiling infrastructure (profiles first 1000 steps of current process)
    profiling_enabled: bool = True
    profiling_step_count: int = 0
    profilers: Dict[str, Profiler] = field(default_factory=dict)
    # NVTX instrumentation for profiling tools (Nsight Systems, etc.)
    nvtx_context: NVTXContext = field(default_factory=lambda: NVTXContext(enabled=True))
    # Data loading metrics
    dataloader_metrics: DataLoadingMetrics = field(default_factory=DataLoadingMetrics)


@dataclass
class TrainingState:
    components: TrainingComponents
    global_step: int
    resume_epoch: int
    resume_iter: int
    stop_requested: bool = False


@dataclass
class EpochContext:
    epoch_loss_sum: Optional[
        torch.Tensor
    ] = None  # Accumulated on GPU, transferred only when needed
    iters_processed: int = 0
    applied_skip: int = 0
    frames_since_last_log: float = 0.0
    last_log_time: float = field(default_factory=time.time)
    skip_remaining: int = 0
    progress_iter_base: int = 0

    # Welford's online variance tracking (on GPU)
    loss_mean: Optional[torch.Tensor] = None  # Running mean on GPU
    loss_m2: Optional[torch.Tensor] = None  # Sum of squared differences on GPU
    loss_count: int = 0  # Number of observations

    def add_loss(self, loss: torch.Tensor) -> None:
        """Accumulate loss on GPU without transferring to CPU."""
        loss_detached = loss.detach()
        if self.epoch_loss_sum is None:
            self.epoch_loss_sum = loss_detached.clone()
        else:
            self.epoch_loss_sum.add_(loss_detached)

        # Welford's algorithm for online variance
        self.loss_count += 1
        if self.loss_mean is None:
            self.loss_mean = loss_detached.clone()
            self.loss_m2 = torch.zeros_like(loss_detached)
        else:
            delta = loss_detached - self.loss_mean
            self.loss_mean.add_(delta / self.loss_count)
            delta2 = loss_detached - self.loss_mean
            self.loss_m2.add_(delta * delta2)

    def get_avg_loss(self) -> float:
        """Get average loss (transfers to CPU only when called)."""
        if self.epoch_loss_sum is None or self.iters_processed == 0:
            return 0.0
        return float(self.epoch_loss_sum.item()) / self.iters_processed

    def get_loss_variance(self) -> float:
        """Get loss variance (transfers to CPU only when called)."""
        if self.loss_m2 is None or self.loss_count < 2:
            return 0.0
        variance = self.loss_m2 / (self.loss_count - 1)  # Sample variance
        return float(variance.item())

    def get_loss_std(self) -> float:
        """Get loss standard deviation."""
        import math

        return math.sqrt(self.get_loss_variance())


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
    head_diagnostics: Dict[
        str, float
    ] = None  # Per-head metrics for instability detection
    imitation_weights: torch.Tensor = None  # Value-based sample weights [B, L]


@dataclass
class LoggingBundle:
    log_lines: List[str]
    payload: Dict[str, float]
