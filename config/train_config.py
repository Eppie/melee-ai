from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch
from pydantic import BaseModel, Field
from pydantic_settings import SettingsConfigDict


class TrainConfig(BaseModel):
    """Pydantic version of TrainConfig with validation."""

    model_config = SettingsConfigDict(
        validate_assignment=True,
        extra="forbid",
    )

    batch_size: int = Field(default=256, ge=1)
    epochs: int = Field(default=32, ge=1)
    lr: float = Field(default=3e-4, gt=0)
    # TODO: Document the effect of this setting
    weight_decay: float = Field(default=0.002, ge=0)
    # TODO: Document the effect of this setting
    betas: Tuple[float, float] = Field(default=(0.9, 0.95))
    warmup_steps: int = Field(default=7500, ge=0)
    num_workers: int = Field(default=16, ge=0)
    prefetch_factor: int = Field(default=4, ge=1)
    max_loader_prefetch_mb: int = Field(default=2048, ge=1)
    pin_memory: bool = Field(default_factory=lambda: _should_pin_memory())
    persistent_workers: bool = True
    stride: int = Field(default=16, ge=1)
    worker_start_method: Optional[Literal["fork", "spawn", "forkserver"]] = None

    # Losses
    grad_clip: float = Field(default=5.0, gt=0)
    label_smoothing: float = Field(default=0.02, ge=0, le=1)
    schedule_warmup_epochs: int = Field(default=1, ge=0)
    schedule_cooldown_epochs: int = Field(default=1, ge=0)
    use_amp: bool = Field(default_factory=lambda: _should_use_amp())
    amp_dtype: str = Field(default_factory=lambda: _get_optimal_amp_dtype())

    # Checkpointing
    out_dir: str = "../checkpoints"
    allow_partial_checkpoint_load: bool = False

    # Performance optimizations
    torch_compile: bool = Field(
        default=True,
        description="Enable torch.compile for model optimization. May have initial overhead.",
    )
    torch_compile_mode: Optional[str] = Field(
        default="default",
        description="torch.compile mode: 'default', 'reduce-overhead', or 'max-autotune'",
    )
    cudnn_benchmark: bool = Field(
        default_factory=lambda: _should_enable_cudnn_benchmark(),
        description="Enable cudnn.benchmark for faster convolutions with consistent input sizes.",
    )


def _should_pin_memory() -> bool:
    """
    Auto-detect if memory pinning should be enabled.
    Pin memory is beneficial for CUDA but not for MPS or CPU.
    """
    if hasattr(torch.backends, "cuda") and torch.cuda.is_available():
        return True
    # MPS (Apple Silicon) doesn't benefit from pinning
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return False
    return False


def _should_use_amp() -> bool:
    """
    Auto-detect if Automatic Mixed Precision should be enabled.
    AMP is beneficial for modern GPUs with tensor cores.
    """
    if hasattr(torch.backends, "cuda") and torch.cuda.is_available():
        return True
    # Apple Silicon supports AMP well
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return True
    return False


def _get_optimal_amp_dtype() -> str:
    """
    Auto-detect optimal AMP dtype based on hardware.
    - float16: GPUs and Apple Silicon with AMP support
    - float32: CPU or unsupported hardware
    """
    if hasattr(torch.backends, "cuda") and torch.cuda.is_available():
        return "float16"
    # Apple Silicon supports float16 well
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "float16"
    return "float32"


def _should_enable_cudnn_benchmark() -> bool:
    """
    Auto-detect if cudnn.benchmark should be enabled.
    Beneficial for CUDA with consistent input sizes (like fixed batch/seq_len).
    """
    if hasattr(torch.backends, "cuda") and torch.cuda.is_available():
        return True
    return False
