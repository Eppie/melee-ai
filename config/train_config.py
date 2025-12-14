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

    batch_size: int = Field(
        default=256,
        ge=1,
        description=(
            "Training batch size. Number of sequences per gradient update. "
            "Effect: Larger batches (512-1024) = more stable gradients, better GPU utilization but more memory; "
            "smaller batches (64-128) = less memory, noisier gradients. Reasonable range: [64, 512]. "
            "Interacts with: lr (larger batches often benefit from higher lr), num_workers, GPU memory."
        ),
    )
    epochs: int = Field(
        default=32,
        ge=1,
        description=(
            "Number of training epochs. One epoch = one pass through the entire dataset. "
            "Effect: More epochs = more training, better fit but risk of overfitting; "
            "fewer epochs = less training, may underfit. Reasonable range: [10, 100]. "
            "Interacts with: lr schedule (cosine decay spans epochs), stride (affects samples per epoch)."
        ),
    )
    lr: float = Field(
        default=2e-4,
        gt=0,
        description=(
            "Learning rate for AdamW optimizer. Reduced from 3e-4 to 2e-4 for more stable imitation learning. "
            "Effect: Lower lr (1e-4 to 2e-4) = more stable but slower learning; higher lr (3e-4 to 5e-4) = faster but less stable. "
            "Reasonable range: [1e-4, 5e-4]. Interacts with: warmup_steps, grad_clip, value_loss_coef."
        ),
    )
    weight_decay: float = Field(
        default=0.002,
        ge=0,
        description=(
            "L2 regularization strength for AdamW optimizer. Prevents overfitting by "
            "penalizing large weights. Effect: Higher values (0.01-0.1) increase regularization "
            "and may reduce overfitting but can hurt capacity; lower values (0.0001-0.005) reduce "
            "regularization. Reasonable range: [0.0001, 0.1]. Interacts with: lr (higher lr often "
            "needs higher weight_decay), model size (larger models may need more regularization)."
        ),
    )
    betas: Tuple[float, float] = Field(
        default=(0.9, 0.95),
        description=(
            "AdamW momentum coefficients (beta1, beta2) for gradient and squared gradient moving averages. "
            "beta1 controls first moment (mean), beta2 controls second moment (variance). "
            "Effect: Higher beta1 (0.9-0.95) = smoother gradient updates, more momentum; "
            "higher beta2 (0.95-0.999) = more stable adaptive learning rates. "
            "Reasonable values: beta1=[0.85, 0.95], beta2=[0.95, 0.999]. "
            "Default (0.9, 0.98) is lower than typical (0.9, 0.999) for faster adaptation. "
            "Interacts with: lr (higher betas may allow higher lr), batch_size (larger batches often use higher betas)."
        ),
    )
    warmup_steps: int = Field(
        default=7500,
        ge=0,
        description="Number of optimizer steps to linearly warm up learning rate from 0 to lr. Helps stabilize early training.",
    )
    num_workers: int = Field(
        default=16,
        ge=0,
        description="Number of dataloader worker processes. More workers = faster data loading but more memory.",
    )
    window_bucket_size: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Group episodes into contiguous buckets of this size and shuffle buckets per epoch. "
            "Within each bucket, window indices are shuffled. Set to None to disable bucketing."
        ),
    )
    worker_episode_cache_size: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of episodes each worker caches in RAM (per-process LRU). "
            "Set to 0 to disable the cache."
        ),
    )
    in_memory_shared: bool = Field(
        default=False,
        description=(
            "Preload all episodes into shared CPU memory so workers reuse a single copy. "
            "Improves dataloader throughput at the cost of RAM roughly equal to the dataset. "
            "Disable if the dataset does not fit in host memory."
        ),
    )
    in_memory_shared_chunk_size: int | str = Field(
        default="auto",
        description=(
            "Number of episodes to preload into shared memory at a time. "
            "Use 'auto' for dynamic sizing based on available RAM, or an integer for fixed size. "
            "Episodes are streamed in chunks to limit RAM usage (e.g., 1000 at a time)."
        ),
    )
    num_overlapping_chunks: int = Field(
        default=2,
        ge=1,
        description=(
            "Number of chunks to keep loaded simultaneously for multi-chunk overlap. "
            "Higher values improve shuffling quality but use more RAM. "
            "With num_overlapping=2, chunks [0,1], [1,2], [2,3] are loaded sequentially, "
            "providing a 2x larger shuffle window compared to single-chunk mode."
        ),
    )
    chunk_size_ram_budget_mb: int | None = Field(
        default=None,
        description=(
            "Override available RAM detection for chunk sizing (in MB). "
            "Useful for remote training where psutil may report incorrect values. "
            "If None, uses psutil.virtual_memory().available."
        ),
    )
    background_chunk_preload: bool = Field(
        default=True,
        description=(
            "Load next chunk in background thread during training to eliminate chunk-switch stalls. "
            "Improves GPU utilization but adds complexity."
        ),
    )
    prefetch_factor: int = Field(
        default=4,
        ge=1,
        description="Number of batches each dataloader worker prefetches. Higher = more memory but smoother training.",
    )
    max_loader_prefetch_mb: int = Field(
        default=2048,
        ge=1,
        description="Maximum memory (MB) for dataloader prefetching. Prevents OOM from excessive prefetching.",
    )
    pin_memory: bool = Field(
        default_factory=lambda: _should_pin_memory(),
        description="Pin tensors in CPU memory for faster GPU transfer. Auto-enabled for CUDA, disabled for MPS/CPU.",
    )
    persistent_workers: bool = Field(
        default=True,
        description="Keep dataloader workers alive between epochs. Faster but uses more memory.",
    )
    stride: int = Field(
        default=8,
        ge=1,
        description=(
            "Stride between consecutive training windows. Lower stride = more overlapping windows = "
            "more data augmentation but slower epochs. stride=1 uses every possible window, stride=block_size "
            "uses non-overlapping windows. Reasonable range: [1, block_size]. Interacts with: block_size, epochs."
        ),
    )
    worker_start_method: Optional[Literal["fork", "spawn", "forkserver"]] = Field(
        default=None,
        description=(
            "Method for starting dataloader worker processes. "
            "Options: 'fork' (copy parent process, fast but can cause issues), "
            "'spawn' (fresh Python interpreter, slower startup but safer), "
            "'forkserver' (compromise between fork and spawn). "
            "None = use platform default (fork on Unix, spawn on Windows). "
            "Recommended: None (auto-detect) or 'spawn' if encountering multiprocessing issues."
        ),
    )

    # Losses
    grad_clip: float = Field(
        default=1.0,
        gt=0,
        description=(
            "Maximum gradient norm for GLOBAL gradient clipping. Prevents exploding gradients. "
            "IMPORTANT: Value head has separate clipping at max_norm=1.0 applied BEFORE this global clip. "
            "Reduced from 5.0 to 1.0 for tighter control after value head instability issues. "
            "Effect: Lower values (0.5-2.0) clip more aggressively, more stable but slower learning; "
            "higher values (5.0-10.0) allow larger updates but risk instability. Reasonable range: [0.5, 5.0]. "
            "Interacts with: lr (higher lr may need lower grad_clip), separate value head clipping (max_norm=1.0)."
        ),
    )
    label_smoothing: float = Field(
        default=0.02,
        ge=0,
        le=1,
        description=(
            "Label smoothing for cross-entropy loss. Replaces hard targets (1.0) with soft targets (1-ε). "
            "Effect: Reduces overconfidence and improves generalization. Higher values (0.05-0.2) = more smoothing. "
            "Reasonable range: [0.0, 0.1]. Too high can hurt performance."
        ),
    )
    schedule_warmup_epochs: int = Field(
        default=1,
        ge=0,
        description="Number of epochs for learning rate warmup phase. Helps stabilize early training.",
    )
    schedule_cooldown_epochs: int = Field(
        default=1,
        ge=0,
        description="Number of epochs at end of training to keep lr at minimum. Allows model to settle.",
    )

    # Imbalance scale scheduling
    imbalance_scale_initial: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description=(
            "Initial imbalance scale value (used during warmup epoch). Controls how aggressively "
            "class balancing and change-based loss weighting are applied at the start of training. "
            "Two-stage training approach: Start with full weighting (1.0) to learn rare actions, "
            "then decay to learn proper timing and frequencies. "
            "Effect: Higher values (0.8-1.0) apply full weighting from start to learn action space; "
            "lower values (0.3-0.5) start gentler. Reasonable range: [0.5, 1.0]. "
            "Interacts with: imbalance_scale_final (determines ramp range), loss config weights (scales all of them). "
            "See loss_weighting_explained.md and loss_weighting_analysis.md for details."
        ),
    )
    imbalance_scale_final: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description=(
            "Final imbalance scale value (used during final portion of training). Controls strength "
            "of class balancing and change-based loss weighting in late training. "
            "Two-stage training approach: After learning rare actions with full weighting, decay to "
            "lighter weighting so model learns true action frequencies and timing. "
            "Effect: Lower values (0.3-0.5) reduce weighting to learn timing; "
            "higher values (0.7-1.0) maintain stronger weighting. Reasonable range: [0.3, 0.7]. "
            "Interacts with: loss config weights (multiplies them), "
            "imbalance_scale_final_fraction (how long to maintain final value)."
        ),
    )
    imbalance_scale_initial_fraction: float = Field(
        default=0.2,
        ge=0,
        le=1,
        description=(
            "Fraction of training to keep at initial imbalance scale before starting decay. "
            "Effect: Higher values (0.3-0.4) maintain full weighting longer to learn action space; "
            "lower values (0.1-0.2) start decay earlier. Reasonable range: [0.1, 0.3]. "
            "Interacts with: epochs (determines absolute duration), imbalance_scale_initial."
        ),
    )
    imbalance_scale_final_fraction: float = Field(
        default=0.2,
        ge=0,
        le=1,
        description=(
            "Fraction of training to keep at final imbalance scale at the end. Determines how long "
            "training stays at reduced weighting to learn timing. Effect: Higher values (0.3-0.5) maintain "
            "reduced weighting longer; lower values (0.1-0.2) reach reduced weighting later. "
            "Reasonable range: [0.1, 0.5]. Interacts with: epochs (determines absolute duration), "
            "imbalance_scale_final (the target scale value)."
        ),
    )
    use_amp: bool = Field(
        default_factory=lambda: _should_use_amp(),
        description=(
            "Enable Automatic Mixed Precision (AMP) training. Uses lower precision (float16/bfloat16) for speed. "
            "Effect: True = ~2x faster training, lower memory, minimal accuracy loss on modern GPUs; "
            "False = full float32 precision, slower but more stable. "
            "Auto-enabled for CUDA and MPS. Recommended: Keep default (auto-detect). "
            "Interacts with: amp_dtype (determines precision format)."
        ),
    )
    amp_dtype: str = Field(
        default_factory=lambda: _get_optimal_amp_dtype(),
        description=(
            "AMP data type. Controls precision format for mixed precision training. "
            "Options: 'bfloat16' (better range, no gradient scaling needed, CUDA default), "
            "'float16' (more hardware support, needs gradient scaling, MPS default), "
            "'float32' (full precision, no AMP). "
            "Auto-selected based on hardware. Recommended: Keep default (auto-detect). "
            "Interacts with: use_amp (only used if AMP enabled)."
        ),
    )

    # Checkpointing
    out_dir: str = Field(
        default="../checkpoints",
        description="Directory for saving model checkpoints during training.",
    )
    allow_partial_checkpoint_load: bool = Field(
        default=False,
        description=(
            "Allow loading checkpoints with mismatched model architecture. "
            "Effect: True = load matching parameters, skip mismatched ones (useful for architecture changes); "
            "False = strict loading, fail if any mismatch (safer). "
            "Recommended: False for normal training, True for transfer learning or architecture experiments."
        ),
    )

    # Wandb
    wandb_project: str = Field(
        default="melee-ai",
        description="Wandb project name.",
    )
    run_name: Optional[str] = Field(
        default=None,
        description="Wandb run name.",
    )
    wandb_mode: str = Field(
        default="online",
        description="Wandb mode. Can be 'online', 'offline', or 'disabled'.",
    )

    # Performance optimizations
    torch_compile: bool = Field(
        default_factory=lambda: _should_enable_torch_compile(),
        description=(
            "Enable torch.compile for model optimization. Auto-disabled on non-CUDA backends; "
            "only CUDA devices attempt compilation. May have initial overhead."
        ),
    )
    torch_compile_mode: Optional[str] = Field(
        default="max-autotune",
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
    - bfloat16: CUDA GPUs (no gradient scaling needed)
    - float16: Apple Silicon MPS (requires gradient scaling)
    - float32: CPU or unsupported hardware
    """
    if hasattr(torch.backends, "cuda") and torch.cuda.is_available():
        # Use bfloat16 for CUDA - no gradient scaling needed
        return "bfloat16"
    # Apple Silicon uses float16 with gradient scaling
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


def _should_enable_torch_compile() -> bool:
    """
    Auto-detect if torch.compile should be enabled.
    Only enable on CUDA devices; disable on MPS/CPU.
    """
    return bool(hasattr(torch.backends, "cuda") and torch.cuda.is_available())
