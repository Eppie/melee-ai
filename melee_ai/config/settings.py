"""
Settings configuration for the Melee AI pipeline.

This module defines the unified configuration system with:
- Hierarchical configuration structure
- Environment variable overrides
- Validation hooks
- Computed properties
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

from .constants import (
    C_STICK_XY_CLUSTER_CENTERS_V0_1,
    FOX_STICK_64,
    MAX_FRAMES,
)


@dataclass
class DataConfig:
    """Configuration for data sources and processing."""

    # Dataset and replay paths
    data_root: str = "dataset_FOX_vs_FOX"
    replay_glob: str = "*.slp"
    replay_dir: Optional[str] = None  # If provided, overrides data_root for replays

    # Data loading
    seq_len: int = 256
    shard_size: int = 100

    # Column/feature selection
    feature_keep: Optional[List[str]] = None
    target_keep: Optional[List[str]] = None


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing."""

    # Quantization palettes (lengths are computed from constants)
    fox_stick_palette_size: int = field(default_factory=lambda: len(FOX_STICK_64))
    c_stick_palette_size: int = field(default_factory=lambda: len(C_STICK_XY_CLUSTER_CENTERS_V0_1))

    # Frame processing
    max_frames: int = MAX_FRAMES

    # Stick processing
    stick_deadzone: float = 0.08  # Radial deadzone threshold


@dataclass
class ModelConfig:
    """Configuration for model architecture."""

    # GPT model parameters
    block_size: int = 256
    n_embd: int = 512
    n_layer: int = 8
    n_head: int = 8
    dropout: float = 0.1
    bias: bool = True

    # Embedding dimensions
    stage_embedding_dim: int = 4
    character_embedding_dim: int = 12
    action_embedding_dim: int = 32

    # Output head configurations
    target_shapes_by_head: dict = field(default_factory=lambda: {
        "main_stick": (len(FOX_STICK_64),),
        "c_stick": (len(C_STICK_XY_CLUSTER_CENTERS_V0_1),),
        "buttons": (5,),
        "shoulder": (3,),
    })


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters."""

    # Core training parameters
    batch_size: int = 64
    epochs: int = 5
    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 0
    max_steps: Optional[int] = None

    # Loss parameters
    grad_clip: float = 1.0
    label_smoothing: float = 0.0

    # Shoulder quantization
    shoulder_centers: Optional[Sequence[float]] = None  # e.g., [0.0, 0.5, 1.0]

    # Data loading mode
    mode: str = "random_windows"  # "episode_linear" or "random_windows"

    # Episode-linear sampler parameters
    episodes_per_epoch: Optional[int] = None
    with_replacement_episodes: bool = False

    # Random windows sampler parameters
    replacement: bool = False
    num_samples: Optional[int] = None

    # Epoch sizing
    windows_per_epoch: Optional[int] = None
    steps_per_epoch: Optional[int] = None

    # Data loading workers
    num_workers: int = 8
    prefetch_factor: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True

    def __post_init__(self):
        """Validate training configuration."""
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.lr <= 0:
            raise ValueError(f"learning rate must be positive, got {self.lr}")
        if self.mode not in ["episode_linear", "random_windows"]:
            raise ValueError(f"mode must be 'episode_linear' or 'random_windows', got {self.mode}")


@dataclass
class LoggingConfig:
    """Configuration for logging and output."""

    # Output directories
    output_dir: str = "output"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # Logging level and format
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    enable_json_logging: bool = False

    # Checkpointing
    save_every_epochs: int = 1

    # Debugging
    debug_sample: bool = False
    debug_sample_size: int = 100


@dataclass
class RuntimeConfig:
    """Configuration for runtime behavior."""

    # Hardware
    device: str = "auto"  # auto, cpu, cuda, mps
    seed: int = 42

    # Performance
    enable_profiling: bool = False
    enable_memory_tracking: bool = False

    # Error handling
    strict_validation: bool = True
    continue_on_error: bool = False


@dataclass
class Settings:
    """Unified configuration for the entire Melee AI pipeline."""

    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def __post_init__(self):
        """Validate configuration after initialization."""
        _apply_env_overrides(self)
        self._validate_config()

    def _validate_config(self):
        """Validate the configuration for consistency."""
        # Data validation
        if not self.data.data_root:
            raise ValueError("data_root cannot be empty")

        # Training validation
        if self.training.steps_per_epoch is not None and self.training.windows_per_epoch is not None:
            raise ValueError("Cannot specify both steps_per_epoch and windows_per_epoch")

        if self.training.mode not in ["episode_linear", "random_windows"]:
            raise ValueError(f"Invalid mode: {self.training.mode}")

        # Model validation
        if self.model.block_size <= 0:
            raise ValueError("block_size must be positive")

        # Preprocessing validation
        if self.preprocessing.stick_deadzone < 0 or self.preprocessing.stick_deadzone >= 1:
            raise ValueError("stick_deadzone must be in [0, 1)")

    @property
    def total_steps(self) -> int:
        """Compute total training steps."""
        if self.training.max_steps is not None:
            return self.training.max_steps

        steps_per_epoch = self.steps_per_epoch
        return self.training.epochs * steps_per_epoch

    @property
    def steps_per_epoch(self) -> int:
        """Compute steps per epoch."""
        if self.training.steps_per_epoch is not None:
            return self.training.steps_per_epoch

        if self.training.windows_per_epoch is not None:
            return self.training.windows_per_epoch // self.training.batch_size

        # Default: assume reasonable number of windows per epoch
        return 1000  # This should be tuned based on dataset size

    @property
    def data_root_path(self) -> Path:
        """Get data root as Path object."""
        return Path(self.data.data_root)

    @property
    def checkpoint_path(self) -> Path:
        """Get checkpoint directory as Path object."""
        return Path(self.logging.checkpoint_dir)

    @property
    def output_path(self) -> Path:
        """Get output directory as Path object."""
        return Path(self.logging.output_dir)


def load_settings(
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[dict] = None,
) -> Settings:
    """
    Load settings from configuration file and environment variables.

    Args:
        config_path: Path to YAML configuration file (optional)
        overrides: Dictionary of configuration overrides

    Returns:
        Configured Settings object
    """
    # Start with default settings
    settings = Settings()

    # Load from YAML file if provided
    if config_path:
        # TODO: Implement YAML loading when we add pydantic or similar
        pass

    # Apply environment variable overrides
    _apply_env_overrides(settings)

    # Apply explicit overrides
    if overrides:
        _apply_dict_overrides(settings, overrides)

    # Validate final configuration
    settings._validate_config()

    return settings


def _apply_env_overrides(settings: Settings):
    """Apply environment variable overrides to settings."""
    # Data configuration
    if "MELEE_AI_DATA_ROOT" in os.environ:
        settings.data.data_root = os.environ["MELEE_AI_DATA_ROOT"]
    if "MELEE_AI_REPLAY_DIR" in os.environ:
        settings.data.replay_dir = os.environ["MELEE_AI_REPLAY_DIR"]

    # Training configuration
    if "MELEE_AI_BATCH_SIZE" in os.environ:
        settings.training.batch_size = int(os.environ["MELEE_AI_BATCH_SIZE"])
    if "MELEE_AI_LEARNING_RATE" in os.environ:
        settings.training.lr = float(os.environ["MELEE_AI_LEARNING_RATE"])
    if "MELEE_AI_EPOCHS" in os.environ:
        settings.training.epochs = int(os.environ["MELEE_AI_EPOCHS"])

    # Logging configuration
    if "MELEE_AI_LOG_LEVEL" in os.environ:
        settings.logging.log_level = os.environ["MELEE_AI_LOG_LEVEL"]
    if "MELEE_AI_OUTPUT_DIR" in os.environ:
        settings.logging.output_dir = os.environ["MELEE_AI_OUTPUT_DIR"]

    # Runtime configuration
    if "MELEE_AI_DEVICE" in os.environ:
        settings.runtime.device = os.environ["MELEE_AI_DEVICE"]
    if "MELEE_AI_SEED" in os.environ:
        settings.runtime.seed = int(os.environ["MELEE_AI_SEED"])


def _apply_dict_overrides(settings: Settings, overrides: dict):
    """Apply dictionary overrides to settings."""
    # Simple implementation - in practice, we'd use a more sophisticated
    # nested dictionary update mechanism
    for key, value in overrides.items():
        if "." in key:
            # Handle nested keys like "training.batch_size"
            parts = key.split(".")
            if len(parts) == 2:
                section, param = parts
                if hasattr(settings, section):
                    section_obj = getattr(settings, section)
                    if hasattr(section_obj, param):
                        setattr(section_obj, param, value)
        else:
            # Handle top-level keys
            if hasattr(settings, key):
                setattr(settings, key, value)
