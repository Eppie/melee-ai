from __future__ import annotations

import platform
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import SettingsConfigDict
from zarr.codecs import BloscCodec, BloscShuffle

from constants import BUTTON_TARGET_NAMES
from controller_utils import (
    CONTROL_STICK_QUANTIZED,
    C_STICK_QUANTIZED,
    SHOULDER_QUANTIZED,
)


def _get_default_paths() -> tuple[str, str, str]:
    """
    Automatically determine default paths based on operating system.
    Returns (input_root, out_root, validation_root)
    """
    system = platform.system()

    if system == "Darwin":
        return (
            "/Users/eppie/Downloads/ALL_REPLAYS/FOX_vs_FOX",
            "/Users/eppie/PycharmProjects/nano-melee/processed_data_250",
            "/Users/eppie/PycharmProjects/nano-melee/validation_set",
        )
    elif system == "Linux":
        return (
            "/home/eppie/hal/replays",
            "/home/eppie/melee-ai/processed_data_250",
            "/home/eppie/melee-ai/validation_set",
        )
    raise ValueError(f"Unknown operating system: {system}")


class ZarrConfig(BaseModel):
    model_config = SettingsConfigDict(
        validate_assignment=True,
        frozen=False,
        extra="forbid",
        str_strip_whitespace=True,
        arbitrary_types_allowed=True,  # Allow BloscCodec
    )

    input_root: str = Field(
        default_factory=lambda: _get_default_paths()[0],
        description="Root directory for input replay files (auto-detected by OS)",
    )
    out_root: str = Field(
        default_factory=lambda: _get_default_paths()[1],
        description="Root directory for processed output data (auto-detected by OS)",
    )
    validation_root: str = Field(
        default_factory=lambda: _get_default_paths()[2],
        description="Root directory for validation data (auto-detected by OS)",
    )
    episode_count: int = Field(
        default=250, ge=1, description="Number of episodes to process"
    )
    validation_count: int = Field(
        default=10,
        ge=1,
        description="Number of validation episodes (automatically matches episode_count by default)",
    )
    shard_size: int = Field(
        default=100, ge=1, description="Number of episodes per shard"
    )
    target_chunk_mb: float = Field(
        default=8.0, gt=0, description="Target chunk size in megabytes"
    )
    seed: int = Field(default=42, description="Random seed for reproducibility")

    # Blosc compressor - must always be a valid codec
    compressor: BloscCodec = Field(
        default_factory=lambda: BloscCodec(
            cname="zstd", clevel=3, shuffle=BloscShuffle.bitshuffle
        ),
        description="Blosc compressor configuration",
    )

    def __init__(self, **data):
        """Override init to debug compressor initialization."""
        super().__init__(**data)
        # Debug: Check if compressor got set
        if self.compressor is None:
            import warnings

            warnings.warn(
                "Compressor is None after __init__, this indicates field_validator "
                "or model_validator is setting it to None. Check validators!",
                UserWarning,
            )

    @classmethod
    @field_validator("compressor", mode="before")
    def validate_compressor(cls, v):
        """Ensure compressor is always a valid BloscCodec."""
        # If None, create default
        if v is None:
            return BloscCodec(cname="zstd", clevel=3, shuffle=BloscShuffle.bitshuffle)

        # If it's a dict (from JSON), reconstruct the BloscCodec
        if isinstance(v, dict):
            # Handle Zarr v3 codec dict format
            if "configuration" in v:
                config = v["configuration"]
                cname = config.get("cname", "zstd")
                clevel = config.get("clevel", 3)
                shuffle_str = config.get("shuffle", "bitshuffle")
                shuffle = (
                    BloscShuffle[shuffle_str]
                    if isinstance(shuffle_str, str)
                    else shuffle_str
                )
                return BloscCodec(cname=cname, clevel=clevel, shuffle=shuffle)

            # Handle simple dict format
            cname = v.get("cname", "zstd")
            clevel = v.get("clevel", 3)
            shuffle_val = v.get("shuffle", BloscShuffle.bitshuffle)

            # Handle shuffle as string or enum
            if isinstance(shuffle_val, str):
                shuffle = BloscShuffle[shuffle_val]
            elif isinstance(shuffle_val, int):
                shuffle = BloscShuffle(shuffle_val)
            else:
                shuffle = shuffle_val

            return BloscCodec(cname=cname, clevel=clevel, shuffle=shuffle)

        if isinstance(v, BloscCodec):
            return v

        return BloscCodec(cname="zstd", clevel=3, shuffle=BloscShuffle.bitshuffle)

    @model_validator(mode="after")
    def validate_paths_and_sharding(self):
        """Validate paths exist and sharding makes sense."""
        if self.compressor is None:
            import warnings

            warnings.warn(
                "Compressor was None in model_validator (after field validation). "
                "This suggests an issue with Pydantic field initialization order.",
                UserWarning,
            )
            # Use object.__setattr__ to bypass frozen config if needed
            object.__setattr__(
                self,
                "compressor",
                BloscCodec(cname="zstd", clevel=3, shuffle=BloscShuffle.bitshuffle),
            )

        # Double-check it's a valid BloscCodec instance
        if not isinstance(self.compressor, BloscCodec):
            import warnings

            warnings.warn(
                f"Compressor is type {type(self.compressor)}, converting to BloscCodec",
                UserWarning,
            )
            object.__setattr__(
                self,
                "compressor",
                BloscCodec(cname="zstd", clevel=7, shuffle=BloscShuffle.bitshuffle),
            )

        # Warn if episode_count < shard_size (not an error, just inefficient)
        if self.episode_count < self.shard_size:
            import warnings

            warnings.warn(
                f"episode_count ({self.episode_count}) < shard_size ({self.shard_size}). "
                f"Consider reducing shard_size for efficiency.",
                UserWarning,
            )

        return self

    def update_out_root_for_episode_count(self) -> None:
        """
        Update out_root to include episode_count in the path.
        Call this after setting episode_count to keep paths in sync.
        """
        # Extract base path without episode count suffix
        base_path = str(self.out_root)
        # Remove any existing _N suffix
        import re

        base_path = re.sub(r"_\d+", "", base_path)
        # Add new episode count
        self.out_root = f"{base_path}_{self.episode_count}"


class TrainConfig(BaseModel):
    """Pydantic version of TrainConfig with validation."""

    model_config = SettingsConfigDict(
        validate_assignment=True,
        extra="forbid",
    )

    batch_size: int = Field(default=128, ge=1)
    epochs: int = Field(default=10, ge=1)
    lr: float = Field(default=1.3e-4, gt=0)
    weight_decay: float = Field(default=0.002, ge=0)
    betas: Tuple[float, float] = Field(default=(0.9, 0.95))
    warmup_steps: int = Field(default=5000, ge=0)
    max_steps: Optional[int] = Field(default=None, ge=1)
    num_workers: int = Field(default=16, ge=0)
    prefetch_factor: int = Field(default=4, ge=1)
    pin_memory: bool = Field(
        default_factory=lambda: _should_pin_memory(),
        description="Pin memory for faster data transfer (auto-detected based on device)",
    )
    persistent_workers: bool = True
    stride: int = Field(default=1, ge=1)

    # Losses
    grad_clip: float = Field(default=5.0, gt=0)
    label_smoothing: float = Field(default=0.02, ge=0, le=1)

    # AMP - auto-detect optimal dtype based on hardware
    use_amp: bool = Field(
        default_factory=lambda: _should_use_amp(),
        description="Use Automatic Mixed Precision (auto-detected based on hardware)",
    )
    amp_dtype: str = Field(
        default_factory=lambda: _get_optimal_amp_dtype(),
        description="AMP dtype (auto-detected: float16 on supported accelerators, float32 on CPU)",
    )

    # Checkpointing
    out_dir: str = "checkpoints"
    save_every_epochs: int = Field(default=1, ge=1)

    @field_validator("betas")
    @classmethod
    def validate_betas(cls, v):
        """Ensure beta values are in valid range."""
        if not (0 <= v[0] < 1 and 0 <= v[1] < 1):
            raise ValueError("Beta values must be in [0, 1)")
        return v

    @field_validator("amp_dtype")
    @classmethod
    def validate_amp_dtype(cls, v):
        """Ensure amp_dtype is valid."""
        valid_dtypes = ["float16", "float32"]
        if v not in valid_dtypes:
            raise ValueError(f"amp_dtype must be one of {valid_dtypes}, got {v}")
        return v


def _should_pin_memory() -> bool:
    """
    Auto-detect if memory pinning should be enabled.
    Pin memory is beneficial for CUDA but not for MPS or CPU.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return True
        # MPS (Apple Silicon) doesn't benefit from pinning
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return False
    except ImportError:
        pass
    return False


def _should_use_amp() -> bool:
    """
    Auto-detect if Automatic Mixed Precision should be enabled.
    AMP is beneficial for modern GPUs with tensor cores.
    """
    try:
        import torch

        # Check for CUDA with compute capability >= 7.0 (Volta+, has tensor cores)
        if torch.cuda.is_available():
            # Get compute capability of first GPU
            major, minor = torch.cuda.get_device_capability(0)
            return major >= 7  # Volta (7.0) and newer
        # Apple Silicon supports AMP well
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return True
    except (ImportError, RuntimeError):
        pass
    return False


def _get_optimal_amp_dtype() -> str:
    """
    Auto-detect optimal AMP dtype based on hardware.
    - float16: GPUs and Apple Silicon with AMP support
    - float32: CPU or unsupported hardware
    """
    try:
        import torch

        if torch.cuda.is_available():
            # Get compute capability
            major, minor = torch.cuda.get_device_capability(0)
            # Volta/Turing (7.x) and newer support float16
            if major >= 7:
                return "float16"
        # Apple Silicon supports float16 well
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "float16"
    except (ImportError, RuntimeError):
        pass
    return "float32"  # Safe fallback when AMP is unavailable


class LossConfig(BaseModel):
    """Configuration related to loss computation and weighting."""

    model_config = SettingsConfigDict(validate_assignment=True, extra="forbid")

    enable_class_balancing: bool = Field(
        default=True,
        description="Enable class-balanced weights for cross-entropy losses.",
    )
    ce_weight_min: float = Field(
        default=0.1,
        gt=0,
        description="Minimum clamp value applied to class weights.",
    )
    ce_weight_max: float = Field(
        default=10.0,
        gt=0,
        description="Maximum clamp value applied to class weights.",
    )
    enable_pos_weighting: bool = Field(
        default=True,
        description="Enable positive-class weighting for multi-label BCE losses.",
    )
    pos_weight_max: float = Field(
        default=10.0,
        gt=0,
        description="Maximum clamp value applied to positive-class weights.",
    )
    use_weighted_component_means: bool = Field(
        default=True,
        description="Apply provided sample weights when averaging component losses.",
    )
    main_change: float = Field(default=5.0, gt=0)
    c_change: float = Field(default=10.0, gt=0)
    shoulder_change: float = Field(default=5.0, gt=0)
    buttons_change_default: float = Field(default=10.0, gt=0)
    button_z: float = Field(default=20.0, gt=0)
    button_b: float = Field(default=12.0, gt=0)
    button_a: float = Field(default=12.0, gt=0)
    button_xy: float = Field(default=10.0, gt=0)
    button_lr: float = Field(default=8.0, gt=0)
    hold_base: float = Field(default=1.0, gt=0)
    value_change: float = Field(default=8.0, gt=0)


class ProfileConfig(BaseModel):
    """Pydantic version of ProfileConfig."""

    model_config = SettingsConfigDict(validate_assignment=True, extra="forbid")

    enable: bool = False
    out_dir: Optional[str] = None
    wait: int = Field(default=1, ge=0)
    warmup: int = Field(default=1, ge=0)
    active: int = Field(default=10, ge=1)
    repeat: int = Field(default=1, ge=1)
    record_shapes: bool = True
    with_stack: bool = False
    profile_memory: bool = False


from constants import BUTTON_TARGET_NAMES
from controller_utils import (
    CONTROL_STICK_QUANTIZED,
    C_STICK_QUANTIZED,
    SHOULDER_QUANTIZED,
)


class GPTConfig(BaseModel):
    """Pydantic version of GPTConfig."""

    model_config = SettingsConfigDict(validate_assignment=True, extra="forbid")

    block_size: int = Field(default=512, ge=1)
    n_embd: int = Field(default=512, ge=1)
    n_layer: int = Field(default=4, ge=1)
    n_head: int = Field(default=8, ge=1)
    dropout: float = Field(default=0.03, ge=0, le=1)
    input_size: int = Field(default=-1, description="Computed dynamically")
    num_stages: int = Field(default=6, ge=1)
    num_characters: int = Field(default=26, ge=1)
    num_actions: int = Field(default=396, ge=1)
    gamma: float = Field(default=0.999, ge=0, le=1)
    norm_type: Literal["layernorm", "rmsnorm"] = "layernorm"
    norm_eps: float = Field(default=1e-7, gt=0)
    norm_affine: bool = True
    norm_placement: Literal["pre", "post", "both"] = "post"
    attention_type: Literal["mha", "gqa", "mqa"] = "gqa"
    n_kv_head: Optional[int] = Field(default=4, ge=1)
    rope_theta: float = Field(default=10000.0, gt=0)
    ffn_mult: float = Field(default=2, gt=0)
    ffn_activation: Literal["gelu", "geglu", "swiglu", "relu"] = "geglu"
    head_flow: Literal["sequential", "parallel"] = "parallel"
    target_shapes_by_head: Dict[str, int] = Field(
        default_factory=lambda: {
            "main_stick": len(CONTROL_STICK_QUANTIZED),
            "c_stick": len(C_STICK_QUANTIZED),
            "buttons": len(BUTTON_TARGET_NAMES),
            "shoulder": len(SHOULDER_QUANTIZED),
        }
    )
    use_value_head: bool = True

    @model_validator(mode="before")
    def compute_input_size(cls, values):
        """Dynamically compute input_size if dimensions are provided in context."""
        if "context" in values and values["context"]:
            context = values["context"]
            gamestate_dim = context.get("gamestate_dim")
            controller_dim = context.get("controller_dim")

            if gamestate_dim is not None and controller_dim is not None:
                values["input_size"] = (
                    values.get("num_stages", 6)
                    + values.get("num_characters", 26) * 2
                    + values.get("num_actions", 396) * 2
                    + gamestate_dim
                    + controller_dim
                )
        return values

    @model_validator(mode="after")
    def validate_attention_heads(self):
        """Ensure n_kv_head is compatible with n_head."""
        if self.attention_type in ("gqa", "mqa"):
            if self.n_kv_head is None:
                raise ValueError(f"{self.attention_type} requires n_kv_head to be set")
            if self.n_head % self.n_kv_head != 0:
                raise ValueError(
                    f"n_head ({self.n_head}) must be divisible by n_kv_head ({self.n_kv_head})"
                )
        return self


class FeatureConfig(BaseModel):
    """Pydantic version of FeatureConfig."""

    model_config = SettingsConfigDict(validate_assignment=True, extra="forbid")

    transforms: list[Dict[str, Any]] = Field(
        default_factory=lambda: [
            {
                "transform": "stick_palette",
                "features": ["main_stick_x", "main_stick_y"],
                "palette": "fox_main",
            },
            {
                "transform": "stick_palette",
                "features": ["c_stick_x", "c_stick_y"],
                "palette": "c_stick",
            },
            {
                "transform": "scale",
                "features": ["facing"],
                "factor": 2.0,
            },
            {
                "transform": "offset",
                "features": ["facing"],
                "delta": -1.0,
            },
            {
                "transform": "scale",
                "features": ["percent"],
                "factor": 1 / 100.0,
            },
            {
                "transform": "scale",
                "features": ["shield_strength"],
                "factor": 1.0 / 60.0,
            },
            {
                "transform": "scale",
                "features": ["stock"],
                "factor": 1 / 4.0,
            },
            {
                "transform": "scale",
                "features": ["position_x", "position_y"],
                "factor": 1 / 20.0,
            },
            {
                "transform": "scale",
                "features": ["jumps_left"],
                "factor": 1 / 6.0,
            },
        ]
    )


class RLConfig(BaseModel):
    """Pydantic version of RLConfig."""

    model_config = SettingsConfigDict(validate_assignment=True, extra="forbid")

    gamma: float = Field(default=0.995, ge=0, le=1)
    value_loss_coef: float = Field(default=0.5, ge=0)
    reward_damage_dealt: float = 0.02
    reward_damage_taken: float = -0.02
    reward_stock_lost: float = -1
    reward_stock_taken: float = 1
    reward_hitlag_opponent: float = 0.02
    reward_hitlag_self: float = -0.02
    reward_low_shield: float = -0.1
    reward_per_frame: float = 0


class PPOConfig(BaseModel):
    """Pydantic version of PPOConfig."""

    model_config = SettingsConfigDict(validate_assignment=True, extra="forbid")

    pool_size: int = Field(default=5, ge=1)
    clip_ratio: float = Field(default=0.2, gt=0)
    entropy_coef: float = Field(default=0.01, ge=0)
    gae_lambda: float = Field(default=0.95, ge=0, le=1)
    lr: float = Field(default=1e-5, gt=0)
    ppo_epochs: int = Field(default=4, ge=1)
    minibatch_size: int = Field(default=64, ge=1)
    max_grad_norm: float = Field(default=0.5, gt=0)
    max_episode_frames: int = Field(default=18000, ge=1)
    num_workers: int = Field(default=1, ge=1)
    normalize_advantages: bool = True
    value_clip: Optional[float] = Field(default=None, gt=0)


class ImitationConfig(BaseModel):
    """Pydantic version of ImitationConfig."""

    model_config = SettingsConfigDict(validate_assignment=True, extra="forbid")

    strategy: Literal[
        "uniform", "value_weighted", "value_advantage", "value_filter", "hybrid"
    ] = "hybrid"
    value_k: float = Field(default=1.0, gt=0)
    value_temperature: float = Field(default=1.0, gt=0)
    value_use_exp: bool = False
    advantage_n_steps: int = Field(default=5, ge=1)
    advantage_alpha: float = Field(default=1.0, gt=0)
    advantage_use_gae: bool = False
    gae_gamma: float = Field(default=0.99, ge=0, le=1)
    gae_lambda: float = Field(default=0.95, ge=0, le=1)
    filter_percentile: float = Field(default=50.0, ge=0, le=100)
    filter_soft: bool = False
    filter_temperature: float = Field(default=1.0, gt=0)
    hybrid_strategies: list[str] = Field(
        default_factory=lambda: ["value_weighted", "value_filter"]
    )
    hybrid_weights: list[float] = Field(default_factory=lambda: [0.5, 0.5])

    @model_validator(mode="after")
    def validate_hybrid(self):
        """Ensure hybrid strategies and weights are consistent."""
        if self.strategy == "hybrid":
            if len(self.hybrid_strategies) != len(self.hybrid_weights):
                raise ValueError(
                    "hybrid_strategies and hybrid_weights must have same length"
                )
            if abs(sum(self.hybrid_weights) - 1.0) > 1e-6:
                raise ValueError("hybrid_weights must sum to 1.0")
        return self


class Config(BaseModel):
    """Main Pydantic configuration with all sub-configs."""

    model_config = SettingsConfigDict(
        validate_assignment=True,
        extra="forbid",
        frozen=False,  # Can be frozen after initialization
    )

    seq_len: int = Field(default=256, ge=1)
    zarr: ZarrConfig = Field(default_factory=ZarrConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    model: GPTConfig = Field(default_factory=GPTConfig)
    profile: ProfileConfig = Field(default_factory=ProfileConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    rl: RLConfig = Field(default_factory=RLConfig)
    ppo: PPOConfig = Field(default_factory=PPOConfig)
    loss_weights: LossConfig = Field(default_factory=LossConfig)
    imitation: ImitationConfig = Field(default_factory=ImitationConfig)

    def freeze(self) -> None:
        """Make config immutable."""
        self.model_config["frozen"] = True

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, s: str) -> "Config":
        """Deserialize from JSON string."""
        return cls.model_validate_json(s)

    def save(self, path: Union[str, Path]) -> Path:
        """Save config to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Config":
        """Load config from JSON file."""
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


def apply_overrides_(cfg: Config, overrides: Dict[str, str]) -> None:
    """
    Apply dotted-key overrides to Pydantic config.
    Example: {"train.lr": "0.001", "model.n_layer": "6"}
    """
    for dotted_key, raw_value in overrides.items():
        parts = dotted_key.split(".")

        # Navigate to the parent
        parent = cfg
        for part in parts[:-1]:
            parent = getattr(parent, part)

        # Set the final attribute (Pydantic will validate automatically)
        field_name = parts[-1]

        # Get the field type for proper coercion
        field_info = parent.model_fields.get(field_name)
        if field_info is None:
            raise ValueError(f"Unknown field: {dotted_key}")

        # Pydantic will handle type conversion, but we can help with common cases
        try:
            if raw_value.lower() in ("true", "false"):
                value = raw_value.lower() == "true"
            elif "." in raw_value or "e" in raw_value.lower():
                value = float(raw_value)
            elif raw_value.isdigit() or (
                raw_value[0] == "-" and raw_value[1:].isdigit()
            ):
                value = int(raw_value)
            else:
                value = raw_value

            setattr(parent, field_name, value)
        except Exception as e:
            raise ValueError(f"Failed to set {dotted_key}={raw_value}: {e}")


_GLOBAL_CONFIG: Optional[Config] = None


def init_config(
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, str]] = None,
    gamestate_dim: Optional[int] = None,
    controller_dim: Optional[int] = None,
    freeze: bool = True,
) -> Config:
    """
    Initialize global config singleton. Replaces old init_config().

    Args:
        config_path: Optional path to JSON config file
        overrides: Optional dict of CLI overrides (e.g., {"train.lr": "0.001"})
        freeze: If True, make config immutable after initialization

    Returns:
        Initialized Config instance

    Example:
        # Initialize with defaults
        config = init_config()

        # Initialize from file
        config = init_config("config.json")

        # Initialize with overrides
        config = init_config(overrides={"train.lr": "0.001"})
    """
    global _GLOBAL_CONFIG

    context = {}
    if gamestate_dim is not None and controller_dim is not None:
        context["gamestate_dim"] = gamestate_dim
        context["controller_dim"] = controller_dim

    # Load from file or create with defaults
    if config_path:
        # When loading from a file, we first load the data, then validate with context
        data = Config.model_validate_json(Path(config_path).read_text(encoding="utf-8"))
        cfg = Config.model_validate(data, context=context)
    else:
        cfg = Config.model_validate({}, context=context)

    # Debug: verify compressor is valid after creation
    if cfg.zarr.compressor is None:
        import warnings

        warnings.warn(
            "CRITICAL: compressor is None after Config creation! Creating default.",
            UserWarning,
        )
        cfg.zarr.compressor = BloscCodec(
            cname="zstd", clevel=7, shuffle=BloscShuffle.bitshuffle
        )

    # Apply CLI overrides if provided
    if overrides:
        apply_overrides_(cfg, overrides)

    # Final safety check
    if cfg.zarr.compressor is None:
        raise RuntimeError(
            "Compressor is None after initialization! This should never happen. "
            "Check your config file or initialization code."
        )

    # Optionally freeze to prevent modifications
    if freeze:
        cfg.model_config["frozen"] = True

    _GLOBAL_CONFIG = cfg
    return cfg


def get_config() -> Config:
    """
    Get global config singleton. Replaces old get_config().

    Returns:
        Config instance

    Raises:
        RuntimeError: If config not initialized (call init_config() first)

    Example:
        # In main script
        init_config()

        # Anywhere else in your code
        cfg = get_config()
        print(cfg.train.lr)
    """
    if _GLOBAL_CONFIG is None:
        raise RuntimeError("Global config not initialized. Call init_config() first.")
    return _GLOBAL_CONFIG


def reset_config() -> None:
    """
    Reset global config. Useful for testing.

    Example:
        def test_something():
            init_config()
            # ... test code ...
            reset_config()  # Clean up for next test
    """
    global _GLOBAL_CONFIG
    _GLOBAL_CONFIG = None


def has_config() -> bool:
    """Check if global config has been initialized."""
    return _GLOBAL_CONFIG is not None


if __name__ == "__main__":
    # Test 1: Create ZarrConfig directly
    print("Test 1: ZarrConfig()")
    zarr_cfg = ZarrConfig()
    print(f"  compressor after init: {zarr_cfg.compressor}")
    print(f"  compressor type: {type(zarr_cfg.compressor)}")

    # Test 2: Create full Config (which creates ZarrConfig)
    print("\nTest 2: Config()")
    cfg = Config()
    print(f"  compressor after init: {cfg.zarr.compressor}")
    print(f"  compressor type: {type(cfg.zarr.compressor)}")

    # Test 3: Explicit compressor
    print("\nTest 3: Explicit compressor")
    zarr_cfg2 = ZarrConfig(compressor=BloscCodec(cname="zstd", clevel=7))
    print(f"  compressor: {zarr_cfg2.compressor}")
