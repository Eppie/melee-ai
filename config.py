from __future__ import annotations

import platform
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator
from pydantic_settings import SettingsConfigDict
from zarr.codecs import BloscCodec, BloscShuffle

import torch

from column_map import ColumnMap
from constants import BUTTON_TARGET_NAMES
from controller_utils import (
    CONTROL_STICK_QUANTIZED,
    C_STICK_QUANTIZED,
    SHOULDER_QUANTIZED,
)
from schema import get_feature_names, get_target_names


def _get_default_paths() -> tuple[str, str, str]:
    """
    Automatically determine default paths based on operating system.
    Returns (input_root, out_root, validation_root)
    """
    system = platform.system()

    if system == "Darwin":
        return (
            "/Users/eppie/Downloads/ALL_REPLAYS/FOX_vs_FOX",
            "/Users/eppie/PycharmProjects/nano-melee/processed_data_1000",
            "/Users/eppie/PycharmProjects/nano-melee/validation_set",
        )
    elif system == "Linux":
        return (
            "/home/eppie/hal/replays",
            "/home/eppie/melee-ai/processed_data_1000",
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

    input_root: str = Field(default_factory=lambda: _get_default_paths()[0])
    out_root: str = Field(default_factory=lambda: _get_default_paths()[1])
    validation_root: str = Field(default_factory=lambda: _get_default_paths()[2])
    episode_count: int = Field(default=6600, ge=1)
    validation_count: int = Field(default=400, ge=1)
    shard_size: int = Field(default=100, ge=1)
    target_chunk_mb: float = Field(default=8.0, gt=0)
    chunk_frames: int = Field(
        default=512,
        ge=1,
        description=(
            "Preferred number of frames per Zarr chunk along the time axis. "
            "Defaults to 512 so 256-frame windows typically stay within a single chunk."
        ),
    )
    seed: int = Field(default=42)
    compressor: BloscCodec = Field(
        default_factory=lambda: BloscCodec(
            cname="zstd", clevel=3, shuffle=BloscShuffle.bitshuffle
        )
    )

    @field_validator("compressor", mode="before")
    def validate_compressor(cls, v):
        """Ensure compressor is always a valid BloscCodec."""
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
            # Use object.__setattr__ to bypass frozen config if needed
            object.__setattr__(
                self,
                "compressor",
                BloscCodec(cname="zstd", clevel=3, shuffle=BloscShuffle.bitshuffle),
            )

        # Double-check it's a valid BloscCodec instance
        if not isinstance(self.compressor, BloscCodec):
            object.__setattr__(
                self,
                "compressor",
                BloscCodec(cname="zstd", clevel=7, shuffle=BloscShuffle.bitshuffle),
            )

        self.update_out_root_for_episode_count()
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

        base_path = re.sub(r"_(\d+)$", "", base_path)
        object.__setattr__(self, "out_root", f"{base_path}_{self.episode_count}")


@lru_cache(maxsize=1)
def _schema_feature_dims() -> Tuple[int, int]:
    """Return canonical (gamestate_dim, controller_dim) from the schema."""
    colmap = ColumnMap(get_feature_names(), get_target_names())
    return len(colmap.gamestate_idxs), len(colmap.controller_idxs)


class TrainConfig(BaseModel):
    """Pydantic version of TrainConfig with validation."""

    model_config = SettingsConfigDict(
        validate_assignment=True,
        extra="forbid",
    )

    batch_size: int = Field(default=128, ge=1)
    epochs: int = Field(default=16, ge=1)
    lr: float = Field(default=1.3e-4, gt=0)
    # TODO: Document the effect of this setting
    weight_decay: float = Field(default=0.002, ge=0)
    # TODO: Document the effect of this setting
    betas: Tuple[float, float] = Field(default=(0.9, 0.95))
    warmup_steps: int = Field(default=30000, ge=0)
    num_workers: int = Field(default=16, ge=0)
    prefetch_factor: int = Field(default=4, ge=1)
    max_loader_prefetch_mb: int = Field(default=2048, ge=1)
    pin_memory: bool = Field(default_factory=lambda: _should_pin_memory())
    persistent_workers: bool = True
    stride: int = Field(default=8, ge=1)
    worker_start_method: Optional[Literal["fork", "spawn", "forkserver"]] = None

    # Losses
    grad_clip: float = Field(default=5.0, gt=0)
    label_smoothing: float = Field(default=0.02, ge=0, le=1)
    schedule_warmup_epochs: int = Field(default=1, ge=0)
    schedule_cooldown_epochs: int = Field(default=1, ge=0)
    use_amp: bool = Field(default_factory=lambda: _should_use_amp())
    amp_dtype: str = Field(default_factory=lambda: _get_optimal_amp_dtype())

    # Checkpointing
    out_dir: str = "checkpoints"


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


class LossConfig(BaseModel):
    """Configuration related to loss computation and weighting."""

    model_config = SettingsConfigDict(validate_assignment=True, extra="forbid")

    enable_class_balancing: bool = Field(default=True)
    ce_weight_min: float = Field(default=0.1, gt=0)
    ce_weight_max: float = Field(default=10.0, gt=0)
    enable_pos_weighting: bool = Field(default=True)
    pos_weight_max: float = Field(default=3.0, gt=0)
    use_weighted_component_means: bool = Field(default=True)
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
    value_change: float = Field(default=4.0, gt=0)


class GPTConfig(BaseModel):
    model_config = SettingsConfigDict(validate_assignment=True, extra="forbid")

    block_size: int = Field(default=512, ge=1)
    n_embd: int = Field(default=512, ge=1)
    n_layer: int = Field(default=16, ge=1)
    n_head: int = Field(default=8, ge=1)
    dropout: float = Field(default=0.03, ge=0, le=1)
    input_size: int = Field(default=-1)
    num_stages: int = Field(default=6, ge=1)
    num_characters: int = Field(default=26, ge=1)
    num_actions: int = Field(default=396, ge=1)
    gamma: float = Field(default=0.999, ge=0, le=1)
    n_kv_head: Optional[int] = Field(default=4, ge=1)
    head_flow: Literal["sequential", "parallel"] = "parallel"
    target_shapes_by_head: Dict[str, int] = Field(
        default_factory=lambda: {
            "main_stick": len(CONTROL_STICK_QUANTIZED),
            "c_stick": len(C_STICK_QUANTIZED),
            "buttons": len(BUTTON_TARGET_NAMES),
            "shoulder": len(SHOULDER_QUANTIZED),
        }
    )

    # TODO: This wasn't working before, so we might have implemented the same logic elsewhere, find it and remove it
    @model_validator(mode="before")
    def compute_input_size(cls, data: Any, info: ValidationInfo):
        """Dynamically compute input_size if context provides dimensions."""
        if not isinstance(data, dict):
            return data

        # Allow explicit overrides to take precedence.
        if "input_size" in data and data["input_size"] not in (-1, None):
            return data

        context = (info.context or {}) if info is not None else {}
        gamestate_dim = context.get("gamestate_dim")
        controller_dim = context.get("controller_dim")

        if gamestate_dim is None or controller_dim is None:
            default_gamestate, default_controller = _schema_feature_dims()
            gamestate_dim = gamestate_dim or default_gamestate
            controller_dim = controller_dim or default_controller

        num_stages = data.get("num_stages", cls.model_fields["num_stages"].default)
        num_characters = data.get(
            "num_characters", cls.model_fields["num_characters"].default
        )
        num_actions = data.get("num_actions", cls.model_fields["num_actions"].default)

        data = dict(data)
        data["input_size"] = (
            num_stages
            + num_characters * 2
            + num_actions * 2
            + gamestate_dim
            + controller_dim
        )
        return data


# TODO: Remove this over-engineering - we are going to stick with these feature transforms
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
    model_config = SettingsConfigDict(validate_assignment=True, extra="forbid")
    # TODO: Document what effect this has
    gamma: float = Field(default=0.995, ge=0, le=1)
    # TODO: Document what effect this has
    value_loss_coef: float = Field(default=0.5, ge=0)
    reward_damage_dealt: float = 0.02
    reward_stock_taken: float = 1
    reward_hitlag_opponent: float = 0.02
    reward_low_shield: float = -0.1


# TODO: This is almost totally untested
class PPOConfig(BaseModel):
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


# TODO: Currently unused, decide to either remove or actually implement.
class ImitationConfig(BaseModel):
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

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain-Python representation (for checkpoints, etc.)."""
        return self.model_dump()

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

    @model_validator(mode="after")
    def apply_context_defaults(self, info: ValidationInfo):
        # TODO: Do we want this, or do we want compute_input_size?
        """Propagate context-aware defaults to nested configs."""
        context = info.context or {}
        gamestate_dim = context.get("gamestate_dim")
        controller_dim = context.get("controller_dim")

        if self.model.input_size in (-1, None):
            if gamestate_dim is None or controller_dim is None:
                default_gamestate, default_controller = _schema_feature_dims()
                gamestate_dim = gamestate_dim or default_gamestate
                controller_dim = controller_dim or default_controller

            self.model.input_size = (
                self.model.num_stages
                + self.model.num_characters * 2
                + self.model.num_actions * 2
                + gamestate_dim
                + controller_dim
            )

        return self


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

    # Apply CLI overrides if provided
    if overrides:
        apply_overrides_(cfg, overrides)

    # Optionally freeze to prevent modifications
    if freeze:
        cfg.model_config["frozen"] = True

    _GLOBAL_CONFIG = cfg
    return cfg


def get_config() -> Config:
    """
    Get global config singleton.

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
