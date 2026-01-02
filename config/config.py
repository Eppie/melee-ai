from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, Field, ValidationInfo, model_validator
from pydantic_settings import SettingsConfigDict

from .feature_config import FeatureConfig
from .gpt_config import GPTConfig, _schema_feature_dims
from .imitation_config import ImitationConfig
from .loss_config import LossConfig
from .rl_config import RLConfig
from .train_config import TrainConfig
from .zarr_config import ZarrConfig


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
    loss_weights: LossConfig = Field(default_factory=LossConfig)
    imitation: ImitationConfig = Field(default_factory=ImitationConfig)

    def freeze(self) -> None:
        """Make config immutable."""
        self.model_config["frozen"] = True

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return self.model_dump_json(indent=indent)

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain-Python representation (for checkpoints, etc.).

        This method produces a JSON-serializable dict that can be safely
        stored in torch checkpoints and reloaded. Complex objects like
        BloscCodec are converted to their dict representation.
        """
        data = self.model_dump()
        # Convert BloscCodec to a serializable dict representation
        if "zarr" in data and "compressor" in data["zarr"]:
            compressor = self.zarr.compressor
            data["zarr"]["compressor"] = {
                "cname": (
                    compressor.cname.value
                    if hasattr(compressor.cname, "value")
                    else compressor.cname
                ),
                "clevel": compressor.clevel,
                "shuffle": (
                    compressor.shuffle.name
                    if hasattr(compressor.shuffle, "name")
                    else str(compressor.shuffle)
                ),
            }
        return data

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
        """Propagate context-aware defaults to nested configs."""
        context = info.context or {}
        gamestate_dim = context.get("gamestate_dim")
        controller_dim = context.get("controller_dim")

        # If context provides explicit dimensions, always use them (overrides nested validator)
        # Otherwise, only compute if input_size is still unset (-1 or None)
        should_compute = (
            gamestate_dim is not None and controller_dim is not None
        ) or self.model.input_size in (-1, None)

        if should_compute:
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
            elif raw_value.isdigit() or (
                raw_value[0] == "-" and raw_value[1:].isdigit()
            ):
                value = int(raw_value)
            elif raw_value.startswith("[") or raw_value.startswith("{"):
                # Parse JSON for lists and dicts
                import json

                value = json.loads(raw_value)
            else:
                # Try to parse as float, but fall back to string if it fails
                try:
                    value = float(raw_value)
                except ValueError:
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


def set_config(config: Config) -> Config:
    """
    Set a Config instance as the global config.

    This is useful when loading a config from a checkpoint or creating
    a config programmatically.

    Args:
        config: The Config instance to set as global.

    Returns:
        The same Config instance.

    Example:
        config = Config.model_validate(checkpoint_config_dict)
        set_config(config)
    """
    global _GLOBAL_CONFIG
    _GLOBAL_CONFIG = config
    return config


def init_config_from_checkpoint(
    checkpoint_path: Union[str, Path],
    overrides: Optional[Dict[str, str]] = None,
    freeze: bool = True,
) -> Config:
    """
    Initialize global config from a checkpoint file.

    This function loads the config stored in a checkpoint and sets it as
    the global config. CLI overrides can be applied on top of the loaded
    config.

    Args:
        checkpoint_path: Path to the checkpoint file.
        overrides: Optional dict of CLI overrides to apply on top of checkpoint config.
        freeze: If True, make config immutable after initialization.

    Returns:
        Initialized Config instance.

    Raises:
        FileNotFoundError: If the checkpoint file doesn't exist.
        ValueError: If the checkpoint doesn't contain a config.

    Example:
        # Resume training with checkpoint's config
        config = init_config_from_checkpoint("checkpoints/model.pt")

        # Resume with overrides
        config = init_config_from_checkpoint(
            "checkpoints/model.pt",
            overrides={"train.lr": "1e-5"}
        )
    """
    import torch

    checkpoint_path = Path(checkpoint_path).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    config_dict = ckpt.get("config")

    if config_dict is None:
        raise ValueError(f"Checkpoint does not contain a config: {checkpoint_path}")

    # Handle legacy checkpoints that only have TrainConfig
    if "train" not in config_dict and "batch_size" in config_dict:
        # This is a legacy checkpoint with only TrainConfig.__dict__
        print(
            f"Warning: Checkpoint contains legacy TrainConfig format, using defaults for other configs"
        )
        config_dict = {"train": config_dict}

    cfg = Config.model_validate(config_dict)

    if overrides:
        apply_overrides_(cfg, overrides)

    if freeze:
        cfg.model_config["frozen"] = True

    global _GLOBAL_CONFIG
    _GLOBAL_CONFIG = cfg
    return cfg
