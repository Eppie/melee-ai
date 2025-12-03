"""PPO-specific configuration."""

from pathlib import Path
from typing import List

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import SettingsConfigDict
from schema import get_feature_names


class PPOConfig(BaseModel):
    """Configuration for hierarchical PPO training system."""

    model_config = SettingsConfigDict(
        validate_assignment=True,
        extra="forbid",
        frozen=False,
    )

    # Infrastructure
    num_shards: int = Field(
        default=12,
        ge=1,
        description="Number of ArenaShard (S8) processes",
    )
    envs_per_shard: int = Field(
        default=8,
        ge=1,
        description="Dolphin instances per shard",
    )
    dolphin_path: str = Field(
        default="/Applications/Slippi Dolphin.app",
        description="Path to Slippi Dolphin executable",
    )
    iso_path: str = Field(
        default="~/Documents/SSBM.iso",
        description="Path to SSBM ISO file",
    )

    # Game settings
    character: str = Field(
        default="FOX",
        description="Character for both players (fixed)",
    )
    stages: List[str] = Field(
        default=["FD", "BF", "YS", "FoD", "PS", "DL"],
        description="Stage pool for randomization (tournament legal)",
    )
    bot_port: int = Field(
        default=1,
        ge=1,
        le=4,
        description="Controller port for ego player",
    )
    opp_port: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Controller port for opponent player",
    )

    # Rollout collection
    rollout_length: int = Field(
        default=1024,
        ge=1,
        description="Frames per rollout before PPO update",
    )
    context_length: int = Field(
        default=256,
        ge=1,
        description="Model context window (ring buffer size)",
    )
    feature_dim: int = Field(
        default_factory=lambda: len(get_feature_names()),
        ge=1,
        description=(
            "Per-frame feature dimension for shared memory/ring buffers. "
            "Coordinator will overwrite this with the schema-derived size."
        ),
    )
    warmup_frames: int = Field(
        default=256,
        ge=0,
        description="Neutral action warmup frames (excluded from training)",
    )
    restart_interval: int = Field(
        default=10000,
        ge=1,
        description="Frames between Dolphin restarts (prevent memory leaks)",
    )

    # PPO hyperparameters
    lr: float = Field(
        default=3e-4,
        gt=0.0,
        description="Learning rate",
    )
    clip_epsilon: float = Field(
        default=0.2,
        gt=0.0,
        description="PPO clipping parameter",
    )
    ppo_epochs: int = Field(
        default=4,
        ge=1,
        description="Number of PPO epochs per update",
    )
    batch_size: int = Field(
        default=128,
        ge=1,
        description="Batch size for PPO training",
    )
    value_coef: float = Field(
        default=0.5,
        ge=0.0,
        description="Value loss coefficient",
    )
    entropy_coef: float = Field(
        default=0.01,
        ge=0.0,
        description="Entropy bonus coefficient",
    )
    gamma: float = Field(
        default=0.995,
        ge=0.0,
        le=1.0,
        description="Discount factor for returns",
    )
    gae_lambda: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="GAE lambda parameter",
    )
    grad_clip: float = Field(
        default=1.0,
        gt=0.0,
        description="Gradient clipping norm",
    )
    weight_decay: float = Field(
        default=0.0,
        ge=0.0,
        description="AdamW weight decay",
    )

    # Opponent pool
    opponent_pool_size: int = Field(
        default=20,
        ge=1,
        description="Number of historical checkpoints to keep in pool",
    )
    opponent_sample_prob: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Probability of using historical opponent (vs self-play)",
    )
    pool_refresh_interval: int = Field(
        default=100,
        ge=1,
        description="Training steps between refreshing opponent pool",
    )
    checkpoint_interval: int = Field(
        default=1000,
        ge=1,
        description="Steps between saving checkpoints",
    )

    # Memory management
    rollouts_per_batch: int = Field(
        default=96,
        ge=1,
        description="Number of rollouts to accumulate before PPO training",
    )

    # Checkpointing
    init_checkpoint: Path = Field(
        description="Initial policy checkpoint to load",
    )
    checkpoint_dir: Path = Field(
        default=Path("ppo_checkpoints"),
        description="Directory for saving PPO checkpoints",
    )

    @property
    def total_envs(self) -> int:
        """Total number of environments (num_shards × envs_per_shard)."""
        return self.num_shards * self.envs_per_shard

    @field_validator("dolphin_path")
    @classmethod
    def validate_dolphin_path(cls, v: str) -> str:
        """Validate Dolphin path exists."""
        path = Path(v).expanduser()
        if not path.exists():
            raise ValueError(
                f"Dolphin path does not exist: {v}\n"
                f"Please install Slippi Dolphin or provide correct path"
            )
        return v

    @field_validator("iso_path")
    @classmethod
    def validate_iso_path(cls, v: str) -> str:
        """Validate ISO path exists."""
        path = Path(v).expanduser()
        if not path.exists():
            raise ValueError(
                f"ISO path does not exist: {v}\n"
                f"Please provide path to SSBM ISO file"
            )
        return v

    @field_validator("stages")
    @classmethod
    def validate_stages(cls, v: List[str]) -> List[str]:
        """Validate stage list."""
        valid_stages = {"FD", "BF", "YS", "FoD", "PS", "DL"}
        invalid = set(v) - valid_stages
        if invalid:
            raise ValueError(
                f"Invalid stages: {invalid}\n" f"Valid stages: {valid_stages}"
            )
        if not v:
            raise ValueError("Must specify at least one stage")
        return v

    @field_validator("character")
    @classmethod
    def validate_character(cls, v: str) -> str:
        """Validate character choice."""
        valid_chars = {"FOX", "FALCO", "MARTH", "SHEIK", "FALCON"}
        if v not in valid_chars:
            raise ValueError(
                f"Invalid character: {v}\n" f"Valid characters: {valid_chars}"
            )
        return v

    @model_validator(mode="after")
    def validate_config(self):
        """Cross-field validation."""
        # Validate total environments
        if self.total_envs < 1:
            raise ValueError("Must have at least 1 environment")

        # Validate context length vs rollout length
        if self.context_length > self.rollout_length:
            raise ValueError(
                f"context_length ({self.context_length}) must be <= "
                f"rollout_length ({self.rollout_length})"
            )

        # Validate warmup vs context
        if self.warmup_frames > self.context_length:
            raise ValueError(
                f"warmup_frames ({self.warmup_frames}) must be <= "
                f"context_length ({self.context_length})"
            )

        # Validate batch size
        if self.batch_size > self.rollout_length:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be <= "
                f"rollout_length ({self.rollout_length})"
            )

        # Validate checkpoint exists
        if not self.init_checkpoint.exists():
            raise ValueError(
                f"Initial checkpoint not found: {self.init_checkpoint}\n"
                f"Please provide a valid checkpoint from imitation learning"
            )

        return self
