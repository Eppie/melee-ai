from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field
from pydantic_settings import SettingsConfigDict


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
    num_workers: int = Field(default=8, ge=1)
    normalize_advantages: bool = True
    value_clip: Optional[float] = Field(default=None, gt=0)

    # Distributed PPO settings
    rollout_length: int = Field(default=5000, ge=100)
    """Number of frames to collect per rollout before training."""

    opponent_rotation_interval: int = Field(default=10000, ge=1000)
    """Frames between opponent model swaps."""

    warmup_frames: int = Field(default=256, ge=1)
    """Frames to buffer before model predictions start. Should match seq_len for consistent shapes."""

    distributed_mode: bool = Field(default=False)
    """Use distributed architecture with centralized GPU inference."""
