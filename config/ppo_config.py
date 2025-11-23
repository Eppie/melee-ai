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
    num_workers: int = Field(default=1, ge=1)
    normalize_advantages: bool = True
    value_clip: Optional[float] = Field(default=None, gt=0)
