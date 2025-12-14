from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_settings import SettingsConfigDict


class RLConfig(BaseModel):
    """RL-specific training knobs (shared reward weights live in RewardConfig)."""

    model_config = SettingsConfigDict(validate_assignment=True, extra="forbid")

    value_loss_coef: float = Field(
        default=1,
        ge=0,
        description=(
            "Coefficient for value function loss in multi-task learning. Balances policy gradient loss vs value prediction loss. "
            "IMPORTANT: During imitation learning, value head trains on full distribution while policy heads train on "
            "filtered high-value states. Low coefficient (0.05) prevents value head gradients from corrupting transformer. "
            "For PPO self-play, increase to 0.5 for better value estimation. "
            "Effect: Higher values (0.5-2.0) prioritize accurate value estimation; lower values (0.05-0.1) prevent instability. "
            "Reasonable range: [0.05, 2.0]. Imitation: 0.05 (stable), PPO: 0.5 (standard). "
            "Interacts with: separate value head gradient clipping (max_norm=1.0), train.grad_clip (global clipping)."
        ),
    )
