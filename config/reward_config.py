from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_settings import SettingsConfigDict


class RewardConfig(BaseModel):
    """Shared reward weights and discounting used by imitation and PPO."""

    model_config = SettingsConfigDict(validate_assignment=True, extra="forbid")

    gamma: float = Field(
        default=0.995,
        ge=0,
        le=1,
        description=(
            "Discount factor for future rewards. Controls how much the agent values future vs immediate rewards. "
            "Effect: Higher gamma (0.99-0.999) makes the agent more far-sighted, optimizing for long-term reward; "
            "lower gamma (0.9-0.98) makes it more myopic, focusing on immediate reward. "
            "At 60 FPS, gamma=0.995 gives effective horizon of ~200 frames (~3.3 seconds). "
            "Reasonable range: [0.99, 0.999]. Formula: effective_horizon ≈ 1/(1-gamma) frames. "
            "Interacts with: reward magnitudes (higher gamma amplifies cumulative rewards), "
            "gae_lambda in PPO (both affect advantage estimation), episode length (longer episodes need higher gamma)."
        ),
    )
    reward_damage_dealt: float = Field(
        default=0.08,
        description=(
            "Reward per percent damage dealt to opponent. Encourages aggressive play and combos. "
            "Reasonable range: [0.01, 0.2]. Interacts with: reward_stock_taken, gamma."
        ),
    )
    reward_stock_taken: float = Field(
        default=1.0,
        description=(
            "Reward for taking an opponent's stock. Large bonus for eliminations. "
            "Should be ~20-100x damage reward for proper scaling. Reasonable range: [1.0, 10.0]."
        ),
    )
    reward_hitlag_opponent: float = Field(
        default=0.08,
        description=(
            "Reward per frame opponent is in hitlag. Encourages landing hits. "
            "Reasonable range: [0.01, 0.2]. Similar scale to damage reward."
        ),
    )
    reward_low_shield: float = Field(
        default=-0.1,
        description=(
            "Penalty per frame when shield is low (encourages shield management). "
            "Negative value discourages shield breaks. Reasonable range: [-0.5, 0.0]."
        ),
    )
