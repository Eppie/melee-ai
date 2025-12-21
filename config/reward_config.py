from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_settings import SettingsConfigDict


class RewardConfig(BaseModel):
    """Shared reward weights and discounting used by imitation."""

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
        default=0.0001,
        description=(
            "Reward per frame opponent is in defensive hitlag (got hit). "
            "Encourages landing hits. Since damage is /100 scaled, this is much smaller. "
            "Reasonable range: [0.00005, 0.0005]."
        ),
    )
    reward_low_shield: float = Field(
        default=-0.1,
        description=(
            "Penalty per frame when shield is low (encourages shield management). "
            "Negative value discourages shield breaks. Reasonable range: [-0.5, 0.0]."
        ),
    )
    reward_hitstun_min_frames: int = Field(
        default=15,
        ge=0,
        description=(
            "Minimum consecutive hitstun frames before reward starts. "
            "Filters out brief hits/pokes. Reasonable range: [0, 30]."
        ),
    )
    reward_hitstun_peak_frames: int = Field(
        default=400,
        ge=1,
        description=(
            "Consecutive hitstun frames where per-frame reward reaches maximum. "
            "Encourages extended combos. Should be > min_frames. Reasonable range: [200, 600]."
        ),
    )
    reward_hitstun_max_frames: int = Field(
        default=600,
        ge=1,
        description=(
            "Consecutive hitstun frames where per-frame reward returns to zero. "
            "Prevents infinite reward accumulation. Should be > peak_frames. Reasonable range: [400, 900]."
        ),
    )
    reward_hitstun_peak_value: float = Field(
        default=0.04,
        description=(
            "Maximum per-frame reward at peak hitstun length. "
            "Should be smaller than damage reward (typically ~0.5x damage reward). "
            "Reasonable range: [0.01, 0.1]."
        ),
    )
