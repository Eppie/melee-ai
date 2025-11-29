from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_settings import SettingsConfigDict


class RLConfig(BaseModel):
    model_config = SettingsConfigDict(validate_assignment=True, extra="forbid")

    gamma: float = Field(
        default=0.995,
        ge=0,
        le=1,
        description=(
            "Discount factor for future rewards in RL. Controls how much the agent values future vs immediate rewards. "
            "Effect: Higher gamma (0.99-0.999) makes the agent more far-sighted, optimizing for long-term reward; "
            "lower gamma (0.9-0.98) makes it more myopic, focusing on immediate reward. "
            "At 60 FPS, gamma=0.995 gives effective horizon of ~200 frames (~3.3 seconds). "
            "Reasonable range: [0.99, 0.999]. Formula: effective_horizon ≈ 1/(1-gamma) frames. "
            "Interacts with: reward magnitudes (higher gamma amplifies cumulative rewards), "
            "gae_lambda in PPO (both affect advantage estimation), episode length (longer episodes need higher gamma)."
        ),
    )
    value_loss_coef: float = Field(
        default=0.05,
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
    reward_damage_dealt: float = Field(
        default=0.08,
        description=(
            "Reward per percent damage dealt to opponent. Encourages aggressive play and combos. "
            "Reasonable range: [0.01, 0.2]. Interacts with: reward_stock_taken, gamma."
        ),
    )
    reward_stock_taken: float = Field(
        default=4.0,
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
