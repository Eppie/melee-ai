from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_settings import SettingsConfigDict


class RLConfig(BaseModel):
    model_config = SettingsConfigDict(validate_assignment=True, extra="forbid")
    # TODO: Document what effect this has
    gamma: float = Field(default=0.995, ge=0, le=1)
    # TODO: Document what effect this has
    value_loss_coef: float = Field(default=0.5, ge=0)
    reward_damage_dealt: float = 0.08
    reward_stock_taken: float = 4
    reward_hitlag_opponent: float = 0.08
    reward_low_shield: float = -0.1
