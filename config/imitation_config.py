from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import SettingsConfigDict


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
