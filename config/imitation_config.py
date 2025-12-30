from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import SettingsConfigDict


class ImitationConfig(BaseModel):
    model_config = SettingsConfigDict(validate_assignment=True, extra="forbid")

    strategy: Literal[
        "uniform", "value_weighted", "value_advantage", "value_filter", "hybrid"
    ] = Field(
        default="hybrid",
        description=(
            "Imitation learning sampling strategy. Controls how training samples are weighted/selected. "
            "Options: 'uniform' = all frames equally likely (baseline); "
            "'value_weighted' = weight by value estimate (learn from high-value states); "
            "'value_advantage' = weight by MODEL-DEPENDENT advantage (learn from frames where expert beat model's expectations); "
            "'value_filter' = filter to top percentile by value (only learn from best states); "
            "'hybrid' = combine multiple strategies (see hybrid_strategies/hybrid_weights). "
            "Default 'hybrid' combines value_weighted and value_filter for balanced learning."
        ),
    )
    value_k: float = Field(
        default=1.4,
        gt=0,
        description=(
            "Scaling factor for value-based weighting. Controls strength of value-based prioritization. "
            "Effect: Higher k (2.0-5.0) = stronger emphasis on high-value states; "
            "lower k (0.5-1.0) = more uniform weighting. Reasonable range: [0.5, 5.0]. "
            "Used in: value_weighted, value_advantage strategies. Interacts with: value_temperature. "
            "Default 1.0 provides gentle emphasis while avoiding extreme outliers."
        ),
    )
    value_temperature: float = Field(
        default=0.8,
        gt=0,
        description=(
            "Temperature for value-based sampling. Controls sharpness of value distribution. "
            "Effect: Lower temperature (0.5-1.0) = sharper distribution, focus on best samples; "
            "higher temperature (1.0-2.0) = smoother distribution, more exploration. "
            "Reasonable range: [0.5, 2.0]. Interacts with: value_k, value_use_exp."
        ),
    )
    value_use_exp: bool = Field(
        default=True,
        description=(
            "Use exponential (softmax) weighting instead of linear for value-based sampling. "
            "Effect: True = sharper focus on high-value states (exponential emphasis); "
            "False = gentler weighting (linear scaling). Default True for superhuman play."
        ),
    )
    advantage_alpha: float = Field(
        default=20.0,
        gt=0,
        description=(
            "Scaling factor for model-dependent advantage weighting. Controls how much to emphasize frames "
            "where expert beat model's predictions. Higher alpha = stronger focus on surprising expert decisions. "
            "Effect: Higher alpha (20-50) = aggressive focus on frames where expert exceeded model expectations; "
            "lower alpha (5-15) = gentler emphasis, more uniform learning. Reasonable range: [5.0, 50.0]. "
            "Since value prediction errors are typically small (0.05-0.3), moderate-to-high alpha needed. "
            "As your model improves, effective_batch_fraction will naturally decrease as fewer frames surprise the model."
        ),
    )
    filter_percentile: float = Field(
        default=15.0,
        ge=0,
        le=100,
        description=(
            "Percentile cutoff for value_filter strategy. Only train on top X% of samples by value. "
            "Effect: Lower percentile (20-40) = train only on best samples (superhuman focus); "
            "higher percentile (60-80) = more diverse samples. Reasonable range: [20, 80]. "
            "Used in: value_filter strategy. Default 15.0 = filter only worst 15% (preserves training signal)."
        ),
    )
    filter_soft: bool = Field(
        default=True,
        description=(
            "Use soft (weighted) filtering instead of hard cutoff for value_filter strategy. "
            "Effect: True = smooth transition around percentile cutoff; "
            "False = sharp cutoff (samples below percentile get 0 weight). "
            "Recommended: True for smoother learning and avoiding abrupt weight transitions. "
            "Interacts with: filter_temperature (controls softness)."
        ),
    )
    filter_temperature: float = Field(
        default=0.5,
        gt=0,
        description=(
            "Temperature for soft filtering. Only used if filter_soft=True. "
            "Effect: Lower temperature (0.5-1.0) = sharper transition; "
            "higher temperature (1.0-2.0) = smoother transition. Reasonable range: [0.5, 2.0]. "
            "Default 0.5 provides sharp but smooth sigmoid transition."
        ),
    )
    hybrid_strategies: list[str] = Field(
        default_factory=lambda: ["value_weighted", "value_filter"],
        description=(
            "List of strategies to combine in hybrid mode. Must match hybrid_weights length. "
            "Example: ['value_weighted', 'value_filter'] combines value-based weighting with filtering. "
            "Recommended combinations: ['value_weighted', 'value_filter'] (quality + filtering), "
            "['value_weighted', 'value_advantage'] (value + surprise)."
        ),
    )
    hybrid_weights: list[float] = Field(
        default_factory=lambda: [0.5, 0.5],
        description=(
            "Weights for combining hybrid strategies. Must sum to 1.0 and match hybrid_strategies length. "
            "Example: [0.7, 0.3] = 70% first strategy, 30% second strategy. "
            "Interacts with: hybrid_strategies (defines what to combine)."
        ),
    )

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
