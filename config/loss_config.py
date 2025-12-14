from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_settings import SettingsConfigDict


class LossConfig(BaseModel):
    """Configuration related to loss computation and weighting."""

    model_config = SettingsConfigDict(validate_assignment=True, extra="forbid")

    use_weighted_component_means: bool = Field(
        default=True,
        description=(
            "Use weighted mean (by sample weights) when averaging loss components. "
            "When True, samples with higher weights (e.g., imitation/value weights) contribute more to final loss."
        ),
    )
