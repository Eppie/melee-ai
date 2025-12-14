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

    use_focal_loss: bool = Field(
        default=True,
        description=(
            "Use focal loss for controller heads (main_stick, c_stick, shoulder, buttons). "
            "Focal loss down-weights easy examples and focuses training on hard examples."
        ),
    )

    focal_gamma: float = Field(
        default=1.0,
        description=(
            "Focusing parameter for focal loss. Higher values increase focus on hard examples. "
            "gamma=0.0 reduces to standard cross-entropy. Typical values: 1.0-2.0."
        ),
    )

    focal_alpha: float | None = Field(
        default=0.25,
        description=(
            "Class balancing weight for focal loss. "
            "Lower values (0.25) give more weight to hard/rare examples. "
            "Set to None to disable class balancing. Typical values: 0.25-0.75."
        ),
    )

    focal_loss_scale: float = Field(
        default=10.0,
        description=(
            "Multiplicative scaling factor for focal loss. "
            "Use this to adjust the overall magnitude of focal loss to match gradient scales "
            "from standard cross-entropy. For example, if focal loss is 10x smaller, "
            "set this to 10.0 to restore original gradient magnitudes."
        ),
    )
