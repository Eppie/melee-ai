from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_settings import SettingsConfigDict


class LossConfig(BaseModel):
    """Configuration related to loss computation and weighting."""

    model_config = SettingsConfigDict(validate_assignment=True, extra="forbid")

    enable_class_balancing: bool = Field(default=True)
    ce_weight_min: float = Field(default=0.1, gt=0)
    ce_weight_max: float = Field(default=10.0, gt=0)
    enable_pos_weighting: bool = Field(default=True)
    pos_weight_max: float = Field(default=3.0, gt=0)
    use_weighted_component_means: bool = Field(default=True)
    main_change: float = Field(default=5.0, gt=0)
    c_change: float = Field(default=10.0, gt=0)
    shoulder_change: float = Field(default=5.0, gt=0)
    buttons_change_default: float = Field(default=10.0, gt=0)
    button_z: float = Field(default=20.0, gt=0)
    button_b: float = Field(default=12.0, gt=0)
    button_a: float = Field(default=12.0, gt=0)
    button_xy: float = Field(default=10.0, gt=0)
    button_lr: float = Field(default=8.0, gt=0)
    hold_base: float = Field(default=1.0, gt=0)
    value_change: float = Field(default=1.0, gt=0)
