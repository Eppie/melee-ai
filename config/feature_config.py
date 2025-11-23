from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field
from pydantic_settings import SettingsConfigDict


class FeatureConfig(BaseModel):
    """Pydantic version of FeatureConfig."""

    model_config = SettingsConfigDict(validate_assignment=True, extra="forbid")

    transforms: list[Dict[str, Any]] = Field(
        default_factory=lambda: [
            {
                "transform": "stick_palette",
                "features": ["main_stick_x", "main_stick_y"],
                "palette": "fox_main",
            },
            {
                "transform": "stick_palette",
                "features": ["c_stick_x", "c_stick_y"],
                "palette": "c_stick",
            },
            {
                "transform": "scale",
                "features": ["facing"],
                "factor": 2.0,
            },
            {
                "transform": "offset",
                "features": ["facing"],
                "delta": -1.0,
            },
            {
                "transform": "scale",
                "features": ["percent"],
                "factor": 1 / 100.0,
            },
            {
                "transform": "scale",
                "features": ["shield_strength"],
                "factor": 1.0 / 60.0,
            },
            {
                "transform": "scale",
                "features": ["stock"],
                "factor": 1 / 4.0,
            },
            {
                "transform": "scale",
                "features": ["position_x", "position_y"],
                "factor": 1 / 20.0,
            },
            {
                "transform": "scale",
                "features": ["jumps_left"],
                "factor": 1 / 6.0,
            },
        ]
    )
