from __future__ import annotations

from pydantic import BaseModel
from pydantic_settings import SettingsConfigDict


class FeatureConfig(BaseModel):
    """Feature configuration - transforms are now hardcoded in feature_transforms.py."""

    model_config = SettingsConfigDict(validate_assignment=True, extra="forbid")
