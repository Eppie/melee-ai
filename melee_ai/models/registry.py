"""
Model registry for Melee AI.

This module provides a registry for different model implementations
and factory functions for model creation.
"""

from typing import Dict, Type

from melee_ai.config import Settings
from .base import BaseModel


class ModelRegistry:
    """Registry for model implementations."""

    _models: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str, model_class: Type[BaseModel]):
        """Register a model class."""
        cls._models[name] = model_class

    @classmethod
    def get_model(cls, name: str, settings: Settings) -> BaseModel:
        """Get model instance by name."""
        if name not in cls._models:
            raise ValueError(f"Unknown model: {name}")

        return cls._models[name](settings)

    @classmethod
    def list_models(cls) -> list:
        """List available model names."""
        return list(cls._models.keys())

    @classmethod
    def create_model_config(cls, settings: Settings) -> Dict:
        """Create model configuration from settings."""
        # This would create the appropriate config object
        # For now, return empty dict
        return {}
