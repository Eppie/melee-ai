"""
Base model interfaces and protocols.

This module defines the core interfaces that all models must implement,
enabling dependency injection and consistent model behavior.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Protocol, Tuple

import torch
import torch.nn as nn

from melee_ai.config import Settings


class ModelAdapter(Protocol):
    """Protocol for model adapters that handle training/inference."""

    def __call__(self, inputs) -> Dict[str, torch.Tensor]:
        """Forward pass through model."""
        ...

    def compute_loss(self, outputs: Dict[str, torch.Tensor], targets) -> torch.Tensor:
        """Compute loss for training."""
        ...

    def parameters(self):
        """Get model parameters for optimization."""
        ...

    def state_dict(self) -> Dict[str, Any]:
        """Get model state dictionary."""
        ...

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load model state dictionary."""
        ...

    def to(self, device) -> 'ModelAdapter':
        """Move model to device."""
        ...

    def train(self, mode: bool = True) -> 'ModelAdapter':
        """Set training mode."""
        ...

    def eval(self) -> 'ModelAdapter':
        """Set evaluation mode."""
        ...


class BaseModel(nn.Module, ModelAdapter):
    """Base class for all models implementing the ModelAdapter protocol."""

    def __init__(self, settings: Settings):
        """
        Initialize base model.

        Args:
            settings: Configuration settings
        """
        super().__init__()
        self.settings = settings

    def _init_weights(self, module) -> None:
        """Initialize weights for the model."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @abstractmethod
    def forward(self, inputs) -> Dict[str, torch.Tensor]:
        """Forward pass implementation."""
        pass

    def compute_loss(self, outputs: Dict[str, torch.Tensor], targets) -> torch.Tensor:
        """
        Default loss computation. Subclasses should override for custom loss.

        Args:
            outputs: Model outputs
            targets: Ground truth targets

        Returns:
            Computed loss tensor
        """
        # Default implementation - subclasses should override
        return torch.tensor(0.0, requires_grad=True)

    def get_input_signature(self) -> Dict[str, Tuple[int, ...]]:
        """
        Get expected input shapes for the model.

        Returns:
            Dictionary mapping input names to expected shapes
        """
        return {}

    def get_output_signature(self) -> Dict[str, Tuple[int, ...]]:
        """
        Get expected output shapes for the model.

        Returns:
            Dictionary mapping output names to expected shapes
        """
        return {}


class ModelRegistry:
    """Registry for model implementations."""

    _models = {}

    @classmethod
    def register(cls, name: str, model_class):
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
