"""
Models module for Melee AI.

This module defines neural network architectures and provides
interfaces for model training and inference.
"""

from .base import ModelAdapter, BaseModel
# from melee_ai.gpt import GPTModel  # Import from root until we move gpt.py - temporarily disabled
from .heads import ModelHeads
from .registry import ModelRegistry

__all__ = [
    "ModelAdapter",
    "BaseModel",
    # "GPTModel",  # Temporarily disabled
    "ModelHeads",
    "ModelRegistry",
]
