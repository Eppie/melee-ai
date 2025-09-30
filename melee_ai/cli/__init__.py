"""
CLI module for Melee AI.

This module provides command-line interfaces for common operations
like training, preprocessing, and evaluation.
"""

from .train import train_command
from .preprocess import preprocess_command
from .evaluate import evaluate_command

__all__ = [
    "train_command",
    "preprocess_command",
    "evaluate_command",
]
