"""
Training module for Melee AI.

This module handles model training orchestration, metrics computation,
checkpointing, and training lifecycle management.
"""

from .trainer import Trainer
from .metrics import MetricAggregator, TrainingMetrics
from .checkpoint import CheckpointManager
from .scheduler import LRScheduler
from .loop import TrainingLoop

__all__ = [
    "Trainer",
    "TrainingLoop",
    "MetricAggregator",
    "TrainingMetrics",
    "CheckpointManager",
    "LRScheduler",
]
