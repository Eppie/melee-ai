"""
Learning rate scheduling for training.

This module provides learning rate schedulers and utilities
for training optimization.
"""

import math
from typing import Optional

import torch
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingLRWithWarmup:
    """Cosine annealing learning rate scheduler with warmup."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_steps: int,
        warmup_steps: int = 0,
        base_lr: float = 3e-4,
        final_lr: float = 1e-6
    ):
        """
        Initialize cosine annealing scheduler.

        Args:
            optimizer: Optimizer to schedule
            max_steps: Total number of training steps
            warmup_steps: Number of warmup steps
            base_lr: Base learning rate
            final_lr: Final learning rate
        """
        self.optimizer = optimizer
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.final_lr = final_lr

        self.current_step = 0

    def step(self):
        """Update learning rate for current step."""
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self) -> float:
        """Get current learning rate."""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.base_lr * (self.current_step + 1) / max(1, self.warmup_steps)

        # Cosine annealing
        progress = (self.current_step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        progress = min(progress, 1.0)

        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.final_lr + (self.base_lr - self.final_lr) * cosine_decay


class LRScheduler:
    """Wrapper for learning rate scheduling utilities."""

    @staticmethod
    def create_cosine_scheduler(
        optimizer: torch.optim.Optimizer,
        settings,
        total_steps: Optional[int] = None
    ) -> CosineAnnealingLRWithWarmup:
        """
        Create cosine annealing scheduler.

        Args:
            optimizer: Optimizer to schedule
            settings: Training settings
            total_steps: Total training steps (computed if None)

        Returns:
            Configured scheduler
        """
        if total_steps is None:
            total_steps = settings.total_steps

        return CosineAnnealingLRWithWarmup(
            optimizer=optimizer,
            max_steps=total_steps,
            warmup_steps=settings.training.warmup_steps,
            base_lr=settings.training.lr
        )

    @staticmethod
    def create_step_scheduler(
        optimizer: torch.optim.Optimizer,
        step_size: int,
        gamma: float = 0.1
    ) -> torch.optim.lr_scheduler.StepLR:
        """
        Create step learning rate scheduler.

        Args:
            optimizer: Optimizer to schedule
            step_size: Period of learning rate decay
            gamma: Multiplicative factor of learning rate decay

        Returns:
            StepLR scheduler
        """
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
