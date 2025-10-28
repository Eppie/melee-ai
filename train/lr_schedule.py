"""Learning rate schedules for training."""

from __future__ import annotations

import math


def cosine_lr_schedule(
        step: int, total_steps: int, base_lr: float, warmup: int = 0
) -> float:
    """Cosine annealing learning rate schedule with optional warmup."""
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def linear_warmup(step: int, warmup_steps: int, base_lr: float) -> float:
    """Linear warmup learning rate schedule."""
    if warmup_steps <= 0:
        return base_lr
    return base_lr * min(1.0, (step + 1) / warmup_steps)
