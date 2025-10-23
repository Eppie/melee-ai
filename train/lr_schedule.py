"""Learning rate schedules for training."""
from __future__ import annotations

import math
from typing import Callable


def cosine_lr_schedule(step: int, total_steps: int, base_lr: float, warmup: int = 0) -> float:
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


def constant_lr(base_lr: float) -> float:
    """Constant learning rate (no schedule)."""
    return base_lr


def get_lr_schedule(schedule_type: str, **kwargs) -> Callable[[int], float]:
    """Factory function for learning rate schedules.
    
    Args:
        schedule_type: One of "cosine", "constant", "warmup"
        **kwargs: Schedule-specific parameters
        
    Returns:
        A function that takes step number and returns learning rate
    """
    if schedule_type == "cosine":
        total_steps = kwargs.get("total_steps", 1000)
        base_lr = kwargs.get("base_lr", 1e-4)
        warmup = kwargs.get("warmup", 0)
        return lambda step: cosine_lr_schedule(step, total_steps, base_lr, warmup)
    
    elif schedule_type == "constant":
        base_lr = kwargs.get("base_lr", 1e-4)
        return lambda step: constant_lr(base_lr)
    
    elif schedule_type == "warmup":
        warmup_steps = kwargs.get("warmup_steps", 100)
        base_lr = kwargs.get("base_lr", 1e-4)
        return lambda step: linear_warmup(step, warmup_steps, base_lr)
    
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")