"""Learning rate schedules for training."""

from __future__ import annotations

import math

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from train.components import TrainingComponents


def cosine_lr_schedule(
    step: int, total_steps: int, base_lr: float, warmup: int = 0
) -> float:
    """Compute the cosine-annealed learning rate for a specific training step.

    Example:
        With ``base_lr=0.001``, ``total_steps=100``, and ``warmup=10``:

        #. For ``step=4`` (during warmup), the function returns
           ``0.001 * (4 + 1) / 10 = 0.0005``.
        #. For ``step=10`` (first step after warmup), ``progress = (10 - 10)/(100 - 10) = 0`` and the
           cosine term equals ``1.0``, so the learning rate is ``0.001``.
        #. For ``step=55``, ``progress ≈ 0.5`` and the cosine becomes ``cos(pi * 0.5) = 0`` yielding
           ``0.0005``.

    Args:
        step: Zero-indexed training step.
        total_steps: Total number of steps in the schedule.
        base_lr: Base learning rate amplitude.
        warmup: Number of warmup steps with linear ramp.

    Returns:
        Floating-point learning rate for the requested step.
    """
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def _update_learning_rate(components: "TrainingComponents", global_step: int) -> float:
    config = components.config
    lr = cosine_lr_schedule(
        global_step,
        components.total_steps,
        config.train.lr,
        config.train.warmup_steps,
    )
    for pg in components.optimizer.param_groups:
        pg["lr"] = lr
    return lr
