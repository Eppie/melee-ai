"""Learning rate schedules for training."""

from __future__ import annotations

import math


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

        The example demonstrates the exact calculations from warmup scaling through cosine decay.

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


def linear_warmup(step: int, warmup_steps: int, base_lr: float) -> float:
    """Linearly ramp the learning rate from zero to ``base_lr`` over ``warmup_steps`` steps.

    Example:
        When ``warmup_steps=5`` and ``base_lr=0.01``:

        * ``step=0`` returns ``0.01 * (0 + 1) / 5 = 0.002``.
        * ``step=4`` returns ``0.01 * 5 / 5 = 0.01`` (end of warmup).
        * ``step=6`` exceeds the warmup window so the function clamps to ``base_lr=0.01``.

        The example walks through each branch showing how the scaling factor is derived.

    Args:
        step: Zero-indexed training step.
        warmup_steps: Number of warmup steps; if ``0`` the base learning rate is returned.
        base_lr: Target learning rate once warmup is complete.

    Returns:
        Warmed learning rate for the provided step.
    """
    if warmup_steps <= 0:
        return base_lr
    return base_lr * min(1.0, (step + 1) / warmup_steps)
