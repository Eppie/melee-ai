"""PPO (Proximal Policy Optimization) self-play training module.

This module provides two training modes:

1. Episode-based training (original):
   - SelfPlayEnvironment manages Dolphin + models
   - TrajectoryBuffer collects per-episode experience
   - Training happens after each episode

2. Distributed training (new):
   - InferenceCoordinator: Centralized GPU inference for all workers
   - SimulationWorker: Stateless emulator processes
   - TrajectorySlicer: Fixed-length rollouts with bootstrapping

Use --distributed flag with train_ppo.py to enable distributed mode.
"""

from ppo.opponent_pool import OpponentPool
from ppo.trajectory import Trajectory, TrajectoryBuffer, Step
from ppo.ppo_loss import compute_total_ppo_loss

__all__ = [
    "OpponentPool",
    "Trajectory",
    "TrajectoryBuffer",
    "Step",
    "compute_total_ppo_loss",
]
