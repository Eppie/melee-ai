"""PPO (Proximal Policy Optimization) self-play training module.

Distributed training architecture:
- InferenceCoordinator: Centralized GPU inference for all workers
- SimulationWorker: Stateless emulator processes
- TrajectorySlicer: Fixed-length rollouts with bootstrapping
- OpponentPool: FIFO pool of frozen past models
"""

from ppo.opponent_pool import OpponentPool
from ppo.trajectory import Trajectory, Step
from ppo.ppo_loss import compute_total_ppo_loss

__all__ = [
    "OpponentPool",
    "Trajectory",
    "Step",
    "compute_total_ppo_loss",
]
