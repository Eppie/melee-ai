from dataclasses import dataclass, field
from typing import Dict, NamedTuple, Optional, List, Union
import numpy as np
import torch

# Type aliases for clarity
ActionDict = Dict[str, np.ndarray]          # Actions for a batch of envs (numpy)
TensorActionDict = Dict[str, torch.Tensor]  # Actions for a batch of envs (torch)
FeatureArray = np.ndarray                   # [Batch, Features] numpy array
FeatureTensor = torch.Tensor                # [Batch, Features] torch tensor

class InferenceRequest(NamedTuple):
    """
    Data sent from a worker process to the inference server.
    Represents a single step of N parallel environments.
    """
    worker_id: int
    obs: FeatureArray          # Shape: [N, F]
    rewards: np.ndarray        # Shape: [N]
    dones: np.ndarray          # Shape: [N] (bool or int)

class InferenceResponse(NamedTuple):
    """
    Data sent from the inference server back to a worker process.
    Contains actions for both players.
    """
    p1_actions: ActionDict     # Keys: main_stick, c_stick, buttons, shoulder
    p2_actions: ActionDict     # Same keys

@dataclass
class PPOConfig:
    """
    Hyperparameters and configuration for PPO training.
    """
    # System Configuration
    num_workers: int = 12
    envs_per_worker: int = 8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Environment Details
    rollout_length: int = 128   # Number of steps per rollout per environment
    feature_dim: int = 45       # Dimension of the observation space (example default)
    
    # Optimization
    learning_rate: float = 2.5e-4
    gamma: float = 0.99         # Discount factor
    gae_lambda: float = 0.95    # GAE smoothing parameter
    clip_range: float = 0.1     # PPO Clip range
    ent_coef: float = 0.01      # Entropy coefficient
    vf_coef: float = 0.5        # Value function coefficient
    max_grad_norm: float = 0.5  # Gradient clipping
    
    # Training Loop
    batch_size: int = 256       # Mini-batch size for PPO updates
    num_epochs: int = 4         # Number of epochs per rollout
    total_timesteps: int = 10_000_000
    
    # Model Paths
    dolphin_path: Optional[str] = None
    iso_path: Optional[str] = None
    opponent_path: Optional[str] = None  # Path to opponent checkpoint (None = self-play)

@dataclass
class RolloutBatch:
    """
    A processed batch of rollout data, ready for PPO training.
    All tensors should be on the training device.
    Dimensions: [Total_Steps, Total_Envs, ...]
    """
    obs: torch.Tensor
    actions: TensorActionDict
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
