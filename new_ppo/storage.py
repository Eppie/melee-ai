import torch
from typing import Generator, Dict
from new_ppo.ppo_types import RolloutBatch
from new_ppo.util import compute_gae

class RolloutBuffer:
    """
    Stores transitions for PPO training.
    Structure: [Time, Batch, Features]
    """
    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        feature_dim: int,
        device: str = "cpu"
    ):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device
        
        # Storage Tensors
        self.obs = torch.zeros((num_steps, num_envs, feature_dim), device=device)
        self.rewards = torch.zeros((num_steps, num_envs), device=device)
        self.dones = torch.zeros((num_steps, num_envs), device=device)
        self.values = torch.zeros((num_steps, num_envs), device=device)
        self.log_probs = torch.zeros((num_steps, num_envs), device=device)
        
        # Actions are a dict of tensors, so we store them as a dict of list of tensors (or pre-allocate)
        # Since keys are fixed, we can pre-allocate.
        self.actions: Dict[str, torch.Tensor] = {
            'main_stick': torch.zeros((num_steps, num_envs), dtype=torch.long, device=device),
            'c_stick': torch.zeros((num_steps, num_envs), dtype=torch.long, device=device),
            'buttons': torch.zeros((num_steps, num_envs, 5), dtype=torch.bool, device=device),
            'shoulder': torch.zeros((num_steps, num_envs), dtype=torch.long, device=device),
        }
        
        # Computed later
        self.advantages = torch.zeros((num_steps, num_envs), device=device)
        self.returns = torch.zeros((num_steps, num_envs), device=device)
        
        self.step = 0
        
    def insert(
        self,
        obs: torch.Tensor,
        actions: Dict[str, torch.Tensor],
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        log_probs: torch.Tensor
    ):
        """Insert a single step of data (from N envs)."""
        self.obs[self.step] = obs
        self.rewards[self.step] = rewards
        self.dones[self.step] = dones
        self.values[self.step] = values
        self.log_probs[self.step] = log_probs
        
        for k, v in actions.items():
            self.actions[k][self.step] = v
            
        self.step += 1

    def compute_returns_and_finish(self, next_value: torch.Tensor, gamma: float, gae_lambda: float):
        """
        Compute GAE and Returns.
        Resets the buffer step pointer for the next round, but does NOT clear data 
        (data is overwritten in next insert).
        """
        with torch.no_grad():
            self.advantages = compute_gae(
                self.rewards,
                self.values,
                self.dones,
                next_value,
                gamma,
                gae_lambda
            )
            self.returns = self.advantages + self.values
            
        self.step = 0 # Reset for next accumulation

    def get_batch(self) -> RolloutBatch:
        """Returns the flattened batch for training."""
        # Flatten time and batch dimensions [T*N, ...]
        b_obs = self.obs.flatten(0, 1)
        b_actions = {k: v.flatten(0, 1) for k, v in self.actions.items()}
        b_log_probs = self.log_probs.flatten(0, 1)
        b_values = self.values.flatten(0, 1)
        b_advantages = self.advantages.flatten(0, 1)
        b_returns = self.returns.flatten(0, 1)
        
        # Note: rewards/dones usually not needed for PPO update step itself, 
        # but kept if needed for metrics.
        b_rewards = self.rewards.flatten(0, 1)
        b_dones = self.dones.flatten(0, 1)

        return RolloutBatch(
            obs=b_obs,
            actions=b_actions,
            rewards=b_rewards,
            dones=b_dones,
            values=b_values,
            log_probs=b_log_probs,
            advantages=b_advantages,
            returns=b_returns
        )
