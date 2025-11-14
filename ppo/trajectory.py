"""Trajectory collection and storage for PPO training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch


@dataclass
class Step:
    """Single step in a trajectory."""

    state: torch.Tensor  # [F] feature tensor
    action_logits: Dict[str, torch.Tensor]  # raw logits from model for each action head
    action_taken: Dict[
        str, torch.Tensor
    ]  # actual actions taken (indices for sticks/shoulders, bool for buttons)
    log_prob: torch.Tensor  # [1] log probability of the action taken
    value: torch.Tensor  # [1] value estimate from critic
    reward: float = 0.0  # immediate reward (assigned post-episode)
    done: bool = False  # episode termination flag


@dataclass
class Trajectory:
    """Complete trajectory (episode) for PPO training."""

    steps: List[Step]

    # Computed after episode ends
    returns: Optional[torch.Tensor] = None  # [T] discounted returns
    advantages: Optional[torch.Tensor] = None  # [T] GAE advantages

    def __len__(self) -> int:
        return len(self.steps)

    def compute_gae(
        self,
        gamma: float = 0.995,
        gae_lambda: float = 0.95,
        normalize: bool = True,
    ) -> None:
        """Compute Generalized Advantage Estimation (GAE) for this trajectory.

        Args:
            gamma: Discount factor for rewards
            gae_lambda: Lambda for GAE (bias-variance tradeoff)
            normalize: Whether to normalize advantages (mean=0, std=1)
        """
        T = len(self.steps)
        if T == 0:
            self.advantages = torch.tensor([])
            self.returns = torch.tensor([])
            return

        # Extract values and rewards
        values = torch.stack([step.value for step in self.steps]).squeeze(-1)  # [T]

        if self.returns is not None and len(self.returns) == T:
            returns = self.returns.to(values.dtype)
            advantages = returns - values
            if normalize and T > 1:
                adv_mean = advantages.mean()
                adv_std = advantages.std(unbiased=False)
                if adv_std > 1e-8:
                    advantages = (advantages - adv_mean) / (adv_std + 1e-8)
                else:
                    advantages = advantages - adv_mean
            self.advantages = advantages
            self.returns = returns
            return

        rewards = torch.tensor(
            [step.reward for step in self.steps], dtype=values.dtype
        )  # [T]

        # Compute TD errors: delta(t) = r(t) + gamma * V(t+1) - V(t)
        # For the last step, V(t+1) = 0 (terminal state)
        next_values = torch.cat([values[1:], torch.zeros(1, dtype=values.dtype)])
        deltas = rewards + gamma * next_values - values  # [T]

        # Compute GAE: A(t) = sum_{l=0}^{inf} (gamma * lambda)^l * delta(t+l)
        # This is a discounted cumulative sum backward in time
        advantages = torch.zeros(T, dtype=values.dtype)
        gae = 0.0
        for t in reversed(range(T)):
            gae = deltas[t] + gamma * gae_lambda * gae
            advantages[t] = gae

        # Normalize advantages if requested
        if normalize and T > 1:
            adv_mean = advantages.mean()
            adv_std = advantages.std()
            if adv_std > 1e-8:  # Only normalize if there's variance
                advantages = (advantages - adv_mean) / (adv_std + 1e-8)
            else:
                # If all advantages are the same, just center them
                advantages = advantages - adv_mean

        self.advantages = advantages

        # Compute returns: R(t) = A(t) + V(t)
        # This is used as the target for the value function
        self.returns = advantages + values

    def to_tensors(self, device: torch.device) -> Dict[str, torch.Tensor]:
        """Convert trajectory to tensors for training.

        Returns:
            Dictionary containing:
                - states: [T, F] stacked states
                - action_logits_*: [T, ...] logits for each action head
                - actions_*: [T, ...] actions taken for each head
                - old_log_probs: [T] log probabilities
                - values: [T] value estimates
                - returns: [T] target returns
                - advantages: [T] GAE advantages
        """
        if self.advantages is None or self.returns is None:
            raise ValueError("Must call compute_gae() before converting to tensors")

        T = len(self.steps)
        if T == 0:
            raise ValueError("Cannot convert empty trajectory to tensors")

        # Stack states
        states = torch.stack([step.state for step in self.steps]).to(device)  # [T, F]

        # Stack action logits and actions for each head
        result = {"states": states}

        # Assuming action heads: main_stick, c_stick, buttons, shoulder
        action_heads = ["main_stick", "c_stick", "buttons", "shoulder"]
        for head in action_heads:
            if head in self.steps[0].action_logits:
                result[f"action_logits_{head}"] = torch.stack(
                    [step.action_logits[head] for step in self.steps]
                ).to(device)
                result[f"actions_{head}"] = torch.stack(
                    [step.action_taken[head] for step in self.steps]
                ).to(device)

        # Stack old log probs, values, returns, advantages
        result["old_log_probs"] = (
            torch.stack([step.log_prob for step in self.steps]).squeeze(-1).to(device)
        )  # [T]
        result["values"] = (
            torch.stack([step.value for step in self.steps]).squeeze(-1).to(device)
        )  # [T]
        result["returns"] = self.returns.to(device)  # [T]
        result["advantages"] = self.advantages.to(device)  # [T]

        return result


class TrajectoryBuffer:
    """Buffer for collecting trajectories during self-play."""

    def __init__(self):
        self.current_trajectory: List[Step] = []
        self.completed_trajectories: List[Trajectory] = []

    def add_step(
        self,
        state: torch.Tensor,
        action_logits: Dict[str, torch.Tensor],
        action_taken: Dict[str, torch.Tensor],
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: float,
        done: bool,
    ) -> None:
        """Add a step to the current trajectory."""
        step = Step(
            state=state.cpu(),
            action_logits={k: v.cpu() for k, v in action_logits.items()},
            action_taken={k: v.cpu() for k, v in action_taken.items()},
            log_prob=log_prob.cpu(),
            value=value.cpu(),
            reward=reward,
            done=done,
        )
        self.current_trajectory.append(step)

    def finish_trajectory(self, returns: Optional[torch.Tensor] = None) -> None:
        """Finish the current trajectory and add to completed list."""
        if len(self.current_trajectory) > 0:
            trajectory = Trajectory(steps=self.current_trajectory)
            if returns is not None:
                returns = returns.detach().clone().reshape(-1)
                if returns.numel() != len(self.current_trajectory):
                    raise ValueError(
                        "Returns length does not match trajectory length"
                    )
                trajectory.returns = returns
            self.completed_trajectories.append(trajectory)
            self.current_trajectory = []

    def compute_all_gae(
        self,
        gamma: float = 0.995,
        gae_lambda: float = 0.95,
        normalize: bool = True,
    ) -> None:
        """Compute GAE for all completed trajectories."""
        for trajectory in self.completed_trajectories:
            trajectory.compute_gae(gamma, gae_lambda, normalize)

    def get_trajectories(self) -> List[Trajectory]:
        """Get all completed trajectories."""
        return self.completed_trajectories

    def clear(self) -> None:
        """Clear all trajectories."""
        self.current_trajectory = []
        self.completed_trajectories = []

    def discard_current(self) -> None:
        """Drop the in-progress trajectory without saving it."""
        self.current_trajectory = []

    def __len__(self) -> int:
        return len(self.completed_trajectories)
