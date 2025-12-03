"""Rollout buffer management and GAE computation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from numpy.lib.stride_tricks import as_strided


@dataclass
class RolloutBuffer:
    """
    Stores one complete rollout for a single environment.

    Rollout length is configurable (default 1024 frames).
    Includes mask for warmup frames (excluded from training).
    """

    rollout_length: int
    feature_dim: int = 908

    def __post_init__(self):
        """Allocate arrays."""
        # Features: [rollout_length, feature_dim] float32
        self.X = np.zeros((self.rollout_length, self.feature_dim), dtype=np.float32)

        # Actions: [rollout_length] structured array
        from .shared_memory import ActionData_dtype

        self.actions = np.zeros(self.rollout_length, dtype=ActionData_dtype)

        # Scalar arrays: [rollout_length]
        self.logp = np.zeros(self.rollout_length, dtype=np.float32)
        self.values = np.zeros(self.rollout_length, dtype=np.float32)
        self.rewards = np.zeros(self.rollout_length, dtype=np.float32)
        self.mask = np.zeros(self.rollout_length, dtype=bool)

        # Computed after rollout completes
        self.advantages: Optional[np.ndarray] = None  # [rollout_length] float32
        self.returns: Optional[np.ndarray] = None  # [rollout_length] float32

        # Tracking
        self.pos = 0  # Current position in buffer
        self.complete = False

    def append(
        self,
        features: np.ndarray,
        action: np.ndarray,
        logp: float,
        value: float,
        reward: float,
        mask: bool,
    ):
        """
        Append a frame to the rollout buffer.

        Args:
            features: [feature_dim] feature vector
            action: Structured numpy array (ActionData_dtype)
            logp: Log probability of action
            value: Value estimate V(s)
            reward: Immediate reward
            mask: True if valid for training, False if warmup
        """
        if self.pos >= self.rollout_length:
            raise ValueError(f"Rollout buffer full (pos={self.pos})")

        self.X[self.pos] = features
        self.actions[self.pos] = action
        self.logp[self.pos] = logp
        self.values[self.pos] = value
        self.rewards[self.pos] = reward
        self.mask[self.pos] = mask

        self.pos += 1

        if self.pos >= self.rollout_length:
            self.complete = True

    def reset(self):
        """Reset buffer for new rollout."""
        self.pos = 0
        self.complete = False
        self.advantages = None
        self.returns = None
        # Note: Arrays are reused without re-allocation

    def compute_advantages(
        self,
        gamma: float = 0.995,
        gae_lambda: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation (GAE) with warmup masking.

        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda parameter

        Returns:
            (advantages, returns): Both [rollout_length] arrays
        """
        if not self.complete:
            raise ValueError("Cannot compute advantages for incomplete rollout")

        advantages = np.zeros(self.rollout_length, dtype=np.float32)
        gae = 0.0
        next_value = 0.0

        # Backward pass to compute GAE
        for t in reversed(range(self.rollout_length)):
            if not self.mask[t]:
                # Skip warmup frames (don't propagate TD error)
                continue

            delta = self.rewards[t] + gamma * next_value - self.values[t]
            gae = delta + gamma * gae_lambda * gae
            advantages[t] = gae
            next_value = self.values[t]

        # Compute returns: advantages + values
        returns = advantages + self.values

        # Normalize advantages (only over valid frames)
        if self.mask.sum() > 0:
            valid_advantages = advantages[self.mask]
            adv_mean = valid_advantages.mean()
            adv_std = valid_advantages.std()
            advantages[self.mask] = (valid_advantages - adv_mean) / (adv_std + 1e-8)

        self.advantages = advantages
        self.returns = returns

        return advantages, returns


@dataclass
class PPOWindow:
    """
    Single window of data for PPO training.

    A window is a contiguous slice of [context_length] frames
    from a rollout, used as a single training example.
    """

    features: np.ndarray  # [context_length, feature_dim]
    actions: np.ndarray  # [context_length] structured
    old_logp: float  # Log prob at final frame
    advantage: float  # Advantage at final frame
    return_: float  # Return at final frame
    mask: np.ndarray  # [context_length] bool


def create_windowed_batches(
    rollouts: List[RolloutBuffer],
    context_length: int = 256,
    batch_size: int = 128,
) -> List[dict]:
    """
    Create overlapping sliding windows from rollout buffers.

    Each rollout of length L produces (L - context_length + 1) windows
    via stride-1 sliding. Uses as_strided for zero-copy windowing.

    Args:
        rollouts: List of completed rollout buffers
        context_length: Context window size (default 256)
        batch_size: Number of windows per batch

    Returns:
        List of batched dicts ready for training
    """
    all_windows = []

    for rollout in rollouts:
        if not rollout.complete:
            raise ValueError("Cannot create windows from incomplete rollout")
        if rollout.advantages is None:
            raise ValueError("Must compute advantages before creating windows")

        # Compute number of windows
        num_windows = rollout.rollout_length - context_length + 1

        # Zero-copy windowing via as_strided
        # Features: [rollout_length, feature_dim] → [num_windows, context_length, feature_dim]
        X_windows = as_strided(
            rollout.X,
            shape=(num_windows, context_length, rollout.feature_dim),
            strides=(rollout.X.strides[0], rollout.X.strides[0], rollout.X.strides[1]),
        )

        # Actions: [rollout_length] → [num_windows, context_length]
        action_windows = as_strided(
            rollout.actions,
            shape=(num_windows, context_length),
            strides=(rollout.actions.strides[0], rollout.actions.strides[0]),
        )

        # Mask: [rollout_length] → [num_windows, context_length]
        mask_windows = as_strided(
            rollout.mask,
            shape=(num_windows, context_length),
            strides=(rollout.mask.strides[0], rollout.mask.strides[0]),
        )

        # Only include windows where all frames are valid (mask=True)
        for i in range(num_windows):
            window_mask = mask_windows[i]
            if not window_mask.all():
                # Skip windows with warmup frames
                continue

            # For PPO, we predict/train on the final frame of each window
            final_idx = i + context_length - 1

            window = PPOWindow(
                features=X_windows[i].copy(),  # Copy to avoid shared memory issues
                actions=action_windows[i].copy(),
                old_logp=rollout.logp[final_idx],
                advantage=rollout.advantages[final_idx],
                return_=rollout.returns[final_idx],
                mask=window_mask.copy(),
            )
            all_windows.append(window)

    # Shuffle windows
    random.shuffle(all_windows)

    # Batch windows
    batches = []
    for i in range(0, len(all_windows), batch_size):
        batch_windows = all_windows[i : i + batch_size]

        # Stack into tensors
        batch = {
            "features": torch.from_numpy(
                np.stack([w.features for w in batch_windows])
            ),  # [B, T, F]
            "actions": np.stack(
                [w.actions for w in batch_windows]
            ),  # [B, T] structured
            "old_logp": torch.tensor(
                [w.old_logp for w in batch_windows], dtype=torch.float32
            ),  # [B]
            "advantages": torch.tensor(
                [w.advantage for w in batch_windows], dtype=torch.float32
            ),  # [B]
            "returns": torch.tensor(
                [w.return_ for w in batch_windows], dtype=torch.float32
            ),  # [B]
            "mask": torch.from_numpy(
                np.stack([w.mask for w in batch_windows])
            ),  # [B, T]
        }
        batches.append(batch)

    return batches


def compute_gae_batch(
    rewards: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute GAE for a batch of rollouts (GPU-accelerated).

    Args:
        rewards: [B, T] reward tensor
        values: [B, T] value estimates
        mask: [B, T] bool mask (False for warmup)
        gamma: Discount factor
        gae_lambda: GAE lambda

    Returns:
        (advantages, returns): Both [B, T] tensors
    """
    B, T = rewards.shape
    device = rewards.device

    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(B, device=device)
    next_value = torch.zeros(B, device=device)

    # Backward pass
    for t in reversed(range(T)):
        # Only update where mask is True
        valid = mask[:, t]

        delta = rewards[:, t] + gamma * next_value - values[:, t]
        gae = torch.where(valid, delta + gamma * gae_lambda * gae, gae)
        advantages[:, t] = gae
        next_value = torch.where(valid, values[:, t], next_value)

    # Compute returns
    returns = advantages + values

    # Normalize advantages (per rollout, only over valid frames)
    for b in range(B):
        valid_mask = mask[b]
        if valid_mask.sum() > 0:
            valid_adv = advantages[b, valid_mask]
            adv_mean = valid_adv.mean()
            adv_std = valid_adv.std()
            advantages[b, valid_mask] = (valid_adv - adv_mean) / (adv_std + 1e-8)

    return advantages, returns
