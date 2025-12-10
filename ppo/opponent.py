"""Opponent pool management for self-play training.

This module manages:
- Historical checkpoint sampling (80%)
- Self-play matches (20%)
- Opponent policy assignment
- Checkpoint discovery and loading
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional

import torch

from model.nano_gpt import GPT


class OpponentPool:
    """
    Manages pool of historical checkpoints for opponent sampling.

    Responsibilities:
    - Discover available checkpoints in directory
    - Sample opponents with 80/20 historical/self-play split
    - Load checkpoint weights on demand
    - Refresh pool periodically
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        historical_ratio: float = 0.8,
        max_pool_size: int = 50,
        refresh_interval: int = 100,  # Refresh every N training steps
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.historical_ratio = historical_ratio
        self.max_pool_size = max_pool_size
        self.refresh_interval = refresh_interval

        # Available checkpoint paths
        self.checkpoint_paths: List[Path] = []

        # Cached checkpoint weights (to avoid repeated loading)
        # Maps checkpoint_path → state_dict
        self.checkpoint_cache: Dict[Path, Dict] = {}

        # Step counter for refresh
        self.steps_since_refresh = 0

        # Discover initial checkpoints
        self.refresh_pool()

    def refresh_pool(self):
        """Scan checkpoint directory for available checkpoints."""
        if not self.checkpoint_dir.exists():
            print(
                f"[OpponentPool] Warning: checkpoint dir {self.checkpoint_dir} does not exist"
            )
            self.checkpoint_paths = []
            return

        # Find all .pt files
        all_checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_*.pt"),
            key=lambda p: p.stat().st_mtime,  # Sort by modification time
        )

        # Keep most recent max_pool_size checkpoints
        self.checkpoint_paths = all_checkpoints[-self.max_pool_size :]

        # Only log on first refresh or if count changed
        if not hasattr(self, '_last_checkpoint_count') or self._last_checkpoint_count != len(self.checkpoint_paths):
            print(f"[OpponentPool] Found {len(self.checkpoint_paths)} checkpoints")
            self._last_checkpoint_count = len(self.checkpoint_paths)

        # Clear cache for removed checkpoints
        valid_paths = set(self.checkpoint_paths)
        self.checkpoint_cache = {
            k: v for k, v in self.checkpoint_cache.items() if k in valid_paths
        }

        self.steps_since_refresh = 0

    def maybe_refresh(self):
        """Periodically refresh pool to pick up new checkpoints."""
        self.steps_since_refresh += 1
        if self.steps_since_refresh >= self.refresh_interval:
            self.refresh_pool()

    def should_use_historical(self) -> bool:
        """Decide if this match should use historical opponent (vs self-play)."""
        return random.random() < self.historical_ratio

    def sample_checkpoint_path(self) -> Optional[Path]:
        """
        Sample a checkpoint for opponent.

        Returns:
            Path to checkpoint, or None for self-play
        """
        if not self.checkpoint_paths:
            # No historical checkpoints available, use self-play
            return None

        if not self.should_use_historical():
            # Self-play match
            return None

        # Sample uniformly from pool
        return random.choice(self.checkpoint_paths)

    def load_checkpoint_weights(
        self,
        checkpoint_path: Path,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """
        Load checkpoint weights (with caching).

        Args:
            checkpoint_path: Path to checkpoint .pt file
            device: Device to load weights to

        Returns:
            state_dict for model
        """
        # Check cache
        if checkpoint_path in self.checkpoint_cache:
            return self.checkpoint_cache[checkpoint_path]

        # Load from disk
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,  # Need to load full checkpoint
        )

        # Extract model state dict
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            # Assume checkpoint is the state dict itself
            state_dict = checkpoint

        # Cache it
        self.checkpoint_cache[checkpoint_path] = state_dict

        # Removed verbose logging (happens frequently during opponent sampling)

        return state_dict


class Matchmaker:
    """
    Assigns opponent policies to environments.

    For each environment:
    - 80% of time: Sample historical checkpoint
    - 20% of time: Self-play (use current policy)

    Maintains match assignments that persist for a full rollout (1024 frames).
    """

    def __init__(
        self,
        opponent_pool: OpponentPool,
        num_envs: int = 96,
        device: torch.device = torch.device("cpu"),
    ):
        self.pool = opponent_pool
        self.num_envs = num_envs
        self.device = device

        # Current match assignments
        # Maps env_id → checkpoint_path (None = self-play)
        self.match_assignments: Dict[int, Optional[Path]] = {}

        # Loaded opponent models
        # Maps checkpoint_path → GPT model instance
        self.loaded_opponents: Dict[Path, GPT] = {}

        # Initialize assignments
        self.reassign_all()

    def reassign_all(self):
        """Reassign all environments (called at start of each rollout)."""
        # Removed verbose logging (happens every rollout)

        # Refresh pool periodically
        self.pool.maybe_refresh()

        # Sample new assignments
        for env_id in range(self.num_envs):
            checkpoint_path = self.pool.sample_checkpoint_path()
            self.match_assignments[env_id] = checkpoint_path

        # Count self-play vs historical
        self_play_count = sum(
            1 for path in self.match_assignments.values() if path is None
        )
        historical_count = self.num_envs - self_play_count

        print(
            f"[Matchmaker] Assigned: {historical_count} historical "
            f"({historical_count/self.num_envs*100:.1f}%), "
            f"{self_play_count} self-play ({self_play_count/self.num_envs*100:.1f}%)"
        )

    def get_opponent_model(
        self,
        env_id: int,
        base_model: GPT,
    ) -> Optional[GPT]:
        """
        Get opponent model for given environment.

        Args:
            env_id: Environment ID (0-95)
            base_model: Base model to clone for opponent (if historical)

        Returns:
            GPT model for opponent, or None for self-play
        """
        checkpoint_path = self.match_assignments.get(env_id)

        if checkpoint_path is None:
            # Self-play: use current policy
            return None

        # Load historical checkpoint
        if checkpoint_path not in self.loaded_opponents:
            # Clone base model and load weights
            opponent_model = self._clone_model(base_model)
            state_dict = self.pool.load_checkpoint_weights(checkpoint_path, self.device)
            opponent_model.load_state_dict(state_dict)
            opponent_model.eval()  # Always in eval mode
            self.loaded_opponents[checkpoint_path] = opponent_model

        return self.loaded_opponents[checkpoint_path]

    def _clone_model(self, model: GPT) -> GPT:
        """Create a copy of model architecture (without weights)."""
        # Create new model with same config
        new_model = GPT(model.config)
        new_model.to(self.device)
        return new_model

    def clear_loaded_opponents(self):
        """Clear loaded opponent models to free memory."""
        # Only log if we're actually clearing models (reduces verbosity)
        if len(self.loaded_opponents) > 0:
            # Don't log every time, this happens frequently
            pass
        self.loaded_opponents.clear()

    def is_self_play(self, env_id: int) -> bool:
        """Check if environment is in self-play mode."""
        return self.match_assignments.get(env_id) is None
