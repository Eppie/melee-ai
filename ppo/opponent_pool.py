"""Opponent pool management for self-play PPO."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional

import torch

from model.nano_gpt import GPT


class OpponentPool:
    """Manages a pool of opponent models for self-play.

    The pool maintains a fixed number of model checkpoints. When a new model
    is added and the pool is full, the oldest model is removed.
    """

    def __init__(self, max_size: int = 5, pool_dir: Optional[Path] = None):
        """Initialize the opponent pool.

        Args:
            max_size: Maximum number of opponents to keep in the pool
            pool_dir: Directory to save/load opponent checkpoints
        """
        self.max_size = max_size
        self.pool_dir = pool_dir or Path("checkpoints/opponent_pool")
        self.pool_dir.mkdir(parents=True, exist_ok=True)

        # List of (checkpoint_path, metadata) tuples
        self.opponents: List[tuple[Path, Dict]] = []

        # Counter for unique opponent IDs
        self.opponent_counter = 0

        # Load existing opponents from disk
        self._load_existing_opponents()

    def _load_existing_opponents(self) -> None:
        """Load existing opponent checkpoints from pool directory."""
        checkpoint_files = sorted(self.pool_dir.glob("opponent_*.pt"))

        for ckpt_path in checkpoint_files:
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                metadata = ckpt.get("metadata", {})
                self.opponents.append((ckpt_path, metadata))

                # Update counter based on opponent ID
                opp_id = metadata.get("opponent_id", 0)
                self.opponent_counter = max(self.opponent_counter, opp_id + 1)
            except Exception as e:
                print(f"Warning: Failed to load opponent checkpoint {ckpt_path}: {e}")

        print(f"Loaded {len(self.opponents)} opponents from {self.pool_dir}")

    def add_opponent(
        self,
        model: GPT,
        metadata: Optional[Dict] = None,
    ) -> Path:
        """Add a new opponent to the pool.

        If the pool is full, removes the oldest opponent first.

        Args:
            model: The model to add as an opponent
            metadata: Optional metadata (episode number, win rate, etc.)

        Returns:
            Path to the saved opponent checkpoint
        """
        # Remove oldest opponent if pool is full
        if len(self.opponents) >= self.max_size:
            oldest_path, oldest_meta = self.opponents.pop(0)
            try:
                oldest_path.unlink()
                print(
                    f"Removed oldest opponent: {oldest_path.name} (id={oldest_meta.get('opponent_id')})"
                )
            except Exception as e:
                print(f"Warning: Failed to remove old opponent {oldest_path}: {e}")

        # Save new opponent
        opponent_id = self.opponent_counter
        self.opponent_counter += 1

        opponent_path = self.pool_dir / f"opponent_{opponent_id:04d}.pt"

        meta = metadata or {}
        meta["opponent_id"] = opponent_id

        checkpoint = {
            "model": model.state_dict(),
            "metadata": meta,
        }

        torch.save(checkpoint, opponent_path)
        self.opponents.append((opponent_path, meta))

        print(
            f"Added opponent {opponent_id} to pool (pool size: {len(self.opponents)}/{self.max_size})"
        )
        return opponent_path

    def sample_opponent(self) -> tuple[Path, Dict]:
        """Randomly sample an opponent from the pool.

        Returns:
            Tuple of (checkpoint_path, metadata)

        Raises:
            ValueError: If the pool is empty
        """
        if len(self.opponents) == 0:
            raise ValueError("Opponent pool is empty. Cannot sample opponent.")

        return random.choice(self.opponents)

    def load_opponent_model(
        self,
        model: GPT,
        checkpoint_path: Optional[Path] = None,
    ) -> GPT:
        """Load an opponent model from a checkpoint.

        Args:
            model: Model instance to load weights into
            checkpoint_path: Path to checkpoint. If None, samples randomly.

        Returns:
            Model with loaded weights
        """
        if checkpoint_path is None:
            checkpoint_path, _ = self.sample_opponent()

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.eval()

        return model

    def get_pool_stats(self) -> Dict:
        """Get statistics about the opponent pool.

        Returns:
            Dictionary with pool statistics
        """
        return {
            "pool_size": len(self.opponents),
            "max_size": self.max_size,
            "total_opponents_created": self.opponent_counter,
            "opponents": [
                {
                    "path": str(path),
                    "metadata": meta,
                }
                for path, meta in self.opponents
            ],
        }

    def is_empty(self) -> bool:
        """Check if the pool is empty."""
        return len(self.opponents) == 0

    def __len__(self) -> int:
        return len(self.opponents)
