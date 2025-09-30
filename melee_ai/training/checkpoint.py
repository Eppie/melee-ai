"""
Checkpoint management for training.

This module handles saving and loading of model checkpoints
during training.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

import torch

from melee_ai.config import Settings


class CheckpointManager:
    """Manages model checkpoint saving and loading."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.checkpoint_dir = Path(settings.logging.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        model,
        optimizer,
        scaler,
        epoch: int,
        global_step: int,
        metrics: Optional[Dict] = None,
        **kwargs
    ) -> str:
        """
        Save model checkpoint.

        Args:
            model: Model to save
            optimizer: Optimizer state
            scaler: Gradient scaler state
            epoch: Current epoch
            global_step: Current global step
            metrics: Optional metrics to save
            **kwargs: Additional state to save

        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = self.checkpoint_dir / f"model_ep{epoch:03d}_{global_step:06d}.pt"

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "settings": self.settings.__dict__,
            "metrics": metrics or {},
            **kwargs
        }

        torch.save(checkpoint, checkpoint_path)
        return str(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str, device: str = "cpu") -> Dict:
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to load to

        Returns:
            Loaded checkpoint dictionary
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)
        return checkpoint

    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to latest checkpoint."""
        if not self.checkpoint_dir.exists():
            return None

        checkpoint_files = list(self.checkpoint_dir.glob("model_ep*.pt"))
        if not checkpoint_files:
            return None

        # Sort by modification time
        latest = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
        return str(latest)

    def list_checkpoints(self) -> List[str]:
        """List all available checkpoints."""
        if not self.checkpoint_dir.exists():
            return []

        return [str(p) for p in self.checkpoint_dir.glob("model_ep*.pt")]
