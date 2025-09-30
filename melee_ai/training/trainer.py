"""
Training orchestration for Melee AI.

This module provides the main training loop and lifecycle management
for model training.
"""

import time
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

from melee_ai.config import Settings
from melee_ai.data import DatasetProvider
from melee_ai.models import ModelAdapter


class TrainingResult:
    """Result of a training epoch."""

    def __init__(self):
        self.epoch: int = 0
        self.loss: float = 0.0
        self.metrics: Dict[str, float] = {}
        self.duration: float = 0.0


class Trainer:
    """Main training orchestration class."""

    def __init__(
        self,
        settings: Settings,
        model: ModelAdapter,
        dataset_provider: DatasetProvider,
        device: Optional[str] = None
    ):
        """
        Initialize trainer.

        Args:
            settings: Configuration settings
            model: Model to train
            dataset_provider: Provider for training data
            device: Device to train on (auto-detected if None)
        """
        self.settings = settings
        self.model = model
        self.dataset_provider = dataset_provider
        self.device = device or self._detect_device()

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.optimizer = None
        self.scaler = None

        # Setup training components
        self._setup_training()

    def _detect_device(self) -> str:
        """Detect available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _setup_training(self):
        """Setup optimizer and training components."""
        self.model.to(self.device)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.settings.training.lr,
            betas=self.settings.training.betas,
            weight_decay=self.settings.training.weight_decay
        )

        self.scaler = GradScaler()

    def train(self) -> Dict[int, TrainingResult]:
        """
        Run full training loop.

        Returns:
            Dictionary mapping epoch numbers to training results
        """
        results = {}

        for epoch in range(self.settings.training.epochs):
            self.current_epoch = epoch

            # Setup for epoch
            if hasattr(self.dataset_provider, "set_epoch"):
                self.dataset_provider.set_epoch(epoch)

            result = self.train_epoch()
            results[epoch] = result

            # Save checkpoint if needed
            if (epoch + 1) % self.settings.logging.save_every_epochs == 0:
                self.save_checkpoint(epoch + 1)

        return results

    def train_epoch(self) -> TrainingResult:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0

        start_time = time.time()

        # Get dataloader
        dataloader = self.dataset_provider.get_dataloader(self.settings)

        for batch in dataloader:
            if self.settings.training.steps_per_epoch and num_batches >= self.settings.training.steps_per_epoch:
                break

            # Training step
            loss = self.train_step(batch)
            epoch_loss += loss
            num_batches += 1

            self.global_step += 1

            # Check for early stopping
            if self.settings.training.max_steps and self.global_step >= self.settings.training.max_steps:
                break

        duration = time.time() - start_time

        return TrainingResult(
            epoch=self.current_epoch,
            loss=epoch_loss / num_batches if num_batches > 0 else 0.0,
            metrics={"avg_batch_loss": epoch_loss / num_batches if num_batches > 0 else 0.0},
            duration=duration
        )

    def train_step(self, batch) -> float:
        """Perform single training step."""
        X, Y = batch
        X = X.to(self.device, non_blocking=True)
        Y = Y.to(self.device, non_blocking=True)

        # Forward pass
        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=self.device, dtype=torch.float16):
            outputs = self.model(X)
            loss = self.model.compute_loss(outputs, Y)

        # Backward pass
        self.scaler.scale(loss).backward()

        # Gradient clipping
        if self.settings.training.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.settings.training.grad_clip)

        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item()

    def save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        checkpoint_path = Path(self.settings.logging.checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "epoch": epoch,
            "global_step": self.global_step,
            "settings": self.settings.__dict__,
        }

        torch.save(checkpoint, checkpoint_path / f"model_ep{epoch:03d}.pt")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.current_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)

        return checkpoint
