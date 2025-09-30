"""
Training loop implementation with reduced complexity.

This module provides a TrainingLoop class that breaks down the complex
train_loop function into smaller, more manageable methods.
"""

import math
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch import GradScaler
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from tensordict import TensorDict

from melee_ai.config import Settings
from melee_ai.data import DatasetProvider
from melee_ai.models import ModelAdapter
from melee_ai.utils import guard_clause, safe_divide
from melee_ai.utils.logging import get_logger
from .metrics import MetricAggregator


class TrainingLoop:
    """Manages the training loop with reduced complexity."""

    def __init__(
        self,
        settings: Settings,
        model: ModelAdapter,
        dataset_provider: DatasetProvider
    ):
        """
        Initialize training loop.

        Args:
            settings: Configuration settings
            model: Model to train
            dataset_provider: Provider for training data
        """
        self.settings = settings
        self.model = model
        self.dataset_provider = dataset_provider
        self.logger = get_logger("training.loop", settings)

        # Setup training components
        self.device = self._setup_device()
        self.model = model.to(self.device)

        self.optimizer = self._setup_optimizer()
        self.scaler = GradScaler()
        self.scheduler = self._setup_scheduler()

        # Training state
        self.global_step = 0
        self.metrics = MetricAggregator(settings)

    def _setup_device(self) -> torch.device:
        """Setup training device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer."""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.settings.training.lr,
            betas=self.settings.training.betas,
            weight_decay=self.settings.training.weight_decay
        )

    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        # Placeholder - would use the LRScheduler class
        return None

    def run_training(self) -> Dict[int, Dict]:
        """
        Run the complete training loop.

        Returns:
            Dictionary mapping epoch numbers to training results
        """
        results = {}

        for epoch in range(self.settings.training.epochs):
            self.logger.info(f"Starting epoch {epoch + 1}/{self.settings.training.epochs}")

            # Setup for epoch
            if hasattr(self.dataset_provider, "set_epoch"):
                self.dataset_provider.set_epoch(epoch)

            # Run epoch
            epoch_result = self.run_epoch(epoch)

            # Log results
            self._log_epoch_results(epoch, epoch_result)

            # Save checkpoint if needed
            if (epoch + 1) % self.settings.logging.save_every_epochs == 0:
                self._save_checkpoint(epoch + 1)

            results[epoch] = epoch_result

        return results

    def run_epoch(self, epoch: int) -> Dict:
        """Run a single training epoch."""
        self.model.train()
        self.metrics.reset()

        epoch_loss = 0.0
        num_batches = 0

        start_time = time.time()

        # Get dataloader
        dataloader = self.dataset_provider.get_dataloader(self.settings)

        for batch_idx, batch in enumerate(dataloader):
            if self._should_stop_batch(batch_idx):
                break

            # Training step
            batch_result = self.train_step(batch)
            epoch_loss += batch_result["loss"]
            num_batches += 1

            self.global_step += 1

        duration = time.time() - start_time

        return {
            "epoch": epoch,
            "loss": safe_divide(epoch_loss, num_batches),
            "duration": duration,
            "batches_processed": num_batches,
        }

    def _should_stop_batch(self, batch_idx: int) -> bool:
        """Check if training should stop for this batch."""
        steps_per_epoch = self.settings.training.steps_per_epoch
        max_steps = self.settings.training.max_steps

        if steps_per_epoch and batch_idx >= steps_per_epoch:
            return True

        if max_steps and self.global_step >= max_steps:
            return True

        return False

    def train_step(self, batch) -> Dict:
        """Perform a single training step."""
        X, Y = batch
        X = X.to(self.device, non_blocking=True)
        Y = Y.to(self.device, non_blocking=True)

        # Forward pass
        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=str(self.device), dtype=torch.float16):
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

        # Update metrics
        self._update_metrics(outputs, Y)

        return {"loss": loss.item()}

    def _update_metrics(self, outputs, targets):
        """Update training metrics."""
        # This would extract predictions and update the MetricAggregator
        # For now, just a placeholder
        pass

    def _log_epoch_results(self, epoch: int, results: Dict):
        """Log training results for the epoch."""
        self.logger.info(
            f"Epoch {epoch + 1} completed: "
            f"loss={results['loss']:.4f}, "
            f"duration={results['duration']:.2f}s, "
            f"batches={results['batches_processed']}"
        )

    def _save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        checkpoint_path = Path(self.settings.logging.output_dir)
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
        self.logger.info(f"Checkpoint saved: {checkpoint_path / f'model_ep{epoch:03d}.pt'}")
