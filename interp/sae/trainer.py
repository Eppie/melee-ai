"""
SAE training with dead feature tracking and resampling.

Provides a clean training interface that handles:
- Batched training from cached activations
- Dead feature detection and optional resampling
- Logging and checkpointing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import torch
from torch import Tensor
from torch.optim import Adam

from interp.sae.topk import TopKSparseAutoencoder

if TYPE_CHECKING:
    from interp.cache import CachedActivations
    from interp.config import SAEConfig


@dataclass
class TrainingStats:
    """Statistics from SAE training."""

    step: int
    reconstruction_loss: float
    avg_active_features: float
    num_dead_features: int
    learning_rate: float


@dataclass
class TrainingHistory:
    """Full training history."""

    stats: List[TrainingStats] = field(default_factory=list)
    final_dead_features: List[int] = field(default_factory=list)

    def add(self, stats: TrainingStats) -> None:
        self.stats.append(stats)

    @property
    def losses(self) -> List[float]:
        return [s.reconstruction_loss for s in self.stats]

    @property
    def steps(self) -> List[int]:
        return [s.step for s in self.stats]


class SAETrainer:
    """
    Trainer for TopK Sparse Autoencoders.

    Handles the training loop with:
    - Dead feature tracking (features that haven't fired for N steps)
    - Optional dead feature resampling
    - Decoder normalization after each step
    - Progress logging

    Usage:
        trainer = SAETrainer(sae, config, device)
        history = trainer.train(cached_activations, show_progress=True)

        # Check for dead features
        dead = trainer.get_dead_features()
    """

    def __init__(
        self,
        sae: TopKSparseAutoencoder,
        config: "SAEConfig",
        device: torch.device,
    ):
        self.sae = sae.to(device)
        self.config = config
        self.device = device

        self.optimizer = Adam(sae.parameters(), lr=config.lr)

        # Dead feature tracking
        self._steps_since_fired = torch.zeros(
            sae.hidden_dim, dtype=torch.long, device=device
        )
        self._dead_threshold = config.dead_feature_threshold

    def train(
        self,
        cached: "CachedActivations",
        show_progress: bool = True,
        log_every: int = 50,
        checkpoint_dir: Optional[Path] = None,
        checkpoint_every: int = 1000,
    ) -> TrainingHistory:
        """
        Train the SAE on cached activations.

        Args:
            cached: CachedActivations object with training data
            show_progress: Whether to print progress
            log_every: Steps between progress logs
            checkpoint_dir: Directory for saving checkpoints (optional)
            checkpoint_every: Steps between checkpoints

        Returns:
            TrainingHistory with loss curves and statistics
        """
        import time

        self.sae.train()
        history = TrainingHistory()

        # Get activations (optionally normalized)
        activations = cached.normalized.to(self.device)
        n_samples = len(activations)

        total_steps = self.config.training_steps
        start_time = time.time()

        if show_progress:
            print(f"\n{'─' * 60}")
            print(f"Training SAE: {total_steps:,} steps")
            print(f"{'─' * 60}")
            print(f"  Activations: {activations.shape[0]:,} samples × {activations.shape[1]} dims")
            print(f"  SAE: {activations.shape[1]} → {self.sae.hidden_dim:,} features (TopK={self.sae.k})")
            print(f"  Batch size: {self.config.batch_size:,}")
            print(f"{'─' * 60}")

        for step in range(total_steps):
            # Sample batch
            indices = torch.randint(0, n_samples, (self.config.batch_size,))
            batch = activations[indices]

            # Forward pass
            output = self.sae(batch)

            # Backward pass
            self.optimizer.zero_grad()
            output.reconstruction_loss.backward()
            self.optimizer.step()

            # Post-step hook (decoder normalization)
            self.sae.post_step_hook()

            # Update dead feature tracking
            self._update_dead_tracking(output.latents)

            # Log progress
            if step % log_every == 0 or step == total_steps - 1:
                stats = self._compute_stats(step, output)
                history.add(stats)

                if show_progress:
                    elapsed = time.time() - start_time
                    steps_done = step + 1
                    steps_per_sec = steps_done / elapsed if elapsed > 0 else 0
                    eta_sec = (total_steps - steps_done) / steps_per_sec if steps_per_sec > 0 else 0
                    pct = 100 * steps_done / total_steps

                    # Format ETA
                    if eta_sec < 60:
                        eta_str = f"{eta_sec:.0f}s"
                    elif eta_sec < 3600:
                        eta_str = f"{eta_sec / 60:.1f}m"
                    else:
                        eta_str = f"{eta_sec / 3600:.1f}h"

                    print(
                        f"  [{pct:5.1f}%] Step {step:>6,}/{total_steps:,} | "
                        f"Loss: {stats.reconstruction_loss:.5f} | "
                        f"Active: {stats.avg_active_features:.1f} | "
                        f"Dead: {stats.num_dead_features:,} | "
                        f"{steps_per_sec:.1f} steps/s | ETA: {eta_str}"
                    )

            # Save checkpoint
            if checkpoint_dir is not None and (step + 1) % checkpoint_every == 0:
                self._save_checkpoint(checkpoint_dir, step + 1)
                if show_progress:
                    print(f"  💾 Checkpoint saved at step {step + 1}")

        # Final stats
        dead_features = self.get_dead_features()
        history.final_dead_features = dead_features.tolist()

        elapsed_total = time.time() - start_time
        if elapsed_total < 60:
            time_str = f"{elapsed_total:.1f}s"
        elif elapsed_total < 3600:
            time_str = f"{elapsed_total / 60:.1f}m"
        else:
            time_str = f"{elapsed_total / 3600:.1f}h"

        if show_progress:
            print(f"{'─' * 60}")
            print(f"Training complete in {time_str}")
            print(f"  Final loss: {history.losses[-1]:.6f}")
            print(f"  Dead features: {len(dead_features):,} / {self.sae.hidden_dim:,} ({100*len(dead_features)/self.sae.hidden_dim:.1f}%)")
            print(f"{'─' * 60}\n")

        # Save final checkpoint
        if checkpoint_dir is not None:
            self._save_checkpoint(checkpoint_dir, self.config.training_steps, final=True)

        return history

    def _update_dead_tracking(self, latents: Tensor) -> None:
        """Update tracking of which features have fired."""
        # Check which features are active in this batch
        active_mask = (latents > 0).any(dim=0)  # [hidden_dim]

        # Reset counter for active features, increment for inactive
        self._steps_since_fired[active_mask] = 0
        self._steps_since_fired[~active_mask] += 1

    def get_dead_features(self) -> Tensor:
        """Get indices of features that haven't fired recently."""
        return torch.where(self._steps_since_fired >= self._dead_threshold)[0]

    def get_feature_firing_rates(self) -> Tensor:
        """Get approximate firing rate for each feature (inverse of steps since fired)."""
        # Lower bound to avoid division by zero
        steps = self._steps_since_fired.clamp(min=1).float()
        return 1.0 / steps

    def resample_dead_features(
        self, cached: "CachedActivations", n_samples: int = 10000
    ) -> int:
        """
        Resample dead features to improve SAE utilization.

        Dead features are reinitialized to point toward high-reconstruction-error
        inputs, giving them a chance to learn useful representations.

        Args:
            cached: CachedActivations to sample from
            n_samples: Number of samples for finding high-error inputs

        Returns:
            Number of features resampled
        """
        dead_features = self.get_dead_features()
        n_dead = len(dead_features)

        if n_dead == 0:
            return 0

        # Sample activations and find high-error inputs
        activations = cached.normalized.to(self.device)
        indices = torch.randint(0, len(activations), (n_samples,))
        batch = activations[indices]

        with torch.no_grad():
            output = self.sae(batch)
            errors = (batch - output.reconstruction).pow(2).sum(dim=-1)

            # Get highest error inputs
            _, top_error_idx = torch.topk(errors, k=min(n_dead, n_samples))
            high_error_inputs = batch[top_error_idx]

        # Resample dead features
        with torch.no_grad():
            for i, feat_idx in enumerate(dead_features):
                if i >= len(high_error_inputs):
                    break

                # Set encoder to point toward this high-error input
                direction = high_error_inputs[i] - self.sae.b_dec
                direction = direction / direction.norm().clamp(min=1e-8)

                self.sae.W_enc[:, feat_idx] = direction
                self.sae.W_dec[feat_idx, :] = direction
                self.sae.b_enc[feat_idx] = 0.0

            # Reset dead tracking for resampled features
            self._steps_since_fired[dead_features] = 0

        return n_dead

    def _compute_stats(self, step: int, output) -> TrainingStats:
        """Compute training statistics."""
        dead_features = self.get_dead_features()

        # Count average active features
        active_count = (output.latents > 0).float().sum(dim=1).mean()

        return TrainingStats(
            step=step,
            reconstruction_loss=output.reconstruction_loss.item(),
            avg_active_features=active_count.item(),
            num_dead_features=len(dead_features),
            learning_rate=self.optimizer.param_groups[0]["lr"],
        )

    def _print_progress(self, stats: TrainingStats) -> None:
        """Print training progress."""
        print(
            f"  Step {stats.step:5d} | "
            f"Loss: {stats.reconstruction_loss:.6f} | "
            f"Active: {stats.avg_active_features:.1f} | "
            f"Dead: {stats.num_dead_features}"
        )

    def _save_checkpoint(
        self, checkpoint_dir: Path, step: int, final: bool = False
    ) -> None:
        """Save training checkpoint."""
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if final:
            path = checkpoint_dir / "sae_final.pt"
        else:
            path = checkpoint_dir / f"sae_step_{step:06d}.pt"

        self.sae.save(str(path))


def train_sae(
    model: "GPT",
    dataloader,
    colmap: "ColumnMap",
    hook_point: "HookPoint",
    config: "SAEConfig",
    device: torch.device,
    max_activation_samples: int = 100000,
    stratified: bool = False,
    show_progress: bool = True,
    checkpoint_dir: Optional[Path] = None,
) -> tuple[TopKSparseAutoencoder, TrainingHistory]:
    """
    Convenience function to train an SAE end-to-end.

    Args:
        model: The GPT model to analyze
        dataloader: DataLoader for training data
        colmap: Column mapping for building inputs
        hook_point: Where to extract activations
        config: SAE configuration
        device: Torch device
        max_activation_samples: How many activations to cache
        stratified: Whether to use loss-stratified sampling
        show_progress: Whether to print progress
        checkpoint_dir: Where to save checkpoints

    Returns:
        Trained SAE and training history
    """
    from interp.cache import ActivationCache

    # Cache activations
    cache = ActivationCache(model, hook_point, device)

    if stratified:
        cached = cache.fill_with_loss_stratification(
            dataloader, colmap, max_samples=max_activation_samples, show_progress=show_progress
        )
    else:
        cached = cache.fill_from_dataloader(
            dataloader, colmap, max_samples=max_activation_samples, show_progress=show_progress
        )

    # Create SAE
    input_dim = cached.activations.shape[1]
    sae = TopKSparseAutoencoder(
        input_dim=input_dim,
        expansion_factor=config.expansion_factor,
        k=config.k,
        normalize_decoder=config.normalize_decoder,
    )

    # Train
    trainer = SAETrainer(sae, config, device)
    history = trainer.train(cached, show_progress=show_progress, checkpoint_dir=checkpoint_dir)

    return sae, history


__all__ = ["SAETrainer", "TrainingStats", "TrainingHistory", "train_sae"]
