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
from torch.amp import autocast
from torch.optim import AdamW

from interp.sae.topk import TopKSparseAutoencoder

if TYPE_CHECKING:
    from interp.cache import CachedActivations
    from interp.config import SAEConfig


@dataclass
class TrainingStats:
    """Statistics from SAE training."""

    step: int
    reconstruction_loss: float
    explained_variance: float  # R² - how much variance is reconstructed
    avg_active_features: float
    mean_activation: float  # mean magnitude of active features
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
        use_amp: bool = True,
        amp_dtype: torch.dtype = torch.bfloat16,
        use_compile: bool = True,
    ):
        self.config = config
        self.device = device

        # AMP settings
        self.use_amp = use_amp and device.type == "cuda"
        self.amp_dtype = amp_dtype

        # Enable cudnn benchmark for consistent input sizes
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        # Compile the SAE for faster execution
        self.use_compile = use_compile and device.type == "cuda"
        sae = sae.to(device)
        if self.use_compile:
            sae = torch.compile(sae, mode="max-autotune")
        self.sae = sae

        # Use fused AdamW for better GPU performance
        use_fused = device.type == "cuda"
        self.optimizer = AdamW(sae.parameters(), lr=config.lr, fused=use_fused)

        # Dead feature tracking
        self._steps_since_fired = torch.zeros(
            sae.hidden_dim, dtype=torch.long, device=device
        )
        self._dead_threshold = config.dead_feature_threshold

    def train(
        self,
        cached,  # CachedActivations or StreamingActivations
        show_progress: bool = True,
        log_every: int = 200,
        checkpoint_dir: Optional[Path] = None,
        checkpoint_every: int = 1000,
    ) -> TrainingHistory:
        """
        Train the SAE on cached or streaming activations.

        Args:
            cached: CachedActivations or StreamingActivations object with training data
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

        # Detect streaming mode
        streaming_mode = hasattr(cached, 'n_samples')

        if streaming_mode:
            # StreamingActivations - samples come normalized from shuffle buffer
            n_samples = cached.n_samples
            activation_dim = cached.activation_dim
            cached.to(self.device)  # Move normalization stats to device
        else:
            # CachedActivations - load all into GPU memory
            activations = cached.normalized.to(self.device)
            n_samples = len(activations)
            activation_dim = activations.shape[1]

        total_steps = self.config.training_steps
        start_time = time.time()
        last_log_time = start_time
        last_log_step = 0
        ema_steps_per_sec = None  # EMA of steps/s
        ema_alpha = 2 / (10 + 1)  # ~10 sample EMA smoothing

        if show_progress:
            print(f"\n{'─' * 60}")
            print(f"Training SAE: {total_steps:,} steps")
            print(f"{'─' * 60}")
            mode_str = "streaming" if streaming_mode else "in-memory"
            print(f"  Activations: {n_samples:,} samples × {activation_dim} dims ({mode_str})")
            print(f"  SAE: {activation_dim} → {self.sae.hidden_dim:,} features (TopK={self.sae.k})")
            print(f"  Batch size: {self.config.batch_size:,}")
            if self.use_amp:
                print(f"  AMP: enabled ({self.amp_dtype})")
            if self.use_compile:
                print(f"  torch.compile: enabled")
            print(f"{'─' * 60}")

        for step in range(total_steps):
            # Sample batch - different paths for streaming vs in-memory
            if streaming_mode:
                batch, _ = cached.sample_batch(self.config.batch_size)
                batch = batch.to(self.device)
            else:
                indices = torch.randint(0, n_samples, (self.config.batch_size,))
                batch = activations[indices]

            # Forward pass with AMP
            with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                output = self.sae(batch)

            # Backward pass (loss is already float32 from autocast)
            self.optimizer.zero_grad()
            output.reconstruction_loss.backward()
            self.optimizer.step()

            # Post-step hook (decoder normalization)
            self.sae.post_step_hook()

            # Update dead feature tracking
            self._update_dead_tracking(output.latents)

            # Log progress
            if step % log_every == 0 or step == total_steps - 1:
                stats = self._compute_stats(step, output, batch)
                history.add(stats)

                if show_progress:
                    current_time = time.time()
                    steps_done = step + 1
                    pct = 100 * steps_done / total_steps

                    # Compute instantaneous steps/s since last log
                    interval_time = current_time - last_log_time
                    interval_steps = step - last_log_step
                    if interval_time > 0 and interval_steps > 0:
                        instant_steps_per_sec = interval_steps / interval_time
                        # Update EMA
                        if ema_steps_per_sec is None:
                            ema_steps_per_sec = instant_steps_per_sec
                        else:
                            ema_steps_per_sec = ema_alpha * instant_steps_per_sec + (1 - ema_alpha) * ema_steps_per_sec

                    last_log_time = current_time
                    last_log_step = step

                    # Use EMA for display and ETA
                    display_steps_per_sec = ema_steps_per_sec if ema_steps_per_sec else 0
                    eta_sec = (total_steps - steps_done) / display_steps_per_sec if display_steps_per_sec > 0 else 0

                    # Format ETA
                    if eta_sec < 60:
                        eta_str = f"{eta_sec:.0f}s"
                    elif eta_sec < 3600:
                        eta_str = f"{eta_sec / 60:.1f}m"
                    else:
                        eta_str = f"{eta_sec / 3600:.1f}h"

                    frames_per_sec = display_steps_per_sec * self.config.batch_size
                    if frames_per_sec >= 1_000_000:
                        throughput_str = f"{frames_per_sec / 1_000_000:.2f}M fr/s"
                    elif frames_per_sec >= 1_000:
                        throughput_str = f"{frames_per_sec / 1_000:.0f}K fr/s"
                    else:
                        throughput_str = f"{frames_per_sec:.0f} fr/s"

                    print(
                        f"  [{pct:5.1f}%] Step {step:>6,}/{total_steps:,} | "
                        f"Loss: {stats.reconstruction_loss:.4f} | "
                        f"R²: {stats.explained_variance:.3f} | "
                        f"Act: {stats.avg_active_features:.0f}/{stats.mean_activation:.2f} | "
                        f"Dead: {stats.num_dead_features:,} | "
                        f"{throughput_str} | ETA: {eta_str}"
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

        # Cleanup streaming resources
        if streaming_mode and hasattr(cached, 'close'):
            cached.close()

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

    def _compute_stats(self, step: int, output, batch: Tensor) -> TrainingStats:
        """Compute training statistics."""
        dead_features = self.get_dead_features()

        # Count average active features
        active_mask = output.latents > 0
        active_count = active_mask.float().sum(dim=1).mean()

        # Mean activation magnitude (of active features only)
        if active_mask.any():
            mean_activation = output.latents[active_mask].mean().item()
        else:
            mean_activation = 0.0

        # Explained variance (R²) = 1 - Var(residual) / Var(input)
        residual = batch - output.reconstruction
        var_residual = residual.var()
        var_input = batch.var()
        if var_input > 0:
            explained_variance = (1 - var_residual / var_input).item()
        else:
            explained_variance = 0.0

        return TrainingStats(
            step=step,
            reconstruction_loss=output.reconstruction_loss.item(),
            explained_variance=explained_variance,
            avg_active_features=active_count.item(),
            mean_activation=mean_activation,
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

        # Handle compiled modules - access original for saving
        sae_to_save = self.sae
        if hasattr(sae_to_save, '_orig_mod'):
            sae_to_save = sae_to_save._orig_mod
        sae_to_save.save(str(path))


def train_sae(
    model: "GPT",
    dataloader,
    colmap: "ColumnMap",
    hook_point: "HookPoint",
    config: "SAEConfig",
    device: torch.device,
    max_activation_samples: int = 100000,
    stratified: bool = False,
    stratified_temperature: float = 1.0,
    show_progress: bool = True,
    checkpoint_dir: Optional[Path] = None,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    use_compile: bool = True,
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
        stratified: Whether to use inverse-density stratified sampling
        stratified_temperature: Temperature for inverse-density weighting (1.0=pure inverse density)
        show_progress: Whether to print progress
        checkpoint_dir: Where to save checkpoints
        use_amp: Whether to use automatic mixed precision
        amp_dtype: Data type for AMP (default: bfloat16)
        use_compile: Whether to use torch.compile (default: True)

    Returns:
        Trained SAE and training history
    """
    from interp.cache import ActivationCache

    # Cache activations
    cache = ActivationCache(model, hook_point, device)

    if stratified:
        cached = cache.fill_with_loss_stratification(
            dataloader, colmap, max_samples=max_activation_samples,
            temperature=stratified_temperature, show_progress=show_progress
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
    trainer = SAETrainer(sae, config, device, use_amp=use_amp, amp_dtype=amp_dtype, use_compile=use_compile)
    history = trainer.train(cached, show_progress=show_progress, checkpoint_dir=checkpoint_dir)

    # Return the underlying SAE (unwrap if compiled)
    trained_sae = trainer.sae
    if hasattr(trained_sae, '_orig_mod'):
        trained_sae = trained_sae._orig_mod

    return trained_sae, history


__all__ = ["SAETrainer", "TrainingStats", "TrainingHistory", "train_sae"]
