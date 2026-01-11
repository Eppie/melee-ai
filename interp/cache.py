"""
Activation caching with metadata for interpretability analysis.

Extends the hook system to store activations along with corresponding
input features for correlation analysis and feature interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from interp.hooks import HookManager, HookPoint, HookPointType

if TYPE_CHECKING:
    from column_map import ColumnMap
    from model.nano_gpt import GPT


@dataclass
class CachedActivations:
    """
    Stored activations with corresponding input features.

    Attributes:
        activations: Flattened activations [N, D] where N = total frames
        inputs: Corresponding input features [N, F] for correlation analysis
        mean: Per-dimension mean of activations (for normalization)
        std: Per-dimension std of activations (for normalization)
    """

    activations: Tensor  # [N, D]
    inputs: Tensor  # [N, F]
    mean: Tensor  # [D]
    std: Tensor  # [D]

    @property
    def normalized(self) -> Tensor:
        """Return activations normalized to zero mean, unit variance."""
        return (self.activations - self.mean) / (self.std + 1e-8)

    def get_batch(self, indices: Tensor) -> Tuple[Tensor, Tensor]:
        """Get a batch of activations and inputs by indices."""
        return self.activations[indices], self.inputs[indices]

    def sample_batch(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        """Sample a random batch of activations and inputs."""
        indices = torch.randperm(len(self.activations))[:batch_size]
        return self.get_batch(indices)


class ActivationCache:
    """
    High-level interface for collecting and caching model activations.

    Handles:
    - Running the model with hooks installed
    - Storing activations with corresponding input features
    - Normalization statistics computation
    - Optional stratified sampling based on model loss

    Usage:
        cache = ActivationCache(model, hook_point, device)
        cache.fill_from_dataloader(dataloader, colmap, max_samples=100000)

        # Get normalized activations for SAE training
        acts = cache.get_cached().normalized

        # Sample batches for training
        act_batch, input_batch = cache.get_cached().sample_batch(4096)
    """

    def __init__(
        self,
        model: "GPT",
        hook_point: HookPoint,
        device: torch.device,
        normalize: bool = True,
    ):
        self.model = model
        self.hook_point = hook_point
        self.device = device
        self.normalize = normalize

        self._hook_manager = HookManager(model)
        self._cached: Optional[CachedActivations] = None

    def fill_from_dataloader(
        self,
        dataloader: DataLoader,
        colmap: "ColumnMap",
        max_samples: int = 100000,
        stratified: bool = False,
        show_progress: bool = True,
    ) -> CachedActivations:
        """
        Fill the cache by running the model on data from the dataloader.

        Args:
            dataloader: DataLoader yielding batches with "X" key
            colmap: Column mapping for building model inputs
            max_samples: Maximum number of activation samples to collect
            stratified: Whether to use loss-stratified sampling (oversample uncertain frames)
            show_progress: Whether to print progress

        Returns:
            CachedActivations object with collected data
        """
        import time

        from train.batch_utils import build_model_inputs

        self.model.eval()
        self._hook_manager.install_hooks([self.hook_point])

        all_activations: List[Tensor] = []
        all_inputs: List[Tensor] = []
        total_samples = 0
        start_time = time.time()

        if show_progress:
            print(f"\n{'─' * 60}")
            print(f"Caching activations at {self.hook_point}")
            print(f"{'─' * 60}")
            print(f"  Target samples: {max_samples:,}")

        try:
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    X = batch["X"].to(self.device)
                    batch_size, seq_len, _ = X.shape

                    # Build model inputs and run forward pass
                    exclude_p1 = getattr(self.model.config.model, 'exclude_p1_controller', False)
                    inputs_td = build_model_inputs(X, colmap, exclude_p1_controller=exclude_p1)
                    _ = self.model(inputs_td)

                    # Get activations from hook
                    acts = self._hook_manager.get_single(self.hook_point)
                    # acts shape: [batch, seq, dim]

                    # Flatten batch and sequence dimensions
                    acts_flat = acts.view(-1, acts.shape[-1])  # [B*L, D]
                    inputs_flat = X.view(-1, X.shape[-1])  # [B*L, F]

                    all_activations.append(acts_flat.cpu())
                    all_inputs.append(inputs_flat.cpu())
                    self._hook_manager.clear()

                    total_samples += acts_flat.shape[0]

                    if show_progress and batch_idx % 10 == 0:
                        elapsed = time.time() - start_time
                        samples_per_sec = total_samples / elapsed if elapsed > 0 else 0
                        remaining = max_samples - total_samples
                        eta_sec = remaining / samples_per_sec if samples_per_sec > 0 else 0
                        pct = 100 * total_samples / max_samples

                        # Format ETA
                        if eta_sec < 60:
                            eta_str = f"{eta_sec:.0f}s"
                        elif eta_sec < 3600:
                            eta_str = f"{eta_sec / 60:.1f}m"
                        else:
                            eta_str = f"{eta_sec / 3600:.1f}h"

                        print(
                            f"  [{pct:5.1f}%] {total_samples:,}/{max_samples:,} samples | "
                            f"{samples_per_sec:,.0f} samples/s | ETA: {eta_str}"
                        )

                    if total_samples >= max_samples:
                        break

        finally:
            self._hook_manager.remove_hooks()

        # Concatenate all collected data
        activations = torch.cat(all_activations, dim=0)[:max_samples]
        inputs = torch.cat(all_inputs, dim=0)[:max_samples]

        # Compute normalization statistics
        mean = activations.mean(dim=0)
        std = activations.std(dim=0)

        # Shuffle the data
        perm = torch.randperm(len(activations))
        activations = activations[perm]
        inputs = inputs[perm]

        self._cached = CachedActivations(
            activations=activations,
            inputs=inputs,
            mean=mean,
            std=std,
        )

        if show_progress:
            elapsed_total = time.time() - start_time
            if elapsed_total < 60:
                time_str = f"{elapsed_total:.1f}s"
            elif elapsed_total < 3600:
                time_str = f"{elapsed_total / 60:.1f}m"
            else:
                time_str = f"{elapsed_total / 3600:.1f}h"

            print(f"{'─' * 60}")
            print(f"Caching complete in {time_str}")
            print(f"  Samples: {len(activations):,}")
            print(f"  Activation dim: {activations.shape[1]}")
            print(f"  Input dim: {inputs.shape[1]}")
            print(f"{'─' * 60}\n")

        return self._cached

    def fill_with_loss_stratification(
        self,
        dataloader: DataLoader,
        colmap: "ColumnMap",
        max_samples: int = 100000,
        num_bins: int = 10,
        high_loss_factor: float = 3.0,
        show_progress: bool = True,
    ) -> CachedActivations:
        """
        Fill cache with loss-stratified sampling to oversample uncertain frames.

        This helps SAEs learn features for rare/interesting game situations
        rather than just common "hold neutral" frames.

        Args:
            dataloader: DataLoader yielding batches with "X" and "Y" keys
            colmap: Column mapping for building model inputs
            max_samples: Maximum number of activation samples to collect
            num_bins: Number of loss bins for stratification
            high_loss_factor: Factor to oversample high-loss frames
            show_progress: Whether to print progress

        Returns:
            CachedActivations object with stratified data
        """
        import time

        from controller_quantization import quantize_targets
        from loss import compute_loss_components
        from train.batch_utils import build_model_inputs

        self.model.eval()
        self._hook_manager.install_hooks([self.hook_point])

        # First pass: collect all activations with their losses
        all_data: List[Tuple[Tensor, Tensor, float]] = []  # (acts, inputs, loss)
        total_collected = 0
        target_collect = max_samples * 2  # Collect 2x for stratification
        start_time = time.time()

        if show_progress:
            print(f"\n{'─' * 60}")
            print(f"Caching activations with loss stratification")
            print(f"{'─' * 60}")
            print(f"  Hook point: {self.hook_point}")
            print(f"  Target samples: {max_samples:,} (collecting {target_collect:,} for stratification)")
            print(f"  Loss bins: {num_bins}, high-loss oversample: {high_loss_factor}x")

        # Get loss config from model config if available
        try:
            from config import get_config

            loss_config = get_config().loss_weights
        except Exception:
            # Fallback to default loss config
            from config.loss_config import LossConfig

            loss_config = LossConfig()

        try:
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    X = batch["X"].to(self.device)
                    Y = batch["Y"].to(self.device)
                    batch_size, seq_len, _ = X.shape

                    # Build model inputs and run forward pass
                    exclude_p1 = getattr(self.model.config.model, 'exclude_p1_controller', False)
                    inputs_td = build_model_inputs(X, colmap, exclude_p1_controller=exclude_p1)
                    outputs = self.model(inputs_td)

                    # Compute per-frame loss
                    target_info = quantize_targets(Y, colmap)
                    losses = compute_loss_components(
                        outputs,
                        target_info,
                        label_smoothing=0.0,
                        loss_config=loss_config,
                    )
                    frame_loss = losses["total"].item()

                    # Get activations
                    acts = self._hook_manager.get_single(self.hook_point)
                    acts_flat = acts.view(-1, acts.shape[-1]).cpu()
                    inputs_flat = X.view(-1, X.shape[-1]).cpu()

                    # Store with loss
                    for i in range(acts_flat.shape[0]):
                        all_data.append((acts_flat[i], inputs_flat[i], frame_loss))

                    self._hook_manager.clear()
                    total_collected += acts_flat.shape[0]

                    if show_progress and batch_idx % 10 == 0:
                        elapsed = time.time() - start_time
                        samples_per_sec = total_collected / elapsed if elapsed > 0 else 0
                        remaining = target_collect - total_collected
                        eta_sec = remaining / samples_per_sec if samples_per_sec > 0 else 0
                        pct = 100 * total_collected / target_collect

                        # Format ETA
                        if eta_sec < 60:
                            eta_str = f"{eta_sec:.0f}s"
                        elif eta_sec < 3600:
                            eta_str = f"{eta_sec / 60:.1f}m"
                        else:
                            eta_str = f"{eta_sec / 3600:.1f}h"

                        print(
                            f"  [{pct:5.1f}%] {total_collected:,}/{target_collect:,} samples | "
                            f"{samples_per_sec:,.0f} samples/s | ETA: {eta_str}"
                        )

                    # Collect more than we need for stratification
                    if total_collected >= target_collect:
                        break

        finally:
            self._hook_manager.remove_hooks()

        # Stratified sampling based on loss
        if show_progress:
            print(f"\n  Stratifying {len(all_data):,} samples into {num_bins} bins...")

        # Sort by loss and bin
        all_data.sort(key=lambda x: x[2])
        bin_size = len(all_data) // num_bins

        # Sample with higher probability for high-loss bins
        sampled_acts = []
        sampled_inputs = []

        for bin_idx in range(num_bins):
            start = bin_idx * bin_size
            end = start + bin_size if bin_idx < num_bins - 1 else len(all_data)
            bin_data = all_data[start:end]

            # Higher bins (higher loss) get more samples
            weight = 1.0 + (high_loss_factor - 1.0) * (bin_idx / (num_bins - 1))
            n_samples = int(max_samples / num_bins * weight)
            n_samples = min(n_samples, len(bin_data))

            # Random sample from this bin
            indices = torch.randperm(len(bin_data))[:n_samples]
            for idx in indices:
                sampled_acts.append(bin_data[idx][0])
                sampled_inputs.append(bin_data[idx][1])

            if show_progress:
                bin_loss_min = bin_data[0][2]
                bin_loss_max = bin_data[-1][2]
                print(
                    f"    Bin {bin_idx + 1}/{num_bins}: loss [{bin_loss_min:.4f}, {bin_loss_max:.4f}] "
                    f"-> {n_samples:,} samples (weight {weight:.2f}x)"
                )

        activations = torch.stack(sampled_acts)[:max_samples]
        inputs = torch.stack(sampled_inputs)[:max_samples]

        # Compute normalization statistics
        mean = activations.mean(dim=0)
        std = activations.std(dim=0)

        # Shuffle
        perm = torch.randperm(len(activations))
        activations = activations[perm]
        inputs = inputs[perm]

        self._cached = CachedActivations(
            activations=activations,
            inputs=inputs,
            mean=mean,
            std=std,
        )

        if show_progress:
            elapsed_total = time.time() - start_time
            if elapsed_total < 60:
                time_str = f"{elapsed_total:.1f}s"
            elif elapsed_total < 3600:
                time_str = f"{elapsed_total / 60:.1f}m"
            else:
                time_str = f"{elapsed_total / 3600:.1f}h"

            print(f"{'─' * 60}")
            print(f"Stratified caching complete in {time_str}")
            print(f"  Final samples: {len(activations):,}")
            print(f"  Activation dim: {activations.shape[1]}")
            print(f"  Input dim: {inputs.shape[1]}")
            print(f"{'─' * 60}\n")

        return self._cached

    def get_cached(self) -> CachedActivations:
        """Get the cached activations, raising if not filled."""
        if self._cached is None:
            raise RuntimeError("Cache not filled. Call fill_from_dataloader first.")
        return self._cached

    @property
    def is_filled(self) -> bool:
        """Whether the cache has been filled."""
        return self._cached is not None

    def clear(self) -> None:
        """Clear the cached activations."""
        self._cached = None


__all__ = ["ActivationCache", "CachedActivations"]
