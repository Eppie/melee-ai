"""
Activation caching with metadata for interpretability analysis.

Extends the hook system to store activations along with corresponding
input features for correlation analysis and feature interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
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
        temperature: float = 1.0,
        show_progress: bool = True,
    ) -> CachedActivations:
        """
        Fill cache with inverse-density stratified sampling to oversample rare frames.

        Uses kernel density estimation (KDE) on the loss distribution to compute
        importance weights. Frames with rare loss values (typically high-loss,
        interesting situations) get upweighted automatically.

        Args:
            dataloader: DataLoader yielding batches with "X" and "Y" keys
            colmap: Column mapping for building model inputs
            max_samples: Maximum number of activation samples to collect
            temperature: Controls strength of inverse-density weighting.
                         1.0 = pure inverse density, <1 = gentler, >1 = more aggressive
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

        # First pass: collect all activations with their losses (vectorized)
        all_activations: List[Tensor] = []
        all_inputs: List[Tensor] = []
        all_losses: List[Tensor] = []
        total_collected = 0
        target_collect = max_samples * 2  # Collect 2x for stratification
        start_time = time.time()

        if show_progress:
            print(f"\n{'─' * 60}")
            print(f"Caching activations with inverse-density stratification")
            print(f"{'─' * 60}")
            print(f"  Hook point: {self.hook_point}")
            print(f"  Target samples: {max_samples:,} (collecting {target_collect:,} for weighting)")
            print(f"  Temperature: {temperature}")

        # Get loss config from model config if available
        try:
            from config import get_config
            loss_config = get_config().loss_weights
        except Exception:
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

                    # Compute batch loss (used as proxy for all frames in batch)
                    target_info = quantize_targets(Y, colmap)
                    losses = compute_loss_components(
                        outputs,
                        target_info,
                        label_smoothing=0.0,
                        loss_config=loss_config,
                    )
                    batch_loss = losses["total"].item()

                    # Get activations
                    acts = self._hook_manager.get_single(self.hook_point)
                    acts_flat = acts.view(-1, acts.shape[-1]).cpu()
                    inputs_flat = X.view(-1, X.shape[-1]).cpu()
                    n_frames = acts_flat.shape[0]

                    # Store tensors (vectorized, not individual tuples)
                    all_activations.append(acts_flat)
                    all_inputs.append(inputs_flat)
                    all_losses.append(torch.full((n_frames,), batch_loss))

                    self._hook_manager.clear()
                    total_collected += n_frames

                    if show_progress and batch_idx % 10 == 0:
                        elapsed = time.time() - start_time
                        samples_per_sec = total_collected / elapsed if elapsed > 0 else 0
                        remaining = target_collect - total_collected
                        eta_sec = remaining / samples_per_sec if samples_per_sec > 0 else 0
                        pct = 100 * total_collected / target_collect

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

                    if total_collected >= target_collect:
                        break

        finally:
            self._hook_manager.remove_hooks()

        # Concatenate all collected data
        activations = torch.cat(all_activations, dim=0)
        inputs = torch.cat(all_inputs, dim=0)
        losses_tensor = torch.cat(all_losses, dim=0)

        if show_progress:
            print(f"\n  Computing inverse-density weights via histogram...")

        # Compute inverse-density weights using histogram (much faster than KDE)
        losses_np = losses_tensor.numpy()
        n_samples = len(losses_np)

        # Use Sturges' rule for number of bins, with a reasonable cap
        n_bins = min(int(np.ceil(np.log2(n_samples)) + 1) * 10, 500)

        # Compute histogram
        hist, bin_edges = np.histogram(losses_np, bins=n_bins)

        # Assign each sample to a bin
        bin_indices = np.digitize(losses_np, bin_edges[:-1]) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        # Compute density as normalized bin counts
        density = hist[bin_indices].astype(np.float64)
        density = density / density.sum()  # Normalize to probability

        # Inverse density weighting (rare = high weight)
        weights = 1.0 / (density + 1e-10)

        # Apply temperature
        weights = weights ** temperature

        # Normalize to probabilities
        weights = weights / weights.sum()

        if show_progress:
            # Show weight statistics
            effective_weight_ratio = weights.max() / weights.min()
            print(f"  Loss range: [{losses_np.min():.4f}, {losses_np.max():.4f}]")
            print(f"  Bins: {n_bins}, effective weight ratio: {effective_weight_ratio:.1f}x")

        # Weighted sampling without replacement
        if show_progress:
            print(f"  Sampling {max_samples:,} frames with importance weights...")

        # Use torch multinomial for weighted sampling
        weights_tensor = torch.from_numpy(weights).float()
        selected_indices = torch.multinomial(weights_tensor, min(max_samples, len(weights_tensor)), replacement=False)

        activations = activations[selected_indices]
        inputs = inputs[selected_indices]

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


class StreamingActivations:
    """
    Memory-efficient streaming activations for training on datasets larger than RAM.

    Uses:
    - Memory-mapped file access (no full load into RAM)
    - Shuffle buffer for randomization without loading everything
    - Background prefetching to hide I/O latency
    - Sequential disk reads (fast) instead of random access (slow)
    - Optional stratified sampling via pre-computed sorted indices

    Maintains the same interface as CachedActivations for drop-in replacement.

    Usage:
        streaming = StreamingActivations.from_cache_dir(cache_dir, layer_idx)
        for step in range(training_steps):
            batch, _ = streaming.sample_batch(batch_size)
            # train on batch...

        # With stratified sampling:
        streaming = StreamingActivations.from_cache_dir(
            cache_dir, layer_idx, stratified=True, stratified_temperature=1.0
        )
    """

    def __init__(
        self,
        activations_path: Path,
        mean: Tensor,
        std: Tensor,
        buffer_size: int = 500_000,
        n_prefetch: int = 2,
        normalize: bool = True,
        stratified_indices: Optional[Tensor] = None,
    ):
        """
        Args:
            activations_path: Path to activations.pt file
            mean: Mean for normalization [D]
            std: Std for normalization [D]
            buffer_size: Number of samples to keep in shuffle buffer
            n_prefetch: Number of buffers to prefetch in background
            normalize: Whether to normalize activations
            stratified_indices: Pre-computed sorted indices for stratified sampling
        """
        import queue
        import threading

        self.activations_path = activations_path
        self.mean = mean
        self.std = std
        self.buffer_size = buffer_size
        self.normalize = normalize

        # Memory-map the activations file
        # Note: torch.load with mmap=True requires PyTorch 2.0+
        self._mmap_data = torch.load(activations_path, map_location="cpu", mmap=True)
        self._total_samples = len(self._mmap_data)
        self.activation_dim = self._mmap_data.shape[1]

        # Stratified sampling state
        self._stratified_indices = stratified_indices
        self._stratified_pos = 0
        self._use_stratified = stratified_indices is not None

        # n_samples reflects actual available samples (stratified subset or all)
        if self._use_stratified:
            self.n_samples = len(stratified_indices)
        else:
            self.n_samples = self._total_samples

        # Current buffer state
        self._buffer: Optional[Tensor] = None
        self._buffer_pos = 0

        # Prefetch queue and thread
        self._prefetch_queue: queue.Queue = queue.Queue(maxsize=n_prefetch)
        self._stop_prefetch = threading.Event()
        self._prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self._prefetch_thread.start()

        # Load first buffer
        self._load_next_buffer()

    @staticmethod
    def _compute_stratified_indices(
        losses: Tensor,
        n_samples: Optional[int] = None,
        temperature: float = 1.0,
    ) -> Tensor:
        """
        Compute stratified sample indices from losses using inverse-density weighting.

        Args:
            losses: Per-sample loss values [total_samples]
            n_samples: Number of indices to select (None = use all)
            temperature: Weight temperature (1.0 = pure inverse density)

        Returns:
            Sorted indices tensor for sequential access
        """
        losses_np = losses.numpy()
        n_total = len(losses_np)
        n_select = n_samples if n_samples else n_total
        n_select = min(n_select, n_total)

        # Histogram-based density estimation (fast)
        n_bins = min(int(np.ceil(np.log2(n_total)) + 1) * 10, 500)
        hist, bin_edges = np.histogram(losses_np, bins=n_bins)
        bin_indices = np.digitize(losses_np, bin_edges[:-1]) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        # Inverse-density weights
        density = hist[bin_indices].astype(np.float64)
        density = density / density.sum()
        weights = 1.0 / (density + 1e-10)
        weights = weights ** temperature
        weights = weights / weights.sum()

        # Weighted sampling
        weights_tensor = torch.from_numpy(weights).float()
        selected = torch.multinomial(weights_tensor, n_select, replacement=False)

        # Sort for sequential access (critical for mmap performance)
        sorted_indices = selected.sort().values

        return sorted_indices

    def _load_buffer_from_disk(self) -> Tensor:
        """Load a shuffled buffer from disk using sequential reads."""
        import random

        if self._use_stratified:
            # Stratified mode: read from pre-computed sorted indices
            remaining = len(self._stratified_indices) - self._stratified_pos
            chunk_size = min(self.buffer_size, remaining)

            if chunk_size <= 0:
                # Wrap around - reshuffle the stratified indices for next epoch
                perm = torch.randperm(len(self._stratified_indices))
                self._stratified_indices = self._stratified_indices[perm].sort().values
                self._stratified_pos = 0
                chunk_size = min(self.buffer_size, len(self._stratified_indices))

            # Get next chunk of sorted indices
            idx_chunk = self._stratified_indices[self._stratified_pos:self._stratified_pos + chunk_size]
            self._stratified_pos += chunk_size

            # Read at sorted indices - mostly sequential access pattern
            buffer = self._mmap_data[idx_chunk].clone()
        else:
            # Uniform mode: sample from multiple random positions for better coverage
            n_chunks = min(10, self._total_samples // (self.buffer_size // 10 + 1))
            n_chunks = max(1, n_chunks)
            chunk_size = self.buffer_size // n_chunks

            chunks = []
            for _ in range(n_chunks):
                # Random starting position
                max_start = max(0, self._total_samples - chunk_size)
                start = random.randint(0, max_start)
                end = min(start + chunk_size, self._total_samples)

                # Sequential read (fast even for mmap)
                chunk = self._mmap_data[start:end].clone()
                chunks.append(chunk)

            # Concatenate
            buffer = torch.cat(chunks, dim=0)

        # Apply normalization if requested
        if self.normalize:
            buffer = (buffer - self.mean) / (self.std + 1e-8)

        # Shuffle the buffer (important for training regardless of mode)
        perm = torch.randperm(len(buffer))
        buffer = buffer[perm]

        return buffer

    def _prefetch_worker(self) -> None:
        """Background thread that prefetches buffers."""
        while not self._stop_prefetch.is_set():
            try:
                buffer = self._load_buffer_from_disk()
                self._prefetch_queue.put(buffer, timeout=1.0)
            except Exception:
                # Queue full or other error, just continue
                pass

    def _load_next_buffer(self) -> None:
        """Load the next buffer from prefetch queue."""
        try:
            self._buffer = self._prefetch_queue.get(timeout=30.0)
            self._buffer_pos = 0
        except Exception as e:
            # Fallback to direct load if prefetch fails
            self._buffer = self._load_buffer_from_disk()
            self._buffer_pos = 0

    def sample_batch(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        """
        Sample a batch of activations.

        Returns:
            Tuple of (activations, inputs) - note: inputs are zeros for streaming
            (we don't store inputs to save memory/disk space)
        """
        # Check if we need a new buffer
        if self._buffer is None or self._buffer_pos + batch_size > len(self._buffer):
            self._load_next_buffer()

        # Get batch from buffer
        batch = self._buffer[self._buffer_pos : self._buffer_pos + batch_size]
        self._buffer_pos += batch_size

        # Return dummy inputs (not available in streaming mode)
        dummy_inputs = torch.zeros(batch_size, 1)

        return batch, dummy_inputs

    @property
    def normalized(self) -> "StreamingActivations":
        """Return self (normalization is applied during sampling)."""
        return self

    def to(self, device: torch.device) -> "StreamingActivations":
        """Move normalization stats to device (data stays on CPU until sampled)."""
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def close(self) -> None:
        """Stop background prefetching."""
        self._stop_prefetch.set()
        self._prefetch_thread.join(timeout=5.0)

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except Exception:
            pass

    @classmethod
    def from_cache_dir(
        cls,
        cache_dir: Path,
        hook_type: str,
        layer_idx: int,
        buffer_size: int = 500_000,
        normalize: bool = True,
        stratified: bool = False,
        stratified_temperature: float = 1.0,
        stratified_n_samples: Optional[int] = None,
    ) -> "StreamingActivations":
        """
        Create StreamingActivations from a cache directory.

        Args:
            cache_dir: Directory containing cached activations
            hook_type: Hook type (e.g., "block_output")
            layer_idx: Layer index
            buffer_size: Shuffle buffer size
            normalize: Whether to normalize activations
            stratified: Whether to use stratified sampling (requires losses.pt)
            stratified_temperature: Temperature for inverse-density weighting
            stratified_n_samples: Number of samples to select (None = use all)
        """
        layer_dir = cache_dir / f"{hook_type}_L{layer_idx}"

        if not layer_dir.exists():
            raise FileNotFoundError(f"Cache not found: {layer_dir}")

        activations_path = layer_dir / "activations.pt"
        mean = torch.load(layer_dir / "mean.pt", map_location="cpu")
        std = torch.load(layer_dir / "std.pt", map_location="cpu")

        # Compute stratified indices if requested
        stratified_indices = None
        if stratified:
            losses_path = layer_dir / "losses.pt"
            if not losses_path.exists():
                raise FileNotFoundError(
                    f"Stratified sampling requires losses.pt in cache. "
                    f"Re-run cache_activations.py with --compute-loss"
                )

            print(f"  Loading losses for stratification...")
            losses = torch.load(losses_path, map_location="cpu")
            n_total = len(losses)

            print(f"  Computing stratified indices (temperature={stratified_temperature})...")
            stratified_indices = cls._compute_stratified_indices(
                losses,
                n_samples=stratified_n_samples,
                temperature=stratified_temperature,
            )

            n_selected = len(stratified_indices)
            print(f"  Selected {n_selected:,} / {n_total:,} samples ({100*n_selected/n_total:.1f}%)")

        return cls(
            activations_path=activations_path,
            mean=mean,
            std=std,
            buffer_size=buffer_size,
            normalize=normalize,
            stratified_indices=stratified_indices,
        )


__all__ = ["ActivationCache", "CachedActivations", "StreamingActivations"]
