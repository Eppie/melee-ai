"""Async GPU-CPU transfer infrastructure using pinned memory.

This module provides utilities for non-blocking GPU to CPU transfers
to eliminate synchronization overhead.
"""

from __future__ import annotations

from typing import Optional

import torch


class PinnedMemoryPool:
    """Manages pinned (page-locked) host memory buffers for async GPU-CPU transfers.

    Pinned memory allows async DMA transfers without blocking the CPU or GPU.
    This eliminates the 130ms+ overhead from pageable memory transfers.

    Usage:
        pool = PinnedMemoryPool()
        buffer = pool.get_scalar_buffer()

        # Non-blocking transfer into pinned buffer
        buffer.copy_(tensor, non_blocking=True)

        # Later, when you need the value:
        torch.cuda.synchronize()  # Only sync when actually needed
        value = buffer.item()
    """

    def __init__(self):
        """Initialize the pinned memory pool."""
        # Scalar buffers for single values (loss, accuracy, etc.)
        self.scalar_f32 = torch.empty(1, dtype=torch.float32, pin_memory=True)
        self.scalar_f64 = torch.empty(1, dtype=torch.float64, pin_memory=True)
        self.scalar_i64 = torch.empty(1, dtype=torch.int64, pin_memory=True)

        # Small buffers for stats vectors (min, max, mean, std)
        self.stats_buffer_small = torch.empty(32, dtype=torch.float32, pin_memory=True)
        self.stats_buffer_medium = torch.empty(256, dtype=torch.float32, pin_memory=True)
        self.stats_buffer_large = torch.empty(2048, dtype=torch.float32, pin_memory=True)

        # Stream for async transfers (separate from compute stream)
        self.transfer_stream: Optional[torch.cuda.Stream] = None
        if torch.cuda.is_available():
            self.transfer_stream = torch.cuda.Stream()

    def get_scalar_buffer(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Get a pinned memory buffer for a single scalar value.

        Args:
            dtype: Data type of the scalar.

        Returns:
            Pinned memory tensor that can be used as output buffer for .cpu()
        """
        if dtype == torch.float32:
            return self.scalar_f32
        elif dtype == torch.float64:
            return self.scalar_f64
        elif dtype == torch.int64:
            return self.scalar_i64
        else:
            # Fallback: create new buffer (will be slower but correct)
            return torch.empty(1, dtype=dtype, pin_memory=True)

    def get_stats_buffer(self, size: int) -> torch.Tensor:
        """Get a pinned memory buffer for statistics arrays.

        Args:
            size: Number of elements needed.

        Returns:
            Pinned memory tensor of appropriate size.
        """
        if size <= 32:
            return self.stats_buffer_small[:size]
        elif size <= 256:
            return self.stats_buffer_medium[:size]
        elif size <= 2048:
            return self.stats_buffer_large[:size]
        else:
            # Large transfer - create dedicated buffer
            return torch.empty(size, dtype=torch.float32, pin_memory=True)

    def transfer_scalar_async(
        self,
        gpu_tensor: torch.Tensor,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Transfer a scalar from GPU to CPU asynchronously.

        This returns immediately. Call .item() on the result later after
        ensuring the transfer is complete (via torch.cuda.synchronize() or
        waiting until logging time).

        Args:
            gpu_tensor: GPU tensor to transfer (should be 0-d or single element).
            dtype: Target dtype.

        Returns:
            Pinned memory buffer containing the result (may not be ready yet!).
        """
        buffer = self.get_scalar_buffer(dtype)

        if self.transfer_stream is not None:
            with torch.cuda.stream(self.transfer_stream):
                buffer.copy_(gpu_tensor, non_blocking=True)
        else:
            buffer.copy_(gpu_tensor, non_blocking=True)

        return buffer

    def transfer_tensor_async(
        self,
        gpu_tensor: torch.Tensor,
        buffer: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Transfer a tensor from GPU to CPU asynchronously.

        Args:
            gpu_tensor: GPU tensor to transfer.
            buffer: Optional pinned memory buffer. If None, creates one.

        Returns:
            Pinned memory buffer containing the result (may not be ready yet!).
        """
        if buffer is None:
            buffer = torch.empty(
                gpu_tensor.shape,
                dtype=gpu_tensor.dtype,
                pin_memory=True
            )

        if self.transfer_stream is not None:
            with torch.cuda.stream(self.transfer_stream):
                buffer.copy_(gpu_tensor, non_blocking=True)
        else:
            buffer.copy_(gpu_tensor, non_blocking=True)

        return buffer

    def synchronize(self):
        """Wait for all async transfers to complete.

        Only call this when you actually need to read the transferred values.
        """
        if self.transfer_stream is not None:
            self.transfer_stream.synchronize()
        else:
            torch.cuda.synchronize()


class DeferredScalarAccumulator:
    """Accumulates scalar values on GPU, only transferring to CPU when needed.

    This is designed to replace patterns like:
        for step in range(1000):
            loss = model(batch)
            loss_value = loss.cpu().item()  # SYNC! (bad)
            tracker.add(loss_value)

    With:
        accumulator = DeferredScalarAccumulator()
        for step in range(1000):
            loss = model(batch)
            accumulator.add(loss)  # No sync!

        # Only sync when logging:
        if should_log:
            values = accumulator.get_and_reset()  # One sync for all values
    """

    def __init__(self, max_size: int = 1000, device: Optional[torch.device] = None):
        """Initialize the accumulator.

        Args:
            max_size: Maximum number of values to accumulate before forcing a transfer.
            device: Device to store accumulated values (defaults to CUDA if available).
        """
        self.max_size = max_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.values: list[torch.Tensor] = []
        self.pinned_buffer = torch.empty(max_size, dtype=torch.float32, pin_memory=True)

    def add(self, value: torch.Tensor):
        """Add a scalar value to the accumulator (keeps on GPU).

        Args:
            value: Scalar tensor (0-d or single element).
        """
        # Detach and keep on GPU
        if value.numel() == 1:
            self.values.append(value.detach().reshape(()))
        else:
            raise ValueError(f"Expected scalar tensor, got shape {value.shape}")

        # Auto-flush if we hit max size
        if len(self.values) >= self.max_size:
            return self.get_and_reset()

        return None

    def get_and_reset(self) -> list[float]:
        """Transfer all accumulated values to CPU and reset.

        This performs a single batched transfer for all accumulated values,
        which is much more efficient than transferring each individually.

        Returns:
            List of float values.
        """
        if not self.values:
            return []

        # Stack all values into single tensor
        stacked = torch.stack(self.values)

        # Transfer via pinned memory
        buffer = self.pinned_buffer[:len(self.values)]
        buffer.copy_(stacked, non_blocking=False)  # Blocking transfer for simplicity

        # Convert to Python list
        result = buffer.tolist()

        # Reset
        self.values.clear()

        return result

    def __len__(self) -> int:
        return len(self.values)


class BatchedStatsTransfer:
    """Helper for transferring multiple statistics tensors in a single operation.

    Instead of:
        stat1 = tensor1.cpu().item()  # Sync 1
        stat2 = tensor2.cpu().item()  # Sync 2
        stat3 = tensor3.cpu().item()  # Sync 3

    Use:
        transfer = BatchedStatsTransfer()
        transfer.add('stat1', tensor1)
        transfer.add('stat2', tensor2)
        transfer.add('stat3', tensor3)
        results = transfer.execute()  # Single sync
    """

    def __init__(self, pool: Optional[PinnedMemoryPool] = None):
        """Initialize the batched transfer helper.

        Args:
            pool: Optional pinned memory pool. Creates one if not provided.
        """
        self.pool = pool or PinnedMemoryPool()
        self.tensors: dict[str, torch.Tensor] = {}

    def add(self, name: str, tensor: torch.Tensor):
        """Add a tensor to be transferred.

        Args:
            name: Key for the result dictionary.
            tensor: Tensor to transfer (will be detached).
        """
        self.tensors[name] = tensor.detach()

    def execute(self, non_blocking: bool = False) -> dict[str, float | list]:
        """Execute the batched transfer.

        Args:
            non_blocking: If True, transfer is async (caller must sync later).

        Returns:
            Dictionary mapping names to transferred values.
        """
        if not self.tensors:
            return {}

        results = {}

        # Group by size for efficient batching
        scalars = {}
        arrays = {}

        for name, tensor in self.tensors.items():
            if tensor.numel() == 1:
                scalars[name] = tensor
            else:
                arrays[name] = tensor

        # Transfer scalars in batch
        if scalars:
            scalar_names = list(scalars.keys())
            scalar_tensors = [scalars[name] for name in scalar_names]
            stacked = torch.stack([t.reshape(()) for t in scalar_tensors])

            buffer = self.pool.get_stats_buffer(len(scalar_tensors))
            buffer.copy_(stacked, non_blocking=non_blocking)

            if not non_blocking:
                for i, name in enumerate(scalar_names):
                    results[name] = float(buffer[i].item())

        # Transfer arrays individually (less common)
        for name, tensor in arrays.items():
            buffer = self.pool.get_stats_buffer(tensor.numel())
            flat = tensor.flatten()
            buffer[:flat.numel()].copy_(flat, non_blocking=non_blocking)

            if not non_blocking:
                results[name] = buffer[:flat.numel()].tolist()

        self.tensors.clear()
        return results
