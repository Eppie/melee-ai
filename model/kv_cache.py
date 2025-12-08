"""
KV Cache implementation for efficient autoregressive inference.

The KV Cache stores keys and values from previous forward passes to avoid
recomputing them during autoregressive generation. This is particularly
effective with ALiBi positional encoding.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor


@dataclass
class LayerKVCache:
    """
    Stores cached keys and values for a single transformer layer.

    Attributes
    ----------
    keys : Tensor
        Cached key states. Shape: [batch_size, num_kv_heads, seq_len, head_dim]
    values : Tensor
        Cached value states. Shape: [batch_size, num_kv_heads, seq_len, head_dim]
    """
    keys: Optional[Tensor] = None
    values: Optional[Tensor] = None

    def update(
        self,
        new_keys: Tensor,
        new_values: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Update the cache with new key-value pairs.

        For autoregressive generation:
        - If cache is empty, initialize with new_keys and new_values
        - If cache exists, concatenate along sequence dimension

        Parameters
        ----------
        new_keys : Tensor
            New key states. Shape: [batch_size, num_kv_heads, new_seq_len, head_dim]
        new_values : Tensor
            New value states. Shape: [batch_size, num_kv_heads, new_seq_len, head_dim]

        Returns
        -------
        keys : Tensor
            Updated keys (cached + new). Shape: [batch_size, num_kv_heads, total_seq_len, head_dim]
        values : Tensor
            Updated values (cached + new). Shape: [batch_size, num_kv_heads, total_seq_len, head_dim]
        """
        if self.keys is None or self.values is None:
            # First forward pass: initialize cache
            self.keys = new_keys
            self.values = new_values
        else:
            # Subsequent passes: concatenate along sequence dimension
            self.keys = torch.cat([self.keys, new_keys], dim=2)
            self.values = torch.cat([self.values, new_values], dim=2)

        return self.keys, self.values

    def clear(self) -> None:
        """Clear the cache."""
        self.keys = None
        self.values = None

    @property
    def is_empty(self) -> bool:
        """Check if cache is empty."""
        return self.keys is None or self.values is None

    @property
    def seq_len(self) -> int:
        """Get the current sequence length in the cache."""
        if self.is_empty:
            return 0
        return self.keys.shape[2]


class KVCache:
    """
    Multi-layer KV cache for transformer models.

    This cache stores keys and values for each transformer layer,
    enabling efficient autoregressive generation by avoiding recomputation
    of attention keys and values for previous tokens.

    The cache is designed to work with ALiBi positional encoding.
    For RoPE models, the cache should be disabled as RoPE requires
    careful handling of positional encodings during caching.

    Parameters
    ----------
    num_layers : int
        Number of transformer layers
    enabled : bool, optional
        Whether the cache is active. Default: True
        Should be False for RoPE models, True for ALiBi models.

    Example
    -------
    >>> # Create cache for 8-layer model
    >>> cache = KVCache(num_layers=8, enabled=True)
    >>>
    >>> # During forward pass for layer 0
    >>> keys, values = cache.update(layer_idx=0, new_keys=k, new_values=v)
    >>>
    >>> # Reset cache for new sequence
    >>> cache.clear()
    """

    def __init__(self, num_layers: int, enabled: bool = True):
        self.num_layers = num_layers
        self.enabled = enabled
        self._layer_caches = [LayerKVCache() for _ in range(num_layers)]

    def update(
        self,
        layer_idx: int,
        new_keys: Tensor,
        new_values: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Update cache for a specific layer.

        Parameters
        ----------
        layer_idx : int
            Index of the transformer layer (0-indexed)
        new_keys : Tensor
            New key states. Shape: [batch_size, num_kv_heads, seq_len, head_dim]
        new_values : Tensor
            New value states. Shape: [batch_size, num_kv_heads, seq_len, head_dim]

        Returns
        -------
        keys : Tensor
            Full key states (cached + new)
        values : Tensor
            Full value states (cached + new)
        """
        if not self.enabled:
            # Cache disabled: return new keys/values as-is
            return new_keys, new_values

        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(
                f"layer_idx {layer_idx} out of range [0, {self.num_layers})"
            )

        return self._layer_caches[layer_idx].update(new_keys, new_values)

    def get(self, layer_idx: int) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """
        Get cached keys and values for a specific layer.

        Parameters
        ----------
        layer_idx : int
            Index of the transformer layer (0-indexed)

        Returns
        -------
        keys : Tensor or None
            Cached keys if available, None otherwise
        values : Tensor or None
            Cached values if available, None otherwise
        """
        if not self.enabled:
            return None, None

        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(
                f"layer_idx {layer_idx} out of range [0, {self.num_layers})"
            )

        cache = self._layer_caches[layer_idx]
        return cache.keys, cache.values

    def clear(self) -> None:
        """Clear all layer caches."""
        for cache in self._layer_caches:
            cache.clear()

    @property
    def is_empty(self) -> bool:
        """Check if all layer caches are empty."""
        return all(cache.is_empty for cache in self._layer_caches)

    @property
    def seq_len(self) -> int:
        """
        Get the current sequence length in the cache.

        Returns 0 if cache is empty, otherwise returns the sequence
        length from the first layer (all layers should have same seq_len).
        """
        if self.is_empty:
            return 0
        return self._layer_caches[0].seq_len

    def __repr__(self) -> str:
        status = "enabled" if self.enabled else "disabled"
        return (
            f"KVCache(num_layers={self.num_layers}, "
            f"enabled={self.enabled}, "
            f"seq_len={self.seq_len}, "
            f"status={status})"
        )
