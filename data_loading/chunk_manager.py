"""ChunkManager: Centralized management of multi-chunk overlap for data loading."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

if TYPE_CHECKING:
    from window_dataset import ZarrCorpusIndex


class ChunkManager:
    """Manages loading and eviction of episode chunks with support for multi-chunk overlap.

    This class implements a multi-chunk overlap strategy where multiple chunks are loaded
    simultaneously to improve shuffling quality. For example, with num_overlapping=2:
    - Iteration 0: Load chunks [0, 1]
    - Iteration 1: Evict chunk 0, load chunk 2 → active chunks [1, 2]
    - Iteration 2: Evict chunk 1, load chunk 3 → active chunks [2, 3]

    This provides a larger shuffle window (e.g., 2000 episodes instead of 1000) while
    controlling memory usage.
    """

    def __init__(
        self,
        index: ZarrCorpusIndex,
        chunk_size: int,
        num_overlapping: int = 2,
        enable_background_load: bool = False,
    ):
        """Initialize the ChunkManager.

        Args:
            index: ZarrCorpusIndex for accessing episode metadata
            chunk_size: Number of episodes per chunk
            num_overlapping: Number of chunks to keep loaded simultaneously
            enable_background_load: If True, preload next chunk in background thread
        """
        self.index = index
        self.chunk_size = chunk_size
        self.num_overlapping = num_overlapping
        self.enable_background_load = enable_background_load

        # Calculate total chunks
        total_episodes = len(self.index.episodes)
        self.total_chunks = max(1, (total_episodes + chunk_size - 1) // chunk_size)

        # Shared memory cache: {(shard_id, episode_id): (features_tensor, targets_tensor)}
        self._shared_cache: Dict[
            Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]
        ] = {}

        # Track which chunks are currently loaded
        self._active_chunk_indices: Set[int] = set()

        # Background preloading
        self._preload_thread: Optional[threading.Thread] = None
        self._preload_lock = threading.Lock()
        self._preload_cache: Optional[
            Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]]
        ] = None
        self._preload_chunk_indices: Optional[List[int]] = None

    def _chunk_bounds(self, chunk_idx: int) -> Tuple[int, int]:
        """Calculate episode index range for a chunk.

        Args:
            chunk_idx: Zero-based chunk index

        Returns:
            (start_episode_idx, end_episode_idx) - end is exclusive
        """
        start = chunk_idx * self.chunk_size
        end = min(len(self.index.episodes), start + self.chunk_size)
        return start, end

    def _load_chunk_into_cache(
        self,
        chunk_idx: int,
        target_cache: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]],
    ) -> List[int]:
        """Load a single chunk's episodes into the specified cache.

        Args:
            chunk_idx: Chunk index to load
            target_cache: Dictionary to store loaded tensors

        Returns:
            List of episode indices that were loaded
        """
        start, end = self._chunk_bounds(chunk_idx)
        episode_indices = list(range(start, end))

        for epi in episode_indices:
            ep = self.index.episodes[epi]
            cache_key = (ep.shard_id, ep.episode_id)

            # Skip if already in cache
            if cache_key in target_cache:
                continue

            # Load episode arrays
            feature_array, target_array = self.index.open_episode_arrays(ep)

            # Convert to shared memory tensors
            f_tensor = torch.from_numpy(
                np.ascontiguousarray(feature_array[:], dtype=np.float32)
            ).share_memory_()
            t_tensor = torch.from_numpy(
                np.ascontiguousarray(target_array[:], dtype=np.float32)
            ).share_memory_()

            target_cache[cache_key] = (f_tensor, t_tensor)

        return episode_indices

    def load_chunks(self, chunk_indices: List[int]) -> np.ndarray:
        """Load multiple chunks and return combined episode indices.

        This method:
        1. Waits for any background preload to complete
        2. Evicts chunks no longer needed
        3. Loads new chunks into shared memory
        4. Kicks off background preload for next iteration (if enabled)

        Args:
            chunk_indices: List of chunk indices to load (typically overlapping, e.g., [0, 1])

        Returns:
            numpy array of episode indices across all loaded chunks
        """
        # Wait for background preload if active
        if self._preload_thread is not None:
            self._preload_thread.join(timeout=30.0)
            if self._preload_thread.is_alive():
                print(
                    "[chunk_manager] WARNING: Background preload timed out, falling back to blocking load"
                )
                self._preload_thread = None
            else:
                # Transfer preloaded cache to main cache
                if self._preload_cache and set(self._preload_chunk_indices) == set(
                    chunk_indices
                ):
                    print(f"[chunk_manager] Using preloaded chunks {chunk_indices}")
                    # Merge preload cache into main cache
                    self._shared_cache.update(self._preload_cache)
                    self._preload_cache = None
                    self._preload_thread = None
                    self._active_chunk_indices = set(chunk_indices)
                    # Compute episode indices and return early
                    all_episode_indices = []
                    for chunk_idx in chunk_indices:
                        start, end = self._chunk_bounds(chunk_idx)
                        all_episode_indices.extend(range(start, end))

                    # Kick off next preload
                    self._maybe_start_background_preload(chunk_indices)
                    return np.array(all_episode_indices, dtype=np.int32)

        chunk_indices_set = set(chunk_indices)

        # Determine which chunks to evict (those not in the new set)
        chunks_to_evict = self._active_chunk_indices - chunk_indices_set

        # Evict old chunks
        if chunks_to_evict:
            evicted_keys = set()
            for chunk_idx in chunks_to_evict:
                start, end = self._chunk_bounds(chunk_idx)
                for epi in range(start, end):
                    ep = self.index.episodes[epi]
                    cache_key = (ep.shard_id, ep.episode_id)
                    evicted_keys.add(cache_key)

            # Remove from cache
            for key in evicted_keys:
                self._shared_cache.pop(key, None)

            print(
                f"[chunk_manager] Evicted {len(chunks_to_evict)} chunk(s): {sorted(chunks_to_evict)}"
            )

        # Determine which chunks need to be loaded
        chunks_to_load = chunk_indices_set - self._active_chunk_indices

        # Load new chunks
        if chunks_to_load:
            sorted_chunks = sorted(chunks_to_load)
            print(
                f"[chunk_manager] Loading {len(sorted_chunks)} new chunk(s): {sorted_chunks}"
            )

            for chunk_idx in sorted_chunks:
                start_time = time.time()
                start, end = self._chunk_bounds(chunk_idx)
                total_to_load = end - start
                print(
                    f"[chunk_manager]   Preloading chunk {chunk_idx + 1}/{self.total_chunks} "
                    f"episodes {start}-{end - 1} into shared memory..."
                )

                loaded_indices = self._load_chunk_into_cache(
                    chunk_idx, self._shared_cache
                )

                elapsed = time.time() - start_time
                print(
                    f"[chunk_manager]   Loaded {len(loaded_indices)} episodes in {elapsed:.2f}s "
                    f"({len(loaded_indices)/elapsed:.1f} eps/s)"
                )

        # Update active chunks
        self._active_chunk_indices = chunk_indices_set

        # Compute combined episode indices
        all_episode_indices = []
        for chunk_idx in chunk_indices:
            start, end = self._chunk_bounds(chunk_idx)
            all_episode_indices.extend(range(start, end))

        episode_array = np.array(all_episode_indices, dtype=np.int32)

        print(
            f"[chunk_manager] Active chunks: {sorted(self._active_chunk_indices)}, "
            f"total episodes: {len(episode_array)}"
        )

        # Kick off background preload for next iteration
        self._maybe_start_background_preload(chunk_indices)

        return episode_array

    def _maybe_start_background_preload(self, current_chunk_indices: List[int]) -> None:
        """Start background preloading of next chunk set if enabled.

        Args:
            current_chunk_indices: Currently loaded chunks (used to predict next chunks)
        """
        if not self.enable_background_load:
            return

        # Compute next chunk indices (rotate forward by 1)
        next_indices = []
        for idx in current_chunk_indices:
            next_idx = (idx + 1) % self.total_chunks
            # Only include if we're not wrapping around to chunks already loaded
            if next_idx not in current_chunk_indices:
                next_indices.append(next_idx)

        if not next_indices:
            return  # No new chunks to preload

        # Combine with chunks that will be retained
        retained_chunks = [
            idx for idx in current_chunk_indices if idx not in next_indices
        ]
        next_full_set = (
            retained_chunks[-(self.num_overlapping - len(next_indices)) :]
            + next_indices
        )

        if not next_full_set or len(next_full_set) > self.num_overlapping:
            return  # Invalid state

        print(f"[chunk_manager] Starting background preload for chunks {next_full_set}")

        self._preload_chunk_indices = next_full_set
        self._preload_cache = {}

        def _preload_worker():
            try:
                for chunk_idx in next_full_set:
                    if chunk_idx not in self._active_chunk_indices:
                        self._load_chunk_into_cache(chunk_idx, self._preload_cache)
                print(
                    f"[chunk_manager] Background preload complete for chunks {next_full_set}"
                )
            except Exception as exc:
                print(f"[chunk_manager] Background preload failed: {exc}")
                self._preload_cache = None

        self._preload_thread = threading.Thread(target=_preload_worker, daemon=True)
        self._preload_thread.start()

    def get_active_episodes(self) -> np.ndarray:
        """Return episode indices for all currently loaded chunks.

        Returns:
            numpy array of episode indices
        """
        all_episode_indices = []
        for chunk_idx in sorted(self._active_chunk_indices):
            start, end = self._chunk_bounds(chunk_idx)
            all_episode_indices.extend(range(start, end))
        return np.array(all_episode_indices, dtype=np.int32)

    def get_cache(self) -> Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]]:
        """Return the shared memory cache for use by WindowDataset.

        Returns:
            Dictionary mapping (shard_id, episode_id) -> (features_tensor, targets_tensor)
        """
        return self._shared_cache

    def shutdown(self) -> None:
        """Clean shutdown of background threads."""
        if self._preload_thread is not None and self._preload_thread.is_alive():
            print("[chunk_manager] Waiting for background preload to finish...")
            self._preload_thread.join(timeout=10.0)
            if self._preload_thread.is_alive():
                print(
                    "[chunk_manager] WARNING: Background thread did not shut down cleanly"
                )
