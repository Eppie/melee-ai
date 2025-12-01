"""Unit tests for ChunkManager."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from data_loading.chunk_manager import ChunkManager


@dataclass
class MockEpisodeInfo:
    """Mock episode info for testing."""

    episode_id: int
    shard_id: int
    num_frames: int
    num_windows: int


class MockZarrCorpusIndex:
    """Mock ZarrCorpusIndex for testing."""

    def __init__(self, num_episodes: int = 100, frames_per_episode: int = 1000):
        self.episodes = [
            MockEpisodeInfo(
                episode_id=i,
                shard_id=0,
                num_frames=frames_per_episode,
                num_windows=frames_per_episode - 255,  # Assuming seq_len=256
            )
            for i in range(num_episodes)
        ]

    def open_episode_arrays(self, ep: MockEpisodeInfo):
        """Return mock feature and target arrays."""
        # Create mock arrays with realistic sizes
        features = np.random.randn(ep.num_frames, 300).astype(np.float32)
        targets = np.random.randn(ep.num_frames, 50).astype(np.float32)
        return features, targets


class TestChunkManager:
    """Test ChunkManager functionality."""

    def test_initialization(self):
        """Test ChunkManager initialization."""
        index = MockZarrCorpusIndex(num_episodes=100)
        manager = ChunkManager(index=index, chunk_size=25, num_overlapping=2)

        assert manager.chunk_size == 25
        assert manager.num_overlapping == 2
        assert manager.total_chunks == 4  # 100 episodes / 25 per chunk = 4 chunks
        assert len(manager._shared_cache) == 0
        assert len(manager._active_chunk_indices) == 0

    def test_chunk_bounds(self):
        """Test chunk boundary calculation."""
        index = MockZarrCorpusIndex(num_episodes=100)
        manager = ChunkManager(index=index, chunk_size=25)

        # Test normal chunks
        assert manager._chunk_bounds(0) == (0, 25)
        assert manager._chunk_bounds(1) == (25, 50)
        assert manager._chunk_bounds(2) == (50, 75)
        assert manager._chunk_bounds(3) == (75, 100)

        # Test with non-divisible total
        index2 = MockZarrCorpusIndex(num_episodes=103)
        manager2 = ChunkManager(index=index2, chunk_size=25)
        assert manager2._chunk_bounds(4) == (100, 103)  # Last chunk is smaller

    def test_load_single_chunk(self):
        """Test loading a single chunk."""
        index = MockZarrCorpusIndex(num_episodes=100)
        manager = ChunkManager(index=index, chunk_size=25, num_overlapping=1)

        # Load first chunk
        episode_indices = manager.load_chunks([0])

        assert len(episode_indices) == 25
        assert np.array_equal(episode_indices, np.arange(0, 25))
        assert manager._active_chunk_indices == {0}
        assert len(manager._shared_cache) == 25

        # Verify tensors are in shared memory
        for cache_key, (f_tensor, t_tensor) in manager._shared_cache.items():
            assert isinstance(f_tensor, torch.Tensor)
            assert isinstance(t_tensor, torch.Tensor)
            assert f_tensor.is_shared()
            assert t_tensor.is_shared()

    def test_load_overlapping_chunks(self):
        """Test loading overlapping chunks."""
        index = MockZarrCorpusIndex(num_episodes=100)
        manager = ChunkManager(index=index, chunk_size=25, num_overlapping=2)

        # Load chunks [0, 1]
        episode_indices = manager.load_chunks([0, 1])

        assert len(episode_indices) == 50
        assert np.array_equal(episode_indices, np.arange(0, 50))
        assert manager._active_chunk_indices == {0, 1}
        assert len(manager._shared_cache) == 50

    def test_chunk_rotation(self):
        """Test rotating chunks with eviction."""
        index = MockZarrCorpusIndex(num_episodes=100)
        manager = ChunkManager(index=index, chunk_size=25, num_overlapping=2)

        # Load chunks [0, 1]
        manager.load_chunks([0, 1])
        initial_cache_size = len(manager._shared_cache)
        assert initial_cache_size == 50

        # Rotate to chunks [1, 2] - should evict chunk 0
        episode_indices = manager.load_chunks([1, 2])

        assert len(episode_indices) == 50
        assert np.array_equal(episode_indices, np.arange(25, 75))
        assert manager._active_chunk_indices == {1, 2}

        # Check that chunk 0 episodes were evicted
        for i in range(25):
            cache_key = (0, i)
            assert cache_key not in manager._shared_cache

        # Check that chunk 1 episodes are still cached
        for i in range(25, 50):
            cache_key = (0, i)
            assert cache_key in manager._shared_cache

    def test_chunk_wrapping(self):
        """Test chunk indices wrapping around."""
        index = MockZarrCorpusIndex(num_episodes=100)
        manager = ChunkManager(index=index, chunk_size=25, num_overlapping=2)

        # Load chunks [3, 0] (wrapping around)
        episode_indices = manager.load_chunks([3, 0])

        # Should load episodes 75-99 and 0-24
        expected = np.concatenate([np.arange(75, 100), np.arange(0, 25)])
        assert len(episode_indices) == 50
        assert np.array_equal(episode_indices, expected)

    def test_get_active_episodes(self):
        """Test retrieving active episode indices."""
        index = MockZarrCorpusIndex(num_episodes=100)
        manager = ChunkManager(index=index, chunk_size=25, num_overlapping=2)

        manager.load_chunks([1, 2])
        active_episodes = manager.get_active_episodes()

        assert len(active_episodes) == 50
        assert np.array_equal(active_episodes, np.arange(25, 75))

    def test_get_cache(self):
        """Test retrieving the shared cache."""
        index = MockZarrCorpusIndex(num_episodes=100)
        manager = ChunkManager(index=index, chunk_size=25)

        manager.load_chunks([0])
        cache = manager.get_cache()

        assert cache is manager._shared_cache
        assert len(cache) == 25

    def test_idempotent_loading(self):
        """Test that loading the same chunks twice doesn't reload."""
        index = MockZarrCorpusIndex(num_episodes=100)
        manager = ChunkManager(index=index, chunk_size=25, num_overlapping=2)

        # Load chunks [0, 1]
        manager.load_chunks([0, 1])
        cache_ids_first = {id(v[0]) for v in manager._shared_cache.values()}

        # Load same chunks again
        manager.load_chunks([0, 1])
        cache_ids_second = {id(v[0]) for v in manager._shared_cache.values()}

        # Tensors should be the same objects (not reloaded)
        assert cache_ids_first == cache_ids_second

    def test_partial_overlap_loading(self):
        """Test loading chunks with partial overlap."""
        index = MockZarrCorpusIndex(num_episodes=100)
        manager = ChunkManager(index=index, chunk_size=25, num_overlapping=2)

        # Load chunks [0, 1]
        manager.load_chunks([0, 1])

        # Load chunks [1, 2] - chunk 1 should be retained
        manager.load_chunks([1, 2])

        # Chunk 1 should still be in cache (not reloaded)
        for i in range(25, 50):
            cache_key = (0, i)
            assert cache_key in manager._shared_cache

    def test_background_preload_disabled(self):
        """Test that background preload doesn't run when disabled."""
        index = MockZarrCorpusIndex(num_episodes=100)
        manager = ChunkManager(
            index=index, chunk_size=25, num_overlapping=2, enable_background_load=False
        )

        manager.load_chunks([0, 1])

        # No background thread should be started
        assert manager._preload_thread is None

    @patch("time.sleep")  # Speed up test by mocking sleep
    def test_background_preload_enabled(self, mock_sleep):
        """Test that background preload starts when enabled."""
        index = MockZarrCorpusIndex(num_episodes=100)
        manager = ChunkManager(
            index=index, chunk_size=25, num_overlapping=2, enable_background_load=True
        )

        manager.load_chunks([0, 1])

        # Background thread should be started for chunks [1, 2]
        assert manager._preload_thread is not None
        assert manager._preload_chunk_indices == [1, 2]

        # Wait for thread to complete
        manager._preload_thread.join(timeout=5.0)
        assert not manager._preload_thread.is_alive()

    def test_shutdown(self):
        """Test clean shutdown of background threads."""
        index = MockZarrCorpusIndex(num_episodes=100)
        manager = ChunkManager(
            index=index, chunk_size=25, num_overlapping=2, enable_background_load=True
        )

        manager.load_chunks([0, 1])
        manager.shutdown()

        # Thread should be stopped
        if manager._preload_thread:
            assert not manager._preload_thread.is_alive()

    def test_empty_chunk_handling(self):
        """Test handling of edge cases with small datasets."""
        index = MockZarrCorpusIndex(num_episodes=10)
        manager = ChunkManager(index=index, chunk_size=25, num_overlapping=1)

        # Only 1 chunk since dataset is smaller than chunk_size
        assert manager.total_chunks == 1

        episode_indices = manager.load_chunks([0])
        assert len(episode_indices) == 10
        assert np.array_equal(episode_indices, np.arange(0, 10))


class TestChunkManagerIntegration:
    """Integration tests for ChunkManager with realistic scenarios."""

    def test_full_epoch_iteration(self):
        """Test iterating through all chunks in an epoch."""
        index = MockZarrCorpusIndex(num_episodes=100)
        manager = ChunkManager(index=index, chunk_size=25, num_overlapping=2)

        # Simulate full epoch with overlapping chunks
        all_episodes_seen = set()

        for iteration in range(manager.total_chunks):
            chunk_indices = [
                (iteration + offset) % manager.total_chunks
                for offset in range(manager.num_overlapping)
            ]
            episode_indices = manager.load_chunks(chunk_indices)
            all_episodes_seen.update(episode_indices.tolist())

        # Should have seen all episodes
        assert all_episodes_seen == set(range(100))

    def test_memory_footprint(self):
        """Test that memory usage is reasonable."""
        index = MockZarrCorpusIndex(num_episodes=100, frames_per_episode=1000)
        manager = ChunkManager(index=index, chunk_size=25, num_overlapping=2)

        manager.load_chunks([0, 1])

        # Calculate expected memory usage
        # 50 episodes * 1000 frames * (300 + 50) features * 4 bytes
        expected_bytes = 50 * 1000 * 350 * 4
        expected_mb = expected_bytes / (1024**2)

        # Verify cache contains expected number of tensors
        assert len(manager._shared_cache) == 50

        # Verify tensors are shared (not duplicated per worker)
        for cache_key, (f_tensor, t_tensor) in manager._shared_cache.items():
            assert f_tensor.is_shared()
            assert t_tensor.is_shared()

        print(f"Expected memory usage: {expected_mb:.2f} MB")

    def test_chunk_transition_consistency(self):
        """Test that chunk transitions maintain data consistency."""
        index = MockZarrCorpusIndex(num_episodes=100)
        manager = ChunkManager(index=index, chunk_size=25, num_overlapping=2)

        # Load initial chunks
        episodes_1 = manager.load_chunks([0, 1])
        cache_1 = {k: v[0].clone() for k, v in manager._shared_cache.items()}

        # Transition to next chunks
        episodes_2 = manager.load_chunks([1, 2])

        # Episodes from chunk 1 should still have same data
        for i in range(25, 50):
            cache_key = (0, i)
            if cache_key in cache_1 and cache_key in manager._shared_cache:
                # Data should be identical (same tensor, not reloaded)
                assert torch.equal(
                    cache_1[cache_key], manager._shared_cache[cache_key][0]
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
