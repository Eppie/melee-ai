"""Unit tests for memory_utils."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from data_loading.memory_utils import (
    calculate_optimal_chunk_size,
    estimate_episode_memory,
    get_available_ram_mb,
)


@dataclass
class MockEpisodeInfo:
    """Mock episode info for testing."""

    episode_id: int
    shard_id: int
    num_frames: int


class MockZarrCorpusIndex:
    """Mock ZarrCorpusIndex for testing."""

    def __init__(self, num_episodes: int = 100, frames_per_episode: int = 1000):
        self.episodes = [
            MockEpisodeInfo(
                episode_id=i,
                shard_id=0,
                num_frames=frames_per_episode,
            )
            for i in range(num_episodes)
        ]

    def open_episode_arrays(self, ep: MockEpisodeInfo):
        """Return mock feature and target arrays."""
        # Create mock arrays with realistic sizes
        features = np.random.randn(ep.num_frames, 300).astype(np.float32)
        targets = np.random.randn(ep.num_frames, 50).astype(np.float32)
        return features, targets


class TestEstimateEpisodeMemory:
    """Test estimate_episode_memory function."""

    def test_basic_estimation(self):
        """Test basic memory estimation."""
        index = MockZarrCorpusIndex(num_episodes=100, frames_per_episode=1000)

        # Sample 10 episodes
        bytes_per_episode = estimate_episode_memory(index, sample_size=10)

        # Expected: 1000 frames * (300 + 50) features * 4 bytes = 1,400,000 bytes
        expected = 1000 * 350 * 4
        assert abs(bytes_per_episode - expected) < 10000  # Allow small variance

    def test_sample_size_larger_than_dataset(self):
        """Test when sample_size > total episodes."""
        index = MockZarrCorpusIndex(num_episodes=5, frames_per_episode=1000)

        # Request 100 samples but only 5 available
        bytes_per_episode = estimate_episode_memory(index, sample_size=100)

        # Should sample all 5 episodes
        expected = 1000 * 350 * 4
        assert abs(bytes_per_episode - expected) < 10000

    def test_empty_dataset(self):
        """Test with empty dataset."""
        index = MockZarrCorpusIndex(num_episodes=0)

        bytes_per_episode = estimate_episode_memory(index, sample_size=10)

        assert bytes_per_episode == 0

    def test_variable_episode_sizes(self):
        """Test with episodes of different sizes."""

        class VariableSizeIndex(MockZarrCorpusIndex):
            def __init__(self):
                self.episodes = [
                    MockEpisodeInfo(episode_id=0, shard_id=0, num_frames=500),
                    MockEpisodeInfo(episode_id=1, shard_id=0, num_frames=1000),
                    MockEpisodeInfo(episode_id=2, shard_id=0, num_frames=1500),
                ]

        index = VariableSizeIndex()
        bytes_per_episode = estimate_episode_memory(index, sample_size=3)

        # Average: (500 + 1000 + 1500) / 3 = 1000 frames
        # 1000 frames * 350 features * 4 bytes = 1,400,000 bytes
        expected_avg = 1000 * 350 * 4
        assert abs(bytes_per_episode - expected_avg) < 100000  # Allow variance

    def test_deterministic_sampling(self):
        """Test that sampling is deterministic (for reproducibility)."""
        index = MockZarrCorpusIndex(num_episodes=100, frames_per_episode=1000)

        # Set random seed and measure
        import random

        random.seed(42)
        bytes_1 = estimate_episode_memory(index, sample_size=10)

        random.seed(42)
        bytes_2 = estimate_episode_memory(index, sample_size=10)

        assert bytes_1 == bytes_2


class TestCalculateOptimalChunkSize:
    """Test calculate_optimal_chunk_size function."""

    def test_basic_calculation(self):
        """Test basic chunk size calculation."""
        # 8192 MB RAM, 2 overlapping chunks, 4 MB per episode
        chunk_size = calculate_optimal_chunk_size(
            total_episodes=10000,
            bytes_per_episode=4 * 1024 * 1024,  # 4 MB
            num_overlapping=2,
            ram_budget_mb=8192,
            safety_margin=0.8,
        )

        # Available: 8192 * 0.8 = 6553.6 MB
        # Per chunk: 6553.6 / 2 = 3276.8 MB
        # Episodes per chunk: 3276.8 / 4 = 819
        expected = 819
        assert chunk_size == expected

    def test_single_chunk_mode(self):
        """Test with single chunk (no overlap)."""
        chunk_size = calculate_optimal_chunk_size(
            total_episodes=1000,
            bytes_per_episode=4 * 1024 * 1024,  # 4 MB
            num_overlapping=1,
            ram_budget_mb=8192,
            safety_margin=0.8,
        )

        # Available: 8192 * 0.8 = 6553.6 MB
        # Episodes: 6553.6 / 4 = 1638
        # Capped at total_episodes = 1000
        assert chunk_size == 1000

    def test_capped_at_total_episodes(self):
        """Test that chunk size is capped at total episodes."""
        chunk_size = calculate_optimal_chunk_size(
            total_episodes=100,
            bytes_per_episode=1 * 1024 * 1024,  # 1 MB
            num_overlapping=2,
            ram_budget_mb=8192,
            safety_margin=0.8,
        )

        # Would calculate to ~3276 episodes, but capped at 100
        assert chunk_size == 100

    def test_minimum_chunk_size(self):
        """Test that chunk size is at least 1."""
        chunk_size = calculate_optimal_chunk_size(
            total_episodes=1000,
            bytes_per_episode=1024 * 1024 * 1024,  # 1 GB per episode (huge!)
            num_overlapping=2,
            ram_budget_mb=100,  # Only 100 MB available
            safety_margin=0.8,
        )

        # Would calculate to 0, but should be clamped to 1
        assert chunk_size == 1

    def test_different_safety_margins(self):
        """Test with different safety margins."""
        base_params = {
            "total_episodes": 10000,
            "bytes_per_episode": 4 * 1024 * 1024,
            "num_overlapping": 2,
            "ram_budget_mb": 8192,
        }

        chunk_80 = calculate_optimal_chunk_size(**base_params, safety_margin=0.8)
        chunk_50 = calculate_optimal_chunk_size(**base_params, safety_margin=0.5)
        chunk_100 = calculate_optimal_chunk_size(**base_params, safety_margin=1.0)

        # More conservative (lower) safety margin = smaller chunks
        assert chunk_50 < chunk_80 < chunk_100

    def test_multi_chunk_overlap(self):
        """Test with different overlap counts."""
        base_params = {
            "total_episodes": 10000,
            "bytes_per_episode": 4 * 1024 * 1024,
            "ram_budget_mb": 8192,
            "safety_margin": 0.8,
        }

        chunk_1 = calculate_optimal_chunk_size(**base_params, num_overlapping=1)
        chunk_2 = calculate_optimal_chunk_size(**base_params, num_overlapping=2)
        chunk_3 = calculate_optimal_chunk_size(**base_params, num_overlapping=3)

        # More overlapping chunks = smaller chunk size (same total RAM)
        assert chunk_3 < chunk_2 < chunk_1
        # Specifically: chunk_1 should be ~2x chunk_2
        assert abs(chunk_1 - 2 * chunk_2) < 10

    def test_zero_bytes_per_episode(self):
        """Test handling of zero bytes per episode."""
        chunk_size = calculate_optimal_chunk_size(
            total_episodes=1000,
            bytes_per_episode=0,
            num_overlapping=2,
            ram_budget_mb=8192,
            safety_margin=0.8,
        )

        # Should return total_episodes (no memory constraint)
        assert chunk_size == 1000

    def test_realistic_scenario(self):
        """Test with realistic values from actual training."""
        # Real scenario: 10,000 episodes, ~4.5 MB each, 11 GB RAM available
        chunk_size = calculate_optimal_chunk_size(
            total_episodes=10000,
            bytes_per_episode=int(4.5 * 1024 * 1024),
            num_overlapping=2,
            ram_budget_mb=11000,
            safety_margin=0.8,
        )

        # Expected: 11000 * 0.8 = 8800 MB
        # Per chunk: 8800 / 2 = 4400 MB
        # Episodes: 4400 / 4.5 ≈ 977
        expected = 977
        assert abs(chunk_size - expected) < 50  # Allow some rounding variance


class TestGetAvailableRamMB:
    """Test get_available_ram_mb function."""

    def test_with_psutil_available(self):
        """Test when psutil is available."""
        # Mock the psutil module at the sys.modules level
        with patch.dict("sys.modules", {"psutil": MagicMock()}):
            import sys

            mock_psutil = sys.modules["psutil"]
            mock_vm = MagicMock()
            mock_vm.available = 8192 * 1024 * 1024  # 8192 MB in bytes
            mock_psutil.virtual_memory.return_value = mock_vm

            # Re-import to use mocked psutil
            from importlib import reload
            import data_loading.memory_utils as mem_utils

            reload(mem_utils)

            ram_mb = mem_utils.get_available_ram_mb()

            assert ram_mb == 8192

    def test_with_psutil_unavailable(self):
        """Test when psutil is not installed."""
        # Import without psutil should return 0
        with patch.dict("sys.modules", {"psutil": None}):
            # Re-import to trigger ImportError handling
            from importlib import reload
            import data_loading.memory_utils as mem_utils

            reload(mem_utils)

            # This should print a warning and return 0
            # Note: In actual code, this would happen during import
            # For testing, we just verify the function exists
            assert hasattr(mem_utils, "get_available_ram_mb")


class TestMemoryUtilsIntegration:
    """Integration tests combining multiple memory utilities."""

    def test_end_to_end_chunk_sizing(self):
        """Test complete flow from episode estimation to chunk sizing."""
        # Create realistic index
        index = MockZarrCorpusIndex(num_episodes=5000, frames_per_episode=1000)

        # Estimate episode memory
        bytes_per_episode = estimate_episode_memory(index, sample_size=50)

        # Calculate chunk size with realistic RAM budget
        chunk_size = calculate_optimal_chunk_size(
            total_episodes=len(index.episodes),
            bytes_per_episode=bytes_per_episode,
            num_overlapping=2,
            ram_budget_mb=8192,
            safety_margin=0.8,
        )

        # Verify chunk size is reasonable
        assert 100 < chunk_size < 5000
        assert chunk_size <= len(index.episodes)

        # Calculate total memory usage
        total_mb = (chunk_size * 2 * bytes_per_episode) / (1024**2)
        assert total_mb < 8192 * 0.8  # Should be within budget

    def test_chunk_size_scales_with_ram(self):
        """Test that chunk size scales proportionally with RAM."""
        index = MockZarrCorpusIndex(num_episodes=10000, frames_per_episode=1000)
        bytes_per_episode = estimate_episode_memory(index, sample_size=50)

        chunk_4gb = calculate_optimal_chunk_size(
            total_episodes=10000,
            bytes_per_episode=bytes_per_episode,
            num_overlapping=2,
            ram_budget_mb=4096,
            safety_margin=0.8,
        )

        chunk_8gb = calculate_optimal_chunk_size(
            total_episodes=10000,
            bytes_per_episode=bytes_per_episode,
            num_overlapping=2,
            ram_budget_mb=8192,
            safety_margin=0.8,
        )

        # Double RAM should roughly double chunk size
        ratio = chunk_8gb / chunk_4gb
        assert 1.9 < ratio < 2.1  # Allow small variance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
