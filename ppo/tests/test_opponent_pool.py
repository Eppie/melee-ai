"""Tests for opponent pool and matchmaking system."""

import tempfile
from pathlib import Path

import pytest
import torch

from ppo.opponent import Matchmaker, OpponentPool


@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary directory with mock checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create mock checkpoint files
        for i in range(5):
            checkpoint_path = tmpdir / f"checkpoint_{i:04d}.pt"
            # Create minimal checkpoint
            torch.save(
                {
                    "model_state_dict": {"dummy": torch.randn(10, 10)},
                    "step": i * 1000,
                },
                checkpoint_path,
            )

        yield tmpdir


def test_opponent_pool_discovery(temp_checkpoint_dir):
    """Test that opponent pool discovers checkpoints."""
    pool = OpponentPool(
        checkpoint_dir=temp_checkpoint_dir,
        historical_ratio=0.8,
        max_pool_size=10,
    )

    assert len(pool.checkpoint_paths) == 5
    print(f"Discovered {len(pool.checkpoint_paths)} checkpoints")


def test_opponent_pool_sampling(temp_checkpoint_dir):
    """Test checkpoint sampling logic."""
    pool = OpponentPool(
        checkpoint_dir=temp_checkpoint_dir,
        historical_ratio=0.8,
        max_pool_size=10,
    )

    # Sample many times and count self-play vs historical
    num_samples = 1000
    historical_count = 0

    for _ in range(num_samples):
        checkpoint = pool.sample_checkpoint_path()
        if checkpoint is not None:
            historical_count += 1

    historical_ratio = historical_count / num_samples

    # Should be approximately 80% historical (within tolerance)
    assert 0.75 < historical_ratio < 0.85, \
        f"Expected ~80% historical, got {historical_ratio*100:.1f}%"

    print(f"Sampled {historical_ratio*100:.1f}% historical (expected ~80%)")


def test_opponent_pool_max_size(temp_checkpoint_dir):
    """Test that pool respects max_pool_size."""
    # Create many more checkpoints
    for i in range(5, 100):
        checkpoint_path = temp_checkpoint_dir / f"checkpoint_{i:04d}.pt"
        torch.save({"dummy": torch.randn(10, 10)}, checkpoint_path)

    pool = OpponentPool(
        checkpoint_dir=temp_checkpoint_dir,
        historical_ratio=0.8,
        max_pool_size=20,
    )

    assert len(pool.checkpoint_paths) == 20
    print(f"Pool size capped at {len(pool.checkpoint_paths)} (max=20)")


def test_opponent_pool_refresh(temp_checkpoint_dir):
    """Test that pool refreshes to pick up new checkpoints."""
    pool = OpponentPool(
        checkpoint_dir=temp_checkpoint_dir,
        historical_ratio=0.8,
        max_pool_size=10,
        refresh_interval=5,
    )

    initial_count = len(pool.checkpoint_paths)

    # Add new checkpoint
    new_checkpoint = temp_checkpoint_dir / "checkpoint_new.pt"
    torch.save({"dummy": torch.randn(10, 10)}, new_checkpoint)

    # Trigger refresh
    for _ in range(10):
        pool.maybe_refresh()

    assert len(pool.checkpoint_paths) == initial_count + 1
    print(f"Pool refreshed: {initial_count} → {len(pool.checkpoint_paths)}")


def test_matchmaker_assignment(temp_checkpoint_dir):
    """Test matchmaker opponent assignment."""
    pool = OpponentPool(
        checkpoint_dir=temp_checkpoint_dir,
        historical_ratio=0.8,
        max_pool_size=10,
    )

    matchmaker = Matchmaker(
        opponent_pool=pool,
        num_envs=96,
        device=torch.device("cpu"),
    )

    # Count self-play vs historical
    self_play_count = sum(
        1 for path in matchmaker.match_assignments.values() if path is None
    )
    historical_count = 96 - self_play_count

    # Should be approximately 80% historical
    historical_ratio = historical_count / 96
    assert 0.7 < historical_ratio < 0.9, \
        f"Expected ~80% historical, got {historical_ratio*100:.1f}%"

    print(f"Matchmaker: {historical_count}/96 historical ({historical_ratio*100:.1f}%)")


def test_matchmaker_is_self_play(temp_checkpoint_dir):
    """Test self-play detection."""
    pool = OpponentPool(
        checkpoint_dir=temp_checkpoint_dir,
        historical_ratio=0.5,  # 50/50 for testing
    )

    matchmaker = Matchmaker(
        opponent_pool=pool,
        num_envs=96,
        device=torch.device("cpu"),
    )

    # Check that is_self_play is consistent with assignments
    for env_id in range(96):
        is_self_play = matchmaker.is_self_play(env_id)
        assignment = matchmaker.match_assignments[env_id]

        assert (is_self_play and assignment is None) or \
               (not is_self_play and assignment is not None)

    print("Self-play detection consistent with assignments")


def test_matchmaker_reassignment(temp_checkpoint_dir):
    """Test that reassign_all changes assignments."""
    pool = OpponentPool(
        checkpoint_dir=temp_checkpoint_dir,
        historical_ratio=0.8,
    )

    matchmaker = Matchmaker(
        opponent_pool=pool,
        num_envs=96,
        device=torch.device("cpu"),
    )

    # Store initial assignments
    initial_assignments = matchmaker.match_assignments.copy()

    # Reassign
    matchmaker.reassign_all()

    # Should have different assignments (with high probability)
    changed_count = sum(
        1 for env_id in range(96)
        if initial_assignments[env_id] != matchmaker.match_assignments[env_id]
    )

    # At least some should have changed (not deterministic, but very likely)
    assert changed_count > 0, "Reassignment should change some assignments"
    print(f"Reassignment changed {changed_count}/96 assignments")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
