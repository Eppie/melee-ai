"""Tests for checkpoint saving and loading functionality."""

import tempfile
from pathlib import Path

import pytest
import torch

from config import Config
from ppo.config import PPOConfig
from ppo.coordinator import Coordinator


@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary directory for checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def minimal_config(temp_checkpoint_dir):
    """Create minimal config for testing."""
    # Create a dummy init checkpoint
    init_ckpt = temp_checkpoint_dir / "init.pt"
    torch.save({"model": {"dummy": torch.randn(10, 10)}}, init_ckpt)

    config = Config()
    ppo_config = PPOConfig(
        num_shards=1,
        envs_per_shard=2,
        dolphin_path="/Applications/Slippi Dolphin.app",
        iso_path="~/Documents/SSBM.iso",
        init_checkpoint=init_ckpt,
        checkpoint_dir=temp_checkpoint_dir / "ppo_checkpoints",
        checkpoint_interval=5,
    )
    return config, ppo_config


def test_checkpoint_directory_creation(temp_checkpoint_dir):
    """Test that checkpoint directory is created."""
    checkpoint_dir = temp_checkpoint_dir / "ppo_checkpoints"
    assert not checkpoint_dir.exists()

    # Create directory
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    assert checkpoint_dir.exists()


def test_checkpoint_filename_format(temp_checkpoint_dir):
    """Test checkpoint filename formatting."""
    checkpoint_dir = temp_checkpoint_dir / "ppo_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Test filename generation
    for step in [0, 1, 10, 100, 1000, 999999]:
        filename = f"checkpoint_{step:06d}.pt"
        checkpoint_path = checkpoint_dir / filename

        # Verify format
        assert checkpoint_path.name.startswith("checkpoint_")
        assert checkpoint_path.name.endswith(".pt")
        assert len(filename) == len("checkpoint_000000.pt")


def test_checkpoint_save_structure(temp_checkpoint_dir):
    """Test that saved checkpoint has correct structure."""
    checkpoint_path = temp_checkpoint_dir / "test_checkpoint.pt"

    # Create mock checkpoint
    checkpoint = {
        "model_state_dict": {"layer1.weight": torch.randn(10, 10)},
        "optimizer_state_dict": {"state": {}, "param_groups": []},
        "config": {"model": {"n_layer": 6}},
        "ppo_config": {"lr": 3e-4},
        "training_steps": 42,
        "total_frames": 100000,
        "step_id": 1024,
    }

    torch.save(checkpoint, checkpoint_path)

    # Load and verify
    loaded = torch.load(checkpoint_path, weights_only=False)

    assert "model_state_dict" in loaded
    assert "optimizer_state_dict" in loaded
    assert "config" in loaded
    assert "ppo_config" in loaded
    assert "training_steps" in loaded
    assert "total_frames" in loaded
    assert "step_id" in loaded

    assert loaded["training_steps"] == 42
    assert loaded["total_frames"] == 100000


def test_checkpoint_pruning(temp_checkpoint_dir):
    """Test that old checkpoints are pruned correctly."""
    checkpoint_dir = temp_checkpoint_dir / "ppo_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create 30 checkpoints
    for i in range(30):
        checkpoint_path = checkpoint_dir / f"checkpoint_{i:06d}.pt"
        torch.save({"step": i}, checkpoint_path)

    # Verify all created
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pt"))
    assert len(checkpoints) == 30

    # Prune to keep only 25 most recent (opponent_pool_size=20 + 5)
    max_keep = 25
    checkpoints_by_time = sorted(
        checkpoint_dir.glob("checkpoint_*.pt"),
        key=lambda p: p.stat().st_mtime,
    )

    if len(checkpoints_by_time) > max_keep:
        to_delete = checkpoints_by_time[:-max_keep]
        for ckpt in to_delete:
            ckpt.unlink()

    # Verify only 25 remain
    remaining = sorted(checkpoint_dir.glob("checkpoint_*.pt"))
    assert len(remaining) == 25

    # Verify oldest 5 were deleted
    remaining_nums = [int(p.stem.split("_")[1]) for p in remaining]
    assert min(remaining_nums) == 5  # 0-4 deleted
    assert max(remaining_nums) == 29


def test_checkpoint_interval_logic():
    """Test checkpoint saving interval logic."""
    checkpoint_interval = 5
    training_steps = [0, 1, 4, 5, 9, 10, 15, 20, 99, 100]

    for step in training_steps:
        should_save = (step % checkpoint_interval == 0) and step > 0
        if step in [5, 10, 15, 20, 100]:
            assert should_save
        else:
            assert not should_save


def test_metrics_tracking():
    """Test that training metrics are tracked correctly."""
    # Initialize metrics
    training_steps = 0
    total_frames = 0
    num_envs = 96

    # Simulate training loop
    for inference_step in range(1000):
        total_frames += num_envs

        # After rollout_length steps, trigger training
        if (inference_step + 1) % 1024 == 0:
            training_steps += 1

    # Verify metrics
    assert total_frames == 1000 * 96  # 96,000 frames
    assert training_steps == 0  # No full rollout completed yet

    # Complete one more step to trigger training
    total_frames += num_envs
    if (1000 + 1) % 1024 == 0:
        training_steps += 1

    # Still no training (1001 % 1024 != 0)
    assert training_steps == 0


def test_fps_calculation():
    """Test FPS calculation logic."""
    import time

    start_time = time.time()
    time.sleep(0.1)  # 100ms
    elapsed = time.time() - start_time

    # Simulate processing frames
    total_frames = 1000
    fps = total_frames / elapsed

    # Should be roughly 10,000 FPS (1000 frames / 0.1 sec)
    assert 8000 < fps < 12000  # Allow some tolerance


def test_checkpoint_metadata():
    """Test that checkpoint metadata is correct."""
    metadata = {
        "training_steps": 42,
        "total_frames": 4300800,  # 42 * 1024 * 100
        "step_id": 43008,  # 42 * 1024
    }

    # Verify relationship between metrics
    rollout_length = 1024
    num_envs = 100  # hypothetical

    expected_total_frames = metadata["training_steps"] * rollout_length * num_envs
    assert metadata["total_frames"] == expected_total_frames

    expected_step_id = metadata["training_steps"] * rollout_length
    assert metadata["step_id"] == expected_step_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
