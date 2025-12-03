"""Tests for PPO configuration validation."""

import tempfile
from pathlib import Path

import pytest
import torch

from ppo.config import PPOConfig


@pytest.fixture
def temp_files():
    """Create temporary files for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create dummy checkpoint
        checkpoint_path = tmpdir / "checkpoint.pt"
        torch.save({"model": {}}, checkpoint_path)

        # Create dummy Dolphin and ISO (just empty files for validation)
        dolphin_path = tmpdir / "Slippi Dolphin.app"
        dolphin_path.mkdir()

        iso_path = tmpdir / "SSBM.iso"
        iso_path.touch()

        yield {
            "checkpoint": checkpoint_path,
            "dolphin": str(dolphin_path),
            "iso": str(iso_path),
        }


def test_valid_config(temp_files):
    """Test that valid config passes validation."""
    config = PPOConfig(
        dolphin_path=temp_files["dolphin"],
        iso_path=temp_files["iso"],
        init_checkpoint=temp_files["checkpoint"],
        character="FOX",
        stages=["FD", "BF"],
    )

    assert config.dolphin_path == temp_files["dolphin"]
    assert config.character == "FOX"
    assert config.stages == ["FD", "BF"]


def test_invalid_dolphin_path():
    """Test that invalid Dolphin path raises error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        checkpoint_path = tmpdir / "checkpoint.pt"
        torch.save({"model": {}}, checkpoint_path)
        iso_path = tmpdir / "SSBM.iso"
        iso_path.touch()

        with pytest.raises(ValueError, match="Dolphin path does not exist"):
            PPOConfig(
                dolphin_path="/nonexistent/dolphin",
                iso_path=str(iso_path),
                init_checkpoint=checkpoint_path,
            )


def test_invalid_iso_path():
    """Test that invalid ISO path raises error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        checkpoint_path = tmpdir / "checkpoint.pt"
        torch.save({"model": {}}, checkpoint_path)
        dolphin_path = tmpdir / "Slippi Dolphin.app"
        dolphin_path.mkdir()

        with pytest.raises(ValueError, match="ISO path does not exist"):
            PPOConfig(
                dolphin_path=str(dolphin_path),
                iso_path="/nonexistent/iso.iso",
                init_checkpoint=checkpoint_path,
            )


def test_invalid_character(temp_files):
    """Test that invalid character raises error."""
    with pytest.raises(ValueError, match="Invalid character"):
        PPOConfig(
            dolphin_path=temp_files["dolphin"],
            iso_path=temp_files["iso"],
            init_checkpoint=temp_files["checkpoint"],
            character="PIKACHU",  # Invalid
        )


def test_invalid_stages(temp_files):
    """Test that invalid stages raise error."""
    with pytest.raises(ValueError, match="Invalid stages"):
        PPOConfig(
            dolphin_path=temp_files["dolphin"],
            iso_path=temp_files["iso"],
            init_checkpoint=temp_files["checkpoint"],
            stages=["FD", "INVALID_STAGE"],
        )


def test_empty_stages(temp_files):
    """Test that empty stages list raises error."""
    with pytest.raises(ValueError, match="Must specify at least one stage"):
        PPOConfig(
            dolphin_path=temp_files["dolphin"],
            iso_path=temp_files["iso"],
            init_checkpoint=temp_files["checkpoint"],
            stages=[],
        )


def test_context_length_validation(temp_files):
    """Test context_length <= rollout_length validation."""
    with pytest.raises(ValueError, match="context_length.*must be <=.*rollout_length"):
        PPOConfig(
            dolphin_path=temp_files["dolphin"],
            iso_path=temp_files["iso"],
            init_checkpoint=temp_files["checkpoint"],
            context_length=2048,  # Larger than rollout_length
            rollout_length=1024,
        )


def test_warmup_validation(temp_files):
    """Test warmup_frames <= context_length validation."""
    with pytest.raises(ValueError, match="warmup_frames.*must be <=.*context_length"):
        PPOConfig(
            dolphin_path=temp_files["dolphin"],
            iso_path=temp_files["iso"],
            init_checkpoint=temp_files["checkpoint"],
            warmup_frames=512,
            context_length=256,  # Smaller than warmup
        )


def test_batch_size_validation(temp_files):
    """Test batch_size <= rollout_length validation."""
    with pytest.raises(ValueError, match="batch_size.*must be <=.*rollout_length"):
        PPOConfig(
            dolphin_path=temp_files["dolphin"],
            iso_path=temp_files["iso"],
            init_checkpoint=temp_files["checkpoint"],
            batch_size=2048,
            rollout_length=1024,  # Smaller than batch_size
        )


def test_missing_checkpoint():
    """Test that missing checkpoint raises error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        dolphin_path = tmpdir / "Slippi Dolphin.app"
        dolphin_path.mkdir()
        iso_path = tmpdir / "SSBM.iso"
        iso_path.touch()

        with pytest.raises(ValueError, match="Initial checkpoint not found"):
            PPOConfig(
                dolphin_path=str(dolphin_path),
                iso_path=str(iso_path),
                init_checkpoint=tmpdir / "nonexistent.pt",
            )


def test_total_envs_property(temp_files):
    """Test total_envs property calculation."""
    config = PPOConfig(
        dolphin_path=temp_files["dolphin"],
        iso_path=temp_files["iso"],
        init_checkpoint=temp_files["checkpoint"],
        num_shards=12,
        envs_per_shard=8,
    )

    assert config.total_envs == 96


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
