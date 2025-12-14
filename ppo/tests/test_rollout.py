"""Tests for rollout buffer and windowing."""

import numpy as np
import torch
import pytest

from ppo.rollout import RolloutBuffer, create_windowed_batches, PPOWindow
from ppo.shared_memory import ActionData, ActionData_dtype
from column_map import ColumnMap
from schema import get_feature_names, get_target_names


@pytest.fixture
def rollout_buffer():
    """Create a rollout buffer for testing."""
    return RolloutBuffer(rollout_length=1024, feature_dim=64)


@pytest.fixture
def column_map():
    """Create column map for testing."""
    feature_names = get_feature_names()
    target_names = get_target_names()
    return ColumnMap(feature_names, target_names)


def test_rollout_buffer_initialization(rollout_buffer):
    """Test that rollout buffer initializes with correct shapes."""
    assert rollout_buffer.X.shape == (1024, 64)
    assert rollout_buffer.actions.shape == (1024,)
    assert rollout_buffer.logp.shape == (1024,)
    assert rollout_buffer.values.shape == (1024,)
    assert rollout_buffer.rewards.shape == (1024,)
    assert rollout_buffer.mask.shape == (1024,)
    assert rollout_buffer.pos == 0
    assert rollout_buffer.complete == False


def test_rollout_buffer_append(rollout_buffer):
    """Test appending frames to rollout buffer."""
    features = np.random.randn(64).astype(np.float32)
    action = np.zeros(1, dtype=ActionData_dtype)[0]
    action["main_idx"] = 10
    action["c_idx"] = 5
    action["shoulder_idx"] = 2
    action["buttons"] = np.array([1, 0, 1, 0, 0], dtype=np.uint8)
    action["logp"] = -1.5
    action["value"] = 0.3

    rollout_buffer.append(
        features=features,
        action=action,
        logp=-1.5,
        value=0.3,
        reward=0.1,
        mask=True,
    )

    assert rollout_buffer.pos == 1
    assert np.allclose(rollout_buffer.X[0], features)
    assert rollout_buffer.actions[0]["main_idx"] == 10
    assert rollout_buffer.logp[0] == -1.5


def test_rollout_buffer_complete():
    """Test that buffer marks complete when full."""
    buffer = RolloutBuffer(rollout_length=10, feature_dim=4)

    for i in range(10):
        buffer.append(
            features=np.zeros(4, dtype=np.float32),
            action=np.zeros(1, dtype=ActionData_dtype)[0],
            logp=0.0,
            value=0.0,
            reward=0.0,
            mask=True,
        )

    assert buffer.complete == True
    assert buffer.pos == 10


def test_gae_computation(rollout_buffer):
    """Test GAE advantage computation."""
    # Fill buffer with dummy data
    for i in range(1024):
        rollout_buffer.append(
            features=np.zeros(64, dtype=np.float32),
            action=np.zeros(1, dtype=ActionData_dtype)[0],
            logp=0.0,
            value=0.5,
            reward=0.1,
            mask=True,
        )

    # Compute advantages
    rollout_buffer.compute_advantages(gamma=0.99, gae_lambda=0.95)

    assert rollout_buffer.advantages is not None
    assert rollout_buffer.returns is not None
    assert rollout_buffer.advantages.shape == (1024,)
    assert rollout_buffer.returns.shape == (1024,)
    assert np.isfinite(rollout_buffer.advantages).all()
    assert np.isfinite(rollout_buffer.returns).all()


def test_gae_with_bootstrap_value():
    """Test that bootstrap value is correctly used in GAE computation."""
    buffer = RolloutBuffer(rollout_length=10, feature_dim=4)

    # Fill buffer with increasing values
    for i in range(10):
        buffer.append(
            features=np.zeros(4, dtype=np.float32),
            action=np.zeros(1, dtype=ActionData_dtype)[0],
            logp=0.0,
            value=float(i),  # Increasing values
            reward=1.0,  # Constant reward
            mask=True,
        )

    # Compute without bootstrap (assumes terminal state, next_value=0)
    buffer_no_bootstrap = RolloutBuffer(rollout_length=10, feature_dim=4)
    buffer_no_bootstrap.X = buffer.X.copy()
    buffer_no_bootstrap.actions = buffer.actions.copy()
    buffer_no_bootstrap.logp = buffer.logp.copy()
    buffer_no_bootstrap.values = buffer.values.copy()
    buffer_no_bootstrap.rewards = buffer.rewards.copy()
    buffer_no_bootstrap.mask = buffer.mask.copy()
    buffer_no_bootstrap.pos = buffer.pos
    buffer_no_bootstrap.complete = True

    adv_no_bootstrap, ret_no_bootstrap = buffer_no_bootstrap.compute_advantages(gamma=0.99, gae_lambda=0.95)

    # Compute with bootstrap value (continuing episode, next_value=10.0)
    buffer.bootstrap_value = 10.0
    adv_with_bootstrap, ret_with_bootstrap = buffer.compute_advantages(gamma=0.99, gae_lambda=0.95)

    # With bootstrap, the returns should be different (returns are not normalized)
    # The bootstrap value affects the temporal difference calculation
    assert not np.allclose(ret_no_bootstrap, ret_with_bootstrap, atol=0.1), \
        "Bootstrap value should affect returns"


def test_bootstrap_value_none_defaults_to_zero():
    """Test that None bootstrap value defaults to 0.0 (terminal assumption)."""
    buffer = RolloutBuffer(rollout_length=5, feature_dim=4)

    for i in range(5):
        buffer.append(
            features=np.zeros(4, dtype=np.float32),
            action=np.zeros(1, dtype=ActionData_dtype)[0],
            logp=0.0,
            value=1.0,
            reward=0.5,
            mask=True,
        )

    # Don't set bootstrap_value (should default to None)
    assert buffer.bootstrap_value is None

    # Compute advantages (should use 0.0 as next_value)
    advantages, returns = buffer.compute_advantages(gamma=0.99, gae_lambda=0.95)

    # Should complete without error
    assert advantages.shape == (5,)
    assert returns.shape == (5,)
    assert np.isfinite(advantages).all()


def test_bootstrap_value_reset():
    """Test that bootstrap_value is reset when buffer is reset."""
    buffer = RolloutBuffer(rollout_length=5, feature_dim=4)

    # Set bootstrap value
    buffer.bootstrap_value = 5.0
    assert buffer.bootstrap_value == 5.0

    # Reset buffer
    buffer.reset()

    # Bootstrap value should be None
    assert buffer.bootstrap_value is None


def test_gae_bootstrap_impact():
    """Test the numerical impact of bootstrap value on advantages."""
    buffer = RolloutBuffer(rollout_length=100, feature_dim=4)

    # Fill with realistic values
    for i in range(100):
        buffer.append(
            features=np.zeros(4, dtype=np.float32),
            action=np.zeros(1, dtype=ActionData_dtype)[0],
            logp=0.0,
            value=0.5,  # Constant value estimates
            reward=0.1 if i % 10 == 0 else 0.0,  # Sparse rewards
            mask=True,
        )

    # Test with high bootstrap value (continuing episode with high value)
    buffer.bootstrap_value = 10.0
    adv_high_bootstrap, ret_high_bootstrap = buffer.compute_advantages(gamma=0.995, gae_lambda=0.95)

    # Reset and test with low bootstrap value (continuing episode with low value)
    buffer.reset()
    for i in range(100):
        buffer.append(
            features=np.zeros(4, dtype=np.float32),
            action=np.zeros(1, dtype=ActionData_dtype)[0],
            logp=0.0,
            value=0.5,
            reward=0.1 if i % 10 == 0 else 0.0,
            mask=True,
        )
    buffer.bootstrap_value = -10.0
    adv_low_bootstrap, ret_low_bootstrap = buffer.compute_advantages(gamma=0.995, gae_lambda=0.95)

    # Returns should differ based on bootstrap value
    # (advantages are normalized so harder to compare directly)
    assert not np.allclose(ret_high_bootstrap, ret_low_bootstrap), \
        "Different bootstrap values should produce different returns"


def test_windowed_batches_shape(column_map):
    """Test that windowed batches have correct shapes."""
    # Create completed rollout
    rollout = RolloutBuffer(rollout_length=1024, feature_dim=64)

    for i in range(1024):
        action = np.zeros(1, dtype=ActionData_dtype)[0]
        action["main_idx"] = i % 64
        action["c_idx"] = i % 9
        action["shoulder_idx"] = i % 5
        action["buttons"] = np.array([0, 0, 0, 0, 0], dtype=np.uint8)

        rollout.append(
            features=np.random.randn(64).astype(np.float32),
            action=action,
            logp=-1.0,
            value=0.0,
            reward=0.0,
            mask=i >= 256,  # Warmup first 256 frames
        )

    rollout.compute_advantages(gamma=0.99, gae_lambda=0.95)

    # Create batches
    batches = create_windowed_batches(
        rollouts=[rollout],
        context_length=256,
        batch_size=128,
    )

    assert len(batches) > 0, "Should create at least one batch"

    # Check first batch shapes
    batch = batches[0]
    assert "features" in batch
    assert "actions" in batch
    assert "old_logp" in batch
    assert "advantages" in batch
    assert "returns" in batch

    B = batch["features"].shape[0]  # Batch size (up to 128)
    assert batch["features"].shape == (B, 256, 64), f"Got {batch['features'].shape}"
    assert batch["actions"].shape == (B, 256), f"Got {batch['actions'].shape}"
    assert batch["old_logp"].shape == (B,), f"Got {batch['old_logp'].shape}"
    assert batch["advantages"].shape == (B,), f"Got {batch['advantages'].shape}"
    assert batch["returns"].shape == (B,), f"Got {batch['returns'].shape}"


def test_windowed_batches_action_extraction():
    """Test that actions can be extracted correctly from windowed batches."""
    rollout = RolloutBuffer(rollout_length=512, feature_dim=32)

    for i in range(512):
        action = np.zeros(1, dtype=ActionData_dtype)[0]
        action["main_idx"] = 10
        action["c_idx"] = 5
        action["shoulder_idx"] = 2
        action["buttons"] = np.array([1, 0, 1, 0, 0], dtype=np.uint8)
        action["logp"] = -1.5
        action["value"] = 0.3

        rollout.append(
            features=np.random.randn(32).astype(np.float32),
            action=action,
            logp=-1.5,
            value=0.3,
            reward=0.1,
            mask=i >= 128,  # Warmup
        )

    rollout.compute_advantages(gamma=0.99, gae_lambda=0.95)

    batches = create_windowed_batches(
        rollouts=[rollout],
        context_length=128,
        batch_size=64,
    )

    batch = batches[0]
    actions = batch["actions"]

    # Extract action fields (final timestep)
    main_idx = actions[:, -1]["main_idx"]
    c_idx = actions[:, -1]["c_idx"]
    shoulder_idx = actions[:, -1]["shoulder_idx"]
    buttons = actions[:, -1]["buttons"]

    # Check shapes after extraction
    B = actions.shape[0]
    assert main_idx.shape == (B,), f"main_idx shape: {main_idx.shape}"
    assert c_idx.shape == (B,), f"c_idx shape: {c_idx.shape}"
    assert shoulder_idx.shape == (B,), f"shoulder_idx shape: {shoulder_idx.shape}"
    assert buttons.shape == (B, 5), f"buttons shape: {buttons.shape}"


def test_multiple_rollouts_batching(column_map):
    """Test batching with multiple rollouts."""
    rollouts = []

    for _ in range(3):
        rollout = RolloutBuffer(rollout_length=512, feature_dim=32)

        for i in range(512):
            action = np.zeros(1, dtype=ActionData_dtype)[0]
            action["main_idx"] = i % 64

            rollout.append(
                features=np.random.randn(32).astype(np.float32),
                action=action,
                logp=-1.0,
                value=0.0,
                reward=0.0,
                mask=i >= 128,
            )

        rollout.compute_advantages(gamma=0.99, gae_lambda=0.95)
        rollouts.append(rollout)

    batches = create_windowed_batches(
        rollouts=rollouts,
        context_length=128,
        batch_size=64,
    )

    # Should create batches from all rollouts
    # Each rollout: (512 - 128 + 1) = 385 windows, minus warmup
    # 3 rollouts * ~385 windows = ~1155 windows
    # With batch size 64, expect ~18 batches
    assert len(batches) > 10, f"Expected many batches, got {len(batches)}"


def test_stride_efficiency():
    """Test that larger stride creates fewer batches."""
    rollout = RolloutBuffer(rollout_length=1024, feature_dim=32)

    # Fill rollout
    for i in range(1024):
        action = np.zeros(1, dtype=ActionData_dtype)[0]
        rollout.append(
            features=np.random.randn(32).astype(np.float32),
            action=action,
            logp=-1.0,
            value=0.5,
            reward=0.1,
            mask=i >= 256,  # Skip first 256 frames (warmup)
        )

    rollout.compute_advantages(gamma=0.99, gae_lambda=0.95)

    # Small stride (more overlap, more batches)
    small_stride_batches = create_windowed_batches(
        rollouts=[rollout],
        context_length=256,
        batch_size=128,
        stride=8,
    )

    # Large stride (less overlap, fewer batches)
    large_stride_batches = create_windowed_batches(
        rollouts=[rollout],
        context_length=256,
        batch_size=128,
        stride=256,
    )

    # Larger stride should create fewer windows (count total windows across all batches)
    small_stride_windows = sum(b["features"].shape[0] for b in small_stride_batches)
    large_stride_windows = sum(b["features"].shape[0] for b in large_stride_batches)

    print(f"Stride 8 windows: {small_stride_windows}, Stride 256 windows: {large_stride_windows}")
    assert large_stride_windows < small_stride_windows, \
        "Larger stride should create fewer windows"

    # Check shapes (all sequence training now)
    batch = large_stride_batches[0]
    assert batch["old_logp"].ndim == 2, "Should be [B, T] for sequence training"
    assert batch["advantages"].ndim == 2, "Should be [B, T] for sequence training"
    assert batch["returns"].ndim == 2, "Should be [B, T] for sequence training"


def test_sequence_training_shapes():
    """Test that sequence training produces correct shapes."""
    rollout = RolloutBuffer(rollout_length=512, feature_dim=32)

    for i in range(512):
        action = np.zeros(1, dtype=ActionData_dtype)[0]
        rollout.append(
            features=np.random.randn(32).astype(np.float32),
            action=action,
            logp=-1.0,
            value=0.5,
            reward=0.1,
            mask=i >= 128,
        )

    rollout.compute_advantages(gamma=0.99, gae_lambda=0.95)

    # Create batches
    batches = create_windowed_batches(
        rollouts=[rollout],
        context_length=128,
        batch_size=32,
        stride=128,
    )

    batch = batches[0]
    B = batch["features"].shape[0]
    assert batch["features"].shape == (B, 128, 32)
    assert batch["old_logp"].shape == (B, 128), f"Got {batch['old_logp'].shape}"
    assert batch["advantages"].shape == (B, 128)
    assert batch["returns"].shape == (B, 128)


def test_stride_window_creation():
    """Test that stride correctly controls window overlap."""
    rollout = RolloutBuffer(rollout_length=512, feature_dim=16)

    for i in range(512):
        action = np.zeros(1, dtype=ActionData_dtype)[0]
        rollout.append(
            features=np.full(16, i, dtype=np.float32),  # Fill with frame number
            action=action,
            logp=-1.0,
            value=0.5,
            reward=0.1,
            mask=True,
        )

    rollout.compute_advantages(gamma=0.99, gae_lambda=0.95)

    # Test different strides
    for stride in [64, 128, 256]:
        batches = create_windowed_batches(
            rollouts=[rollout],
            context_length=256,
            batch_size=128,
            stride=stride,
        )

        # Check number of windows: (rollout_length - context_length) // stride + 1
        expected_windows = (512 - 256) // stride + 1
        total_windows = sum(b["features"].shape[0] for b in batches)
        assert total_windows == expected_windows, \
            f"Stride {stride}: Expected {expected_windows} windows, got {total_windows}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
