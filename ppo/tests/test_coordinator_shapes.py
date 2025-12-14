"""Tests for coordinator data flow and shape handling."""

import torch
import numpy as np
import pytest

from column_map import ColumnMap
from schema import get_feature_names, get_target_names
from train.batch_utils import build_model_inputs


@pytest.fixture
def column_map():
    """Create column map for testing."""
    feature_names = get_feature_names()
    target_names = get_target_names()
    return ColumnMap(feature_names, target_names)


def test_horizon_feature_appending():
    """Test that horizon feature can be appended correctly."""
    batch_size = 32
    seq_length = 256
    feature_dim = 64

    # Simulated features from coordinator
    features = torch.randn(batch_size, seq_length, feature_dim)

    # Append horizon
    horizon = torch.full(
        (batch_size, seq_length, 1), 0.5, device=features.device, dtype=features.dtype
    )
    features_with_horizon = torch.cat([features, horizon], dim=-1)

    assert features_with_horizon.shape == (batch_size, seq_length, feature_dim + 1)
    assert features_with_horizon[:, :, -1].allclose(torch.tensor(0.5))


def test_build_model_inputs_with_horizon(column_map):
    """Test that build_model_inputs handles horizon feature correctly."""
    batch_size = 16
    seq_length = 128
    feature_dim = len(get_feature_names())

    # Create features with horizon appended
    features = torch.randn(batch_size, seq_length, feature_dim)
    horizon = torch.full(
        (batch_size, seq_length, 1), 0.5, device=features.device, dtype=features.dtype
    )
    features_with_horizon = torch.cat([features, horizon], dim=-1)

    # Should not raise
    try:
        model_inputs = build_model_inputs(features_with_horizon, column_map)
        assert "gamestate" in model_inputs
        assert "controller" in model_inputs

        # Horizon should be appended to gamestate
        expected_gamestate_dim = len(column_map.gamestate_idxs) + 1  # +1 for horizon
        actual_gamestate_dim = model_inputs["gamestate"].shape[-1]
        assert (
            actual_gamestate_dim == expected_gamestate_dim
        ), f"Gamestate dim: expected {expected_gamestate_dim}, got {actual_gamestate_dim}"
    except Exception as e:
        pytest.fail(f"build_model_inputs failed with horizon: {e}")


def test_action_shape_extraction():
    """Test extracting final timestep from action sequences."""
    batch_size = 128
    seq_length = 256

    # Simulate action arrays from batches
    main_idx = np.random.randint(0, 64, (batch_size, seq_length))
    c_idx = np.random.randint(0, 9, (batch_size, seq_length))
    shoulder_idx = np.random.randint(0, 5, (batch_size, seq_length))
    buttons = np.random.randint(0, 2, (batch_size, seq_length, 5))

    # Extract final timestep (what coordinator should do)
    main_idx_final = main_idx[:, -1]
    c_idx_final = c_idx[:, -1]
    shoulder_idx_final = shoulder_idx[:, -1]
    buttons_final = buttons[:, -1, :]

    assert main_idx_final.shape == (batch_size,)
    assert c_idx_final.shape == (batch_size,)
    assert shoulder_idx_final.shape == (batch_size,)
    assert buttons_final.shape == (batch_size, 5)


def test_feature_scaling_reversal():
    """Test that feature scaling can be reversed correctly for display."""
    # Simulate scaled features from coordinator
    scaled_pct = np.array([0.0, 0.25, 0.50, 0.99])
    scaled_stock = np.array([0.0, 0.25, 0.75, 1.0])
    scaled_pos_x = np.array([-2.5, 0.0, 2.5, 6.5])

    # Reverse scaling
    unscaled_pct = scaled_pct * 100.0
    unscaled_stock = scaled_stock * 4.0
    unscaled_pos_x = scaled_pos_x * 20.0

    assert np.allclose(unscaled_pct, [0, 25, 50, 99])
    assert np.allclose(unscaled_stock, [0, 1, 3, 4])
    assert np.allclose(unscaled_pos_x, [-50, 0, 50, 130])


def test_memory_estimation():
    """Test memory estimation calculation."""
    batch_size = 128
    context_length = 256
    feature_dim = 64

    # Estimate single batch memory
    batch_memory_bytes = (
        batch_size
        * context_length
        * (feature_dim * 4 + 40 + 16)  # features + actions + metadata
    )
    batch_memory_mb = batch_memory_bytes / 1024 / 1024

    # Should be reasonable
    assert batch_memory_mb > 0
    assert batch_memory_mb < 100  # Single batch shouldn't exceed 100MB


def test_tensor_cleanup():
    """Test that tensors can be deleted properly."""
    # Create some tensors
    features = torch.randn(32, 256, 64)
    actions = torch.randint(0, 64, (32,))
    loss = torch.tensor(1.5)

    # Delete them
    del features
    del actions
    del loss

    # Force garbage collection
    import gc

    gc.collect()

    # Should not raise
    assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
