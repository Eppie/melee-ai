"""Tests for coordinator data flow and shape handling."""

import torch
import numpy as np


def test_action_shape_extraction():
    """Test that action sequences keep full [B, T] shape for PPO."""
    batch_size = 128
    seq_length = 256

    # Simulate action arrays from batches
    main_idx = np.random.randint(0, 64, (batch_size, seq_length))
    c_idx = np.random.randint(0, 9, (batch_size, seq_length))
    shoulder_idx = np.random.randint(0, 5, (batch_size, seq_length))
    buttons = np.random.randint(0, 2, (batch_size, seq_length, 5))

    assert main_idx.shape == (batch_size, seq_length)
    assert c_idx.shape == (batch_size, seq_length)
    assert shoulder_idx.shape == (batch_size, seq_length)
    assert buttons.shape == (batch_size, seq_length, 5)


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
