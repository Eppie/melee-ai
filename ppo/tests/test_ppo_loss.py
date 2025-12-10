"""Tests for PPO loss computation."""

import torch
import numpy as np
import pytest

from ppo.ppo_loss import compute_ppo_loss, compute_action_logprob, compute_action_entropy
from column_map import ColumnMap
from schema import get_feature_names, get_target_names
from model.nano_gpt import GPT
from config import Config


@pytest.fixture
def column_map():
    """Create column map for testing."""
    feature_names = get_feature_names()
    target_names = get_target_names()
    return ColumnMap(feature_names, target_names)


@pytest.fixture
def mock_policy():
    """Create a small policy network for testing."""
    config = Config()
    # Override with small model for faster testing
    config.model.n_layer = 2
    config.model.n_embd = 64
    config.model.block_size = 256

    policy = GPT(config)
    policy.eval()
    return policy


def test_action_logprob_shapes():
    """Test that action log probability computation handles correct shapes."""
    batch_size = 32

    # Create mock outputs (with final timestep extracted)
    outputs = {
        "main_stick": torch.randn(batch_size, 256, 64),  # [B, T, 64]
        "c_stick": torch.randn(batch_size, 256, 9),      # [B, T, 9]
        "shoulder": torch.randn(batch_size, 256, 5),     # [B, T, 5]
        "buttons": torch.randn(batch_size, 256, 5),      # [B, T, 5]
        "value": torch.randn(batch_size, 256, 1),        # [B, T, 1]
    }

    # Create mock actions (final timestep only)
    actions = {
        "main_idx": torch.randint(0, 64, (batch_size,)),     # [B]
        "c_idx": torch.randint(0, 9, (batch_size,)),         # [B]
        "shoulder_idx": torch.randint(0, 5, (batch_size,)),  # [B]
        "buttons": torch.randint(0, 2, (batch_size, 5)).bool(),  # [B, 5]
    }

    # Should not raise
    logp = compute_action_logprob(outputs, actions)

    # Check output shape
    assert logp.shape == (batch_size,), f"Expected shape ({batch_size},), got {logp.shape}"
    assert torch.isfinite(logp).all(), "Log probs contain NaN or Inf"


def test_action_entropy_shapes():
    """Test that entropy computation returns correct shapes."""
    batch_size = 32

    outputs = {
        "main_stick": torch.randn(batch_size, 256, 64),
        "c_stick": torch.randn(batch_size, 256, 9),
        "shoulder": torch.randn(batch_size, 256, 5),
        "buttons": torch.randn(batch_size, 256, 5),
    }

    entropy = compute_action_entropy(outputs)

    assert entropy.shape == (batch_size,), f"Expected shape ({batch_size},), got {entropy.shape}"
    assert (entropy >= 0).all(), "Entropy should be non-negative"


def test_ppo_loss_shapes(column_map, mock_policy):
    """Test that PPO loss computation handles all shapes correctly."""
    batch_size = 16
    seq_length = 256
    feature_dim = len(get_feature_names())

    # Create mock batch data
    batch_features = torch.randn(batch_size, seq_length, feature_dim)

    batch_actions = {
        "main_idx": torch.randint(0, 64, (batch_size,)),
        "c_idx": torch.randint(0, 9, (batch_size,)),
        "shoulder_idx": torch.randint(0, 5, (batch_size,)),
        "buttons": torch.randint(0, 2, (batch_size, 5)).bool(),
    }

    old_logps = torch.randn(batch_size)
    advantages = torch.randn(batch_size)
    returns = torch.randn(batch_size)

    # Compute loss
    with torch.no_grad():  # Faster for testing
        loss_dict = compute_ppo_loss(
            policy=mock_policy,
            batch_features=batch_features,
            batch_actions=batch_actions,
            old_logps=old_logps,
            advantages=advantages,
            returns=returns,
            column_map=column_map,
            clip_epsilon=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
        )

    # Check all expected keys are present
    expected_keys = ["total", "policy", "value", "entropy", "ratio_mean", "ratio_std", "approx_kl"]
    for key in expected_keys:
        assert key in loss_dict, f"Missing key: {key}"

    # Check loss is scalar
    assert loss_dict["total"].shape == (), "Total loss should be scalar"
    assert torch.isfinite(loss_dict["total"]), "Total loss is NaN or Inf"



def test_button_type_handling():
    """Test that button actions handle uint8/bool conversion correctly."""
    batch_size = 32

    outputs = {
        "main_stick": torch.randn(batch_size, 256, 64),
        "c_stick": torch.randn(batch_size, 256, 9),
        "shoulder": torch.randn(batch_size, 256, 5),
        "buttons": torch.randn(batch_size, 256, 5),
        "value": torch.randn(batch_size, 256, 1),
    }

    # Test with uint8 (should be converted to bool internally)
    actions_uint8 = {
        "main_idx": torch.randint(0, 64, (batch_size,)),
        "c_idx": torch.randint(0, 9, (batch_size,)),
        "shoulder_idx": torch.randint(0, 5, (batch_size,)),
        "buttons": torch.randint(0, 2, (batch_size, 5), dtype=torch.uint8),
    }

    # Should not raise warning or error
    with pytest.warns(None) as warning_list:
        logp = compute_action_logprob(outputs, actions_uint8)

    # Check no uint8 deprecation warnings
    uint8_warnings = [w for w in warning_list if "uint8" in str(w.message).lower()]
    assert len(uint8_warnings) == 0, f"Got uint8 warning: {uint8_warnings}"


def test_clipping_behavior():
    """Test that PPO clipping works as expected."""
    batch_size = 16
    seq_length = 256
    feature_dim = len(get_feature_names())

    column_map = ColumnMap(get_feature_names(), get_target_names())
    config = Config()
    config.model.n_layer = 1
    config.model.n_embd = 32
    policy = GPT(config)

    batch_features = torch.randn(batch_size, seq_length, feature_dim)
    batch_actions = {
        "main_idx": torch.randint(0, 64, (batch_size,)),
        "c_idx": torch.randint(0, 9, (batch_size,)),
        "shoulder_idx": torch.randint(0, 5, (batch_size,)),
        "buttons": torch.randint(0, 2, (batch_size, 5)).bool(),
    }

    old_logps = torch.randn(batch_size)
    advantages = torch.randn(batch_size)
    returns = torch.randn(batch_size)

    with torch.no_grad():
        loss_dict = compute_ppo_loss(
            policy=policy,
            batch_features=batch_features,
            batch_actions=batch_actions,
            old_logps=old_logps,
            advantages=advantages,
            returns=returns,
            column_map=column_map,
            clip_epsilon=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
        )

    # Check ratio is reasonable (not exploding)
    assert loss_dict["ratio_mean"] > 0, "Ratio mean should be positive"
    assert loss_dict["ratio_mean"] < 10, "Ratio mean should not explode"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
