"""Tests for PPO loss computation."""

import torch
import pytest

from ppo.ppo_loss import (
    compute_ppo_loss,
    compute_action_logprob,
    compute_action_entropy,
)
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


def make_valid_features(
    batch_size: int, seq_length: int, column_map: ColumnMap, *, add_noise: bool = True
) -> torch.Tensor:
    """Create feature tensors with categorical columns in-bounds."""
    feature_dim = len(get_feature_names())
    features = torch.zeros(batch_size, seq_length, feature_dim)

    if add_noise:
        features += torch.randn_like(features) * 0.1

    features[..., column_map.stage_idx] = 0  # valid stage id
    features[..., column_map.ego_char_idx] = 0
    features[..., column_map.opp_char_idx] = 0
    features[..., column_map.ego_action_idx] = 0
    features[..., column_map.opp_action_idx] = 0

    return features


def test_action_logprob_shapes():
    """Test that action log probability computation handles correct shapes."""
    batch_size = 32
    seq_length = 16

    outputs = {
        "main_stick": torch.randn(batch_size, seq_length, 64),  # [B, T, 64]
        "c_stick": torch.randn(batch_size, seq_length, 9),  # [B, T, 9]
        "shoulder": torch.randn(batch_size, seq_length, 5),  # [B, T, 5]
        "buttons": torch.randn(batch_size, seq_length, 5),  # [B, T, 5]
        "value": torch.randn(batch_size, seq_length, 1),  # [B, T, 1]
    }

    actions = {
        "main_idx": torch.randint(0, 64, (batch_size, seq_length)),  # [B, T]
        "c_idx": torch.randint(0, 9, (batch_size, seq_length)),  # [B, T]
        "shoulder_idx": torch.randint(0, 5, (batch_size, seq_length)),  # [B, T]
        "buttons": torch.randint(0, 2, (batch_size, seq_length, 5)).bool(),  # [B, T, 5]
    }

    # Should not raise
    logp = compute_action_logprob(outputs, actions)

    # Check output shape
    assert logp.shape == (
        batch_size,
        seq_length,
    ), f"Expected shape ({batch_size}, {seq_length}), got {logp.shape}"
    assert torch.isfinite(logp).all(), "Log probs contain NaN or Inf"


def test_action_entropy_shapes():
    """Test that entropy computation returns correct shapes."""
    batch_size = 32
    seq_length = 16

    outputs = {
        "main_stick": torch.randn(batch_size, seq_length, 64),
        "c_stick": torch.randn(batch_size, seq_length, 9),
        "shoulder": torch.randn(batch_size, seq_length, 5),
        "buttons": torch.randn(batch_size, seq_length, 5),
    }

    entropy = compute_action_entropy(outputs)

    assert entropy.shape == (
        batch_size,
        seq_length,
    ), f"Expected shape ({batch_size}, {seq_length}), got {entropy.shape}"
    assert (entropy >= 0).all(), "Entropy should be non-negative"


def test_ppo_loss_shapes(column_map, mock_policy):
    """Test that PPO loss computation handles all shapes correctly."""
    batch_size = 16
    seq_length = 16

    # Create mock batch data
    batch_features = make_valid_features(batch_size, seq_length, column_map)

    batch_actions = {
        "main_idx": torch.randint(0, 64, (batch_size, seq_length)),
        "c_idx": torch.randint(0, 9, (batch_size, seq_length)),
        "shoulder_idx": torch.randint(0, 5, (batch_size, seq_length)),
        "buttons": torch.randint(0, 2, (batch_size, seq_length, 5)).bool(),
    }

    old_logps = torch.randn(batch_size, seq_length)
    advantages = torch.randn(batch_size, seq_length)
    returns = torch.randn(batch_size, seq_length)

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
    expected_keys = [
        "total",
        "policy",
        "value",
        "entropy",
        "ratio_mean",
        "ratio_std",
        "approx_kl",
    ]
    for key in expected_keys:
        assert key in loss_dict, f"Missing key: {key}"

    # Check loss is scalar
    assert loss_dict["total"].shape == (), "Total loss should be scalar"
    assert torch.isfinite(loss_dict["total"]), "Total loss is NaN or Inf"


def test_button_type_handling():
    """Test that button actions handle uint8/bool conversion correctly."""
    batch_size = 32
    seq_length = 8

    outputs = {
        "main_stick": torch.randn(batch_size, seq_length, 64),
        "c_stick": torch.randn(batch_size, seq_length, 9),
        "shoulder": torch.randn(batch_size, seq_length, 5),
        "buttons": torch.randn(batch_size, seq_length, 5),
        "value": torch.randn(batch_size, seq_length, 1),
    }

    # Test with uint8 (should be converted to bool internally)
    actions_uint8 = {
        "main_idx": torch.randint(0, 64, (batch_size, seq_length)),
        "c_idx": torch.randint(0, 9, (batch_size, seq_length)),
        "shoulder_idx": torch.randint(0, 5, (batch_size, seq_length)),
        "buttons": torch.randint(0, 2, (batch_size, seq_length, 5), dtype=torch.uint8),
    }

    # Should not raise warning or error
    compute_action_logprob(outputs, actions_uint8)


def test_clipping_behavior():
    """Test that PPO clipping works as expected."""
    batch_size = 16
    seq_length = 8

    column_map = ColumnMap(get_feature_names(), get_target_names())
    config = Config()
    config.model.n_layer = 1
    config.model.n_embd = 32
    config.model.block_size = max(config.model.block_size, seq_length)
    policy = GPT(config)

    batch_features = make_valid_features(batch_size, seq_length, column_map)
    batch_actions = {
        "main_idx": torch.randint(0, 64, (batch_size, seq_length)),
        "c_idx": torch.randint(0, 9, (batch_size, seq_length)),
        "shoulder_idx": torch.randint(0, 5, (batch_size, seq_length)),
        "buttons": torch.randint(0, 2, (batch_size, seq_length, 5)).bool(),
    }

    old_logps = torch.randn(batch_size, seq_length)
    advantages = torch.randn(batch_size, seq_length)
    returns = torch.randn(batch_size, seq_length)

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


def test_action_shape_mismatch_raises():
    """Ensure mismatched action shapes trigger an error (regression guard)."""
    batch_size = 4
    seq_length = 3

    outputs = {
        "main_stick": torch.randn(batch_size, seq_length, 64),
        "c_stick": torch.randn(batch_size, seq_length, 9),
        "shoulder": torch.randn(batch_size, seq_length, 5),
        "buttons": torch.randn(batch_size, seq_length, 5),
        "value": torch.randn(batch_size, seq_length, 1),
    }

    # Incorrectly sliced actions (previous bug)
    bad_actions = {
        "main_idx": torch.randint(0, 64, (batch_size,)),
        "c_idx": torch.randint(0, 9, (batch_size,)),
        "shoulder_idx": torch.randint(0, 5, (batch_size,)),
        "buttons": torch.randint(0, 2, (batch_size, 5)).bool(),
    }

    with pytest.raises(ValueError):
        compute_action_logprob(outputs, bad_actions)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
