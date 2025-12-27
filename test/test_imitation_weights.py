import pytest
import torch
import torch.nn.functional as F
from config.imitation_config import ImitationConfig
from train.imitation_weights import (
    compute_value_weighted_weights,
    compute_value_filter_weights,
    compute_advantage_weights,
    compute_hybrid_weights,
    compute_imitation_weights,
)


@pytest.fixture
def dummy_values():
    # Shape [B, L] = [2, 5]
    return torch.tensor(
        [[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1]], dtype=torch.float32
    )


def test_compute_value_weighted_weights_linear(dummy_values):
    weights = compute_value_weighted_weights(
        dummy_values, k=1.0, temperature=1.0, use_exp=False
    )
    assert weights.shape == dummy_values.shape
    assert torch.all(weights >= 0)
    # Normalized to mean=1
    assert torch.abs(weights.mean() - 1.0) < 1e-5

    # Check linear relationship: higher value -> higher weight
    # In first row, values increase, so weights should increase
    assert weights[0, 0] < weights[0, 4]
    # In second row, values decrease, so weights should decrease
    assert weights[1, 0] > weights[1, 4]


def test_compute_value_weighted_weights_exp(dummy_values):
    weights = compute_value_weighted_weights(
        dummy_values, k=1.0, temperature=1.0, use_exp=True
    )
    assert weights.shape == dummy_values.shape
    assert torch.all(weights > 0)  # exp is always positive
    assert torch.abs(weights.mean() - 1.0) < 1e-5

    # Check relationship
    assert weights[0, 0] < weights[0, 4]
    assert weights[1, 0] > weights[1, 4]


def test_compute_value_filter_weights_hard(dummy_values):
    # Percentile 40, means top 60% kept.
    # Flat values: 0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.4, 0.3, 0.2, 0.1
    # Sorted: 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 0.5, 0.5
    # 40th percentile is roughly 0.2-0.3 boundary.
    # Let's use median (50th percentile) to be safe. Median is 0.3.

    weights = compute_value_filter_weights(
        dummy_values, percentile=50.0, soft=False, temperature=1.0
    )

    assert weights.shape == dummy_values.shape
    # Should be binary
    unique_vals = torch.unique(weights)
    assert torch.all(torch.isin(unique_vals, torch.tensor([0.0, 1.0])))

    # Values >= median (0.3) should be 1.0
    threshold = torch.quantile(dummy_values.flatten(), 0.5)
    expected = (dummy_values >= threshold).float()
    assert torch.allclose(weights, expected)


def test_compute_value_filter_weights_soft(dummy_values):
    weights = compute_value_filter_weights(
        dummy_values, percentile=50.0, soft=True, temperature=1.0
    )
    assert weights.shape == dummy_values.shape
    assert torch.all(weights >= 0.0)
    assert torch.all(weights <= 1.0)

    # Check monotonicity with respect to values
    # Higher values should have higher weights
    assert weights[0, 0] < weights[0, 4]


def test_compute_advantage_weights(dummy_values):
    # Row 0: 0.1, 0.2... increasing values -> positive advantage
    # Row 1: 0.5, 0.4... decreasing values -> negative advantage

    weights = compute_advantage_weights(dummy_values, alpha=1.0)

    assert weights.shape == dummy_values.shape
    assert torch.all(weights >= 0)

    # Row 0: increasing values -> positive advantage -> weight > 1.0 (boosted)
    assert torch.all(weights[0, :-1] > 1.0)

    # Row 1: decreasing values -> negative advantage -> weight < 1.0 (suppressed)
    assert torch.all(weights[1, :-1] < 1.0)

    # Positive advantage weighted higher than negative
    assert weights[0, 0] > weights[1, 0]


def test_compute_hybrid_weights(dummy_values):
    # Hybrid: value_weighted (0.5) + value_filter (0.5)
    config = ImitationConfig(
        strategy="hybrid",
        hybrid_strategies=["value_weighted", "value_filter"],
        hybrid_weights=[0.5, 0.5],
        value_k=1.0,
        filter_percentile=50.0,
        filter_soft=False,
    )

    weights = compute_hybrid_weights(dummy_values, config)
    assert weights.shape == dummy_values.shape
    # Mean should be approx 1
    assert torch.abs(weights.mean() - 1.0) < 1e-5


def test_compute_hybrid_weights_mismatch_error(dummy_values):
    # Although config validation catches this, we can also test the function directly
    # if we bypass config validation or manually construct an invalid state if pydantic allows (it shouldn't usually)
    # But here we pass config. config has validator.
    # We can try to manually patch config if needed, but let's rely on pydantic raising error at init
    with pytest.raises(ValueError):
        ImitationConfig(
            strategy="hybrid",
            hybrid_strategies=["value_weighted"],
            hybrid_weights=[0.5, 0.5],
        )


def test_compute_imitation_weights_integration(dummy_values):
    # Mock input X: [B, L, F]
    # Let's say F=10, and value_idx=5
    B, L = dummy_values.shape
    F_dim = 10
    X = torch.zeros(B, L, F_dim)
    value_idx = 5
    X[:, :, value_idx] = dummy_values

    config = ImitationConfig(strategy="value_weighted", value_k=2.0)

    weights = compute_imitation_weights(X, value_idx, config)
    assert weights.shape == (B, L)
    assert torch.abs(weights.mean() - 1.0) < 1e-5

    # Test uniform
    config_uniform = ImitationConfig(strategy="uniform")
    weights_uniform = compute_imitation_weights(X, value_idx, config_uniform)
    assert torch.all(weights_uniform == 1.0)

    # Test value_filter
    config_filter = ImitationConfig(strategy="value_filter", filter_percentile=50.0)
    weights_filter = compute_imitation_weights(X, value_idx, config_filter)
    assert weights_filter.shape == (B, L)

    # Test value_advantage
    config_adv = ImitationConfig(strategy="value_advantage")
    weights_adv = compute_imitation_weights(X, value_idx, config_adv)
    assert weights_adv.shape == (B, L)

    # Test hybrid
    config_hybrid = ImitationConfig(strategy="hybrid")
    weights_hybrid = compute_imitation_weights(X, value_idx, config_hybrid)
    assert weights_hybrid.shape == (B, L)


def test_compute_imitation_weights_unknown_strategy(dummy_values):
    B, L = dummy_values.shape
    X = torch.zeros(B, L, 10)
    X[:, :, 0] = dummy_values

    # We have to bypass pydantic validation to test the runtime error in the function
    # or use a mock object that looks like config
    class MockConfig:
        strategy = "unknown_strategy"
        value_k = 1.0
        value_temperature = 1.0
        value_use_exp = True

    with pytest.raises(ValueError, match="Unknown imitation strategy"):
        compute_imitation_weights(X, 0, MockConfig())


def test_numerical_stability_large_values():
    # Test with large values to ensure exp doesn't explode before normalization
    values = torch.tensor([[1000.0, 1001.0], [1000.0, 999.0]])
    weights = compute_value_weighted_weights(
        values, k=1.0, temperature=1.0, use_exp=True
    )
    assert not torch.isnan(weights).any()
    assert not torch.isinf(weights).any()
    assert torch.abs(weights.mean() - 1.0) < 1e-5


def test_short_sequence_advantage():
    # L=1: no temporal difference possible
    values = torch.tensor([[0.5]], dtype=torch.float32)
    # advantages all 0 -> weights all 1.0
    weights = compute_advantage_weights(values, alpha=1.0)
    assert weights.shape == values.shape
    assert torch.all(weights == 1.0)


def test_hybrid_unknown_strategy_direct():
    values = torch.tensor([[0.1, 0.2]], dtype=torch.float32)

    # Mock config object with invalid strategy string in list
    # We can't use ImitationConfig easily here because of validation, so we use a simple object
    class MockConfig:
        hybrid_strategies = ["unknown_strategy"]
        hybrid_weights = [1.0]

    with pytest.raises(ValueError, match="Unknown strategy in hybrid"):
        compute_hybrid_weights(values, MockConfig())


def test_hybrid_weights_length_mismatch(dummy_values):
    # Directly test compute_hybrid_weights raising error when lengths mismatch
    # We need a mock config that passes the validator or bypasses it
    class MockConfig:
        hybrid_strategies = ["value_weighted", "value_filter"]
        hybrid_weights = [0.5]  # Mismatch

    with pytest.raises(
        ValueError, match="hybrid_strategies and hybrid_weights must have same length"
    ):
        compute_hybrid_weights(dummy_values, MockConfig())


def test_hybrid_weights_with_advantage(dummy_values):
    # Test hybrid strategy including "value_advantage"
    class MockConfig:
        hybrid_strategies = ["value_advantage"]
        hybrid_weights = [1.0]
        advantage_alpha = 1.0

    weights = compute_hybrid_weights(dummy_values, MockConfig())
    assert weights.shape == dummy_values.shape
    assert torch.abs(weights.mean() - 1.0) < 1e-5


def test_hybrid_weights_with_uniform(dummy_values):
    # Test hybrid strategy including "uniform"
    class MockConfig:
        hybrid_strategies = ["uniform"]
        hybrid_weights = [1.0]

    weights = compute_hybrid_weights(dummy_values, MockConfig())
    assert weights.shape == dummy_values.shape
    assert torch.allclose(weights, torch.ones_like(weights))
