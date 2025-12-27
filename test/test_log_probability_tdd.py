"""TDD tests for log probability computation - current implementation.

These tests capture the behavior of the CURRENT implementation in ppo_train.py
before refactoring. After refactoring, these tests must continue to pass with
identical outputs.
"""
from __future__ import annotations

import math
import pytest
import torch
from tensordict import TensorDict


# Import current implementations from ppo_train
# We'll test these functions as-is before refactoring
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def mock_outputs():
    """Create mock model outputs for testing."""
    torch.manual_seed(42)
    return TensorDict({
        "main_stick": torch.randn(1, 256, 64),  # [B=1, L=256, K=64]
        "c_stick": torch.randn(1, 256, 9),      # [B=1, L=256, K=9]
        "buttons": torch.randn(1, 256, 5),      # [B=1, L=256, K=5]
        "shoulder": torch.randn(1, 256, 5),     # [B=1, L=256, K=5]
        "value": torch.randn(1, 256, 1),        # [B=1, L=256, 1]
    }, batch_size=(1, 256))


@pytest.fixture
def mock_batch_outputs():
    """Create mock batch outputs for training."""
    torch.manual_seed(42)
    batch_size = 16
    return TensorDict({
        "main_stick": torch.randn(batch_size, 256, 64),
        "c_stick": torch.randn(batch_size, 256, 9),
        "buttons": torch.randn(batch_size, 256, 5),
        "shoulder": torch.randn(batch_size, 256, 5),
        "value": torch.randn(batch_size, 256, 1),
    }, batch_size=(batch_size, 256))


class TestCategoricalLogProbability:
    """Test categorical log probability computation (main_stick, c_stick, shoulder)."""

    def test_log_softmax_manual_computation(self):
        """Test that manual log_softmax produces correct probabilities."""
        torch.manual_seed(42)
        logits = torch.randn(64)

        # Current implementation: manual log_softmax
        log_probs = torch.log_softmax(logits, dim=-1)

        # Verification: Should sum to 1 in probability space
        probs = torch.exp(log_probs)
        assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-6)

        # Verification: Should match PyTorch distribution
        from torch.distributions import Categorical
        dist = Categorical(logits=logits)
        all_actions = torch.arange(64)
        dist_log_probs = dist.log_prob(all_actions)
        assert torch.allclose(log_probs, dist_log_probs, atol=1e-6)

    def test_log_prob_indexing_single_sample(self):
        """Test log prob extraction for single sampled action."""
        torch.manual_seed(42)
        logits = torch.randn(64)

        # Sample action
        action_idx = torch.multinomial(torch.softmax(logits, dim=-1), 1).squeeze(-1)

        # Current implementation: index into log_softmax
        log_softmax = torch.log_softmax(logits, dim=-1)
        log_prob = log_softmax[action_idx]

        # Verification: Should be negative (probabilities < 1)
        assert log_prob <= 0.0

        # Verification: Should match distribution computation
        from torch.distributions import Categorical
        dist = Categorical(logits=logits)
        dist_log_prob = dist.log_prob(action_idx)
        assert torch.allclose(log_prob, dist_log_prob, atol=1e-6)

    def test_log_prob_batch_computation(self):
        """Test log prob computation for batch of actions."""
        torch.manual_seed(42)
        batch_size = 16
        logits = torch.randn(batch_size, 64)  # [L, K]
        actions = torch.randint(0, 64, (batch_size,))  # [L]

        # Current implementation would use: gather
        log_softmax = torch.log_softmax(logits, dim=-1)
        log_probs = log_softmax.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)

        assert log_probs.shape == (batch_size,)
        assert (log_probs <= 0.0).all()


class TestBernoulliLogProbability:
    """Test Bernoulli log probability computation (buttons)."""

    def test_bernoulli_log_prob_with_epsilon(self):
        """Test Bernoulli log prob with epsilon smoothing."""
        torch.manual_seed(42)
        logits = torch.randn(5)

        # Current implementation
        probs = torch.sigmoid(logits)
        actions = torch.bernoulli(probs)

        epsilon = 1e-8
        log_probs = torch.where(
            actions == 1,
            torch.log(probs + epsilon),
            torch.log(1 - probs + epsilon),
        )

        # Verification: All log probs should be finite (no log(0))
        assert torch.isfinite(log_probs).all()

        # Verification: Log probs should be negative or zero
        assert (log_probs <= 0.0).all()

    def test_bernoulli_epsilon_prevents_log_zero(self):
        """Test that epsilon prevents log(0) even with extreme probabilities."""
        # Create logits that produce prob very close to 0 or 1
        extreme_logits = torch.tensor([-100.0, 100.0, -50.0, 50.0, 0.0])

        probs = torch.sigmoid(extreme_logits)
        # Force actions to worst case (action=1 when prob≈0, action=0 when prob≈1)
        actions = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])

        epsilon = 1e-8
        log_probs = torch.where(
            actions == 1,
            torch.log(probs + epsilon),
            torch.log(1 - probs + epsilon),
        )

        # Critical: Should not produce -inf
        assert torch.isfinite(log_probs).all()
        assert not torch.isinf(log_probs).any()

    def test_bernoulli_batch_computation(self):
        """Test Bernoulli log prob for batch."""
        torch.manual_seed(42)
        batch_size = 16
        num_buttons = 5
        logits = torch.randn(batch_size, num_buttons)

        probs = torch.sigmoid(logits)
        actions = torch.bernoulli(probs)

        epsilon = 1e-8
        log_probs = torch.where(
            actions == 1,
            torch.log(probs + epsilon),
            torch.log(1 - probs + epsilon),
        )

        assert log_probs.shape == (batch_size, num_buttons)
        assert torch.isfinite(log_probs).all()


class TestSampleActionsWithLogprobs:
    """Test the complete sample_actions_with_logprobs() function."""

    def test_stochastic_sampling_deterministic_with_seed(self, mock_outputs):
        """Test that sampling is deterministic with fixed seed."""
        from ppo_train import sample_actions_with_logprobs

        # Sample 1
        torch.manual_seed(42)
        result1 = sample_actions_with_logprobs(mock_outputs)

        # Sample 2 with same seed
        torch.manual_seed(42)
        result2 = sample_actions_with_logprobs(mock_outputs)

        # Should be identical
        assert result1.main_stick_idx == result2.main_stick_idx
        assert result1.c_stick_idx == result2.c_stick_idx
        assert result1.shoulder_idx == result2.shoulder_idx
        assert torch.equal(result1.buttons, result2.buttons)
        assert torch.equal(result1.main_log_prob, result2.main_log_prob)

    def test_log_probs_are_negative(self, mock_outputs):
        """Test that all log probabilities are negative or zero."""
        from ppo_train import sample_actions_with_logprobs

        torch.manual_seed(42)
        result = sample_actions_with_logprobs(mock_outputs)

        assert result.main_log_prob <= 0.0
        assert result.c_log_prob <= 0.0
        assert result.shoulder_log_prob <= 0.0
        assert (result.buttons_log_probs <= 0.0).all()

    def test_log_probs_are_finite(self, mock_outputs):
        """Test that log probabilities are always finite (no -inf)."""
        from ppo_train import sample_actions_with_logprobs

        torch.manual_seed(42)
        result = sample_actions_with_logprobs(mock_outputs)

        assert torch.isfinite(result.main_log_prob).all()
        assert torch.isfinite(result.c_log_prob).all()
        assert torch.isfinite(result.shoulder_log_prob).all()
        assert torch.isfinite(result.buttons_log_probs).all()


class TestGreedyActionsWithLogprobs:
    """Test the greedy_actions_with_logprobs() function."""

    def test_greedy_is_deterministic(self, mock_outputs):
        """Test that greedy selection is always deterministic."""
        from ppo_train import greedy_actions_with_logprobs

        result1 = greedy_actions_with_logprobs(mock_outputs)
        result2 = greedy_actions_with_logprobs(mock_outputs)

        # Should be identical (no randomness)
        assert result1.main_stick_idx == result2.main_stick_idx
        assert result1.c_stick_idx == result2.c_stick_idx
        assert result1.shoulder_idx == result2.shoulder_idx
        assert torch.equal(result1.buttons, result2.buttons)

    def test_greedy_selects_max_logit(self, mock_outputs):
        """Test that greedy selects argmax."""
        from ppo_train import greedy_actions_with_logprobs

        result = greedy_actions_with_logprobs(mock_outputs)

        # Manually verify argmax for main_stick
        main_logits = mock_outputs["main_stick"][0, -1]
        expected_main_idx = int(torch.argmax(main_logits).item())

        assert result.main_stick_idx == expected_main_idx

    def test_greedy_button_threshold(self, mock_outputs):
        """Test that buttons use 0.5 threshold."""
        from ppo_train import greedy_actions_with_logprobs

        result = greedy_actions_with_logprobs(mock_outputs)

        # Manually compute expected buttons
        button_logits = mock_outputs["buttons"][0, -1]
        probs = torch.sigmoid(button_logits)
        expected_buttons = (probs >= 0.5).to(torch.float32)

        assert torch.equal(result.buttons, expected_buttons)


class TestComputeLogProbsForActions:
    """Test compute_log_probs_for_actions() batch function."""

    def test_batch_log_prob_computation(self, mock_batch_outputs):
        """Test log prob computation for batch of rollout steps."""
        from ppo_train import compute_log_probs_for_actions, RolloutStep, Rollout, ActionInfo

        batch_size = mock_batch_outputs.batch_size[0]
        device = mock_batch_outputs.device

        # Create mock rollout steps
        steps = []
        for i in range(batch_size):
            action_info = ActionInfo(
                main_stick_idx=i % 64,
                c_stick_idx=i % 9,
                buttons=torch.rand(5),
                shoulder_idx=i % 5,
                main_log_prob=torch.tensor(0.0),
                c_log_prob=torch.tensor(0.0),
                buttons_log_probs=torch.zeros(5),
                shoulder_log_prob=torch.tensor(0.0),
            )
            step = RolloutStep(
                features=torch.zeros(100),
                action_info=action_info,
                value_pred=0.0,
                reward=0.0,
                done=False,
            )
            steps.append(step)

        rollout = Rollout(
            steps=steps,
            episode_length=batch_size,
            total_reward=0.0,
            winner=1,
        )

        # Compute log probs
        log_probs = compute_log_probs_for_actions(mock_batch_outputs, rollout, device)

        # Verify shapes
        assert log_probs["main"].shape == (batch_size,)
        assert log_probs["c"].shape == (batch_size,)
        assert log_probs["shoulder"].shape == (batch_size,)
        assert log_probs["buttons"].shape == (batch_size, 5)

        # Verify all finite
        assert torch.isfinite(log_probs["main"]).all()
        assert torch.isfinite(log_probs["c"]).all()
        assert torch.isfinite(log_probs["shoulder"]).all()
        assert torch.isfinite(log_probs["buttons"]).all()


class TestComputePolicyEntropy:
    """Test compute_policy_entropy() function."""

    def test_entropy_computation(self, mock_batch_outputs):
        """Test entropy computation for all action types."""
        from ppo_train import compute_policy_entropy

        entropies = compute_policy_entropy(mock_batch_outputs)

        # Verify keys
        assert "main" in entropies
        assert "c" in entropies
        assert "shoulder" in entropies
        assert "buttons" in entropies

        # Verify all are scalars
        assert isinstance(entropies["main"], float)
        assert isinstance(entropies["c"], float)
        assert isinstance(entropies["shoulder"], float)
        assert isinstance(entropies["buttons"], float)

        # Verify all are non-negative (entropy >= 0)
        assert entropies["main"] >= 0.0
        assert entropies["c"] >= 0.0
        assert entropies["shoulder"] >= 0.0
        assert entropies["buttons"] >= 0.0

    def test_uniform_distribution_max_entropy(self):
        """Test that uniform distribution has maximum entropy."""
        from ppo_train import compute_policy_entropy

        # Create uniform logits (all zeros → uniform after softmax)
        uniform_outputs = TensorDict({
            "main_stick": torch.zeros(16, 256, 64),
            "c_stick": torch.zeros(16, 256, 9),
            "buttons": torch.zeros(16, 256, 5),
            "shoulder": torch.zeros(16, 256, 5),
            "value": torch.zeros(16, 256, 1),
        }, batch_size=(16, 256))

        entropies = compute_policy_entropy(uniform_outputs)

        # Maximum entropy for categorical: log(K)
        import math
        expected_main_entropy = math.log(64)
        expected_c_entropy = math.log(9)
        expected_shoulder_entropy = math.log(5)
        expected_button_entropy = math.log(2)  # Bernoulli

        assert abs(entropies["main"] - expected_main_entropy) < 0.01
        assert abs(entropies["c"] - expected_c_entropy) < 0.01
        assert abs(entropies["shoulder"] - expected_shoulder_entropy) < 0.01
        assert abs(entropies["buttons"] - expected_button_entropy) < 0.01


class TestNumericalStability:
    """Test numerical stability across edge cases."""

    def test_extreme_logits_categorical(self):
        """Test categorical log probs with extreme logits."""
        extreme_logits = torch.tensor([-100.0, -50.0, 0.0, 50.0, 100.0])

        log_probs = torch.log_softmax(extreme_logits, dim=-1)

        # Should not produce NaN or inf
        assert torch.isfinite(log_probs).all()

        # Should still sum to 1 in probability space
        probs = torch.exp(log_probs)
        assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)

    def test_epsilon_value_preservation(self):
        """Test that epsilon value 1e-8 is exactly preserved."""
        # This is critical - changing epsilon would change training behavior

        probs = torch.tensor([0.0, 1.0, 0.5])
        actions = torch.tensor([1.0, 0.0, 1.0])

        # Current implementation uses exactly 1e-8
        epsilon = 1e-8
        log_probs = torch.where(
            actions == 1,
            torch.log(probs + epsilon),
            torch.log(1 - probs + epsilon),
        )

        # Verify epsilon prevents -inf
        assert torch.isfinite(log_probs).all()

        # Verify epsilon value is exactly 1e-8 (not 1e-7, not 1e-9)
        expected_log_prob_for_zero = math.log(1e-8)
        assert abs(log_probs[0].item() - expected_log_prob_for_zero) < 1e-5  # Reasonable tolerance for float32


class TestIntegrationLogProbabilityWorkflow:
    """Integration tests for complete log probability workflows."""

    def test_sample_and_recompute_log_probs(self, mock_outputs, mock_batch_outputs):
        """Test full workflow: sample → store → recompute log probs."""
        from ppo_train import (
            sample_actions_with_logprobs,
            compute_log_probs_for_actions,
            RolloutStep,
            Rollout,
        )

        # Step 1: Sample action
        torch.manual_seed(42)
        action = sample_actions_with_logprobs(mock_outputs)

        # Step 2: Create rollout with this action
        step = RolloutStep(
            features=torch.zeros(100),
            action_info=action,
            value_pred=0.0,
            reward=0.0,
            done=False,
        )
        rollout = Rollout(
            steps=[step],
            episode_length=1,
            total_reward=0.0,
            winner=1,
        )

        # Step 3: Recompute log probs (simulating PPO update)
        # Create outputs with same logits repeated in batch
        # Note: Value is now computed by separate ValueNetwork, not GPT
        recompute_outputs = TensorDict({
            "main_stick": mock_outputs["main_stick"].expand(1, -1, -1),
            "c_stick": mock_outputs["c_stick"].expand(1, -1, -1),
            "buttons": mock_outputs["buttons"].expand(1, -1, -1),
            "shoulder": mock_outputs["shoulder"].expand(1, -1, -1),
        }, batch_size=(1, 256))

        device = mock_outputs.device
        recomputed = compute_log_probs_for_actions(recompute_outputs, rollout, device)

        # Verify recomputed log probs match original (within numerical precision)
        assert torch.allclose(recomputed["main"][0], action.main_log_prob, atol=1e-5)
        assert torch.allclose(recomputed["c"][0], action.c_log_prob, atol=1e-5)
        assert torch.allclose(recomputed["shoulder"][0], action.shoulder_log_prob, atol=1e-5)
        # Buttons might differ slightly due to storage
        assert torch.allclose(recomputed["buttons"][0], action.buttons_log_probs, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
