"""Tests for PPO sequence handling - proper transformer training with full sequences."""

import pytest
import torch

from ppo.trajectory import Step, Trajectory
from ppo.ppo_loss import compute_total_ppo_loss, compute_log_probs, compute_entropy


class TestComputeTotalPPOLossSequence:
    """Tests for compute_total_ppo_loss with sequence inputs."""

    def test_sequence_loss_basic(self):
        """Test basic sequence loss computation."""
        B, seq_len = 4, 32
        num_main = 64
        num_c = 9
        num_buttons = 5
        num_shoulder = 5

        new_action_logits = {
            "main_stick": torch.randn(B, seq_len, num_main),
            "c_stick": torch.randn(B, seq_len, num_c),
            "buttons": torch.randn(B, seq_len, num_buttons),
            "shoulder": torch.randn(B, seq_len, num_shoulder),
        }
        old_action_logits = {
            "main_stick": torch.randn(B, seq_len, num_main),
            "c_stick": torch.randn(B, seq_len, num_c),
            "buttons": torch.randn(B, seq_len, num_buttons),
            "shoulder": torch.randn(B, seq_len, num_shoulder),
        }
        actions_taken = {
            "main_stick": torch.randint(0, num_main, (B, seq_len)),
            "c_stick": torch.randint(0, num_c, (B, seq_len)),
            "buttons": torch.randint(0, 2, (B, seq_len, num_buttons)).float(),
            "shoulder": torch.randint(0, num_shoulder, (B, seq_len)),
        }

        new_values = torch.randn(B, seq_len)
        old_values = torch.randn(B, seq_len)
        old_log_probs = torch.randn(B, seq_len) - 5.0  # Negative log probs
        advantages = torch.randn(B, seq_len)
        returns = torch.randn(B, seq_len)

        loss, metrics = compute_total_ppo_loss(
            new_action_logits=new_action_logits,
            new_values=new_values,
            old_values=old_values,
            actions_taken=actions_taken,
            old_log_probs=old_log_probs,
            advantages=advantages,
            returns=returns,
        )

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert "ppo/total_loss" in metrics
        assert "ppo/num_valid_positions" in metrics
        assert metrics["ppo/num_valid_positions"] == B * seq_len

    def test_sequence_loss_with_mask(self):
        """Test sequence loss with loss_mask."""
        B, seq_len = 4, 32
        num_main = 64
        num_c = 9
        num_buttons = 5
        num_shoulder = 5

        new_action_logits = {
            "main_stick": torch.randn(B, seq_len, num_main),
            "c_stick": torch.randn(B, seq_len, num_c),
            "buttons": torch.randn(B, seq_len, num_buttons),
            "shoulder": torch.randn(B, seq_len, num_shoulder),
        }
        old_action_logits = {
            "main_stick": torch.randn(B, seq_len, num_main),
            "c_stick": torch.randn(B, seq_len, num_c),
            "buttons": torch.randn(B, seq_len, num_buttons),
            "shoulder": torch.randn(B, seq_len, num_shoulder),
        }
        actions_taken = {
            "main_stick": torch.randint(0, num_main, (B, seq_len)),
            "c_stick": torch.randint(0, num_c, (B, seq_len)),
            "buttons": torch.randint(0, 2, (B, seq_len, num_buttons)).float(),
            "shoulder": torch.randint(0, num_shoulder, (B, seq_len)),
        }

        new_values = torch.randn(B, seq_len)
        old_values = torch.randn(B, seq_len)
        old_log_probs = torch.randn(B, seq_len) - 5.0
        advantages = torch.randn(B, seq_len)
        returns = torch.randn(B, seq_len)

        # Mask out first 8 positions (warmup)
        loss_mask = torch.ones(B, seq_len, dtype=torch.bool)
        loss_mask[:, :8] = False

        loss, metrics = compute_total_ppo_loss(
            new_action_logits=new_action_logits,
            new_values=new_values,
            old_values=old_values,
            actions_taken=actions_taken,
            old_log_probs=old_log_probs,
            advantages=advantages,
            returns=returns,
            loss_mask=loss_mask,
        )

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        # Should only count non-masked positions
        expected_valid = B * (seq_len - 8)
        assert metrics["ppo/num_valid_positions"] == expected_valid

    def test_sequence_loss_all_masked_returns_zero(self):
        """Test that fully masked input returns zero loss."""
        B, seq_len = 4, 32
        num_main = 64

        new_action_logits = {"main_stick": torch.randn(B, seq_len, num_main)}
        old_action_logits = {"main_stick": torch.randn(B, seq_len, num_main)}
        actions_taken = {"main_stick": torch.randint(0, num_main, (B, seq_len))}

        new_values = torch.randn(B, seq_len)
        old_values = torch.randn(B, seq_len)
        old_log_probs = torch.randn(B, seq_len) - 5.0
        advantages = torch.randn(B, seq_len)
        returns = torch.randn(B, seq_len)

        # All masked
        loss_mask = torch.zeros(B, seq_len, dtype=torch.bool)

        loss, metrics = compute_total_ppo_loss(
            new_action_logits=new_action_logits,
            new_values=new_values,
            old_values=old_values,
            actions_taken=actions_taken,
            old_log_probs=old_log_probs,
            advantages=advantages,
            returns=returns,
            loss_mask=loss_mask,
        )

        assert loss.item() == 0.0
        assert metrics["ppo/num_valid_positions"] == 0

    def test_sequence_loss_gradient_flow(self):
        """Test that gradients flow correctly through sequence loss."""
        B, seq_len = 2, 16
        num_main = 64

        new_action_logits = {
            "main_stick": torch.randn(B, seq_len, num_main, requires_grad=True)
        }
        old_action_logits = {"main_stick": torch.randn(B, seq_len, num_main)}
        actions_taken = {"main_stick": torch.randint(0, num_main, (B, seq_len))}

        new_values = torch.randn(B, seq_len, requires_grad=True)
        old_values = torch.randn(B, seq_len)
        old_log_probs = torch.randn(B, seq_len) - 5.0
        advantages = torch.randn(B, seq_len)
        returns = torch.randn(B, seq_len)

        loss_mask = torch.ones(B, seq_len, dtype=torch.bool)
        loss_mask[:, :4] = False  # Mask warmup

        loss, _ = compute_total_ppo_loss(
            new_action_logits=new_action_logits,
            new_values=new_values,
            old_values=old_values,
            actions_taken=actions_taken,
            old_log_probs=old_log_probs,
            advantages=advantages,
            returns=returns,
            loss_mask=loss_mask,
        )

        loss.backward()

        # Check gradients exist
        assert new_action_logits["main_stick"].grad is not None
        assert new_values.grad is not None
        assert not torch.isnan(new_action_logits["main_stick"].grad).any()
        assert not torch.isnan(new_values.grad).any()


class TestComputeLogProbs:
    """Tests for compute_log_probs helper."""

    def test_log_probs_categorical(self):
        """Test log prob computation for categorical actions."""
        B = 8
        num_classes = 64

        action_logits = {
            "main_stick": torch.randn(B, num_classes),
        }
        actions_taken = {
            "main_stick": torch.randint(0, num_classes, (B,)),
        }

        log_probs = compute_log_probs(action_logits, actions_taken)

        assert log_probs.shape == (B,)
        assert (log_probs <= 0).all()  # Log probs should be negative
        assert not torch.isnan(log_probs).any()

    def test_log_probs_buttons(self):
        """Test log prob computation for binary button actions."""
        B = 8
        num_buttons = 5

        action_logits = {
            "buttons": torch.randn(B, num_buttons),
        }
        actions_taken = {
            "buttons": torch.randint(0, 2, (B, num_buttons)).float(),
        }

        log_probs = compute_log_probs(action_logits, actions_taken)

        assert log_probs.shape == (B,)
        assert not torch.isnan(log_probs).any()


class TestComputeEntropy:
    """Tests for compute_entropy helper."""

    def test_entropy_categorical(self):
        """Test entropy computation for categorical distributions."""
        B = 8
        num_classes = 64

        action_logits = {
            "main_stick": torch.randn(B, num_classes),
        }

        entropy = compute_entropy(action_logits)

        assert entropy.shape == (B,)
        assert (entropy >= 0).all()  # Entropy should be non-negative
        assert not torch.isnan(entropy).any()

    def test_entropy_uniform_is_high(self):
        """Test that uniform distribution has high entropy."""
        B = 8
        num_classes = 64

        # Uniform logits
        action_logits = {
            "main_stick": torch.zeros(B, num_classes),
        }

        entropy = compute_entropy(action_logits)

        # Entropy of uniform distribution over 64 classes is log(64) ~ 4.16
        expected_entropy = torch.log(torch.tensor(num_classes, dtype=torch.float32))
        assert torch.allclose(entropy, expected_entropy.expand(B), atol=0.01)

    def test_entropy_peaked_is_low(self):
        """Test that peaked distribution has low entropy."""
        B = 8
        num_classes = 64

        # Very peaked logits (one class has high probability)
        action_logits = {
            "main_stick": torch.full((B, num_classes), -100.0),
        }
        action_logits["main_stick"][:, 0] = 100.0  # First class has all probability

        entropy = compute_entropy(action_logits)

        # Entropy should be very close to 0
        assert (entropy < 0.1).all()
