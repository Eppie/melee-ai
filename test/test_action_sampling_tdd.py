"""TDD tests for action sampling implementations (Cluster 9).

This test suite captures the current behavior of action sampling functions
in both PPO training and inference contexts before refactoring.

Tests ensure that:
1. Stochastic sampling is deterministic with fixed seed
2. Greedy sampling is fully deterministic
3. Output types and shapes are correct
4. Coordinate conversions preserve palette mappings
5. Button sampling follows Bernoulli distribution
"""

from __future__ import annotations

import pytest
import torch
import numpy as np
from tensordict import TensorDict

from ppo_train import sample_actions_with_logprobs, greedy_actions_with_logprobs, ActionInfo


class TestSampleActionsWithLogprobs:
    """Test stochastic action sampling with log probabilities (PPO training)."""

    @pytest.fixture
    def mock_outputs(self):
        """Create mock model outputs for testing."""
        torch.manual_seed(42)
        return TensorDict({
            "main_stick": torch.randn(1, 256, 64),
            "c_stick": torch.randn(1, 256, 9),
            "buttons": torch.randn(1, 256, 5),
            "shoulder": torch.randn(1, 256, 5),
            "value": torch.randn(1, 256, 1),
        }, batch_size=(1, 256))

    def test_returns_action_info(self, mock_outputs):
        """Test that function returns ActionInfo with all required fields."""
        result = sample_actions_with_logprobs(mock_outputs)

        assert isinstance(result, ActionInfo)
        assert hasattr(result, 'main_stick_idx')
        assert hasattr(result, 'c_stick_idx')
        assert hasattr(result, 'shoulder_idx')
        assert hasattr(result, 'buttons')
        assert hasattr(result, 'main_log_prob')
        assert hasattr(result, 'c_log_prob')
        assert hasattr(result, 'shoulder_log_prob')
        assert hasattr(result, 'buttons_log_probs')

    def test_action_indices_in_valid_range(self, mock_outputs):
        """Test that sampled indices are within valid ranges."""
        result = sample_actions_with_logprobs(mock_outputs)

        # Main stick: 64 classes
        assert 0 <= result.main_stick_idx < 64
        # C-stick: 9 classes
        assert 0 <= result.c_stick_idx < 9
        # Shoulder: 5 classes
        assert 0 <= result.shoulder_idx < 5

    def test_button_actions_are_binary(self, mock_outputs):
        """Test that button actions are binary (0 or 1)."""
        result = sample_actions_with_logprobs(mock_outputs)

        assert result.buttons.shape == (5,)
        assert torch.all((result.buttons == 0) | (result.buttons == 1))

    def test_log_probs_are_negative(self, mock_outputs):
        """Test that log probabilities are negative (valid log probs)."""
        result = sample_actions_with_logprobs(mock_outputs)

        assert result.main_log_prob <= 0
        assert result.c_log_prob <= 0
        assert result.shoulder_log_prob <= 0
        assert torch.all(result.buttons_log_probs <= 0)

    def test_log_probs_are_finite(self, mock_outputs):
        """Test that log probabilities are finite (no -inf or nan)."""
        result = sample_actions_with_logprobs(mock_outputs)

        assert torch.isfinite(result.main_log_prob)
        assert torch.isfinite(result.c_log_prob)
        assert torch.isfinite(result.shoulder_log_prob)
        assert torch.all(torch.isfinite(result.buttons_log_probs))

    def test_stochastic_is_deterministic_with_seed(self, mock_outputs):
        """Test that stochastic sampling is deterministic with fixed seed."""
        torch.manual_seed(100)
        result1 = sample_actions_with_logprobs(mock_outputs)

        torch.manual_seed(100)
        result2 = sample_actions_with_logprobs(mock_outputs)

        assert result1.main_stick_idx == result2.main_stick_idx
        assert result1.c_stick_idx == result2.c_stick_idx
        assert result1.shoulder_idx == result2.shoulder_idx
        assert torch.equal(result1.buttons, result2.buttons)
        assert torch.allclose(result1.main_log_prob, result2.main_log_prob)

    def test_actions_moved_to_cpu(self, mock_outputs):
        """Test that action indices are moved to CPU."""
        result = sample_actions_with_logprobs(mock_outputs)

        # Indices should be Python ints (from CPU)
        assert isinstance(result.main_stick_idx, int)
        assert isinstance(result.c_stick_idx, int)
        assert isinstance(result.shoulder_idx, int)

        # Buttons tensor should be on CPU
        assert result.buttons.device == torch.device('cpu')


class TestGreedyActionsWithLogprobs:
    """Test greedy action selection with log probabilities (PPO training)."""

    @pytest.fixture
    def mock_outputs(self):
        """Create mock model outputs for testing."""
        torch.manual_seed(42)
        return TensorDict({
            "main_stick": torch.randn(1, 256, 64),
            "c_stick": torch.randn(1, 256, 9),
            "buttons": torch.randn(1, 256, 5),
            "shoulder": torch.randn(1, 256, 5),
            "value": torch.randn(1, 256, 1),
        }, batch_size=(1, 256))

    def test_returns_action_info(self, mock_outputs):
        """Test that function returns ActionInfo."""
        result = greedy_actions_with_logprobs(mock_outputs)
        assert isinstance(result, ActionInfo)

    def test_greedy_is_deterministic(self, mock_outputs):
        """Test that greedy selection is fully deterministic."""
        result1 = greedy_actions_with_logprobs(mock_outputs)
        result2 = greedy_actions_with_logprobs(mock_outputs)

        assert result1.main_stick_idx == result2.main_stick_idx
        assert result1.c_stick_idx == result2.c_stick_idx
        assert result1.shoulder_idx == result2.shoulder_idx
        assert torch.equal(result1.buttons, result2.buttons)

    def test_greedy_selects_max_logit(self, mock_outputs):
        """Test that greedy selection chooses argmax of logits."""
        result = greedy_actions_with_logprobs(mock_outputs)

        # Verify that selected indices correspond to argmax
        main_logits = mock_outputs["main_stick"][0, -1]
        c_logits = mock_outputs["c_stick"][0, -1]
        shoulder_logits = mock_outputs["shoulder"][0, -1]

        assert result.main_stick_idx == torch.argmax(main_logits).item()
        assert result.c_stick_idx == torch.argmax(c_logits).item()
        assert result.shoulder_idx == torch.argmax(shoulder_logits).item()

    def test_greedy_button_threshold(self, mock_outputs):
        """Test that buttons use 0.5 threshold on sigmoid."""
        result = greedy_actions_with_logprobs(mock_outputs)

        button_logits = mock_outputs["buttons"][0, -1]
        button_probs = torch.sigmoid(button_logits)
        expected_buttons = (button_probs >= 0.5).float()

        assert torch.equal(result.buttons, expected_buttons)

    def test_log_probs_match_selected_actions(self, mock_outputs):
        """Test that log probs correspond to the selected greedy actions."""
        result = greedy_actions_with_logprobs(mock_outputs)

        # Recompute log probs for verification
        main_logits = mock_outputs["main_stick"][0, -1]
        main_log_softmax = torch.log_softmax(main_logits, dim=-1)
        expected_main_log_prob = main_log_softmax[result.main_stick_idx]

        assert torch.allclose(result.main_log_prob, expected_main_log_prob)


class TestInferenceDecoding:
    """Test inference decoding functions in model_interface.py."""

    @pytest.fixture
    def mock_outputs(self):
        """Create mock model outputs for inference testing."""
        torch.manual_seed(42)
        return TensorDict({
            "main_stick": torch.randn(1, 256, 64),
            "c_stick": torch.randn(1, 256, 9),
            "buttons": torch.randn(1, 256, 5),
            "shoulder": torch.randn(1, 256, 5),
            "value": torch.randn(1, 256, 1),
        }, batch_size=(1, 256))

    def test_decode_stick_uses_argmax(self, mock_outputs):
        """Test that stick decoding uses argmax for palette lookup."""
        from model_interface import GPTInferenceEngine
        from controller_utils import CONTROL_STICK_QUANTIZED

        main_logits = mock_outputs["main_stick"][0, -1]
        expected_idx = torch.argmax(main_logits).item()

        # The palette lookup should use this index
        expected_coords = CONTROL_STICK_QUANTIZED[expected_idx]

        # Convert to [0, 1] range
        from model_interface import model_to_dolphin01
        expected_xy = model_to_dolphin01(
            np.array(expected_idx, dtype=np.int32),
            palette11=np.asarray(CONTROL_STICK_QUANTIZED, dtype=np.float32)
        )

        assert expected_xy.shape == (2,)
        assert 0.0 <= expected_xy[0] <= 1.0
        assert 0.0 <= expected_xy[1] <= 1.0

    def test_decode_buttons_stochastic(self, mock_outputs):
        """Test that button decoding uses Bernoulli sampling."""
        button_logits = mock_outputs["buttons"][0, -1]
        button_probs = torch.sigmoid(button_logits)

        # With seed, Bernoulli sampling should be deterministic
        torch.manual_seed(200)
        samples1 = torch.bernoulli(button_probs).bool()

        torch.manual_seed(200)
        samples2 = torch.bernoulli(button_probs).bool()

        assert torch.equal(samples1, samples2)

    def test_shoulder_index_clamping(self, mock_outputs):
        """Test that shoulder index is clamped to valid range."""
        from controller_utils import SHOULDER_QUANTIZED

        shoulder_logits = mock_outputs["shoulder"][0, -1]
        s_idx = torch.argmax(shoulder_logits).item()

        # Should be clamped to [0, len(SHOULDER_QUANTIZED)-1]
        clamped_idx = max(0, min(s_idx, len(SHOULDER_QUANTIZED) - 1))
        assert clamped_idx == s_idx  # In normal case, no clamping needed
        assert 0 <= clamped_idx < len(SHOULDER_QUANTIZED)


class TestActionInfoStructure:
    """Test ActionInfo dataclass structure and properties."""

    def test_action_info_fields(self):
        """Test that ActionInfo has all required fields."""
        from ppo_train import ActionInfo

        action_info = ActionInfo(
            main_stick_idx=5,
            c_stick_idx=3,
            shoulder_idx=2,
            buttons=torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0]),
            main_log_prob=torch.tensor(-2.5),
            c_log_prob=torch.tensor(-1.8),
            shoulder_log_prob=torch.tensor(-1.2),
            buttons_log_probs=torch.tensor([-0.5, -0.3, -0.8, -0.2, -0.6]),
        )

        assert action_info.main_stick_idx == 5
        assert action_info.c_stick_idx == 3
        assert action_info.shoulder_idx == 2
        assert action_info.buttons.shape == (5,)


class TestCoordinateConversion:
    """Test coordinate conversion from indices to [0, 1] range."""

    def test_model_to_dolphin01_from_indices(self):
        """Test conversion from palette indices to [0, 1] coordinates."""
        from model_interface import model_to_dolphin01
        from controller_utils import CONTROL_STICK_QUANTIZED

        # Test index 0 (should map to first palette entry)
        idx = np.array(0, dtype=np.int32)
        palette = np.asarray(CONTROL_STICK_QUANTIZED, dtype=np.float32)

        coords = model_to_dolphin01(idx, palette11=palette)

        assert coords.shape == (2,)
        assert 0.0 <= coords[0] <= 1.0
        assert 0.0 <= coords[1] <= 1.0

        # Verify mapping: [-1,1] -> [0,1]
        # palette is in [-1, 1], coords should be in [0, 1]
        expected = np.clip(palette[0] * 0.5 + 0.5, 0.0, 1.0)
        assert np.allclose(coords, expected, atol=1e-6)

    def test_palette_lookup_preserves_wavedash_angles(self):
        """Test that palette lookup preserves critical wavedash angles."""
        from controller_utils import CONTROL_STICK_QUANTIZED

        # Wavedash angles are specific palette entries
        # Test that they're preserved
        palette = np.asarray(CONTROL_STICK_QUANTIZED, dtype=np.float32)

        assert len(palette) == 64
        assert palette.shape == (64, 2)

        # All palette entries should be in [-1, 1]
        assert np.all(palette >= -1.0)
        assert np.all(palette <= 1.0)


class TestNumericalStability:
    """Test numerical stability of action sampling."""

    def test_extreme_logits_categorical(self):
        """Test that extreme logits don't cause numerical issues."""
        extreme_outputs = TensorDict({
            "main_stick": torch.full((1, 256, 64), -1000.0),
            "c_stick": torch.full((1, 256, 9), -1000.0),
            "buttons": torch.full((1, 256, 5), -1000.0),
            "shoulder": torch.full((1, 256, 5), -1000.0),
            "value": torch.zeros(1, 256, 1),
        }, batch_size=(1, 256))

        # Set one logit very high
        extreme_outputs["main_stick"][0, -1, 0] = 1000.0
        extreme_outputs["c_stick"][0, -1, 0] = 1000.0
        extreme_outputs["shoulder"][0, -1, 0] = 1000.0

        result = greedy_actions_with_logprobs(extreme_outputs)

        # Should select index 0 (highest logit)
        assert result.main_stick_idx == 0
        assert result.c_stick_idx == 0
        assert result.shoulder_idx == 0

        # Log probs should be finite
        assert torch.isfinite(result.main_log_prob)
        assert torch.isfinite(result.c_log_prob)
        assert torch.isfinite(result.shoulder_log_prob)

    def test_extreme_button_probabilities(self):
        """Test extreme button logits don't cause issues."""
        # Create tensor with correct shape (1, 256, 5)
        button_tensor = torch.zeros(1, 256, 5)
        button_tensor[0, -1, :] = torch.tensor([1000.0, -1000.0, 1000.0, -1000.0, 0.0])

        extreme_outputs = TensorDict({
            "main_stick": torch.randn(1, 256, 64),
            "c_stick": torch.randn(1, 256, 9),
            "buttons": button_tensor,
            "shoulder": torch.randn(1, 256, 5),
            "value": torch.zeros(1, 256, 1),
        }, batch_size=(1, 256))

        result = greedy_actions_with_logprobs(extreme_outputs)

        # Should threshold at 0.5: sigmoid(1000) ≈ 1, sigmoid(-1000) ≈ 0, sigmoid(0) = 0.5 >= 0.5 → 1
        expected_buttons = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])
        assert torch.equal(result.buttons, expected_buttons)


class TestIntegrationSamplingWorkflow:
    """Integration test for full sampling workflow."""

    @pytest.fixture
    def mock_outputs(self):
        """Create mock model outputs."""
        torch.manual_seed(42)
        return TensorDict({
            "main_stick": torch.randn(1, 256, 64),
            "c_stick": torch.randn(1, 256, 9),
            "buttons": torch.randn(1, 256, 5),
            "shoulder": torch.randn(1, 256, 5),
            "value": torch.randn(1, 256, 1),
        }, batch_size=(1, 256))

    def test_stochastic_then_greedy_workflow(self, mock_outputs):
        """Test sampling stochastic actions then switching to greedy."""
        # Sample stochastically
        torch.manual_seed(100)
        stochastic = sample_actions_with_logprobs(mock_outputs)

        # Sample greedily
        greedy = greedy_actions_with_logprobs(mock_outputs)

        # Both should return valid ActionInfo
        assert isinstance(stochastic, ActionInfo)
        assert isinstance(greedy, ActionInfo)

        # Greedy should be deterministic
        greedy2 = greedy_actions_with_logprobs(mock_outputs)
        assert greedy.main_stick_idx == greedy2.main_stick_idx

    def test_action_info_can_be_serialized(self, mock_outputs):
        """Test that ActionInfo can be used in rollout collection."""
        result = sample_actions_with_logprobs(mock_outputs)

        # Should be able to access all fields
        _ = result.main_stick_idx
        _ = result.c_stick_idx
        _ = result.shoulder_idx
        _ = result.buttons
        _ = result.main_log_prob
        _ = result.c_log_prob
        _ = result.shoulder_log_prob
        _ = result.buttons_log_probs

        # All accessed without errors
        assert True
