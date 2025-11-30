"""Comprehensive tests for ppo/trajectory.py - GAE computation and trajectory management."""

from __future__ import annotations

import pytest
import torch
import numpy as np

from ppo.trajectory import Step, Trajectory


class TestStep:
    """Tests for the Step dataclass."""

    def test_step_creation(self):
        """Test basic Step creation with all fields."""
        state = torch.randn(50)
        action_taken = {
            "main_stick": torch.tensor(10),
            "c_stick": torch.tensor(5),
            "buttons": torch.tensor([1, 0, 1, 0, 0], dtype=torch.float32),
            "shoulder": torch.tensor(2),
        }
        log_prob = torch.tensor(-2.5)
        value = torch.tensor(0.75)

        step = Step(
            state=state,
            action_taken=action_taken,
            log_prob=log_prob,
            value=value,
            reward=0.1,
            done=False,
        )

        assert torch.equal(step.state, state)
        assert step.action_taken["main_stick"] == 10
        assert step.log_prob == -2.5
        assert step.value == 0.75
        assert step.reward == 0.1
        assert step.done is False

    def test_step_default_values(self):
        """Test Step default values for reward and done."""
        state = torch.randn(10)
        action_taken = {"main_stick": torch.tensor(0)}
        log_prob = torch.tensor(0.0)
        value = torch.tensor(0.0)

        step = Step(
            state=state,
            action_taken=action_taken,
            log_prob=log_prob,
            value=value,
        )

        assert step.reward == 0.0
        assert step.done is False


class TestTrajectoryComputeGAE:
    """Tests for Trajectory.compute_gae() method."""

    def test_empty_trajectory_gae(self):
        """Test GAE computation on empty trajectory."""
        trajectory = Trajectory(steps=[])
        trajectory.compute_gae()

        assert len(trajectory.advantages) == 0
        assert len(trajectory.returns) == 0

    def test_single_step_gae(self):
        """Test GAE on single-step trajectory."""
        step = Step(
            state=torch.randn(10),
            action_taken={"main_stick": torch.tensor(0)},
            log_prob=torch.tensor(-1.0),
            value=torch.tensor(0.5),
            reward=1.0,
        )
        trajectory = Trajectory(steps=[step])
        trajectory.compute_gae(gamma=0.99, gae_lambda=0.95, normalize=False)

        # For single step: delta = reward + 0 - value = 1.0 - 0.5 = 0.5
        # advantage = delta = 0.5
        # return = advantage + value = 0.5 + 0.5 = 1.0
        assert len(trajectory.advantages) == 1
        assert pytest.approx(trajectory.advantages[0].item(), abs=1e-6) == 0.5
        assert pytest.approx(trajectory.returns[0].item(), abs=1e-6) == 1.0

    def test_multi_step_gae_calculation(self):
        """Test GAE calculation matches manual computation."""
        # Create trajectory with known values
        values = [0.5, 0.6, 0.7]
        rewards = [0.1, 0.2, 0.3]
        steps = []

        for v, r in zip(values, rewards):
            step = Step(
                state=torch.randn(10),
                action_taken={"main_stick": torch.tensor(0)},
                log_prob=torch.tensor(-1.0),
                value=torch.tensor(v),
                reward=r,
            )
            steps.append(step)

        trajectory = Trajectory(steps=steps)
        gamma = 0.99
        gae_lambda = 0.95
        trajectory.compute_gae(gamma=gamma, gae_lambda=gae_lambda, normalize=False)

        # Manual GAE calculation
        # delta[0] = r[0] + gamma * v[1] - v[0] = 0.1 + 0.99 * 0.6 - 0.5 = 0.194
        # delta[1] = r[1] + gamma * v[2] - v[1] = 0.2 + 0.99 * 0.7 - 0.6 = 0.293
        # delta[2] = r[2] + gamma * 0 - v[2] = 0.3 + 0 - 0.7 = -0.4
        #
        # GAE backward:
        # gae[2] = delta[2] = -0.4
        # gae[1] = delta[1] + gamma * lambda * gae[2] = 0.293 + 0.99 * 0.95 * (-0.4)
        # gae[0] = delta[0] + gamma * lambda * gae[1]

        delta_0 = rewards[0] + gamma * values[1] - values[0]
        delta_1 = rewards[1] + gamma * values[2] - values[1]
        delta_2 = rewards[2] + gamma * 0 - values[2]

        gae_2 = delta_2
        gae_1 = delta_1 + gamma * gae_lambda * gae_2
        gae_0 = delta_0 + gamma * gae_lambda * gae_1

        assert pytest.approx(trajectory.advantages[0].item(), abs=1e-6) == gae_0
        assert pytest.approx(trajectory.advantages[1].item(), abs=1e-6) == gae_1
        assert pytest.approx(trajectory.advantages[2].item(), abs=1e-6) == gae_2

        # Returns should be advantages + values
        assert pytest.approx(trajectory.returns[0].item(), abs=1e-6) == gae_0 + values[0]
        assert pytest.approx(trajectory.returns[1].item(), abs=1e-6) == gae_1 + values[1]
        assert pytest.approx(trajectory.returns[2].item(), abs=1e-6) == gae_2 + values[2]

    def test_gae_normalization(self):
        """Test that advantage normalization works correctly."""
        steps = []
        for i in range(10):
            step = Step(
                state=torch.randn(10),
                action_taken={"main_stick": torch.tensor(0)},
                log_prob=torch.tensor(-1.0),
                value=torch.tensor(float(i) / 10.0),
                reward=float(i) / 10.0,
            )
            steps.append(step)

        trajectory = Trajectory(steps=steps)
        trajectory.compute_gae(gamma=0.99, gae_lambda=0.95, normalize=True)

        # Normalized advantages should have mean ~0 and std ~1
        mean = trajectory.advantages.mean()
        std = trajectory.advantages.std()

        assert pytest.approx(mean.item(), abs=1e-6) == 0.0
        assert pytest.approx(std.item(), abs=0.01) == 1.0

    def test_gae_no_normalization(self):
        """Test that normalize=False preserves raw advantages."""
        steps = []
        for i in range(5):
            step = Step(
                state=torch.randn(10),
                action_taken={"main_stick": torch.tensor(0)},
                log_prob=torch.tensor(-1.0),
                value=torch.tensor(1.0),
                reward=2.0,
            )
            steps.append(step)

        trajectory = Trajectory(steps=steps)
        trajectory.compute_gae(gamma=0.99, gae_lambda=0.95, normalize=False)

        # Without normalization, mean should NOT be 0 (unless raw GAE happens to be 0)
        # Just check that values are computed (not zeros)
        assert not torch.allclose(trajectory.advantages, torch.zeros_like(trajectory.advantages))

    def test_gae_zero_variance_advantages(self):
        """Test GAE when all advantages are identical (zero variance)."""
        # Create trajectory where all deltas will be the same
        steps = []
        for _ in range(5):
            step = Step(
                state=torch.randn(10),
                action_taken={"main_stick": torch.tensor(0)},
                log_prob=torch.tensor(-1.0),
                value=torch.tensor(1.0),
                reward=1.0,
            )
            steps.append(step)

        trajectory = Trajectory(steps=steps)
        trajectory.compute_gae(gamma=0.99, gae_lambda=0.95, normalize=True)

        # With zero variance, normalization should just center (mean = 0)
        mean = trajectory.advantages.mean()
        assert pytest.approx(mean.item(), abs=1e-6) == 0.0

        # All advantages should be equal (to each other)
        assert torch.allclose(
            trajectory.advantages,
            trajectory.advantages[0].expand_as(trajectory.advantages)
        )

    def test_gae_with_precomputed_returns(self):
        """Test GAE when returns are already provided."""
        steps = []
        for i in range(5):
            step = Step(
                state=torch.randn(10),
                action_taken={"main_stick": torch.tensor(0)},
                log_prob=torch.tensor(-1.0),
                value=torch.tensor(float(i)),
                reward=0.0,  # Ignored when returns are provided
            )
            steps.append(step)

        trajectory = Trajectory(steps=steps)
        # Pre-set returns
        precomputed_returns = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
        trajectory.returns = precomputed_returns

        trajectory.compute_gae(gamma=0.99, gae_lambda=0.95, normalize=False)

        # Advantages should be returns - values
        expected_advantages = precomputed_returns - torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
        assert torch.allclose(trajectory.advantages, expected_advantages, atol=1e-6)

        # Returns should be preserved
        assert torch.equal(trajectory.returns, precomputed_returns)

    def test_gae_gamma_effects(self):
        """Test that different gamma values affect GAE differently."""
        steps = []
        for i in range(5):
            step = Step(
                state=torch.randn(10),
                action_taken={"main_stick": torch.tensor(0)},
                log_prob=torch.tensor(-1.0),
                value=torch.tensor(0.0),
                reward=1.0,
            )
            steps.append(step)

        # High gamma (more future-oriented)
        traj_high_gamma = Trajectory(steps=[s for s in steps])
        traj_high_gamma.compute_gae(gamma=0.99, gae_lambda=0.95, normalize=False)

        # Low gamma (more myopic)
        traj_low_gamma = Trajectory(steps=[s for s in steps])
        traj_low_gamma.compute_gae(gamma=0.5, gae_lambda=0.95, normalize=False)

        # Earlier timesteps should have higher advantages with high gamma
        # (they accumulate more future rewards)
        assert traj_high_gamma.advantages[0] > traj_low_gamma.advantages[0]

    def test_gae_lambda_effects(self):
        """Test that different lambda values affect bias-variance tradeoff."""
        steps = []
        for i in range(10):
            step = Step(
                state=torch.randn(10),
                action_taken={"main_stick": torch.tensor(0)},
                log_prob=torch.tensor(-1.0),
                value=torch.tensor(float(i) * 0.1),
                reward=float(i) * 0.1,
            )
            steps.append(step)

        # High lambda (more biased toward Monte Carlo)
        traj_high_lambda = Trajectory(steps=[s for s in steps])
        traj_high_lambda.compute_gae(gamma=0.99, gae_lambda=0.99, normalize=False)

        # Low lambda (more biased toward TD)
        traj_low_lambda = Trajectory(steps=[s for s in steps])
        traj_low_lambda.compute_gae(gamma=0.99, gae_lambda=0.1, normalize=False)

        # Advantages should be different
        assert not torch.allclose(
            traj_high_lambda.advantages,
            traj_low_lambda.advantages,
            atol=1e-3
        )


class TestTrajectoryToTensors:
    """Tests for Trajectory.to_tensors() method."""

    def test_to_tensors_requires_gae(self):
        """Test that to_tensors() fails if GAE hasn't been computed."""
        step = Step(
            state=torch.randn(10),
            action_taken={"main_stick": torch.tensor(0)},
            log_prob=torch.tensor(-1.0),
            value=torch.tensor(0.5),
        )
        trajectory = Trajectory(steps=[step])

        with pytest.raises(ValueError, match="Must call compute_gae"):
            trajectory.to_tensors(torch.device("cpu"))

    def test_to_tensors_empty_trajectory(self):
        """Test that to_tensors() fails on empty trajectory."""
        trajectory = Trajectory(steps=[])
        trajectory.compute_gae()

        with pytest.raises(ValueError, match="empty trajectory"):
            trajectory.to_tensors(torch.device("cpu"))

    def test_to_tensors_device_placement(self):
        """Test that tensors are moved to correct device."""
        steps = []
        for i in range(5):
            step = Step(
                state=torch.randn(10),
                action_taken={
                    "main_stick": torch.tensor(i),
                    "c_stick": torch.tensor(0),
                },
                log_prob=torch.tensor(-1.0),
                value=torch.tensor(0.5),
                reward=0.1,
            )
            steps.append(step)

        trajectory = Trajectory(steps=steps)
        trajectory.compute_gae()

        device = torch.device("cpu")
        result = trajectory.to_tensors(device)

        assert result["states"].device == device
        assert result["old_log_probs"].device == device
        assert result["values"].device == device
        assert result["returns"].device == device
        assert result["advantages"].device == device

    def test_to_tensors_shapes(self):
        """Test that to_tensors() produces correct tensor shapes."""
        T = 10
        F = 50
        steps = []

        for i in range(T):
            step = Step(
                state=torch.randn(F),
                action_taken={
                    "main_stick": torch.tensor(i % 64),
                    "c_stick": torch.tensor(i % 9),
                },
                log_prob=torch.tensor(-1.0),
                value=torch.tensor(0.5),
                reward=0.1,
            )
            steps.append(step)

        trajectory = Trajectory(steps=steps)
        trajectory.compute_gae()

        result = trajectory.to_tensors(torch.device("cpu"))

        assert result["states"].shape == (T, F)
        assert result["old_log_probs"].shape == (T,)
        assert result["values"].shape == (T,)
        assert result["returns"].shape == (T,)
        assert result["advantages"].shape == (T,)


class TestTrajectoryEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_trajectory_length(self):
        """Test __len__ method."""
        steps = [
            Step(
                state=torch.randn(10),
                action_taken={"main_stick": torch.tensor(i)},
                log_prob=torch.tensor(-1.0),
                value=torch.tensor(0.5),
            )
            for i in range(7)
        ]
        trajectory = Trajectory(steps=steps)

        assert len(trajectory) == 7

    def test_very_long_trajectory_gae(self):
        """Test GAE computation on long trajectory (performance/numerical stability)."""
        T = 1000
        steps = []

        for i in range(T):
            step = Step(
                state=torch.randn(10),
                action_taken={"main_stick": torch.tensor(i % 64)},
                log_prob=torch.tensor(-1.0),
                value=torch.tensor(np.sin(i / 100.0)),
                reward=np.cos(i / 100.0),
            )
            steps.append(step)

        trajectory = Trajectory(steps=steps)
        trajectory.compute_gae(gamma=0.995, gae_lambda=0.95, normalize=True)

        # Should complete without error
        assert len(trajectory.advantages) == T
        assert len(trajectory.returns) == T

        # Check no NaN or Inf
        assert not torch.isnan(trajectory.advantages).any()
        assert not torch.isinf(trajectory.advantages).any()
        assert not torch.isnan(trajectory.returns).any()
        assert not torch.isinf(trajectory.returns).any()

    def test_negative_rewards_gae(self):
        """Test GAE with negative rewards."""
        steps = []
        for i in range(5):
            step = Step(
                state=torch.randn(10),
                action_taken={"main_stick": torch.tensor(0)},
                log_prob=torch.tensor(-1.0),
                value=torch.tensor(1.0),
                reward=-0.5,  # Negative reward
            )
            steps.append(step)

        trajectory = Trajectory(steps=steps)
        trajectory.compute_gae(gamma=0.99, gae_lambda=0.95, normalize=False)

        # Should handle negative rewards correctly
        assert len(trajectory.advantages) == 5
        # All advantages should be negative (negative rewards, positive values)
        assert all(adv < 0 for adv in trajectory.advantages)

    def test_mixed_reward_trajectory(self):
        """Test trajectory with mix of positive and negative rewards."""
        rewards = [1.0, -0.5, 2.0, -1.0, 0.5]
        steps = []

        for r in rewards:
            step = Step(
                state=torch.randn(10),
                action_taken={"main_stick": torch.tensor(0)},
                log_prob=torch.tensor(-1.0),
                value=torch.tensor(0.0),
                reward=r,
            )
            steps.append(step)

        trajectory = Trajectory(steps=steps)
        trajectory.compute_gae(gamma=0.99, gae_lambda=0.95, normalize=False)

        # Should complete without error
        assert len(trajectory.advantages) == 5
        assert len(trajectory.returns) == 5
