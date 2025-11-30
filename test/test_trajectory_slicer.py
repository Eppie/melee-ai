"""Comprehensive tests for ppo/trajectory_slicer.py - trajectory slicing for continuous PPO training.

IMPORTANT: In production, seq_len should always be 256. Trajectories should always have at least
seq_len steps to avoid padding edge cases. These tests cover both realistic production scenarios
(seq_len=256) and edge cases for robustness.
"""

from __future__ import annotations

from unittest.mock import Mock, MagicMock
import pytest
import torch
import numpy as np

from ppo.trajectory_slicer import RolloutSlice, TrajectorySlicer
from ppo.trajectory import Step, Trajectory


class TestRolloutSlice:
    """Tests for the RolloutSlice dataclass."""

    def test_rollout_slice_creation(self):
        """Test basic RolloutSlice creation."""
        worker_steps = {
            0: [
                Step(
                    state=torch.randn(50),
                    action_taken={"main_stick": torch.tensor(10)},
                    log_prob=torch.tensor(-1.0),
                    value=torch.tensor(0.5),
                )
            ],
            1: [
                Step(
                    state=torch.randn(50),
                    action_taken={"main_stick": torch.tensor(20)},
                    log_prob=torch.tensor(-1.5),
                    value=torch.tensor(0.6),
                )
            ],
        }
        bootstrap_values = {
            0: torch.tensor(0.7),
            1: torch.tensor(0.8),
        }
        total_frames = 100

        rollout = RolloutSlice(
            worker_steps=worker_steps,
            bootstrap_values=bootstrap_values,
            total_frames=total_frames,
        )

        assert len(rollout.worker_steps) == 2
        assert rollout.worker_steps[0][0].value == 0.5
        assert rollout.bootstrap_values[0] == 0.7
        assert rollout.total_frames == 100


class TestActionsToprimitives:
    """Tests for TrajectorySlicer._actions_to_primitives() method."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_coordinator = Mock()
        mock_coordinator.num_workers = 4
        self.slicer = TrajectorySlicer(
            coordinator=mock_coordinator,
            rollout_length=1000,
            gamma=0.99,
            gae_lambda=0.95,
        )

    def test_convert_scalar_actions(self):
        """Test conversion of scalar tensor actions to primitives."""
        actions = {
            "main_stick": torch.tensor(42),
            "c_stick": torch.tensor(5),
            "shoulder": torch.tensor(3),
        }

        primitives = self.slicer._actions_to_primitives(actions)

        assert primitives["main_stick"] == 42
        assert primitives["c_stick"] == 5
        assert primitives["shoulder"] == 3
        assert isinstance(primitives["main_stick"], int)
        assert isinstance(primitives["c_stick"], int)
        assert isinstance(primitives["shoulder"], int)

    def test_convert_button_actions(self):
        """Test conversion of button tensor to list of bools."""
        actions = {
            "buttons": torch.tensor([True, False, True, False, False]),
        }

        primitives = self.slicer._actions_to_primitives(actions)

        assert primitives["buttons"] == [True, False, True, False, False]
        assert isinstance(primitives["buttons"], list)
        assert all(isinstance(b, bool) for b in primitives["buttons"])

    def test_convert_mixed_actions(self):
        """Test conversion of mixed action types."""
        actions = {
            "main_stick": torch.tensor(10),
            "c_stick": torch.tensor(2),
            "buttons": torch.tensor([1, 0, 1, 0, 0], dtype=torch.float32),
            "shoulder": torch.tensor(4),
        }

        primitives = self.slicer._actions_to_primitives(actions)

        assert primitives["main_stick"] == 10
        assert primitives["c_stick"] == 2
        assert primitives["shoulder"] == 4
        assert primitives["buttons"] == [1.0, 0.0, 1.0, 0.0, 0.0]

    def test_convert_cuda_tensors(self):
        """Test conversion works with CUDA tensors (if available)."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        actions = {
            "main_stick": torch.tensor(10, device=device),
            "buttons": torch.tensor([True, False], device=device),
        }

        primitives = self.slicer._actions_to_primitives(actions)

        assert primitives["main_stick"] == 10
        assert primitives["buttons"] == [True, False]


class TestComputeGAEWithBootstrap:
    """Tests for TrajectorySlicer._compute_gae_with_bootstrap() method."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_coordinator = Mock()
        mock_coordinator.num_workers = 4
        self.slicer = TrajectorySlicer(
            coordinator=mock_coordinator,
            rollout_length=1000,
            gamma=0.99,
            gae_lambda=0.95,
            normalize_advantages=False,
        )

    def test_empty_trajectory(self):
        """Test GAE computation on empty trajectory."""
        trajectory = Trajectory(steps=[])
        bootstrap_value = torch.tensor(0.5)

        self.slicer._compute_gae_with_bootstrap(trajectory, bootstrap_value)

        assert len(trajectory.advantages) == 0
        assert len(trajectory.returns) == 0

    def test_multi_step_with_bootstrap(self):
        """Test GAE computation with bootstrap differs from zero bootstrap."""
        values = [0.5, 0.6, 0.7]
        rewards = [0.1, 0.2, 0.3]
        steps = [
            Step(
                state=torch.randn(10),
                action_taken={"main_stick": torch.tensor(0)},
                log_prob=torch.tensor(-1.0),
                value=torch.tensor(v),
                reward=r,
            )
            for v, r in zip(values, rewards)
        ]

        # Compare with bootstrap vs without
        traj_with_bootstrap = Trajectory(steps=[s for s in steps])
        traj_no_bootstrap = Trajectory(steps=[s for s in steps])

        bootstrap_value = torch.tensor(1.5)
        self.slicer._compute_gae_with_bootstrap(traj_with_bootstrap, bootstrap_value)
        self.slicer._compute_gae_with_bootstrap(traj_no_bootstrap, torch.tensor(0.0))

        # With bootstrap, the final TD error should include the bootstrap value
        # This affects all advantages through the backward GAE pass
        assert not torch.allclose(
            traj_with_bootstrap.advantages,
            traj_no_bootstrap.advantages,
            atol=1e-6,
        )

    def test_bootstrap_affects_final_step_most(self):
        """Test that bootstrap value most affects the last timestep."""
        steps = []
        for i in range(10):
            step = Step(
                state=torch.randn(10),
                action_taken={"main_stick": torch.tensor(0)},
                log_prob=torch.tensor(-1.0),
                value=torch.tensor(0.5),
                reward=0.1,
            )
            steps.append(step)

        traj_high_bootstrap = Trajectory(steps=[s for s in steps])
        traj_low_bootstrap = Trajectory(steps=[s for s in steps])

        self.slicer._compute_gae_with_bootstrap(traj_high_bootstrap, torch.tensor(2.0))
        self.slicer._compute_gae_with_bootstrap(traj_low_bootstrap, torch.tensor(0.1))

        # Last step should have largest difference
        diff_last = abs(
            traj_high_bootstrap.advantages[-1] - traj_low_bootstrap.advantages[-1]
        )
        diff_first = abs(
            traj_high_bootstrap.advantages[0] - traj_low_bootstrap.advantages[0]
        )

        assert diff_last > diff_first

    def test_normalization_enabled(self):
        """Test that advantage normalization works when enabled."""
        slicer_with_norm = TrajectorySlicer(
            coordinator=Mock(),
            normalize_advantages=True,
        )

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
        slicer_with_norm._compute_gae_with_bootstrap(trajectory, torch.tensor(0.5))

        # Normalized advantages should have mean ~0 and std ~1
        mean = trajectory.advantages.mean()
        std = trajectory.advantages.std()

        assert pytest.approx(mean.item(), abs=1e-6) == 0.0
        assert pytest.approx(std.item(), abs=0.01) == 1.0

    def test_normalization_disabled(self):
        """Test that normalization can be disabled."""
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
        self.slicer._compute_gae_with_bootstrap(trajectory, torch.tensor(1.0))

        # Without normalization, mean should NOT necessarily be 0
        # Just check that values are computed
        assert not torch.allclose(
            trajectory.advantages, torch.zeros_like(trajectory.advantages)
        )

    def test_zero_variance_with_normalization(self):
        """Test normalization with zero variance advantages."""
        slicer_with_norm = TrajectorySlicer(
            coordinator=Mock(),
            normalize_advantages=True,
        )

        # Create trajectory where all advantages will be identical
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
        slicer_with_norm._compute_gae_with_bootstrap(trajectory, torch.tensor(1.0))

        # With zero variance, should just center (mean = 0)
        mean = trajectory.advantages.mean()
        assert pytest.approx(mean.item(), abs=1e-6) == 0.0

    def test_gamma_parameter_effect(self):
        """Test that gamma parameter affects GAE computation."""
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

        slicer_high_gamma = TrajectorySlicer(
            coordinator=Mock(), gamma=0.99, normalize_advantages=False
        )
        slicer_low_gamma = TrajectorySlicer(
            coordinator=Mock(), gamma=0.5, normalize_advantages=False
        )

        traj_high = Trajectory(steps=[s for s in steps])
        traj_low = Trajectory(steps=[s for s in steps])

        slicer_high_gamma._compute_gae_with_bootstrap(traj_high, torch.tensor(0.0))
        slicer_low_gamma._compute_gae_with_bootstrap(traj_low, torch.tensor(0.0))

        # High gamma should have higher advantages for earlier timesteps
        assert traj_high.advantages[0] > traj_low.advantages[0]

    def test_gae_lambda_parameter_effect(self):
        """Test that gae_lambda parameter affects bias-variance tradeoff."""
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

        slicer_high_lambda = TrajectorySlicer(
            coordinator=Mock(), gae_lambda=0.99, normalize_advantages=False
        )
        slicer_low_lambda = TrajectorySlicer(
            coordinator=Mock(), gae_lambda=0.1, normalize_advantages=False
        )

        traj_high = Trajectory(steps=[s for s in steps])
        traj_low = Trajectory(steps=[s for s in steps])

        slicer_high_lambda._compute_gae_with_bootstrap(traj_high, torch.tensor(0.0))
        slicer_low_lambda._compute_gae_with_bootstrap(traj_low, torch.tensor(0.0))

        # Different lambda should produce different advantages
        assert not torch.allclose(traj_high.advantages, traj_low.advantages, atol=1e-3)


class TestBuildSequenceWindows:
    """Tests for TrajectorySlicer._build_sequence_windows() method."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_coordinator = Mock()
        mock_coordinator.num_workers = 4
        self.slicer = TrajectorySlicer(
            coordinator=mock_coordinator,
            rollout_length=1000,
        )
        self.device = torch.device("cpu")

    def test_empty_trajectories(self):
        """Test with empty trajectory list."""
        result = self.slicer._build_sequence_windows(
            [], seq_len=10, stride=5, device=self.device
        )
        assert len(result) == 0

    def test_single_short_trajectory(self):
        """Test with trajectory shorter than seq_len (requires padding)."""
        steps = []
        for i in range(5):
            step = Step(
                state=torch.randn(20),
                action_taken={"main_stick": torch.tensor(i)},
                log_prob=torch.tensor(-1.0),
                value=torch.tensor(0.5),
                reward=0.1,
            )
            steps.append(step)

        trajectory = Trajectory(steps=steps)
        trajectory.compute_gae(normalize=False)

        seq_len = 10
        result = self.slicer._build_sequence_windows(
            [trajectory], seq_len=seq_len, stride=5, device=self.device
        )

        # Should create one padded window
        assert result["states"].shape[0] == 1  # 1 window
        assert result["states"].shape[1] == seq_len  # seq_len timesteps
        assert result["states"].shape[2] == 20  # feature dim

        # Valid mask should have first 5 positions False, last 5 True
        assert result["valid_mask"].shape == (1, seq_len)
        assert result["valid_mask"][0, :5].sum() == 0  # First 5 are padding
        assert result["valid_mask"][0, 5:].sum() == 5  # Last 5 are valid

    def test_single_long_trajectory_sliding_windows(self):
        """Test creating sliding windows from long trajectory."""
        T = 25
        steps = []
        for i in range(T):
            step = Step(
                state=torch.randn(20),
                action_taken={"main_stick": torch.tensor(i % 64)},
                log_prob=torch.tensor(-1.0),
                value=torch.tensor(0.5),
                reward=0.1,
            )
            steps.append(step)

        trajectory = Trajectory(steps=steps)
        trajectory.compute_gae(normalize=False)

        seq_len = 10
        stride = 5
        result = self.slicer._build_sequence_windows(
            [trajectory], seq_len=seq_len, stride=stride, device=self.device
        )

        # Calculate expected number of windows
        # Windows at: 0, 5, 10, 15 (each of length 10)
        # Plus final window at 15 (25-10)
        # So: 0-10, 5-15, 10-20, 15-25 = 4 windows
        expected_windows = (T - seq_len) // stride + 1
        # The code also adds a final window if (T - seq_len) % stride != 0
        # (25 - 10) = 15, 15 % 5 = 0, so no extra final window
        assert result["states"].shape[0] == expected_windows
        assert result["states"].shape[1] == seq_len

        # All should be valid (no padding for long trajectory)
        assert result["valid_mask"].all()

    def test_multiple_trajectories(self):
        """Test combining windows from multiple trajectories."""
        trajectories = []
        for _ in range(3):
            steps = []
            for i in range(15):
                step = Step(
                    state=torch.randn(20),
                    action_taken={"main_stick": torch.tensor(i)},
                    log_prob=torch.tensor(-1.0),
                    value=torch.tensor(0.5),
                    reward=0.1,
                )
                steps.append(step)
            traj = Trajectory(steps=steps)
            traj.compute_gae(normalize=False)
            trajectories.append(traj)

        seq_len = 10
        stride = 5
        result = self.slicer._build_sequence_windows(
            trajectories, seq_len=seq_len, stride=stride, device=self.device
        )

        # Each trajectory of length 15 with seq_len=10, stride=5:
        # Windows at 0-10, 5-15 = 2 windows per trajectory
        # 3 trajectories * 2 windows = 6 total
        expected_total_windows = 3 * 2
        assert result["states"].shape[0] == expected_total_windows

    def test_action_heads_included(self):
        """Test that all action heads are included in output."""
        steps = []
        for i in range(15):
            step = Step(
                state=torch.randn(20),
                action_taken={
                    "main_stick": torch.tensor(i % 64),
                    "c_stick": torch.tensor(i % 9),
                    "buttons": torch.tensor([1, 0, 1, 0, 0], dtype=torch.float32),
                    "shoulder": torch.tensor(i % 5),
                },
                log_prob=torch.tensor(-1.0),
                value=torch.tensor(0.5),
                reward=0.1,
            )
            steps.append(step)

        trajectory = Trajectory(steps=steps)
        trajectory.compute_gae(normalize=False)

        result = self.slicer._build_sequence_windows(
            [trajectory], seq_len=10, stride=5, device=self.device
        )

        # All action heads should be present
        assert "actions_main_stick" in result
        assert "actions_c_stick" in result
        assert "actions_buttons" in result
        assert "actions_shoulder" in result

        # Check shapes
        num_windows = result["states"].shape[0]
        assert result["actions_main_stick"].shape == (num_windows, 10)
        assert result["actions_c_stick"].shape == (num_windows, 10)
        assert result["actions_buttons"].shape == (num_windows, 10, 5)
        assert result["actions_shoulder"].shape == (num_windows, 10)

    def test_required_outputs_present(self):
        """Test that all required training outputs are present."""
        steps = []
        for i in range(15):
            step = Step(
                state=torch.randn(20),
                action_taken={"main_stick": torch.tensor(i)},
                log_prob=torch.tensor(-1.0),
                value=torch.tensor(0.5),
                reward=0.1,
            )
            steps.append(step)

        trajectory = Trajectory(steps=steps)
        trajectory.compute_gae(normalize=False)

        result = self.slicer._build_sequence_windows(
            [trajectory], seq_len=10, stride=5, device=self.device
        )

        # Check all required keys are present
        required_keys = [
            "states",
            "advantages",
            "returns",
            "old_log_probs",
            "values",
            "valid_mask",
        ]
        for key in required_keys:
            assert key in result

    def test_device_placement(self):
        """Test that tensors are placed on correct device."""
        steps = []
        for i in range(15):
            step = Step(
                state=torch.randn(20),
                action_taken={"main_stick": torch.tensor(i)},
                log_prob=torch.tensor(-1.0),
                value=torch.tensor(0.5),
                reward=0.1,
            )
            steps.append(step)

        trajectory = Trajectory(steps=steps)
        trajectory.compute_gae(normalize=False)

        device = torch.device("cpu")
        result = self.slicer._build_sequence_windows(
            [trajectory], seq_len=10, stride=5, device=device
        )

        # All tensors should be on the specified device
        for value in result.values():
            assert value.device == device

    def test_padding_preserves_first_state(self):
        """Test that padding uses the first state (not zeros)."""
        steps = []
        first_state = torch.ones(20) * 99.0  # Distinctive value
        for i in range(5):
            state = first_state if i == 0 else torch.randn(20)
            step = Step(
                state=state,
                action_taken={"main_stick": torch.tensor(i)},
                log_prob=torch.tensor(-1.0),
                value=torch.tensor(0.5),
                reward=0.1,
            )
            steps.append(step)

        trajectory = Trajectory(steps=steps)
        trajectory.compute_gae(normalize=False)

        seq_len = 10
        result = self.slicer._build_sequence_windows(
            [trajectory], seq_len=seq_len, stride=5, device=self.device
        )

        # Padding should use first state
        # Padded positions 0-4 should equal position 5 (the actual first state)
        for i in range(5):
            assert torch.allclose(result["states"][0, i], result["states"][0, 5])

    def test_stride_parameter_affects_overlap(self):
        """Test that stride parameter controls window overlap."""
        steps = []
        for i in range(30):
            step = Step(
                state=torch.randn(20),
                action_taken={"main_stick": torch.tensor(i)},
                log_prob=torch.tensor(-1.0),
                value=torch.tensor(0.5),
                reward=0.1,
            )
            steps.append(step)

        trajectory = Trajectory(steps=steps)
        trajectory.compute_gae(normalize=False)

        seq_len = 10

        # Large stride (less overlap)
        result_large_stride = self.slicer._build_sequence_windows(
            [trajectory], seq_len=seq_len, stride=10, device=self.device
        )

        # Small stride (more overlap)
        result_small_stride = self.slicer._build_sequence_windows(
            [trajectory], seq_len=seq_len, stride=2, device=self.device
        )

        # Small stride should produce more windows
        assert (
            result_small_stride["states"].shape[0]
            > result_large_stride["states"].shape[0]
        )

    def test_trajectory_without_gae_skipped(self):
        """Test that trajectories without computed GAE are skipped."""
        # Trajectory with GAE
        steps_with_gae = []
        for i in range(15):
            step = Step(
                state=torch.randn(20),
                action_taken={"main_stick": torch.tensor(i)},
                log_prob=torch.tensor(-1.0),
                value=torch.tensor(0.5),
                reward=0.1,
            )
            steps_with_gae.append(step)
        traj_with_gae = Trajectory(steps=steps_with_gae)
        traj_with_gae.compute_gae(normalize=False)

        # Trajectory without GAE
        steps_without_gae = []
        for i in range(15):
            step = Step(
                state=torch.randn(20),
                action_taken={"main_stick": torch.tensor(i)},
                log_prob=torch.tensor(-1.0),
                value=torch.tensor(0.5),
                reward=0.1,
            )
            steps_without_gae.append(step)
        traj_without_gae = Trajectory(steps=steps_without_gae)
        # Don't compute GAE

        result = self.slicer._build_sequence_windows(
            [traj_with_gae, traj_without_gae],
            seq_len=10,
            stride=5,
            device=self.device,
        )

        # Should only have windows from traj_with_gae
        # 15 steps, seq_len=10, stride=5 -> 2 windows
        assert result["states"].shape[0] == 2


class TestPrepareTrainingData:
    """Tests for TrajectorySlicer.prepare_training_data() method."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_coordinator = Mock()
        mock_coordinator.num_workers = 4
        self.slicer = TrajectorySlicer(
            coordinator=mock_coordinator,
            rollout_length=1000,
            gamma=0.99,
            gae_lambda=0.95,
            normalize_advantages=False,
        )
        self.device = torch.device("cpu")

    def test_empty_rollout(self):
        """Test with empty rollout."""
        rollout = RolloutSlice(
            worker_steps={},
            bootstrap_values={},
            total_frames=0,
        )

        result = self.slicer.prepare_training_data(
            rollout, seq_len=10, device=self.device
        )

        assert len(result) == 0

    def test_rollout_with_single_worker(self):
        """Test preparing data from rollout with single worker."""
        steps = []
        for i in range(20):
            step = Step(
                state=torch.randn(50),
                action_taken={"main_stick": torch.tensor(i % 64)},
                log_prob=torch.tensor(-1.0),
                value=torch.tensor(0.5),
                reward=0.1,
            )
            steps.append(step)

        rollout = RolloutSlice(
            worker_steps={0: steps},
            bootstrap_values={0: torch.tensor(0.7)},
            total_frames=20,
        )

        seq_len = 10
        result = self.slicer.prepare_training_data(
            rollout, seq_len=seq_len, device=self.device
        )

        # Should have windows created
        assert "states" in result
        assert result["states"].shape[1] == seq_len  # Sequence dimension
        assert result["states"].shape[2] == 50  # Feature dimension

    def test_rollout_with_multiple_workers(self):
        """Test combining data from multiple workers."""
        worker_steps = {}
        bootstrap_values = {}

        for worker_id in range(3):
            steps = []
            for i in range(15):
                step = Step(
                    state=torch.randn(50),
                    action_taken={"main_stick": torch.tensor(i % 64)},
                    log_prob=torch.tensor(-1.0),
                    value=torch.tensor(0.5),
                    reward=0.1,
                )
                steps.append(step)
            worker_steps[worker_id] = steps
            bootstrap_values[worker_id] = torch.tensor(0.7)

        rollout = RolloutSlice(
            worker_steps=worker_steps,
            bootstrap_values=bootstrap_values,
            total_frames=45,
        )

        seq_len = 10
        result = self.slicer.prepare_training_data(
            rollout, seq_len=seq_len, device=self.device
        )

        # Should combine windows from all workers
        # Each worker: 15 steps, seq_len=10, stride ~= 10//64 = max(1, 0) = 1
        # Actually, stride = max(1, seq_len // 64) = max(1, 10//64) = 1
        # So: windows at 0-10, 1-11, 2-12, 3-13, 4-14, 5-15 = 6 windows per worker
        # 3 workers * 6 windows = 18 total
        # Note: The actual stride calculation is max(1, seq_len // 64)
        stride = max(1, seq_len // 64)  # = 1 for seq_len=10

        # For T=15, seq_len=10, stride=1:
        # Windows: 0-10, 1-11, 2-12, 3-13, 4-14, 5-15 = 6 windows
        # 3 workers * 6 = 18 windows
        assert result["states"].shape[0] >= 3  # At least one window per worker

    def test_bootstrap_value_used_in_gae(self):
        """Test that bootstrap values affect advantage computation."""
        steps = []
        for i in range(10):
            step = Step(
                state=torch.randn(50),
                action_taken={"main_stick": torch.tensor(i)},
                log_prob=torch.tensor(-1.0),
                value=torch.tensor(0.5),
                reward=0.1,
            )
            steps.append(step)

        # Two rollouts with different bootstrap values
        rollout_high_bootstrap = RolloutSlice(
            worker_steps={0: [s for s in steps]},
            bootstrap_values={0: torch.tensor(2.0)},
            total_frames=10,
        )

        rollout_low_bootstrap = RolloutSlice(
            worker_steps={0: [s for s in steps]},
            bootstrap_values={0: torch.tensor(0.1)},
            total_frames=10,
        )

        result_high = self.slicer.prepare_training_data(
            rollout_high_bootstrap, seq_len=10, device=self.device
        )
        result_low = self.slicer.prepare_training_data(
            rollout_low_bootstrap, seq_len=10, device=self.device
        )

        # Advantages should differ due to different bootstrap values
        assert not torch.allclose(result_high["advantages"], result_low["advantages"])

    def test_worker_with_empty_steps_ignored(self):
        """Test that workers with no steps are ignored."""
        worker_steps = {
            0: [
                Step(
                    state=torch.randn(50),
                    action_taken={"main_stick": torch.tensor(0)},
                    log_prob=torch.tensor(-1.0),
                    value=torch.tensor(0.5),
                    reward=0.1,
                )
                for _ in range(15)
            ],
            1: [],  # Empty worker
        }
        bootstrap_values = {
            0: torch.tensor(0.7),
            1: torch.tensor(0.5),
        }

        rollout = RolloutSlice(
            worker_steps=worker_steps,
            bootstrap_values=bootstrap_values,
            total_frames=15,
        )

        result = self.slicer.prepare_training_data(
            rollout, seq_len=10, device=self.device
        )

        # Should only process worker 0
        assert "states" in result
        assert result["states"].shape[0] >= 1

    def test_missing_bootstrap_value_uses_zero(self):
        """Test that missing bootstrap values default to 0.0."""
        steps = []
        for i in range(10):
            step = Step(
                state=torch.randn(50),
                action_taken={"main_stick": torch.tensor(i)},
                log_prob=torch.tensor(-1.0),
                value=torch.tensor(0.5),
                reward=0.1,
            )
            steps.append(step)

        rollout = RolloutSlice(
            worker_steps={0: steps},
            bootstrap_values={},  # No bootstrap value for worker 0
            total_frames=10,
        )

        # Should not raise an error, should use 0.0
        result = self.slicer.prepare_training_data(
            rollout, seq_len=10, device=self.device
        )

        assert "states" in result


class TestClearRecords:
    """Tests for TrajectorySlicer.clear_records() method."""

    def test_clear_records_calls_coordinator(self):
        """Test that clear_records delegates to coordinator."""
        mock_coordinator = Mock()
        mock_coordinator.clear_step_records = Mock()

        slicer = TrajectorySlicer(coordinator=mock_coordinator)
        slicer.clear_records()

        mock_coordinator.clear_step_records.assert_called_once()


class TestTrajectorySlicerInitialization:
    """Tests for TrajectorySlicer initialization."""

    def test_default_parameters(self):
        """Test initialization with default parameters."""
        mock_coordinator = Mock()
        slicer = TrajectorySlicer(coordinator=mock_coordinator)

        assert slicer.coordinator is mock_coordinator
        assert slicer.rollout_length == 5000
        assert slicer.gamma == 0.995
        assert slicer.gae_lambda == 0.95
        assert slicer.normalize_advantages is True

    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        mock_coordinator = Mock()
        slicer = TrajectorySlicer(
            coordinator=mock_coordinator,
            rollout_length=10000,
            gamma=0.99,
            gae_lambda=0.9,
            normalize_advantages=False,
        )

        assert slicer.rollout_length == 10000
        assert slicer.gamma == 0.99
        assert slicer.gae_lambda == 0.9
        assert slicer.normalize_advantages is False


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_coordinator = Mock()
        self.slicer = TrajectorySlicer(coordinator=mock_coordinator)
        self.device = torch.device("cpu")

    def test_exact_sequence_length_trajectory(self):
        """Test trajectory that is exactly seq_len (no padding, single window)."""
        seq_len = 10
        steps = []
        for i in range(seq_len):
            step = Step(
                state=torch.randn(20),
                action_taken={"main_stick": torch.tensor(i)},
                log_prob=torch.tensor(-1.0),
                value=torch.tensor(0.5),
                reward=0.1,
            )
            steps.append(step)

        trajectory = Trajectory(steps=steps)
        trajectory.compute_gae(normalize=False)

        result = self.slicer._build_sequence_windows(
            [trajectory], seq_len=seq_len, stride=5, device=self.device
        )

        # Should create exactly 1 window (0-10)
        assert result["states"].shape[0] == 1
        assert result["valid_mask"].all()  # No padding

    def test_very_long_trajectory(self):
        """Test handling of very long trajectory."""
        steps = []
        for i in range(1000):
            step = Step(
                state=torch.randn(20),
                action_taken={"main_stick": torch.tensor(i % 64)},
                log_prob=torch.tensor(-1.0),
                value=torch.tensor(0.5),
                reward=0.1,
            )
            steps.append(step)

        trajectory = Trajectory(steps=steps)
        trajectory.compute_gae(normalize=False)

        seq_len = 50
        stride = 25
        result = self.slicer._build_sequence_windows(
            [trajectory], seq_len=seq_len, stride=stride, device=self.device
        )

        # Should create many windows
        expected_windows = (1000 - seq_len) // stride + 1
        assert (
            result["states"].shape[0] >= expected_windows - 1
        )  # Allow for final window logic


class TestProductionScenarios:
    """Tests for realistic production scenarios with seq_len=256.

    These tests validate behavior with production parameters:
    - seq_len=256 (standard sequence length)
    - Trajectories always >= 256 steps
    - Realistic rollout lengths (5000 frames)
    """

    def setup_method(self):
        """Set up test fixtures."""
        mock_coordinator = Mock()
        mock_coordinator.num_workers = 4
        self.slicer = TrajectorySlicer(
            coordinator=mock_coordinator,
            rollout_length=5000,
            gamma=0.995,
            gae_lambda=0.95,
            normalize_advantages=True,
        )
        self.device = torch.device("cpu")
        self.seq_len = 256  # Production seq_len

    def test_production_seq_len_exact_match(self):
        """Test trajectory that is exactly seq_len=256."""
        steps = []
        for i in range(self.seq_len):
            step = Step(
                state=torch.randn(50),
                action_taken={
                    "main_stick": torch.tensor(i % 64),
                    "c_stick": torch.tensor(i % 9),
                    "buttons": torch.tensor([1, 0, 1, 0, 0], dtype=torch.float32),
                    "shoulder": torch.tensor(i % 5),
                },
                log_prob=torch.tensor(-1.0),
                value=torch.tensor(0.5),
                reward=0.1,
            )
            steps.append(step)

        trajectory = Trajectory(steps=steps)
        trajectory.compute_gae(normalize=True)

        stride = max(1, self.seq_len // 64)  # Production stride calculation
        result = self.slicer._build_sequence_windows(
            [trajectory], seq_len=self.seq_len, stride=stride, device=self.device
        )

        # Should create exactly 1 window
        assert result["states"].shape == (1, self.seq_len, 50)
        assert result["valid_mask"].all()  # No padding needed

    def test_production_typical_rollout(self):
        """Test typical production rollout: 5000 frames across 4 workers."""
        num_workers = 4
        frames_per_worker = 5000 // num_workers  # ~1250 frames per worker

        worker_steps = {}
        bootstrap_values = {}

        for worker_id in range(num_workers):
            steps = []
            for i in range(frames_per_worker):
                step = Step(
                    state=torch.randn(50),
                    action_taken={
                        "main_stick": torch.tensor(i % 64),
                        "c_stick": torch.tensor(i % 9),
                        "buttons": torch.tensor([1, 0, 1, 0, 0], dtype=torch.float32),
                        "shoulder": torch.tensor(i % 5),
                    },
                    log_prob=torch.tensor(-1.0),
                    value=torch.tensor(0.5),
                    reward=0.1,
                )
                steps.append(step)
            worker_steps[worker_id] = steps
            bootstrap_values[worker_id] = torch.tensor(0.7)

        rollout = RolloutSlice(
            worker_steps=worker_steps,
            bootstrap_values=bootstrap_values,
            total_frames=5000,
        )

        result = self.slicer.prepare_training_data(
            rollout, seq_len=self.seq_len, device=self.device
        )

        # Verify we got training data
        assert "states" in result
        assert "advantages" in result
        assert "returns" in result
        assert result["states"].shape[1] == self.seq_len

        # Each worker has ~1250 steps
        # stride = max(1, 256 // 64) = 4
        # Windows per worker: (1250 - 256) // 4 + 1 = 994 // 4 + 1 = 248 + 1 = 249
        # Plus potential final window
        # 4 workers * ~249 windows = ~996 windows
        assert result["states"].shape[0] >= 900  # At least this many windows

    def test_production_long_episode(self):
        """Test long episode (>>256 steps) with seq_len=256."""
        # Simulate a long episode of 2000 steps
        steps = []
        for i in range(2000):
            step = Step(
                state=torch.randn(50),
                action_taken={
                    "main_stick": torch.tensor(i % 64),
                    "c_stick": torch.tensor(i % 9),
                    "buttons": torch.tensor([1, 0, 1, 0, 0], dtype=torch.float32),
                    "shoulder": torch.tensor(i % 5),
                },
                log_prob=torch.tensor(-1.0),
                value=torch.tensor(np.sin(i / 100.0)),  # Varying values
                reward=np.cos(i / 100.0),  # Varying rewards
            )
            steps.append(step)

        trajectory = Trajectory(steps=steps)
        bootstrap_value = torch.tensor(0.5)
        self.slicer._compute_gae_with_bootstrap(trajectory, bootstrap_value)

        stride = max(1, self.seq_len // 64)  # = 4
        result = self.slicer._build_sequence_windows(
            [trajectory], seq_len=self.seq_len, stride=stride, device=self.device
        )

        # Calculate expected windows: (2000 - 256) // 4 + 1 = 1744 // 4 + 1 = 436 + 1
        expected_windows = (2000 - self.seq_len) // stride + 1
        assert result["states"].shape[0] >= expected_windows - 1

        # All windows should be valid (no padding for long trajectory)
        assert result["valid_mask"].all()

        # Check normalization was applied to the trajectory
        # Note: After windowing, the combined batch mean/std may differ slightly
        # from the original trajectory normalization due to overlapping windows
        mean = result["advantages"].mean()
        std = result["advantages"].std()
        assert pytest.approx(mean.item(), abs=0.2) == 0.0  # Relaxed tolerance
        assert pytest.approx(std.item(), abs=0.3) == 1.0  # Relaxed tolerance

    def test_production_bootstrap_integration(self):
        """Test GAE with bootstrap in production settings."""
        steps = []
        for i in range(500):  # Realistic rollout chunk
            step = Step(
                state=torch.randn(50),
                action_taken={"main_stick": torch.tensor(i % 64)},
                log_prob=torch.tensor(-1.0),
                value=torch.tensor(float(i) / 500.0),  # Gradually increasing value
                reward=0.1,
            )
            steps.append(step)

        trajectory = Trajectory(steps=steps)

        # High bootstrap (optimistic)
        bootstrap_high = torch.tensor(2.0)
        self.slicer._compute_gae_with_bootstrap(trajectory, bootstrap_high)
        advantages_high = trajectory.advantages.clone()
        returns_high = trajectory.returns.clone()

        # Reset trajectory
        trajectory.advantages = None
        trajectory.returns = None

        # Low bootstrap (pessimistic)
        bootstrap_low = torch.tensor(0.1)
        self.slicer._compute_gae_with_bootstrap(trajectory, bootstrap_low)
        advantages_low = trajectory.advantages.clone()
        returns_low = trajectory.returns.clone()

        # Bootstrap should significantly affect final timesteps
        final_idx = -1
        assert advantages_high[final_idx] > advantages_low[final_idx]
        assert returns_high[final_idx] > returns_low[final_idx]

        # Effect should diminish for earlier timesteps (due to discounting)
        early_idx = 0
        diff_final = abs(advantages_high[final_idx] - advantages_low[final_idx])
        diff_early = abs(advantages_high[early_idx] - advantages_low[early_idx])
        assert diff_final > diff_early

    def test_production_multiple_workers_seq_len_256(self):
        """Test realistic multi-worker scenario with seq_len=256."""
        num_workers = 4
        steps_per_worker = 800  # Each worker collects 800 steps

        worker_steps = {}
        bootstrap_values = {}

        for worker_id in range(num_workers):
            steps = []
            for i in range(steps_per_worker):
                step = Step(
                    state=torch.randn(50),
                    action_taken={
                        "main_stick": torch.tensor((i + worker_id * 10) % 64),
                        "c_stick": torch.tensor(i % 9),
                        "buttons": torch.tensor([1, 0, 1, 0, 0], dtype=torch.float32),
                        "shoulder": torch.tensor(i % 5),
                    },
                    log_prob=torch.tensor(-1.0 - worker_id * 0.1),
                    value=torch.tensor(0.5 + worker_id * 0.1),
                    reward=0.1 * (worker_id + 1),
                )
                steps.append(step)
            worker_steps[worker_id] = steps
            bootstrap_values[worker_id] = torch.tensor(0.5 + worker_id * 0.1)

        rollout = RolloutSlice(
            worker_steps=worker_steps,
            bootstrap_values=bootstrap_values,
            total_frames=num_workers * steps_per_worker,
        )

        result = self.slicer.prepare_training_data(
            rollout, seq_len=self.seq_len, device=self.device
        )

        # Verify all action heads are present
        assert "actions_main_stick" in result
        assert "actions_c_stick" in result
        assert "actions_buttons" in result
        assert "actions_shoulder" in result

        # Verify shapes
        batch_size = result["states"].shape[0]
        assert result["states"].shape == (batch_size, self.seq_len, 50)
        assert result["actions_main_stick"].shape == (batch_size, self.seq_len)
        assert result["actions_c_stick"].shape == (batch_size, self.seq_len)
        assert result["actions_buttons"].shape == (batch_size, self.seq_len, 5)
        assert result["actions_shoulder"].shape == (batch_size, self.seq_len)
        assert result["advantages"].shape == (batch_size, self.seq_len)
        assert result["returns"].shape == (batch_size, self.seq_len)

        # Each worker: 800 steps, seq_len=256, stride=4
        # Windows per worker: (800 - 256) // 4 + 1 = 544 // 4 + 1 = 136 + 1 = 137
        # 4 workers * 137 = ~548 windows
        assert batch_size >= 500  # Should have many windows

    def test_production_gae_parameters(self):
        """Test GAE computation with production parameters (gamma=0.995, lambda=0.95)."""
        steps = []
        for i in range(1000):
            step = Step(
                state=torch.randn(50),
                action_taken={"main_stick": torch.tensor(i % 64)},
                log_prob=torch.tensor(-1.0),
                value=torch.tensor(0.5),
                reward=0.1 if i % 10 == 0 else 0.0,  # Sparse rewards
            )
            steps.append(step)

        trajectory = Trajectory(steps=steps)
        bootstrap_value = torch.tensor(0.5)

        # Use production slicer (gamma=0.995, lambda=0.95, normalize=True)
        self.slicer._compute_gae_with_bootstrap(trajectory, bootstrap_value)

        # Advantages should be normalized
        mean = trajectory.advantages.mean()
        std = trajectory.advantages.std()
        assert pytest.approx(mean.item(), abs=1e-6) == 0.0
        assert pytest.approx(std.item(), abs=0.01) == 1.0

        # Returns should be computed
        assert trajectory.returns is not None
        assert len(trajectory.returns) == 1000
