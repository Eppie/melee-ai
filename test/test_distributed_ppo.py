"""Unit tests for distributed PPO training components."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from config import init_config, reset_config, get_config


@pytest.fixture(autouse=True)
def setup_config():
    """Initialize config before each test."""
    reset_config()
    init_config()
    yield
    reset_config()


class TestPPOConfig:
    """Test new PPO config fields."""

    def test_default_rollout_length(self):
        cfg = get_config()
        assert cfg.ppo.rollout_length == 5000

    def test_default_opponent_rotation_interval(self):
        cfg = get_config()
        assert cfg.ppo.opponent_rotation_interval == 10000

    def test_default_warmup_frames(self):
        cfg = get_config()
        assert cfg.ppo.warmup_frames == 256

    def test_default_distributed_mode(self):
        cfg = get_config()
        assert cfg.ppo.distributed_mode is False

    def test_rollout_length_validation(self):
        """rollout_length must be >= 100."""
        from pydantic import ValidationError
        from config.ppo_config import PPOConfig

        with pytest.raises(ValidationError):
            PPOConfig(rollout_length=50)

    def test_opponent_rotation_interval_validation(self):
        """opponent_rotation_interval must be >= 1000."""
        from pydantic import ValidationError
        from config.ppo_config import PPOConfig

        with pytest.raises(ValidationError):
            PPOConfig(opponent_rotation_interval=500)


class TestInferenceCoordinator:
    """Test InferenceCoordinator functionality."""

    @pytest.fixture
    def coordinator(self):
        """Create a coordinator with a small model for testing."""
        from model.nano_gpt import GPT
        from ppo.inference_coordinator import InferenceCoordinator

        config = get_config()
        device = torch.device("cpu")
        model = GPT(config).to(device)
        model.eval()

        seq_len = 64
        return InferenceCoordinator(
            learner_model=model,
            device=device,
            num_workers=2,
            seq_len=seq_len,
            warmup_frames=seq_len,  # Must equal seq_len
        )

    @pytest.fixture
    def valid_features(self):
        """Create valid feature tensor."""
        from schema import get_feature_names

        feature_names = get_feature_names()
        feature_dim = len(feature_names)
        features = torch.zeros(feature_dim, dtype=torch.float32)

        # Set valid indices for categorical features
        for i, name in enumerate(feature_names):
            if "character" in name or "action" in name or name == "stage":
                features[i] = 0.0
            elif "stock" in name:
                features[i] = 4.0

        return features

    def test_worker_state_initialization(self, coordinator):
        """Test that worker states are properly initialized."""
        assert len(coordinator.worker_states) == 2
        for worker_id in range(2):
            state = coordinator.worker_states[worker_id]
            assert state.worker_id == worker_id
            assert len(state.learner_buffer) == 0
            assert len(state.opponent_buffer) == 0
            assert state.frames_collected == 0

    def test_reset_worker(self, coordinator, valid_features):
        """Test worker reset clears buffers."""
        # Add some frames
        state = coordinator.worker_states[0]
        state.learner_buffer.append(valid_features)
        state.opponent_buffer.append(valid_features)
        state.frames_collected = 10

        # Reset
        coordinator.reset_worker(0)

        assert len(state.learner_buffer) == 0
        assert len(state.opponent_buffer) == 0
        assert state.frames_collected == 0

    def test_warmup_returns_neutral_actions(self, coordinator, valid_features):
        """During warmup, should return neutral actions."""
        worker_states = [(0, valid_features.clone(), 0.0, False)]
        results, timings = coordinator.process_states(worker_states)

        # Should have result for worker 0
        assert 0 in results
        p1_actions, p2_actions = results[0]

        # During warmup, actions should be neutral
        assert p1_actions["main_stick"].item() == 0
        assert p1_actions["c_stick"].item() == 0

        # Check that timings are returned
        assert timings is not None
        assert timings.total >= 0

    def test_inference_after_warmup(self, coordinator, valid_features):
        """After warmup frames, should get real model actions."""
        # Fill up warmup frames
        for frame in range(coordinator.warmup_frames + 1):
            worker_states = [
                (0, valid_features.clone(), 0.0, False),
                (1, valid_features.clone(), 0.0, False),
            ]
            results, timings = coordinator.process_states(worker_states)

        # After warmup, should have recorded steps
        assert coordinator.get_total_steps() > 0

    def test_step_recording(self, coordinator, valid_features):
        """Test that steps are recorded after warmup."""
        # Fill warmup and get a few real steps
        # Inference starts at frame warmup_frames (0-indexed), so +5 gives 6 steps
        for frame in range(coordinator.warmup_frames + 5):
            worker_states = [
                (0, valid_features.clone(), 0.1, False),
                (1, valid_features.clone(), 0.2, False),
            ]
            results, timings = coordinator.process_states(worker_states)

        # Check step records
        records_0 = coordinator.get_step_records(0)
        records_1 = coordinator.get_step_records(1)

        # Steps are recorded starting at frame warmup_frames (inclusive)
        # So frames [16, 17, 18, 19, 20] = 6 steps (frames 16-20 when warmup=16)
        assert len(records_0) == 6
        assert len(records_1) == 6

        # Check record structure
        record = records_0[0]
        assert record.worker_id == 0
        assert record.state.shape[0] == coordinator.feature_dim
        assert "main_stick" in record.action_logits
        assert "main_stick" in record.action_taken
        assert record.log_prob is not None
        assert record.value is not None

    def test_clear_step_records(self, coordinator, valid_features):
        """Test clearing step records."""
        # Generate some steps
        for frame in range(coordinator.warmup_frames + 3):
            worker_states = [(0, valid_features.clone(), 0.0, False)]
            results, timings = coordinator.process_states(worker_states)

        assert coordinator.get_total_steps() > 0

        coordinator.clear_step_records()

        assert coordinator.get_total_steps() == 0

    def test_done_resets_worker(self, coordinator, valid_features):
        """Test that done=True resets the worker."""
        # Fill some frames
        for frame in range(coordinator.warmup_frames):
            worker_states = [(0, valid_features.clone(), 0.0, False)]
            results, timings = coordinator.process_states(worker_states)

        state = coordinator.worker_states[0]
        assert state.frames_collected == coordinator.warmup_frames

        # Send done signal
        worker_states = [(0, valid_features.clone(), 1.0, True)]
        results, timings = coordinator.process_states(worker_states)

        assert state.frames_collected == 0
        assert len(state.learner_buffer) == 0

    def test_bootstrap_values(self, coordinator, valid_features):
        """Test bootstrap value computation."""
        # Fill warmup frames
        for frame in range(coordinator.warmup_frames + 1):
            worker_states = [
                (0, valid_features.clone(), 0.0, False),
                (1, valid_features.clone(), 0.0, False),
            ]
            results, timings = coordinator.process_states(worker_states)

        bootstrap = coordinator.bootstrap_values()

        assert 0 in bootstrap
        assert 1 in bootstrap
        assert isinstance(bootstrap[0], torch.Tensor)
        assert isinstance(bootstrap[1], torch.Tensor)


class TestTrajectorySlicer:
    """Test TrajectorySlicer functionality."""

    @pytest.fixture
    def mock_coordinator(self):
        """Create a mock coordinator for testing."""
        from ppo.inference_coordinator import StepRecord

        coordinator = MagicMock()
        coordinator.num_workers = 2

        # Create some fake step records
        feature_dim = 64
        records = {
            0: [
                StepRecord(
                    worker_id=0,
                    state=torch.randn(feature_dim),
                    action_logits={
                        "main_stick": torch.randn(64),
                        "c_stick": torch.randn(9),
                        "buttons": torch.randn(5),
                        "shoulder": torch.randn(5),
                    },
                    action_taken={
                        "main_stick": torch.tensor(0),
                        "c_stick": torch.tensor(0),
                        "buttons": torch.zeros(5, dtype=torch.bool),
                        "shoulder": torch.tensor(0),
                    },
                    log_prob=torch.tensor(-1.0),
                    value=torch.tensor(0.5),
                    reward=0.1 * i,
                )
                for i in range(100)
            ],
            1: [
                StepRecord(
                    worker_id=1,
                    state=torch.randn(feature_dim),
                    action_logits={
                        "main_stick": torch.randn(64),
                        "c_stick": torch.randn(9),
                        "buttons": torch.randn(5),
                        "shoulder": torch.randn(5),
                    },
                    action_taken={
                        "main_stick": torch.tensor(0),
                        "c_stick": torch.tensor(0),
                        "buttons": torch.zeros(5, dtype=torch.bool),
                        "shoulder": torch.tensor(0),
                    },
                    log_prob=torch.tensor(-1.0),
                    value=torch.tensor(0.5),
                    reward=0.1 * i,
                )
                for i in range(100)
            ],
        }

        coordinator.get_all_step_records.return_value = records
        coordinator.bootstrap_values.return_value = {
            0: torch.tensor(0.5),
            1: torch.tensor(0.5),
        }

        return coordinator

    def test_gae_with_bootstrap(self, mock_coordinator):
        """Test GAE computation with bootstrap value."""
        from ppo.trajectory_slicer import TrajectorySlicer
        from ppo.trajectory import Trajectory, Step

        slicer = TrajectorySlicer(
            coordinator=mock_coordinator,
            rollout_length=100,
            gamma=0.99,
            gae_lambda=0.95,
        )

        # Create a simple trajectory
        steps = [
            Step(
                state=torch.randn(64),
                action_logits={"main_stick": torch.randn(64)},
                action_taken={"main_stick": torch.tensor(0)},
                log_prob=torch.tensor(-1.0),
                value=torch.tensor(float(i) * 0.1),
                reward=1.0,
                done=False,
            )
            for i in range(10)
        ]
        traj = Trajectory(steps=steps)

        bootstrap_value = torch.tensor(5.0)
        slicer._compute_gae_with_bootstrap(traj, bootstrap_value)

        assert traj.advantages is not None
        assert traj.returns is not None
        assert len(traj.advantages) == 10
        assert len(traj.returns) == 10

        # Returns should be reasonable (not NaN or inf)
        assert not torch.isnan(traj.returns).any()
        assert not torch.isinf(traj.returns).any()

    def test_prepare_training_data_creates_windows(self, mock_coordinator):
        """Test that training data is properly windowed."""
        from ppo.trajectory_slicer import TrajectorySlicer, RolloutSlice
        from ppo.trajectory import Step

        slicer = TrajectorySlicer(
            coordinator=mock_coordinator,
            rollout_length=100,
        )

        # Create rollout slice
        worker_steps = {
            0: [
                Step(
                    state=torch.randn(64),
                    action_logits={
                        "main_stick": torch.randn(64),
                        "c_stick": torch.randn(9),
                        "buttons": torch.randn(5),
                        "shoulder": torch.randn(5),
                    },
                    action_taken={
                        "main_stick": torch.tensor(0),
                        "c_stick": torch.tensor(0),
                        "buttons": torch.zeros(5, dtype=torch.bool),
                        "shoulder": torch.tensor(0),
                    },
                    log_prob=torch.tensor(-1.0),
                    value=torch.tensor(0.5),
                    reward=0.1,
                    done=False,
                )
                for _ in range(100)
            ]
        }

        rollout = RolloutSlice(
            worker_steps=worker_steps,
            bootstrap_values={0: torch.tensor(0.5)},
            total_frames=100,
        )

        windows = slicer.prepare_training_data(
            rollout, seq_len=32, device=torch.device("cpu")
        )

        assert "states" in windows
        assert "advantages" in windows
        assert "returns" in windows
        assert "valid_mask" in windows
        assert windows["states"].shape[1] == 32  # seq_len


class TestOpponentPoolAssignment:
    """Test new OpponentPool assignment methods."""

    @pytest.fixture
    def pool_with_opponents(self):
        """Create a pool with some opponents."""
        from ppo.opponent_pool import OpponentPool
        from model.nano_gpt import GPT

        with tempfile.TemporaryDirectory() as tmpdir:
            pool = OpponentPool(max_size=5, pool_dir=Path(tmpdir))

            config = get_config()
            model = GPT(config)

            # Add a few opponents
            pool.add_opponent(model, metadata={"episode": 1})
            pool.add_opponent(model, metadata={"episode": 2})
            pool.add_opponent(model, metadata={"episode": 3})

            yield pool

    def test_get_opponent_path_existing(self, pool_with_opponents):
        """Test getting path for existing opponent."""
        pool = pool_with_opponents

        # Opponent IDs are 0, 1, 2
        path = pool.get_opponent_path(0)
        assert path is not None
        assert path.exists()

    def test_get_opponent_path_nonexistent(self, pool_with_opponents):
        """Test getting path for non-existent opponent."""
        pool = pool_with_opponents

        path = pool.get_opponent_path(999)
        assert path is None

    def test_get_all_opponent_paths(self, pool_with_opponents):
        """Test getting all opponent paths."""
        pool = pool_with_opponents

        paths = pool.get_all_opponent_paths()
        assert len(paths) == 3
        for path in paths:
            assert path.exists()

    def test_get_latest_opponent(self, pool_with_opponents):
        """Test getting latest opponent."""
        pool = pool_with_opponents

        latest = pool.get_latest_opponent()
        assert latest is not None
        path, meta = latest
        assert meta["episode"] == 3  # Most recent

    def test_get_latest_opponent_empty_pool(self):
        """Test getting latest from empty pool."""
        from ppo.opponent_pool import OpponentPool

        with tempfile.TemporaryDirectory() as tmpdir:
            pool = OpponentPool(max_size=5, pool_dir=Path(tmpdir))
            assert pool.get_latest_opponent() is None


class TestWorkerConfig:
    """Test WorkerConfig dataclass."""

    def test_worker_config_defaults(self):
        """Test WorkerConfig default values."""
        from ppo.simulation_worker import WorkerConfig

        config = WorkerConfig(
            worker_id=0,
            dolphin_path="/path/to/dolphin",
            iso_path="/path/to/melee.iso",
        )

        assert config.worker_id == 0
        assert config.learner_port == 1
        assert config.opponent_port == 2

    def test_worker_config_custom_ports(self):
        """Test WorkerConfig with custom ports."""
        from ppo.simulation_worker import WorkerConfig

        config = WorkerConfig(
            worker_id=5,
            dolphin_path="/path/to/dolphin",
            iso_path="/path/to/melee.iso",
            learner_port=3,
            opponent_port=4,
        )

        assert config.learner_port == 3
        assert config.opponent_port == 4


class TestStepRecord:
    """Test StepRecord dataclass."""

    def test_step_record_creation(self):
        """Test creating a StepRecord."""
        from ppo.inference_coordinator import StepRecord

        record = StepRecord(
            worker_id=0,
            state=torch.randn(64),
            action_logits={"main_stick": torch.randn(64)},
            action_taken={"main_stick": torch.tensor(5)},
            log_prob=torch.tensor(-2.5),
            value=torch.tensor(0.8),
            reward=0.5,
        )

        assert record.worker_id == 0
        assert record.reward == 0.5
        assert record.state.shape == (64,)

    def test_step_record_default_reward(self):
        """Test StepRecord default reward is 0."""
        from ppo.inference_coordinator import StepRecord

        record = StepRecord(
            worker_id=0,
            state=torch.randn(64),
            action_logits={},
            action_taken={},
            log_prob=torch.tensor(0.0),
            value=torch.tensor(0.0),
        )

        assert record.reward == 0.0


class TestRolloutSlice:
    """Test RolloutSlice dataclass."""

    def test_rollout_slice_creation(self):
        """Test creating a RolloutSlice."""
        from ppo.trajectory_slicer import RolloutSlice
        from ppo.trajectory import Step

        worker_steps = {
            0: [
                Step(
                    state=torch.randn(64),
                    action_logits={},
                    action_taken={},
                    log_prob=torch.tensor(0.0),
                    value=torch.tensor(0.0),
                    reward=0.0,
                    done=False,
                )
            ]
        }

        rollout = RolloutSlice(
            worker_steps=worker_steps,
            bootstrap_values={0: torch.tensor(0.5)},
            total_frames=100,
        )

        assert rollout.total_frames == 100
        assert 0 in rollout.worker_steps
        assert 0 in rollout.bootstrap_values
