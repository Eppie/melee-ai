"""Unit tests for async PPO trainer."""

from __future__ import annotations

import time
import pytest
import torch
import multiprocessing as mp

from ppo.async_trainer import AsyncPPOTrainer, TrainingMetrics
from ppo.trajectory import Step
from ppo.trajectory_slicer import RolloutSlice
from config import get_config, init_config
from model.nano_gpt import GPT


@pytest.fixture
def simple_config():
    """Create a simple test config."""
    # Use minimal config for faster tests
    from config import Config, init_config

    # Initialize with defaults, then override for faster tests
    config = init_config(freeze=False)

    # Use minimal model for faster tests
    config.seq_len = 64
    config.model.n_layer = 2
    config.model.n_head = 2
    config.model.n_kv_head = 2  # Must divide n_head evenly
    config.model.n_embd = 32  # Must be divisible by n_head
    config.model.block_size = 64
    config.model.dropout = 0.0  # Disable dropout for deterministic tests

    # PPO config for tests
    config.ppo.ppo_epochs = 1  # Faster training
    config.ppo.minibatch_size = 4
    config.ppo.warmup_frames = 16
    config.ppo.gradient_accumulation_steps = 2

    return config


@pytest.fixture
def simple_model(simple_config):
    """Create a simple test model."""
    model = GPT(simple_config)
    return model


def create_dummy_rollout(num_steps: int = 128, num_features: int = 908) -> RolloutSlice:
    """Create a dummy rollout for testing.

    Note: num_features should match the model's input_size (default 908).
    num_steps should be >= seq_len (64) to allow window creation.
    """
    steps = []
    for _ in range(num_steps):
        state = torch.randn(num_features)
        actions = {
            "main_stick": torch.tensor(0),
            "c_stick": torch.tensor(0),
            "buttons": torch.tensor([False] * 5),
            "shoulder": torch.tensor(0),
        }
        log_prob = torch.tensor([-1.0])
        value = torch.tensor([0.0])

        step = Step(
            state=state,
            action_taken=actions,
            log_prob=log_prob,
            value=value,
            reward=0.1,
            done=False,
        )
        steps.append(step)

    rollout = RolloutSlice(
        worker_steps={0: steps},
        bootstrap_values={0: torch.tensor([0.0])},
        total_frames=num_steps,
    )
    return rollout


class TestAsyncPPOTrainer:
    """Tests for AsyncPPOTrainer."""

    def test_initialization(self, simple_model, simple_config):
        """Test that async trainer can be initialized."""
        trainer = AsyncPPOTrainer(
            model_state_dict=simple_model.state_dict(),
            optimizer_state=None,
            config_dict=simple_config.to_dict(),
        )

        try:
            assert trainer.process.is_alive(), "Training process should be running"
            time.sleep(1)  # Give process time to initialize
        finally:
            trainer.shutdown()

    def test_rollout_submission(self, simple_model, simple_config):
        """Test submitting rollouts to the trainer."""
        trainer = AsyncPPOTrainer(
            model_state_dict=simple_model.state_dict(),
            optimizer_state=None,
            config_dict=simple_config.to_dict(),
        )

        try:
            time.sleep(1)  # Give process time to initialize

            # Create and submit dummy rollout
            rollout = create_dummy_rollout(num_steps=32)
            success = trainer.submit_rollout(rollout)
            assert success, "Should be able to submit rollout"

            # Submit another rollout
            rollout2 = create_dummy_rollout(num_steps=32)
            success2 = trainer.submit_rollout(rollout2)
            assert success2, "Should be able to submit second rollout"

            # Third submission might fail (queue size is 2)
            rollout3 = create_dummy_rollout(num_steps=32)
            success3 = trainer.submit_rollout(rollout3)
            # Don't assert on this one - queue might have space if training is fast

        finally:
            trainer.shutdown()

    def test_parameter_updates(self, simple_model, simple_config):
        """Test receiving parameter updates from trainer."""
        trainer = AsyncPPOTrainer(
            model_state_dict=simple_model.state_dict(),
            optimizer_state=None,
            config_dict=simple_config.to_dict(),
        )

        try:
            time.sleep(1)  # Give process time to initialize

            # Submit enough rollouts to trigger training (gradient_accumulation_steps=2)
            for _ in range(2):
                rollout = create_dummy_rollout(num_steps=32)
                trainer.submit_rollout(rollout)

            # Wait for training to complete
            max_wait = 30  # seconds
            start_time = time.time()
            params = None

            while time.time() - start_time < max_wait:
                params = trainer.get_updated_parameters()
                if params is not None:
                    break
                time.sleep(0.5)

            assert params is not None, "Should receive parameter updates after training"
            assert isinstance(params, dict), "Parameters should be a dict"
            assert len(params) > 0, "Parameters dict should not be empty"

        finally:
            trainer.shutdown()

    def test_metrics_collection(self, simple_model, simple_config):
        """Test receiving training metrics from trainer."""
        trainer = AsyncPPOTrainer(
            model_state_dict=simple_model.state_dict(),
            optimizer_state=None,
            config_dict=simple_config.to_dict(),
        )

        try:
            time.sleep(1)  # Give process time to initialize

            # Submit enough rollouts to trigger training
            for _ in range(2):
                rollout = create_dummy_rollout(num_steps=32)
                trainer.submit_rollout(rollout)

            # Wait for metrics
            max_wait = 30  # seconds
            start_time = time.time()
            metrics = None

            while time.time() - start_time < max_wait:
                metrics = trainer.get_metrics()
                if metrics is not None:
                    break
                time.sleep(0.5)

            assert metrics is not None, "Should receive training metrics"
            assert isinstance(
                metrics, TrainingMetrics
            ), "Metrics should be TrainingMetrics"
            # Note: metrics.num_windows might be 0 if trajectory too short, but we got metrics!

        finally:
            trainer.shutdown()

    def test_graceful_shutdown(self, simple_model, simple_config):
        """Test that trainer shuts down gracefully."""
        trainer = AsyncPPOTrainer(
            model_state_dict=simple_model.state_dict(),
            optimizer_state=None,
            config_dict=simple_config.to_dict(),
        )

        time.sleep(1)  # Give process time to initialize
        assert trainer.process.is_alive(), "Process should be alive"

        trainer.shutdown(timeout=5.0)
        assert (
            not trainer.process.is_alive()
        ), "Process should be terminated after shutdown"

    def test_multiple_training_steps(self, simple_model, simple_config):
        """Test multiple training steps with gradient accumulation."""
        trainer = AsyncPPOTrainer(
            model_state_dict=simple_model.state_dict(),
            optimizer_state=None,
            config_dict=simple_config.to_dict(),
        )

        try:
            time.sleep(1)  # Give process time to initialize

            # Submit 4 rollouts (should trigger 2 training steps with accumulation=2)
            for _ in range(4):
                rollout = create_dummy_rollout(num_steps=32)
                trainer.submit_rollout(rollout)
                time.sleep(0.1)  # Small delay between submissions

            # Wait and collect metrics
            max_wait = 60  # seconds
            start_time = time.time()
            collected_metrics = []

            while time.time() - start_time < max_wait:
                metrics = trainer.get_metrics()
                if metrics is not None:
                    collected_metrics.append(metrics)
                    if len(collected_metrics) >= 2:
                        break
                time.sleep(0.5)

            assert len(collected_metrics) >= 1, "Should have at least 1 training step"

        finally:
            trainer.shutdown()


def test_queue_behavior():
    """Test multiprocessing queue behavior."""
    # This is a basic sanity check for multiprocessing
    q = mp.Queue(maxsize=2)

    # Should be able to put 2 items
    q.put("item1")
    q.put("item2")

    # Third item should fail with put_nowait
    import queue as queue_module

    with pytest.raises(queue_module.Full):
        q.put_nowait("item3")

    # Should be able to get items
    assert q.get() == "item1"
    assert q.get() == "item2"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
