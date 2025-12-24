"""TDD tests for training session builder (Cluster 1).

This test suite captures the requirements for the TrainingSessionBuilder
before implementation, following TDD principles.

The builder pattern should:
1. Enforce dependency order (config first, model required)
2. Support fresh model creation and checkpoint loading
3. Support custom factories for optimizer/dataloader
4. Handle AMP configuration with optional override
5. Provide convenience functions for common patterns
"""

from __future__ import annotations

import pytest
import torch
from pathlib import Path


class TestTrainingSessionBuilderBasics:
    """Test basic builder functionality and structure."""

    def test_builder_can_be_instantiated(self):
        """Test that builder can be created."""
        from train.session_builder import TrainingSessionBuilder

        builder = TrainingSessionBuilder()
        assert builder is not None

    def test_minimal_session_requires_config_and_model(self):
        """Test that minimal session needs config and model."""
        from train.session_builder import TrainingSessionBuilder

        builder = TrainingSessionBuilder()

        # Should fail without config
        with pytest.raises(ValueError, match="config"):
            builder.with_fresh_model()

        # Should fail without model
        with pytest.raises(ValueError, match="model"):
            builder.with_config().build()

    def test_minimal_session_succeeds_with_config_and_model(self):
        """Test that minimal session can be built with config and model."""
        from train.session_builder import TrainingSessionBuilder

        builder = TrainingSessionBuilder()
        session = builder.with_config().with_fresh_model().build()

        assert session.config is not None
        assert session.model is not None
        assert session.device is not None
        assert session.amp is not None

    def test_session_has_required_fields(self):
        """Test that session has all expected fields."""
        from train.session_builder import TrainingSessionBuilder

        builder = TrainingSessionBuilder()
        session = builder.with_config().with_fresh_model().build()

        # Required fields
        assert hasattr(session, 'config')
        assert hasattr(session, 'model')
        assert hasattr(session, 'device')
        assert hasattr(session, 'amp')

        # Optional fields
        assert hasattr(session, 'optimizer')
        assert hasattr(session, 'scaler')
        assert hasattr(session, 'loader')
        assert hasattr(session, 'dataset')
        assert hasattr(session, 'sampler')


class TestBuilderDependencyOrder:
    """Test that builder enforces correct dependency order."""

    def test_config_must_be_first(self):
        """Test that config must be set before other methods."""
        from train.session_builder import TrainingSessionBuilder

        builder = TrainingSessionBuilder()

        # Should fail to call with_fresh_model before with_config
        with pytest.raises(ValueError, match="config"):
            builder.with_fresh_model()

        # Should fail to call with_dataloader before with_config
        with pytest.raises(ValueError, match="config"):
            builder.with_dataloader()

    def test_model_required_before_build(self):
        """Test that model must be set before building."""
        from train.session_builder import TrainingSessionBuilder

        builder = TrainingSessionBuilder()

        with pytest.raises(ValueError, match="model"):
            builder.with_config().build()

    def test_build_order_is_enforced(self):
        """Test that build enforces proper initialization order."""
        from train.session_builder import TrainingSessionBuilder

        # Correct order should succeed
        builder = TrainingSessionBuilder()
        session = (
            builder.with_config()
            .with_fresh_model()
            .with_optimizer()
            .with_amp()
            .build()
        )

        assert session.optimizer is not None


class TestFreshModelCreation:
    """Test fresh model creation."""

    def test_with_fresh_model_creates_new_model(self):
        """Test that with_fresh_model creates a new model from config."""
        from train.session_builder import TrainingSessionBuilder

        builder = TrainingSessionBuilder()
        session = builder.with_config().with_fresh_model().build()

        assert session.model is not None
        assert hasattr(session.model, 'forward')
        # Should be a GPT model
        from model.nano_gpt import GPT
        assert isinstance(session.model, GPT)

    def test_model_moved_to_device(self):
        """Test that model is moved to correct device."""
        from train.session_builder import TrainingSessionBuilder

        builder = TrainingSessionBuilder()
        session = builder.with_config().with_fresh_model().build()

        # Model should be on the same device as session.device
        # Compare device types (mps/cuda/cpu) since index may differ
        model_device = next(session.model.parameters()).device
        assert model_device.type == session.device.type


class TestCheckpointLoading:
    """Test checkpoint loading functionality."""

    @pytest.mark.skip(reason="Requires actual checkpoint file")
    def test_with_model_from_checkpoint_loads_model(self):
        """Test loading model from checkpoint."""
        from train.session_builder import TrainingSessionBuilder

        checkpoint_path = Path("checkpoints/test_checkpoint.pt")
        if not checkpoint_path.exists():
            pytest.skip("No checkpoint available")

        builder = TrainingSessionBuilder()
        session = (
            builder.with_config()
            .with_model_from_checkpoint(checkpoint_path)
            .build()
        )

        assert session.model is not None


class TestOptimizerConfiguration:
    """Test optimizer configuration."""

    def test_with_optimizer_uses_default_factory(self):
        """Test that with_optimizer uses default factory."""
        from train.session_builder import TrainingSessionBuilder

        builder = TrainingSessionBuilder()
        session = (
            builder.with_config()
            .with_fresh_model()
            .with_optimizer()
            .build()
        )

        assert session.optimizer is not None
        # Should be an optimizer
        assert isinstance(session.optimizer, torch.optim.Optimizer)

    def test_with_optimizer_accepts_custom_factory(self):
        """Test custom optimizer factory."""
        from train.session_builder import TrainingSessionBuilder

        def custom_optimizer(model, config):
            return torch.optim.SGD(model.parameters(), lr=0.001)

        builder = TrainingSessionBuilder()
        session = (
            builder.with_config()
            .with_fresh_model()
            .with_optimizer(factory=custom_optimizer)
            .build()
        )

        assert isinstance(session.optimizer, torch.optim.SGD)

    def test_session_without_optimizer_is_valid(self):
        """Test that optimizer is optional (for validation)."""
        from train.session_builder import TrainingSessionBuilder

        builder = TrainingSessionBuilder()
        session = builder.with_config().with_fresh_model().build()

        assert session.optimizer is None


class TestAMPConfiguration:
    """Test AMP (Automatic Mixed Precision) configuration."""

    def test_amp_is_configured_by_default(self):
        """Test that AMP is configured even without explicit call."""
        from train.session_builder import TrainingSessionBuilder

        builder = TrainingSessionBuilder()
        session = builder.with_config().with_fresh_model().build()

        assert session.amp is not None
        assert hasattr(session.amp, 'enabled')
        assert hasattr(session.amp, 'dtype')

    def test_amp_override_changes_enabled_state(self):
        """Test that AMP override works."""
        from train.session_builder import TrainingSessionBuilder

        # Test with override=False
        builder = TrainingSessionBuilder()
        session = (
            builder.with_config()
            .with_fresh_model()
            .with_amp(override_use_amp=False)
            .build()
        )

        # AMP should be disabled
        assert session.amp.enabled is False

    def test_grad_scaler_created_when_amp_enabled(self):
        """Test that GradScaler is created when AMP enabled with optimizer."""
        from train.session_builder import TrainingSessionBuilder

        builder = TrainingSessionBuilder()
        session = (
            builder.with_config()
            .with_fresh_model()
            .with_optimizer()
            .with_amp(override_use_amp=True)
            .build()
        )

        # If AMP is enabled and optimizer present, scaler should exist
        if session.amp.enabled:
            assert session.scaler is not None


class TestDataLoaderConfiguration:
    """Test DataLoader configuration."""

    def test_dataloader_is_optional(self):
        """Test that dataloader is optional."""
        from train.session_builder import TrainingSessionBuilder

        builder = TrainingSessionBuilder()
        session = builder.with_config().with_fresh_model().build()

        assert session.loader is None
        assert session.dataset is None
        assert session.sampler is None

    @pytest.mark.skip(reason="Requires preprocessed data files")
    def test_with_dataloader_creates_loader(self):
        """Test that with_dataloader creates loader."""
        from train.session_builder import TrainingSessionBuilder

        builder = TrainingSessionBuilder()
        session = (
            builder.with_config()
            .with_fresh_model()
            .with_dataloader()
            .build()
        )

        assert session.loader is not None
        assert session.dataset is not None
        # sampler may or may not be present depending on config


class TestConvenienceFunctions:
    """Test convenience functions for common patterns."""

    @pytest.mark.skip(reason="Requires preprocessed data files")
    def test_build_imitation_session(self):
        """Test imitation learning convenience function."""
        from train.session_builder import build_imitation_session

        session = build_imitation_session(cli_overrides={})

        assert session.config is not None
        assert session.model is not None
        assert session.loader is not None
        assert session.optimizer is not None

    @pytest.mark.skip(reason="Requires checkpoint")
    def test_build_validation_session(self):
        """Test validation convenience function."""
        from train.session_builder import build_validation_session

        checkpoint_path = Path("checkpoints/test.pt")
        if not checkpoint_path.exists():
            pytest.skip("No checkpoint available")

        session = build_validation_session(
            cli_overrides={}, checkpoint_path=checkpoint_path, with_dataloader=True
        )

        assert session.model is not None
        assert session.loader is not None
        assert session.optimizer is None  # No optimizer for validation


class TestFluentInterface:
    """Test fluent interface (method chaining)."""

    def test_methods_return_builder(self):
        """Test that builder methods return self for chaining."""
        from train.session_builder import TrainingSessionBuilder

        builder = TrainingSessionBuilder()

        # All methods should return builder for chaining
        result = builder.with_config()
        assert isinstance(result, TrainingSessionBuilder)

        result = result.with_fresh_model()
        assert isinstance(result, TrainingSessionBuilder)

        result = result.with_optimizer()
        assert isinstance(result, TrainingSessionBuilder)

    @pytest.mark.skip(reason="Requires preprocessed data files")
    def test_full_chain_builds_successfully(self):
        """Test full method chain."""
        from train.session_builder import TrainingSessionBuilder

        session = (
            TrainingSessionBuilder()
            .with_config()
            .with_fresh_model()
            .with_dataloader()
            .with_optimizer()
            .with_amp()
            .build()
        )

        assert session is not None
        assert session.model is not None
        assert session.optimizer is not None
        assert session.loader is not None


class TestErrorHandling:
    """Test error handling and validation."""

    def test_invalid_order_raises_error(self):
        """Test that invalid call order raises clear errors."""
        from train.session_builder import TrainingSessionBuilder

        builder = TrainingSessionBuilder()

        # Calling methods before with_config should fail
        with pytest.raises(ValueError):
            builder.with_fresh_model()

    def test_build_without_model_raises_error(self):
        """Test that building without model raises error."""
        from train.session_builder import TrainingSessionBuilder

        builder = TrainingSessionBuilder()

        with pytest.raises(ValueError):
            builder.with_config().build()

    def test_double_model_specification_works(self):
        """Test that specifying model twice uses latest."""
        from train.session_builder import TrainingSessionBuilder

        builder = TrainingSessionBuilder()

        # Should use the last model specification
        session = (
            builder.with_config()
            .with_fresh_model()  # First specification
            .with_fresh_model()  # Second specification (should override)
            .build()
        )

        assert session.model is not None
