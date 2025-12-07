"""Unit tests for optional future_x and future_y heads."""

import pytest
import torch

from column_map import ColumnMap
from config import init_config, reset_config
from model.nano_gpt import GPT
from schema import get_feature_names, get_target_names
from train.batch_utils import build_model_inputs


@pytest.fixture(autouse=True)
def clear_global_config():
    """Ensure init_config() global state does not leak between tests."""
    reset_config()
    yield
    reset_config()


class TestOptionalFutureHeads:
    """Test the optional future heads functionality."""

    @pytest.fixture
    def sample_inputs(self):
        """Create sample inputs for the model using proper schema."""
        batch_size = 2
        sequence_length = 16
        device = torch.device("cpu")

        # Use the actual schema to get proper dimensions
        colmap = ColumnMap(get_feature_names(), get_target_names())
        # Include horizon feature (+1) to mirror training inputs
        # Use zeros to avoid negative values in categorical columns
        X = torch.zeros(batch_size, sequence_length, len(colmap.feat_names) + 1, device=device)
        inputs = build_model_inputs(X, colmap)

        return inputs

    def test_future_heads_disabled_by_default(self):
        """Test that future heads are disabled by default."""
        config = init_config()
        assert config.model.use_future_heads is False

    def test_model_without_future_heads(self, sample_inputs):
        """Test model creation and forward pass without future heads."""
        config = init_config(
            overrides={
                "model.use_future_heads": "false",
                "model.block_size": "64",
                "model.n_embd": "128",
                "model.n_layer": "2",
            }
        )
        model = GPT(config)
        model.eval()

        # Check that future heads don't exist
        assert not hasattr(model, "future_x_head")
        assert not hasattr(model, "future_y_head")
        assert not model.use_future_heads

        # Forward pass should work
        with torch.no_grad():
            outputs = model(sample_inputs)

        # Check that future heads are not in outputs
        assert "future_x" not in outputs
        assert "future_y" not in outputs

        # Check that other heads are still present
        assert "buttons" in outputs
        assert "main_stick" in outputs
        assert "c_stick" in outputs
        assert "shoulder" in outputs
        assert "value" in outputs

        # Check output shapes
        batch_size, sequence_length = sample_inputs.batch_size
        assert outputs["buttons"].shape[:2] == (batch_size, sequence_length)
        assert outputs["value"].shape == (batch_size, sequence_length, 1)

    def test_model_with_future_heads(self, sample_inputs):
        """Test model creation and forward pass with future heads enabled."""
        config = init_config(
            overrides={
                "model.use_future_heads": "true",
                "model.block_size": "64",
                "model.n_embd": "128",
                "model.n_layer": "2",
            }
        )
        model = GPT(config)
        model.eval()

        # Check that future heads exist
        assert hasattr(model, "future_x_head")
        assert hasattr(model, "future_y_head")
        assert model.use_future_heads

        # Forward pass should work
        with torch.no_grad():
            outputs = model(sample_inputs)

        # Check that all heads are in outputs
        assert "buttons" in outputs
        assert "main_stick" in outputs
        assert "c_stick" in outputs
        assert "shoulder" in outputs
        assert "future_x" in outputs
        assert "future_y" in outputs
        assert "value" in outputs

        # Check output shapes
        batch_size, sequence_length = sample_inputs.batch_size
        assert outputs["buttons"].shape[:2] == (batch_size, sequence_length)
        assert outputs["future_x"].shape == (
            batch_size,
            sequence_length,
            config.model.target_shapes_by_head["future_x"],
        )
        assert outputs["future_y"].shape == (
            batch_size,
            sequence_length,
            config.model.target_shapes_by_head["future_y"],
        )
        assert outputs["value"].shape == (batch_size, sequence_length, 1)

    def test_parameter_count_difference(self):
        """Test that disabling future heads reduces parameter count."""
        config_with_future = init_config(
            overrides={
                "model.use_future_heads": "true",
                "model.block_size": "64",
                "model.n_embd": "128",
                "model.n_layer": "2",
            }
        )
        model_with_future = GPT(config_with_future)
        params_with_future = sum(p.numel() for p in model_with_future.parameters())

        reset_config()

        config_without_future = init_config(
            overrides={
                "model.use_future_heads": "false",
                "model.block_size": "64",
                "model.n_embd": "128",
                "model.n_layer": "2",
            }
        )
        model_without_future = GPT(config_without_future)
        params_without_future = sum(p.numel() for p in model_without_future.parameters())

        # Model without future heads should have fewer parameters
        assert params_without_future < params_with_future

        # The difference should be roughly the size of two future heads
        # Each head has: hidden_dim * (embedding_dim + 1) + output_dim * (hidden_dim + 1)
        # With embedding_dim=128, hidden_dim=128, output_dim=32
        # Each head ≈ 128*(128+1) + 32*(128+1) ≈ 16512 + 4128 = 20640 params
        # Two heads ≈ 41280 params
        param_diff = params_with_future - params_without_future
        assert 35000 < param_diff < 50000, f"Unexpected param diff: {param_diff}"

    def test_gradient_flow_without_future_heads(self, sample_inputs):
        """Test that gradients flow correctly without future heads."""
        config = init_config(
            overrides={
                "model.use_future_heads": "false",
                "model.block_size": "64",
                "model.n_embd": "128",
                "model.n_layer": "2",
            }
        )
        model = GPT(config)
        model.train()

        outputs = model(sample_inputs)

        # Create a dummy loss from all outputs
        loss = (
            outputs["buttons"].sum()
            + outputs["main_stick"].sum()
            + outputs["c_stick"].sum()
            + outputs["shoulder"].sum()
            + outputs["value"].sum()
        )
        loss.backward()

        # Check that gradients exist and are finite for all parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    def test_gradient_flow_with_future_heads(self, sample_inputs):
        """Test that gradients flow correctly with future heads."""
        config = init_config(
            overrides={
                "model.use_future_heads": "true",
                "model.block_size": "64",
                "model.n_embd": "128",
                "model.n_layer": "2",
            }
        )
        model = GPT(config)
        model.train()

        outputs = model(sample_inputs)

        # Create a dummy loss from all outputs including future heads
        loss = (
            outputs["buttons"].sum()
            + outputs["main_stick"].sum()
            + outputs["c_stick"].sum()
            + outputs["shoulder"].sum()
            + outputs["future_x"].sum()
            + outputs["future_y"].sum()
            + outputs["value"].sum()
        )
        loss.backward()

        # Check that gradients exist and are finite for all parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    def test_outputs_are_finite_without_future_heads(self, sample_inputs):
        """Test that outputs are finite without future heads."""
        config = init_config(
            overrides={
                "model.use_future_heads": "false",
                "model.block_size": "64",
            }
        )
        model = GPT(config)
        model.eval()

        with torch.no_grad():
            outputs = model(sample_inputs)

        # All outputs should be finite
        for key in ["buttons", "main_stick", "c_stick", "shoulder", "value"]:
            assert torch.isfinite(outputs[key]).all(), f"Non-finite values in {key}"

    def test_different_head_flow_modes_without_future_heads(self, sample_inputs):
        """Test that future heads work correctly with different head_flow modes."""
        for head_flow in ["sequential", "parallel", "mix"]:
            reset_config()
            config = init_config(
                overrides={
                    "model.use_future_heads": "false",
                    "model.head_flow": head_flow,
                    "model.block_size": "64",
                    "model.n_embd": "128",
                    "model.n_layer": "2",
                }
            )
            model = GPT(config)
            model.eval()

            with torch.no_grad():
                outputs = model(sample_inputs)

            # Future heads should not be present regardless of head_flow mode
            assert "future_x" not in outputs
            assert "future_y" not in outputs

            # Other heads should be present
            assert "buttons" in outputs
            assert "main_stick" in outputs
            assert "value" in outputs

    def test_config_override_via_cli(self):
        """Test that use_future_heads can be set via CLI override."""
        # Test enabling
        config = init_config(overrides={"model.use_future_heads": "true"})
        assert config.model.use_future_heads is True

        reset_config()

        # Test disabling (explicit)
        config = init_config(overrides={"model.use_future_heads": "false"})
        assert config.model.use_future_heads is False
