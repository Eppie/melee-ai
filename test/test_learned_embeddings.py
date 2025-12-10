"""Unit tests for learned embeddings vs one-hot encoding."""

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


class TestLearnedEmbeddings:
    """Test the learned embeddings functionality."""

    @pytest.fixture
    def sample_inputs(self):
        """Create sample inputs for the model using proper schema."""
        batch_size = 2
        sequence_length = 16
        device = torch.device("cpu")

        # Use the actual schema to get proper dimensions
        colmap = ColumnMap(get_feature_names(), get_target_names())
        X = torch.zeros(batch_size, sequence_length, len(colmap.feat_names), device=device)
        inputs = build_model_inputs(X, colmap)

        return inputs

    def test_learned_embeddings_enabled_by_default(self):
        """Test that learned embeddings are enabled by default."""
        config = init_config()
        assert config.model.use_learned_embeddings is True

    def test_model_with_learned_embeddings(self, sample_inputs):
        """Test model creation and forward pass with learned embeddings."""
        config = init_config(
            overrides={
                "model.use_learned_embeddings": "true",
                "model.block_size": "64",
                "model.n_embd": "128",
                "model.n_layer": "2",
            }
        )
        model = GPT(config)
        model.eval()

        # Check that embedding layers exist
        assert hasattr(model, "stage_embedding")
        assert hasattr(model, "character_embedding")
        assert hasattr(model, "action_embedding")
        assert model.use_learned_embeddings

        # Check embedding dimensions
        assert model.stage_embedding.embedding_dim == config.model.embedding_dim_stage
        assert model.character_embedding.embedding_dim == config.model.embedding_dim_character
        assert model.action_embedding.embedding_dim == config.model.embedding_dim_action

        # Forward pass should work
        with torch.no_grad():
            outputs = model(sample_inputs)

        # Check that outputs are valid
        batch_size, sequence_length = sample_inputs.batch_size
        assert outputs["buttons"].shape[:2] == (batch_size, sequence_length)
        assert outputs["value"].shape == (batch_size, sequence_length, 1)
        assert torch.isfinite(outputs["buttons"]).all()
        assert torch.isfinite(outputs["value"]).all()

    def test_model_with_one_hot_encoding(self, sample_inputs):
        """Test model creation and forward pass with one-hot encoding."""
        config = init_config(
            overrides={
                "model.use_learned_embeddings": "false",
                "model.block_size": "64",
                "model.n_embd": "128",
                "model.n_layer": "2",
            }
        )
        model = GPT(config)
        model.eval()

        # Check that embedding layers don't exist
        assert not hasattr(model, "stage_embedding")
        assert not hasattr(model, "character_embedding")
        assert not hasattr(model, "action_embedding")
        assert not model.use_learned_embeddings

        # Forward pass should work
        with torch.no_grad():
            outputs = model(sample_inputs)

        # Check that outputs are valid
        batch_size, sequence_length = sample_inputs.batch_size
        assert outputs["buttons"].shape[:2] == (batch_size, sequence_length)
        assert outputs["value"].shape == (batch_size, sequence_length, 1)
        assert torch.isfinite(outputs["buttons"]).all()
        assert torch.isfinite(outputs["value"]).all()

    def test_input_size_difference(self):
        """Test that learned embeddings produce smaller input_size than one-hot."""
        config_learned = init_config(
            overrides={
                "model.use_learned_embeddings": "true",
                "model.block_size": "64",
            }
        )
        input_size_learned = config_learned.model.input_size

        reset_config()

        config_onehot = init_config(
            overrides={
                "model.use_learned_embeddings": "false",
                "model.block_size": "64",
            }
        )
        input_size_onehot = config_onehot.model.input_size

        # Learned embeddings should produce smaller input size
        # One-hot: 6 + 26*2 + 396*2 = 850 categorical dims
        # Learned: 8 + 16*2 + 32*2 = 104 categorical dims
        assert input_size_learned < input_size_onehot

        # Verify the expected difference (850 - 104 = 746)
        categorical_diff = (
            config_onehot.model.num_stages +
            config_onehot.model.num_characters * 2 +
            config_onehot.model.num_actions * 2
        ) - (
            config_learned.model.embedding_dim_stage +
            config_learned.model.embedding_dim_character * 2 +
            config_learned.model.embedding_dim_action * 2
        )
        assert input_size_onehot - input_size_learned == categorical_diff

    def test_parameter_count_difference(self):
        """Test parameter counts between learned embeddings and one-hot."""
        config_learned = init_config(
            overrides={
                "model.use_learned_embeddings": "true",
                "model.block_size": "64",
                "model.n_embd": "128",
                "model.n_layer": "2",
            }
        )
        model_learned = GPT(config_learned)
        params_learned = sum(p.numel() for p in model_learned.parameters())

        reset_config()

        config_onehot = init_config(
            overrides={
                "model.use_learned_embeddings": "false",
                "model.block_size": "64",
                "model.n_embd": "128",
                "model.n_layer": "2",
            }
        )
        model_onehot = GPT(config_onehot)
        params_onehot = sum(p.numel() for p in model_onehot.parameters())

        # Models will have different parameter counts due to:
        # 1. Embedding layers in learned model
        # 2. Different input_size affecting projection_down layer
        # The learned model should have fewer parameters overall
        # because the projection_down layer is much smaller
        assert params_learned < params_onehot

    def test_gradient_flow_with_learned_embeddings(self, sample_inputs):
        """Test that gradients flow correctly through learned embeddings."""
        config = init_config(
            overrides={
                "model.use_learned_embeddings": "true",
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

        # Check that embedding layers have gradients
        assert model.stage_embedding.weight.grad is not None
        assert model.character_embedding.weight.grad is not None
        assert model.action_embedding.weight.grad is not None
        assert torch.isfinite(model.stage_embedding.weight.grad).all()
        assert torch.isfinite(model.character_embedding.weight.grad).all()
        assert torch.isfinite(model.action_embedding.weight.grad).all()

        # Check that other parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    def test_gradient_flow_with_one_hot(self, sample_inputs):
        """Test that gradients flow correctly with one-hot encoding."""
        config = init_config(
            overrides={
                "model.use_learned_embeddings": "false",
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

        # Check that all parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    def test_custom_embedding_dimensions(self, sample_inputs):
        """Test that custom embedding dimensions work correctly."""
        config = init_config(
            overrides={
                "model.use_learned_embeddings": "true",
                "model.embedding_dim_stage": "16",
                "model.embedding_dim_character": "32",
                "model.embedding_dim_action": "64",
                "model.block_size": "64",
                "model.n_embd": "128",
                "model.n_layer": "2",
            }
        )
        model = GPT(config)
        model.eval()

        # Check custom embedding dimensions
        assert model.stage_embedding.embedding_dim == 16
        assert model.character_embedding.embedding_dim == 32
        assert model.action_embedding.embedding_dim == 64

        # Forward pass should work
        with torch.no_grad():
            outputs = model(sample_inputs)

        assert torch.isfinite(outputs["buttons"]).all()
        assert torch.isfinite(outputs["value"]).all()

    def test_outputs_have_same_shapes(self, sample_inputs):
        """Test that both modes produce outputs with the same shapes."""
        config_learned = init_config(
            overrides={
                "model.use_learned_embeddings": "true",
                "model.block_size": "64",
                "model.n_embd": "128",
                "model.n_layer": "2",
            }
        )
        model_learned = GPT(config_learned)
        model_learned.eval()

        reset_config()

        config_onehot = init_config(
            overrides={
                "model.use_learned_embeddings": "false",
                "model.block_size": "64",
                "model.n_embd": "128",
                "model.n_layer": "2",
            }
        )
        model_onehot = GPT(config_onehot)
        model_onehot.eval()

        with torch.no_grad():
            outputs_learned = model_learned(sample_inputs)
            outputs_onehot = model_onehot(sample_inputs)

        # Output shapes should be identical
        for key in ["buttons", "main_stick", "c_stick", "shoulder", "value"]:
            assert outputs_learned[key].shape == outputs_onehot[key].shape, \
                f"Shape mismatch for {key}"

    def test_different_head_flow_modes(self, sample_inputs):
        """Test that learned embeddings work with different head_flow modes."""
        for head_flow in ["sequential", "parallel"]:
            reset_config()
            config = init_config(
                overrides={
                    "model.use_learned_embeddings": "true",
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

            assert "buttons" in outputs
            assert "value" in outputs
            assert torch.isfinite(outputs["buttons"]).all()
            assert torch.isfinite(outputs["value"]).all()

    def test_config_override_via_cli(self):
        """Test that use_learned_embeddings can be set via CLI override."""
        # Test enabling (default)
        config = init_config()
        assert config.model.use_learned_embeddings is True

        reset_config()

        # Test disabling
        config = init_config(overrides={"model.use_learned_embeddings": "false"})
        assert config.model.use_learned_embeddings is False

        reset_config()

        # Test custom embedding dimensions
        config = init_config(
            overrides={
                "model.embedding_dim_stage": "4",
                "model.embedding_dim_character": "8",
                "model.embedding_dim_action": "16",
            }
        )
        assert config.model.embedding_dim_stage == 4
        assert config.model.embedding_dim_character == 8
        assert config.model.embedding_dim_action == 16
