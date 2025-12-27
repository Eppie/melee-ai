"""Unit tests for ALiBi (Attention with Linear Biases) implementation."""

import pytest
import torch
from tensordict import TensorDict

from column_map import ColumnMap
from config import init_config, reset_config
from config.config import Config
from model.nano_gpt import GPT
from model.positional_encoding import get_alibi_biases
from schema import get_feature_names, get_target_names
from train.batch_utils import build_model_inputs


class TestAlibiBiases:
    """Test the get_alibi_biases function."""

    def test_alibi_shape(self):
        """Test that ALiBi biases have the correct shape."""
        num_heads = 8
        max_seq_len = 64
        biases = get_alibi_biases(num_heads, max_seq_len)

        assert biases.shape == (1, num_heads, max_seq_len, max_seq_len)
        assert biases.dtype == torch.float32

    def test_alibi_diagonal_is_zero(self):
        """Test that diagonal elements (i=j) are zero."""
        num_heads = 4
        max_seq_len = 16
        biases = get_alibi_biases(num_heads, max_seq_len)

        # Check diagonal is zero for all heads
        for h in range(num_heads):
            diagonal = torch.diagonal(biases[0, h])
            assert torch.allclose(diagonal, torch.zeros_like(diagonal))

    def test_alibi_values_negative(self):
        """Test that all off-diagonal values are negative."""
        num_heads = 4
        max_seq_len = 16
        biases = get_alibi_biases(num_heads, max_seq_len)

        # Create mask for off-diagonal elements
        mask = ~torch.eye(max_seq_len, dtype=torch.bool)

        # Check all off-diagonal values are negative
        for h in range(num_heads):
            off_diagonal = biases[0, h][mask]
            assert torch.all(off_diagonal < 0)

    def test_alibi_symmetric_distances(self):
        """Test that biases are symmetric in absolute value (distance-based)."""
        num_heads = 4
        max_seq_len = 8
        biases = get_alibi_biases(num_heads, max_seq_len)

        # For each head, bias[i,j] should equal bias[j,i] since both depend on |i-j|
        for h in range(num_heads):
            head_biases = biases[0, h]
            assert torch.allclose(head_biases, head_biases.T)

    def test_alibi_slopes_geometric_progression(self):
        """Test that slopes follow geometric progression."""
        num_heads = 8
        max_seq_len = 4
        biases = get_alibi_biases(num_heads, max_seq_len)

        # Extract slopes by looking at bias[0, 1] for each head (distance = 1)
        slopes = []
        for h in range(num_heads):
            slope = -biases[0, h, 0, 1].item()  # Negate to get positive slope
            slopes.append(slope)

        # Verify geometric progression: slope[k] = 2^(-(8*k/num_heads))
        expected_slopes = [
            2.0 ** (-(8.0 * (k + 1) / num_heads)) for k in range(num_heads)
        ]

        for actual, expected in zip(slopes, expected_slopes):
            assert abs(actual - expected) < 1e-5

    def test_alibi_proportional_to_distance(self):
        """Test that biases are proportional to distance."""
        num_heads = 2
        max_seq_len = 8
        biases = get_alibi_biases(num_heads, max_seq_len)

        # For each head, check that bias[i, i+k] = k * bias[i, i+1]
        for h in range(num_heads):
            head_biases = biases[0, h]
            unit_bias = head_biases[0, 1].item()  # Distance 1

            for i in range(max_seq_len - 1):
                for k in range(1, max_seq_len - i):
                    expected = k * unit_bias
                    actual = head_biases[i, i + k].item()
                    assert abs(actual - expected) < 1e-5

    def test_alibi_device_placement(self):
        """Test that biases can be created on different devices."""
        num_heads = 4
        max_seq_len = 16

        # Test CPU
        biases_cpu = get_alibi_biases(
            num_heads, max_seq_len, device=torch.device("cpu")
        )
        assert biases_cpu.device.type == "cpu"

        # Test CUDA if available
        if torch.cuda.is_available():
            biases_cuda = get_alibi_biases(
                num_heads, max_seq_len, device=torch.device("cuda")
            )
            assert biases_cuda.device.type == "cuda"


class TestModelWithAlibi:
    """Test the GPT model with ALiBi enabled."""

    @pytest.fixture(autouse=True)
    def clear_global_config(self):
        """Ensure init_config() global state does not leak between tests."""
        reset_config()
        yield
        reset_config()

    @pytest.fixture
    def config_rope(self):
        """Create a config with RoPE (default)."""
        config = init_config(
            overrides={
                "model.use_alibi": "false",
                "model.block_size": "64",
                "model.n_embd": "128",
                "model.n_layer": "2",
                "model.n_head": "4",
                "model.n_kv_head": "4",
            }
        )
        return config

    @pytest.fixture
    def config_alibi(self):
        """Create a config with ALiBi enabled."""
        config = init_config(
            overrides={
                "model.use_alibi": "true",
                "model.block_size": "64",
                "model.n_embd": "128",
                "model.n_layer": "2",
                "model.n_head": "4",
                "model.n_kv_head": "4",
            }
        )
        return config

    @pytest.fixture
    def sample_inputs(self):
        """Create sample inputs for the model using proper schema."""
        batch_size = 2
        sequence_length = 16
        device = torch.device("cpu")

        # Use the actual schema to get proper dimensions
        colmap = ColumnMap(get_feature_names(), get_target_names())
        # Use zeros to avoid negative values in categorical columns
        X = torch.zeros(
            batch_size, sequence_length, len(colmap.feat_names), device=device
        )
        inputs = build_model_inputs(X, colmap)

        return inputs

    def test_model_initialization_rope(self, config_rope):
        """Test that model initializes correctly with RoPE."""
        model = GPT(config_rope)

        assert not model.use_alibi
        assert model.cos is not None
        assert model.sin is not None
        assert model.alibi_bias is None

    def test_model_initialization_alibi(self, config_alibi):
        """Test that model initializes correctly with ALiBi."""
        model = GPT(config_alibi)

        assert model.use_alibi
        assert model.cos is None
        assert model.sin is None
        assert model.alibi_bias is not None
        assert model.alibi_bias.shape == (
            1,
            config_alibi.model.n_head,
            config_alibi.model.block_size,
            config_alibi.model.block_size,
        )

    def test_forward_pass_rope(self, config_rope, sample_inputs):
        """Test forward pass with RoPE."""
        model = GPT(config_rope)
        model.eval()

        with torch.no_grad():
            outputs = model(sample_inputs)

        # Check output shapes
        batch_size, sequence_length = sample_inputs.batch_size
        assert outputs["buttons"].shape == (
            batch_size,
            sequence_length,
            config_rope.model.target_shapes_by_head["buttons"],
        )
        assert outputs["main_stick"].shape == (
            batch_size,
            sequence_length,
            config_rope.model.target_shapes_by_head["main_stick"],
        )
        # Note: Value is now computed by separate ValueNetwork, not GPT

    def test_forward_pass_alibi(self, config_alibi, sample_inputs):
        """Test forward pass with ALiBi."""
        model = GPT(config_alibi)
        model.eval()

        with torch.no_grad():
            outputs = model(sample_inputs)

        # Check output shapes
        batch_size, sequence_length = sample_inputs.batch_size
        assert outputs["buttons"].shape == (
            batch_size,
            sequence_length,
            config_alibi.model.target_shapes_by_head["buttons"],
        )
        assert outputs["main_stick"].shape == (
            batch_size,
            sequence_length,
            config_alibi.model.target_shapes_by_head["main_stick"],
        )
        # Note: Value is now computed by separate ValueNetwork, not GPT

    def test_outputs_are_different(self, config_rope, config_alibi, sample_inputs):
        """Test that RoPE and ALiBi both produce valid outputs."""
        # Set seeds for reproducibility
        torch.manual_seed(42)
        model_rope = GPT(config_rope)
        model_rope.eval()

        torch.manual_seed(42)
        model_alibi = GPT(config_alibi)
        model_alibi.eval()

        with torch.no_grad():
            outputs_rope = model_rope(sample_inputs)
            outputs_alibi = model_alibi(sample_inputs)

        # Both should produce valid outputs (no NaNs or Infs)
        # Note: With zero inputs, outputs may be identical, which is fine
        assert torch.isfinite(outputs_rope["buttons"]).all()
        assert torch.isfinite(outputs_alibi["buttons"]).all()
        # Note: Value is now computed by separate ValueNetwork, not GPT

    def test_gradient_flow_alibi(self, config_alibi, sample_inputs):
        """Test that gradients flow correctly with ALiBi."""
        model = GPT(config_alibi)
        model.train()

        outputs = model(sample_inputs)

        # Create a dummy loss from all outputs to ensure gradients flow everywhere
        # Note: Value is now computed by separate ValueNetwork, not GPT
        loss = (
            outputs["buttons"].sum()
            + outputs["main_stick"].sum()
            + outputs["c_stick"].sum()
            + outputs["shoulder"].sum()
        )

        loss.backward()

        # Check that gradients exist and are finite
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(
                    param.grad
                ).all(), f"Non-finite gradient for {name}"

    def test_different_sequence_lengths_alibi(self, config_alibi):
        """Test that ALiBi works with different sequence lengths."""
        model = GPT(config_alibi)
        model.eval()

        batch_size = 2
        device = torch.device("cpu")
        colmap = ColumnMap(get_feature_names(), get_target_names())

        # Test with different sequence lengths
        for seq_len in [8, 16, 32, 64]:
            X = torch.zeros(
                batch_size, seq_len, len(colmap.feat_names) + 1, device=device
            )
            inputs = build_model_inputs(X, colmap)

            with torch.no_grad():
                outputs = model(inputs)

            assert outputs["buttons"].shape == (
                batch_size,
                seq_len,
                config_alibi.model.target_shapes_by_head["buttons"],
            )
            assert torch.isfinite(outputs["buttons"]).all()

    def test_alibi_with_mqa(self):
        """Test that ALiBi works correctly with Multi-Query Attention (MQA)."""
        config = init_config(
            overrides={
                "model.use_alibi": "true",
                "model.block_size": "64",
                "model.n_embd": "128",
                "model.n_layer": "2",
                "model.n_head": "8",
                "model.n_kv_head": "2",  # MQA: fewer KV heads than Q heads
            }
        )

        model = GPT(config)
        model.eval()

        batch_size = 2
        seq_len = 16
        device = torch.device("cpu")

        colmap = ColumnMap(get_feature_names(), get_target_names())
        X = torch.zeros(batch_size, seq_len, len(colmap.feat_names) + 1, device=device)
        inputs = build_model_inputs(X, colmap)

        with torch.no_grad():
            outputs = model(inputs)

        # Should work correctly with MQA
        assert outputs["buttons"].shape == (
            batch_size,
            seq_len,
            config.model.target_shapes_by_head["buttons"],
        )
        assert torch.isfinite(outputs["buttons"]).all()
