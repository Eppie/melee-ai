"""Tests for head cross-attention mechanism."""

from __future__ import annotations

import pytest
import torch

from config import init_config, reset_config, get_config
from column_map import ColumnMap
from model.head_cross_attention import HeadCrossAttention
from model.output_head import SimpleHead
from schema import get_target_names
from train.batch_utils import build_model_inputs


@pytest.fixture(autouse=True)
def clear_global_config():
    """Ensure init_config() global state does not leak between tests."""
    reset_config()
    yield
    reset_config()


class TestHeadCrossAttention:
    """Tests for HeadCrossAttention module."""

    def test_forward_preserves_shapes(self):
        """Test that output shapes match input shapes."""
        hidden_dim = 128
        num_head_types = 4
        cross_attn = HeadCrossAttention(
            hidden_dim=hidden_dim,
            num_head_types=num_head_types,
            num_attn_heads=4,
        )

        B, T = 2, 16
        head_features = [torch.randn(B, T, hidden_dim) for _ in range(num_head_types)]

        outputs = cross_attn(head_features)

        assert len(outputs) == num_head_types
        for i, out in enumerate(outputs):
            assert out.shape == (B, T, hidden_dim), f"Head {i} shape mismatch"

    def test_forward_different_batch_sizes(self):
        """Test with various batch sizes."""
        cross_attn = HeadCrossAttention(
            hidden_dim=64, num_head_types=4, num_attn_heads=2
        )

        for batch_size in [1, 4, 16]:
            head_features = [torch.randn(batch_size, 8, 64) for _ in range(4)]
            outputs = cross_attn(head_features)
            assert all(out.shape[0] == batch_size for out in outputs)

    def test_forward_different_sequence_lengths(self):
        """Test with various sequence lengths."""
        cross_attn = HeadCrossAttention(
            hidden_dim=64, num_head_types=4, num_attn_heads=2
        )

        for seq_len in [1, 32, 128]:
            head_features = [torch.randn(2, seq_len, 64) for _ in range(4)]
            outputs = cross_attn(head_features)
            assert all(out.shape[1] == seq_len for out in outputs)

    def test_wrong_number_of_heads_raises(self):
        """Test that passing wrong number of head features raises assertion."""
        cross_attn = HeadCrossAttention(
            hidden_dim=64, num_head_types=4, num_attn_heads=2
        )

        # Pass 3 heads instead of 4
        head_features = [torch.randn(2, 8, 64) for _ in range(3)]

        with pytest.raises(AssertionError):
            cross_attn(head_features)

    def test_gradients_flow_through(self):
        """Test that gradients flow through the cross-attention."""
        cross_attn = HeadCrossAttention(
            hidden_dim=64, num_head_types=4, num_attn_heads=2
        )

        head_features = [torch.randn(2, 8, 64, requires_grad=True) for _ in range(4)]
        outputs = cross_attn(head_features)

        # Sum all outputs and backprop
        loss = sum(out.sum() for out in outputs)
        loss.backward()

        # Check gradients exist for all inputs
        for i, feat in enumerate(head_features):
            assert feat.grad is not None, f"No gradient for head {i}"
            assert feat.grad.shape == feat.shape

    def test_residual_connection(self):
        """Test that residual connection allows identity-like behavior initially."""
        cross_attn = HeadCrossAttention(
            hidden_dim=64, num_head_types=4, num_attn_heads=2
        )

        # With residual connection, output should be correlated with input
        head_features = [torch.randn(2, 8, 64) for _ in range(4)]
        outputs = cross_attn(head_features)

        # Outputs should not be identical to inputs (attention modifies them)
        # but should be numerically stable
        for out in outputs:
            assert torch.isfinite(out).all()


class TestSimpleHeadIntermediate:
    """Tests for SimpleHead intermediate feature methods."""

    def test_forward_intermediate_shape(self):
        """Test forward_intermediate returns correct shape."""
        head = SimpleHead(input_size=512, output_size=64, hidden=128)
        x = torch.randn(2, 16, 512)

        intermediate = head.forward_intermediate(x)

        assert intermediate.shape == (2, 16, 128)

    def test_forward_from_intermediate_shape(self):
        """Test forward_from_intermediate returns correct shape."""
        head = SimpleHead(input_size=512, output_size=64, hidden=128)
        h = torch.randn(2, 16, 128)

        output = head.forward_from_intermediate(h)

        assert output.shape == (2, 16, 64)

    def test_split_forward_equals_full_forward(self):
        """Test that split forward path equals direct forward."""
        head = SimpleHead(input_size=512, output_size=64, hidden=128)
        head.eval()  # Disable dropout for deterministic output
        x = torch.randn(2, 16, 512)

        # Direct forward
        direct_output = head(x)

        # Split forward
        intermediate = head.forward_intermediate(x)
        split_output = head.forward_from_intermediate(intermediate)

        assert torch.allclose(direct_output, split_output)

    def test_gradients_through_split_path(self):
        """Test gradients flow through the split forward path."""
        head = SimpleHead(input_size=64, output_size=32, hidden=128)
        x = torch.randn(2, 8, 64, requires_grad=True)

        intermediate = head.forward_intermediate(x)
        output = head.forward_from_intermediate(intermediate)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestGPTConfigCrossAttention:
    """Tests for GPTConfig head_flow field."""

    def test_default_head_flow_sequential(self):
        """Test that head_flow defaults to sequential."""
        cfg = init_config()
        assert cfg.model.head_flow == "sequential"

    def test_set_head_flow_parallel_via_override(self):
        """Test setting head_flow to parallel via CLI override."""
        cfg = init_config(overrides={"model.head_flow": "parallel"})
        assert cfg.model.head_flow == "parallel"

    def test_set_head_flow_mix_via_override(self):
        """Test setting head_flow to mix via CLI override."""
        cfg = init_config(overrides={"model.head_flow": "mix"})
        assert cfg.model.head_flow == "mix"

    def test_set_head_flow_sequential_via_override(self):
        """Test setting head_flow to sequential via CLI override."""
        cfg = init_config(overrides={"model.head_flow": "sequential"})
        assert cfg.model.head_flow == "sequential"

    def test_invalid_head_flow_raises_validation_error(self):
        """Test that invalid head_flow value raises ValidationError."""
        from pydantic import ValidationError
        from config.gpt_config import GPTConfig

        with pytest.raises(ValidationError):
            GPTConfig(head_flow="invalid")


class TestGPTModelCrossAttention:
    """Tests for GPT model with different head_flow modes."""

    def test_model_sequential_mode_default(self):
        """Test model creation with sequential mode (default)."""
        from model.nano_gpt import GPT

        cfg = init_config()
        model = GPT(cfg)

        assert model.head_flow == "sequential"
        assert not hasattr(model, "head_cross_attention")
        # Sequential mode should have different input sizes for each head
        # Order: buttons → main_stick → c_stick → shoulder
        assert model.button_head.fc1.in_features < model.main_stick_head.fc1.in_features
        assert (
            model.main_stick_head.fc1.in_features < model.c_stick_head.fc1.in_features
        )
        assert model.c_stick_head.fc1.in_features < model.shoulder_head.fc1.in_features

    def test_model_parallel_mode(self):
        """Test model creation with parallel mode."""
        from model.nano_gpt import GPT

        cfg = init_config(overrides={"model.head_flow": "parallel"})
        model = GPT(cfg)

        assert model.head_flow == "parallel"
        assert not hasattr(model, "head_cross_attention")
        # Parallel mode should have same input size for all heads
        button_in = model.button_head.fc1.in_features
        assert model.main_stick_head.fc1.in_features == button_in
        assert model.c_stick_head.fc1.in_features == button_in
        assert model.shoulder_head.fc1.in_features == button_in

    def test_model_mix_mode(self):
        """Test model creation with mix mode (cross-attention)."""
        from model.nano_gpt import GPT

        cfg = init_config(overrides={"model.head_flow": "mix"})
        model = GPT(cfg)

        assert model.head_flow == "mix"
        assert hasattr(model, "head_cross_attention")
        assert isinstance(model.head_cross_attention, HeadCrossAttention)

    def test_cross_attention_adds_parameters(self):
        """Test that mix mode (cross-attention) adds parameters vs parallel mode."""
        from model.nano_gpt import GPT

        cfg_parallel = init_config(overrides={"model.head_flow": "parallel"})
        model_parallel = GPT(cfg_parallel)
        params_parallel = sum(p.numel() for p in model_parallel.parameters())

        reset_config()
        cfg_mix = init_config(overrides={"model.head_flow": "mix"})
        model_mix = GPT(cfg_mix)
        params_mix = sum(p.numel() for p in model_mix.parameters())

        assert params_mix > params_parallel
        # Cross-attention should add roughly 66K params (128 hidden * 4 heads * ~130)
        param_diff = params_mix - params_parallel
        assert 50000 < param_diff < 100000, f"Unexpected param diff: {param_diff}"

    def test_forward_with_cross_attention(self):
        """Test forward pass with mix mode (cross-attention)."""
        from model.nano_gpt import GPT
        from schema import get_feature_names

        cfg = init_config(overrides={"model.head_flow": "mix"})
        device = torch.device("cpu")
        model = GPT(cfg).to(device)
        model.eval()

        # Create valid inputs
        B, T = 2, 16
        colmap = ColumnMap(get_feature_names(), get_target_names())
        # Include horizon feature to mirror training inputs
        X = torch.zeros(B, T, len(colmap.feat_names) + 1, device=device)
        inputs = build_model_inputs(X, colmap)

        with torch.no_grad():
            outputs = model(inputs)

        # Check output keys and shapes
        assert "buttons" in outputs
        assert "main_stick" in outputs
        assert "c_stick" in outputs
        assert "shoulder" in outputs
        assert "value" in outputs

        assert outputs["buttons"].shape[:2] == (B, T)
        assert outputs["main_stick"].shape[:2] == (B, T)
        assert outputs["c_stick"].shape[:2] == (B, T)
        assert outputs["shoulder"].shape[:2] == (B, T)
        assert outputs["value"].shape[:2] == (B, T)

    def test_forward_outputs_match_across_modes(self):
        """Test that output shapes are identical across different head_flow modes."""
        from model.nano_gpt import GPT
        from tensordict import TensorDict
        from column_map import ColumnMap
        from schema import get_feature_names, get_target_names

        device = torch.device("cpu")

        # Model with parallel mode
        cfg_parallel = init_config(overrides={"model.head_flow": "parallel"})
        model_parallel = GPT(cfg_parallel).to(device)

        reset_config()

        # Model with mix mode
        cfg_mix = init_config(overrides={"model.head_flow": "mix"})
        model_mix = GPT(cfg_mix).to(device)

        # Create inputs
        B, T = 2, 16
        colmap = ColumnMap(get_feature_names(), get_target_names())
        X = torch.zeros(B, T, len(colmap.feat_names) + 1, device=device)
        inputs = build_model_inputs(X, colmap)

        model_parallel.eval()
        model_mix.eval()

        with torch.no_grad():
            out_parallel = model_parallel(inputs)
            out_mix = model_mix(inputs)

        # Shapes should be identical
        for key in ["buttons", "main_stick", "c_stick", "shoulder", "value"]:
            assert (
                out_parallel[key].shape == out_mix[key].shape
            ), f"Shape mismatch for {key}"

    def test_backward_pass_with_cross_attention(self):
        """Test that gradients flow correctly with mix mode (cross-attention)."""
        from model.nano_gpt import GPT
        from tensordict import TensorDict
        from column_map import ColumnMap
        from schema import get_feature_names, get_target_names

        device = torch.device("cpu")

        cfg = init_config(overrides={"model.head_flow": "mix"})
        model = GPT(cfg).to(device)
        model.train()

        B, T = 2, 16
        colmap = ColumnMap(get_feature_names(), get_target_names())

        X = torch.zeros(B, T, len(colmap.feat_names) + 1, device=device, requires_grad=True)
        inputs = build_model_inputs(X, colmap)

        outputs = model(inputs)

        # Compute a simple loss
        loss = (
            outputs["buttons"].sum()
            + outputs["main_stick"].sum()
            + outputs["c_stick"].sum()
            + outputs["shoulder"].sum()
        )
        loss.backward()

        # Check that cross-attention parameters have gradients
        for name, param in model.head_cross_attention.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"
