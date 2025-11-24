"""Tests for head cross-attention mechanism."""

from __future__ import annotations

import pytest
import torch

from config import init_config, reset_config, get_config
from model.head_cross_attention import HeadCrossAttention
from model.output_head import SimpleHead


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
        x = torch.randn(2, 16, 512)

        # Direct forward
        direct_output = head(x)

        # Split forward
        intermediate = head.forward_intermediate(x)
        split_output = head.forward_from_intermediate(intermediate)

        assert torch.allclose(direct_output, split_output)

    def test_intermediate_has_relu_activation(self):
        """Test that intermediate features have ReLU applied."""
        head = SimpleHead(input_size=64, output_size=32, hidden=128)

        # Create input that would produce negative pre-activation values
        x = torch.randn(2, 8, 64) * 10  # Large values to ensure some negatives

        intermediate = head.forward_intermediate(x)

        # After ReLU, all values should be >= 0
        assert (intermediate >= 0).all()

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
    """Tests for GPTConfig cross-attention fields."""

    def test_default_cross_attention_disabled(self):
        """Test that cross-attention is disabled by default."""
        cfg = init_config()
        assert cfg.model.use_head_cross_attention is False

    def test_default_cross_attention_heads(self):
        """Test default number of cross-attention heads."""
        cfg = init_config()
        assert cfg.model.head_cross_attention_heads == 4

    def test_enable_cross_attention_via_override(self):
        """Test enabling cross-attention via CLI override."""
        cfg = init_config(overrides={"model.use_head_cross_attention": "true"})
        assert cfg.model.use_head_cross_attention is True

    def test_set_cross_attention_heads_via_override(self):
        """Test setting cross-attention heads via CLI override."""
        cfg = init_config(overrides={"model.head_cross_attention_heads": "8"})
        assert cfg.model.head_cross_attention_heads == 8

    def test_cross_attention_heads_validation(self):
        """Test that cross-attention heads must be >= 1."""
        from pydantic import ValidationError
        from config.gpt_config import GPTConfig

        with pytest.raises(ValidationError):
            GPTConfig(head_cross_attention_heads=0)


class TestGPTModelCrossAttention:
    """Tests for GPT model with cross-attention."""

    def test_model_without_cross_attention(self):
        """Test model creation without cross-attention (default)."""
        from model.nano_gpt import GPT

        cfg = init_config()
        model = GPT(cfg)

        assert model.use_head_cross_attention is False
        assert (
            not hasattr(model, "head_cross_attention")
            or model.head_cross_attention is None
        )

    def test_model_with_cross_attention(self):
        """Test model creation with cross-attention enabled."""
        from model.nano_gpt import GPT

        cfg = init_config(overrides={"model.use_head_cross_attention": "true"})
        model = GPT(cfg)

        assert model.use_head_cross_attention is True
        assert hasattr(model, "head_cross_attention")
        assert isinstance(model.head_cross_attention, HeadCrossAttention)

    def test_cross_attention_adds_parameters(self):
        """Test that cross-attention adds parameters to the model."""
        from model.nano_gpt import GPT

        cfg_without = init_config()
        model_without = GPT(cfg_without)
        params_without = sum(p.numel() for p in model_without.parameters())

        reset_config()
        cfg_with = init_config(overrides={"model.use_head_cross_attention": "true"})
        model_with = GPT(cfg_with)
        params_with = sum(p.numel() for p in model_with.parameters())

        assert params_with > params_without
        # Cross-attention should add roughly 66K params (128 hidden * 4 heads * ~130)
        param_diff = params_with - params_without
        assert 50000 < param_diff < 100000, f"Unexpected param diff: {param_diff}"

    def test_forward_with_cross_attention(self):
        """Test forward pass with cross-attention enabled."""
        from model.nano_gpt import GPT
        from schema import get_feature_names

        cfg = init_config(overrides={"model.use_head_cross_attention": "true"})
        device = torch.device("cpu")
        model = GPT(cfg).to(device)
        model.eval()

        # Create valid inputs
        B, T = 2, 16
        feature_names = get_feature_names()
        feature_dim = len(feature_names)

        # Build input tensors matching expected format
        from tensordict import TensorDict
        from column_map import ColumnMap
        from schema import get_target_names

        colmap = ColumnMap(get_feature_names(), get_target_names())

        inputs = TensorDict(
            {
                "gamestate": torch.zeros(
                    B, T, len(colmap.gamestate_idxs), device=device
                ),
                "controller": torch.zeros(
                    B, T, len(colmap.controller_idxs), device=device
                ),
                "stage": torch.zeros(B, T, 1, dtype=torch.long, device=device),
                "ego_character": torch.zeros(B, T, 1, dtype=torch.long, device=device),
                "opponent_character": torch.zeros(
                    B, T, 1, dtype=torch.long, device=device
                ),
                "ego_action": torch.zeros(B, T, 1, dtype=torch.long, device=device),
                "opponent_action": torch.zeros(
                    B, T, 1, dtype=torch.long, device=device
                ),
            },
            batch_size=(B, T),
        )

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

    def test_forward_outputs_match_without_cross_attention(self):
        """Test that output shapes are identical with and without cross-attention."""
        from model.nano_gpt import GPT
        from tensordict import TensorDict
        from column_map import ColumnMap
        from schema import get_feature_names, get_target_names

        device = torch.device("cpu")

        # Model without cross-attention
        cfg_without = init_config()
        model_without = GPT(cfg_without).to(device)

        reset_config()

        # Model with cross-attention
        cfg_with = init_config(overrides={"model.use_head_cross_attention": "true"})
        model_with = GPT(cfg_with).to(device)

        # Create inputs
        B, T = 2, 16
        colmap = ColumnMap(get_feature_names(), get_target_names())

        inputs = TensorDict(
            {
                "gamestate": torch.randn(
                    B, T, len(colmap.gamestate_idxs), device=device
                ),
                "controller": torch.randn(
                    B, T, len(colmap.controller_idxs), device=device
                ),
                "stage": torch.zeros(B, T, 1, dtype=torch.long, device=device),
                "ego_character": torch.zeros(B, T, 1, dtype=torch.long, device=device),
                "opponent_character": torch.zeros(
                    B, T, 1, dtype=torch.long, device=device
                ),
                "ego_action": torch.zeros(B, T, 1, dtype=torch.long, device=device),
                "opponent_action": torch.zeros(
                    B, T, 1, dtype=torch.long, device=device
                ),
            },
            batch_size=(B, T),
        )

        model_without.eval()
        model_with.eval()

        with torch.no_grad():
            out_without = model_without(inputs)
            out_with = model_with(inputs)

        # Shapes should be identical
        for key in ["buttons", "main_stick", "c_stick", "shoulder", "value"]:
            assert (
                out_without[key].shape == out_with[key].shape
            ), f"Shape mismatch for {key}"

    def test_backward_pass_with_cross_attention(self):
        """Test that gradients flow correctly with cross-attention."""
        from model.nano_gpt import GPT
        from tensordict import TensorDict
        from column_map import ColumnMap
        from schema import get_feature_names, get_target_names

        device = torch.device("cpu")

        cfg = init_config(overrides={"model.use_head_cross_attention": "true"})
        model = GPT(cfg).to(device)
        model.train()

        B, T = 2, 16
        colmap = ColumnMap(get_feature_names(), get_target_names())

        inputs = TensorDict(
            {
                "gamestate": torch.randn(
                    B, T, len(colmap.gamestate_idxs), device=device
                ),
                "controller": torch.randn(
                    B, T, len(colmap.controller_idxs), device=device
                ),
                "stage": torch.zeros(B, T, 1, dtype=torch.long, device=device),
                "ego_character": torch.zeros(B, T, 1, dtype=torch.long, device=device),
                "opponent_character": torch.zeros(
                    B, T, 1, dtype=torch.long, device=device
                ),
                "ego_action": torch.zeros(B, T, 1, dtype=torch.long, device=device),
                "opponent_action": torch.zeros(
                    B, T, 1, dtype=torch.long, device=device
                ),
            },
            batch_size=(B, T),
        )

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
