"""Tests for the separate ValueNetwork module."""

import pytest
import torch
from tensordict import TensorDict

from config.config import Config
from model.value_network import (
    PreLNFFWResBlock,
    ResidualLSTM,
    ValueNetwork,
)


@pytest.fixture
def config():
    """Create a test config with value network enabled."""
    cfg = Config.model_validate({})
    # Ensure value network is enabled
    cfg.value_network.enabled = True
    cfg.value_network.hidden_dim = 128  # Smaller for faster tests
    cfg.value_network.ffw_expansion = 2
    cfg.value_network.lstm_num_layers = 1
    cfg.value_network.dropout = 0.0  # No dropout for deterministic tests
    return cfg


@pytest.fixture
def mock_inputs(config):
    """Create mock input TensorDict matching expected structure."""
    batch_size = 2
    seq_len = 16

    # Get dimensions from config to ensure consistency
    num_stages = config.model.num_stages
    num_characters = config.model.num_characters
    num_actions = config.model.num_actions

    # Get actual continuous feature dimensions from schema
    from config.gpt_config import _schema_feature_dims

    gamestate_dim, controller_dim = _schema_feature_dims()

    return TensorDict(
        {
            "stage": torch.randint(0, num_stages, (batch_size, seq_len, 1)),
            "ego_character": torch.randint(0, num_characters, (batch_size, seq_len, 1)),
            "opponent_character": torch.randint(
                0, num_characters, (batch_size, seq_len, 1)
            ),
            "ego_action": torch.randint(0, num_actions, (batch_size, seq_len, 1)),
            "opponent_action": torch.randint(0, num_actions, (batch_size, seq_len, 1)),
            "gamestate": torch.randn(batch_size, seq_len, gamestate_dim),
            "controller": torch.randn(batch_size, seq_len, controller_dim),
        },
        batch_size=(batch_size, seq_len),
    )


class TestResidualLSTM:
    """Tests for the ResidualLSTM module."""

    def test_forward_shape(self):
        """Test that output has same shape as input."""
        hidden_dim = 64
        lstm = ResidualLSTM(hidden_dim=hidden_dim)

        batch_size, seq_len = 4, 32
        x = torch.randn(batch_size, seq_len, hidden_dim)

        output, hidden = lstm(x)

        assert output.shape == x.shape
        assert hidden[0].shape == (1, batch_size, hidden_dim)  # h
        assert hidden[1].shape == (1, batch_size, hidden_dim)  # c

    def test_residual_connection(self):
        """Test that output is different from input (LSTM contributes)."""
        hidden_dim = 64
        lstm = ResidualLSTM(hidden_dim=hidden_dim)

        x = torch.randn(2, 10, hidden_dim)
        output, _ = lstm(x)

        # Output should be different from input (LSTM adds something)
        assert not torch.allclose(x, output)

    def test_with_initial_hidden(self):
        """Test that providing initial hidden state works."""
        hidden_dim = 64
        lstm = ResidualLSTM(hidden_dim=hidden_dim)

        batch_size, seq_len = 4, 16
        x = torch.randn(batch_size, seq_len, hidden_dim)

        # Create initial hidden state
        h_0 = torch.randn(1, batch_size, hidden_dim)
        c_0 = torch.randn(1, batch_size, hidden_dim)

        output, (h_n, c_n) = lstm(x, (h_0, c_0))

        assert output.shape == x.shape
        assert h_n.shape == h_0.shape
        assert c_n.shape == c_0.shape

    def test_get_initial_state(self):
        """Test creation of initial hidden state."""
        hidden_dim = 64
        lstm = ResidualLSTM(hidden_dim=hidden_dim, num_layers=2)

        batch_size = 4
        device = torch.device("cpu")

        h_0, c_0 = lstm.get_initial_state(batch_size, device)

        assert h_0.shape == (2, batch_size, hidden_dim)
        assert c_0.shape == (2, batch_size, hidden_dim)
        assert torch.all(h_0 == 0)
        assert torch.all(c_0 == 0)


class TestPreLNFFWResBlock:
    """Tests for the PreLNFFWResBlock module."""

    def test_forward_shape(self):
        """Test that output has same shape as input."""
        hidden_dim = 64
        block = PreLNFFWResBlock(hidden_dim=hidden_dim, expansion=2)

        x = torch.randn(2, 10, hidden_dim)
        output = block(x)

        assert output.shape == x.shape

    def test_zero_init_starts_as_identity(self):
        """Test that fc2 is zero-initialized so block starts as identity."""
        hidden_dim = 64
        block = PreLNFFWResBlock(hidden_dim=hidden_dim)

        # Check that fc2 weights and bias are zero
        assert torch.all(block.fc2.weight == 0)
        assert torch.all(block.fc2.bias == 0)

        # This means the block should start as (approximately) identity
        # (LayerNorm changes the values slightly, but the fc2 output is zero)
        x = torch.randn(2, 10, hidden_dim)
        output = block(x)

        # Output should be very close to input
        # (not exactly equal due to LayerNorm normalization + fc1 contribution,
        # but fc2(0) = 0 + bias(0) = 0, so residual is just input)
        # Actually, due to zero-init of fc2, the FFW output is 0, so output = x
        assert torch.allclose(x, output, atol=1e-5)

    def test_gradient_flow(self):
        """Test that gradients flow through the block."""
        hidden_dim = 64
        block = PreLNFFWResBlock(hidden_dim=hidden_dim)

        x = torch.randn(2, 10, hidden_dim, requires_grad=True)
        output = block(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        for name, param in block.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


class TestValueNetwork:
    """Tests for the ValueNetwork module."""

    def test_forward_shape(self, config, mock_inputs):
        """Test that value network produces correct output shape."""
        vn = ValueNetwork(config)

        values, hidden = vn(mock_inputs)

        batch_size, seq_len = mock_inputs["gamestate"].shape[:2]
        assert values.shape == (batch_size, seq_len, 1)
        assert hidden[0].shape == (1, batch_size, config.value_network.hidden_dim)
        assert hidden[1].shape == (1, batch_size, config.value_network.hidden_dim)

    def test_gradient_flow(self, config, mock_inputs):
        """Test that gradients flow through all components."""
        vn = ValueNetwork(config)

        values, _ = vn(mock_inputs)
        loss = values.sum()
        loss.backward()

        # Check all parameters have gradients
        for name, param in vn.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_stateful_inference(self, config, mock_inputs):
        """Test that providing hidden state works for stateful inference."""
        vn = ValueNetwork(config)
        batch_size = mock_inputs["gamestate"].shape[0]

        # First step: get initial hidden state
        initial_hidden = vn.get_initial_state(
            batch_size, torch.device("cpu"), torch.float32
        )

        # Process with initial hidden
        values1, hidden1 = vn(mock_inputs, hidden=initial_hidden)

        # Process again with updated hidden (simulating next timestep)
        values2, hidden2 = vn(mock_inputs, hidden=hidden1)

        assert values1.shape == values2.shape
        # Hidden states should be different after processing
        assert not torch.allclose(hidden1[0], hidden2[0])

    def test_encoder_initialization(self, config):
        """Test that encoder is properly initialized."""
        vn = ValueNetwork(config)

        # Encoder should be Xavier initialized
        # Just check it's not all zeros or ones
        assert not torch.all(vn.encoder.weight == 0)
        assert torch.all(vn.encoder.bias == 0)

    def test_value_head_initialization(self, config):
        """Test that value head is properly initialized."""
        vn = ValueNetwork(config)

        # Value head should be Xavier initialized
        assert not torch.all(vn.value_head.weight == 0)
        assert torch.all(vn.value_head.bias == 0)

    def test_ffw_block_starts_as_identity(self, config):
        """Test that FFW block starts as identity."""
        vn = ValueNetwork(config)

        # fc2 of FFW block should be zero-initialized
        assert torch.all(vn.ffw_block.fc2.weight == 0)
        assert torch.all(vn.ffw_block.fc2.bias == 0)

    def test_different_hidden_dims(self):
        """Test value network with different hidden dimensions."""
        for hidden_dim in [64, 128, 256, 512]:
            cfg = Config.model_validate({})
            cfg.value_network.hidden_dim = hidden_dim

            vn = ValueNetwork(cfg)

            assert vn.hidden_dim == hidden_dim
            assert vn.encoder.out_features == hidden_dim
            assert vn.value_head.in_features == hidden_dim

    def test_different_ffw_expansion(self):
        """Test value network with different FFW expansion factors."""
        for expansion in [1, 2, 4]:
            cfg = Config.model_validate({})
            cfg.value_network.hidden_dim = 64
            cfg.value_network.ffw_expansion = expansion

            vn = ValueNetwork(cfg)

            assert vn.ffw_block.fc1.out_features == 64 * expansion
            assert vn.ffw_block.fc2.in_features == 64 * expansion


class TestValueNetworkIntegration:
    """Integration tests for value network with config system."""

    def test_config_defaults(self):
        """Test that default config values are sensible."""
        cfg = Config.model_validate({})

        assert cfg.value_network.enabled is True
        assert cfg.value_network.hidden_dim == 512
        assert cfg.value_network.ffw_expansion == 2
        assert cfg.value_network.lstm_num_layers == 1
        assert cfg.value_network.dropout == 0.1
        assert cfg.value_network.grad_clip == 1.0

    def test_gamma_default(self):
        """Test that gamma default is updated to 0.9971."""
        cfg = Config.model_validate({})

        assert cfg.reward.gamma == 0.9971
