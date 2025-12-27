"""Separate value network with LSTM + FFW ResBlock architecture.

This module implements a standalone value network for advantage estimation,
separate from the policy transformer. The architecture follows the reference:
- Encoder: Linear(input_size -> hidden_dim)
- Residual LSTM: output = input + LSTM(input)
- Pre-LN FFW ResBlock: LayerNorm -> FFW with zero-init output
- Value Head: Linear(hidden_dim -> 1)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict


class ResidualLSTM(nn.Module):
    """LSTM with residual connection: output = input + LSTM(input).

    The residual connection allows the network to learn incremental updates
    to the input rather than learning the full transformation.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with residual connection.

        Args:
            x: Input tensor of shape [B, L, hidden_dim]
            hidden: Optional tuple (h_0, c_0) for LSTM initial state.
                Each has shape [num_layers, B, hidden_dim].
                If None, zeros are used.

        Returns:
            output: Tensor of shape [B, L, hidden_dim] with residual connection
            hidden_new: Updated tuple (h_n, c_n) for stateful inference
        """
        lstm_out, hidden_new = self.lstm(x, hidden)
        # Residual connection: output = input + lstm_output
        return x + lstm_out, hidden_new

    def get_initial_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create zero-initialized hidden state for the LSTM.

        Args:
            batch_size: Batch size for the hidden state
            device: Device to create tensors on
            dtype: Data type for tensors

        Returns:
            Tuple (h_0, c_0) each of shape [num_layers, batch_size, hidden_dim]
        """
        h_0 = torch.zeros(
            self.num_layers, batch_size, self.hidden_dim, device=device, dtype=dtype
        )
        c_0 = torch.zeros(
            self.num_layers, batch_size, self.hidden_dim, device=device, dtype=dtype
        )
        return h_0, c_0


class PreLNFFWResBlock(nn.Module):
    """Pre-LN FFW ResBlock with zero-initialized output projection.

    Architecture:
        output = input + Linear2(GELU(Linear1(LayerNorm(input))))

    The second linear layer (fc2) is zero-initialized so the block starts
    as an identity function. This allows the network to "grow" complexity
    during training.
    """

    def __init__(
        self,
        hidden_dim: int,
        expansion: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        intermediate_dim = hidden_dim * expansion

        # Pre-LN: LayerNorm before FFW layers
        self.norm = nn.LayerNorm(hidden_dim)

        # FFW layers
        self.fc1 = nn.Linear(hidden_dim, intermediate_dim, bias=True)
        self.fc2 = nn.Linear(intermediate_dim, hidden_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

        # Zero-initialize fc2 so residual block starts as identity
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with Pre-LN and residual connection.

        Args:
            x: Input tensor of shape [B, L, hidden_dim]

        Returns:
            Output tensor of shape [B, L, hidden_dim]
        """
        # Pre-LN: normalize before FFW
        h = self.norm(x)

        # FFW: Linear -> GELU -> Dropout -> Linear
        h = self.fc1(h)
        h = F.gelu(h)
        h = self.dropout(h)
        h = self.fc2(h)

        # Residual connection
        return x + h


class ValueNetwork(nn.Module):
    """Separate value network for advantage estimation.

    This network is independent from the policy transformer and has its own
    encoder. It uses an LSTM for temporal modeling and outputs scalar values
    for each timestep.

    Architecture:
        1. Encoder: Linear(input_size -> hidden_dim)
        2. Residual LSTM: output = input + LSTM(input)
        3. Pre-LN FFW ResBlock: LayerNorm + FFW with zero-init
        4. Value Head: Linear(hidden_dim -> 1)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        vn_config = config.value_network
        model_config = config.model

        # Store model config for _embed_inputs
        self.num_stages = model_config.num_stages
        self.num_characters = model_config.num_characters
        self.num_actions = model_config.num_actions
        self.hidden_dim = vn_config.hidden_dim

        # Compute input size for one-hot encoding (always used by value network)
        # This is independent of the policy model's use_learned_embeddings setting
        from config.gpt_config import _schema_feature_dims

        gamestate_dim, controller_dim = _schema_feature_dims()
        categorical_size = (
            self.num_stages + self.num_characters * 2 + self.num_actions * 2
        )
        self.input_size = categorical_size + gamestate_dim + controller_dim

        # 1. Encoder projection (independent from transformer)
        self.encoder = nn.Linear(self.input_size, self.hidden_dim, bias=True)

        # 2. Residual LSTM
        self.residual_lstm = ResidualLSTM(
            hidden_dim=self.hidden_dim,
            num_layers=vn_config.lstm_num_layers,
            dropout=vn_config.dropout,
        )

        # 3. Pre-LN FFW ResBlock
        self.ffw_block = PreLNFFWResBlock(
            hidden_dim=self.hidden_dim,
            expansion=vn_config.ffw_expansion,
            dropout=vn_config.dropout,
        )

        # 4. Value head
        self.value_head = nn.Linear(self.hidden_dim, 1, bias=True)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights following best practices."""
        # Xavier/Glorot initialization for encoder and value head
        nn.init.xavier_uniform_(self.encoder.weight)
        if self.encoder.bias is not None:
            nn.init.zeros_(self.encoder.bias)

        nn.init.xavier_uniform_(self.value_head.weight)
        if self.value_head.bias is not None:
            nn.init.zeros_(self.value_head.bias)

        # LSTM uses default PyTorch initialization (orthogonal for recurrent weights)
        # FFW ResBlock fc2 is already zero-initialized in __init__

    def _embed_inputs(self, inputs: TensorDict) -> torch.Tensor:
        """Convert TensorDict inputs to flat feature tensor.

        Uses one-hot encoding for categorical features, matching the GPT model's
        input processing when use_learned_embeddings=False.

        Args:
            inputs: TensorDict with keys:
                - stage: [B, L, 1] int tensor
                - ego_character: [B, L, 1] int tensor
                - opponent_character: [B, L, 1] int tensor
                - ego_action: [B, L, 1] int tensor
                - opponent_action: [B, L, 1] int tensor
                - gamestate: [B, L, gamestate_dim] float tensor
                - controller: [B, L, controller_dim] float tensor

        Returns:
            Flat feature tensor of shape [B, L, input_size]
        """
        # One-hot encoding for categorical features
        categorical_features = [
            F.one_hot(
                inputs["stage"].squeeze(-1).long(),
                num_classes=self.num_stages,
            ).float(),
            F.one_hot(
                inputs["ego_character"].squeeze(-1).long(),
                num_classes=self.num_characters,
            ).float(),
            F.one_hot(
                inputs["opponent_character"].squeeze(-1).long(),
                num_classes=self.num_characters,
            ).float(),
            F.one_hot(
                inputs["ego_action"].squeeze(-1).long(),
                num_classes=self.num_actions,
            ).float(),
            F.one_hot(
                inputs["opponent_action"].squeeze(-1).long(),
                num_classes=self.num_actions,
            ).float(),
        ]

        return torch.cat(
            categorical_features
            + [
                inputs["gamestate"],
                inputs["controller"],
            ],
            dim=-1,
        )

    def forward(
        self,
        inputs: TensorDict,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the value network.

        Args:
            inputs: TensorDict with gamestate and controller features
            hidden: Optional LSTM hidden state (h, c) for stateful inference.
                If None, zeros are used (appropriate for training).

        Returns:
            values: [B, L, 1] value predictions per timestep
            hidden_new: Updated LSTM hidden state for stateful inference
        """
        # Embed inputs to flat features
        x = self._embed_inputs(inputs)  # [B, L, input_size]

        # 1. Encoder
        h = self.encoder(x)  # [B, L, hidden_dim]

        # 2. Residual LSTM
        h, hidden_new = self.residual_lstm(h, hidden)  # [B, L, hidden_dim]

        # 3. FFW ResBlock
        h = self.ffw_block(h)  # [B, L, hidden_dim]

        # 4. Value head
        values = self.value_head(h)  # [B, L, 1]

        return values, hidden_new

    def get_initial_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create zero-initialized hidden state for stateful inference.

        Args:
            batch_size: Batch size for the hidden state
            device: Device to create tensors on
            dtype: Data type for tensors

        Returns:
            Tuple (h_0, c_0) for LSTM initialization
        """
        return self.residual_lstm.get_initial_state(batch_size, device, dtype)
