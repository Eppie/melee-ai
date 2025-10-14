"""Value head for reinforcement learning critic network.

The value head estimates the expected discounted return (value) of a state,
which is used in actor-critic algorithms like PPO and A2C.
"""

import torch
import torch.nn as nn
from typing import Optional

from config import get_config
from model.feed_forward import ActivationFFN
from model.norm import _create_norm


class ValueHead(nn.Module):
    """Value head that estimates state value for RL.
    
    This is a critic network that takes the transformer's output and predicts
    a scalar value representing the expected discounted return from that state.
    
    Architecture:
        - Optional normalization layer
        - FFN block (for expressiveness)
        - Linear projection to scalar value
    """
    
    def __init__(self, input_dim: int, hidden_mult: float = 2.0, bias: bool = True) -> None:
        """Initialize value head.
        
        Args:
            input_dim: Dimension of input features (typically n_embd from transformer)
            hidden_mult: Multiplier for hidden dimension in FFN
            bias: Whether to use bias in linear layers
        """
        super().__init__()
        
        config = get_config().model
        
        # Normalization before value head (helps stability)
        self.norm = _create_norm(input_dim)
        
        # FFN for learning nonlinear value function
        self.ffn = ActivationFFN(
            input_dim,
            mult=hidden_mult,
            activation=config.ffn_activation,
            bias=bias,
        )
        
        # Project to scalar value
        self.value_proj = nn.Linear(input_dim, 1, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to estimate state values.
        
        Args:
            x: Input tensor of shape [B, L, D] where
               B = batch size, L = sequence length, D = input_dim
        
        Returns:
            Value estimates of shape [B, L, 1]
        """
        x = self.norm(x)
        x = self.ffn(x)
        value = self.value_proj(x)
        return value


class TinyValueHead(nn.Module):
    """Lightweight value head with single hidden layer.
    
    Similar to TinyMLPHead but for value prediction. Uses a small hidden
    dimension for efficiency while maintaining expressiveness.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden: int = 128,
        activation: str = "gelu",
        bias: bool = True,
    ) -> None:
        """Initialize tiny value head.
        
        Args:
            input_dim: Dimension of input features
            hidden: Hidden layer size
            activation: Activation function name
            bias: Whether to use bias terms
        """
        super().__init__()
        
        self.net = nn.Sequential(
            _create_norm(input_dim),
            nn.Linear(input_dim, hidden, bias=bias),
            self._get_activation(activation),
            nn.Linear(hidden, 1, bias=bias),
        )
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        name = name.lower()
        if name == "relu":
            return nn.ReLU()
        elif name == "gelu":
            return nn.GELU()
        elif name == "silu" or name == "swish":
            return nn.SiLU()
        elif name == "tanh":
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {name}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [B, L, input_dim]
        
        Returns:
            Value estimates [B, L, 1]
        """
        return self.net(x)

