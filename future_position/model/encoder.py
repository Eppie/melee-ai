"""Transformer encoder for temporal modeling."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from ..constants import D_MODEL, N_LAYERS, N_HEADS, MLP_RATIO
from .embeddings import RotaryPositionalEmbedding


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    More efficient than LayerNorm, no mean centering.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x) * self.weight


class MultiHeadAttention(nn.Module):
    """Multi-head attention with RoPE support."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout_p = dropout
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryPositionalEmbedding,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.size()  # Batch, Sequence Length, Channels (d_model)

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, T, head_dim)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, T, head_dim)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, T, head_dim)

        # Apply RoPE
        q, k = rope(q, k, T)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5) # (B, n_heads, T, T)

        if mask is not None:
            # Mask should be (1, 1, T, T) or (T, T) and broadcastable
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, v) # (B, n_heads, T, head_dim)

        # Concatenate heads and project back
        output = output.transpose(1, 2).contiguous().view(B, T, C) # (B, T, d_model)
        output = self.out_proj(output)

        return output


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer.

    Structure:
    - Multi-head self-attention with RoPE
    - RMSNorm
    - MLP with ReLU
    - RMSNorm
    - Residual connections
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.norm2 = RMSNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * mlp_ratio, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryPositionalEmbedding,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention with residual connection
        x_norm = self.norm1(x)
        x = x + self.attn(x_norm, rope, mask)

        # MLP with residual connection
        x_norm = self.norm2(x)
        x = x + self.mlp(x_norm)

        return x


class TransformerEncoder(nn.Module):
    """Stack of Transformer encoder layers with RoPE.

    Uses:
    - Rotary positional embeddings
    - RMSNorm (instead of LayerNorm)
    - No bias in linear layers
    - Causal attention mask (optional)
    """

    def __init__(
        self,
        d_model: int = D_MODEL,
        n_layers: int = N_LAYERS,
        n_heads: int = N_HEADS,
        mlp_ratio: int = MLP_RATIO,
        dropout: float = 0.0,
        causal: bool = False,
    ):
        super().__init__()
        self.rope = RotaryPositionalEmbedding(dim=d_model // n_heads) # RoPE dimension is head_dim
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, mlp_ratio, dropout)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model) # Final normalization
        self.causal = causal

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through encoder.

        Args:
            x: [B, T, d_model] input sequence
            mask: Optional [T, T] attention mask (if causal is False, otherwise generated)

        Returns:
            [B, T, d_model] encoded sequence
        """
        if self.causal:
            T = x.size(1)
            # Create a causal mask (lower triangular)
            causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
            if mask is None:
                mask = causal_mask
            else:
                mask = mask & causal_mask

        for layer in self.layers:
            x = layer(x, self.rope, mask)
        
        return self.norm(x)
