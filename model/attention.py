from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn as nn
from torch.nn import functional as F

from model.norm import norm
from model.positional_encoding import apply_rotary_emb


class CausalSelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_key_value_heads, dropout):
        super().__init__()
        self.num_query_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.embedding_dim = embedding_dim
        self.head_dim = embedding_dim // num_heads
        self.dropout = dropout
        assert embedding_dim % num_heads == 0
        assert num_key_value_heads <= num_heads and num_heads % num_key_value_heads == 0

        # Query, key, and value projections
        self.query_projection = nn.Linear(
            embedding_dim, num_heads * self.head_dim, bias=False
        )
        self.key_projection = nn.Linear(
            embedding_dim, num_key_value_heads * self.head_dim, bias=False
        )
        self.value_projection = nn.Linear(
            embedding_dim, num_key_value_heads * self.head_dim, bias=False
        )
        self.output_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.residual_dropout = nn.Dropout(dropout)

    def _prepare_qkv(
        self, hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare query, key, value tensors with RoPE and normalization."""
        batch_size, sequence_length, channels = hidden_states.size()

        # Project the input to get queries, keys, and values
        query_states = self.query_projection(hidden_states).view(
            batch_size, sequence_length, self.num_query_heads, self.head_dim
        )
        key_states = self.key_projection(hidden_states).view(
            batch_size, sequence_length, self.num_key_value_heads, self.head_dim
        )
        value_states = self.value_projection(hidden_states).view(
            batch_size, sequence_length, self.num_key_value_heads, self.head_dim
        )

        # Apply rotary positional embeddings
        query_states = apply_rotary_emb(query_states, cos, sin)
        key_states = apply_rotary_emb(key_states, cos, sin)

        # Normalize queries and keys
        query_states = norm(query_states)
        key_states = norm(key_states)

        # Transpose to (batch_size, num_heads, sequence_length, head_dim)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Repeat key-value heads to match query heads for grouped-query attention
        num_repetitions = self.num_query_heads // self.num_key_value_heads
        key_states = repeat_key_value_heads(key_states, num_repetitions)
        value_states = repeat_key_value_heads(value_states, num_repetitions)

        return query_states, key_states, value_states

    def forward(
        self, hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        batch_size, sequence_length, channels = hidden_states.size()

        query_states, key_states, value_states = self._prepare_qkv(
            hidden_states, cos, sin
        )

        attention_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        # Reshape back to (batch_size, sequence_length, embedding_dim)
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, channels)
        )
        attention_output = self.residual_dropout(
            self.output_projection(attention_output)
        )
        return attention_output

    def forward_with_attention_weights(
        self, hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that also returns attention weights for interpretability.

        This is slower than the standard forward() since it manually computes
        attention weights instead of using the fused SDPA kernel.

        Args:
            hidden_states: [batch, seq_len, embedding_dim]
            cos, sin: Rotary embedding components

        Returns:
            attention_output: [batch, seq_len, embedding_dim]
            attention_weights: [batch, num_heads, seq_len, seq_len]
        """
        batch_size, sequence_length, channels = hidden_states.size()

        query_states, key_states, value_states = self._prepare_qkv(
            hidden_states, cos, sin
        )

        # Compute attention scores manually
        # query_states, key_states: [batch, num_heads, seq_len, head_dim]
        scale = 1.0 / math.sqrt(self.head_dim)
        attention_scores = torch.matmul(
            query_states, key_states.transpose(-2, -1)
        ) * scale
        # attention_scores: [batch, num_heads, seq_len, seq_len]

        # Apply causal mask (lower triangular)
        causal_mask = torch.triu(
            torch.ones(sequence_length, sequence_length, device=hidden_states.device),
            diagonal=1,
        ).bool()
        attention_scores = attention_scores.masked_fill(causal_mask, float("-inf"))

        # Softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply dropout during training
        if self.training and self.dropout > 0:
            attention_weights = F.dropout(attention_weights, p=self.dropout)

        # Apply attention to values
        attention_output = torch.matmul(attention_weights, value_states)

        # Reshape back to (batch_size, sequence_length, embedding_dim)
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, channels)
        )
        attention_output = self.residual_dropout(
            self.output_projection(attention_output)
        )

        return attention_output, attention_weights

    def get_attention_weights(
        self, hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute only attention weights without the full forward pass.

        Useful for visualization when you don't need the output.

        Returns:
            attention_weights: [batch, num_heads, seq_len, seq_len]
        """
        batch_size, sequence_length, _ = hidden_states.size()

        query_states, key_states, _ = self._prepare_qkv(hidden_states, cos, sin)

        # Compute attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        attention_scores = torch.matmul(
            query_states, key_states.transpose(-2, -1)
        ) * scale

        # Apply causal mask
        causal_mask = torch.triu(
            torch.ones(sequence_length, sequence_length, device=hidden_states.device),
            diagonal=1,
        ).bool()
        attention_scores = attention_scores.masked_fill(causal_mask, float("-inf"))

        # Softmax
        attention_weights = F.softmax(attention_scores, dim=-1)

        return attention_weights


def repeat_key_value_heads(hidden_states, num_repetitions):
    """
    Repeats key/value heads to match the number of query heads.
    This is used for grouped-query attention where multiple query heads share the same key/value heads.
    """
    if num_repetitions == 1:
        return hidden_states

    return torch.repeat_interleave(hidden_states, dim=1, repeats=num_repetitions)
