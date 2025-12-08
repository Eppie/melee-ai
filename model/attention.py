import torch
from torch import nn as nn
from torch.nn import functional as F
from typing import Optional, Tuple

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

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor = None,
        sin: torch.Tensor = None,
        alibi_bias: torch.Tensor = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with optional KV caching.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Input tensor of shape [batch_size, sequence_length, channels]
        cos : torch.Tensor, optional
            Cosine for RoPE
        sin : torch.Tensor, optional
            Sine for RoPE
        alibi_bias : torch.Tensor, optional
            ALiBi bias matrix
        kv_cache : tuple of (keys, values), optional
            Cached keys and values from previous forward passes.
            Keys shape: [batch_size, num_kv_heads, past_seq_len, head_dim]
            Values shape: [batch_size, num_kv_heads, past_seq_len, head_dim]
        use_cache : bool
            Whether to return updated cache. Only works with ALiBi (not RoPE).

        Returns
        -------
        attention_output : torch.Tensor
            Output tensor of shape [batch_size, sequence_length, channels]
        updated_cache : tuple of (keys, values) or None
            Updated cache if use_cache=True, None otherwise
        """
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

        # Apply positional encoding: either RoPE or ALiBi
        if alibi_bias is None:
            # Use RoPE (default)
            # KV cache is not supported with RoPE (would need position adjustment)
            if use_cache:
                raise ValueError(
                    "KV caching is only supported with ALiBi, not RoPE. "
                    "Set use_alibi=True in model config to enable caching."
                )

            query_states = apply_rotary_emb(query_states, cos, sin)
            key_states = apply_rotary_emb(key_states, cos, sin)
            attn_mask = None
            is_causal = True
        else:
            # Use ALiBi - no rotary embeddings needed
            # ALiBi supports KV caching naturally since biases are position-based
            pass

        # Normalize queries and keys
        query_states = norm(query_states)
        key_states = norm(key_states)

        # Transpose to (batch_size, num_heads, sequence_length, head_dim)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Handle KV cache
        updated_cache = None
        if use_cache and alibi_bias is not None:
            # Cache only works with ALiBi
            if kv_cache is not None:
                # Concatenate cached keys/values with new ones
                cached_keys, cached_values = kv_cache
                key_states = torch.cat([cached_keys, key_states], dim=2)
                value_states = torch.cat([cached_values, value_states], dim=2)

            # Store updated cache (before repeating for MQA/GQA)
            updated_cache = (key_states, value_states)

        # Get total sequence length (including cache)
        total_seq_len = key_states.shape[2]

        # Setup attention mask for ALiBi
        if alibi_bias is not None:
            # Slice alibi_bias to match query and key sequence lengths
            # For KV cache: query_len = sequence_length, key_len = total_seq_len
            attn_mask = alibi_bias[:, :, :sequence_length, :total_seq_len]

            # Apply causal mask by setting future positions to -inf
            # Create causal mask: lower triangular matrix
            causal_mask = torch.tril(
                torch.ones(sequence_length, total_seq_len, device=hidden_states.device)
            )
            # Expand to match attn_mask shape and apply
            causal_mask = causal_mask.view(1, 1, sequence_length, total_seq_len)
            attn_mask = attn_mask.masked_fill(causal_mask == 0, float('-inf'))
            is_causal = False  # We've already applied causal masking
        else:
            attn_mask = None
            is_causal = True

        # Repeat key-value heads to match query heads for grouped-query attention
        num_repetitions = self.num_query_heads // self.num_key_value_heads
        key_states = repeat_key_value_heads(key_states, num_repetitions)
        value_states = repeat_key_value_heads(value_states, num_repetitions)

        # Note: ALiBi bias is already sized for all query heads (num_query_heads),
        # so no repetition needed even with MQA/GQA

        attention_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
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
        return attention_output, updated_cache


def repeat_key_value_heads(hidden_states, num_repetitions):
    """
    Repeats key/value heads to match the number of query heads.
    This is used for grouped-query attention where multiple query heads share the same key/value heads.
    """
    if num_repetitions == 1:
        return hidden_states

    return torch.repeat_interleave(hidden_states, dim=1, repeats=num_repetitions)
