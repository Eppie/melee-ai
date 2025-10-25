import torch
from torch import nn as nn
from torch.nn import functional as F

from model.norm import norm
from model.positional_encoding import apply_rotary_emb


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, n_kv_head, dropout):
        super().__init__()
        self.num_query_heads = n_head
        self.num_kv_heads = n_kv_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.dropout = dropout
        assert n_embd % n_head == 0
        assert n_kv_head <= n_head and n_head % n_kv_head == 0
        self.c_q = nn.Linear(
            n_embd, n_head * self.head_dim, bias=False
        )  # query projection
        self.c_k = nn.Linear(
            n_embd, n_kv_head * self.head_dim, bias=False
        )  # key projection
        self.c_v = nn.Linear(
            n_embd, n_kv_head * self.head_dim, bias=False
        )  # value projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)  # output projection

        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        batch_size, sequence_length, C = x.size()

        # Project the input to get queries, keys, and values
        query_states = self.c_q(x).view(
            batch_size, sequence_length, self.num_query_heads, self.head_dim
        )
        key_states = self.c_k(x).view(
            batch_size, sequence_length, self.num_kv_heads, self.head_dim
        )
        value_states = self.c_v(x).view(
            batch_size, sequence_length, self.num_kv_heads, self.head_dim
        )

        query_states = apply_rotary_emb(query_states, cos, sin)
        key_states = apply_rotary_emb(key_states, cos, sin)

        query_states = norm(query_states)
        key_states = norm(key_states)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        value_states = value_states.transpose(1, 2)

        num_repetitions = self.num_query_heads // self.num_kv_heads
        key_states = repeat_kv(key_states, num_repetitions)
        value_states = repeat_kv(value_states, num_repetitions)

        attention_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, C)
        )
        attention_output = self.residual_dropout(self.c_proj(attention_output))
        return attention_output


def repeat_kv(x, n_rep):
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    if n_rep == 1:
        return x
    bs, n_kv_heads, slen, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )
