import math
from typing import Optional

import torch
from torch import nn as nn

from config import get_config
from model.norm import QKNorm
from model.positional_encoding import _ntk_scaled_theta, _rope_cache, apply_rope_inplace, _alibi_bias


class CausalSelfAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        cfg = get_config().model
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = self.n_embd // self.n_head

        attn_type = cfg.attention_type.lower()
        if attn_type == "mqa":
            self.n_kv_head = 1
        elif attn_type == "gqa":
            self.n_kv_head = cfg.n_kv_head or max(1, self.n_head // 4)
        else:
            self.n_kv_head = self.n_head
        assert self.n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head"
        self.n_groups = self.n_head // self.n_kv_head

        self.q_proj = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=cfg.bias)
        self.k_proj = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=cfg.bias)
        self.v_proj = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=cfg.bias)
        self.o_proj = nn.Linear(self.n_embd, self.n_embd, bias=cfg.bias)

        self.dropout = cfg.dropout
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")

        self.qk_norm = None
        if cfg.qk_norm:
            qk_type = cfg.qk_norm_type or cfg.norm_type
            self.qk_norm = QKNorm(self.head_dim, qk_type, cfg.norm_eps, cfg.norm_affine)

        self.pe_type = cfg.pe_type.lower()
        self.rope_theta = cfg.rope_theta
        self.rope_scaling = cfg.rope_scaling.lower() if cfg.rope_scaling else None
        self.rope_scaling_factor = cfg.rope_scaling_factor

        if not self.flash:
            self.register_buffer(
                "bias_mask",
                torch.tril(torch.ones(cfg.block_size, cfg.block_size)).to(torch.bool),
                persistent=False,
            )

    def forward(self, x: torch.Tensor):
        B, L, _ = x.size()
        H, D = self.n_head, self.head_dim
        q = self.q_proj(x).view(B, L, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_kv_head, D).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_kv_head, D).transpose(1, 2)

        if self.n_groups > 1:
            k = k.repeat_interleave(self.n_groups, dim=1)
            v = v.repeat_interleave(self.n_groups, dim=1)

        if self.qk_norm is not None:
            q, k = self.qk_norm(q, k)

        alibi_bias: Optional[torch.Tensor] = None
        if self.pe_type == "rope":
            theta = self.rope_theta
            if self.rope_scaling in {"ntk", "yarn"}:
                theta = _ntk_scaled_theta(theta, self.rope_scaling_factor)
            cos, sin = _rope_cache(L, D, theta, q.device)
            q, k = apply_rope_inplace(q, k, cos, sin)
        elif self.pe_type == "alibi":
            alibi_bias = _alibi_bias(B, H, L, q.device)
        else:
            raise ValueError(f"Unsupported positional encoding type '{self.pe_type}'.")

        if self.flash and alibi_bias is None:
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(D))
            if alibi_bias is not None:
                att = att + alibi_bias
            else:
                if not hasattr(self, "bias_mask"):
                    raise RuntimeError("Causal mask buffer missing for non-flash attention path.")
                mask = self.bias_mask[:L, :L].unsqueeze(0).unsqueeze(0)
                att = att.masked_fill(~mask, float("-inf"))
            att = att.softmax(dim=-1)
            if self.dropout > 0:
                att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, L, self.n_embd)
        y = self.resid_dropout(self.o_proj(y))
        return y
