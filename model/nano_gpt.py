"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Multi-Query Attention (MQA) support for more efficient inference
"""

import math
from re import S
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from torch.distributions import Bernoulli, Categorical

def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last time into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3) # re-assemble
    out = out.to(x.dtype) # ensure input/output dtypes match
    return out


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


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, n_kv_head, dropout, bias = False):
        super().__init__()
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.dropout = dropout
        assert n_embd % n_head == 0
        assert n_kv_head <= n_head and n_head % n_kv_head == 0
        self.c_q = nn.Linear(n_embd, n_head * self.head_dim, bias=bias)
        self.c_k = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=bias)
        self.c_v = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)


        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        q = norm(q)
        k = norm(k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        nrep = self.n_head // self.n_kv_head
        k = repeat_kv(k, nrep)
        v = repeat_kv(v, nrep)

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, n_embd, bias = False):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, n_embd, n_head, n_kv_head, dropout, bias = False):
        super().__init__()
        self.attn = CausalSelfAttention(n_embd, n_head, n_kv_head, dropout, bias)
        self.mlp = MLP(n_embd, bias)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(norm(x), cos, sin)
        x = x + self.mlp(norm(x))
        return x

class SimpleHead(nn.Module):
    def __init__(self, input_size, output_size, hidden = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden, bias=False),
            nn.ReLU(),
            nn.Linear(hidden, output_size, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ButtonHead(nn.Module):
    def __init__(self, input_size, output_size, hidden = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden, bias=False),
            nn.ReLU(),
            nn.Linear(hidden, output_size, bias=False),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.net(norm(x))
        probs = torch.sigmoid(logits)
        return logits, probs


class ValueHead(nn.Module):
    def __init__(self, input_dim, hidden = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden, bias=False),
            nn.ReLU(),
            nn.Linear(hidden, 1, bias=False),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(norm(x))



class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        cfg = config.model
        self.block_size = cfg.block_size
        self.n_embd: int = cfg.n_embd
        self.input_size: int = cfg.input_size

        self.stage_emb = nn.Embedding(cfg.num_stages, cfg.stage_embedding_dim)
        self.character_emb = nn.Embedding(cfg.num_characters, cfg.character_embedding_dim)
        self.action_emb = nn.Embedding(cfg.num_actions, cfg.action_embedding_dim)

        self.proj_down = nn.Linear(self.input_size, self.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
        
        self.blocks = nn.ModuleList([
            Block(self.n_embd, cfg.n_head, cfg.n_kv_head, cfg.dropout, bias=False)
            for _ in range(cfg.n_layer)
        ])

        self.target_shapes_by_head = cfg.target_shapes_by_head
        self.shoulder_output_size = self.target_shapes_by_head["shoulder"]
        self.c_stick_output_size = self.target_shapes_by_head["c_stick"]
        self.main_stick_output_size = self.target_shapes_by_head["main_stick"]
        self.button_output_size = self.target_shapes_by_head["buttons"]


        head_hidden_dim = 128

        self.button_head = ButtonHead(self.n_embd, self.button_output_size, hidden=head_hidden_dim)

        main_stick_input_size = self.n_embd + self.button_output_size
        self.main_stick_head = SimpleHead(main_stick_input_size, self.main_stick_output_size, hidden=head_hidden_dim)

        c_stick_input_size = self.n_embd + self.button_output_size + self.main_stick_output_size
        self.c_stick_head = SimpleHead(c_stick_input_size, self.c_stick_output_size, hidden=head_hidden_dim)

        shoulder_input_size = self.n_embd + self.button_output_size + self.main_stick_output_size + self.c_stick_output_size
        self.shoulder_head = SimpleHead(shoulder_input_size, self.shoulder_output_size, hidden=head_hidden_dim)
        self.value_head = ValueHead(self.n_embd, hidden=head_hidden_dim)


        self.rotary_seq_len = self.block_size * 2
        head_dim = cfg.n_embd // cfg.n_head
        rope_base = getattr(cfg, "rope_theta", 10000.0)
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim, rope_base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.apply(self._init_weights)

        for block in self.blocks:
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.zeros_(block.attn.c_proj.weight)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # https://arxiv.org/pdf/2310.17813
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)


    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # autodetect the device from model embeddings
        
        if device is None:
            device = self.stage_emb.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16() # keep them in bfloat16
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def _embed_inputs(self, inputs: TensorDict) -> torch.Tensor:
        """Includes categorical embeddings, one-hot encodings, and numerical features."""
        return torch.cat(
            [
                F.one_hot(inputs["stage"].squeeze(-1).long(), num_classes=self.config.model.num_stages).float(),
                F.one_hot(inputs["ego_character"].squeeze(-1).long(), num_classes=self.config.model.num_characters).float(),
                F.one_hot(inputs["opponent_character"].squeeze(-1).long(), num_classes=self.config.model.num_characters).float(),
                F.one_hot(inputs["ego_action"].squeeze(-1).long(), num_classes=self.config.model.num_actions).float(),
                F.one_hot(inputs["opponent_action"].squeeze(-1).long(), num_classes=self.config.model.num_actions).float(),
                self.stage_emb(inputs["stage"]).squeeze(-2),
                self.character_emb(inputs["ego_character"]).squeeze(-2),
                self.character_emb(inputs["opponent_character"]).squeeze(-2),
                self.action_emb(inputs["ego_action"]).squeeze(-2),
                self.action_emb(inputs["opponent_action"]).squeeze(-2),
                inputs["gamestate"],
                inputs["controller"],
            ],
            dim=-1,
        )

    def forward(self, inputs: TensorDict, *, actions = None, return_rl_outputs = None) -> TensorDict:
        B, L, _ = inputs["gamestate"].shape
        assert L <= self.block_size, f"Cannot forward sequence of length {L}, block size is only {self.block_size}"

        combined_inputs = self._embed_inputs(inputs)
        x = self.proj_down(combined_inputs)
        x = self.drop(x)
        cos = self.cos[:, :L]
        sin = self.sin[:, :L]
        
        for block in self.blocks:
            x = block(x, cos, sin)

        x = norm(x)

        base = x
        # Reordered computation: buttons first, then main_stick, then c_stick, then shoulder
        button_logits, button_probs = self.button_head(base)
        main_stick = self.main_stick_head(torch.cat((base, button_logits), dim=-1))
        c_stick = self.c_stick_head(torch.cat((base, button_logits, main_stick), dim=-1))
        shoulder = self.shoulder_head(torch.cat((base, button_logits, main_stick, c_stick), dim=-1))

        outputs = TensorDict(
            {
                "buttons": button_logits,
                "buttons_probs": button_probs,
                "main_stick": main_stick,
                "c_stick": c_stick,
                "shoulder": shoulder,
            },
            batch_size=(B, L),
        )

        value = self.value_head(x)
        outputs.set("value", value)
        
        return outputs
