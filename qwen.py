"""Qwen-style controller models with hybrid attention and sparse MoE options."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from preprocess import C_STICK_XY_CLUSTER_CENTERS_V0_1, FOX_STICK_64


def _rope(q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors."""

    d = q.size(-1)
    device = q.device
    positions = torch.arange(q.size(-2), device=device, dtype=q.dtype)
    freqs = torch.einsum(
        "l,d->l d",
        positions,
        10000.0 ** (-2 * torch.arange(d // 2, device=device, dtype=q.dtype) / d),
    )
    cos = freqs.cos()[None, None, :, :]
    sin = freqs.sin()[None, None, :, :]

    def _apply(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)

    return _apply(q), _apply(k)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.scale


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network with configurable hidden size."""

    def __init__(self, dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.w = nn.Linear(dim, hidden, bias=False)
        self.v = nn.Linear(dim, hidden, bias=False)
        self.out = nn.Linear(hidden, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = F.silu(self.w(x)) * self.v(x)
        return self.dropout(self.out(gated))


@dataclass(frozen=True)
class QwenConfig:
    name: str
    block_size: int
    n_layer: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    d_ff: int
    dropout: float = 0.0
    attn_dropout: float = 0.0
    bias: bool = True
    linear_ratio: int = 3
    full_attention_every: int = 4
    moe_num_experts: int = 0
    moe_top_k: int = 0
    moe_has_shared: bool = False
    input_size: int = 130
    num_stages: int = 7
    num_characters: int = 26
    num_actions: int = 396
    stage_embedding_dim: int = 4
    character_embedding_dim: int = 12
    action_embedding_dim: int = 32
    gamma: float = 0.999
    target_shapes_by_head: Dict[str, Tuple[int, ...]] = field(
        default_factory=lambda: {
            "main_stick": (len(FOX_STICK_64),),
            "c_stick": (len(C_STICK_XY_CLUSTER_CENTERS_V0_1),),
            "buttons": (5,),
            "shoulder": (3,),
        }
    )

    def validate(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.full_attention_every < 1:
            raise ValueError("full_attention_every must be >= 1")
        if self.moe_num_experts and self.moe_top_k <= 0:
            raise ValueError("moe_top_k must be positive when using MoE")


def _feature_map(x: torch.Tensor) -> torch.Tensor:
    return F.elu(x) + 1.0


class HybridAttention(nn.Module):
    def __init__(self, config: QwenConfig, *, use_full_attention: bool) -> None:
        super().__init__()
        dim = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = dim // config.n_heads
        self.dropout = config.attn_dropout
        self.use_full_attention = use_full_attention

        self.q_proj = nn.Linear(dim, dim, bias=config.bias)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=config.bias)
        self.out_proj = nn.Linear(dim, dim, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

    def _reshape(self, tensor: torch.Tensor, heads: int) -> torch.Tensor:
        b, l, d = tensor.shape
        return tensor.view(b, l, heads, d // heads).transpose(1, 2)

    def _expand_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_kv_heads == self.n_heads:
            return x
        repeat = self.n_heads // self.n_kv_heads
        b, h, l, d = x.shape
        x = x[:, :, None].repeat(1, 1, repeat, 1, 1)
        return x.reshape(b, self.n_heads, l, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self._reshape(self.q_proj(x), self.n_heads)
        k = self._reshape(self.k_proj(x), self.n_kv_heads)
        v = self._reshape(self.v_proj(x), self.n_kv_heads)

        q, k = _rope(q, k)
        k = self._expand_kv(k)
        v = self._expand_kv(v)

        if self.use_full_attention:
            attn = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            phi_q = _feature_map(q)
            phi_k = _feature_map(k)
            kv = torch.einsum("bhlm,bhlv->bhmv", phi_k, v)
            z = 1.0 / (torch.einsum("bhlm,bhm->bhl", phi_q, phi_k.sum(dim=2)) + 1e-6)
            attn = torch.einsum("bhlm,bhmv->bhlv", phi_q, kv)
            attn = attn * z.unsqueeze(-1)

        attn = attn.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], -1)
        attn = self.out_proj(attn)
        return self.resid_dropout(attn)


class DenseFFN(nn.Module):
    def __init__(self, config: QwenConfig) -> None:
        super().__init__()
        self.norm = RMSNorm(config.d_model)
        self.ffn = SwiGLUFFN(config.d_model, config.d_ff, dropout=config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.norm(x))


class MoEExpertPool(nn.Module):
    def __init__(self, dim: int, hidden: int, num_experts: int, dropout: float) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            [SwiGLUFFN(dim, hidden, dropout=dropout) for _ in range(num_experts)]
        )

    def forward(self, expert_idx: int, x: torch.Tensor) -> torch.Tensor:
        return self.experts[expert_idx](x)


class SparseMoE(nn.Module):
    def __init__(
        self,
        config: QwenConfig,
        pool: MoEExpertPool,
    ) -> None:
        super().__init__()
        self.pool = pool
        self.dim = config.d_model
        self.router = nn.Linear(self.dim, config.moe_num_experts - (1 if config.moe_has_shared else 0))
        self.shared_proj = (
            nn.Linear(self.dim, 1, bias=True) if config.moe_has_shared else None
        )
        self.top_k = config.moe_top_k
        self.num_experts = config.moe_num_experts
        self.has_shared = config.moe_has_shared
        self.shared_index = self.num_experts - 1 if self.has_shared else None
        self.norm = RMSNorm(self.dim)

    def _dispatch(
        self, x: torch.Tensor, top_idx: torch.Tensor, gate: torch.Tensor
    ) -> torch.Tensor:
        b_l, k = top_idx.shape
        out = torch.zeros(b_l, self.dim, device=x.device, dtype=x.dtype)
        for slot in range(k):
            expert_ids = top_idx[:, slot]
            if expert_ids.numel() == 0:
                continue
            unique_ids = expert_ids.unique()
            for expert_id in unique_ids.tolist():
                mask = expert_ids == expert_id
                if not mask.any():
                    continue
                expert_inp = x[mask]
                expert_out = self.pool.forward(expert_id, expert_inp)
                out[mask] += expert_out * gate[mask, slot, None]
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, d = x.shape
        normed = self.norm(x)
        routed = self.router(normed)
        top_scores, top_idx = routed.topk(self.top_k, dim=-1)

        if self.has_shared:
            shared_scores = self.shared_proj(normed)
            scores = torch.cat([top_scores, shared_scores], dim=-1)
            probs = F.softmax(scores, dim=-1)
            top_gate = probs[..., : self.top_k]
            shared_gate = probs[..., self.top_k :]
        else:
            top_gate = F.softmax(top_scores, dim=-1)
            shared_gate = None

        flat_x = normed.reshape(-1, d)
        flat_idx = top_idx.reshape(-1, self.top_k)
        flat_gate = top_gate.reshape(-1, self.top_k)
        dispatched = self._dispatch(flat_x, flat_idx, flat_gate)

        if self.has_shared and self.shared_index is not None and shared_gate is not None:
            shared_out = self.pool.forward(self.shared_index, flat_x)
            gate = shared_gate.reshape(-1)
            dispatched += shared_out * gate[:, None]

        return x + dispatched.view(b, l, d)


class HybridBlock(nn.Module):
    def __init__(
        self,
        config: QwenConfig,
        *,
        use_full_attention: bool,
        moe_pool: Optional[MoEExpertPool],
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = HybridAttention(config, use_full_attention=use_full_attention)

        if moe_pool is not None:
            self.ffn = SparseMoE(config, pool=moe_pool)
        else:
            self.ffn = DenseFFN(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = self.ffn(x)
        return x


class MultiLabelButtonHead(nn.Module):
    def __init__(self, input_size: int, output_size: int, *, bias: bool) -> None:
        super().__init__()
        hidden = max(1, max(input_size // 2, output_size * 2))
        mult = hidden / input_size if input_size > 0 else 1.0
        self.net = nn.Sequential(
            RMSNorm(input_size),
            SwiGLUFFN(input_size, int(input_size * mult), dropout=0.0),
            nn.Linear(input_size, output_size, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.net(x)
        return logits, torch.sigmoid(logits)


class QwenBase(nn.Module):
    def __init__(self, config: QwenConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config

        self.stage_emb = nn.Embedding(config.num_stages, config.stage_embedding_dim)
        self.character_emb = nn.Embedding(config.num_characters, config.character_embedding_dim)
        self.action_emb = nn.Embedding(config.num_actions, config.action_embedding_dim)

        self.proj_in = nn.Linear(config.input_size, config.d_model, bias=config.bias)
        self.drop = nn.Dropout(config.dropout)

        moe_pool = (
            MoEExpertPool(config.d_model, config.d_ff, config.moe_num_experts, config.dropout)
            if config.moe_num_experts
            else None
        )

        blocks = []
        for idx in range(config.n_layer):
            if config.n_layer < config.full_attention_every:
                use_full = idx == config.n_layer - 1
            else:
                use_full = (idx + 1) % config.full_attention_every == 0
            blocks.append(
                HybridBlock(
                    config,
                    use_full_attention=use_full,
                    moe_pool=moe_pool,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = RMSNorm(config.d_model)

        targets = config.target_shapes_by_head
        shoulder_size = targets["shoulder"][0]
        c_stick_size = targets["c_stick"][0]
        main_size = targets["main_stick"][0]
        button_size = targets["buttons"][0]

        self.shoulder_head = nn.Sequential(
            RMSNorm(config.d_model),
            SwiGLUFFN(config.d_model, max(config.d_model // 2, 1), dropout=config.dropout),
            nn.Linear(config.d_model, shoulder_size, bias=config.bias),
        )

        c_input = config.d_model + shoulder_size
        self.c_stick_head = nn.Sequential(
            RMSNorm(c_input),
            SwiGLUFFN(c_input, max(c_input // 2, 1), dropout=config.dropout),
            nn.Linear(c_input, c_stick_size, bias=config.bias),
        )

        m_input = config.d_model + shoulder_size + c_stick_size
        self.main_stick_head = nn.Sequential(
            RMSNorm(m_input),
            SwiGLUFFN(m_input, max(m_input // 2, 1), dropout=config.dropout),
            nn.Linear(m_input, main_size, bias=config.bias),
        )

        b_input = config.d_model + shoulder_size + c_stick_size + main_size
        self.button_head = MultiLabelButtonHead(
            b_input,
            button_size,
            bias=config.bias,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def block_size(self) -> int:
        return self.config.block_size

    def _embed(self, inputs: TensorDict) -> torch.Tensor:
        return torch.cat(
            [
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

    def forward(self, inputs: TensorDict) -> TensorDict:
        batch, seq, _ = inputs["gamestate"].shape
        if seq > self.block_size:
            raise ValueError(
                f"Cannot forward sequence of length {seq}; block size is {self.block_size}"
            )

        combined = self._embed(inputs)
        x = self.drop(self.proj_in(combined))
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)

        shoulder = self.shoulder_head(x)
        c_stick = self.c_stick_head(torch.cat([x, shoulder.detach()], dim=-1))
        main = self.main_stick_head(torch.cat([x, shoulder.detach(), c_stick.detach()], dim=-1))
        button_logits, button_probs = self.button_head(
            torch.cat([x, shoulder.detach(), c_stick.detach(), main.detach()], dim=-1)
        )

        return TensorDict(
            {
                "buttons": button_logits,
                "buttons_probs": button_probs,
                "main_stick": main,
                "c_stick": c_stick,
                "shoulder": shoulder,
            },
            batch_size=(batch, seq),
        )


PRESETS: Dict[str, QwenConfig] = {
    "qwen_tiny_100k": QwenConfig(
        name="qwen_tiny_100k",
        block_size=256,
        n_layer=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=1,
        d_ff=192,
        dropout=0.0,
        attn_dropout=0.0,
    ),
    "qwen_small_1m": QwenConfig(
        name="qwen_small_1m",
        block_size=256,
        n_layer=5,
        d_model=128,
        n_heads=4,
        n_kv_heads=2,
        d_ff=384,
        dropout=0.0,
        attn_dropout=0.0,
    ),
    "qwen_medium_10m": QwenConfig(
        name="qwen_medium_10m",
        block_size=256,
        n_layer=10,
        d_model=256,
        n_heads=8,
        n_kv_heads=4,
        d_ff=1024,
        dropout=0.0,
        attn_dropout=0.0,
    ),
    "qwen_moe_50m": QwenConfig(
        name="qwen_moe_50m",
        block_size=256,
        n_layer=8,
        d_model=320,
        n_heads=16,
        n_kv_heads=1,
        d_ff=1536,
        dropout=0.0,
        attn_dropout=0.0,
        moe_num_experts=33,
        moe_top_k=4,
        moe_has_shared=True,
    ),
}


class QwenTiny(QwenBase):
    def __init__(self, config: Optional[QwenConfig] = None) -> None:
        super().__init__(config or PRESETS["qwen_tiny_100k"])


class QwenSmall(QwenBase):
    def __init__(self, config: Optional[QwenConfig] = None) -> None:
        super().__init__(config or PRESETS["qwen_small_1m"])


class QwenMedium(QwenBase):
    def __init__(self, config: Optional[QwenConfig] = None) -> None:
        super().__init__(config or PRESETS["qwen_medium_10m"])


class QwenMoE(QwenBase):
    def __init__(self, config: Optional[QwenConfig] = None) -> None:
        super().__init__(config or PRESETS["qwen_moe_50m"])


def available_qwen_configs() -> Iterable[str]:
    return PRESETS.keys()
