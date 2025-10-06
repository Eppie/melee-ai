"""Adapted from Karpathy's nanoGPT: https://github.com/karpathy/nanoGPT."""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Categorical
from tensordict import TensorDict

from config import get_config


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6, affine: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d)) if affine else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        y = x * torch.rsqrt(norm + self.eps)
        if self.weight is not None:
            y = y * self.weight
        return y


def _make_norm(dim: int, norm_type: str, eps: float, affine: bool) -> nn.Module:
    kind = norm_type.lower()
    alias = {
        "qk_norm": "rmsnorm",
        "qknorm": "rmsnorm",
    }
    kind = alias.get(kind, kind)
    if kind == "rmsnorm":
        return RMSNorm(dim, eps=eps, affine=affine)
    if kind == "layernorm":
        return nn.LayerNorm(dim, eps=eps, elementwise_affine=affine)
    raise ValueError(f"Unsupported normalization type '{norm_type}'.")


def _create_norm(dim: int, *, override_type: Optional[str] = None) -> nn.Module:
    config = get_config()
    norm_type = override_type or config.model.norm_type
    return _make_norm(dim, norm_type, config.model.norm_eps, config.model.norm_affine)


class QKNorm(nn.Module):
    def __init__(self, dim: int, norm_type: str, eps: float, affine: bool) -> None:
        super().__init__()
        self.q_norm = _make_norm(dim, norm_type, eps, affine)
        self.k_norm = _make_norm(dim, norm_type, eps, affine)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q_norm(q), self.k_norm(k)


def _ntk_scaled_theta(theta: float, factor: float) -> float:
    return theta * factor


def _rope_cache(seq_len: int, head_dim: int, theta: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    half = head_dim // 2
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = theta ** (-2 * torch.arange(half, device=device, dtype=torch.float32) / head_dim)
    angles = torch.einsum("l,d->l d", positions, freqs)
    cos = angles.cos()[None, None, :, :]
    sin = angles.sin()[None, None, :, :]
    return cos, sin


def apply_rope_inplace(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[
    torch.Tensor, torch.Tensor]:
    q_even, q_odd = q[..., ::2], q[..., 1::2]
    k_even, k_odd = k[..., ::2], k[..., 1::2]
    q[..., ::2] = q_even * cos - q_odd * sin
    q[..., 1::2] = q_even * sin + q_odd * cos
    k[..., ::2] = k_even * cos - k_odd * sin
    k[..., 1::2] = k_even * sin + k_odd * cos
    return q, k


def _alibi_slopes(n_head: int) -> torch.Tensor:
    def get_slopes(n: int) -> torch.Tensor:
        import math

        def pow2(x: int) -> int:
            return 2 ** math.floor(math.log2(x))

        m = pow2(n)
        slopes = torch.pow(2, -(torch.arange(1, m + 1, dtype=torch.float32) / m))
        if m < n:
            extra = torch.tensor([slopes[-1] * (i + 2) for i in range(n - m)], dtype=torch.float32)
            slopes = torch.cat([slopes, extra], dim=0)
        return slopes

    return get_slopes(n_head)


def _alibi_bias(batch: int, n_head: int, seq_len: int, device: torch.device) -> torch.Tensor:
    slopes = _alibi_slopes(n_head).to(device)
    positions = torch.arange(seq_len, device=device)
    distance = (positions[None, :] - positions[:, None]).clamp(min=0).to(torch.float32)
    bias = -slopes[:, None, None] * distance[None, :, :]
    return bias.unsqueeze(0).expand(batch, -1, -1, -1)


class ActivationFFN(nn.Module):
    def __init__(self, d: int, mult: float, activation: str, bias: bool = False) -> None:
        super().__init__()
        inner = max(1, int(mult * d))
        act = activation.lower()
        if act == "swiglu":
            self.w1 = nn.Linear(d, inner, bias=bias)
            self.v1 = nn.Linear(d, inner, bias=bias)
            self.w2 = nn.Linear(inner, d, bias=bias)
            self.activation = "swiglu"
        elif act == "gelu":
            self.net = nn.Sequential(
                nn.Linear(d, inner, bias=bias),
                nn.GELU(),
                nn.Linear(inner, d, bias=bias),
            )
            self.activation = "gelu"
        elif act == "geglu":
            self.w1 = nn.Linear(d, inner, bias=bias)
            self.v1 = nn.Linear(d, inner, bias=bias)
            self.w2 = nn.Linear(inner, d, bias=bias)
            self.activation = "geglu"
        else:
            raise ValueError(f"Unsupported FFN activation '{activation}'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "swiglu":
            return self.w2(torch.nn.functional.silu(self.w1(x)) * self.v1(x))
        if self.activation == "geglu":
            return self.w2(torch.nn.functional.gelu(self.w1(x)) * self.v1(x))
        return self.net(x)


class MoEFFN(nn.Module):
    """Mixture-of-Experts feed-forward network with optional shared expert path."""

    def __init__(
        self,
        d: int,
        num_experts: int,
        num_active: int,
        mult: float,
        activation: str,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if num_active > num_experts:
            raise ValueError("moe_num_active must be <= moe_num_experts")

        cfg = get_config().model
        self.num_experts = num_experts
        self.num_active = num_active
        self.d = d

        self.router = nn.Linear(d, num_experts, bias=False)
        self.experts = nn.ModuleList(
            ActivationFFN(d, mult, activation, bias) for _ in range(num_experts)
        )

        self.use_shared_expert = cfg.moe_shared_expert
        if self.use_shared_expert:
            self.shared_expert = ActivationFFN(d, mult, activation, bias)
            self.shared_expert_weight = nn.Parameter(torch.ones(1))

        self.capacity_factor = cfg.moe_expert_capacity_factor
        self.use_capacity = self.capacity_factor is not None
        self.jitter_eps = cfg.moe_jitter_eps
        self.normalize_weights = cfg.moe_normalize_expert_weights

    def _compute_expert_capacity(self, num_tokens: int) -> int:
        if not self.use_capacity:
            return num_tokens
        tokens_per_expert = (num_tokens * self.num_active) / max(1, self.num_experts)
        capacity = int(tokens_per_expert * float(self.capacity_factor))
        return max(1, capacity)

    def _add_routing_noise(self, logits: torch.Tensor) -> torch.Tensor:
        if self.training and self.jitter_eps > 0:
            noise = torch.randn_like(logits) * self.jitter_eps
            return logits * (1.0 + noise)
        return logits

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        x_flat = x.view(-1, D)
        num_tokens = x_flat.shape[0]

        shared_output = None
        shared_weight = None
        if self.use_shared_expert:
            shared_output = self.shared_expert(x_flat)
            shared_weight = torch.sigmoid(self.shared_expert_weight)

        router_logits = self.router(x_flat)
        router_logits = self._add_routing_noise(router_logits)

        topk_values, topk_indices = torch.topk(router_logits, self.num_active, dim=-1)
        if self.normalize_weights:
            routing_weights = torch.softmax(topk_values, dim=-1)
        else:
            routing_weights = torch.relu(topk_values)

        expert_capacity = self._compute_expert_capacity(num_tokens)

        output = torch.zeros_like(x_flat)
        expert_mask = torch.zeros(
            num_tokens, self.num_experts, device=x.device, dtype=torch.float32
        )

        for expert_idx in range(self.num_experts):
            assignments = (topk_indices == expert_idx).nonzero(as_tuple=False)
            if assignments.numel() == 0:
                continue

            token_indices = assignments[:, 0]
            slot_indices = assignments[:, 1]
            weights = routing_weights[token_indices, slot_indices]

            if self.use_capacity and token_indices.numel() > expert_capacity:
                kept_weights, keep_idx = torch.topk(
                    weights, expert_capacity, sorted=False
                )
                token_indices = token_indices[keep_idx]
                slot_indices = slot_indices[keep_idx]
                weights = kept_weights

            if token_indices.numel() == 0:
                continue

            tokens = x_flat[token_indices]
            expert_out = self.experts[expert_idx](tokens)
            expert_mask[token_indices, expert_idx] = 1.0
            output[token_indices] += weights.unsqueeze(-1) * expert_out

        if self.use_shared_expert and shared_output is not None and shared_weight is not None:
            output = (1.0 - shared_weight) * output + shared_weight * shared_output

        output = output.view(B, L, D)

        expert_usage = expert_mask.mean(dim=0)
        router_probs = torch.softmax(router_logits, dim=-1).mean(dim=0)
        aux_loss = self.num_experts * (expert_usage * router_probs).sum()

        return output, aux_loss


class MoEMLP(nn.Module):
    """Transformer MLP block with optional MoE FFN."""

    def __init__(self) -> None:
        super().__init__()
        cfg = get_config().model
        if cfg.use_moe:
            self.ffn = MoEFFN(
                cfg.n_embd,
                num_experts=cfg.moe_num_experts,
                num_active=cfg.moe_num_active,
                mult=cfg.ffn_mult,
                activation=cfg.ffn_activation,
                bias=cfg.bias,
            )
            self.is_moe = True
        else:
            self.ffn = ActivationFFN(
                cfg.n_embd,
                mult=cfg.ffn_mult,
                activation=cfg.ffn_activation,
                bias=cfg.bias,
            )
            self.is_moe = False
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.is_moe:
            out, aux_loss = self.ffn(x)
            return self.dropout(out), aux_loss
        return self.dropout(self.ffn(x)), None


class MultiLabelButtonHead(nn.Module):
    """Predict independent button probabilities via shared features."""

    def __init__(self, input_size: int, output_size: int, *, bias: bool) -> None:
        super().__init__()
        cfg = get_config().model
        self.net = nn.Sequential(
            _create_norm(input_size),
            ActivationFFN(
                input_size,
                mult=cfg.ffn_mult,
                activation=cfg.ffn_activation,
                bias=bias,
            ),
            nn.Linear(input_size, output_size, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.net(x)
        probs = torch.sigmoid(logits)
        return logits, probs


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


class Block(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        config = get_config()
        self.attn = CausalSelfAttention()
        self.mlp = MoEMLP()
        placement = config.model.norm_placement.lower()
        if placement not in {"pre", "post", "both"}:
            raise ValueError(f"Unsupported norm placement '{config.model.norm_placement}'.")

        use_pre = placement in {"pre", "both"}
        use_post = placement in {"post", "both"}

        self.pre_attn_norm = _create_norm(config.model.n_embd) if use_pre else None
        self.post_attn_norm = _create_norm(config.model.n_embd) if use_post else None
        self.pre_mlp_norm = _create_norm(config.model.n_embd) if use_pre else None
        self.post_mlp_norm = _create_norm(config.model.n_embd) if use_post else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = x
        attn_input = self.pre_attn_norm(x) if self.pre_attn_norm is not None else x
        attn_out = self.attn(attn_input)
        if self.post_attn_norm is not None:
            attn_out = self.post_attn_norm(attn_out)
        x = residual + attn_out

        residual = x
        mlp_input = self.pre_mlp_norm(x) if self.pre_mlp_norm is not None else x
        mlp_out, aux_loss = self.mlp(mlp_input)
        if self.post_mlp_norm is not None:
            mlp_out = self.post_mlp_norm(mlp_out)
        x = residual + mlp_out
        return x, aux_loss


class BaseGPT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        config = get_config()
        self.block_size = config.model.block_size

    def get_num_params(self, non_embedding=True):
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, inputs: TensorDict):
        raise NotImplementedError


class GPTv7(BaseGPT):
    def __init__(self) -> None:
        super().__init__()
        config = get_config()
        self.input_size = config.model.input_size  # G
        self.n_embd = config.model.n_embd  # D

        # Categorical input embeddings
        self.stage_emb = nn.Embedding(config.model.num_stages, config.model.stage_embedding_dim)
        self.character_emb = nn.Embedding(config.model.num_characters, config.model.character_embedding_dim)
        self.action_emb = nn.Embedding(config.model.num_actions, config.model.action_embedding_dim)

        self.transformer = nn.ModuleDict(
            dict(
                proj_down=nn.Linear(self.input_size, config.model.n_embd),  # G -> D
                drop=nn.Dropout(config.model.dropout),
                h=nn.ModuleList([Block() for _ in range(config.model.n_layer)]),
                ln_f=_create_norm(self.n_embd),
            )
        )

        # Output heads
        self.target_shapes_by_head = config.model.target_shapes_by_head
        shoulder_output_size = self.target_shapes_by_head["shoulder"][0]

        c_stick_input_size = self.n_embd + shoulder_output_size
        c_stick_output_size = self.target_shapes_by_head["c_stick"][0]

        main_stick_input_size = self.n_embd + shoulder_output_size + c_stick_output_size
        main_stick_output_size = self.target_shapes_by_head["main_stick"][0]

        button_input_size = self.n_embd + shoulder_output_size + c_stick_output_size + main_stick_output_size
        button_output_size = self.target_shapes_by_head["buttons"][0]

        # Put shoulder and c-stick first because they are less complex and they modify/override other inputs
        shoulder_hidden = max(1, self.n_embd // 2)

        def head_block(input_dim: int, output_dim: int) -> nn.Sequential:
            return nn.Sequential(
                _create_norm(input_dim),
                ActivationFFN(
                    input_dim,
                    mult=config.model.ffn_mult,
                    activation=config.model.ffn_activation,
                    bias=config.model.bias,
                ),
                nn.Linear(input_dim, output_dim, bias=config.model.bias),
            )

        self.shoulder_head = head_block(self.n_embd, shoulder_output_size)
        self.c_stick_head = head_block(c_stick_input_size, c_stick_output_size)
        self.main_stick_head = head_block(main_stick_input_size, main_stick_output_size)
        self.button_head = MultiLabelButtonHead(
            button_input_size,
            button_output_size,
            bias=config.model.bias,
        )

        self.enable_rl_heads = bool(getattr(config.model, "enable_rl_heads", False))
        self._rl_value_head: Optional[nn.Module] = None
        self._initialize_rl_heads(config)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.model.n_layer))

    def _embed_inputs(self, inputs: TensorDict) -> torch.Tensor:
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

    def forward(
        self,
        inputs: TensorDict,
        *,
        actions: Optional[TensorDict] = None,
        return_rl_outputs: Optional[bool] = None,
    ) -> TensorDict:
        # loguru.logger.info(f"{inputs}")
        # B = batch size
        # L = sequence length
        B, L, _ = inputs["gamestate"].shape
        assert L <= self.block_size, f"Cannot forward sequence of length {L}, block size is only {self.block_size}"

        # Concatenate embeddings and numerical inputs -> project down
        combined_inputs_BLG = self._embed_inputs(inputs)
        proj_inputs_BLD = self.transformer.proj_down(combined_inputs_BLG)

        x_BLD = self.transformer.drop(proj_inputs_BLD)
        total_aux_loss: Optional[torch.Tensor] = None
        num_moe_layers = 0
        for block in self.transformer.h:
            x_BLD, aux_loss = block(x_BLD)
            if aux_loss is not None:
                total_aux_loss = aux_loss if total_aux_loss is None else total_aux_loss + aux_loss
                num_moe_layers += 1
        x_BLD = self.transformer.ln_f(x_BLD)

        head_flow = get_config().model.head_flow.lower()

        if head_flow == "parallel":
            base = x_BLD
            shoulder = self.shoulder_head(base)
            c_stick = self.c_stick_head(torch.cat((base, shoulder), dim=-1))
            main_stick = self.main_stick_head(torch.cat((base, shoulder, c_stick), dim=-1))
            button_logits, button_probs = self.button_head(
                torch.cat((base, shoulder, c_stick, main_stick), dim=-1)
            )
        elif head_flow == "sequential":
            shoulder = self.shoulder_head(x_BLD)
            c_input = torch.cat((x_BLD, shoulder.detach()), dim=-1)
            c_stick = self.c_stick_head(c_input)
            main_input = torch.cat((x_BLD, shoulder.detach(), c_stick.detach()), dim=-1)
            main_stick = self.main_stick_head(main_input)
            button_input = torch.cat(
                (x_BLD, shoulder.detach(), c_stick.detach(), main_stick.detach()),
                dim=-1,
            )
            button_logits, button_probs = self.button_head(button_input)
        else:
            raise ValueError(f"Unsupported head_flow '{get_config().model.head_flow}'.")

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

        cfg = get_config().model
        if cfg.use_moe and total_aux_loss is not None and num_moe_layers > 0:
            avg_aux_loss = total_aux_loss / num_moe_layers
            outputs.set("moe_aux_loss", avg_aux_loss.reshape(1, 1).expand(B, L))

        add_rl = return_rl_outputs if return_rl_outputs is not None else self.enable_rl_heads
        if not add_rl:
            return outputs

        if not self.enable_rl_heads:
            raise RuntimeError(
                "RL outputs requested but `enable_rl_heads` is False in the config."
            )

        rl_data = TensorDict({}, batch_size=(B, L))
        if self._rl_value_head is not None:
            value = self._rl_value_head(x_BLD).squeeze(-1)
            rl_data.set("value", value)

        policy = TensorDict({}, batch_size=(B, L))
        policy.set("buttons_logits", button_logits)
        policy.set("buttons_probs", button_probs)
        policy.set("shoulder_logits", shoulder)
        policy.set("c_stick_logits", c_stick)
        policy.set("main_stick_logits", main_stick)
        rl_data.set("policy", policy)

        if actions is not None:
            log_prob, entropy = self._rl_evaluate_actions(policy, actions)
            rl_data.set("action_log_prob", log_prob)
            rl_data.set("policy_entropy", entropy)

        outputs.update(rl_data)
        return outputs

    def _initialize_rl_heads(
        self,
        config,
    ) -> None:
        if not self.enable_rl_heads:
            return

        bias = config.model.bias
        self._rl_value_head = nn.Sequential(
            _create_norm(self.n_embd),
            nn.Linear(self.n_embd, 1, bias=bias),
        )

    def _rl_evaluate_actions(
        self,
        policy: TensorDict,
        actions: TensorDict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = next(iter(policy.values()))
        log_prob = sample.new_zeros(policy.batch_size)
        entropy = sample.new_zeros(policy.batch_size)

        if "buttons" in actions.keys() and "buttons_logits" in policy.keys():
            buttons_logits = policy.get("buttons_logits")
            buttons_actions = actions.get("buttons").to(buttons_logits.device, buttons_logits.dtype)
            btn_dist = Bernoulli(logits=buttons_logits)
            log_prob = log_prob + btn_dist.log_prob(buttons_actions).sum(dim=-1)
            entropy = entropy + btn_dist.entropy().sum(dim=-1)

        if "main_stick" in actions.keys() and "main_stick_logits" in policy.keys():
            logits = policy.get("main_stick_logits")
            main_actions = actions.get("main_stick").to(logits.device).long()
            dist = Categorical(logits=logits)
            log_prob = log_prob + dist.log_prob(main_actions)
            entropy = entropy + dist.entropy()

        if "c_stick" in actions.keys() and "c_stick_logits" in policy.keys():
            logits = policy.get("c_stick_logits")
            c_actions = actions.get("c_stick").to(logits.device).long()
            dist = Categorical(logits=logits)
            log_prob = log_prob + dist.log_prob(c_actions)
            entropy = entropy + dist.entropy()

        if "shoulder" in actions.keys() and "shoulder_logits" in policy.keys():
            logits = policy.get("shoulder_logits")
            sh_actions = actions.get("shoulder").to(logits.device).long()
            dist = Categorical(logits=logits)
            log_prob = log_prob + dist.log_prob(sh_actions)
            entropy = entropy + dist.entropy()

        return log_prob, entropy
