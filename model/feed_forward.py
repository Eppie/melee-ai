from typing import Tuple

import torch
from torch import nn as nn

from config import get_config


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
