from typing import Optional, Tuple

import torch
from torch import nn as nn

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
