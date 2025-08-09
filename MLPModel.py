from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from typing import Sequence

import torch
from torch import nn


def gelu_init(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(
            m.weight, nonlinearity="relu"
        )  # good for GELU/Relu-family
        nn.init.zeros_(m.bias)


@dataclass(slots=True)
class MLPConfig:
    """Hyperparameters for the feed-forward network."""

    input_dim: int
    output_dim: int
    hidden_dims: Sequence[int] = (2048, 1024, 512, 256)
    dropout: float = 0.1
    act: type[nn.Module] = nn.GELU
    weight_init: Callable[[nn.Module], None] = gelu_init


class FeedForwardNet(nn.Module):
    """Simple fully-connected network that flattens sequence input."""

    def __init__(self, cfg: MLPConfig, feature_dim: int | None = None) -> None:
        super().__init__()
        # Normalize per time step across F before flattening
        self.pre_norm = nn.LayerNorm(feature_dim) if feature_dim is not None else None
        self.flatten = nn.Flatten()

        layers: list[nn.Module] = []
        dims = (cfg.input_dim, *cfg.hidden_dims, cfg.output_dim)
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            if out_dim != cfg.output_dim:
                layers.append(cfg.act())
                if cfg.dropout > 0:
                    layers.append(nn.Dropout(cfg.dropout))
        self.net = nn.Sequential(*layers)

        if cfg.weight_init is not None:
            self.apply(cfg.weight_init)

        self.net = nn.Sequential(*layers)

        if cfg.weight_init is not None:
            self.apply(cfg.weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, sequence_length, num_features)
        # Flatten to: (batch_size, sequence_length * num_features)
        x = self.flatten(x)
        return self.net(x)
