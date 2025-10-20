from typing import Tuple

import torch
from torch import nn as nn

from config import get_config


class ActivationFFN(nn.Module):
    def __init__(self, d: int, mult: float, activation: str) -> None:
        super().__init__()
        inner = max(1, int(mult * d))
        act = activation.lower()
        if act == "swiglu":
            self.w1 = nn.Linear(d, inner, bias=False)
            self.v1 = nn.Linear(d, inner, bias=False)
            self.w2 = nn.Linear(inner, d, bias=False)
            self.activation = "swiglu"
        elif act == "gelu":
            self.net = nn.Sequential(
                nn.Linear(d, inner, bias=False),
                nn.GELU(),
                nn.Linear(inner, d, bias=False),
            )
            self.activation = "gelu"
        elif act == "geglu":
            self.w1 = nn.Linear(d, inner, bias=False)
            self.v1 = nn.Linear(d, inner, bias=False)
            self.w2 = nn.Linear(inner, d, bias=False)
            self.activation = "geglu"
        else:
            raise ValueError(f"Unsupported FFN activation '{activation}'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "swiglu":
            return self.w2(torch.nn.functional.silu(self.w1(x)) * self.v1(x))
        if self.activation == "geglu":
            return self.w2(torch.nn.functional.gelu(self.w1(x)) * self.v1(x))
        return self.net(x)