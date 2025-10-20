from typing import Tuple

import torch
import torch.nn as nn

from config import get_config
from model.feed_forward import ActivationFFN
from model.norm import _create_norm


class LinearHead(nn.Module):
    """Small head: Norm -> Linear."""

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.norm = _create_norm(input_size)
        self.fc = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.norm(x))


class MLPHead(nn.Module):
    """A standard MLP head: Norm -> FFN -> Linear."""

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        cfg = get_config().model
        # Use a hidden dimension that is a fraction of the input, e.g., half.
        # This is a common practice for heads to keep them lightweight.
        hidden_dim = input_size // 2

        self.net = nn.Sequential(
            _create_norm(input_size),
            # An FFN block for non-linearity
            ActivationFFN(
                input_size,
                mult=cfg.ffn_mult,  # Or you could use a smaller, fixed multiplier
                activation=cfg.ffn_activation,
            ),
            # Final projection to the output size
            nn.Linear(input_size, output_size, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiLabelButtonHeadLinear(nn.Module):
    """Small multi-label head: Norm -> Linear -> Sigmoid (for probs)."""

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.norm = _create_norm(input_size)
        self.fc = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.fc(self.norm(x))
        probs = torch.sigmoid(logits)
        return logits, probs


class MultiLabelButtonHead(nn.Module):
    """Predict independent button probabilities via shared features."""

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        cfg = get_config().model
        self.net = nn.Sequential(
            _create_norm(input_size),
            ActivationFFN(
                input_size,
                mult=cfg.ffn_mult,
                activation=cfg.ffn_activation,
            ),
            nn.Linear(input_size, output_size, bias=False),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.net(x)
        probs = torch.sigmoid(logits)
        return logits, probs


class TinyMLPHead(nn.Module):
    """Norm -> Linear(d->h) -> Act -> Linear(h->out)."""

    def __init__(
            self,
            input_size: int,
            output_size: int,
            *,
            hidden: int,
            activation: str,
    ) -> None:
        super().__init__()
        self.norm = _create_norm(input_size)
        act = activation.lower()
        if act == "gelu":
            act_layer: nn.Module = nn.GELU()
        elif act == "silu":
            act_layer = nn.SiLU()
        else:
            raise ValueError(f"Unsupported head activation '{activation}'")
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden, bias=False),
            act_layer,
            nn.Linear(hidden, output_size, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(x))


class TinyGLUHead(nn.Module):
    """Norm -> [w1(x) ⊗ act(v1(x))] -> w2 -> out. Much smaller than full FFN."""

    def __init__(
            self,
            input_size: int,
            output_size: int,
            *,
            hidden: int,
            activation: str,
    ) -> None:
        super().__init__()
        self.norm = _create_norm(input_size)
        self.w1 = nn.Linear(input_size, hidden, bias=False)
        self.v1 = nn.Linear(input_size, hidden, bias=False)
        self.w2 = nn.Linear(hidden, output_size, bias=False)
        act = activation.lower()
        if act == "swiglu":
            self._gate = torch.nn.functional.silu
        elif act == "geglu":
            self._gate = torch.nn.functional.gelu
        else:
            raise ValueError(f"Unsupported GLU activation '{activation}'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return self.w2(self._gate(self.v1(x)) * self.w1(x))


class LowRankAdapterHead(nn.Module):
    """
    Norm -> (x + Up(Act(Down(x)))) -> Linear(out)
    Nonlinearity lives inside the tiny adapter; very parameter efficient.
    """

    def __init__(
            self,
            input_size: int,
            output_size: int,
            *,
            rank: int,
            activation: str,
    ) -> None:
        super().__init__()
        self.norm = _create_norm(input_size)
        self.down = nn.Linear(input_size, rank, bias=False)
        self.up = nn.Linear(rank, input_size, bias=False)
        self.out = nn.Linear(input_size, output_size, bias=False)
        act = activation.lower()
        if act == "gelu":
            self.act: nn.Module = nn.GELU()
        elif act == "silu":
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unsupported adapter activation '{activation}'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        adapted = x + self.up(self.act(self.down(x)))
        return self.out(adapted)


class MultiLabelButtonHeadTiny(nn.Module):
    """Wraps any of the small heads above but returns (logits, probs)."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.inner(x)
        return logits, torch.sigmoid(logits)
