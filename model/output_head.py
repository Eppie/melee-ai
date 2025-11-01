import torch
from torch import nn as nn

from model.norm import norm


# TODO: Might want to enable bias here actually
class SimpleHead(nn.Module):
    def __init__(self, input_size, output_size, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden, bias=False),
            nn.ReLU(),
            nn.Linear(hidden, output_size, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ButtonHead(nn.Module):
    def __init__(self, input_size, output_size, hidden=128):
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
