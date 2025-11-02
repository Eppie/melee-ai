import torch
from torch import nn as nn


class SimpleHead(nn.Module):
    def __init__(self, input_size, output_size, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden, bias=True),
            nn.ReLU(),
            nn.Linear(hidden, output_size, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ButtonHead(nn.Module):
    def __init__(self, input_size, output_size, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden, bias=True),
            nn.ReLU(),
            nn.Linear(hidden, output_size, bias=True),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO: Why are we doing the sigmoid here?
        logits = self.net(x)
        probs = torch.sigmoid(logits)
        return logits, probs
