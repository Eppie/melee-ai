import torch
from torch import nn as nn


# TODO: We don't really need this, we can just use a SimpleHead
class ValueHead(nn.Module):
    def __init__(self, input_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden, bias=True),
            nn.ReLU(),
            nn.Linear(hidden, 1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
