import torch
from torch import nn


class SimpleHead(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden, bias=True),
            nn.ReLU(),
            nn.Linear(hidden, output_size, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
