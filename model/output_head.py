import torch
import torch.nn.functional as F
from torch import nn


class SimpleHead(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden: int = 128,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, output_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass through both layers."""
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        return self.fc2(h)
