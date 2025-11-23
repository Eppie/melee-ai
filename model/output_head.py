import torch
import torch.nn.functional as F
from torch import nn


class SimpleHead(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, output_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass through both layers."""
        h = F.relu(self.fc1(x))
        return self.fc2(h)

    def forward_intermediate(self, x: torch.Tensor) -> torch.Tensor:
        """Returns intermediate features after first layer (before final projection)."""
        return F.relu(self.fc1(x))

    def forward_from_intermediate(self, h: torch.Tensor) -> torch.Tensor:
        """Projects from intermediate features to output."""
        return self.fc2(h)
