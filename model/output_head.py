from typing import Tuple

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
        self.hidden_dim = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass through both layers."""
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        return self.fc2(h)

    def forward_with_hidden(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass that also returns the hidden activations.

        Returns:
            logits: Output logits [B, L, output_size]
            hidden: Pre-dropout hidden activations [B, L, hidden_dim]

        The hidden activations are returned pre-dropout to provide a richer
        signal about the head's internal reasoning, useful for conditioning
        downstream heads.
        """
        h = F.relu(self.fc1(x))
        h_drop = self.dropout(h)
        logits = self.fc2(h_drop)
        return logits, h  # return pre-dropout hidden for richer signal
