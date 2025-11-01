"""
Auxiliary prediction heads for self-supervised learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OpponentActionHead(nn.Module):
    """Predicts opponent's next action from current state representation"""

    def __init__(self, n_embd, num_actions, hidden=128):
        super().__init__()
        self.num_actions = num_actions
        self.fc1 = nn.Linear(n_embd, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, num_actions, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, n_embd) state representation
        Returns:
            logits: (B, L, num_actions)
        """
        h = F.relu(self.fc1(x)).square()  # Match MLP activation
        logits = self.fc2(h)
        return logits


class DamageDifferentialHead(nn.Module):
    """Predicts net damage differential over next N frames"""

    def __init__(self, n_embd, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, n_embd) state representation
        Returns:
            damage_diff: (B, L, 1) predicted net damage
        """
        h = F.relu(self.fc1(x)).square()
        damage_diff = self.fc2(h)
        return damage_diff


class ActionEffectivenessHead(nn.Module):
    """Predicts if ego action will cause opponent hitlag within K frames"""

    def __init__(self, n_embd, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, n_embd) state representation
        Returns:
            logits: (B, L, 1) logits for hitlag prediction
        """
        h = F.relu(self.fc1(x)).square()
        logits = self.fc2(h)
        return logits
