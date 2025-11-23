"""
Cross-attention mechanism for output heads.

Allows output heads to share information by applying self-attention across
the head types, enabling each head to be aware of what other heads are predicting.
"""

import torch
import torch.nn as nn


class HeadCrossAttention(nn.Module):
    """
    Cross-attention mechanism that allows output heads to share information.

    Takes intermediate features from each head and applies self-attention across
    the head types, allowing each head to be aware of what other heads are predicting.
    """

    def __init__(
        self, hidden_dim: int, num_head_types: int = 4, num_attn_heads: int = 4
    ):
        super().__init__()
        self.num_head_types = num_head_types
        self.hidden_dim = hidden_dim

        # Multi-head attention to attend across head types
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attn_heads,
            batch_first=True,
            dropout=0.0,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, head_features_list: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Args:
            head_features_list: List of tensors, each (B, T, hidden_dim)
                               One tensor per head type (buttons, main_stick, c_stick, shoulder)

        Returns:
            List of tensors with same shape, but now informed by other heads
        """
        # Stack: (B, T, num_head_types, hidden_dim)
        stacked = torch.stack(head_features_list, dim=2)
        B, T, H, D = stacked.shape
        assert (
            H == self.num_head_types
        ), f"Expected {self.num_head_types} heads, got {H}"

        # Reshape: treat each timestep independently, attend across head types
        # (B*T, num_head_types, hidden_dim)
        x = stacked.reshape(B * T, H, D)

        # Self-attention across head types
        attended, _ = self.cross_attn(x, x, x, need_weights=False)

        # Residual + norm
        attended = self.norm(x + attended)

        # Reshape back: (B, T, num_head_types, hidden_dim)
        attended = attended.reshape(B, T, H, D)

        # Unstack back to list
        return [attended[:, :, i] for i in range(H)]
