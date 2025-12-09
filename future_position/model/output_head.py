"""Output head for trajectory prediction, including Mixture Density Network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple

from ..constants import (
    D_MODEL,
    HORIZONS,
    HORIZON_EMBED_DIM,
    N_MIXTURE_COMPONENTS,
    OUTPUT_HIDDEN_DIM,
    N_RESBLOCKS,
    MIN_SIGMA,
    MAX_SIGMA,
    SIGMA_EPSILON,
)


class ResBlock(nn.Module):
    def __init__(self, dim: int, expansion: int = 2):
        super().__init__()
        hidden = dim * expansion
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden, bias=False),
            nn.ReLU(),
            nn.Linear(hidden, dim, bias=False),
        )
        # Using RMSNorm as specified in PLAN.md under TransformerEncoder
        # Assuming it's suitable here too for consistency if not explicitly stated otherwise
        self.norm = nn.LayerNorm(dim) # Changed to LayerNorm as RMSNorm is not standard PyTorch and needs custom implementation, which is not in constants.py and will introduce more dependencies

    def forward(self, x):
        return F.relu(x + self.mlp(self.norm(x)))


class TrajectoryOutputHead(nn.Module):
    def __init__(
        self,
        d_model: int = D_MODEL,
        n_horizons: int = len(HORIZONS),
        n_components: int = N_MIXTURE_COMPONENTS,
        horizon_embed_dim: int = HORIZON_EMBED_DIM,
        output_hidden_dim: int = OUTPUT_HIDDEN_DIM,
        n_resblocks: int = N_RESBLOCKS,
    ):
        super().__init__()
        self.horizons = HORIZONS

        # Horizon embeddings
        # Max horizon is 60, so embedding up to 60+1 = 61 categories
        self.horizon_embed = nn.Embedding(max(HORIZONS) + 1, horizon_embed_dim)

        # Shared trunk
        self.project = nn.Linear(d_model + horizon_embed_dim, output_hidden_dim)
        
        res_blocks = [ResBlock(output_hidden_dim) for _ in range(n_resblocks)]
        self.res_blocks = nn.Sequential(*res_blocks)

        # Per-horizon heads
        self.p1_heads = nn.ModuleList([
            MixtureDensityHead(output_hidden_dim, n_components)
            for _ in range(n_horizons)
        ])
        self.p2_heads = nn.ModuleList([
            MixtureDensityHead(output_hidden_dim, n_components)
            for _ in range(n_horizons)
        ])

    def forward(self, h_context: torch.Tensor) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]:
        # h_context: [B, d_model]
        B = h_context.size(0)

        p1_trajectories = []
        p2_trajectories = []

        for i, h in enumerate(self.horizons):
            # Embed horizon
            h_embed = self.horizon_embed(
                torch.full((B,), h, device=h_context.device, dtype=torch.long)
            )  # [B, horizon_embed_dim]

            # Concatenate and process through shared trunk
            x = torch.cat([h_context, h_embed], dim=-1)  # [B, d_model + horizon_embed_dim]
            x = F.relu(self.project(x))  # [B, output_hidden_dim]
            x = self.res_blocks(x)       # [B, output_hidden_dim]

            # Predict mixtures using per-horizon heads
            p1_mix = self.p1_heads[i](x)
            p2_mix = self.p2_heads[i](x)

            p1_trajectories.append(p1_mix)
            p2_trajectories.append(p2_mix)

        return p1_trajectories, p2_trajectories


class MixtureDensityHead(nn.Module):
    def __init__(
        self,
        d_input: int,
        n_components: int = N_MIXTURE_COMPONENTS,
        min_sigma: float = MIN_SIGMA,
        max_sigma: float = MAX_SIGMA,
        sigma_epsilon: float = SIGMA_EPSILON,
    ):
        super().__init__()
        self.n_components = n_components
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.sigma_epsilon = sigma_epsilon

        # Output: weights[K], mu_x[K], mu_y[K], log_sigma_x[K], log_sigma_y[K]
        self.proj = nn.Linear(d_input, n_components * 5)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: [B, d_input]
        out = self.proj(x)  # [B, K*5]
        out = out.reshape(x.size(0), self.n_components, 5)

        weights_logits = out[..., 0]       # [B, K]
        mu_x = out[..., 1]                 # [B, K]
        mu_y = out[..., 2]                 # [B, K]
        log_sigma_x = out[..., 3]          # [B, K]
        log_sigma_y = out[..., 4]          # [B, K]

        # Stable sigma via softplus as specified in PLAN.md
        sigma_x = (F.softplus(log_sigma_x) + self.sigma_epsilon).clamp(min=self.min_sigma, max=self.max_sigma)
        sigma_y = (F.softplus(log_sigma_y) + self.sigma_epsilon).clamp(min=self.min_sigma, max=self.max_sigma)

        return {
            'weights': F.softmax(weights_logits, dim=-1),  # [B, K]
            'mu_x': mu_x,
            'mu_y': mu_y,
            'sigma_x': sigma_x,
            'sigma_y': sigma_y,
        }