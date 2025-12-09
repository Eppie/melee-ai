"""Main model for future position prediction."""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple

from ..constants import (
    D_MODEL,
    N_LAYERS,
    N_HEADS,
    MLP_RATIO,
    TOTAL_FEATURE_DIM,
    CONTEXT_LENGTH,
    NUM_CONTINUOUS_FEATURES,
)
from .embeddings import FeatureEmbedder
from .encoder import TransformerEncoder
from .output_head import TrajectoryOutputHead


class FuturePositionPredictor(nn.Module):
    """Predicts future player positions using a Transformer encoder and MDN output heads."""

    def __init__(
        self,
        input_feature_dim: int = TOTAL_FEATURE_DIM,
        n_continuous_features: int = NUM_CONTINUOUS_FEATURES,
        d_model: int = D_MODEL,
        n_layers: int = N_LAYERS,
        n_heads: int = N_HEADS,
        mlp_ratio: int = MLP_RATIO,
        dropout: float = 0.0,
        context_length: int = CONTEXT_LENGTH,
    ):
        super().__init__()
        # input_feature_dim reflects the total feature width in the NPZ (categorical + continuous)
        # n_continuous_features should exclude categorical slots to keep projection shapes aligned.
        self.feature_embedder = FeatureEmbedder(
            d_model=d_model,
            n_continuous_features=n_continuous_features,
        )
        self.encoder = TransformerEncoder(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            causal=False, # For a sequence of past frames, not necessarily causal here
        )
        self.output_head = TrajectoryOutputHead(d_model=d_model)

    def forward(self, x: torch.Tensor) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]:
        """Forward pass of the FuturePositionPredictor.

        Args:
            x: Input feature tensor [B, T, input_feature_dim] representing a sequence of frames.

        Returns:
            Tuple of lists, each containing dictionaries of mixture parameters for P1 and P2
            across different horizons.
        """
        # Embed the input features
        # x_embedded: [B, T, d_model]
        x_embedded = self.feature_embedder(x)

        # Encode the sequence of embedded features
        # h_context: [B, T, d_model]
        # We need the representation of the last frame for prediction
        encoded_features = self.encoder(x_embedded)
        h_context = encoded_features[:, -1, :] # Take the last timestep's output as context [B, d_model]

        # Predict future trajectories using the output head
        p1_trajectories, p2_trajectories = self.output_head(h_context)

        return p1_trajectories, p2_trajectories
