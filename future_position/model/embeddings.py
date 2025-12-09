"""Embedding layers for categorical features."""

import torch
import torch.nn as nn
from typing import Dict

from ..constants import (
    N_CHARACTERS,
    CHARACTER_EMBED_DIM,
    MAX_ACTION_STATES,
    ACTION_STATE_EMBED_DIM,
    N_STAGES,
    STAGE_EMBED_DIM,
    HORIZON_EMBED_DIM,
    D_MODEL,
    NUM_CONTINUOUS_FEATURES, # Added
    P1_CHAR_ID_GLOBAL_IDX,   # Added
    P2_CHAR_ID_GLOBAL_IDX,   # Added
    STAGE_ID_GLOBAL_IDX,     # Added
    P1_ACTION_STATE_GLOBAL_IDX, # Added
    P2_ACTION_STATE_GLOBAL_IDX, # Added
)


class FeatureEmbedder(nn.Module):
    """Embeds categorical features and projects to d_model.

    Handles:
    - Character ID embeddings
    - Action state ID embeddings
    - Stage ID embeddings
    - Continuous feature concatenation
    - Linear projection to d_model
    """

    def __init__(
        self,
        d_model: int = D_MODEL,
        character_embed_dim: int = CHARACTER_EMBED_DIM,
        action_state_embed_dim: int = ACTION_STATE_EMBED_DIM,
        stage_embed_dim: int = STAGE_EMBED_DIM,
        n_continuous_features: int = NUM_CONTINUOUS_FEATURES,
    ):
        """Initialize embeddings.

        Args:
            d_model: Output dimension after projection
            character_embed_dim: Character embedding dimension
            action_state_embed_dim: Action state embedding dimension
            stage_embed_dim: Stage embedding dimension
            n_continuous_features: Number of continuous features (after removing categoricals)
        """
        super().__init__()
        # Embedding layers
        self.character_embed = nn.Embedding(N_CHARACTERS, character_embed_dim)
        self.action_state_embed = nn.Embedding(MAX_ACTION_STATES, action_state_embed_dim)
        self.stage_embed = nn.Embedding(N_STAGES, stage_embed_dim)

        # Compute total input dimension after embedding
        total_embedded_dim = (
            2 * character_embed_dim  # P1 and P2 characters
            + 2 * action_state_embed_dim  # P1 and P2 action states
            + stage_embed_dim             # Stage ID
        )
        total_input_dim = n_continuous_features + total_embedded_dim
        
        # Linear projection to d_model
        self.projection = nn.Linear(total_input_dim, d_model)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Embed and project features.

        Args:
            features: [B, T, TOTAL_FEATURE_DIM] raw features

        Returns:
            [B, T, d_model] embedded and projected features

        Notes:
            - Extracts categorical indices from features
            - Embeds categoricals
            - Concatenates with continuous features
            - Projects to d_model
        """
        B, T, _ = features.shape

        # Extract categorical IDs (ensure they are long tensors for embedding lookup)
        p1_char_id = features[:, :, P1_CHAR_ID_GLOBAL_IDX].long()
        p2_char_id = features[:, :, P2_CHAR_ID_GLOBAL_IDX].long()
        stage_id = features[:, :, STAGE_ID_GLOBAL_IDX].long()
        p1_action_state = features[:, :, P1_ACTION_STATE_GLOBAL_IDX].long()
        p2_action_state = features[:, :, P2_ACTION_STATE_GLOBAL_IDX].long()

        # Embed categorical features
        p1_char_embedded = self.character_embed(p1_char_id)
        p2_char_embedded = self.character_embed(p2_char_id)
        # Clamp stage IDs to embedding size to avoid OOB if new stages appear
        stage_id = stage_id.clamp(min=0, max=self.stage_embed.num_embeddings - 1)
        stage_embedded = self.stage_embed(stage_id)
        p1_action_state_embedded = self.action_state_embed(p1_action_state)
        p2_action_state_embedded = self.action_state_embed(p2_action_state)

        # Separate continuous features
        # Create a mask to identify categorical features in the original tensor
        categorical_indices = sorted([
            P1_ACTION_STATE_GLOBAL_IDX,
            P2_ACTION_STATE_GLOBAL_IDX,
            STAGE_ID_GLOBAL_IDX,
            P1_CHAR_ID_GLOBAL_IDX,
            P2_CHAR_ID_GLOBAL_IDX,
        ])
        
        all_indices = torch.arange(features.shape[2], device=features.device)
        continuous_mask = torch.ones(features.shape[2], dtype=torch.bool, device=features.device)
        continuous_mask[categorical_indices] = False
        
        continuous_features = features[:, :, continuous_mask]

        # Concatenate continuous features with embeddings
        combined_features = torch.cat([
            continuous_features,
            p1_char_embedded,
            p2_char_embedded,
            stage_embedded,
            p1_action_state_embedded,
            p2_action_state_embedded,
        ], dim=-1)

        # Project to d_model
        return self.projection(combined_features)


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional embeddings for horizons.

    Uses sin/cos functions of different frequencies for position encoding.
    """

    def __init__(self, max_len: int = 61, embed_dim: int = HORIZON_EMBED_DIM):
        """Initialize sinusoidal embeddings.

        Args:
            max_len: Maximum sequence length (61 for horizons 0-60)
            embed_dim: Embedding dimension
        """
        super().__init__()
        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer so it's not a trained parameter but moved with the model
        self.register_buffer('pe', pe)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Get embeddings for positions.

        Args:
            positions: [B] or [B, T] integer positions

        Returns:
            [B, embed_dim] or [B, T, embed_dim] embeddings
        """
        return self.pe[positions]


class RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embeddings (RoPE) for Transformer.

    Applies rotations to query and key vectors based on position.
    """

    def __init__(self, dim: int, max_len: int = 1024):
        """Initialize RoPE.

        Args:
            dim: Dimension per head (should be even)
            max_len: Maximum sequence length
        """
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Pre-compute cos and sin terms
        t = torch.arange(max_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> tuple:
        """Apply rotary embeddings to queries and keys.

        Args:
            q: [B, n_heads, T, head_dim] queries
            k: [B, n_heads, T, head_dim] keys
            seq_len: Sequence length

        Returns:
            (q_rot, k_rot) with rotary embeddings applied
        """
        # x: [B, n_heads, T, head_dim]
        # self.cos_cached and self.sin_cached are [1, 1, max_len, head_dim]
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]

        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
