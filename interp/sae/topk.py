"""
TopK Sparse Autoencoder for interpretability.

Uses hard TopK sparsity instead of L1 penalty, which provides:
- Exact sparsity control (no L1 coefficient tuning)
- No shrinkage problem (L1 shrinks all activations)
- Better suited for already-sparse ReLU² activations
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class SAEOutput:
    """Output from SAE forward pass."""

    reconstruction: Tensor  # [batch, input_dim]
    latents: Tensor  # [batch, hidden_dim] (sparse)
    latent_indices: Tensor  # [batch, k] (indices of active features)
    latent_values: Tensor  # [batch, k] (values of active features)
    reconstruction_loss: Tensor  # scalar


class TopKSparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder with TopK activation sparsity.

    Architecture:
        encode: x -> (x - b_dec) @ W_enc + b_enc -> pre_acts
        topk:   pre_acts -> keep only top-k, apply ReLU
        decode: sparse_acts @ W_dec + b_dec -> reconstruction

    The decoder columns are optionally normalized to unit length after
    each optimization step, which helps with feature interpretability.

    Args:
        input_dim: Dimension of input activations
        expansion_factor: Ratio of SAE features to input dimension
        k: Number of active features per input (sparsity level)
        normalize_decoder: Whether to normalize decoder columns to unit length
    """

    def __init__(
        self,
        input_dim: int,
        expansion_factor: int = 8,
        k: int = 32,
        normalize_decoder: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = input_dim * expansion_factor
        self.k = k
        self.normalize_decoder = normalize_decoder

        # Encoder weights
        self.W_enc = nn.Parameter(torch.empty(input_dim, self.hidden_dim))
        self.b_enc = nn.Parameter(torch.zeros(self.hidden_dim))

        # Decoder weights (tied bias with encoder preprocessing)
        self.W_dec = nn.Parameter(torch.empty(self.hidden_dim, input_dim))
        self.b_dec = nn.Parameter(torch.zeros(input_dim))

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with appropriate scaling."""
        # Kaiming initialization for encoder
        nn.init.kaiming_uniform_(self.W_enc, nonlinearity="relu")

        # Initialize decoder as transpose of encoder (approximate inverse)
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.T.clone())

        # Normalize decoder columns if requested
        if self.normalize_decoder:
            self._normalize_decoder()

    @torch.no_grad()
    def _normalize_decoder(self) -> None:
        """Normalize decoder columns to unit length."""
        norms = self.W_dec.norm(dim=1, keepdim=True)
        self.W_dec.div_(norms.clamp(min=1e-8))

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Encode input to sparse latent representation.

        Args:
            x: Input activations [batch, input_dim]

        Returns:
            latents: Sparse latent activations [batch, hidden_dim]
            indices: Indices of active features [batch, k]
            values: Values of active features [batch, k]
        """
        # Subtract decoder bias and project
        pre_acts = (x - self.b_dec) @ self.W_enc + self.b_enc

        # Get top-k activations
        topk_values, topk_indices = torch.topk(pre_acts, k=self.k, dim=-1)

        # Apply ReLU to top-k values (may zero some if negative)
        topk_values = F.relu(topk_values)

        # Create sparse activation tensor
        latents = torch.zeros_like(pre_acts)
        latents.scatter_(dim=-1, index=topk_indices, src=topk_values)

        return latents, topk_indices, topk_values

    def decode(self, latents: Tensor) -> Tensor:
        """
        Decode sparse latents back to input space.

        Args:
            latents: Sparse latent activations [batch, hidden_dim]

        Returns:
            Reconstruction [batch, input_dim]
        """
        return latents @ self.W_dec + self.b_dec

    def forward(self, x: Tensor) -> SAEOutput:
        """
        Full forward pass: encode -> decode with loss computation.

        Args:
            x: Input activations [batch, input_dim]

        Returns:
            SAEOutput with reconstruction, latents, and losses
        """
        latents, indices, values = self.encode(x)
        reconstruction = self.decode(latents)

        # Compute reconstruction loss (MSE)
        reconstruction_loss = F.mse_loss(reconstruction, x)

        return SAEOutput(
            reconstruction=reconstruction,
            latents=latents,
            latent_indices=indices,
            latent_values=values,
            reconstruction_loss=reconstruction_loss,
        )

    def post_step_hook(self) -> None:
        """Call after each optimizer step to maintain decoder normalization."""
        if self.normalize_decoder:
            self._normalize_decoder()

    def get_feature_activations(
        self, x: Tensor, feature_idx: int
    ) -> Tensor:
        """
        Get activation of a specific feature across inputs.

        Args:
            x: Input activations [batch, input_dim]
            feature_idx: Index of feature to examine

        Returns:
            Feature activations [batch]
        """
        latents, _, _ = self.encode(x)
        return latents[:, feature_idx]

    def get_decoder_direction(self, feature_idx: int) -> Tensor:
        """
        Get the decoder direction for a specific feature.

        This is the direction in activation space that the feature represents.

        Args:
            feature_idx: Index of feature

        Returns:
            Decoder direction [input_dim]
        """
        return self.W_dec[feature_idx].clone()

    def compute_feature_stats(self, x: Tensor) -> dict:
        """
        Compute statistics about feature activations.

        Args:
            x: Input activations [batch, input_dim]

        Returns:
            Dictionary with activation statistics
        """
        output = self.forward(x)

        # Count how often each feature fires
        active_mask = output.latents > 0
        feature_counts = active_mask.sum(dim=0)

        # Mean activation when active
        mean_when_active = torch.zeros(self.hidden_dim, device=x.device)
        for i in range(self.hidden_dim):
            mask = active_mask[:, i]
            if mask.any():
                mean_when_active[i] = output.latents[mask, i].mean()

        return {
            "feature_counts": feature_counts,
            "feature_frequencies": feature_counts.float() / len(x),
            "mean_when_active": mean_when_active,
            "reconstruction_loss": output.reconstruction_loss.item(),
            "avg_active_features": active_mask.float().sum(dim=1).mean().item(),
        }

    def find_dead_features(self, threshold: int = 0) -> Tensor:
        """
        Find features that never activated in stats computation.

        Should be called after compute_feature_stats with a large batch.

        Args:
            threshold: Count threshold below which feature is considered dead

        Returns:
            Indices of dead features
        """
        # This is a placeholder - actual tracking happens in trainer
        raise NotImplementedError(
            "Use SAETrainer.get_dead_features() for proper dead feature tracking"
        )

    def save(self, path: str) -> None:
        """Save SAE checkpoint."""
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "input_dim": self.input_dim,
                    "hidden_dim": self.hidden_dim,
                    "k": self.k,
                    "normalize_decoder": self.normalize_decoder,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> "TopKSparseAutoencoder":
        """Load SAE from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint["config"]

        sae = cls(
            input_dim=config["input_dim"],
            expansion_factor=config["hidden_dim"] // config["input_dim"],
            k=config["k"],
            normalize_decoder=config["normalize_decoder"],
        )
        sae.load_state_dict(checkpoint["state_dict"])

        if device is not None:
            sae = sae.to(device)

        return sae


__all__ = ["TopKSparseAutoencoder", "SAEOutput"]
