"""Configuration for interpretability toolkit."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field


class SAEConfig(BaseModel):
    """Configuration for Sparse Autoencoder training."""

    expansion_factor: int = Field(
        default=8,
        ge=1,
        description="Ratio of SAE features to input dimension (e.g., 8 = 8x features)",
    )
    k: int = Field(
        default=32,
        ge=1,
        description="Number of active features per input (TopK sparsity)",
    )
    training_steps: int = Field(
        default=10000,
        ge=1,
        description="Number of training steps for SAE",
    )
    batch_size: int = Field(
        default=4096,
        ge=1,
        description="Batch size for SAE training",
    )
    lr: float = Field(
        default=3e-4,
        gt=0,
        description="Learning rate for SAE training",
    )
    normalize_decoder: bool = Field(
        default=True,
        description="Whether to normalize decoder columns to unit length",
    )
    dead_feature_threshold: int = Field(
        default=1000,
        ge=1,
        description="Steps without activation before considering a feature dead",
    )


class SamplingConfig(BaseModel):
    """Configuration for activation sampling."""

    stratified: bool = Field(
        default=True,
        description="Whether to use loss-stratified sampling",
    )
    num_loss_bins: int = Field(
        default=10,
        ge=2,
        description="Number of bins for loss stratification",
    )
    buffer_size: int = Field(
        default=100000,
        ge=1000,
        description="Number of activations to store in buffer",
    )
    high_loss_oversample: float = Field(
        default=3.0,
        ge=1.0,
        description="Factor to oversample high-loss frames",
    )


class InterpConfig(BaseModel):
    """Main configuration for interpretability toolkit."""

    # SAE settings
    sae: SAEConfig = Field(default_factory=SAEConfig)

    # Sampling settings
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)

    # Default layer for analysis
    default_layer: int = Field(
        default=4,
        ge=0,
        description="Default layer index for SAE training (middle layer recommended)",
    )

    # Layers to analyze for multi-layer analysis
    analysis_layers: Optional[List[int]] = Field(
        default=None,
        description="Layer indices for multi-layer analysis (None = auto: [0, mid, last])",
    )

    # Output paths
    sae_checkpoint_dir: Path = Field(
        default=Path("interp_checkpoints"),
        description="Directory to save trained SAE checkpoints",
    )
    reports_dir: Path = Field(
        default=Path("interp_reports"),
        description="Directory to save analysis reports",
    )

    # Feature interpretation
    top_features_to_show: int = Field(
        default=10,
        ge=1,
        description="Number of top SAE features to show in explanations",
    )
    top_inputs_to_show: int = Field(
        default=10,
        ge=1,
        description="Number of top input features to show in explanations",
    )
    correlation_threshold: float = Field(
        default=0.1,
        ge=0,
        le=1,
        description="Minimum correlation to consider meaningful for feature interpretation",
    )

    def get_analysis_layers(self, n_layers: int) -> List[int]:
        """Get layer indices for analysis, auto-computing if not specified."""
        if self.analysis_layers is not None:
            return [l for l in self.analysis_layers if 0 <= l < n_layers]
        # Default: input, middle, output
        return [0, n_layers // 2, n_layers - 1]


__all__ = ["InterpConfig", "SAEConfig", "SamplingConfig"]
