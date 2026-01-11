"""
Feature importance analysis tools for the Nano-Melee interpretability toolkit.

Provides gradient-based and ablation-based methods to understand which
input features matter most for model predictions.

Usage:
    from interp.feature_importance import FeatureImportanceAnalyzer

    analyzer = FeatureImportanceAnalyzer(model, column_map, device)

    # Gradient-based importance on validation data
    importance = analyzer.compute_gradient_importance(val_dataset, n_samples=100)
    print(importance.summary())

    # Ablation-based importance
    ablation_results = analyzer.compute_ablation_importance(val_dataset, n_samples=50)

    # Compare P1 vs P2 feature importance
    comparison = analyzer.compare_player_importance(val_dataset)
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from column_map import ColumnMap
    from model.nano_gpt import GPT
    from window_dataset import WindowDataset


@dataclass
class FeatureImportanceResult:
    """Results from feature importance analysis."""

    # Per-feature importance scores
    feature_importance: Dict[str, float]

    # Grouped by category
    category_importance: Dict[str, float]

    # Analysis metadata
    n_samples: int
    method: str  # "gradient" or "ablation"

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = ["=" * 70]
        lines.append(f"FEATURE IMPORTANCE ANALYSIS ({self.method.upper()})")
        lines.append(f"Samples analyzed: {self.n_samples}")
        lines.append("=" * 70)

        # Sort features by importance
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )

        lines.append("\nTOP 20 MOST IMPORTANT FEATURES:")
        for i, (name, imp) in enumerate(sorted_features[:20], 1):
            lines.append(f"  {i:2d}. {name:40s}: {imp:.6f}")

        lines.append("\nBOTTOM 10 LEAST IMPORTANT FEATURES:")
        for i, (name, imp) in enumerate(sorted_features[-10:], 1):
            lines.append(f"  {i:2d}. {name:40s}: {imp:.6f}")

        lines.append("\nIMPORTANCE BY CATEGORY:")
        sorted_cats = sorted(
            self.category_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        for cat, imp in sorted_cats:
            lines.append(f"  {cat:20s}: {imp:.5f}")

        return "\n".join(lines)

    def top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top N most important features."""
        return sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n]


def categorize_feature(name: str) -> str:
    """Categorize a feature name into semantic groups."""
    if name == "stage":
        return "stage"
    if "_character" in name and "holding" not in name:
        return "character"
    if "_action" in name:
        return "action"
    if "position_x" in name or "position_y" in name:
        return "position"
    if "speed" in name:
        return "speed"
    if "percent" in name:
        return "percent"
    if "facing" in name:
        return "facing"
    if "on_ground" in name:
        return "on_ground"
    if "jumps" in name:
        return "jumps"
    if "is_" in name or "off_stage" in name or "l_cancel" in name:
        return "boolean_flags"
    if "main_stick" in name or "c_stick" in name:
        return "stick_inputs"
    if "button" in name:
        return "buttons"
    if "shoulder" in name:
        return "shoulder"
    return "other"


class FeatureImportanceAnalyzer:
    """
    Analyzes feature importance for model predictions.

    Supports two methods:
    - Gradient-based: Compute gradient magnitude w.r.t. each input feature
    - Ablation-based: Measure prediction change when features are zeroed
    """

    def __init__(
        self,
        model: "GPT",
        column_map: "ColumnMap",
        device: torch.device,
    ):
        self.model = model
        self.model.eval()
        self.model.to(device)
        self.column_map = column_map
        self.device = device
        self.feature_names = column_map.feat_names

    def compute_gradient_importance(
        self,
        dataset: "WindowDataset",
        n_samples: int = 100,
        output_head: str = "main_stick",
        verbose: bool = True,
    ) -> FeatureImportanceResult:
        """
        Compute gradient-based feature importance.

        For each sample, computes the gradient of the output w.r.t. inputs
        and averages the absolute gradient magnitudes across samples.

        Args:
            dataset: Validation dataset to sample from
            n_samples: Number of samples to analyze
            output_head: Which output head to compute gradients for
            verbose: Whether to print progress

        Returns:
            FeatureImportanceResult with per-feature and per-category importance
        """
        from train.batch_utils import build_model_inputs

        importance_accum = defaultdict(list)

        if verbose:
            print(f"Computing gradient importance over {n_samples} samples...")

        for i in range(min(n_samples, len(dataset))):
            try:
                # Sample spread throughout dataset
                idx = (i * 100) % len(dataset)
                sample = dataset[idx]
                inputs = sample["X"].unsqueeze(0).to(self.device)
                inputs.requires_grad_(True)

                # Build model inputs
                inputs_td = build_model_inputs(inputs, self.column_map)

                # Forward pass
                outputs = self.model(inputs_td)

                # Compute gradient w.r.t. specified output head
                logits = outputs[output_head]
                probs = torch.softmax(logits, dim=-1)

                # Use sum of max probs as scalar objective
                max_probs = probs.max(dim=-1).values.sum()
                max_probs.backward()

                # Extract gradients (average over sequence)
                if inputs.grad is not None:
                    grads = inputs.grad[0].abs().mean(dim=0)

                    for j, name in enumerate(self.feature_names):
                        if j < len(grads):
                            importance_accum[name].append(grads[j].item())

                # Clean up
                inputs.grad = None
                self.model.zero_grad()

                if verbose and (i + 1) % 20 == 0:
                    print(f"  Processed {i+1}/{n_samples} samples")

            except Exception as e:
                if verbose:
                    print(f"  Error on sample {i}: {e}")
                continue

        # Compute averages
        feature_importance = {
            name: np.mean(values) if values else 0.0
            for name, values in importance_accum.items()
        }

        # Group by category
        category_totals = defaultdict(float)
        for name, imp in feature_importance.items():
            cat = categorize_feature(name)
            category_totals[cat] += imp

        return FeatureImportanceResult(
            feature_importance=feature_importance,
            category_importance=dict(category_totals),
            n_samples=n_samples,
            method="gradient",
        )

    def compute_ablation_importance(
        self,
        dataset: "WindowDataset",
        n_samples: int = 50,
        output_head: str = "main_stick",
        verbose: bool = True,
    ) -> FeatureImportanceResult:
        """
        Compute ablation-based feature importance.

        For each sample, measures how much the output distribution changes
        when each feature is zeroed out.

        Args:
            dataset: Validation dataset to sample from
            n_samples: Number of samples to analyze
            output_head: Which output head to measure
            verbose: Whether to print progress

        Returns:
            FeatureImportanceResult with per-feature importance (KL divergence)
        """
        from train.batch_utils import build_model_inputs

        importance_accum = defaultdict(list)

        if verbose:
            print(f"Computing ablation importance over {n_samples} samples...")

        for i in range(min(n_samples, len(dataset))):
            try:
                idx = (i * 100) % len(dataset)
                sample = dataset[idx]
                inputs = sample["X"].unsqueeze(0).to(self.device)

                # Get baseline prediction
                with torch.no_grad():
                    inputs_td = build_model_inputs(inputs.clone(), self.column_map)
                    baseline_outputs = self.model(inputs_td)
                    baseline_probs = torch.softmax(
                        baseline_outputs[output_head][:, -1], dim=-1
                    )

                # Ablate each continuous feature and measure KL divergence
                for j, name in enumerate(self.feature_names):
                    # Skip categorical features (they go through embeddings)
                    if name in ["stage", "p1_action", "p2_action",
                               "p1_character", "p2_character"]:
                        importance_accum[name].append(0.0)
                        continue

                    ablated = inputs.clone()
                    ablated[:, :, j] = 0.0  # Zero out this feature

                    with torch.no_grad():
                        ablated_td = build_model_inputs(ablated, self.column_map)
                        ablated_outputs = self.model(ablated_td)
                        ablated_probs = torch.softmax(
                            ablated_outputs[output_head][:, -1], dim=-1
                        )

                    # KL divergence from baseline
                    kl_div = F.kl_div(
                        ablated_probs.log(),
                        baseline_probs,
                        reduction="batchmean"
                    ).item()
                    importance_accum[name].append(max(0, kl_div))

                if verbose and (i + 1) % 10 == 0:
                    print(f"  Processed {i+1}/{n_samples} samples")

            except Exception as e:
                if verbose:
                    print(f"  Error on sample {i}: {e}")
                continue

        # Compute averages
        feature_importance = {
            name: np.mean(values) if values else 0.0
            for name, values in importance_accum.items()
        }

        # Group by category
        category_totals = defaultdict(float)
        for name, imp in feature_importance.items():
            cat = categorize_feature(name)
            category_totals[cat] += imp

        return FeatureImportanceResult(
            feature_importance=feature_importance,
            category_importance=dict(category_totals),
            n_samples=n_samples,
            method="ablation",
        )

    def compare_player_importance(
        self,
        dataset: "WindowDataset",
        n_samples: int = 100,
    ) -> Dict[str, float]:
        """
        Compare total importance of P1 vs P2 features.

        Returns dict with keys "p1_total", "p2_total", and "ratio".
        """
        result = self.compute_gradient_importance(dataset, n_samples, verbose=False)

        p1_total = sum(
            imp for name, imp in result.feature_importance.items()
            if name.startswith("p1_")
        )
        p2_total = sum(
            imp for name, imp in result.feature_importance.items()
            if name.startswith("p2_")
        )

        return {
            "p1_total": p1_total,
            "p2_total": p2_total,
            "ratio": p1_total / p2_total if p2_total > 0 else float("inf"),
        }


__all__ = [
    "FeatureImportanceAnalyzer",
    "FeatureImportanceResult",
    "categorize_feature",
]
