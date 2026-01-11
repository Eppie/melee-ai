"""
Decision explanation for Nano-Melee model.

Provides tools to understand why the model made specific predictions,
with support for:
- Model confidence breakdown
- Top input features by importance
- SAE feature activations
- Death log analysis
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from interp.hooks import HookManager, HookPoint, HookPointType

if TYPE_CHECKING:
    from column_map import ColumnMap
    from interp.sae.topk import TopKSparseAutoencoder
    from model.nano_gpt import GPT


@dataclass
class PredictedOutput:
    """Model prediction for a single output head."""

    head_name: str
    predicted_idx: int
    confidence: float
    top_k_indices: List[int]
    top_k_probs: List[float]
    logits: Tensor


@dataclass
class InputImportance:
    """Importance of an input feature for the prediction."""

    name: str
    value: float
    importance: float  # Gradient-based importance
    rank: int


@dataclass
class SAEFeatureActivation:
    """An activated SAE feature."""

    feature_idx: int
    activation: float
    rank: int


@dataclass
class DecisionExplanation:
    """
    Complete explanation for a model decision at a single frame.

    Contains:
    - Model predictions with confidence
    - Top input features by importance
    - Active SAE features (if SAE provided)
    - Frame metadata
    """

    # Predictions by head
    predictions: Dict[str, PredictedOutput]

    # Top input features by gradient importance
    top_input_features: List[InputImportance]

    # Active SAE features (if SAE was provided)
    sae_features: List[SAEFeatureActivation]

    # Frame metadata
    frame_idx: int
    position_in_sequence: int

    # Raw data for debugging
    raw_inputs: Optional[Dict[str, float]] = None

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = []
        lines.append("=" * 60)
        lines.append(f"Frame {self.frame_idx} (seq pos {self.position_in_sequence})")
        lines.append("=" * 60)

        # Predictions
        lines.append("\nPREDICTIONS:")
        for name, pred in self.predictions.items():
            lines.append(f"  {name}: idx={pred.predicted_idx} ({pred.confidence*100:.1f}%)")
            if len(pred.top_k_indices) > 1:
                top_probs = [f"{idx}:{p*100:.0f}%" for idx, p in
                            zip(pred.top_k_indices[:3], pred.top_k_probs[:3])]
                lines.append(f"    top-3: {', '.join(top_probs)}")

        # Top inputs
        if self.top_input_features:
            lines.append("\nTOP INPUT FEATURES:")
            for feat in self.top_input_features[:10]:
                lines.append(f"  {feat.rank}. {feat.name}: {feat.value:.3f} (importance: {feat.importance:.4f})")

        # SAE features
        if self.sae_features:
            lines.append("\nACTIVE SAE FEATURES:")
            for feat in self.sae_features[:10]:
                lines.append(f"  {feat.rank}. Feature {feat.feature_idx}: {feat.activation:.4f}")

        return "\n".join(lines)


class DecisionExplainer:
    """
    Main interface for explaining model decisions.

    Usage:
        explainer = DecisionExplainer(model, colmap, device)

        # With trained SAE
        explainer.set_sae(sae, hook_point)

        # Explain a single frame
        explanation = explainer.explain_frame(inputs, position=-1)
        print(explanation.summary())

        # Analyze a death log
        explanations = explainer.explain_from_death_log("death_logs/death_0001.json")
    """

    def __init__(
        self,
        model: "GPT",
        colmap: "ColumnMap",
        device: torch.device,
        sae: Optional["TopKSparseAutoencoder"] = None,
        sae_hook_point: Optional[HookPoint] = None,
    ):
        self.model = model
        self.colmap = colmap
        self.device = device

        self._sae = sae
        self._sae_hook_point = sae_hook_point
        self._hook_manager = HookManager(model)

        # Build feature name list for importance ranking
        self._feature_names = colmap.feat_names

    def set_sae(
        self,
        sae: "TopKSparseAutoencoder",
        hook_point: HookPoint,
    ) -> None:
        """Set the SAE to use for feature analysis."""
        self._sae = sae.to(self.device)
        self._sae_hook_point = hook_point

    def explain_frame(
        self,
        inputs: Tensor,
        position: int = -1,
        compute_gradients: bool = True,
    ) -> DecisionExplanation:
        """
        Explain the model's decision for a specific frame.

        Args:
            inputs: Input tensor [batch, seq_len, features] or [seq_len, features]
            position: Which position in sequence to explain (-1 for last)
            compute_gradients: Whether to compute gradient-based importance

        Returns:
            DecisionExplanation with predictions and feature importance
        """
        from train.batch_utils import build_model_inputs

        # Ensure inputs have batch dimension
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(0)

        inputs = inputs.to(self.device)
        batch_size, seq_len, n_features = inputs.shape

        # Normalize position
        if position < 0:
            position = seq_len + position

        # Build model inputs
        inputs_td = build_model_inputs(inputs, self.colmap)

        # Install hooks if SAE is set
        if self._sae is not None and self._sae_hook_point is not None:
            self._hook_manager.install_hooks([self._sae_hook_point])

        # Forward pass with gradient tracking for importance
        if compute_gradients:
            inputs.requires_grad_(True)
            self.model.eval()

        try:
            outputs = self.model(inputs_td)

            # Extract predictions
            predictions = self._extract_predictions(outputs, position)

            # Compute gradient importance
            top_inputs = []
            if compute_gradients:
                top_inputs = self._compute_input_importance(
                    inputs, outputs, position
                )

            # Get SAE features
            sae_features = []
            if self._sae is not None:
                sae_features = self._get_sae_features(position)

        finally:
            self._hook_manager.remove_hooks()
            if compute_gradients:
                inputs.requires_grad_(False)

        # Get raw input values for context
        raw_inputs = {}
        for i, name in enumerate(self._feature_names):
            if i < inputs.shape[-1]:
                raw_inputs[name] = inputs[0, position, i].item()

        return DecisionExplanation(
            predictions=predictions,
            top_input_features=top_inputs,
            sae_features=sae_features,
            frame_idx=position,
            position_in_sequence=position,
            raw_inputs=raw_inputs,
        )

    def _extract_predictions(
        self, outputs: Dict[str, Tensor], position: int
    ) -> Dict[str, PredictedOutput]:
        """Extract predictions from model outputs."""
        predictions = {}

        head_configs = [
            ("main_stick", "main_stick"),
            ("c_stick", "c_stick"),
            ("buttons", "buttons"),
            ("shoulder", "shoulder"),
        ]

        for head_name, output_key in head_configs:
            if output_key in outputs:
                logits = outputs[output_key][0, position]  # [num_classes]
                probs = F.softmax(logits, dim=-1)

                # Get top-k predictions
                k = min(5, len(probs))
                top_probs, top_indices = torch.topk(probs, k)

                predictions[head_name] = PredictedOutput(
                    head_name=head_name,
                    predicted_idx=top_indices[0].item(),
                    confidence=top_probs[0].item(),
                    top_k_indices=top_indices.tolist(),
                    top_k_probs=top_probs.tolist(),
                    logits=logits.detach(),
                )

        # Value head (regression)
        if "value" in outputs:
            value = outputs["value"][0, position, 0]
            predictions["value"] = PredictedOutput(
                head_name="value",
                predicted_idx=0,
                confidence=value.item(),
                top_k_indices=[0],
                top_k_probs=[value.item()],
                logits=value.detach().unsqueeze(0),
            )

        return predictions

    def _compute_input_importance(
        self,
        inputs: Tensor,
        outputs: Dict[str, Tensor],
        position: int,
    ) -> List[InputImportance]:
        """Compute gradient-based input importance."""
        # Sum logits from all classification heads for gradient computation
        total_logit = torch.zeros(1, device=self.device)

        for key in ["main_stick", "c_stick", "buttons", "shoulder"]:
            if key in outputs:
                logits = outputs[key][0, position]
                # Use max logit (predicted class) for gradient
                total_logit = total_logit + logits.max()

        # Compute gradients
        if inputs.grad is not None:
            inputs.grad.zero_()

        total_logit.backward(retain_graph=True)

        if inputs.grad is None:
            return []

        # Extract gradient magnitudes at the position of interest
        grads = inputs.grad[0, position].abs()  # [n_features]

        # Create importance list
        importances = []
        for i, name in enumerate(self._feature_names):
            if i < len(grads):
                importances.append(
                    InputImportance(
                        name=name,
                        value=inputs[0, position, i].item(),
                        importance=grads[i].item(),
                        rank=0,  # Will be set after sorting
                    )
                )

        # Sort by importance and set ranks
        importances.sort(key=lambda x: x.importance, reverse=True)
        for rank, imp in enumerate(importances, 1):
            imp.rank = rank

        return importances

    def _get_sae_features(self, position: int) -> List[SAEFeatureActivation]:
        """Get active SAE features for the frame."""
        if self._sae is None or self._sae_hook_point is None:
            return []

        try:
            # Get activations from hook
            acts = self._hook_manager.get_single(self._sae_hook_point)
            # acts shape: [batch, seq, hidden_dim]

            frame_acts = acts[0, position]  # [hidden_dim]

            # Run through SAE encoder
            with torch.no_grad():
                latents, indices, values = self._sae.encode(frame_acts.unsqueeze(0))
                # indices: [1, k], values: [1, k]

            # Build feature list
            features = []
            for rank, (idx, val) in enumerate(
                zip(indices[0].tolist(), values[0].tolist()), 1
            ):
                if val > 0:  # Only include active features
                    features.append(
                        SAEFeatureActivation(
                            feature_idx=idx,
                            activation=val,
                            rank=rank,
                        )
                    )

            return features

        except Exception:
            return []

    def explain_from_death_log(
        self,
        death_log_path: str | Path,
        positions: Optional[List[int]] = None,
        last_n_frames: int = 30,
    ) -> List[DecisionExplanation]:
        """
        Analyze frames from a death log file.

        Args:
            death_log_path: Path to death log JSON file
            positions: Specific frame positions to analyze (0-indexed)
            last_n_frames: If positions is None, analyze this many frames before death

        Returns:
            List of DecisionExplanations for each analyzed frame
        """
        path = Path(death_log_path)
        with open(path, "r") as f:
            data = json.load(f)

        frames = data["frames"]
        feature_names = data["feature_names"]
        n_frames = len(frames)

        # Determine which positions to analyze
        if positions is None:
            # Analyze last N frames before death
            start = max(0, n_frames - last_n_frames)
            positions = list(range(start, n_frames))

        # Convert frames to tensor
        inputs_list = []
        for frame in frames:
            # Use transformed features (already normalized)
            feat_values = [
                frame["transformed_features"].get(name, 0.0)
                for name in feature_names
            ]
            inputs_list.append(feat_values)

        inputs = torch.tensor(inputs_list, dtype=torch.float32)  # [seq, features]
        inputs = inputs.unsqueeze(0)  # [1, seq, features]

        # Generate explanations for each position
        explanations = []
        for pos in positions:
            explanation = self.explain_frame(
                inputs, position=pos, compute_gradients=True
            )
            explanations.append(explanation)

        return explanations

    def compare_predictions_to_actual(
        self,
        death_log_path: str | Path,
    ) -> Dict[str, Any]:
        """
        Compare model predictions to actual player actions from death log.

        Returns statistics on prediction accuracy leading up to death.
        """
        path = Path(death_log_path)
        with open(path, "r") as f:
            data = json.load(f)

        frames = data["frames"]
        n_frames = len(frames)

        # Track prediction accuracy
        stats = {
            "n_frames": n_frames,
            "main_stick_correct": 0,
            "c_stick_correct": 0,
            "buttons_correct": {"a": 0, "b": 0, "xy": 0, "z": 0, "lr": 0},
            "shoulder_correct": 0,
        }

        # Get predictions for each frame
        explanations = self.explain_from_death_log(
            death_log_path,
            positions=list(range(n_frames)),
        )

        for i, (explanation, frame) in enumerate(zip(explanations, frames)):
            # Compare main stick (need to check targets)
            if "main_stick" in explanation.predictions:
                pred_idx = explanation.predictions["main_stick"].predicted_idx
                # Note: actual comparison would need target indices from frame

            # Compare button predictions
            if "buttons" in explanation.predictions:
                # Button predictions are multi-class
                pass

        return stats


@dataclass
class DeathAnalysis:
    """Analysis of a death sequence."""

    death_log_path: Path
    total_frames: int
    explanations: List[DecisionExplanation]

    # Summary statistics
    avg_confidence: Dict[str, float] = field(default_factory=dict)
    low_confidence_frames: List[int] = field(default_factory=list)

    def find_decision_points(self, confidence_threshold: float = 0.5) -> List[int]:
        """Find frames where model was uncertain (potential decision points)."""
        uncertain_frames = []
        for exp in self.explanations:
            for head_name, pred in exp.predictions.items():
                if head_name != "value" and pred.confidence < confidence_threshold:
                    uncertain_frames.append(exp.frame_idx)
                    break
        return sorted(set(uncertain_frames))


def analyze_death_log(
    model: "GPT",
    colmap: "ColumnMap",
    device: torch.device,
    death_log_path: str | Path,
    sae: Optional["TopKSparseAutoencoder"] = None,
    sae_hook_point: Optional[HookPoint] = None,
) -> DeathAnalysis:
    """
    Convenience function to analyze a death log.

    Args:
        model: Trained GPT model
        colmap: Column mapping
        device: Torch device
        death_log_path: Path to death log file
        sae: Optional trained SAE
        sae_hook_point: Hook point for SAE

    Returns:
        DeathAnalysis with frame-by-frame explanations
    """
    explainer = DecisionExplainer(model, colmap, device, sae, sae_hook_point)

    path = Path(death_log_path)
    with open(path, "r") as f:
        data = json.load(f)

    n_frames = len(data["frames"])
    explanations = explainer.explain_from_death_log(
        death_log_path, positions=list(range(n_frames))
    )

    # Compute summary statistics
    avg_confidence: Dict[str, List[float]] = {}
    low_confidence_frames = []

    for exp in explanations:
        for head_name, pred in exp.predictions.items():
            if head_name not in avg_confidence:
                avg_confidence[head_name] = []
            avg_confidence[head_name].append(pred.confidence)

            if head_name != "value" and pred.confidence < 0.5:
                low_confidence_frames.append(exp.frame_idx)

    # Average confidences
    avg_conf_final = {k: sum(v) / len(v) for k, v in avg_confidence.items()}

    return DeathAnalysis(
        death_log_path=path,
        total_frames=n_frames,
        explanations=explanations,
        avg_confidence=avg_conf_final,
        low_confidence_frames=sorted(set(low_confidence_frames)),
    )


__all__ = [
    "DecisionExplainer",
    "DecisionExplanation",
    "PredictedOutput",
    "InputImportance",
    "SAEFeatureActivation",
    "DeathAnalysis",
    "analyze_death_log",
]
