"""
Logit lens for understanding prediction evolution through layers.

The logit lens technique applies output heads to intermediate layer
activations to see when the model's predictions "crystallize" - at which
layer does it commit to a particular action?

This helps answer questions like:
- "When did the model decide to wavedash?"
- "At which layer does the prediction become confident?"
- "How does the prediction evolve through the network?"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from interp.hooks import HookManager, HookPoint, HookPointType

if TYPE_CHECKING:
    from column_map import ColumnMap
    from model.nano_gpt import GPT


@dataclass
class LayerPrediction:
    """Prediction from a single layer's activations."""

    layer_idx: int
    logits: Dict[str, Tensor]  # head_name -> logits
    probs: Dict[str, Tensor]  # head_name -> probabilities
    predicted_class: Dict[str, int]  # head_name -> argmax
    confidence: Dict[str, float]  # head_name -> max probability


@dataclass
class PredictionEvolution:
    """How predictions evolve through all layers."""

    layers: List[LayerPrediction]
    final_prediction: Dict[str, int]

    def get_crystallization_layer(
        self,
        head_name: str = "main_stick",
        threshold: float = 0.8,
    ) -> Optional[int]:
        """
        Find the layer where prediction becomes stable.

        Returns the first layer where:
        1. The predicted class matches the final prediction
        2. Confidence exceeds the threshold

        Args:
            head_name: Which output head to analyze
            threshold: Confidence threshold for "crystallized"

        Returns:
            Layer index where prediction crystallized, or None if never
        """
        final_class = self.final_prediction.get(head_name)
        if final_class is None:
            return None

        for layer in self.layers:
            if (layer.predicted_class.get(head_name) == final_class and
                layer.confidence.get(head_name, 0) >= threshold):
                return layer.layer_idx

        return None

    def get_confidence_trajectory(
        self, head_name: str = "main_stick"
    ) -> List[Tuple[int, float]]:
        """Get confidence at each layer for the final predicted class."""
        final_class = self.final_prediction.get(head_name)
        if final_class is None:
            return []

        trajectory = []
        for layer in self.layers:
            if head_name in layer.probs:
                prob = layer.probs[head_name][final_class].item()
                trajectory.append((layer.layer_idx, prob))

        return trajectory

    def get_prediction_changes(
        self, head_name: str = "main_stick"
    ) -> List[Tuple[int, int, int]]:
        """
        Find layers where the predicted class changes.

        Returns list of (layer_idx, old_class, new_class) tuples.
        """
        changes = []
        prev_class = None

        for layer in self.layers:
            curr_class = layer.predicted_class.get(head_name)
            if curr_class is not None and curr_class != prev_class:
                if prev_class is not None:
                    changes.append((layer.layer_idx, prev_class, curr_class))
                prev_class = curr_class

        return changes

    def summary(self, head_name: str = "main_stick") -> str:
        """Generate a human-readable summary."""
        lines = ["=" * 60, "PREDICTION EVOLUTION", "=" * 60]

        final = self.final_prediction.get(head_name, -1)
        crystal_layer = self.get_crystallization_layer(head_name)

        lines.append(f"\nFinal prediction: {head_name}={final}")
        lines.append(f"Crystallization layer: {crystal_layer}")

        lines.append(f"\nLayer-by-layer predictions ({head_name}):")
        for layer in self.layers:
            pred = layer.predicted_class.get(head_name, -1)
            conf = layer.confidence.get(head_name, 0)
            match = "✓" if pred == final else " "
            bar = "█" * int(conf * 20)
            lines.append(f"  Layer {layer.layer_idx}: {pred:3d} ({conf:.2f}) {bar} {match}")

        changes = self.get_prediction_changes(head_name)
        if changes:
            lines.append("\nPrediction changes:")
            for layer_idx, old, new in changes:
                lines.append(f"  Layer {layer_idx}: {old} -> {new}")

        return "\n".join(lines)


class LogitLens:
    """
    Apply output heads to intermediate layer activations.

    The logit lens reveals how the model's predictions evolve through
    the network. By applying the final output heads to each layer's
    output, we can see:

    1. When predictions become stable (crystallization)
    2. Which layers cause prediction changes
    3. How confidence builds through the network

    Usage:
        lens = LogitLens(model, colmap, device)

        # Analyze prediction evolution
        evolution = lens.analyze(inputs, position=-1)
        print(evolution.summary())

        # Find when the model committed to its prediction
        crystal_layer = evolution.get_crystallization_layer("main_stick")

        # Get confidence trajectory
        trajectory = evolution.get_confidence_trajectory("main_stick")
    """

    def __init__(
        self,
        model: "GPT",
        colmap: "ColumnMap",
        device: torch.device,
    ):
        self.model = model
        self.colmap = colmap
        self.device = device
        self._hook_manager = HookManager(model)

    def _apply_output_heads(
        self,
        hidden_states: Tensor,
        position: int,
    ) -> Dict[str, Tensor]:
        """
        Apply the model's output heads to hidden states.

        This mimics what the model does at the end, but applied to
        intermediate layer outputs. The heads are applied sequentially
        with information flow between them, matching the model architecture.
        """
        from model.norm import norm

        # Apply final normalization (as the model does before output heads)
        normed = norm(hidden_states)

        # Get hidden states at the target position
        h = normed[:, position, :]  # [batch, embedding_dim]

        outputs = {}

        # Sequential heads: buttons → main_stick → c_stick → shoulder
        # Each head receives concatenated outputs from previous heads

        # Button head (takes just hidden states)
        if hasattr(self.model, "button_head"):
            button_logits = self.model.button_head(h)
            outputs["buttons"] = button_logits
        else:
            button_logits = torch.zeros(h.shape[0], 5, device=h.device)

        # Main stick head (hidden + buttons)
        if hasattr(self.model, "main_stick_head"):
            main_stick_input = torch.cat((h, button_logits.detach()), dim=-1)
            main_stick = self.model.main_stick_head(main_stick_input)
            outputs["main_stick"] = main_stick
        else:
            main_stick = torch.zeros(h.shape[0], 64, device=h.device)

        # C-stick head (hidden + buttons + main_stick)
        if hasattr(self.model, "c_stick_head"):
            c_stick_input = torch.cat(
                (h, button_logits.detach(), main_stick.detach()), dim=-1
            )
            c_stick = self.model.c_stick_head(c_stick_input)
            outputs["c_stick"] = c_stick
        else:
            c_stick = torch.zeros(h.shape[0], 9, device=h.device)

        # Shoulder head (hidden + buttons + main_stick + c_stick)
        if hasattr(self.model, "shoulder_head"):
            shoulder_input = torch.cat(
                (h, button_logits.detach(), main_stick.detach(), c_stick.detach()),
                dim=-1,
            )
            outputs["shoulder"] = self.model.shoulder_head(shoulder_input)

        # Value head (takes just hidden states)
        if hasattr(self.model, "value_head"):
            outputs["value"] = self.model.value_head(h)

        return outputs

    def _run_to_layer(
        self,
        inputs: Tensor,
        target_layer: int,
    ) -> Tensor:
        """
        Run the model up to a specific layer and return hidden states.
        """
        from train.batch_utils import build_model_inputs

        inputs_td = build_model_inputs(inputs, self.colmap)

        # Embed inputs
        combined_inputs = self.model._embed_inputs(inputs_td)
        hidden_states = self.model.projection_down(combined_inputs)
        hidden_states = self.model.dropout(hidden_states)

        # Get RoPE embeddings from precomputed buffers
        seq_len = hidden_states.shape[1]
        cos = self.model.cos[:, :seq_len]
        sin = self.model.sin[:, :seq_len]

        # Run through blocks up to target layer
        for i in range(target_layer + 1):
            hidden_states = self.model.blocks[i](hidden_states, cos, sin)

        return hidden_states

    def get_layer_prediction(
        self,
        inputs: Tensor,
        layer_idx: int,
        position: int = -1,
    ) -> LayerPrediction:
        """
        Get predictions by applying output heads to a specific layer.

        Args:
            inputs: Input tensor [batch, seq, features]
            layer_idx: Which layer's output to analyze
            position: Sequence position to predict for

        Returns:
            LayerPrediction with logits, probs, and predicted class
        """
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(0)
        inputs = inputs.to(self.device)

        seq_len = inputs.shape[1]
        if position < 0:
            position = seq_len + position

        self.model.eval()

        with torch.no_grad():
            hidden_states = self._run_to_layer(inputs, layer_idx)
            outputs = self._apply_output_heads(hidden_states, position)

        # Compute probabilities and predictions
        logits = {}
        probs = {}
        predicted_class = {}
        confidence = {}

        for head_name, logit in outputs.items():
            if logit.dim() == 1:
                logit = logit.unsqueeze(0)

            logits[head_name] = logit[0]

            if head_name != "value":  # Classification heads
                prob = F.softmax(logit[0], dim=-1)
                probs[head_name] = prob
                predicted_class[head_name] = prob.argmax().item()
                confidence[head_name] = prob.max().item()
            else:  # Value head (regression)
                probs[head_name] = logit[0]
                predicted_class[head_name] = 0
                confidence[head_name] = logit[0].item()

        return LayerPrediction(
            layer_idx=layer_idx,
            logits=logits,
            probs=probs,
            predicted_class=predicted_class,
            confidence=confidence,
        )

    def analyze(
        self,
        inputs: Tensor,
        position: int = -1,
        layers: Optional[List[int]] = None,
    ) -> PredictionEvolution:
        """
        Analyze prediction evolution through all (or specified) layers.

        Args:
            inputs: Input tensor [batch, seq, features]
            position: Sequence position to analyze
            layers: Specific layers to analyze (None = all)

        Returns:
            PredictionEvolution with per-layer predictions
        """
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(0)
        inputs = inputs.to(self.device)

        n_layers = len(self.model.blocks)

        if layers is None:
            layers = list(range(n_layers))

        layer_predictions = []
        for layer_idx in layers:
            pred = self.get_layer_prediction(inputs, layer_idx, position)
            layer_predictions.append(pred)

        # Get final prediction (from last layer)
        final_pred = layer_predictions[-1].predicted_class

        return PredictionEvolution(
            layers=layer_predictions,
            final_prediction=final_pred,
        )

    def compare_predictions(
        self,
        inputs: Tensor,
        layer_a: int,
        layer_b: int,
        position: int = -1,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare predictions between two layers.

        Useful for understanding what changes between specific layers.

        Args:
            inputs: Input tensor
            layer_a: First layer
            layer_b: Second layer
            position: Sequence position

        Returns:
            Dictionary with comparison metrics per head
        """
        pred_a = self.get_layer_prediction(inputs, layer_a, position)
        pred_b = self.get_layer_prediction(inputs, layer_b, position)

        comparison = {}
        for head in pred_a.probs:
            if head in pred_b.probs and head != "value":
                probs_a = pred_a.probs[head]
                probs_b = pred_b.probs[head]

                # KL divergence
                kl_div = F.kl_div(
                    probs_a.log(), probs_b, reduction="sum"
                ).item()

                # Whether predicted class changed
                class_changed = pred_a.predicted_class[head] != pred_b.predicted_class[head]

                # Confidence change
                conf_change = pred_b.confidence[head] - pred_a.confidence[head]

                comparison[head] = {
                    "kl_divergence": kl_div,
                    "class_changed": float(class_changed),
                    "confidence_change": conf_change,
                    "class_a": pred_a.predicted_class[head],
                    "class_b": pred_b.predicted_class[head],
                }

        return comparison

    def find_decision_layers(
        self,
        inputs: Tensor,
        head_name: str = "main_stick",
        position: int = -1,
        change_threshold: float = 0.1,
    ) -> List[int]:
        """
        Find layers where significant prediction changes occur.

        These are the "decision layers" where the model commits to
        different actions.

        Args:
            inputs: Input tensor
            head_name: Which head to analyze
            position: Sequence position
            change_threshold: Minimum probability change to consider significant

        Returns:
            List of layer indices where significant changes occur
        """
        evolution = self.analyze(inputs, position)

        decision_layers = []
        prev_probs = None

        for layer in evolution.layers:
            if head_name not in layer.probs:
                continue

            curr_probs = layer.probs[head_name]

            if prev_probs is not None:
                # Compute total probability change
                change = (curr_probs - prev_probs).abs().sum().item()
                if change > change_threshold:
                    decision_layers.append(layer.layer_idx)

            prev_probs = curr_probs

        return decision_layers


def visualize_evolution(
    evolution: PredictionEvolution,
    head_name: str = "main_stick",
    top_k: int = 3,
) -> str:
    """
    Create an ASCII visualization of prediction evolution.

    Shows how the top-k predictions change through layers.
    """
    lines = [f"Prediction Evolution for {head_name}", "=" * 60]

    final_class = evolution.final_prediction.get(head_name, -1)

    for layer in evolution.layers:
        if head_name not in layer.probs:
            continue

        probs = layer.probs[head_name]
        values, indices = torch.topk(probs, min(top_k, len(probs)))

        # Format predictions
        pred_strs = []
        for idx, val in zip(indices.tolist(), values.tolist()):
            marker = "*" if idx == final_class else " "
            pred_strs.append(f"{idx}{marker}:{val:.2f}")

        lines.append(f"  L{layer.layer_idx}: {' | '.join(pred_strs)}")

    return "\n".join(lines)


__all__ = [
    "LogitLens",
    "LayerPrediction",
    "PredictionEvolution",
    "visualize_evolution",
]
