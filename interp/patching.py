"""
Activation patching for causal analysis.

Provides tools to understand which layers are responsible for model decisions
by patching activations from one run into another.

Use cases:
- "What made the model predict grab instead of dash attack?"
- "Which layer decided to use this recovery option?"
- "How much does changing opponent position affect the decision?"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from interp.hooks import HookManager, HookPoint, HookPointType

if TYPE_CHECKING:
    from column_map import ColumnMap
    from model.nano_gpt import GPT


@dataclass
class PatchingResult:
    """Result from a single patching experiment."""

    hook_point: HookPoint
    clean_logits: Dict[str, Tensor]  # head_name -> logits
    patched_logits: Dict[str, Tensor]
    effect: Dict[str, float]  # Change in target logit/probability


@dataclass
class CausalTrace:
    """Results from tracing causal effects across all layers."""

    clean_prediction: Dict[str, int]  # head_name -> predicted class
    corrupted_prediction: Dict[str, int]
    layer_effects: Dict[int, Dict[str, float]]  # layer_idx -> head_name -> effect
    total_effect: Dict[str, float]  # head_name -> total effect

    def get_critical_layer(self, head_name: str = "main_stick") -> int:
        """Find the layer with the largest causal effect."""
        effects = [(idx, self.layer_effects[idx].get(head_name, 0.0))
                   for idx in self.layer_effects]
        if not effects:
            return 0
        return max(effects, key=lambda x: abs(x[1]))[0]

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = ["=" * 60, "CAUSAL TRACE SUMMARY", "=" * 60]

        lines.append("\nPrediction change:")
        for head in self.clean_prediction:
            clean = self.clean_prediction[head]
            corrupt = self.corrupted_prediction[head]
            lines.append(f"  {head}: {clean} -> {corrupt}")

        lines.append("\nLayer effects (main_stick):")
        for layer_idx in sorted(self.layer_effects.keys()):
            effect = self.layer_effects[layer_idx].get("main_stick", 0.0)
            bar = "█" * int(abs(effect) * 20)
            sign = "+" if effect > 0 else "-"
            lines.append(f"  Layer {layer_idx}: {sign}{abs(effect):.3f} {bar}")

        critical = self.get_critical_layer("main_stick")
        lines.append(f"\nCritical layer: {critical}")

        return "\n".join(lines)


class ActivationPatcher:
    """
    Perform activation patching experiments for causal analysis.

    Activation patching works by:
    1. Running the model on "clean" input and caching activations
    2. Running the model on "corrupted" input (e.g., different opponent position)
    3. Re-running the corrupted input, but patching in clean activations at
       specific layers
    4. Measuring how much the output changes toward the clean prediction

    This identifies which layers are "responsible" for the difference between
    clean and corrupted behavior.

    Usage:
        patcher = ActivationPatcher(model, colmap, device)

        # Compare two situations
        clean_inputs = ...  # Normal game state
        corrupted_inputs = ...  # Same but opponent in different position

        trace = patcher.trace_all_layers(
            clean_inputs, corrupted_inputs,
            target_head="main_stick",
            position=-1
        )

        print(f"Critical layer: {trace.get_critical_layer()}")
        print(trace.summary())
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

    def _get_all_block_points(self) -> List[HookPoint]:
        """Get hook points for all transformer blocks."""
        n_layers = len(self.model.blocks)
        return [HookPoint(HookPointType.BLOCK_OUTPUT, i) for i in range(n_layers)]

    def _run_and_cache(
        self,
        inputs: Tensor,
        hook_points: List[HookPoint],
    ) -> Tuple[Dict[str, Tensor], Dict[HookPoint, Tensor]]:
        """Run model and cache activations at specified hook points."""
        from train.batch_utils import build_model_inputs

        self._hook_manager.install_hooks(hook_points)

        try:
            inputs_td = build_model_inputs(inputs.to(self.device), self.colmap)

            with torch.no_grad():
                outputs = self.model(inputs_td)

            # Get cached activations
            activations = {}
            for point in hook_points:
                activations[point] = self._hook_manager.get_single(point).clone()

            self._hook_manager.clear()

        finally:
            self._hook_manager.remove_hooks()

        return outputs, activations

    def _run_with_patch(
        self,
        inputs: Tensor,
        patch_point: HookPoint,
        patch_activations: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Run model on inputs but patch in different activations at one layer.

        This requires modifying the forward pass to inject the patched activations.
        We do this by using a forward hook that replaces the output.
        """
        from train.batch_utils import build_model_inputs

        patched_output = {}

        def patch_hook(module, inputs, output):
            """Replace the output with patched activations."""
            return patch_activations

        # Get the module to patch
        module = self._hook_manager._get_module(patch_point)
        if module is None:
            raise ValueError(f"Cannot find module for hook point: {patch_point}")

        # Register the patching hook
        handle = module.register_forward_hook(patch_hook)

        try:
            inputs_td = build_model_inputs(inputs.to(self.device), self.colmap)

            with torch.no_grad():
                outputs = self.model(inputs_td)

            patched_output = {k: v.clone() for k, v in outputs.items()
                             if isinstance(v, Tensor)}

        finally:
            handle.remove()

        return patched_output

    def patch_single_layer(
        self,
        clean_inputs: Tensor,
        corrupted_inputs: Tensor,
        hook_point: HookPoint,
        target_head: str = "main_stick",
        position: int = -1,
    ) -> PatchingResult:
        """
        Patch activations at a single layer and measure the effect.

        Args:
            clean_inputs: The "correct" input [batch, seq, features]
            corrupted_inputs: The "corrupted" input [batch, seq, features]
            hook_point: Which layer to patch
            target_head: Which output head to measure
            position: Which sequence position to analyze

        Returns:
            PatchingResult with effect measurements
        """
        # Ensure batch dimension
        if clean_inputs.dim() == 2:
            clean_inputs = clean_inputs.unsqueeze(0)
        if corrupted_inputs.dim() == 2:
            corrupted_inputs = corrupted_inputs.unsqueeze(0)

        seq_len = clean_inputs.shape[1]
        if position < 0:
            position = seq_len + position

        # Run clean input and cache activations
        clean_outputs, clean_acts = self._run_and_cache(
            clean_inputs, [hook_point]
        )
        clean_act = clean_acts[hook_point]

        # Run corrupted input (no patching)
        corrupted_outputs, _ = self._run_and_cache(
            corrupted_inputs, []
        )

        # Run corrupted input with clean activations patched in
        patched_outputs = self._run_with_patch(
            corrupted_inputs, hook_point, clean_act
        )

        # Compute effect: how much did patching change the corrupted output
        # toward the clean output?
        effect = {}
        for head in [target_head]:
            if head in clean_outputs and head in corrupted_outputs:
                clean_logit = clean_outputs[head][0, position]
                corrupt_logit = corrupted_outputs[head][0, position]
                patched_logit = patched_outputs[head][0, position]

                # Effect = how much patching recovered the clean prediction
                # Measured as correlation with the direction from corrupt -> clean
                direction = clean_logit - corrupt_logit
                recovery = patched_logit - corrupt_logit

                # Normalized effect: 1.0 means fully recovered clean behavior
                denom = (direction * direction).sum().sqrt()
                if denom > 1e-8:
                    effect[head] = (
                        (direction * recovery).sum() / (denom * denom)
                    ).item()
                else:
                    effect[head] = 0.0

        return PatchingResult(
            hook_point=hook_point,
            clean_logits={k: v[0, position].clone() for k, v in clean_outputs.items()
                         if isinstance(v, Tensor) and v.dim() >= 2},
            patched_logits={k: v[0, position].clone() for k, v in patched_outputs.items()
                           if isinstance(v, Tensor) and v.dim() >= 2},
            effect=effect,
        )

    def trace_all_layers(
        self,
        clean_inputs: Tensor,
        corrupted_inputs: Tensor,
        target_head: str = "main_stick",
        position: int = -1,
    ) -> CausalTrace:
        """
        Trace causal effects across all transformer layers.

        For each layer, patches in clean activations and measures how much
        the output recovers toward the clean prediction.

        Args:
            clean_inputs: The "correct" input [batch, seq, features]
            corrupted_inputs: The "corrupted" input [batch, seq, features]
            target_head: Which output head to analyze
            position: Which sequence position to analyze

        Returns:
            CausalTrace with per-layer effect measurements
        """
        # Ensure batch dimension
        if clean_inputs.dim() == 2:
            clean_inputs = clean_inputs.unsqueeze(0)
        if corrupted_inputs.dim() == 2:
            corrupted_inputs = corrupted_inputs.unsqueeze(0)

        seq_len = clean_inputs.shape[1]
        if position < 0:
            position = seq_len + position

        # Get predictions without patching
        clean_outputs, _ = self._run_and_cache(clean_inputs, [])
        corrupted_outputs, _ = self._run_and_cache(corrupted_inputs, [])

        clean_pred = {}
        corrupt_pred = {}
        for head in ["main_stick", "c_stick", "buttons", "shoulder"]:
            if head in clean_outputs:
                clean_pred[head] = clean_outputs[head][0, position].argmax().item()
                corrupt_pred[head] = corrupted_outputs[head][0, position].argmax().item()

        # Trace each layer
        layer_effects = {}
        block_points = self._get_all_block_points()

        for point in block_points:
            result = self.patch_single_layer(
                clean_inputs, corrupted_inputs,
                point, target_head, position
            )
            layer_effects[point.layer_idx] = result.effect

        # Total effect (difference between clean and corrupted)
        total_effect = {}
        for head in ["main_stick", "c_stick", "buttons", "shoulder"]:
            if head in clean_outputs:
                clean_probs = F.softmax(clean_outputs[head][0, position], dim=-1)
                corrupt_probs = F.softmax(corrupted_outputs[head][0, position], dim=-1)
                total_effect[head] = (clean_probs - corrupt_probs).abs().sum().item()

        return CausalTrace(
            clean_prediction=clean_pred,
            corrupted_prediction=corrupt_pred,
            layer_effects=layer_effects,
            total_effect=total_effect,
        )

    def compute_direct_effect(
        self,
        inputs: Tensor,
        ablate_point: HookPoint,
        target_head: str = "main_stick",
        position: int = -1,
        ablation_type: str = "zero",
    ) -> Dict[str, float]:
        """
        Compute the direct effect of a layer by ablating it.

        Instead of patching, this zeros out or mean-ablates a layer's
        contribution to see how important it is.

        Args:
            inputs: Input tensor [batch, seq, features]
            ablate_point: Which layer to ablate
            target_head: Which output head to measure
            position: Which sequence position
            ablation_type: "zero" or "mean" ablation

        Returns:
            Dictionary mapping head names to effect magnitudes
        """
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(0)

        seq_len = inputs.shape[1]
        if position < 0:
            position = seq_len + position

        # Get baseline output
        baseline_outputs, baseline_acts = self._run_and_cache(inputs, [ablate_point])

        # Create ablated activation
        if ablation_type == "zero":
            ablated_act = torch.zeros_like(baseline_acts[ablate_point])
        else:  # mean
            ablated_act = baseline_acts[ablate_point].mean(dim=1, keepdim=True)
            ablated_act = ablated_act.expand_as(baseline_acts[ablate_point])

        # Run with ablated activations
        ablated_outputs = self._run_with_patch(inputs, ablate_point, ablated_act)

        # Compute effect
        effect = {}
        for head in ["main_stick", "c_stick", "buttons", "shoulder"]:
            if head in baseline_outputs and head in ablated_outputs:
                baseline_probs = F.softmax(baseline_outputs[head][0, position], dim=-1)
                ablated_probs = F.softmax(ablated_outputs[head][0, position], dim=-1)
                effect[head] = (baseline_probs - ablated_probs).abs().sum().item()

        return effect


def create_corrupted_input(
    clean_inputs: Tensor,
    colmap: "ColumnMap",
    corruption_type: str = "swap_players",
) -> Tensor:
    """
    Create a corrupted version of clean inputs for patching experiments.

    Corruption types:
    - "swap_players": Swap P1 and P2 features
    - "zero_opponent": Zero out P2 features
    - "random_opponent": Randomize P2 features

    Args:
        clean_inputs: Original input tensor
        colmap: Column mapping
        corruption_type: Type of corruption to apply

    Returns:
        Corrupted input tensor
    """
    corrupted = clean_inputs.clone()

    if corruption_type == "swap_players":
        # Find P1 and P2 feature ranges
        p1_indices = [i for i, name in enumerate(colmap.feat_names)
                     if name.startswith("p1_")]
        p2_indices = [i for i, name in enumerate(colmap.feat_names)
                     if name.startswith("p2_")]

        if len(p1_indices) == len(p2_indices):
            # Swap P1 and P2 features
            temp = corrupted[..., p1_indices].clone()
            corrupted[..., p1_indices] = corrupted[..., p2_indices]
            corrupted[..., p2_indices] = temp

    elif corruption_type == "zero_opponent":
        # Zero out P2 features
        p2_indices = [i for i, name in enumerate(colmap.feat_names)
                     if name.startswith("p2_")]
        corrupted[..., p2_indices] = 0.0

    elif corruption_type == "random_opponent":
        # Randomize P2 features
        p2_indices = [i for i, name in enumerate(colmap.feat_names)
                     if name.startswith("p2_")]
        corrupted[..., p2_indices] = torch.randn_like(corrupted[..., p2_indices])

    return corrupted


__all__ = [
    "ActivationPatcher",
    "PatchingResult",
    "CausalTrace",
    "create_corrupted_input",
]
