"""
Circuit analysis for understanding behavior implementations.

Circuits are the computational subgraphs that implement specific behaviors.
This module provides tools to:
- Define behavior detectors (wavedash, recovery, edgeguard, etc.)
- Find which layers are critical for each behavior
- Identify SAE features that implement behaviors
- Analyze information flow through the network

Usage:
    from interp.circuits import CircuitAnalyzer, MELEE_CIRCUITS

    analyzer = CircuitAnalyzer(model, colmap, device)

    # Find which layers implement wavedashing
    circuit = MELEE_CIRCUITS["wavedash"]
    analysis = analyzer.analyze_circuit(circuit, dataloader)
    print(f"Critical layers: {analysis.critical_layers}")

    # Find SAE features for recovery
    recovery = MELEE_CIRCUITS["recovery"]
    features = analyzer.find_circuit_features(recovery, sae, hook_point, dataloader)
    print(f"Top recovery features: {features[:10]}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from column_map import ColumnMap
    from interp.hooks import HookPoint
    from interp.sae.topk import TopKSparseAutoencoder
    from model.nano_gpt import GPT
    from tensordict import TensorDict


# Type for behavior detectors
# Takes (inputs, outputs) and returns boolean mask [batch] or [batch, seq]
BehaviorDetector = Callable[[Tensor, "TensorDict"], Tensor]

# Type for input conditions
# Takes inputs and returns boolean mask
InputCondition = Callable[[Tensor], Tensor]


@dataclass
class CircuitSpec:
    """
    Specification for a behavior circuit.

    A circuit is defined by:
    - A detector function that identifies when the behavior occurs
    - Optional input conditions (e.g., "when offstage")
    - Metadata for interpretation
    """

    name: str
    description: str
    detector: BehaviorDetector
    input_condition: Optional[InputCondition] = None

    # Metadata
    output_heads: List[str] = field(default_factory=lambda: ["main_stick"])
    expected_frequency: float = 0.1  # Expected rate in dataset

    def detect(
        self,
        inputs: Tensor,
        outputs: "TensorDict",
        position: int = -1,
    ) -> Tensor:
        """
        Detect behavior occurrences.

        Args:
            inputs: Input tensor [batch, seq, features]
            outputs: Model outputs TensorDict
            position: Sequence position to check

        Returns:
            Boolean mask [batch] indicating behavior occurrence
        """
        mask = self.detector(inputs, outputs)

        # Apply input condition if specified
        if self.input_condition is not None:
            input_mask = self.input_condition(inputs)
            if input_mask.dim() > 1:
                input_mask = input_mask[:, position]
            mask = mask & input_mask

        return mask


@dataclass
class LayerImportance:
    """Importance of a layer for a circuit."""

    layer_idx: int
    causal_effect: float  # Effect of patching this layer
    ablation_effect: float  # Effect of ablating this layer
    attention_contribution: float  # Attention-based importance

    @property
    def combined_score(self) -> float:
        """Combined importance score."""
        return (self.causal_effect + self.ablation_effect) / 2


@dataclass
class FeatureImportance:
    """Importance of an SAE feature for a circuit."""

    feature_idx: int
    activation_rate: float  # How often it activates during behavior
    specificity: float  # How specific to this behavior (vs baseline)
    causal_effect: float  # Effect of ablating this feature

    @property
    def combined_score(self) -> float:
        """Combined importance score."""
        return self.activation_rate * self.specificity


@dataclass
class CircuitAnalysis:
    """Complete analysis of a behavior circuit."""

    circuit: CircuitSpec
    n_samples: int
    behavior_rate: float  # How often behavior was detected

    # Layer analysis
    layer_importance: List[LayerImportance]
    critical_layers: List[int]  # Layers with highest importance

    # Feature analysis (if SAE provided)
    feature_importance: Optional[List[FeatureImportance]] = None
    critical_features: Optional[List[int]] = None

    # Attention analysis
    attention_patterns: Optional[Dict[int, float]] = None  # layer -> attention score

    def get_layer_ranking(self) -> List[Tuple[int, float]]:
        """Get layers ranked by importance."""
        return sorted(
            [(li.layer_idx, li.combined_score) for li in self.layer_importance],
            key=lambda x: -x[1],
        )

    def get_feature_ranking(self) -> List[Tuple[int, float]]:
        """Get features ranked by importance."""
        if not self.feature_importance:
            return []
        return sorted(
            [(fi.feature_idx, fi.combined_score) for fi in self.feature_importance],
            key=lambda x: -x[1],
        )

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "=" * 60,
            f"CIRCUIT ANALYSIS: {self.circuit.name}",
            "=" * 60,
            f"\n{self.circuit.description}",
            f"\nSamples analyzed: {self.n_samples}",
            f"Behavior rate: {self.behavior_rate:.1%}",
        ]

        lines.append("\n--- LAYER IMPORTANCE ---")
        for layer_idx, score in self.get_layer_ranking()[:5]:
            bar = "█" * int(score * 40)
            lines.append(f"  Layer {layer_idx}: {score:.3f} {bar}")

        if self.critical_layers:
            lines.append(f"\nCritical layers: {self.critical_layers}")

        if self.feature_importance:
            lines.append("\n--- FEATURE IMPORTANCE ---")
            for feat_idx, score in self.get_feature_ranking()[:10]:
                lines.append(f"  Feature {feat_idx}: {score:.3f}")

        return "\n".join(lines)


# =============================================================================
# Melee-specific behavior detectors
# =============================================================================


def _get_output_prediction(outputs: "TensorDict", head: str, position: int = -1) -> Tensor:
    """Get predicted class from outputs."""
    logits = outputs[head]
    if logits.dim() == 3:
        logits = logits[:, position, :]
    return logits.argmax(dim=-1)


def _get_output_probs(outputs: "TensorDict", head: str, position: int = -1) -> Tensor:
    """Get class probabilities from outputs."""
    logits = outputs[head]
    if logits.dim() == 3:
        logits = logits[:, position, :]
    return F.softmax(logits, dim=-1)


# Wavedash angles in the main_stick palette (approximate indices)
# These are diagonal down angles for wavedashing
WAVEDASH_ANGLES = list(range(48, 56))  # Approximate wavedash angle indices


def detect_wavedash(inputs: Tensor, outputs: "TensorDict") -> Tensor:
    """
    Detect wavedash prediction.

    Wavedash = diagonal down main_stick + shoulder press.
    """
    main_stick = _get_output_prediction(outputs, "main_stick")
    shoulder = _get_output_prediction(outputs, "shoulder")

    # Check if main_stick is a wavedash angle
    is_wavedash_angle = torch.zeros_like(main_stick, dtype=torch.bool)
    for angle in WAVEDASH_ANGLES:
        is_wavedash_angle = is_wavedash_angle | (main_stick == angle)

    # Check if shoulder is pressed (not neutral = index 0)
    shoulder_pressed = shoulder > 0

    return is_wavedash_angle & shoulder_pressed


def detect_shield(inputs: Tensor, outputs: "TensorDict") -> Tensor:
    """
    Detect shield prediction.

    Shield = shoulder pressed + no attack buttons.
    """
    shoulder = _get_output_prediction(outputs, "shoulder")
    buttons = outputs["buttons"]
    if buttons.dim() == 3:
        buttons = buttons[:, -1, :]

    # Shoulder pressed (any non-zero level)
    shoulder_pressed = shoulder > 0

    # No attack buttons (A, B are usually indices 0, 1)
    button_probs = torch.sigmoid(buttons)
    no_attack = (button_probs[:, 0] < 0.5) & (button_probs[:, 1] < 0.5)

    return shoulder_pressed & no_attack


def detect_aerial(inputs: Tensor, outputs: "TensorDict") -> Tensor:
    """
    Detect aerial attack prediction.

    Aerial = attack button + c-stick or main stick in air.
    """
    buttons = outputs["buttons"]
    if buttons.dim() == 3:
        buttons = buttons[:, -1, :]

    # A button pressed (usually index 0)
    button_probs = torch.sigmoid(buttons)
    a_pressed = button_probs[:, 0] > 0.5

    return a_pressed


def detect_grab(inputs: Tensor, outputs: "TensorDict") -> Tensor:
    """
    Detect grab prediction.

    Grab = Z button or shield + A.
    """
    buttons = outputs["buttons"]
    shoulder = _get_output_prediction(outputs, "shoulder")

    if buttons.dim() == 3:
        buttons = buttons[:, -1, :]

    button_probs = torch.sigmoid(buttons)

    # Z button (usually index 3)
    z_pressed = button_probs[:, 3] > 0.5 if buttons.shape[-1] > 3 else torch.zeros_like(shoulder, dtype=torch.bool)

    # Shield + A
    a_pressed = button_probs[:, 0] > 0.5
    shield_pressed = shoulder > 0
    shield_grab = a_pressed & shield_pressed

    return z_pressed | shield_grab


def detect_dash(inputs: Tensor, outputs: "TensorDict") -> Tensor:
    """
    Detect dash/run prediction.

    Dash = main stick fully left or right.
    """
    main_stick = _get_output_prediction(outputs, "main_stick")

    # Full left and right are typically at specific indices
    # In a 64-position palette, let's assume 0-7 are leftward, 56-63 are rightward
    is_left = main_stick < 8
    is_right = main_stick >= 56

    return is_left | is_right


def detect_neutral_stick(inputs: Tensor, outputs: "TensorDict") -> Tensor:
    """
    Detect neutral stick prediction.

    Neutral = main stick near center.
    """
    main_stick = _get_output_prediction(outputs, "main_stick")

    # Assuming center/neutral is around index 32 in a 64-position palette
    # Let's say indices 28-36 are "neutral"
    is_neutral = (main_stick >= 28) & (main_stick <= 36)

    return is_neutral


def make_offstage_condition(colmap: "ColumnMap") -> InputCondition:
    """Create condition for P1 being offstage."""
    try:
        offstage_idx = colmap.feat_names.index("p1_off_stage")
    except ValueError:
        # Fallback: use position
        try:
            x_idx = colmap.feat_names.index("p1_position_x")
            y_idx = colmap.feat_names.index("p1_position_y")

            def offstage_by_position(inputs: Tensor) -> Tensor:
                x = inputs[..., x_idx]
                y = inputs[..., y_idx]
                # Rough offstage bounds
                return (x.abs() > 80) | (y < -20)

            return offstage_by_position
        except ValueError:
            return lambda x: torch.ones(x.shape[0], dtype=torch.bool, device=x.device)

    def offstage_condition(inputs: Tensor) -> Tensor:
        return inputs[..., offstage_idx] > 0.5

    return offstage_condition


def make_opponent_offstage_condition(colmap: "ColumnMap") -> InputCondition:
    """Create condition for P2 (opponent) being offstage."""
    try:
        offstage_idx = colmap.feat_names.index("p2_off_stage")
    except ValueError:
        try:
            x_idx = colmap.feat_names.index("p2_position_x")
            y_idx = colmap.feat_names.index("p2_position_y")

            def offstage_by_position(inputs: Tensor) -> Tensor:
                x = inputs[..., x_idx]
                y = inputs[..., y_idx]
                return (x.abs() > 80) | (y < -20)

            return offstage_by_position
        except ValueError:
            return lambda x: torch.ones(x.shape[0], dtype=torch.bool, device=x.device)

    def offstage_condition(inputs: Tensor) -> Tensor:
        return inputs[..., offstage_idx] > 0.5

    return offstage_condition


def make_shielding_condition(colmap: "ColumnMap") -> InputCondition:
    """Create condition for P1 currently shielding."""
    # Check for shield-related action states
    try:
        action_idx = colmap.feat_names.index("p1_action")
        # Shield action states are typically in range 178-182
        SHIELD_ACTIONS = [178, 179, 180, 181, 182]

        def shielding_condition(inputs: Tensor) -> Tensor:
            action = inputs[..., action_idx].long()
            is_shielding = torch.zeros_like(action, dtype=torch.bool)
            for shield_action in SHIELD_ACTIONS:
                is_shielding = is_shielding | (action == shield_action)
            return is_shielding

        return shielding_condition
    except ValueError:
        return lambda x: torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)


# Predefined Melee circuits
def get_melee_circuits(colmap: Optional["ColumnMap"] = None) -> Dict[str, CircuitSpec]:
    """
    Get predefined Melee behavior circuits.

    Args:
        colmap: Column map for input conditions (optional)

    Returns:
        Dictionary of circuit specifications
    """
    circuits = {
        "wavedash": CircuitSpec(
            name="wavedash",
            description="Wavedash: diagonal down angle + shoulder trigger",
            detector=detect_wavedash,
            output_heads=["main_stick", "shoulder"],
            expected_frequency=0.05,
        ),
        "shield": CircuitSpec(
            name="shield",
            description="Shield: shoulder pressed without attack",
            detector=detect_shield,
            output_heads=["shoulder", "buttons"],
            expected_frequency=0.1,
        ),
        "aerial": CircuitSpec(
            name="aerial",
            description="Aerial attack: A button pressed",
            detector=detect_aerial,
            output_heads=["buttons"],
            expected_frequency=0.15,
        ),
        "grab": CircuitSpec(
            name="grab",
            description="Grab: Z button or shield+A",
            detector=detect_grab,
            output_heads=["buttons", "shoulder"],
            expected_frequency=0.03,
        ),
        "dash": CircuitSpec(
            name="dash",
            description="Dash: full horizontal stick input",
            detector=detect_dash,
            output_heads=["main_stick"],
            expected_frequency=0.2,
        ),
        "neutral": CircuitSpec(
            name="neutral",
            description="Neutral stick: center position",
            detector=detect_neutral_stick,
            output_heads=["main_stick"],
            expected_frequency=0.3,
        ),
    }

    # Add input-conditioned circuits if colmap provided
    if colmap is not None:
        # Recovery: any action while offstage
        circuits["recovery"] = CircuitSpec(
            name="recovery",
            description="Recovery: any action while P1 offstage",
            detector=lambda i, o: torch.ones(i.shape[0], dtype=torch.bool, device=i.device),
            input_condition=make_offstage_condition(colmap),
            output_heads=["main_stick", "buttons"],
            expected_frequency=0.05,
        )

        # Edgeguard: attack while opponent offstage
        circuits["edgeguard"] = CircuitSpec(
            name="edgeguard",
            description="Edgeguard: attack while opponent offstage",
            detector=detect_aerial,
            input_condition=make_opponent_offstage_condition(colmap),
            output_heads=["buttons", "main_stick"],
            expected_frequency=0.02,
        )

        # Shield pressure: action while opponent shielding
        # (This would need opponent shield detection which is complex)

    return circuits


# Global circuits (without colmap-dependent ones)
MELEE_CIRCUITS: Dict[str, CircuitSpec] = get_melee_circuits()


class CircuitAnalyzer:
    """
    Analyze circuits that implement specific behaviors.

    Uses activation patching, ablation, and SAE analysis to understand
    which parts of the network are responsible for each behavior.
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

        # Update MELEE_CIRCUITS with colmap-dependent ones
        self.circuits = get_melee_circuits(colmap)

    def analyze_circuit(
        self,
        circuit: CircuitSpec,
        dataloader: DataLoader,
        n_batches: int = 50,
        critical_threshold: float = 0.1,
    ) -> CircuitAnalysis:
        """
        Analyze which layers implement a behavior circuit.

        Args:
            circuit: Circuit specification
            dataloader: DataLoader for inputs
            n_batches: Number of batches to analyze
            critical_threshold: Threshold for "critical" layer designation

        Returns:
            CircuitAnalysis with layer importance
        """
        from interp.hooks import HookManager, HookPoint, HookPointType
        from train.batch_utils import build_model_inputs

        n_layers = len(self.model.blocks)
        hook_points = [
            HookPoint(HookPointType.BLOCK_OUTPUT, i)
            for i in range(n_layers)
        ]

        # Accumulators
        layer_causal_effects = [0.0] * n_layers
        layer_ablation_effects = [0.0] * n_layers
        behavior_count = 0
        total_count = 0

        hook_manager = HookManager(self.model)
        self.model.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= n_batches:
                    break

                X = batch["X"].to(self.device)
                batch_size = X.shape[0]
                total_count += batch_size

                # Get baseline outputs and detect behavior
                inputs_td = build_model_inputs(X, self.colmap)
                outputs = self.model(inputs_td)
                behavior_mask = circuit.detect(X, outputs)
                behavior_count += behavior_mask.sum().item()

                # Skip if no behavior in this batch
                if not behavior_mask.any():
                    continue

                # Cache baseline activations
                hook_manager.install_hooks(hook_points)
                _ = self.model(inputs_td)
                baseline_acts = {
                    point: hook_manager.get_single(point).clone()
                    for point in hook_points
                }
                hook_manager.clear()
                hook_manager.remove_hooks()

                # For each layer, measure effect of ablation
                for layer_idx in range(n_layers):
                    point = hook_points[layer_idx]

                    # Mean ablation
                    ablated_act = baseline_acts[point].mean(dim=1, keepdim=True)
                    ablated_act = ablated_act.expand_as(baseline_acts[point])

                    # Run with ablated activations
                    def patch_hook(module, inp, out):
                        return ablated_act

                    block = self.model.blocks[layer_idx]
                    handle = block.register_forward_hook(patch_hook)

                    try:
                        ablated_outputs = self.model(inputs_td)
                        ablated_mask = circuit.detect(X, ablated_outputs)

                        # Effect = how many behaviors were lost
                        lost = (behavior_mask & ~ablated_mask).sum().item()
                        effect = lost / max(behavior_mask.sum().item(), 1)
                        layer_ablation_effects[layer_idx] += effect

                    finally:
                        handle.remove()

        # Normalize
        n_behavior_batches = max(1, behavior_count / (total_count / n_batches))
        layer_ablation_effects = [e / n_behavior_batches for e in layer_ablation_effects]

        # Create layer importance list
        layer_importance = []
        for layer_idx in range(n_layers):
            importance = LayerImportance(
                layer_idx=layer_idx,
                causal_effect=layer_ablation_effects[layer_idx],  # Using ablation as proxy
                ablation_effect=layer_ablation_effects[layer_idx],
                attention_contribution=0.0,  # Would need separate analysis
            )
            layer_importance.append(importance)

        # Find critical layers
        critical_layers = [
            li.layer_idx for li in layer_importance
            if li.combined_score >= critical_threshold
        ]

        return CircuitAnalysis(
            circuit=circuit,
            n_samples=total_count,
            behavior_rate=behavior_count / max(total_count, 1),
            layer_importance=layer_importance,
            critical_layers=critical_layers,
        )

    def find_circuit_features(
        self,
        circuit: CircuitSpec,
        sae: "TopKSparseAutoencoder",
        hook_point: "HookPoint",
        dataloader: DataLoader,
        n_batches: int = 50,
        top_k: int = 50,
    ) -> List[FeatureImportance]:
        """
        Find SAE features that implement a circuit.

        Compares feature activations during behavior vs baseline.

        Args:
            circuit: Circuit specification
            sae: Trained sparse autoencoder
            hook_point: Where SAE is applied
            dataloader: DataLoader for inputs
            n_batches: Number of batches to analyze
            top_k: Number of top features to return

        Returns:
            List of FeatureImportance sorted by importance
        """
        from interp.hooks import HookManager
        from train.batch_utils import build_model_inputs

        hook_manager = HookManager(self.model)
        hook_manager.install_hooks([hook_point])

        n_features = sae.hidden_dim
        behavior_activations = torch.zeros(n_features, device=self.device)
        baseline_activations = torch.zeros(n_features, device=self.device)
        behavior_count = 0
        baseline_count = 0

        self.model.eval()
        sae.eval()

        try:
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    if batch_idx >= n_batches:
                        break

                    X = batch["X"].to(self.device)

                    # Forward pass
                    inputs_td = build_model_inputs(X, self.colmap)
                    outputs = self.model(inputs_td)

                    # Get activations
                    acts = hook_manager.get_single(hook_point)[:, -1, :]
                    hook_manager.clear()

                    # SAE encode - returns (sparse_acts, topk_vals, topk_idx)
                    sae_acts, _, _ = sae.encode(acts)

                    # Detect behavior (ensure 1D mask matching batch size)
                    behavior_mask = circuit.detect(X, outputs)
                    if behavior_mask.dim() > 1:
                        behavior_mask = behavior_mask[:, -1]
                    behavior_mask = behavior_mask.bool()

                    # Accumulate for behavior samples
                    n_behavior = behavior_mask.sum().item()
                    if n_behavior > 0:
                        behavior_sae = sae_acts[behavior_mask]
                        behavior_activations += behavior_sae.sum(dim=0)
                        behavior_count += n_behavior

                    # Accumulate for baseline samples
                    baseline_mask = ~behavior_mask
                    n_baseline = baseline_mask.sum().item()
                    if n_baseline > 0:
                        baseline_sae = sae_acts[baseline_mask]
                        baseline_activations += baseline_sae.sum(dim=0)
                        baseline_count += n_baseline

        finally:
            hook_manager.remove_hooks()

        # Compute average activations
        if behavior_count > 0:
            behavior_activations /= behavior_count
        if baseline_count > 0:
            baseline_activations /= baseline_count

        # Compute specificity (how much more active during behavior)
        epsilon = 1e-6
        specificity = (behavior_activations + epsilon) / (baseline_activations + epsilon)

        # Create feature importance list
        feature_importance = []
        for feat_idx in range(n_features):
            importance = FeatureImportance(
                feature_idx=feat_idx,
                activation_rate=behavior_activations[feat_idx].item(),
                specificity=specificity[feat_idx].item(),
                causal_effect=0.0,  # Would need ablation study
            )
            feature_importance.append(importance)

        # Sort by combined score
        feature_importance.sort(key=lambda x: -x.combined_score)

        return feature_importance[:top_k]

    def compare_circuits(
        self,
        circuits: List[CircuitSpec],
        dataloader: DataLoader,
        n_batches: int = 30,
    ) -> Dict[str, CircuitAnalysis]:
        """
        Compare multiple circuits to find shared vs unique layers.

        Args:
            circuits: List of circuits to compare
            dataloader: DataLoader for inputs
            n_batches: Number of batches per circuit

        Returns:
            Dictionary mapping circuit name to analysis
        """
        analyses = {}
        for circuit in circuits:
            analysis = self.analyze_circuit(circuit, dataloader, n_batches)
            analyses[circuit.name] = analysis

        return analyses

    def find_shared_layers(
        self,
        analyses: Dict[str, CircuitAnalysis],
        threshold: float = 0.1,
    ) -> List[int]:
        """
        Find layers that are critical for multiple circuits.

        Args:
            analyses: Dictionary of circuit analyses
            threshold: Minimum importance to count as critical

        Returns:
            List of layer indices important for multiple circuits
        """
        layer_counts: Dict[int, int] = {}

        for analysis in analyses.values():
            for li in analysis.layer_importance:
                if li.combined_score >= threshold:
                    layer_counts[li.layer_idx] = layer_counts.get(li.layer_idx, 0) + 1

        # Return layers critical for more than half the circuits
        n_circuits = len(analyses)
        shared = [
            layer_idx for layer_idx, count in layer_counts.items()
            if count > n_circuits // 2
        ]

        return sorted(shared)

    def find_unique_layers(
        self,
        analyses: Dict[str, CircuitAnalysis],
        threshold: float = 0.1,
    ) -> Dict[str, List[int]]:
        """
        Find layers that are uniquely critical for each circuit.

        Args:
            analyses: Dictionary of circuit analyses
            threshold: Minimum importance to count as critical

        Returns:
            Dictionary mapping circuit name to unique critical layers
        """
        # Get critical layers for each circuit
        circuit_layers: Dict[str, set] = {}
        for name, analysis in analyses.items():
            circuit_layers[name] = {
                li.layer_idx for li in analysis.layer_importance
                if li.combined_score >= threshold
            }

        # Find unique layers
        unique = {}
        for name, layers in circuit_layers.items():
            other_layers = set()
            for other_name, other in circuit_layers.items():
                if other_name != name:
                    other_layers |= other
            unique[name] = sorted(layers - other_layers)

        return unique


def create_custom_circuit(
    name: str,
    description: str,
    output_head: str,
    target_classes: List[int],
    input_condition: Optional[InputCondition] = None,
) -> CircuitSpec:
    """
    Create a custom circuit for specific output classes.

    Args:
        name: Circuit name
        description: Human-readable description
        output_head: Which output head to check
        target_classes: List of class indices that trigger the circuit
        input_condition: Optional input condition

    Returns:
        CircuitSpec for the custom circuit
    """
    target_set = set(target_classes)

    def detector(inputs: Tensor, outputs: "TensorDict") -> Tensor:
        pred = _get_output_prediction(outputs, output_head)
        mask = torch.zeros_like(pred, dtype=torch.bool)
        for target in target_set:
            mask = mask | (pred == target)
        return mask

    return CircuitSpec(
        name=name,
        description=description,
        detector=detector,
        input_condition=input_condition,
        output_heads=[output_head],
    )


__all__ = [
    "CircuitSpec",
    "CircuitAnalysis",
    "LayerImportance",
    "FeatureImportance",
    "CircuitAnalyzer",
    "MELEE_CIRCUITS",
    "get_melee_circuits",
    "create_custom_circuit",
    # Detectors
    "detect_wavedash",
    "detect_shield",
    "detect_aerial",
    "detect_grab",
    "detect_dash",
    "detect_neutral_stick",
    # Conditions
    "make_offstage_condition",
    "make_opponent_offstage_condition",
    "make_shielding_condition",
]
