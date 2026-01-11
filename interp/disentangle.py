"""
P1/P2 disentanglement analysis.

Provides tools to understand how the model separates player information:
- How much attention goes to P1 vs P2 features?
- What happens when we swap players?
- Which SAE features encode opponent state?
- How sensitive is the output to each player's features?

This helps answer questions like:
- "Is the model thinking about the opponent or just itself?"
- "Which layers do opponent modeling?"
- "What features represent opponent state?"

Usage:
    from interp.disentangle import EntityDisentangler

    disentangler = EntityDisentangler(model, colmap, device)

    # How much does the model attend to the opponent?
    attention = disentangler.compute_player_attention_ratio(inputs, layer_idx=4)
    print(f"P1 attention: {attention.p1_ratio:.1%}, P2: {attention.p2_ratio:.1%}")

    # What happens if we swap players?
    swap = disentangler.analyze_swap_effect(inputs)
    print(f"Prediction changed: {swap.prediction_changed}")

    # Find SAE features that encode opponent state
    features = disentangler.find_opponent_modeling_features(sae, dataset)
    print(f"Top opponent features: {features.top_features[:5]}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from column_map import ColumnMap
    from interp.hooks import HookPoint
    from interp.sae.topk import TopKSparseAutoencoder
    from model.nano_gpt import GPT


@dataclass
class PlayerAttentionRatio:
    """Attention distribution between P1, P2, and common features."""

    layer_idx: int
    p1_ratio: float  # Fraction of attention to P1 features
    p2_ratio: float  # Fraction of attention to P2 features
    common_ratio: float  # Fraction to common features (stage, etc.)

    # Per-head breakdown
    per_head_p1: List[float] = field(default_factory=list)
    per_head_p2: List[float] = field(default_factory=list)
    per_head_common: List[float] = field(default_factory=list)

    @property
    def opponent_modeling_ratio(self) -> float:
        """How much the model attends to opponent vs self."""
        total_player = self.p1_ratio + self.p2_ratio
        if total_player < 1e-8:
            return 0.5
        return self.p2_ratio / total_player

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Layer {self.layer_idx} Attention Distribution:",
            f"  P1 (self):    {self.p1_ratio:.1%}",
            f"  P2 (opponent):{self.p2_ratio:.1%}",
            f"  Common:       {self.common_ratio:.1%}",
            f"  Opponent modeling ratio: {self.opponent_modeling_ratio:.1%}",
        ]
        return "\n".join(lines)


@dataclass
class SwapEffect:
    """Effect of swapping P1 and P2 features."""

    # Predictions
    original_prediction: Dict[str, int]
    swapped_prediction: Dict[str, int]
    prediction_changed: Dict[str, bool]

    # Confidence changes
    original_confidence: Dict[str, float]
    swapped_confidence: Dict[str, float]

    # Logit space changes
    logit_l2_distance: Dict[str, float]
    logit_cosine_similarity: Dict[str, float]

    # Activation changes per layer
    activation_l2_per_layer: Dict[int, float] = field(default_factory=dict)

    @property
    def any_prediction_changed(self) -> bool:
        """Whether any head's prediction changed."""
        return any(self.prediction_changed.values())

    @property
    def total_logit_change(self) -> float:
        """Sum of logit L2 distances across heads."""
        return sum(self.logit_l2_distance.values())

    def summary(self) -> str:
        """Human-readable summary."""
        lines = ["PLAYER SWAP EFFECT", "=" * 40]

        for head in self.original_prediction:
            orig = self.original_prediction[head]
            swap = self.swapped_prediction[head]
            changed = self.prediction_changed[head]
            marker = "CHANGED" if changed else "same"

            lines.append(f"\n{head}:")
            lines.append(f"  Original: {orig} -> Swapped: {swap} [{marker}]")
            lines.append(f"  Logit L2: {self.logit_l2_distance.get(head, 0):.3f}")
            lines.append(
                f"  Cosine sim: {self.logit_cosine_similarity.get(head, 0):.3f}"
            )

        if self.activation_l2_per_layer:
            lines.append("\nActivation changes per layer:")
            for layer_idx in sorted(self.activation_l2_per_layer.keys()):
                change = self.activation_l2_per_layer[layer_idx]
                bar = "█" * int(change * 20)
                lines.append(f"  Layer {layer_idx}: {change:.3f} {bar}")

        return "\n".join(lines)


@dataclass
class PlayerSensitivity:
    """Sensitivity of outputs to P1 vs P2 feature perturbations."""

    head_name: str
    p1_sensitivity: float  # Output change per unit P1 perturbation
    p2_sensitivity: float  # Output change per unit P2 perturbation

    # Per-feature sensitivities
    p1_feature_sensitivity: Dict[str, float] = field(default_factory=dict)
    p2_feature_sensitivity: Dict[str, float] = field(default_factory=dict)

    @property
    def sensitivity_ratio(self) -> float:
        """P2 sensitivity relative to P1."""
        total = self.p1_sensitivity + self.p2_sensitivity
        if total < 1e-8:
            return 0.5
        return self.p2_sensitivity / total

    def top_p1_features(self, k: int = 5) -> List[Tuple[str, float]]:
        """Top k most sensitive P1 features."""
        sorted_feats = sorted(
            self.p1_feature_sensitivity.items(), key=lambda x: -x[1]
        )
        return sorted_feats[:k]

    def top_p2_features(self, k: int = 5) -> List[Tuple[str, float]]:
        """Top k most sensitive P2 features."""
        sorted_feats = sorted(
            self.p2_feature_sensitivity.items(), key=lambda x: -x[1]
        )
        return sorted_feats[:k]


@dataclass
class OpponentModelingFeatures:
    """SAE features that encode opponent state."""

    # Features that activate more for P2 changes
    opponent_features: List[int]
    opponent_scores: List[float]

    # Features that activate more for P1 changes
    self_features: List[int]
    self_scores: List[float]

    # Features invariant to player swap
    invariant_features: List[int]
    invariant_scores: List[float]

    def top_opponent_features(self, k: int = 10) -> List[Tuple[int, float]]:
        """Top k features most responsive to opponent state."""
        return list(zip(self.opponent_features[:k], self.opponent_scores[:k]))

    def top_self_features(self, k: int = 10) -> List[Tuple[int, float]]:
        """Top k features most responsive to self state."""
        return list(zip(self.self_features[:k], self.self_scores[:k]))

    def summary(self) -> str:
        """Human-readable summary."""
        lines = ["OPPONENT MODELING FEATURES", "=" * 40]

        lines.append("\nTop opponent-responsive features:")
        for feat_idx, score in self.top_opponent_features(5):
            lines.append(f"  Feature {feat_idx}: {score:.3f}")

        lines.append("\nTop self-responsive features:")
        for feat_idx, score in self.top_self_features(5):
            lines.append(f"  Feature {feat_idx}: {score:.3f}")

        lines.append(f"\nInvariant features: {len(self.invariant_features)}")

        return "\n".join(lines)


@dataclass
class DisentanglementSummary:
    """Complete disentanglement analysis."""

    attention_by_layer: Dict[int, PlayerAttentionRatio]
    swap_effect: SwapEffect
    sensitivity: Dict[str, PlayerSensitivity]
    opponent_features: Optional[OpponentModelingFeatures] = None

    def mean_opponent_attention(self) -> float:
        """Mean opponent attention ratio across layers."""
        if not self.attention_by_layer:
            return 0.0
        ratios = [a.opponent_modeling_ratio for a in self.attention_by_layer.values()]
        return sum(ratios) / len(ratios)

    def summary(self) -> str:
        """Comprehensive summary."""
        lines = [
            "=" * 60,
            "P1/P2 DISENTANGLEMENT ANALYSIS",
            "=" * 60,
        ]

        # Attention summary
        lines.append("\n--- ATTENTION ANALYSIS ---")
        lines.append(f"Mean opponent attention: {self.mean_opponent_attention():.1%}")
        lines.append("\nBy layer:")
        for layer_idx in sorted(self.attention_by_layer.keys()):
            ratio = self.attention_by_layer[layer_idx]
            bar = "█" * int(ratio.opponent_modeling_ratio * 40)
            lines.append(
                f"  Layer {layer_idx}: P1={ratio.p1_ratio:.1%} "
                f"P2={ratio.p2_ratio:.1%} {bar}"
            )

        # Swap effect summary
        lines.append("\n--- SWAP EFFECT ---")
        lines.append(f"Predictions changed: {self.swap_effect.any_prediction_changed}")
        for head, changed in self.swap_effect.prediction_changed.items():
            lines.append(f"  {head}: {'CHANGED' if changed else 'same'}")

        # Sensitivity summary
        lines.append("\n--- SENSITIVITY ---")
        for head, sens in self.sensitivity.items():
            lines.append(
                f"  {head}: P1={sens.p1_sensitivity:.3f} "
                f"P2={sens.p2_sensitivity:.3f} "
                f"(ratio={sens.sensitivity_ratio:.1%})"
            )

        return "\n".join(lines)


class EntityDisentangler:
    """
    Analyze how the model separates P1 and P2 information.

    Provides methods to understand opponent modeling:
    - Attention analysis: How much does each layer attend to P1 vs P2?
    - Swap analysis: What changes when we swap P1 and P2?
    - Sensitivity analysis: How sensitive is output to each player?
    - Feature analysis: Which SAE features encode opponent state?
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

        # Cache player feature indices
        self._p1_indices = [
            i for i, name in enumerate(colmap.feat_names)
            if name.startswith("p1_")
        ]
        self._p2_indices = [
            i for i, name in enumerate(colmap.feat_names)
            if name.startswith("p2_")
        ]
        self._common_indices = [
            i for i, name in enumerate(colmap.feat_names)
            if not name.startswith("p1_") and not name.startswith("p2_")
        ]

        # Map P1 features to corresponding P2 features
        self._p1_to_p2_map = self._build_player_feature_map()

    def _build_player_feature_map(self) -> Dict[int, int]:
        """Build mapping from P1 feature indices to P2 feature indices."""
        p1_to_p2 = {}
        for i, name in enumerate(self.colmap.feat_names):
            if name.startswith("p1_"):
                p2_name = "p2_" + name[3:]
                try:
                    p2_idx = self.colmap.feat_names.index(p2_name)
                    p1_to_p2[i] = p2_idx
                except ValueError:
                    pass  # No corresponding P2 feature
        return p1_to_p2

    def _prepare_inputs(self, inputs: Tensor) -> Tensor:
        """Ensure inputs have batch dimension and are on device."""
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(0)
        return inputs.to(self.device)

    def swap_players(self, inputs: Tensor) -> Tensor:
        """
        Swap P1 and P2 features in the input.

        Args:
            inputs: Input tensor [batch, seq, features]

        Returns:
            Tensor with P1 and P2 features swapped
        """
        swapped = inputs.clone()

        for p1_idx, p2_idx in self._p1_to_p2_map.items():
            temp = swapped[..., p1_idx].clone()
            swapped[..., p1_idx] = swapped[..., p2_idx]
            swapped[..., p2_idx] = temp

        return swapped

    def compute_player_attention_ratio(
        self,
        inputs: Tensor,
        layer_idx: int,
        position: int = -1,
    ) -> PlayerAttentionRatio:
        """
        Compute how much attention goes to P1 vs P2 vs common features.

        Note: In this architecture, all features are mixed at each position,
        so we use temporal attention as a proxy. Earlier positions may
        correlate with different feature influences.

        For true per-feature attention, you'd need per-token embeddings
        for each feature.

        Args:
            inputs: Input tensor [batch, seq, features]
            layer_idx: Which layer to analyze
            position: Query position

        Returns:
            PlayerAttentionRatio with attention breakdown
        """
        from interp.attention import AttentionExtractor

        inputs = self._prepare_inputs(inputs)
        seq_len = inputs.shape[1]

        if position < 0:
            position = seq_len + position

        extractor = AttentionExtractor(self.model, self.colmap, self.device)
        pattern = extractor.get_layer_attention(inputs, layer_idx)

        # Get attention weights for query position
        # Shape: [batch, heads, seq_len]
        attn = pattern.get_attention_to_position(position)

        # Compute per-head ratios using temporal proxy
        # Divide sequence into thirds: early=common, middle=P2, recent=P1
        third = seq_len // 3
        early_end = third
        middle_end = 2 * third

        per_head_p1 = []
        per_head_p2 = []
        per_head_common = []

        for head_idx in range(pattern.num_heads):
            head_attn = attn[0, head_idx, :]  # [seq_len]

            common_attn = head_attn[:early_end].sum().item()
            p2_attn = head_attn[early_end:middle_end].sum().item()
            p1_attn = head_attn[middle_end:].sum().item()

            total = common_attn + p2_attn + p1_attn
            if total > 0:
                per_head_common.append(common_attn / total)
                per_head_p2.append(p2_attn / total)
                per_head_p1.append(p1_attn / total)
            else:
                per_head_common.append(1/3)
                per_head_p2.append(1/3)
                per_head_p1.append(1/3)

        # Mean across heads
        mean_p1 = sum(per_head_p1) / len(per_head_p1) if per_head_p1 else 1/3
        mean_p2 = sum(per_head_p2) / len(per_head_p2) if per_head_p2 else 1/3
        mean_common = sum(per_head_common) / len(per_head_common) if per_head_common else 1/3

        return PlayerAttentionRatio(
            layer_idx=layer_idx,
            p1_ratio=mean_p1,
            p2_ratio=mean_p2,
            common_ratio=mean_common,
            per_head_p1=per_head_p1,
            per_head_p2=per_head_p2,
            per_head_common=per_head_common,
        )

    def analyze_swap_effect(
        self,
        inputs: Tensor,
        position: int = -1,
        track_activations: bool = True,
    ) -> SwapEffect:
        """
        Analyze how predictions change when P1 and P2 are swapped.

        This reveals how much the model's output depends on which
        player is which, vs just the game state.

        Args:
            inputs: Input tensor [batch, seq, features]
            position: Sequence position to analyze
            track_activations: Also track per-layer activation changes

        Returns:
            SwapEffect with prediction and activation changes
        """
        from train.batch_utils import build_model_inputs
        from interp.hooks import HookManager, HookPoint, HookPointType

        inputs = self._prepare_inputs(inputs)
        swapped = self.swap_players(inputs)

        seq_len = inputs.shape[1]
        if position < 0:
            position = seq_len + position

        self.model.eval()

        # Set up activation tracking
        if track_activations:
            n_layers = len(self.model.blocks)
            hook_points = [
                HookPoint(HookPointType.BLOCK_OUTPUT, i)
                for i in range(n_layers)
            ]
            hook_manager = HookManager(self.model)
            hook_manager.install_hooks(hook_points)

        try:
            with torch.no_grad():
                # Original forward pass
                orig_td = build_model_inputs(inputs, self.colmap)
                orig_outputs = self.model(orig_td)

                if track_activations:
                    orig_activations = {
                        point: hook_manager.get_single(point).clone()
                        for point in hook_points
                    }
                    hook_manager.clear()

                # Swapped forward pass
                swap_td = build_model_inputs(swapped, self.colmap)
                swap_outputs = self.model(swap_td)

                if track_activations:
                    swap_activations = {
                        point: hook_manager.get_single(point).clone()
                        for point in hook_points
                    }
                    hook_manager.clear()

        finally:
            if track_activations:
                hook_manager.remove_hooks()

        # Compare outputs
        original_prediction = {}
        swapped_prediction = {}
        prediction_changed = {}
        original_confidence = {}
        swapped_confidence = {}
        logit_l2_distance = {}
        logit_cosine_similarity = {}

        for head in ["main_stick", "c_stick", "buttons", "shoulder"]:
            if head not in orig_outputs:
                continue

            orig_logits = orig_outputs[head][0, position]
            swap_logits = swap_outputs[head][0, position]

            # Predictions
            orig_pred = orig_logits.argmax().item()
            swap_pred = swap_logits.argmax().item()
            original_prediction[head] = orig_pred
            swapped_prediction[head] = swap_pred
            prediction_changed[head] = orig_pred != swap_pred

            # Confidence
            orig_probs = F.softmax(orig_logits, dim=-1)
            swap_probs = F.softmax(swap_logits, dim=-1)
            original_confidence[head] = orig_probs.max().item()
            swapped_confidence[head] = swap_probs.max().item()

            # Logit space metrics
            logit_l2_distance[head] = (orig_logits - swap_logits).norm().item()

            # Cosine similarity
            cos_sim = F.cosine_similarity(
                orig_logits.unsqueeze(0), swap_logits.unsqueeze(0)
            ).item()
            logit_cosine_similarity[head] = cos_sim

        # Activation changes per layer
        activation_l2_per_layer = {}
        if track_activations:
            for point in hook_points:
                orig_act = orig_activations[point][0, position]
                swap_act = swap_activations[point][0, position]
                l2_dist = (orig_act - swap_act).norm().item()
                # Normalize by dimension
                l2_dist /= orig_act.numel() ** 0.5
                activation_l2_per_layer[point.layer_idx] = l2_dist

        return SwapEffect(
            original_prediction=original_prediction,
            swapped_prediction=swapped_prediction,
            prediction_changed=prediction_changed,
            original_confidence=original_confidence,
            swapped_confidence=swapped_confidence,
            logit_l2_distance=logit_l2_distance,
            logit_cosine_similarity=logit_cosine_similarity,
            activation_l2_per_layer=activation_l2_per_layer,
        )

    def compute_player_sensitivity(
        self,
        inputs: Tensor,
        head_name: str = "main_stick",
        position: int = -1,
        perturbation_scale: float = 0.1,
        n_samples: int = 10,
    ) -> PlayerSensitivity:
        """
        Compute sensitivity of output to P1 vs P2 feature perturbations.

        Uses finite differences to estimate gradients.

        Args:
            inputs: Input tensor [batch, seq, features]
            head_name: Which output head to analyze
            position: Sequence position
            perturbation_scale: Size of perturbations
            n_samples: Number of random perturbations to average

        Returns:
            PlayerSensitivity with per-player and per-feature sensitivities
        """
        from train.batch_utils import build_model_inputs

        inputs = self._prepare_inputs(inputs)
        seq_len = inputs.shape[1]
        if position < 0:
            position = seq_len + position

        self.model.eval()

        # Get baseline output
        with torch.no_grad():
            base_td = build_model_inputs(inputs, self.colmap)
            base_outputs = self.model(base_td)
            base_logits = base_outputs[head_name][0, position]

        p1_sensitivities = []
        p2_sensitivities = []
        p1_feature_sens = {self.colmap.feat_names[i]: [] for i in self._p1_indices}
        p2_feature_sens = {self.colmap.feat_names[i]: [] for i in self._p2_indices}

        for _ in range(n_samples):
            # Perturb P1 features
            p1_perturbed = inputs.clone()
            p1_noise = torch.randn(len(self._p1_indices), device=self.device)
            p1_noise = p1_noise / p1_noise.norm() * perturbation_scale

            for idx, feat_idx in enumerate(self._p1_indices):
                p1_perturbed[0, position, feat_idx] += p1_noise[idx]

            with torch.no_grad():
                p1_td = build_model_inputs(p1_perturbed, self.colmap)
                p1_outputs = self.model(p1_td)
                p1_logits = p1_outputs[head_name][0, position]

            p1_change = (p1_logits - base_logits).norm().item()
            p1_sensitivities.append(p1_change / perturbation_scale)

            # Perturb P2 features
            p2_perturbed = inputs.clone()
            p2_noise = torch.randn(len(self._p2_indices), device=self.device)
            p2_noise = p2_noise / p2_noise.norm() * perturbation_scale

            for idx, feat_idx in enumerate(self._p2_indices):
                p2_perturbed[0, position, feat_idx] += p2_noise[idx]

            with torch.no_grad():
                p2_td = build_model_inputs(p2_perturbed, self.colmap)
                p2_outputs = self.model(p2_td)
                p2_logits = p2_outputs[head_name][0, position]

            p2_change = (p2_logits - base_logits).norm().item()
            p2_sensitivities.append(p2_change / perturbation_scale)

        # Per-feature sensitivity (using individual perturbations)
        for feat_idx in self._p1_indices:
            feat_name = self.colmap.feat_names[feat_idx]
            perturbed = inputs.clone()
            perturbed[0, position, feat_idx] += perturbation_scale

            with torch.no_grad():
                td = build_model_inputs(perturbed, self.colmap)
                outputs = self.model(td)
                logits = outputs[head_name][0, position]

            change = (logits - base_logits).norm().item()
            p1_feature_sens[feat_name] = change / perturbation_scale

        for feat_idx in self._p2_indices:
            feat_name = self.colmap.feat_names[feat_idx]
            perturbed = inputs.clone()
            perturbed[0, position, feat_idx] += perturbation_scale

            with torch.no_grad():
                td = build_model_inputs(perturbed, self.colmap)
                outputs = self.model(td)
                logits = outputs[head_name][0, position]

            change = (logits - base_logits).norm().item()
            p2_feature_sens[feat_name] = change / perturbation_scale

        return PlayerSensitivity(
            head_name=head_name,
            p1_sensitivity=sum(p1_sensitivities) / len(p1_sensitivities),
            p2_sensitivity=sum(p2_sensitivities) / len(p2_sensitivities),
            p1_feature_sensitivity=p1_feature_sens,
            p2_feature_sensitivity=p2_feature_sens,
        )

    def find_opponent_modeling_features(
        self,
        sae: "TopKSparseAutoencoder",
        dataloader: DataLoader,
        hook_point: "HookPoint",
        n_batches: int = 50,
        threshold: float = 0.1,
    ) -> OpponentModelingFeatures:
        """
        Find SAE features that encode opponent (P2) state.

        Compares SAE activations on original vs player-swapped inputs
        to identify features that respond differently.

        Args:
            sae: Trained sparse autoencoder
            dataloader: DataLoader for inputs
            hook_point: Where SAE is applied
            n_batches: Number of batches to analyze
            threshold: Minimum activation difference to count

        Returns:
            OpponentModelingFeatures with feature lists and scores
        """
        from interp.hooks import HookManager
        from train.batch_utils import build_model_inputs

        hook_manager = HookManager(self.model)
        hook_manager.install_hooks([hook_point])

        # Track activation differences per feature
        n_features = sae.num_features
        p2_response = torch.zeros(n_features, device=self.device)
        p1_response = torch.zeros(n_features, device=self.device)
        feature_counts = torch.zeros(n_features, device=self.device)

        self.model.eval()
        sae.eval()

        try:
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    if batch_idx >= n_batches:
                        break

                    X = batch["X"].to(self.device)
                    swapped = self.swap_players(X)

                    # Original activations
                    orig_td = build_model_inputs(X, self.colmap)
                    _ = self.model(orig_td)
                    orig_acts = hook_manager.get_single(hook_point)
                    hook_manager.clear()

                    # Swapped activations
                    swap_td = build_model_inputs(swapped, self.colmap)
                    _ = self.model(swap_td)
                    swap_acts = hook_manager.get_single(hook_point)
                    hook_manager.clear()

                    # Get SAE features for last position
                    orig_sae = sae.encode(orig_acts[:, -1, :])
                    swap_sae = sae.encode(swap_acts[:, -1, :])

                    # Features that increase with swap = opponent features
                    # Features that decrease with swap = self features
                    diff = swap_sae - orig_sae  # [batch, n_features]

                    # Accumulate
                    p2_response += diff.clamp(min=0).sum(dim=0)
                    p1_response += (-diff).clamp(min=0).sum(dim=0)
                    feature_counts += (orig_sae > threshold).float().sum(dim=0)

        finally:
            hook_manager.remove_hooks()

        # Normalize by counts
        p2_scores = p2_response / (feature_counts + 1)
        p1_scores = p1_response / (feature_counts + 1)

        # Sort by score
        p2_sorted = torch.argsort(p2_scores, descending=True)
        p1_sorted = torch.argsort(p1_scores, descending=True)

        # Find invariant features (small difference either way)
        total_diff = p2_response + p1_response
        invariant_mask = total_diff < threshold * feature_counts
        invariant_indices = torch.where(invariant_mask)[0]

        return OpponentModelingFeatures(
            opponent_features=p2_sorted.tolist(),
            opponent_scores=p2_scores[p2_sorted].tolist(),
            self_features=p1_sorted.tolist(),
            self_scores=p1_scores[p1_sorted].tolist(),
            invariant_features=invariant_indices.tolist(),
            invariant_scores=[0.0] * len(invariant_indices),
        )

    def full_analysis(
        self,
        inputs: Tensor,
        position: int = -1,
        layers: Optional[List[int]] = None,
    ) -> DisentanglementSummary:
        """
        Run complete disentanglement analysis.

        Args:
            inputs: Input tensor [batch, seq, features]
            position: Sequence position to analyze
            layers: Which layers to analyze (None = all)

        Returns:
            DisentanglementSummary with all analyses
        """
        inputs = self._prepare_inputs(inputs)

        n_layers = len(self.model.blocks)
        if layers is None:
            layers = list(range(n_layers))

        # Attention analysis per layer
        attention_by_layer = {}
        for layer_idx in layers:
            attention_by_layer[layer_idx] = self.compute_player_attention_ratio(
                inputs, layer_idx, position
            )

        # Swap effect
        swap_effect = self.analyze_swap_effect(inputs, position)

        # Sensitivity per head
        sensitivity = {}
        for head in ["main_stick", "c_stick", "buttons", "shoulder"]:
            sensitivity[head] = self.compute_player_sensitivity(
                inputs, head, position
            )

        return DisentanglementSummary(
            attention_by_layer=attention_by_layer,
            swap_effect=swap_effect,
            sensitivity=sensitivity,
        )


def analyze_opponent_modeling(
    model: "GPT",
    colmap: "ColumnMap",
    dataloader: DataLoader,
    device: torch.device,
    n_samples: int = 100,
) -> Dict[str, float]:
    """
    Quick analysis of opponent modeling across a dataset.

    Returns summary statistics about how much the model attends to
    and is sensitive to opponent state.

    Args:
        model: The GPT model
        colmap: Column mapping
        dataloader: DataLoader for inputs
        device: Torch device
        n_samples: Number of samples to analyze

    Returns:
        Dictionary with summary statistics
    """
    disentangler = EntityDisentangler(model, colmap, device)

    swap_changed_count = 0
    total_samples = 0
    p2_attention_ratios = []
    p2_sensitivity_ratios = []

    for batch in dataloader:
        if total_samples >= n_samples:
            break

        X = batch["X"].to(device)
        batch_size = X.shape[0]

        for i in range(min(batch_size, n_samples - total_samples)):
            inputs = X[i:i+1]

            # Swap analysis
            swap = disentangler.analyze_swap_effect(inputs, track_activations=False)
            if swap.any_prediction_changed:
                swap_changed_count += 1

            # Attention (just middle layer)
            n_layers = len(model.blocks)
            attn = disentangler.compute_player_attention_ratio(
                inputs, n_layers // 2
            )
            p2_attention_ratios.append(attn.opponent_modeling_ratio)

            # Sensitivity (just main_stick)
            sens = disentangler.compute_player_sensitivity(
                inputs, "main_stick", n_samples=3
            )
            p2_sensitivity_ratios.append(sens.sensitivity_ratio)

            total_samples += 1

    return {
        "swap_change_rate": swap_changed_count / total_samples if total_samples > 0 else 0,
        "mean_p2_attention": sum(p2_attention_ratios) / len(p2_attention_ratios) if p2_attention_ratios else 0,
        "mean_p2_sensitivity": sum(p2_sensitivity_ratios) / len(p2_sensitivity_ratios) if p2_sensitivity_ratios else 0,
        "n_samples": total_samples,
    }


__all__ = [
    "EntityDisentangler",
    "PlayerAttentionRatio",
    "SwapEffect",
    "PlayerSensitivity",
    "OpponentModelingFeatures",
    "DisentanglementSummary",
    "analyze_opponent_modeling",
]
