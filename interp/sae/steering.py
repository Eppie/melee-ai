"""
SAE-based activation steering for intervention experiments.

Enables modifying model behavior by injecting SAE feature directions
into the residual stream during inference. Use cases:
- Amplify "recovery" feature to test more aggressive recoveries
- Suppress "approach" feature to see if model camps
- Measure off-target effects on value head output
- Find the steering "sweet spot" where behavior changes without capability loss

Usage:
    from interp.sae.steering import SAESteering

    steering = SAESteering(model, sae, hook_point, device)

    # Amplify a feature
    with steering.steer(feature_idx=42, strength=2.0):
        outputs = model(inputs)

    # Ablate (remove) a feature
    with steering.ablate(feature_idx=42):
        outputs = model(inputs)

    # Steer with multiple features
    with steering.multi_steer({42: 2.0, 17: -1.0}):
        outputs = model(inputs)

    # Find optimal steering strength
    sweet_spot = steering.find_steering_sweet_spot(
        feature_idx=42,
        dataloader=loader,
        colmap=colmap,
    )
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, Generator, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from interp.hooks import HookPoint, HookPointType

if TYPE_CHECKING:
    from column_map import ColumnMap
    from model.nano_gpt import GPT
    from interp.sae.topk import TopKSparseAutoencoder


@dataclass
class SteeringEffect:
    """Results from measuring steering effect on model outputs."""

    feature_idx: int
    strength: float

    # Action distribution changes
    main_stick_kl: float  # KL divergence from baseline
    c_stick_kl: float
    buttons_diff: float  # L1 diff in button probabilities
    shoulder_kl: float

    # Value head change
    value_delta: float  # Change in value prediction

    # Overall capability metric
    total_action_change: float  # Combined action distribution change


@dataclass
class SweetSpotResult:
    """Result from finding optimal steering strength."""

    feature_idx: int
    optimal_strength: float
    strength_range: Tuple[float, float]  # (min, max) for useful steering

    # Effects at optimal strength
    effect_at_optimal: SteeringEffect

    # Sweep data
    strengths_tested: List[float]
    effects: List[SteeringEffect]


class SAESteering:
    """
    Inject SAE feature directions into model activations.

    The steering works by:
    1. Getting the decoder direction for a feature (what the feature "means")
    2. Adding/subtracting this direction from activations during forward pass
    3. Measuring how outputs change

    Steering modes:
    - "add": Simply add strength * decoder_direction to activations
    - "project_add": Project out current feature activation, then add new value
    - "ablate": Set feature activation to zero

    Steering positions:
    - "all": Modify all timesteps (default)
    - "last": Only modify the last timestep (current decision)
    - "first_k": Only modify first k timesteps
    """

    def __init__(
        self,
        model: "GPT",
        sae: "TopKSparseAutoencoder",
        hook_point: HookPoint,
        device: torch.device,
    ):
        self.model = model
        self.sae = sae.to(device)
        self.hook_point = hook_point
        self.device = device

        # Validate hook point
        if hook_point.hook_type not in {
            HookPointType.BLOCK_OUTPUT,
            HookPointType.MLP_POST_ACT,
            HookPointType.FINAL_NORM,
        }:
            raise ValueError(
                f"Steering requires block_output, mlp_post_act, or final_norm hook. "
                f"Got: {hook_point.hook_type}"
            )

        # Get the module to hook
        self._module = self._get_module()
        self._hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None

        # Current steering config (set during context manager)
        self._steering_fn: Optional[Callable[[Tensor], Tensor]] = None

    def _get_module(self) -> nn.Module:
        """Get the module to install hook on."""
        if self.hook_point.hook_type == HookPointType.BLOCK_OUTPUT:
            return self.model.blocks[self.hook_point.layer_idx]
        elif self.hook_point.hook_type == HookPointType.MLP_POST_ACT:
            return self.model.blocks[self.hook_point.layer_idx].mlp.fully_connected
        elif self.hook_point.hook_type == HookPointType.FINAL_NORM:
            return self.model.blocks[-1]
        else:
            raise ValueError(f"Unsupported hook type: {self.hook_point.hook_type}")

    def _make_hook_fn(self) -> Callable:
        """Create the hook function that applies steering."""

        def hook_fn(module: nn.Module, inputs: Tuple[Tensor, ...], output: Tensor) -> Tensor:
            if self._steering_fn is None:
                return output

            # For MLP_POST_ACT, output is pre-activation; apply ReLU² first
            if self.hook_point.hook_type == HookPointType.MLP_POST_ACT:
                import torch.nn.functional as F
                output = F.relu(output).square()

            # Apply steering
            steered = self._steering_fn(output)

            return steered

        return hook_fn

    def _install_hook(self) -> None:
        """Install the steering hook."""
        if self._hook_handle is not None:
            return
        self._hook_handle = self._module.register_forward_hook(self._make_hook_fn())

    def _remove_hook(self) -> None:
        """Remove the steering hook."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def get_decoder_direction(self, feature_idx: int) -> Tensor:
        """Get the decoder direction for a feature (normalized to unit length)."""
        direction = self.sae.get_decoder_direction(feature_idx)
        return direction / (direction.norm() + 1e-8)

    def _make_add_steering_fn(
        self,
        feature_idx: int,
        strength: float,
        position: str = "last",
    ) -> Callable[[Tensor], Tensor]:
        """Create steering function that adds feature direction."""
        direction = self.get_decoder_direction(feature_idx).to(self.device)

        def steering_fn(activations: Tensor) -> Tensor:
            # activations: [batch, seq, dim]
            steered = activations.clone()

            if position == "all":
                steered = steered + strength * direction
            elif position == "last":
                steered[:, -1, :] = steered[:, -1, :] + strength * direction
            elif position.startswith("first_"):
                k = int(position.split("_")[1])
                steered[:, :k, :] = steered[:, :k, :] + strength * direction
            else:
                raise ValueError(f"Unknown position: {position}")

            return steered

        return steering_fn

    def _make_multi_steering_fn(
        self,
        feature_strengths: Dict[int, float],
        position: str = "last",
    ) -> Callable[[Tensor], Tensor]:
        """Create steering function for multiple features."""
        # Pre-compute weighted direction sum
        total_direction = torch.zeros(self.sae.input_dim, device=self.device)
        for feat_idx, strength in feature_strengths.items():
            direction = self.get_decoder_direction(feat_idx).to(self.device)
            total_direction = total_direction + strength * direction

        def steering_fn(activations: Tensor) -> Tensor:
            steered = activations.clone()

            if position == "all":
                steered = steered + total_direction
            elif position == "last":
                steered[:, -1, :] = steered[:, -1, :] + total_direction
            else:
                raise ValueError(f"Unknown position: {position}")

            return steered

        return steering_fn

    def _make_ablate_steering_fn(
        self,
        feature_idx: int,
        position: str = "last",
    ) -> Callable[[Tensor], Tensor]:
        """Create steering function that removes a feature's contribution."""
        direction = self.get_decoder_direction(feature_idx).to(self.device)

        def steering_fn(activations: Tensor) -> Tensor:
            steered = activations.clone()

            if position == "all":
                # Project out the feature direction
                proj = torch.einsum("bsd,d->bs", steered, direction)
                steered = steered - proj.unsqueeze(-1) * direction
            elif position == "last":
                proj = torch.einsum("bd,d->b", steered[:, -1, :], direction)
                steered[:, -1, :] = steered[:, -1, :] - proj.unsqueeze(-1) * direction
            else:
                raise ValueError(f"Unknown position: {position}")

            return steered

        return steering_fn

    @contextmanager
    def steer(
        self,
        feature_idx: int,
        strength: float = 1.0,
        position: str = "last",
    ) -> Generator[None, None, None]:
        """
        Context manager that steers model by adding feature direction.

        Args:
            feature_idx: Index of SAE feature to amplify
            strength: Multiplier for feature direction (negative to suppress)
            position: Where to apply steering: "all", "last", "first_k"

        Usage:
            with steering.steer(feature_idx=42, strength=2.0):
                outputs = model(inputs)
        """
        self._steering_fn = self._make_add_steering_fn(feature_idx, strength, position)
        self._install_hook()
        try:
            yield
        finally:
            self._steering_fn = None
            self._remove_hook()

    @contextmanager
    def ablate(
        self,
        feature_idx: int,
        position: str = "last",
    ) -> Generator[None, None, None]:
        """
        Context manager that removes a feature's contribution.

        Args:
            feature_idx: Index of SAE feature to ablate
            position: Where to apply ablation

        Usage:
            with steering.ablate(feature_idx=42):
                outputs = model(inputs)
        """
        self._steering_fn = self._make_ablate_steering_fn(feature_idx, position)
        self._install_hook()
        try:
            yield
        finally:
            self._steering_fn = None
            self._remove_hook()

    @contextmanager
    def multi_steer(
        self,
        feature_strengths: Dict[int, float],
        position: str = "last",
    ) -> Generator[None, None, None]:
        """
        Context manager that steers with multiple features.

        Args:
            feature_strengths: Dict mapping feature_idx -> strength
            position: Where to apply steering

        Usage:
            with steering.multi_steer({42: 2.0, 17: -1.0}):
                outputs = model(inputs)
        """
        self._steering_fn = self._make_multi_steering_fn(feature_strengths, position)
        self._install_hook()
        try:
            yield
        finally:
            self._steering_fn = None
            self._remove_hook()

    def measure_steering_effect(
        self,
        inputs,
        feature_idx: int,
        strength: float,
        position: str = "last",
    ) -> SteeringEffect:
        """
        Measure the effect of steering on model outputs.

        Args:
            inputs: TensorDict of model inputs
            feature_idx: Feature to steer
            strength: Steering strength
            position: Where to apply steering

        Returns:
            SteeringEffect with distribution changes
        """
        import torch.nn.functional as F

        # Get baseline outputs
        with torch.no_grad():
            baseline = self.model(inputs)

        # Get steered outputs
        with torch.no_grad():
            with self.steer(feature_idx, strength, position):
                steered = self.model(inputs)

        # Compute KL divergences for discrete outputs
        def kl_div(p_logits: Tensor, q_logits: Tensor) -> float:
            p = F.softmax(p_logits, dim=-1)
            q = F.softmax(q_logits, dim=-1)
            # Only compute for last timestep
            p = p[:, -1, :]
            q = q[:, -1, :]
            kl = (p * (p.log() - q.log())).sum(dim=-1).mean()
            return kl.item()

        main_kl = kl_div(baseline.main_stick_logits, steered.main_stick_logits)
        c_kl = kl_div(baseline.c_stick_logits, steered.c_stick_logits)
        shoulder_kl = kl_div(baseline.shoulder_logits, steered.shoulder_logits)

        # Button probability difference (sigmoid outputs)
        baseline_buttons = torch.sigmoid(baseline.button_logits[:, -1, :])
        steered_buttons = torch.sigmoid(steered.button_logits[:, -1, :])
        buttons_diff = (baseline_buttons - steered_buttons).abs().mean().item()

        # Value head change
        value_delta = (steered.value[:, -1] - baseline.value[:, -1]).mean().item()

        total_change = main_kl + c_kl + shoulder_kl + buttons_diff

        return SteeringEffect(
            feature_idx=feature_idx,
            strength=strength,
            main_stick_kl=main_kl,
            c_stick_kl=c_kl,
            buttons_diff=buttons_diff,
            shoulder_kl=shoulder_kl,
            value_delta=value_delta,
            total_action_change=total_change,
        )

    def find_steering_sweet_spot(
        self,
        feature_idx: int,
        dataloader,
        colmap: "ColumnMap",
        strength_range: Tuple[float, float] = (-3.0, 3.0),
        n_strengths: int = 13,
        n_batches: int = 10,
        position: str = "last",
        value_drop_threshold: float = 0.1,
    ) -> SweetSpotResult:
        """
        Find the optimal steering strength for a feature.

        The "sweet spot" is where:
        - Action distribution changes meaningfully (behavior shifts)
        - Value head doesn't drop too much (capability preserved)

        Args:
            feature_idx: Feature to test
            dataloader: DataLoader for test data
            colmap: Column mapping
            strength_range: (min, max) strength to test
            n_strengths: Number of strength values to test
            n_batches: Number of batches to average over
            position: Where to apply steering
            value_drop_threshold: Maximum acceptable value drop

        Returns:
            SweetSpotResult with optimal strength and sweep data
        """
        from train.batch_utils import build_model_inputs

        strengths = torch.linspace(
            strength_range[0], strength_range[1], n_strengths
        ).tolist()

        all_effects: List[SteeringEffect] = []

        self.model.eval()
        data_iter = iter(dataloader)

        for strength in strengths:
            effects_at_strength = []

            for _ in range(n_batches):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)

                X = batch["X"].to(self.device)
                inputs = build_model_inputs(X, colmap)

                effect = self.measure_steering_effect(
                    inputs, feature_idx, strength, position
                )
                effects_at_strength.append(effect)

            # Average effects
            avg_effect = SteeringEffect(
                feature_idx=feature_idx,
                strength=strength,
                main_stick_kl=sum(e.main_stick_kl for e in effects_at_strength)
                / len(effects_at_strength),
                c_stick_kl=sum(e.c_stick_kl for e in effects_at_strength)
                / len(effects_at_strength),
                buttons_diff=sum(e.buttons_diff for e in effects_at_strength)
                / len(effects_at_strength),
                shoulder_kl=sum(e.shoulder_kl for e in effects_at_strength)
                / len(effects_at_strength),
                value_delta=sum(e.value_delta for e in effects_at_strength)
                / len(effects_at_strength),
                total_action_change=sum(e.total_action_change for e in effects_at_strength)
                / len(effects_at_strength),
            )
            all_effects.append(avg_effect)

        # Find sweet spot: max action change while value doesn't drop too much
        valid_effects = [
            e for e in all_effects if abs(e.value_delta) < value_drop_threshold
        ]

        if valid_effects:
            optimal = max(valid_effects, key=lambda e: e.total_action_change)
        else:
            # If all cause too much value drop, pick smallest value drop
            optimal = min(all_effects, key=lambda e: abs(e.value_delta))

        # Find useful range (where action change > 10% of optimal)
        threshold = optimal.total_action_change * 0.1
        useful = [e for e in all_effects if e.total_action_change > threshold]
        if useful:
            useful_range = (min(e.strength for e in useful), max(e.strength for e in useful))
        else:
            useful_range = (optimal.strength, optimal.strength)

        return SweetSpotResult(
            feature_idx=feature_idx,
            optimal_strength=optimal.strength,
            strength_range=useful_range,
            effect_at_optimal=optimal,
            strengths_tested=strengths,
            effects=all_effects,
        )


def compute_persona_vector(
    model: "GPT",
    positive_dataloader,
    negative_dataloader,
    colmap: "ColumnMap",
    hook_point: HookPoint,
    device: torch.device,
    max_samples: int = 1000,
) -> Tensor:
    """
    Compute a persona-style steering vector from contrastive data.

    This implements the "difference of means" approach from persona vectors:
    v = mean(activations | positive) - mean(activations | negative)

    Args:
        model: GPT model
        positive_dataloader: DataLoader for positive examples (e.g., aggressive play)
        negative_dataloader: DataLoader for negative examples (e.g., passive play)
        colmap: Column mapping
        hook_point: Where to extract activations
        device: Torch device
        max_samples: Maximum samples from each distribution

    Returns:
        Steering vector [d_model] to add to activations

    Example:
        # Compute aggression vector
        v_aggr = compute_persona_vector(
            model,
            aggressive_loader,
            passive_loader,
            colmap,
            HookPoint(HookPointType.BLOCK_OUTPUT, 4),
            device,
        )

        # Use for steering
        # hidden_states[:, -1, :] += alpha * v_aggr
    """
    from interp.cache import ActivationCache

    cache = ActivationCache(model, hook_point, device)

    # Collect positive activations
    print("Collecting positive activations...")
    pos_cached = cache.fill_from_dataloader(
        positive_dataloader, colmap, max_samples=max_samples, show_progress=True
    )
    pos_mean = pos_cached.activations.mean(dim=0)
    cache.clear()

    # Collect negative activations
    print("Collecting negative activations...")
    neg_cached = cache.fill_from_dataloader(
        negative_dataloader, colmap, max_samples=max_samples, show_progress=True
    )
    neg_mean = neg_cached.activations.mean(dim=0)

    # Compute difference
    persona_vector = pos_mean - neg_mean

    print(f"Persona vector norm: {persona_vector.norm().item():.4f}")

    return persona_vector.to(device)


def decompose_vector_into_sae_features(
    vector: Tensor,
    sae: "TopKSparseAutoencoder",
    top_k: int = 20,
) -> List[Tuple[int, float]]:
    """
    Decompose a steering vector into SAE feature contributions.

    Useful for understanding what a persona vector (e.g., "aggression")
    is made of in terms of interpretable features.

    Args:
        vector: Steering vector [d_model]
        sae: Trained SAE
        top_k: Number of top features to return

    Returns:
        List of (feature_idx, cosine_similarity) sorted by similarity

    Example:
        # Decompose aggression into features
        components = decompose_vector_into_sae_features(v_aggr, sae, top_k=10)
        for feat_idx, similarity in components:
            print(f"Feature {feat_idx}: {similarity:.3f}")
    """
    # Normalize vector
    vector = vector / (vector.norm() + 1e-8)

    # Get decoder directions (each row is a feature's direction)
    W_dec = sae.W_dec.detach()  # [hidden_dim, input_dim]

    # Normalize decoder directions
    W_dec_norm = W_dec / (W_dec.norm(dim=1, keepdim=True) + 1e-8)

    # Compute cosine similarities
    similarities = W_dec_norm @ vector  # [hidden_dim]

    # Get top features (by absolute similarity for both aligned and anti-aligned)
    top_indices = similarities.abs().topk(top_k).indices

    result = []
    for idx in top_indices:
        result.append((idx.item(), similarities[idx].item()))

    # Sort by absolute similarity (descending)
    result.sort(key=lambda x: abs(x[1]), reverse=True)

    return result


__all__ = [
    "SAESteering",
    "SteeringEffect",
    "SweetSpotResult",
    "compute_persona_vector",
    "decompose_vector_into_sae_features",
]
