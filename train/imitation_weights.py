"""Value-based sample weighting for imitation learning.

This module computes per-frame weights based on the value_target feature to prioritize
learning from high-value states and critical decisions. Used to train models that imitate
winning play rather than average play.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from config.imitation_config import ImitationConfig


def compute_value_weighted_weights(
    values: Tensor,
    *,
    k: float,
    temperature: float,
    use_exp: bool,
) -> Tensor:
    """Weight samples by their value (higher value = higher weight).

    Args:
        values: [B, L] value targets
        k: Scaling factor (higher = more emphasis on high values)
        temperature: Temperature for softening distribution
        use_exp: If True, use exp(k*value); else use linear scaling

    Returns:
        [B, L] normalized weights
    """
    # Ensure float dtype for arithmetic operations
    values = values.float()

    if use_exp:
        # Exponential weighting: strongly emphasizes high-value states
        # Subtract max for numerical stability
        scaled = k * values / temperature
        scaled = scaled - scaled.max()  # prevent overflow
        weights = torch.exp(scaled)
    else:
        # Linear weighting: gentler emphasis
        weights = (k * values / temperature).clamp_min(0.0)

    # Normalize to mean=1 to maintain loss scale
    return weights / (weights.mean() + 1e-8)


def compute_value_filter_weights(
    values: Tensor,
    *,
    percentile: float,
    soft: bool,
    temperature: float,
) -> Tensor:
    """Filter to only train on top percentile by value.

    Args:
        values: [B, L] value targets
        percentile: Percentile cutoff (e.g., 30 = only train on top 70%)
        soft: If True, use smooth sigmoid cutoff; else hard threshold
        temperature: Temperature for soft filtering

    Returns:
        [B, L] binary or smooth weights
    """
    # Ensure float dtype for quantile computation
    values = values.float()

    # Compute threshold across entire batch
    threshold = torch.quantile(values.flatten(), percentile / 100.0)

    if soft:
        # Smooth sigmoid transition around threshold
        weights = torch.sigmoid((values - threshold) / temperature)
    else:
        # Hard cutoff: 1 if above threshold, 0 otherwise
        weights = (values >= threshold).float()

    return weights


def compute_advantage_weights(
    values: Tensor,
    *,
    alpha: float,
    use_gae: bool = False,
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
    return_advantages: bool = False,
):
    """Weight by advantage (temporal difference in value).

    Frames where value increases (good decisions) get higher weight.

    Args:
        values: [B, L] value targets
        alpha: Scaling factor for advantage weighting
        gamma: Discount factor for GAE
        gae_lambda: Lambda for GAE
        return_advantages: If True, return (weights, advantages); else just weights

    Returns:
        If return_advantages=False: [B, L] weights based on advantage magnitude
        If return_advantages=True: tuple of ([B, L] weights, [B, L] advantages)
    """
    # Ensure float dtype for arithmetic operations
    values = values.float()

    B, L = values.shape

    # GAE advantage estimation (more sophisticated)
    advantages = torch.zeros_like(values)
    deltas = torch.zeros_like(values)

    # Compute TD errors
    deltas[:, :-1] = values[:, 1:] - values[:, :-1]

    # Compute GAE
    gae = torch.zeros(B, device=values.device, dtype=values.dtype)
    for t in reversed(range(L - 1)):
        gae = deltas[:, t] + gamma * gae_lambda * gae
        advantages[:, t] = gae

    # Weight by advantage: boost positive (value improvements), suppress negative (mistakes/noise)
    # For imitation learning, we want to learn from frames where expert decisions led to value gains,
    # not from frames where value declined (likely mistakes or unavoidable bad situations)
    weights = torch.where(
        advantages > 0,
        1.0 + alpha * advantages,  # strongly boost improvements
        1.0
        / (
            1.0 + alpha * 0.5 * torch.abs(advantages)
        ),  # suppress declines (down to ~1/12 weight)
    )

    weights = weights.clamp_min(0.0)

    if return_advantages:
        return weights, advantages
    return weights


def compute_model_advantage_weights(
    value_pred: Tensor,
    value_target: Tensor,
    *,
    alpha: float,
    return_advantages: bool = False,
):
    """Weight by model-dependent advantage (how much better expert was vs model prediction).

    This computes advantages as: advantage = ground_truth - model_prediction

    Frames where the actual outcome was better than the model predicted get higher weight.
    This focuses learning on situations where the expert did better than our current model
    would have expected, which is exactly what we want for imitation learning.

    Args:
        value_pred: [B, L] model's value predictions
        value_target: [B, L] ground truth value targets
        alpha: Scaling factor for advantage weighting
        return_advantages: If True, return (weights, advantages); else just weights

    Returns:
        If return_advantages=False: [B, L] weights based on advantage magnitude
        If return_advantages=True: tuple of ([B, L] weights, [B, L] advantages)
    """
    # Ensure float dtype and detach predictions (don't backprop through advantage computation)
    value_pred = value_pred.detach().float()
    value_target = value_target.float()

    # Compute model-dependent advantage: how much better was the actual outcome vs prediction
    # Positive = expert achieved better outcome than model expected (learn from this!)
    # Negative = expert achieved worse outcome than model expected (model overestimate or expert error)
    advantages = value_target - value_pred  # [B, L]

    # Weight by advantage: boost positive (expert beat model), suppress negative (model was too optimistic)
    # For imitation learning, we want to learn from frames where the expert did better than our
    # model predicted, as these represent situations where we underestimate the expert's capability
    weights = torch.where(
        advantages > 0,
        1.0 + alpha * advantages,  # strongly boost when expert beat our prediction
        1.0
        / (
            1.0 + alpha * 0.5 * torch.abs(advantages)
        ),  # suppress when model overestimated
    )

    weights = weights.clamp_min(0.0)

    # Normalize to mean=1 to maintain loss scale
    weights = weights / (weights.mean() + 1e-8)

    if return_advantages:
        return weights, advantages
    return weights


def compute_hybrid_weights(
    values: Tensor,
    config: ImitationConfig,
) -> Tensor:
    """Combine multiple weighting strategies.

    Args:
        values: [B, L] value targets
        config: Imitation config with hybrid settings

    Returns:
        [B, L] combined weights
    """
    if len(config.hybrid_strategies) != len(config.hybrid_weights):
        raise ValueError("hybrid_strategies and hybrid_weights must have same length")

    weights_list = []

    for strategy_name in config.hybrid_strategies:
        if strategy_name == "value_weighted":
            w = compute_value_weighted_weights(
                values,
                k=config.value_k,
                temperature=config.value_temperature,
                use_exp=config.value_use_exp,
            )
        elif strategy_name == "value_filter":
            w = compute_value_filter_weights(
                values,
                percentile=config.filter_percentile,
                soft=config.filter_soft,
                temperature=config.filter_temperature,
            )
        elif strategy_name == "value_advantage":
            w = compute_advantage_weights(
                values,
                alpha=config.advantage_alpha,
            )
        elif strategy_name == "uniform":
            w = torch.ones_like(values)
        else:
            raise ValueError(f"Unknown strategy in hybrid: {strategy_name}")

        weights_list.append(w)

    # Combine strategies: multiply each weight by its hybrid coefficient, then multiply together
    combined = torch.ones_like(values)
    for w, coef in zip(weights_list, config.hybrid_weights):
        # Raise weight to power of coefficient (geometric mean when coefs sum to 1)
        combined = combined * torch.pow(w, coef)

    # Normalize to mean=1
    return combined / (combined.mean() + 1e-8)


def compute_imitation_weights(
    X: Tensor,
    value_idx: int,
    config: ImitationConfig,
) -> Tensor:
    """Compute per-sample weights based on imitation learning strategy.

    Args:
        X: [B, L, F] input features
        value_idx: Column index of value_target feature
        config: Imitation learning configuration

    Returns:
        [B, L] sample weights (normalized to mean=1)
    """
    # Extract value targets
    values = X[:, :, value_idx]  # [B, L]

    strategy = config.strategy

    if strategy == "uniform":
        # Baseline: no weighting
        return torch.ones_like(values)

    elif strategy == "value_weighted":
        return compute_value_weighted_weights(
            values,
            k=config.value_k,
            temperature=config.value_temperature,
            use_exp=config.value_use_exp,
        )

    elif strategy == "value_filter":
        return compute_value_filter_weights(
            values,
            percentile=config.filter_percentile,
            soft=config.filter_soft,
            temperature=config.filter_temperature,
        )

    elif strategy == "value_advantage":
        return compute_advantage_weights(
            values,
            alpha=config.advantage_alpha,
        )

    elif strategy == "hybrid":
        return compute_hybrid_weights(values, config)

    else:
        raise ValueError(f"Unknown imitation strategy: {strategy}")
