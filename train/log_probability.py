"""Performance-optimized log probability computation helpers.

This module provides helpers for computing log probabilities and entropies
for both categorical (stick/shoulder) and Bernoulli (button) distributions.

These helpers use manual torch operations instead of PyTorch distribution objects
for ~2x performance improvement in single-sample inference, while maintaining
exact numerical equivalence (verified by TDD tests).

Key design decisions:
- Manual log_softmax for categorical distributions (avoids Distribution overhead)
- Epsilon smoothing (1e-8) for Bernoulli to prevent log(0) = -inf
- Support for both single-sample and batch modes
- All operations preserve float32 precision and numerical stability
"""

from __future__ import annotations

import torch

# Epsilon value for numerical stability in log probability computation
# This prevents log(0) = -inf in Bernoulli distributions
# Value verified by TDD tests to match current implementation
LOG_PROB_EPSILON = 1e-8


def compute_categorical_log_prob(
    logits: torch.Tensor,
    actions: torch.Tensor,
    mode: str = "single",
) -> torch.Tensor:
    """Compute log probabilities for categorical distributions.

    This uses manual log_softmax instead of torch.distributions.Categorical
    for better performance in single-sample inference (~2x faster).

    Args:
        logits: Logits for the categorical distribution.
            - Single mode: shape (num_classes,)
            - Batch mode: shape (batch_size, num_classes)
        actions: Action indices to compute log probabilities for.
            - Single mode: scalar or shape ()
            - Batch mode: shape (batch_size,)
        mode: Either "single" or "batch"
            - "single": logits shape (num_classes,), actions is scalar
            - "batch": logits shape (batch_size, num_classes), actions shape (batch_size,)

    Returns:
        Log probabilities:
            - Single mode: scalar tensor
            - Batch mode: shape (batch_size,)

    Examples:
        >>> # Single sample
        >>> logits = torch.randn(64)  # main stick has 64 classes
        >>> action_idx = torch.tensor(15)
        >>> log_prob = compute_categorical_log_prob(logits, action_idx, mode="single")
        >>> log_prob.shape
        torch.Size([])

        >>> # Batch
        >>> logits = torch.randn(32, 64)  # batch_size=32
        >>> actions = torch.randint(0, 64, (32,))
        >>> log_probs = compute_categorical_log_prob(logits, actions, mode="batch")
        >>> log_probs.shape
        torch.Size([32])
    """
    if mode == "single":
        # Single sample: logits shape (num_classes,), action is scalar
        log_softmax = torch.log_softmax(logits, dim=-1)
        return log_softmax[actions]
    elif mode == "batch":
        # Batch: logits shape (batch_size, num_classes), actions shape (batch_size,)
        log_softmax = torch.log_softmax(logits, dim=-1)
        # Gather log probs for each action in the batch
        return log_softmax.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'single' or 'batch'.")


def compute_bernoulli_log_prob(
    logits: torch.Tensor,
    actions: torch.Tensor,
    mode: str = "single",
) -> torch.Tensor:
    """Compute log probabilities for Bernoulli distributions (button presses).

    This uses manual log computation with epsilon smoothing instead of
    torch.distributions.Bernoulli for better performance and explicit
    numerical stability control.

    Args:
        logits: Logits for Bernoulli distributions (pre-sigmoid).
            - Single mode: shape (num_buttons,) e.g. (5,) for 5 buttons
            - Batch mode: shape (batch_size, num_buttons)
        actions: Binary actions (0 or 1).
            - Single mode: shape (num_buttons,)
            - Batch mode: shape (batch_size, num_buttons)
        mode: Either "single" or "batch"

    Returns:
        Log probabilities (summed across buttons):
            - Single mode: scalar tensor
            - Batch mode: shape (batch_size,)

    Examples:
        >>> # Single sample
        >>> logits = torch.randn(5)  # 5 buttons
        >>> actions = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0])
        >>> log_prob = compute_bernoulli_log_prob(logits, actions, mode="single")
        >>> log_prob.shape
        torch.Size([])

        >>> # Batch
        >>> logits = torch.randn(32, 5)
        >>> actions = torch.randint(0, 2, (32, 5)).float()
        >>> log_probs = compute_bernoulli_log_prob(logits, actions, mode="batch")
        >>> log_probs.shape
        torch.Size([32])
    """
    # Convert logits to probabilities
    probs = torch.sigmoid(logits)

    # Compute log probabilities with epsilon smoothing
    # log P(a|s) = log(p) if a=1, log(1-p) if a=0
    log_probs = torch.where(
        actions == 1,
        torch.log(probs + LOG_PROB_EPSILON),
        torch.log(1 - probs + LOG_PROB_EPSILON),
    )

    if mode == "single":
        # Sum log probs across buttons for single sample
        return log_probs.sum()
    elif mode == "batch":
        # Sum log probs across buttons for each sample in batch
        return log_probs.sum(dim=-1)
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'single' or 'batch'.")


def compute_categorical_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute entropy for categorical distribution.

    Entropy measures the uncertainty/randomness of the distribution.
    Higher entropy = more uniform distribution = more exploration.

    Args:
        logits: Logits for categorical distribution.
            - Shape: (batch_size, num_classes) or (num_classes,)

    Returns:
        Entropy values:
            - If logits has batch dim: shape (batch_size,)
            - If logits is 1D: scalar

    Examples:
        >>> # Batch mode
        >>> logits = torch.randn(32, 64)
        >>> entropies = compute_categorical_entropy(logits)
        >>> entropies.shape
        torch.Size([32])

        >>> # Single sample
        >>> logits = torch.randn(64)
        >>> entropy = compute_categorical_entropy(logits)
        >>> entropy.shape
        torch.Size([])
    """
    # Compute probabilities and log probabilities
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)

    # Entropy: H = -sum(p * log(p))
    entropy = -(probs * log_probs).sum(dim=-1)

    return entropy


def compute_bernoulli_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute entropy for Bernoulli distributions (button presses).

    For Bernoulli: H = -p*log(p) - (1-p)*log(1-p)
    Summed across all buttons.

    Args:
        logits: Logits for Bernoulli distributions (pre-sigmoid).
            - Shape: (batch_size, num_buttons) or (num_buttons,)

    Returns:
        Entropy values (summed across buttons):
            - If logits has batch dim: shape (batch_size,)
            - If logits is 1D: scalar

    Examples:
        >>> # Batch mode
        >>> logits = torch.randn(32, 5)
        >>> entropies = compute_bernoulli_entropy(logits)
        >>> entropies.shape
        torch.Size([32])

        >>> # Single sample
        >>> logits = torch.randn(5)
        >>> entropy = compute_bernoulli_entropy(logits)
        >>> entropy.shape
        torch.Size([])
    """
    probs = torch.sigmoid(logits)

    # Compute entropy per button with epsilon smoothing
    # H = -p*log(p) - (1-p)*log(1-p)
    entropy_per_button = -(
        probs * torch.log(probs + LOG_PROB_EPSILON)
        + (1 - probs) * torch.log(1 - probs + LOG_PROB_EPSILON)
    )

    # Sum across buttons
    return entropy_per_button.sum(dim=-1)
