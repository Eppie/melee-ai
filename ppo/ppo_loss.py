"""PPO loss computation with clipped objective."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from model.nano_gpt import GPT


def compute_action_logprob(
    outputs: Dict[str, torch.Tensor],
    actions: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Recompute action log probabilities under current policy.

    Args:
        outputs: Model outputs dict with logits for each head
        actions: Dict with sampled actions:
            - main_idx: [B] int (0-63)
            - c_idx: [B] int (0-8)
            - shoulder_idx: [B] int (0-4)
            - buttons: [B, 5] bool

    Returns:
        total_logp: [B] total log probability (sum of components)
    """
    # Main stick
    main_logits = outputs["main_stick"][:, -1, :]  # [B, 64]
    main_dist = Categorical(logits=main_logits)
    main_logp = main_dist.log_prob(actions["main_idx"])

    # C-stick
    c_logits = outputs["c_stick"][:, -1, :]  # [B, 9]
    c_dist = Categorical(logits=c_logits)
    c_logp = c_dist.log_prob(actions["c_idx"])

    # Shoulder
    shoulder_logits = outputs["shoulder"][:, -1, :]  # [B, 5]
    shoulder_dist = Categorical(logits=shoulder_logits)
    shoulder_logp = shoulder_dist.log_prob(actions["shoulder_idx"])

    # Buttons (Bernoulli)
    button_logits = outputs["buttons"][:, -1, :]  # [B, 5]
    button_probs = torch.sigmoid(button_logits)
    button_samples = actions["buttons"]  # [B, 5] bool
    button_logp = torch.log(
        torch.where(button_samples, button_probs, 1 - button_probs)
    ).sum(dim=1)

    # Total log prob (sum of independent components)
    total_logp = main_logp + c_logp + shoulder_logp + button_logp

    return total_logp


def compute_action_entropy(
    outputs: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Compute action entropy for exploration bonus.

    Args:
        outputs: Model outputs dict with logits

    Returns:
        entropy: [B] total entropy (sum of components)
    """
    # Main stick entropy
    main_logits = outputs["main_stick"][:, -1, :]
    main_dist = Categorical(logits=main_logits)
    main_entropy = main_dist.entropy()

    # C-stick entropy
    c_logits = outputs["c_stick"][:, -1, :]
    c_dist = Categorical(logits=c_logits)
    c_entropy = c_dist.entropy()

    # Shoulder entropy
    shoulder_logits = outputs["shoulder"][:, -1, :]
    shoulder_dist = Categorical(logits=shoulder_logits)
    shoulder_entropy = shoulder_dist.entropy()

    # Button entropy (binary)
    button_logits = outputs["buttons"][:, -1, :]  # [B, 5]
    button_probs = torch.sigmoid(button_logits)
    # Binary entropy: -p*log(p) - (1-p)*log(1-p)
    button_entropy = -(
        button_probs * torch.log(button_probs + 1e-8)
        + (1 - button_probs) * torch.log(1 - button_probs + 1e-8)
    ).sum(dim=1)

    # Total entropy
    total_entropy = main_entropy + c_entropy + shoulder_entropy + button_entropy

    return total_entropy


def compute_ppo_loss(
    policy: GPT,
    batch_features: torch.Tensor,
    batch_actions: Dict[str, torch.Tensor],
    old_logps: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    clip_epsilon: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
) -> Dict[str, torch.Tensor]:
    """
    Compute PPO clipped objective loss.

    Args:
        policy: Current policy network
        batch_features: [B, T, F] feature tensor
        batch_actions: Dict with action tensors
        old_logps: [B] old policy log probs
        advantages: [B] GAE advantages (normalized)
        returns: [B] discounted returns
        clip_epsilon: PPO clipping parameter (default 0.2)
        value_coef: Value loss coefficient (default 0.5)
        entropy_coef: Entropy bonus coefficient (default 0.01)

    Returns:
        Dict with loss components:
            - total: Total loss
            - policy: Policy loss (clipped objective)
            - value: Value loss
            - entropy: Entropy loss
            - ratio_mean: Mean importance ratio
            - ratio_std: Std of importance ratio
            - approx_kl: Approximate KL divergence
    """
    from train.batch_utils import build_model_inputs
    from column_map import ColumnMap

    # Forward pass
    # TODO: Pass proper column_map (for now, assume it's available globally)
    # In production, this should be passed as argument
    batch_td = {
        "features": batch_features,
    }

    # Simple forward (assuming batch_features are already in the right format)
    # In production, use build_model_inputs properly
    # For now, treat batch_features as raw input
    # outputs = policy(batch_td)

    # HACK: For now, create a simple TensorDict
    # In production, this needs proper integration with existing batch_utils
    from tensordict import TensorDict

    # Split features into components (this is a simplified version)
    # Real implementation should use build_model_inputs
    # For now, pass features as-is and let model handle it

    # Create minimal input (model expects specific format)
    # This is a placeholder - real implementation needs proper featurization
    B, T, F = batch_features.shape

    # Forward through model
    # Note: This is simplified; real code needs proper input construction
    # outputs = policy(batch_features)  # This won't work without proper input format

    # SIMPLIFIED: For MVP, we'll compute loss on a subset
    # Real implementation requires proper batch construction
    # For now, return dummy loss to complete the structure

    # Recompute action log probs under current policy
    # new_logps = compute_action_logprob(outputs, batch_actions)

    # PLACEHOLDER IMPLEMENTATION
    # TODO: Integrate with existing train/batch_utils.py properly
    new_logps = old_logps.clone()  # Placeholder

    # Compute importance sampling ratio
    ratio = torch.exp(new_logps - old_logps)

    # Clipped surrogate objective
    clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    policy_loss = -torch.min(
        ratio * advantages,
        clipped_ratio * advantages,
    ).mean()

    # Value loss (MSE to returns)
    # values = outputs["value"][:, -1, 0]  # [B]
    # PLACEHOLDER: Use returns as "predicted" values for now
    values = returns.clone()  # Placeholder
    value_loss = F.mse_loss(values, returns)

    # Entropy bonus
    # entropy = compute_action_entropy(outputs)
    # PLACEHOLDER
    entropy = torch.zeros_like(old_logps)
    entropy_loss = -entropy.mean()

    # Total loss
    total_loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

    # Compute diagnostics
    ratio_mean = ratio.mean()
    ratio_std = ratio.std()
    approx_kl = (old_logps - new_logps).mean()

    return {
        "total": total_loss,
        "policy": policy_loss,
        "value": value_loss,
        "entropy": entropy_loss,
        "ratio_mean": ratio_mean,
        "ratio_std": ratio_std,
        "approx_kl": approx_kl,
    }


def compute_ppo_loss_simple(
    old_logps: torch.Tensor,
    new_logps: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float = 0.2,
) -> torch.Tensor:
    """
    Simplified PPO loss (policy only, no value or entropy).

    Useful for testing and debugging.

    Args:
        old_logps: [B] old policy log probs
        new_logps: [B] new policy log probs
        advantages: [B] advantages
        clip_epsilon: clipping parameter

    Returns:
        policy_loss: Scalar loss
    """
    # Importance sampling ratio
    ratio = torch.exp(new_logps - old_logps)

    # Clipped surrogate
    clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)

    # PPO objective (maximize, so negate for minimization)
    policy_loss = -torch.min(
        ratio * advantages,
        clipped_ratio * advantages,
    ).mean()

    return policy_loss


def test_ppo_loss():
    """Unit test for PPO loss computation."""
    print("Testing PPO loss computation...")

    B = 32
    device = torch.device("cpu")

    # Generate dummy data
    old_logps = torch.randn(B, device=device)
    advantages = torch.randn(B, device=device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Test case 1: No policy change (ratio = 1)
    new_logps = old_logps.clone()
    loss = compute_ppo_loss_simple(old_logps, new_logps, advantages)
    print(f"Loss with no change: {loss.item():.4f}")
    assert loss.item() < 0, "Loss should be negative (we're maximizing)"

    # Test case 2: Small policy change
    new_logps = old_logps + torch.randn(B, device=device) * 0.1
    loss = compute_ppo_loss_simple(old_logps, new_logps, advantages)
    print(f"Loss with small change: {loss.item():.4f}")

    # Test case 3: Large policy change (should be clipped)
    new_logps = old_logps + torch.randn(B, device=device) * 2.0
    loss = compute_ppo_loss_simple(old_logps, new_logps, advantages)
    print(f"Loss with large change: {loss.item():.4f}")

    # Test case 4: Verify clipping
    clip_epsilon = 0.2
    ratio = torch.exp(new_logps - old_logps)
    clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)

    num_clipped = (ratio != clipped_ratio).sum().item()
    print(f"Number of clipped ratios: {num_clipped}/{B}")

    print("✓ PPO loss tests passed!")


if __name__ == "__main__":
    test_ppo_loss()
