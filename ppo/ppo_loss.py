"""PPO loss computation for multi-head action outputs."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def compute_log_probs(
    action_logits: Dict[str, torch.Tensor],
    actions_taken: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Compute log probabilities for actions taken.
    
    Args:
        action_logits: Dictionary of action logits for each head
            - main_stick: [B, num_stick_bins]
            - c_stick: [B, num_c_stick_bins]
            - shoulder: [B, num_shoulder_bins]
            - buttons: [B, num_buttons]
        actions_taken: Dictionary of actions taken (indices or bool)
            - main_stick: [B] indices
            - c_stick: [B] indices
            - shoulder: [B] indices
            - buttons: [B, num_buttons] bool
    
    Returns:
        [B] total log probabilities
    """
    log_probs = []
    
    # Check for NaN in inputs
    for head, logits in action_logits.items():
        if torch.isnan(logits).any():
            print(f"  [compute_log_probs] NaN detected in {head} logits! count={torch.isnan(logits).sum()}")
    for head, actions in actions_taken.items():
        if torch.isnan(actions).any():
            print(f"  [compute_log_probs] NaN detected in {head} actions! count={torch.isnan(actions).sum()}")

    # Sticks and shoulder (categorical)
    for head in ["main_stick", "c_stick", "shoulder"]:
        if head in action_logits:
            logits_orig = action_logits[head]  # [B, num_classes]
            actions = actions_taken[head].long()  # [B]

            # Clamp logits for numerical stability (loose bounds)
            logits = torch.clamp(logits_orig, min=-10000.0, max=10000.0)
            
            # Log if clamping occurred
            if not torch.equal(logits, logits_orig):
                n_clamped = ((logits_orig < -10000.0) | (logits_orig > 10000.0)).sum().item()
                print(f"  [{head}] Clamped {n_clamped} logits: "
                      f"range [{logits_orig.min():.2f}, {logits_orig.max():.2f}] "
                      f"-> [{logits.min():.2f}, {logits.max():.2f}]")
            
            log_prob = F.log_softmax(logits, dim=-1)  # [B, num_classes]
            
            # Check for NaN after log_softmax
            if torch.isnan(log_prob).any():
                print(f"  [{head}] NaN after log_softmax! logits range: [{logits.min():.2f}, {logits.max():.2f}]")
            
            selected_log_prob_orig = torch.gather(
                log_prob, -1, actions.unsqueeze(-1)
            ).squeeze(
                -1
            )  # [B]
            
            # Check for NaN after gather
            if torch.isnan(selected_log_prob_orig).any():
                print(f"  [{head}] NaN after gather! actions range: [{actions.min()}, {actions.max()}]")
            
            # Clamp log probs to prevent extreme values (loose bounds)
            selected_log_prob = torch.clamp(selected_log_prob_orig, min=-100.0, max=0.0)
            
            # Log if log prob clamping occurred
            if not torch.equal(selected_log_prob, selected_log_prob_orig):
                n_clamped = ((selected_log_prob_orig < -100.0) | (selected_log_prob_orig > 0.0)).sum().item()
                print(f"  [{head}] Clamped {n_clamped} log_probs: "
                      f"range [{selected_log_prob_orig.min():.2f}, {selected_log_prob_orig.max():.2f}] "
                      f"-> [{selected_log_prob.min():.2f}, {selected_log_prob.max():.2f}]")
            
            log_probs.append(selected_log_prob)

    # Buttons (independent Bernoulli)
    if "buttons" in action_logits:
        logits_orig = action_logits["buttons"]  # [B, num_buttons]
        actions = actions_taken["buttons"].float()  # [B, num_buttons]

        # Clamp logits for numerical stability (loose bounds)
        logits = torch.clamp(logits_orig, min=-200.0, max=200.0)
        
        # Log if clamping occurred
        if not torch.equal(logits, logits_orig):
            n_clamped = ((logits_orig < -200.0) | (logits_orig > 200.0)).sum().item()
            print(f"  [buttons] Clamped {n_clamped} logits: "
                  f"range [{logits_orig.min():.2f}, {logits_orig.max():.2f}] "
                  f"-> [{logits.min():.2f}, {logits.max():.2f}]")
        
        # Log probability for each button: action * log(p) + (1-action) * log(1-p)
        probs_orig = torch.sigmoid(logits)
        # Clamp probabilities away from 0 and 1
        probs = torch.clamp(probs_orig, min=1e-7, max=1.0 - 1e-7)
        
        
        button_log_probs = actions * torch.log(probs) + (
            1 - actions
        ) * torch.log(1 - probs)
        
        # Check for NaN in button log probs
        if torch.isnan(button_log_probs).any():
            print(f"  [buttons] NaN in button_log_probs! probs range: [{probs.min():.6f}, {probs.max():.6f}]")
        
        # Sum over buttons
        log_probs.append(button_log_probs.sum(dim=-1))  # [B]

    # Sum all log probs
    total_log_probs = torch.stack(log_probs, dim=0).sum(dim=0)  # [B]
    
    # Check for NaN in final result
    if torch.isnan(total_log_probs).any():
        print(f"  [compute_log_probs] NaN in final total_log_probs! count={torch.isnan(total_log_probs).sum()}")
        for i, lp in enumerate(log_probs):
            if torch.isnan(lp).any():
                print(f"    log_probs[{i}] has NaN: count={torch.isnan(lp).sum()}")
    
    return total_log_probs


def compute_entropy(action_logits: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Compute entropy of the action distribution.

    Higher entropy = more exploration.

    Args:
        action_logits: Dictionary of action logits for each head

    Returns:
        [B] entropy values
    """
    entropies = []

    # Sticks and shoulder (categorical entropy)
    for head in ["main_stick", "c_stick", "shoulder"]:
        if head in action_logits:
            logits = action_logits[head]  # [B, num_classes]
            probs = F.softmax(logits, dim=-1)  # [B, num_classes]
            log_probs = F.log_softmax(logits, dim=-1)

            # H(X) = -sum(p * log(p))
            entropy = -(probs * log_probs).sum(dim=-1)  # [B]
            entropies.append(entropy)

    # Buttons (Bernoulli entropy)
    if "buttons" in action_logits:
        logits = action_logits["buttons"]  # [B, num_buttons]
        probs = torch.sigmoid(logits)  # [B, num_buttons]

        # H(Bernoulli) = -p*log(p) - (1-p)*log(1-p)
        button_entropy = -(
            probs * torch.log(probs + 1e-8) + (1 - probs) * torch.log(1 - probs + 1e-8)
        )
        # Sum over buttons
        entropies.append(button_entropy.sum(dim=-1))  # [B]

    # Sum all entropies
    return torch.stack(entropies, dim=0).sum(dim=0)  # [B]


def compute_ppo_loss(
    new_action_logits: Dict[str, torch.Tensor],
    old_action_logits: Dict[str, torch.Tensor],
    actions_taken: Dict[str, torch.Tensor],
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_ratio: float = 0.2,
    entropy_coef: float = 0.01,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Compute PPO clipped surrogate loss.

    Args:
        new_action_logits: Current policy action logits
        old_action_logits: Old policy action logits (for logging)
        actions_taken: Actions that were taken
        old_log_probs: Log probs from old policy
        advantages: GAE advantages
        clip_ratio: Clipping epsilon (policy ratio will be clipped to [1-eps, 1+eps])
        entropy_coef: Coefficient for entropy bonus

    Returns:
        Tuple of (loss, metrics_dict)
    """
    # Compute new log probs
    new_log_probs = compute_log_probs(new_action_logits, actions_taken)  # [B]
    
    # Debug: Check old and new log probs
    if torch.isnan(old_log_probs).any():
        print(f"  [loss] NaN in old_log_probs! count={torch.isnan(old_log_probs).sum()}")
    if torch.isnan(new_log_probs).any():
        print(f"  [loss] NaN in new_log_probs! count={torch.isnan(new_log_probs).sum()}")
        print(f"  [loss] new_log_probs stats: min={new_log_probs[~torch.isnan(new_log_probs)].min() if (~torch.isnan(new_log_probs)).any() else 'all NaN'}, max={new_log_probs[~torch.isnan(new_log_probs)].max() if (~torch.isnan(new_log_probs)).any() else 'all NaN'}")
    
    # Compute ratio: pi_new(a|s) / pi_old(a|s)
    # Clamp the log prob difference to prevent numerical issues (loose bounds for exp safety)
    log_ratio_orig = new_log_probs - old_log_probs
    log_ratio = torch.clamp(log_ratio_orig, min=-50.0, max=50.0)
    
    # Log if clamping occurred
    if not torch.equal(log_ratio, log_ratio_orig):
        n_clamped = ((log_ratio_orig < -50.0) | (log_ratio_orig > 50.0)).sum().item()
        print(f"  [ratio] Clamped {n_clamped} log_ratios: "
              f"range [{log_ratio_orig.min():.2f}, {log_ratio_orig.max():.2f}] "
              f"-> [{log_ratio.min():.2f}, {log_ratio.max():.2f}]")
    
    ratio = torch.exp(log_ratio)  # [B]
    
    # Check ratio
    if torch.isnan(ratio).any() or torch.isinf(ratio).any():
        print(f"  [loss] NaN/Inf in ratio! nan_count={torch.isnan(ratio).sum()}, inf_count={torch.isinf(ratio).sum()}")

    # Clipped surrogate objective
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # Entropy bonus (encourage exploration)
    entropy = compute_entropy(new_action_logits).mean()
    entropy_loss = -entropy_coef * entropy

    # Total loss
    total_loss = policy_loss + entropy_loss

    # Metrics for logging
    metrics = {
        "ppo/policy_loss": policy_loss.item(),
        "ppo/entropy": entropy.item(),
        "ppo/entropy_loss": entropy_loss.item(),
        "ppo/ratio_mean": ratio.mean().item(),
        "ppo/ratio_std": ratio.std().item(),
        "ppo/ratio_min": ratio.min().item(),
        "ppo/ratio_max": ratio.max().item(),
        "ppo/advantages_mean": advantages.mean().item(),
        "ppo/advantages_std": advantages.std().item(),
        "ppo/clipped_fraction": (
            (ratio < 1.0 - clip_ratio) | (ratio > 1.0 + clip_ratio)
        )
        .float()
        .mean()
        .item(),
    }

    return total_loss, metrics


def compute_value_loss(
    value_pred: torch.Tensor,
    value_target: torch.Tensor,
    old_value_pred: Optional[torch.Tensor] = None,
    value_clip: Optional[float] = None,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Compute value function loss.

    Optionally uses clipped value loss as in PPO paper.

    Args:
        value_pred: [B] predicted values from current critic
        value_target: [B] target returns
        old_value_pred: [B] predicted values from old critic (for clipping)
        value_clip: Clipping range for value function. If None, no clipping.

    Returns:
        Tuple of (loss, metrics_dict)
    """
    # Simple MSE loss
    value_loss = F.mse_loss(value_pred, value_target)

    # Optional: clipped value loss
    if value_clip is not None and old_value_pred is not None:
        value_pred_clipped = old_value_pred + torch.clamp(
            value_pred - old_value_pred,
            -value_clip,
            value_clip,
        )
        value_loss_clipped = F.mse_loss(value_pred_clipped, value_target)
        value_loss = torch.max(value_loss, value_loss_clipped)

    metrics = {
        "ppo/value_loss": value_loss.item(),
        "ppo/value_pred_mean": value_pred.mean().item(),
        "ppo/value_target_mean": value_target.mean().item(),
        "ppo/value_error": (value_pred - value_target).abs().mean().item(),
    }

    return value_loss, metrics


def compute_total_ppo_loss(
    new_action_logits: Dict[str, torch.Tensor],
    new_values: torch.Tensor,
    old_action_logits: Dict[str, torch.Tensor],
    old_values: torch.Tensor,
    actions_taken: Dict[str, torch.Tensor],
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    clip_ratio: float = 0.2,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    value_clip: Optional[float] = None,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Compute total PPO loss (policy + value + entropy).

    Args:
        new_action_logits: Current policy action logits
        new_values: Current critic value predictions
        old_action_logits: Old policy action logits
        old_values: Old critic value predictions
        actions_taken: Actions that were taken
        old_log_probs: Log probs from old policy
        advantages: GAE advantages
        returns: Target returns for value function
        clip_ratio: Clipping epsilon for policy ratio
        entropy_coef: Coefficient for entropy bonus
        value_coef: Coefficient for value loss
        value_clip: Optional clipping range for value function

    Returns:
        Tuple of (total_loss, metrics_dict)
    """
    # Policy loss
    policy_loss, policy_metrics = compute_ppo_loss(
        new_action_logits=new_action_logits,
        old_action_logits=old_action_logits,
        actions_taken=actions_taken,
        old_log_probs=old_log_probs,
        advantages=advantages,
        clip_ratio=clip_ratio,
        entropy_coef=entropy_coef,
    )

    # Value loss
    value_loss, value_metrics = compute_value_loss(
        value_pred=new_values,
        value_target=returns,
        old_value_pred=old_values,
        value_clip=value_clip,
    )

    # Total loss
    total_loss = policy_loss + value_coef * value_loss

    # Combined metrics
    metrics = {
        **policy_metrics,
        **value_metrics,
        "ppo/total_loss": total_loss.item(),
    }

    return total_loss, metrics
