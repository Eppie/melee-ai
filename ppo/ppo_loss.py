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
        
        # Compute log probs safely with clamping to prevent log(0)
        # Note: We need to clamp both probs and (1-probs) due to floating point precision
        log_probs_pos = torch.log(torch.clamp(probs, min=1e-7))
        log_probs_neg = torch.log(torch.clamp(1 - probs, min=1e-7))
        
        # Check for Inf/NaN in log values
        if torch.isinf(log_probs_pos).any() or torch.isnan(log_probs_pos).any():
            print(f"  [buttons] Inf/NaN in log(probs)!")
            print(f"    probs min/max: {probs.min():.10f}, {probs.max():.10f}")
            print(f"    probs_orig min/max: {probs_orig.min():.10f}, {probs_orig.max():.10f}")
            print(f"    inf_count: {torch.isinf(log_probs_pos).sum()}, nan_count: {torch.isnan(log_probs_pos).sum()}")
            # Find which indices have issues
            bad_idx = (torch.isinf(log_probs_pos) | torch.isnan(log_probs_pos)).nonzero(as_tuple=True)
            if len(bad_idx[0]) > 0:
                print(f"    First bad indices (batch, button): {bad_idx[0][0]}, {bad_idx[1][0]}")
                print(f"    probs value: {probs[bad_idx[0][0], bad_idx[1][0]]}")
                print(f"    probs_orig value: {probs_orig[bad_idx[0][0], bad_idx[1][0]]}")
        
        if torch.isinf(log_probs_neg).any() or torch.isnan(log_probs_neg).any():
            print(f"  [buttons] Inf/NaN in log(1-probs)!")
            print(f"    1-probs min/max: {(1-probs).min():.10f}, {(1-probs).max():.10f}")
            print(f"    inf_count: {torch.isinf(log_probs_neg).sum()}, nan_count: {torch.isnan(log_probs_neg).sum()}")
        
        button_log_probs = actions * log_probs_pos + (1 - actions) * log_probs_neg
        
        # Check for NaN in button log probs
        if torch.isnan(button_log_probs).any():
            print(f"  [buttons] NaN in button_log_probs!")
            print(f"    actions min/max: {actions.min():.6f}, {actions.max():.6f}")
            nan_mask = torch.isnan(button_log_probs)
            print(f"    NaN positions: {nan_mask.nonzero(as_tuple=True)}")
        
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
            logits = action_logits[head].float()  # [B, num_classes]
            probs = F.softmax(logits, dim=-1)  # [B, num_classes]
            log_probs = F.log_softmax(logits, dim=-1)

            # H(X) = -sum(p * log(p))
            entropy = -(probs * log_probs).sum(dim=-1)  # [B]

            # Check for NaN in categorical entropy
            if torch.isnan(entropy).any() or torch.isinf(entropy).any():
                print(f"  [entropy] NaN/Inf in {head} entropy! nan_count={torch.isnan(entropy).sum()}, inf_count={torch.isinf(entropy).sum()}")
                print(f"    logits range: [{logits.min():.4f}, {logits.max():.4f}]")

            entropies.append(entropy)

    # Buttons (Bernoulli entropy) — numerically stable, compute in float32
    if "buttons" in action_logits:
        logits_in = action_logits["buttons"]  # [B, num_buttons]
        logits = logits_in.float()  # promote to fp32 to avoid fp16 underflow/overflow
        # H(Bernoulli) = -p*log p - (1-p)*log(1-p)
        # Use a stable formulation that avoids log(0) and 0 * -inf in low-precision:
        # log p = log_sigmoid(x); log(1-p) = log_sigmoid(-x)
        # Compute p in fp32 to keep multiplications well-behaved even when p≈0 or p≈1.
        p = torch.sigmoid(logits)  # [B, num_buttons]
        log_p = F.logsigmoid(logits)       # stable log σ(x)
        log_one_minus_p = F.logsigmoid(-logits)  # stable log σ(-x) = log(1 - σ(x))

        # Entropy per button (fp32): -(p*log p + (1-p)*log(1-p))
        button_entropy = -(p * log_p + (1.0 - p) * log_one_minus_p)  # [B, num_buttons]

        # Debug guard (should not trigger now, but helpful if upstream produced NaNs/Infs)
        if torch.isnan(button_entropy).any() or torch.isinf(button_entropy).any():
            sat_hi = (p >= 1.0 - 2**-10).sum().item()  # approx fp16 eps threshold
            sat_lo = (p <= 2**-10).sum().item()
            print(f"  [entropy] NaN/Inf in button_entropy (stable path)! "
                  f"nan_count={torch.isnan(button_entropy).sum()}, inf_count={torch.isinf(button_entropy).sum()}, "
                  f"saturated_hi={sat_hi}, saturated_lo={sat_lo}, dtype_in={logits_in.dtype}")

        # Sum over buttons to get [B]
        entropies.append(button_entropy.sum(dim=-1))  # [B]

    # Sum all entropies
    total_entropy = torch.stack(entropies, dim=0).sum(dim=0)  # [B]
    
    # Check for NaN in final entropy
    if torch.isnan(total_entropy).any() or torch.isinf(total_entropy).any():
        print(f"  [entropy] NaN/Inf in total_entropy! nan_count={torch.isnan(total_entropy).sum()}, inf_count={torch.isinf(total_entropy).sum()}")
    
    return total_entropy


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

    # Check advantages
    if torch.isnan(advantages).any() or torch.isinf(advantages).any():
        print(f"  [loss] NaN/Inf in advantages! nan_count={torch.isnan(advantages).sum()}, inf_count={torch.isinf(advantages).sum()}")
    
    # Clipped surrogate objective
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    
    # Check surrogate objectives
    if torch.isnan(surr1).any() or torch.isinf(surr1).any():
        print(f"  [loss] NaN/Inf in surr1! nan_count={torch.isnan(surr1).sum()}, inf_count={torch.isinf(surr1).sum()}")
    if torch.isnan(surr2).any() or torch.isinf(surr2).any():
        print(f"  [loss] NaN/Inf in surr2! nan_count={torch.isnan(surr2).sum()}, inf_count={torch.isinf(surr2).sum()}")
    
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # Check policy loss
    if torch.isnan(policy_loss) or torch.isinf(policy_loss):
        print(f"  [loss] NaN/Inf in policy_loss! value={policy_loss}")

    # Entropy bonus (encourage exploration)
    entropy = compute_entropy(new_action_logits).mean()
    
    # Check entropy
    if torch.isnan(entropy) or torch.isinf(entropy):
        print(f"  [loss] NaN/Inf in entropy! value={entropy}")
    
    entropy_loss = -entropy_coef * entropy
    
    # Check entropy loss
    if torch.isnan(entropy_loss) or torch.isinf(entropy_loss):
        print(f"  [loss] NaN/Inf in entropy_loss! value={entropy_loss}")

    # Total loss
    total_loss = policy_loss + entropy_loss
    
    # Check total loss
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        print(f"  [loss] NaN/Inf in total_loss! policy_loss={policy_loss}, entropy_loss={entropy_loss}")

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
    # Check inputs
    if torch.isnan(value_pred).any() or torch.isinf(value_pred).any():
        print(f"  [value_loss] NaN/Inf in value_pred! nan_count={torch.isnan(value_pred).sum()}, inf_count={torch.isinf(value_pred).sum()}")
    if torch.isnan(value_target).any() or torch.isinf(value_target).any():
        print(f"  [value_loss] NaN/Inf in value_target! nan_count={torch.isnan(value_target).sum()}, inf_count={torch.isinf(value_target).sum()}")
    
    # Simple MSE loss
    value_loss = F.mse_loss(value_pred, value_target)
    
    # Check MSE result
    if torch.isnan(value_loss) or torch.isinf(value_loss):
        print(f"  [value_loss] NaN/Inf in MSE value_loss! value={value_loss}")

    # Optional: clipped value loss
    if value_clip is not None and old_value_pred is not None:
        if torch.isnan(old_value_pred).any() or torch.isinf(old_value_pred).any():
            print(f"  [value_loss] NaN/Inf in old_value_pred! nan_count={torch.isnan(old_value_pred).sum()}, inf_count={torch.isinf(old_value_pred).sum()}")
        
        value_pred_clipped = old_value_pred + torch.clamp(
            value_pred - old_value_pred,
            -value_clip,
            value_clip,
        )
        value_loss_clipped = F.mse_loss(value_pred_clipped, value_target)
        value_loss = torch.max(value_loss, value_loss_clipped)
        
        # Check clipped loss
        if torch.isnan(value_loss) or torch.isinf(value_loss):
            print(f"  [value_loss] NaN/Inf in clipped value_loss! value={value_loss}")

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
    # Check all inputs
    if torch.isnan(new_values).any() or torch.isinf(new_values).any():
        print(f"  [total_loss] NaN/Inf in new_values! nan_count={torch.isnan(new_values).sum()}, inf_count={torch.isinf(new_values).sum()}")
    if torch.isnan(old_values).any() or torch.isinf(old_values).any():
        print(f"  [total_loss] NaN/Inf in old_values! nan_count={torch.isnan(old_values).sum()}, inf_count={torch.isinf(old_values).sum()}")
    if torch.isnan(returns).any() or torch.isinf(returns).any():
        print(f"  [total_loss] NaN/Inf in returns! nan_count={torch.isnan(returns).sum()}, inf_count={torch.isinf(returns).sum()}")
    
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
    
    # Check policy loss
    if torch.isnan(policy_loss) or torch.isinf(policy_loss):
        print(f"  [total_loss] NaN/Inf in policy_loss after compute! value={policy_loss}")

    # Value loss
    value_loss, value_metrics = compute_value_loss(
        value_pred=new_values,
        value_target=returns,
        old_value_pred=old_values,
        value_clip=value_clip,
    )
    
    # Check value loss
    if torch.isnan(value_loss) or torch.isinf(value_loss):
        print(f"  [total_loss] NaN/Inf in value_loss after compute! value={value_loss}")

    # Total loss
    total_loss = policy_loss + value_coef * value_loss
    
    # Check final total loss
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        print(f"  [total_loss] NaN/Inf in FINAL total_loss! policy_loss={policy_loss}, value_loss={value_loss}, value_coef={value_coef}")

    # Combined metrics
    metrics = {
        **policy_metrics,
        **value_metrics,
        "ppo/total_loss": total_loss.item(),
    }

    return total_loss, metrics
