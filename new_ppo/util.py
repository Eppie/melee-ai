import torch
from typing import List
from schema import get_feature_names

def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float
) -> torch.Tensor:
    """
    Compute Generalized Advantage Estimation (GAE).
    
    Args:
        rewards: [T, N] tensor of rewards
        values: [T, N] tensor of value estimates
        dones: [T, N] tensor of done flags (0.0 for not done, 1.0 for done)
        next_value: [N] tensor of value estimate for the state after the last step
        gamma: Discount factor
        gae_lambda: GAE smoothing parameter
        
    Returns:
        advantages: [T, N] tensor of advantages
    """
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae_lam = torch.zeros(N, device=rewards.device)
    
    for t in reversed(range(T)):
        if t == T - 1:
            next_non_terminal = 1.0 - dones[t]
            next_val = next_value
        else:
            next_non_terminal = 1.0 - dones[t]
            next_val = values[t + 1]
            
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        advantages[t] = last_gae_lam
        
    return advantages

def get_feature_swap_indices() -> List[int]:
    """
    Returns a list of indices to permute a feature vector 
    such that P1 features become P2 features and vice-versa.
    Relies on naming convention 'p1_...' and 'p2_...'.
    """
    names = get_feature_names()
    indices = list(range(len(names)))
    
    for i, name in enumerate(names):
        if name.startswith("p1_"):
            # Construct the p2 equivalent name
            p2_name = "p2_" + name[3:]
            try:
                j = names.index(p2_name)
                # Swap the mapping
                indices[i] = j
                indices[j] = i
            except ValueError:
                # If p2 version doesn't exist, we leave it (or warn)
                pass
                
    return indices

def swap_player_features(obs: torch.Tensor, swap_indices: torch.Tensor) -> torch.Tensor:
    """
    Permute the observation tensor to switch perspectives.
    
    Args:
        obs: [..., F] Feature tensor
        swap_indices: [F] LongTensor of indices
        
    Returns:
        Swapped tensor of same shape
    """
    return obs[..., swap_indices]
