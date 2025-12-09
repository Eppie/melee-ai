"""Loss functions for Mixture Density Networks."""

import torch
import torch.nn.functional as F
from typing import Dict, Optional


def mixture_nll_loss(mixture_params: Dict[str, torch.Tensor], target_delta: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    mixture_params: dict with keys [weights, mu_x, mu_y, sigma_x, sigma_y]
                    each [B, K]
    target_delta: [B, 2] true (dx, dy) in normalized units
    valid_mask: [B] bool mask for valid samples (optional)
    """
    weights = mixture_params['weights']      # [B, K]
    mu_x = mixture_params['mu_x']            # [B, K]
    mu_y = mixture_params['mu_y']            # [B, K]
    sigma_x = mixture_params['sigma_x']      # [B, K]
    sigma_y = mixture_params['sigma_y']      # [B, K]

    # Expand target_delta to match mixture components [B, 1, K]
    target_x = target_delta[:, 0].unsqueeze(-1).expand_as(mu_x) # [B, K]
    target_y = target_delta[:, 1].unsqueeze(-1).expand_as(mu_y) # [B, K]

    # Log probability under each component
    log_px = gaussian_log_prob(target_x, mu_x, sigma_x)  # [B, K]
    log_py = gaussian_log_prob(target_y, mu_y, sigma_y)  # [B, K]
    log_pxy = log_px + log_py  # Independence assumption between x and y

    # Mixture log probability
    log_weights = torch.log(weights + 1e-8) # Add small epsilon for numerical stability
    log_mixture = torch.logsumexp(log_weights + log_pxy, dim=-1)  # [B]

    # Negative log likelihood
    nll = -log_mixture

    # Apply mask if provided
    if valid_mask is not None:
        # Ensure valid_mask is float for multiplication
        nll = nll * valid_mask.float()
        return nll.sum() / valid_mask.sum().clamp(min=1) # Average over valid samples
    else:
        return nll.mean()


def gaussian_log_prob(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """
    Calculates the log probability density of a Gaussian distribution.

    x: [B, K] or [B] (target value)
    mu: [B, K] (mean of each component)
    sigma: [B, K] (standard deviation of each component)
    Returns: [B, K] log probabilities
    """
    # Ensure x has the same dimensions as mu and sigma for element-wise operations
    if x.ndim < mu.ndim:
        x = x.unsqueeze(-1) # Add a K dimension to x for broadcasting
    
    return -0.5 * (
        torch.log(2 * torch.pi * sigma**2) +
        ((x - mu) / sigma)**2
    )
