from typing import Tuple

import torch


def _rope_cache(seq_len: int, head_dim: int, theta: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    half = head_dim // 2
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = theta ** (-2 * torch.arange(half, device=device, dtype=torch.float32) / head_dim)
    angles = torch.einsum("l,d->l d", positions, freqs)
    cos = angles.cos()[None, None, :, :]
    sin = angles.sin()[None, None, :, :]
    return cos, sin


def apply_rope_inplace(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[
    torch.Tensor, torch.Tensor]:
    q_even, q_odd = q[..., ::2], q[..., 1::2]
    k_even, k_odd = k[..., ::2], k[..., 1::2]
    q[..., ::2] = q_even * cos - q_odd * sin
    q[..., 1::2] = q_even * sin + q_odd * cos
    k[..., ::2] = k_even * cos - k_odd * sin
    k[..., 1::2] = k_even * sin + k_odd * cos
    return q, k