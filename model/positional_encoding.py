from typing import Tuple

import torch


def _ntk_scaled_theta(theta: float, factor: float) -> float:
    return theta * factor


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


def _alibi_slopes(n_head: int) -> torch.Tensor:
    def get_slopes(n: int) -> torch.Tensor:
        import math

        def pow2(x: int) -> int:
            return 2 ** math.floor(math.log2(x))

        m = pow2(n)
        slopes = torch.pow(2, -(torch.arange(1, m + 1, dtype=torch.float32) / m))
        if m < n:
            extra = torch.tensor([slopes[-1] * (i + 2) for i in range(n - m)], dtype=torch.float32)
            slopes = torch.cat([slopes, extra], dim=0)
        return slopes

    return get_slopes(n_head)


def _alibi_bias(batch: int, n_head: int, seq_len: int, device: torch.device) -> torch.Tensor:
    slopes = _alibi_slopes(n_head).to(device)
    positions = torch.arange(seq_len, device=device)
    distance = (positions[None, :] - positions[:, None]).clamp(min=0).to(torch.float32)
    bias = -slopes[:, None, None] * distance[None, :, :]
    return bias.unsqueeze(0).expand(batch, -1, -1, -1)
