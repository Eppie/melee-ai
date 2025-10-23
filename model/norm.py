import torch
from torch.nn import functional as F

try:
    _rms_norm = F.rms_norm  # type: ignore[attr-defined]
except AttributeError:
    _rms_norm = None


def _fallback_rms_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    rms = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(rms + eps)


def norm(x):
    # Purely functional rmsnorm with no learnable params
    if _rms_norm is not None:
        return _rms_norm(x, (x.size(-1),))
    return _fallback_rms_norm(x)
