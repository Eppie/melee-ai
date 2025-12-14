from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from column_map import ColumnMap
from constants import (
    _MAIN_STICK_PALETTE_CPU,
    _C_STICK_PALETTE_CPU,
    _SHOULDER_PALETTE_CPU,
    _MAIN_STICK_NORM_SQ_CPU,
    _C_STICK_NORM_SQ_CPU,
    _MAIN_STICK_CACHE,
    _MAIN_STICK_NORM_CACHE,
    _C_STICK_CACHE,
    _C_STICK_NORM_CACHE,
    _SHOULDER_CACHE,
)
from controller_utils import (
    SHOULDER_QUANTIZED,
)
from controller_quantization_shared import (
    InputDomain,
    clamp_unit_circle,
    quantize_stick_indices_unit01,
    quantize_stick_indices_unit11,
    sticks01_to_unit11,
)

_clamp_unit_circle = clamp_unit_circle


def _device_cache_key(device: torch.device) -> Tuple[str, Optional[int]]:
    """Canonicalize a :class:`torch.device` for palette caching with an example.

    Example
    -------
    * ``torch.device("cuda")`` with CUDA available resolves to
      ``("cuda", current_device())`` so that the cache differentiates GPU cards.
    * ``torch.device("cpu")`` becomes ``("cpu", None)`` to share CPU copies.

    Given two successive calls with these devices, the quantization helper will
    reuse the cached palettes because the tuple keys are stable regardless of how
    the device objects were constructed.
    """
    if device.type == "cuda":
        index = device.index
        if index is None and torch.cuda.is_available():
            index = torch.cuda.current_device()
        return device.type, index
    return device.type, device.index


# TODO: Why do we need this function?
def _palette_for_device(
    cpu_palette: torch.Tensor,
    cache: Dict[Tuple[str, Optional[int]], torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Move a palette tensor to ``device`` once and reuse the cached copy.

    Example
    -------
    Suppose ``cpu_palette`` is ``tensor([[0., 0.], [1., 0.]])`` and ``device`` is
    ``cuda:0``:

    1. The cache is initially empty so ``key=('cuda', 0)`` is missing.
    2. The palette is transferred using ``cpu_palette.to(device)`` and stored.
    3. A second call with the same device hits the cache and returns the already
       moved tensor without copying again.

    Passing ``torch.device('cpu')`` simply returns ``cpu_palette`` directly,
    showing how CPU execution avoids needless clones.
    """
    if device.type == "cpu":
        return cpu_palette
    key = _device_cache_key(device)
    cached = cache.get(key)
    if cached is None or cached.device != device:
        cached = cpu_palette.to(device=device)
        cache[key] = cached
    return cached


def _quantize_stick(
    xy: torch.Tensor,
    palette: torch.Tensor,
    palette_norm_sq: torch.Tensor,
    input_domain: InputDomain,
    batch_size: int,
    seq_len: int,
) -> torch.Tensor:
    """Convert continuous stick coordinates to palette indices via distance."""
    if input_domain == "unit01":
        idx = quantize_stick_indices_unit01(xy, palette, palette_norm_sq)
    elif input_domain == "unit11":
        idx = quantize_stick_indices_unit11(xy, palette, palette_norm_sq)
    else:
        raise ValueError(f"Unknown input domain: {input_domain}")
    return idx.view(batch_size, seq_len)


def quantize_targets(
    targets: Tensor,
    column_map: ColumnMap,
    *,
    input_domain: InputDomain,
) -> Dict[str, torch.Tensor]:
    """Quantize raw controller targets with a frame-by-frame walkthrough.

    Example
    -------
    Imagine ``batch_Y`` has shape ``(2, 3, F)`` where the column map says the
    first two feature columns are ``p1_main_stick_(x, y)``, the next two are
    ``p1_c_stick_(x, y)``, and button probabilities follow. For the first batch
    element we might have::

        main stick  -> [[0.2, 0.8], [1.0, 0.4], [0.5, 0.5]]
        c-stick     -> [[0.1, 1.0], [0.4, 0.4], [0.8, 0.1]]
        buttons     -> [[0.7, 0.1, 0.9, 0.0, 0.2], ...]

    ``quantize_targets`` performs the following steps for each batch:

    1. Slice stick blocks using ``column_map`` and move the precomputed palettes to
       the tensor's device with :func:`_palette_for_device`.
    2. Call :func:`_quantize_stick` to map every ``(x, y)`` vector to the nearest
       palette entry. For the first main-stick frame above the nearest palette
       might be ``[-0.125, 0.875]`` at index ``14``.
    3. Repeat for the C-stick and (optionally) shoulder analog values, producing
       integer index tensors shaped like ``(batch_size, seq_len)``. Shoulder values are snapped
       to the largest palette element that does not exceed the raw value.
    4. Clamp button probabilities into ``[0, 1]`` without otherwise changing
       their shape, keeping them ready for BCE losses.

    The returned dictionary contains the quantized indices, the button tensor,
    and metadata about palette cardinalities so callers can set up embeddings or
    classification heads without re-deriving these values.
    """
    batch_size, seq_len, _ = targets.shape
    device = targets.device

    # Main stick quantization
    main_xy = targets[..., list(column_map.y_main)]
    P_main = _palette_for_device(_MAIN_STICK_PALETTE_CPU, _MAIN_STICK_CACHE, device)
    P_main_norm_sq = _palette_for_device(
        _MAIN_STICK_NORM_SQ_CPU, _MAIN_STICK_NORM_CACHE, device
    )
    y_main_idx = _quantize_stick(
        main_xy, P_main, P_main_norm_sq, input_domain, batch_size, seq_len
    )

    # C-stick quantization
    c_xy = targets[..., list(column_map.y_c)]
    P_c = _palette_for_device(_C_STICK_PALETTE_CPU, _C_STICK_CACHE, device)
    P_c_norm_sq = _palette_for_device(_C_STICK_NORM_SQ_CPU, _C_STICK_NORM_CACHE, device)
    y_c_idx = _quantize_stick(c_xy, P_c, P_c_norm_sq, input_domain, batch_size, seq_len)

    # Buttons remain probabilistic targets
    btn_cols = column_map.y_buttons
    y_buttons = targets[..., btn_cols].to(torch.float32)
    y_buttons = torch.clamp(y_buttons, 0.0, 1.0)

    centers = _palette_for_device(_SHOULDER_PALETTE_CPU, _SHOULDER_CACHE, device)
    centers = centers.view(-1)
    s = targets[..., column_map.y_shoulder].to(dtype=centers.dtype).contiguous()
    y_shoulder_idx = torch.searchsorted(centers, s, right=True) - 1
    y_shoulder_idx = torch.clamp(y_shoulder_idx, min=0, max=int(centers.shape[0] - 1))
    shoulder_K = len(SHOULDER_QUANTIZED)

    return {
        "main_idx": y_main_idx,
        "c_idx": y_c_idx,
        "buttons": y_buttons,
        "shoulder_idx": y_shoulder_idx,
        "main_K": P_main.shape[0],
        "c_K": P_c.shape[0],
        "buttons_K": y_buttons.shape[-1],
        "shoulder_K": shoulder_K,
    }
