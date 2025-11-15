"""Batch processing and preparation utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence

import torch
from tensordict import TensorDict
from torch import Tensor

from constants import CONTROLLER_KEY_GROUPS
from column_map import ColumnMap
from controller_quantization import quantize_targets


def build_model_inputs(batch_X: torch.FloatTensor, column_map: ColumnMap) -> TensorDict:
    """Convert raw feature tensors into the structured ``TensorDict`` expected by the model.

    The function slices the ``batch_X`` tensor using indices stored in ``colmap`` and casts
    categorical features to ``torch.long`` so they can be consumed by embedding layers. Continuous
    features (game state and controller values) remain floating point. The resulting dictionary is
    wrapped in a ``TensorDict`` with the same batch shape as the input so downstream code can rely
    on consistent key names.

    Example:
        Suppose ``batch_X`` is shaped ``[2, 3, 6]`` and the column map encodes indices such that
        stage is at column 0, ego character at column 1, opponent character at column 2, ego action
        at column 3, opponent action at column 4, and the remaining columns correspond to
        ``gamestate`` (column 5 onwards) and ``controller`` (the last two columns). The first batch
        might look like::

            batch_X = torch.tensor([
                [
                    [3.0, 10.0, 20.0, 4.0, 12.0, 0.1, 0.2],
                    [3.0, 10.0, 20.0, 4.0, 12.0, 0.3, 0.4],
                    [2.0, 11.0, 21.0, 5.0, 13.0, 0.5, 0.6],
                ]
            ])

        ``build_model_inputs`` will slice each column group, cast the five categorical columns to
        integer type, and keep the last two columns as floating point. The returned ``TensorDict``
        contains entries like ``{"stage": tensor([[[3], [3], [2]]], dtype=torch.long)}`` and
        ``{"controller": tensor([[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]])}``, demonstrating how each
        slice of the original tensor is repackaged for the model.

    Args:
        batch_X: ``[B, L, F]`` float32 features of the current frame sequence.
        column_map: Column mapping for feature indices.

    Returns:
        ``TensorDict`` with the keys the model expects: ``stage``, ``ego_character``,
        ``opponent_character``, ``ego_action``, ``opponent_action``, ``gamestate``, and
        ``controller``.
    """
    B, L, _ = batch_X.shape

    # Categoricals back to long indices
    stage = batch_X[..., column_map.stage_idx].to(torch.long).unsqueeze(-1)  # [B,L,1]
    ego_character = batch_X[..., column_map.ego_char_idx].to(torch.long).unsqueeze(-1)
    opp_character = batch_X[..., column_map.opp_char_idx].to(torch.long).unsqueeze(-1)
    ego_action = batch_X[..., column_map.ego_action_idx].to(torch.long).unsqueeze(-1)
    opp_action = batch_X[..., column_map.opp_action_idx].to(torch.long).unsqueeze(-1)

    gamestate = batch_X[..., column_map.gamestate_idxs]  # [B,L,Gg]
    controller = batch_X[..., column_map.controller_idxs]  # [B,L,Gc]

    return TensorDict(
        {
            "stage": stage,
            "ego_character": ego_character,
            "opponent_character": opp_character,
            "ego_action": ego_action,
            "opponent_action": opp_action,
            "gamestate": gamestate,
            "controller": controller,
        },
        batch_size=(B, L),
    )


# TODO: What is the point of this?
def quantize_controller_targets(
    batch_Y: torch.Tensor, colmap: ColumnMap, input_domain: str = "unit11"
) -> Dict[str, torch.Tensor]:
    """Quantize controller outputs to the discrete bins used by the loss functions.

    This is a thin wrapper around :func:`controller_quantization.quantize_targets` that provides a
    consistent entry point for the rest of the training code.

    Example:
        If ``batch_Y`` contains a single sequence ``[[[0.0, 0.5], [0.2, -0.1]]]`` and the column map
        reports that the first column is the main stick and the second is the C-stick, the wrapped
        quantizer will convert those continuous values into categorical indices. With a quantization
        scheme that maps ``0.0`` to bin ``4`` and ``0.5`` to bin ``7``, the resulting dictionary
        includes tensors like ``{"main_idx": tensor([[4, 5]]), "c_idx": tensor([[7, 3]])}`` along
        with masks describing which frames changed. The example shows how continuous values are
        transformed step by step before being returned.

    Args:
        batch_Y: ``[B, L, Y]`` target controller values to quantize.
        colmap: Column mapping for the target indices.
        input_domain: Domain of input values (``"unit11"`` or ``"unit01"``).

    Returns:
        Dictionary with quantized targets and metadata as produced by the underlying quantizer.
    """
    return quantize_targets(batch_Y, colmap, input_domain=input_domain)


@dataclass(frozen=True)
class SampleWeightRatios:
    """How much to upweight 'change' frames vs 'hold' frames, per component."""

    main_change: float = 8.0
    c_change: float = 10.0
    shoulder_change: float = 10.0
    buttons_change_default: float = 10.0
    buttons_change_per_key: Dict[str, float] = field(default_factory=dict)
    hold_base: float = 1.0
    value_change: Optional[float] = None  # None => reuse "main" weights


def _normalize(w: Tensor) -> Tensor:
    """Scale weights so their mean is exactly one, avoiding degenerate zeros.

    Example:
        Passing ``w = tensor([[2.0, 4.0], [6.0, 8.0]])`` results in a mean of ``5.0``. The function
        divides every entry by ``5.0 + 1e-12`` to produce
        ``tensor([[0.4, 0.8], [1.2, 1.6]])``. The step-by-step scaling ensures that subsequent loss
        computations treat the average weight as neutral while preserving the relative emphasis of
        each element.

    Args:
        w: Tensor of arbitrary shape containing positive sample weights.

    Returns:
        Tensor with the same shape as ``w`` where the mean value is one (up to numerical precision).
    """
    return w / (w.mean() + 1e-12)


# TODO: Make this optional via config (might be done already?)
# TODO: Reduce duplication within this function
def compute_component_sample_weights(
    target_info: Mapping[str, Tensor],
    device: torch.device,
    *,
    ratios: Optional[SampleWeightRatios] = None,
    button_names: Optional[Sequence[str]] = None,
    change_scale: float = 1.0,
) -> Dict[str, Tensor]:
    """Construct dynamic loss weights that emphasize frames where actions change.

    The function inspects quantized controller targets to find frames where each component (main
    stick, C-stick, shoulders, buttons) differs from the previous frame. Change frames receive the
    up-weighting factors from :class:`SampleWeightRatios`, while hold frames receive the base weight.
    The helper then normalizes each component so the average weight stays at one, ensuring the total
    loss magnitude is stable.

    Example:
        Consider a tiny batch with ``B=1`` and ``L=4`` where the main stick indices are
        ``[1, 1, 3, 3]`` and a single button toggles ``[0, 1, 1, 0]``. Using the default ratios,
        ``compute_component_sample_weights``:

        #. Detects that the main stick only changes at frame 2 (index ``3``) and assigns the
           ``main_change`` weight ``8.0`` there while giving ``1.0`` to the hold frames.
        #. For the button, it spots changes at frames 1 and 3, so those frames are weighted ``10.0``
           while the others remain ``1.0``.
        #. After normalizing, the returned tensors might look like ``main = tensor([[0.5, 0.5, 2.0, 2.0]])``
           and ``buttons = tensor([[[0.4], [1.6], [1.6], [0.4]]])``, illustrating how each component
           is scaled relative to the original ratios yet keeps a mean of one.

    Args:
        target_info: Mapping containing quantized indices and button states produced by
            :func:`quantize_controller_targets`.
        device: Target device for the constructed tensors.
        ratios: Optional override for the default :class:`SampleWeightRatios`.
        button_names: Optional list of button names that aligns with the ``buttons`` tensor.
        change_scale: Multiplier applied to all change weights (1.0 keeps original ratios).

    Returns:
        Dictionary with per-component weight tensors:

        * ``"main"`` and ``"c"``: ``[B, L]``
        * ``"shoulder"``: ``[B, L]`` (or ones if shoulders are absent)
        * ``"buttons"``: ``[B, L, K]``
        * ``"global"``: ``[B, L]`` union of all change indicators
    """
    r = ratios or SampleWeightRatios()
    B, L = target_info["main_idx"].shape
    change_scale = float(max(min(change_scale, 1.0), 0.0))

    hold_weight = torch.as_tensor(r.hold_base, device=device, dtype=torch.float32)

    def _blend(value: float) -> torch.Tensor:
        base = torch.as_tensor(value, device=device, dtype=torch.float32)
        return hold_weight + (base - hold_weight) * change_scale

    main_change_weight = _blend(r.main_change)
    c_change_weight = _blend(r.c_change)
    shoulder_change_weight = _blend(r.shoulder_change)
    value_change_weight = _blend(r.value_change) if r.value_change is not None else None

    # --- MAIN ---
    main_idx = target_info["main_idx"]  # [B, L]
    main_change = torch.zeros((B, L), device=device, dtype=torch.bool)
    if L > 1:
        main_change[:, 1:] = main_idx[:, 1:] != main_idx[:, :-1]
    w_main = torch.where(main_change, main_change_weight, hold_weight).to(torch.float32)
    w_main = _normalize(w_main)

    # --- C-STICK ---
    c_idx = target_info["c_idx"]
    c_change = torch.zeros((B, L), device=device, dtype=torch.bool)
    if L > 1:
        c_change[:, 1:] = c_idx[:, 1:] != c_idx[:, :-1]
    w_c = torch.where(c_change, c_change_weight, hold_weight).to(torch.float32)
    w_c = _normalize(w_c)

    sh_idx = target_info.get("shoulder_idx")

    sh_change = torch.zeros((B, L), device=device, dtype=torch.bool)
    if L > 1:
        sh_change[:, 1:] = sh_idx[:, 1:] != sh_idx[:, :-1]
    w_shoulder = torch.where(sh_change, shoulder_change_weight, hold_weight).to(
        torch.float32
    )
    w_shoulder = _normalize(w_shoulder)

    # --- BUTTONS (per-button) ---
    btn_t = target_info["buttons"].to(torch.float32)  # [B, L, K], {0,1}
    K = btn_t.shape[-1]
    if button_names is None:
        button_names = CONTROLLER_KEY_GROUPS["buttons"]

    # Change mask per button at frame t>0
    btn_change = torch.zeros((B, L, K), device=device, dtype=torch.bool)
    if L > 1:
        btn_change[:, 1:, :] = btn_t[:, 1:, :] != btn_t[:, :-1, :]

    # Build per-button change ratios
    base_button = float(_blend(r.buttons_change_default).item())
    per_button_ratio = torch.full(
        (K,),
        base_button,
        device=device,
        dtype=torch.float32,
    )
    for k, name in enumerate(button_names):
        if name in r.buttons_change_per_key:
            per_button_ratio[k] = float(_blend(r.buttons_change_per_key[name]).item())

    # weights = hold_base on holds; ratio_k on changes of button k
    w_buttons = torch.where(
        btn_change,
        per_button_ratio.view(1, 1, K),
        hold_weight.view(1, 1, 1),
    ).to(torch.float32)
    w_buttons = _normalize(w_buttons)

    # --- GLOBAL (union-of-changes) ---
    union_change = main_change | c_change
    union_change = union_change | sh_change
    union_change = union_change | btn_change.any(dim=-1)

    value_ratio = (
        value_change_weight if value_change_weight is not None else main_change_weight
    )
    w_global = torch.where(
        union_change,
        value_ratio,
        hold_weight,
    ).to(torch.float32)
    w_global = _normalize(w_global)

    return {
        "main": w_main,  # [B, L]
        "c": w_c,  # [B, L]
        "shoulder": w_shoulder,  # [B, L]
        "buttons": w_buttons,  # [B, L, K]
        "global": w_global,  # [B, L]
    }
