"""Batch processing and preparation utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Sequence

import torch
from tensordict import TensorDict
from torch import Tensor

from column_map import ColumnMap


def build_model_inputs(features_batch: Tensor, column_map: ColumnMap) -> TensorDict:
    """Convert raw feature tensors into the structured ``TensorDict`` expected by the model.

    The function slices the ``features_batch`` tensor using indices stored in ``column_map`` and casts
    categorical features to ``torch.long`` so they can be consumed by embedding layers. Continuous
    features (game state and controller values) remain floating point. The resulting dictionary is
    wrapped in a ``TensorDict`` with the same batch shape as the input so downstream code can rely
    on consistent key names.

    Example:
        Suppose ``features_batch`` is shaped ``[2, 3, 6]`` and the column map encodes indices such that
        stage is at column 0, ego character at column 1, opponent character at column 2, ego action
        at column 3, opponent action at column 4, and the remaining columns correspond to
        ``gamestate`` (column 5 onwards) and ``controller`` (the last two columns). The first batch
        might look like::

            features_batch = torch.tensor([
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
        features_batch: ``[batch_size, sequence_length, F]`` float32 features of the current frame sequence.
        column_map: Column mapping for feature indices.

    Returns:
        ``TensorDict`` with the keys the model expects: ``stage``, ``ego_character``,
        ``opponent_character``, ``ego_action``, ``opponent_action``, ``gamestate``, and
        ``controller``.
    """
    batch_size, sequence_length, _ = features_batch.shape

    # Categoricals back to long indices
    stage = (
        features_batch[..., column_map.stage_idx].to(torch.long).unsqueeze(-1)
    )  # [batch_size,sequence_length,1]
    ego_character = (
        features_batch[..., column_map.ego_char_idx].to(torch.long).unsqueeze(-1)
    )
    opp_character = (
        features_batch[..., column_map.opp_char_idx].to(torch.long).unsqueeze(-1)
    )
    ego_action = (
        features_batch[..., column_map.ego_action_idx].to(torch.long).unsqueeze(-1)
    )
    opp_action = (
        features_batch[..., column_map.opp_action_idx].to(torch.long).unsqueeze(-1)
    )

    gamestate = features_batch[
        ..., column_map.gamestate_idxs
    ]  # [batch_size,sequence_length,Gg]
    controller = features_batch[
        ..., column_map.controller_idxs
    ]  # [batch_size,sequence_length,Gc]

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
        batch_size=(batch_size, sequence_length),
    )


@dataclass(frozen=True)
class SampleWeightRatios:
    """How much to upweight 'change' frames vs 'hold' frames, per component."""

    main_change: float = 8.0
    c_change: float = 10.0
    shoulder_change: float = 10.0
    buttons_change_default: float = 10.0
    buttons_change_per_key: Dict[str, float] = field(default_factory=dict)
    hold_base: float = 1.0
    value_change: float = 8.0


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
    ratios: SampleWeightRatios,
    button_names: Sequence[str],
    change_scale: float = 1.0,
) -> Dict[str, Tensor]:
    """Construct dynamic loss weights that emphasize frames where actions change.

    The function inspects quantized controller targets to find frames where each component (main
    stick, C-stick, shoulders, buttons) differs from the previous frame. Change frames receive the
    up-weighting factors from :class:`SampleWeightRatios`, while hold frames receive the base weight.
    The helper then normalizes each component so the average weight stays at one, ensuring the total
    loss magnitude is stable.

    Example:
        Consider a tiny batch with ``batch_size=1`` and ``sequence_length=4`` where the main stick indices are
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
        ratios: Override for the default :class:`SampleWeightRatios`.
        button_names: List of button names that aligns with the ``buttons`` tensor.
        change_scale: Multiplier applied to all change weights (1.0 keeps original ratios).

    Returns:
        Dictionary with per-component weight tensors:

        * ``"main"`` and ``"c"``: ``[batch_size, sequence_length]``
        * ``"shoulder"``: ``[batch_size, sequence_length]``
        * ``"buttons"``: ``[batch_size, sequence_length, K]``
        * ``"global"``: ``[batch_size, sequence_length]`` union of all change indicators
    """
    batch_size, sequence_length = target_info["main_idx"].shape
    change_scale = float(max(min(change_scale, 1.0), 0.0))

    hold_weight = torch.as_tensor(ratios.hold_base, device=device, dtype=torch.float32)

    def _blend(value: float) -> torch.Tensor:
        base = torch.as_tensor(value, device=device, dtype=torch.float32)
        return hold_weight + (base - hold_weight) * change_scale

    main_change_weight = _blend(ratios.main_change)
    c_change_weight = _blend(ratios.c_change)
    shoulder_change_weight = _blend(ratios.shoulder_change)
    value_change_weight = _blend(ratios.value_change)

    # --- MAIN ---
    main_idx = target_info["main_idx"]  # [batch_size, sequence_length]
    main_change = torch.zeros(
        (batch_size, sequence_length), device=device, dtype=torch.bool
    )
    main_change[:, 1:] = main_idx[:, 1:] != main_idx[:, :-1]
    w_main = torch.where(main_change, main_change_weight, hold_weight).to(torch.float32)
    w_main = _normalize(w_main)

    # --- C-STICK ---
    c_idx = target_info["c_idx"]
    c_change = torch.zeros(
        (batch_size, sequence_length), device=device, dtype=torch.bool
    )
    c_change[:, 1:] = c_idx[:, 1:] != c_idx[:, :-1]
    w_c = torch.where(c_change, c_change_weight, hold_weight).to(torch.float32)
    w_c = _normalize(w_c)

    sh_idx = target_info["shoulder_idx"]
    sh_change = torch.zeros(
        (batch_size, sequence_length), device=device, dtype=torch.bool
    )
    sh_change[:, 1:] = sh_idx[:, 1:] != sh_idx[:, :-1]
    w_shoulder = torch.where(sh_change, shoulder_change_weight, hold_weight).to(
        torch.float32
    )
    w_shoulder = _normalize(w_shoulder)

    # --- BUTTONS (per-button) ---
    btn_t = target_info["buttons"].to(
        torch.float32
    )  # [batch_size, sequence_length, K], {0,1}
    K = btn_t.shape[-1]

    # Change mask per button at frame t>0
    btn_change = torch.zeros(
        (batch_size, sequence_length, K), device=device, dtype=torch.bool
    )
    btn_change[:, 1:, :] = btn_t[:, 1:, :] != btn_t[:, :-1, :]

    # Build per-button change ratios
    base_button = float(_blend(ratios.buttons_change_default).item())
    per_button_ratio = torch.full(
        (K,),
        base_button,
        device=device,
        dtype=torch.float32,
    )
    for k, name in enumerate(button_names):
        if name in ratios.buttons_change_per_key:
            per_button_ratio[k] = float(
                _blend(ratios.buttons_change_per_key[name]).item()
            )

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

    w_global = torch.where(
        union_change,
        value_change_weight,
        hold_weight,
    ).to(torch.float32)
    w_global = _normalize(w_global)

    return {
        "main": w_main,  # [batch_size, sequence_length]
        "c": w_c,  # [batch_size, sequence_length]
        "shoulder": w_shoulder,  # [batch_size, sequence_length]
        "buttons": w_buttons,  # [batch_size, sequence_length, K]
        "global": w_global,  # [batch_size, sequence_length]
    }
