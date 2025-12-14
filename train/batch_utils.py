"""Batch processing and preparation utilities."""

from __future__ import annotations

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
