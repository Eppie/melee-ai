"""Unified action sampling for training and inference.

Provides flexible sampling interface supporting both:
- Training mode: Returns ActionInfo with discrete indices and log probabilities
- Inference mode: Returns ControllerState with continuous [0,1] coordinates

This module consolidates the duplicated action sampling logic from:
- ppo_train.py: sample_actions_with_logprobs(), greedy_actions_with_logprobs()
- model_interface.py: _decode_outputs(), _decode_stick(), _decode_buttons()

Key features:
- Stochastic (multinomial) and greedy (argmax) sampling modes
- Returns either ActionInfo (training) or ControllerState (inference)
- Uses train.log_probability helpers for efficient log prob computation
- Minimal GPU->CPU transfers for performance
"""

from __future__ import annotations

from typing import Literal, TYPE_CHECKING

import numpy as np
import torch

from controller_utils import (
    CONTROL_STICK_QUANTIZED,
    C_STICK_QUANTIZED,
    SHOULDER_QUANTIZED,
)
from train.log_probability import (
    compute_categorical_log_prob,
    LOG_PROB_EPSILON,
)

# Avoid circular imports
if TYPE_CHECKING:
    from ppo_train import ActionInfo
    from model_interface import ControllerState


def sample_controller_outputs(
    outputs: dict,
    mode: Literal["stochastic", "greedy"] = "greedy",
    return_type: Literal["action_info", "controller_state"] = "controller_state",
    return_log_probs: bool = True,
):
    """Unified action sampling for training and inference.

    Args:
        outputs: Model outputs dict/TensorDict with keys:
            - "main_stick": Logits [1, seq_len, 64]
            - "c_stick": Logits [1, seq_len, 9]
            - "buttons": Logits [1, seq_len, 5]
            - "shoulder": Logits [1, seq_len, 5]
        mode: Sampling mode
            - "stochastic": Sample from distribution (multinomial for sticks, Bernoulli for buttons)
            - "greedy": Select argmax (deterministic)
        return_type: Output format
            - "action_info": Training format with discrete indices
            - "controller_state": Inference format with continuous [0,1] coordinates
        return_log_probs: Whether to compute log probabilities (required for action_info)

    Returns:
        ActionInfo: If return_type="action_info"
            - Discrete indices for sticks/shoulder
            - Binary tensor for buttons
            - Log probabilities for all actions
        ControllerState: If return_type="controller_state"
            - Continuous [0,1] coordinates for sticks
            - Float shoulder value
            - Boolean button states

    Performance:
        - Uses manual log_softmax instead of distribution objects (~2x faster)
        - Minimizes GPU->CPU transfers by batching conversions
        - Critical for 60Hz inference loop (must complete in <16ms)

    Examples:
        >>> # PPO training (stochastic)
        >>> action_info = sample_controller_outputs(
        ...     outputs, mode="stochastic", return_type="action_info"
        ... )

        >>> # Inference (greedy)
        >>> controller = sample_controller_outputs(
        ...     outputs, mode="greedy", return_type="controller_state", return_log_probs=False
        ... )
    """
    # Extract last timestep logits [1, seq_len, K] -> [K]
    main_logits = outputs["main_stick"][0, -1]  # [64]
    c_logits = outputs["c_stick"][0, -1]  # [9]
    button_logits = outputs["buttons"][0, -1]  # [5]
    shoulder_logits = outputs["shoulder"][0, -1]  # [5]

    device = main_logits.device

    # Sample or select greedy actions
    if mode == "stochastic":
        # Stochastic: Sample from distributions
        main_idx = torch.multinomial(torch.softmax(main_logits, dim=-1), 1).squeeze(-1)
        c_idx = torch.multinomial(torch.softmax(c_logits, dim=-1), 1).squeeze(-1)
        shoulder_idx = torch.multinomial(
            torch.softmax(shoulder_logits, dim=-1), 1
        ).squeeze(-1)

        # Buttons: Independent Bernoulli sampling
        button_probs = torch.sigmoid(button_logits)
        button_samples = torch.bernoulli(button_probs)
    else:  # greedy
        # Greedy: Argmax selection
        main_idx = torch.argmax(main_logits, dim=-1)
        c_idx = torch.argmax(c_logits, dim=-1)
        shoulder_idx = torch.argmax(shoulder_logits, dim=-1)

        # Buttons: Threshold at 0.5
        button_probs = torch.sigmoid(button_logits)
        button_samples = (button_probs >= 0.5).to(button_probs.dtype)

    # Compute log probabilities if needed
    if return_log_probs or return_type == "action_info":
        main_log_prob = compute_categorical_log_prob(
            main_logits, main_idx, mode="single"
        )
        c_log_prob = compute_categorical_log_prob(c_logits, c_idx, mode="single")
        shoulder_log_prob = compute_categorical_log_prob(
            shoulder_logits, shoulder_idx, mode="single"
        )

        # Button log probs (per-button, not summed)
        button_log_probs = torch.where(
            button_samples == 1,
            torch.log(button_probs + LOG_PROB_EPSILON),
            torch.log(1 - button_probs + LOG_PROB_EPSILON),
        )

    if return_type == "action_info":
        # Training mode: Return ActionInfo with discrete indices
        from ppo_train import ActionInfo, _pack_actions_for_cpu

        main_idx_cpu, c_idx_cpu, shoulder_idx_cpu, buttons_cpu = _pack_actions_for_cpu(
            main_idx, c_idx, shoulder_idx, button_samples
        )

        return ActionInfo(
            main_stick_idx=main_idx_cpu,
            c_stick_idx=c_idx_cpu,
            buttons=buttons_cpu,
            shoulder_idx=shoulder_idx_cpu,
            main_log_prob=main_log_prob.detach(),
            c_log_prob=c_log_prob.detach(),
            buttons_log_probs=button_log_probs.detach(),
            shoulder_log_prob=shoulder_log_prob.detach(),
        )
    else:
        # Inference mode: Return ControllerState with continuous coordinates
        from model_interface import ControllerState, model_to_dolphin01

        # Convert indices to continuous [0, 1] coordinates
        main_idx_np = main_idx.cpu().numpy().astype(np.int32)
        c_idx_np = c_idx.cpu().numpy().astype(np.int32)
        shoulder_idx_np = shoulder_idx.cpu().numpy().astype(np.int32)

        # Palette lookup and coordinate conversion
        main_xy = model_to_dolphin01(
            main_idx_np, palette11=np.asarray(CONTROL_STICK_QUANTIZED, dtype=np.float32)
        ).reshape(-1)
        c_xy = model_to_dolphin01(
            c_idx_np, palette11=np.asarray(C_STICK_QUANTIZED, dtype=np.float32)
        ).reshape(-1)

        # Shoulder: Direct index into palette
        shoulder_val = float(SHOULDER_QUANTIZED[shoulder_idx_np])

        # Buttons: Convert to boolean list
        buttons_bool = button_samples.bool().cpu().tolist()

        return ControllerState(
            main_stick_x=float(main_xy[0]),
            main_stick_y=float(main_xy[1]),
            c_stick_x=float(c_xy[0]),
            c_stick_y=float(c_xy[1]),
            shoulder_analog=shoulder_val,
            button_a=bool(buttons_bool[0]),
            button_b=bool(buttons_bool[1]),
            button_xy=bool(buttons_bool[2]),
            button_z=bool(buttons_bool[3]),
            button_lr=bool(buttons_bool[4]),
        )


def action_info_to_controller_state(action_info) -> ControllerState:
    """Convert ActionInfo (discrete indices) to ControllerState (continuous).

    Useful for debugging or visualization when training.

    Args:
        action_info: ActionInfo with discrete indices

    Returns:
        ControllerState with continuous [0,1] coordinates
    """
    from model_interface import ControllerState, model_to_dolphin01

    # Convert stick indices to continuous coordinates
    main_xy = model_to_dolphin01(
        action_info.main_stick_idx,
        palette11=np.asarray(CONTROL_STICK_QUANTIZED, dtype=np.float32),
    ).reshape(-1)
    c_xy = model_to_dolphin01(
        action_info.c_stick_idx,
        palette11=np.asarray(C_STICK_QUANTIZED, dtype=np.float32),
    ).reshape(-1)

    # Shoulder: Direct palette lookup
    shoulder_val = float(SHOULDER_QUANTIZED[action_info.shoulder_idx])

    # Buttons: Convert tensor to boolean list
    buttons_bool = action_info.buttons.bool().cpu().tolist()

    return ControllerState(
        main_stick_x=float(main_xy[0]),
        main_stick_y=float(main_xy[1]),
        c_stick_x=float(c_xy[0]),
        c_stick_y=float(c_xy[1]),
        shoulder_analog=shoulder_val,
        button_a=bool(buttons_bool[0]),
        button_b=bool(buttons_bool[1]),
        button_xy=bool(buttons_bool[2]),
        button_z=bool(buttons_bool[3]),
        button_lr=bool(buttons_bool[4]),
    )
