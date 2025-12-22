from __future__ import annotations

from typing import List

import numpy as np
import torch

from controller_utils import (
    CONTROL_STICK_QUANTIZED,
    C_STICK_QUANTIZED,
    SHOULDER_QUANTIZED,
)

# Game-specific constants
try:
    from libmelee.melee.enums import Character, Stage

    # Legal tournament stages for competitive Melee
    LEGAL_TOURNAMENT_STAGES = [
        Stage.BATTLEFIELD,
        Stage.YOSHIS_STORY,
        Stage.POKEMON_STADIUM,
        Stage.DREAMLAND,
        Stage.FINAL_DESTINATION,
        Stage.FOUNTAIN_OF_DREAMS,
    ]

    # Supported characters for training
    SUPPORTED_CHARS = [Character.FOX]
except ImportError:
    # Fallback if libmelee not available
    LEGAL_TOURNAMENT_STAGES = []
    SUPPORTED_CHARS = []

# Port assignments
BOT_PORT = 1  # Port for AI-controlled player
OPP_PORT = 2  # Port for opponent

# Training constants
MIN_EPISODE_LENGTH = 64  # Minimum frames (~1 second) for meaningful training

CONTROLLER_KEY_GROUPS = {
    "main": ("main_stick_x", "main_stick_y"),
    "c": ("c_stick_x", "c_stick_y"),
    "buttons": ("button_a", "button_b", "button_xy", "button_z", "button_lr"),
    "shoulder": ("shoulder_analog",),
}
BUTTON_TARGET_NAMES = tuple(f"p1_{name}" for name in CONTROLLER_KEY_GROUPS["buttons"])
_MAIN_STICK_PALETTE_CPU = torch.as_tensor(
    np.asarray(CONTROL_STICK_QUANTIZED, dtype=np.float32)
)
_C_STICK_PALETTE_CPU = torch.as_tensor(np.asarray(C_STICK_QUANTIZED, dtype=np.float32))
_SHOULDER_PALETTE_CPU = torch.as_tensor(
    np.asarray(SHOULDER_QUANTIZED, dtype=np.float32)
)
# Precompute palette norm squared for faster distance calculations
_MAIN_STICK_NORM_SQ_CPU = (_MAIN_STICK_PALETTE_CPU * _MAIN_STICK_PALETTE_CPU).sum(dim=1)
_C_STICK_NORM_SQ_CPU = (_C_STICK_PALETTE_CPU * _C_STICK_PALETTE_CPU).sum(dim=1)


_MAIN_STICK_CACHE: dict[tuple[str, int | None], torch.Tensor] = {}
_C_STICK_CACHE: dict[tuple[str, int | None], torch.Tensor] = {}
_SHOULDER_CACHE: dict[tuple[str, int | None], torch.Tensor] = {}
_MAIN_STICK_NORM_CACHE: dict[tuple[str, int | None], torch.Tensor] = {}
_C_STICK_NORM_CACHE: dict[tuple[str, int | None], torch.Tensor] = {}
MAX_FRAMES: int = (60 * 60 * 8) + 123  # Full 8 minute replay
_MAIN_STICK_LABELS: List[str] = [
    f"({x:.2f},{y:.2f})" for x, y in CONTROL_STICK_QUANTIZED
]
_BUTTON_PRETTY = {
    "button_a": "A",
    "button_b": "B",
    "button_xy": "X/Y",
    "button_z": "Z",
    "button_lr": "L/R",
}
