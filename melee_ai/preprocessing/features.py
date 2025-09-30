"""
Feature processing utilities for preprocessing.

This module contains utilities for normalizing, quantizing, and transforming
controller inputs and game state features.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np

from libmelee.melee.enums import Action
from melee_ai.config import Settings


class ActionMapper:
    """Maps game actions to dense indices and categories."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._build_action_mapping()

    def _build_action_mapping(self):
        """Build internal action mapping tables."""
        # For now, we'll use the action enum value directly
        # In the future, this could be expanded to include category mappings
        self.num_actions = len(Action)

    def action_to_index(self, action: Action) -> int:
        """Convert Action enum to dense index."""
        return action.value

    def index_to_action(self, index: int) -> Action:
        """Convert dense index back to Action enum."""
        return Action(index)


class Quantizer:
    """Handles quantization of continuous controller inputs."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.action_mapper = ActionMapper(settings)

    def quantize_stick(self, x: float, y: float, palette: np.ndarray) -> Tuple[int, Tuple[float, float]]:
        """
        Quantize stick position to nearest palette entry.

        Args:
            x: X coordinate in [0, 1]
            y: Y coordinate in [0, 1]
            palette: Array of palette positions, shape (N, 2)

        Returns:
            Tuple of (index, (x_quantized, y_quantized))
        """
        # Convert to [-1, 1] range
        x_norm = x * 2.0 - 1.0
        y_norm = y * 2.0 - 1.0

        # Clamp to unit circle
        norm = np.sqrt(x_norm**2 + y_norm**2)
        if norm > 1.0:
            x_norm /= norm
            y_norm /= norm

        # Find nearest neighbor in palette
        distances = np.sum((palette - [x_norm, y_norm])**2, axis=1)
        index = int(np.argmin(distances))

        return index, tuple(palette[index])

    def quantize_buttons(self, buttons: dict) -> dict:
        """
        Quantize button states to discrete values.

        Args:
            buttons: Dictionary of button states

        Returns:
            Dictionary of quantized button states
        """
        quantized = {}

        # Most buttons are already binary
        for button_name, value in buttons.items():
            if button_name in ["button_a", "button_b", "button_z"]:
                quantized[button_name] = 1.0 if value > 0.5 else 0.0
            elif button_name == "button_xy":
                # X/Y are already OR'd together
                quantized[button_name] = 1.0 if value > 0.5 else 0.0
            elif button_name == "button_lr":
                # L/R are already OR'd together
                quantized[button_name] = 1.0 if value > 0.5 else 0.0

        return quantized


class FeatureNormalizer:
    """Normalizes game state features to consistent ranges."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.quantizer = Quantizer(settings)

    def normalize_position(self, x: float, y: float, stage: int) -> Tuple[float, float]:
        """
        Normalize position coordinates for a given stage.

        Args:
            x: X position
            y: Y position
            stage: Stage enum value

        Returns:
            Normalized (x, y) coordinates
        """
        # Stage-specific normalization
        stage_bounds = {
            1: (100.0, 50.0),  # Final Destination
            2: (120.0, 60.0),  # Battlefield
            3: (140.0, 70.0),  # Pokemon Stadium
            4: (130.0, 65.0),  # Dreamland
            5: (110.0, 55.0),  # Fountain of Dreams
            6: (115.0, 58.0),  # Yoshi's Story
        }

        stage_width, stage_height = stage_bounds.get(stage, (100.0, 50.0))

        return x / stage_width, y / stage_height

    def normalize_percent(self, percent: int) -> float:
        """Normalize damage percentage."""
        return min(percent / 300.0, 1.0)  # Cap at 300%

    def normalize_stick_deadzone(self, x: float, y: float) -> Tuple[float, float]:
        """
        Apply radial deadzone to stick inputs.

        Args:
            x: X coordinate in [-1, 1]
            y: Y coordinate in [-1, 1]

        Returns:
            Deadzone-normalized (x, y) coordinates
        """
        deadzone = self.settings.preprocessing.stick_deadzone
        norm = np.sqrt(x**2 + y**2)

        if norm <= deadzone:
            return 0.0, 0.0
        else:
            # Scale to fill the remaining range
            scale = (norm - deadzone) / (1.0 - deadzone)
            return (x / norm) * scale, (y / norm) * scale
