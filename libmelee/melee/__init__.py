"""Python package initialization for libmelee.melee."""

from .console import Console  # re-export common entry points
from .gamestate import GameState

__all__ = ["Console", "GameState"]
