"""
Unified configuration system for Melee AI.

This module provides a single source of truth for all configuration parameters
used across the entire Melee AI pipeline.
"""

from .settings import Settings, load_settings
from .constants import (
    FOX_STICK_64,
    C_STICK_XY_CLUSTER_CENTERS_V0_1,
    STICK_XY_CLUSTER_CENTERS_V2,
    MAX_FRAMES,
)
from .cli import create_argument_parser, load_settings_from_args, print_settings

__all__ = [
    "Settings",
    "load_settings",
    "load_settings_from_args",
    "create_argument_parser",
    "print_settings",
    "FOX_STICK_64",
    "C_STICK_XY_CLUSTER_CENTERS_V0_1",
    "STICK_XY_CLUSTER_CENTERS_V2",
    "MAX_FRAMES",
]
