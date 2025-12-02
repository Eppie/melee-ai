"""Formatting utilities for statistics output."""

from __future__ import annotations

import math
from typing import Optional


def format_number(
    value: Optional[float],
    *,
    precision: int = 4,
    is_int: bool = False,
    use_commas: bool = True,
) -> str:
    """Format a number for display.

    Args:
        value: The value to format.
        precision: Decimal precision for floats.
        is_int: If True, format as integer.
        use_commas: If True, use comma separators for large numbers.

    Returns:
        Formatted string.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    if is_int or isinstance(value, int):
        if use_commas:
            return f"{int(round(value)):,}"
        return str(int(round(value)))
    return f"{value:.{precision}f}"


def format_percent(value: Optional[float], precision: int = 2) -> str:
    """Format a percentage value.

    Args:
        value: The percentage value (0-100).
        precision: Decimal precision.

    Returns:
        Formatted percentage string.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:.{precision}f}%"


def format_bytes(num_bytes: int) -> str:
    """Format byte count as human-readable size.

    Args:
        num_bytes: Number of bytes.

    Returns:
        Formatted size string (e.g., "1.5 GB").
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"
