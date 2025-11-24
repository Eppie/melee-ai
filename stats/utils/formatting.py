"""Formatting utilities for statistics output."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence


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


def format_duration(frames: int, fps: int = 60) -> str:
    """Format frame count as human-readable duration.

    Args:
        frames: Number of frames.
        fps: Frames per second.

    Returns:
        Duration string (e.g., "1m 30s" or "45s").
    """
    seconds = frames / fps
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    if minutes < 60:
        return f"{minutes}m {remaining_seconds:.0f}s"
    hours = minutes // 60
    remaining_minutes = minutes % 60
    return f"{hours}h {remaining_minutes}m"


def format_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    indent: int = 2,
    min_width: int = 0,
) -> str:
    """Format data as an ASCII table.

    Args:
        headers: Column headers.
        rows: Row data (list of lists of strings).
        indent: Number of spaces to indent the table.
        min_width: Minimum column width.

    Returns:
        Formatted table string.
    """
    if not headers:
        return ""

    # Calculate column widths
    widths = [max(min_width, len(h)) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            if idx < len(widths) and len(str(cell)) > widths[idx]:
                widths[idx] = len(str(cell))

    indent_str = " " * indent

    def format_row(row: Sequence[str]) -> str:
        cells = [str(cell).ljust(widths[idx]) for idx, cell in enumerate(row)]
        return indent_str + "  ".join(cells)

    lines = [
        format_row(headers),
        indent_str + "  ".join("-" * w for w in widths),
    ]
    for row in rows:
        lines.append(format_row(row))

    return "\n".join(lines)


def truncate_string(s: str, max_length: int = 50, suffix: str = "...") -> str:
    """Truncate a string if it exceeds max length.

    Args:
        s: String to truncate.
        max_length: Maximum length.
        suffix: Suffix to add if truncated.

    Returns:
        Truncated string.
    """
    if len(s) <= max_length:
        return s
    return s[: max_length - len(suffix)] + suffix


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


def indent_text(text: str, spaces: int = 2) -> str:
    """Indent all lines of text.

    Args:
        text: Text to indent.
        spaces: Number of spaces to indent.

    Returns:
        Indented text.
    """
    indent = " " * spaces
    return "\n".join(indent + line for line in text.split("\n"))
