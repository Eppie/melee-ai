"""
Common programming patterns for Melee AI.

This module provides utilities for common patterns like guard clauses
and safe operations that improve code readability.
"""

from typing import Any, Callable, TypeVar

T = TypeVar('T')


def guard_clause(condition: bool, message: str = "Guard condition failed") -> None:
    """
    Guard clause that raises ValueError if condition is false.

    Args:
        condition: Condition to check
        message: Error message if condition fails

    Raises:
        ValueError: If condition is false
    """
    if not condition:
        raise ValueError(message)


def guard_not_none(value: T, message: str = "Value cannot be None") -> T:
    """
    Guard that ensures value is not None.

    Args:
        value: Value to check
        message: Error message if value is None

    Returns:
        The value if not None

    Raises:
        ValueError: If value is None
    """
    if value is None:
        raise ValueError(message)
    return value


def safe_divide(n: float, d: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers with a default value.

    Args:
        n: Numerator
        d: Denominator
        default: Default value if division by zero

    Returns:
        n / d or default if d is zero
    """
    try:
        return n / d if d != 0 else default
    except (ZeroDivisionError, OverflowError):
        return default


def safe_get(dictionary: dict, key: str, default: Any = None) -> Any:
    """
    Safely get a value from a dictionary.

    Args:
        dictionary: Dictionary to get from
        key: Key to look for
        default: Default value if key not found

    Returns:
        Value from dictionary or default
    """
    return dictionary.get(key, default)


def with_default(value: T, default: T) -> T:
    """
    Return value if not None, otherwise return default.

    Args:
        value: Value to check
        default: Default value

    Returns:
        Value or default
    """
    return value if value is not None else default


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp a value between min and max.

    Args:
        value: Value to clamp
        min_val: Minimum value
        max_val: Maximum value

    Returns:
        Clamped value
    """
    return max(min_val, min(value, max_val))
