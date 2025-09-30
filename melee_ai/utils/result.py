"""
Structured result pattern for error handling.

This module provides a Result type that encapsulates either success or failure,
making error handling more explicit and composable than exceptions.
"""

from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar, Union

T = TypeVar('T')
E = TypeVar('E')


@dataclass
class Result(Generic[T, E]):
    """A result that can be either Ok(value) or Err(error)."""

    _value: Union[T, E, None] = None
    _is_ok: bool = True

    def __post_init__(self) -> None:
        if not hasattr(self, '_value') or self._value is None:
            raise ValueError("Result must be created with Ok() or Err()")

    @classmethod
    def Ok(cls, value: T) -> 'Result[T, E]':
        """Create a successful result."""
        return cls(_value=value, _is_ok=True)

    @classmethod
    def Err(cls, error: E) -> 'Result[T, E]':
        """Create a failed result."""
        return cls(_value=error, _is_ok=False)

    def is_ok(self) -> bool:
        """Check if result is successful."""
        return self._is_ok

    def is_err(self) -> bool:
        """Check if result is failed."""
        return not self._is_ok

    def unwrap(self) -> T:
        """Get the value, raising an exception if failed."""
        if self._is_ok:
            return self._value  # type: ignore
        else:
            raise ValueError(f"Called unwrap() on an Err result: {self._value}")

    def unwrap_or(self, default: T) -> T:
        """Get the value or return default if failed."""
        return self._value if self._is_ok else default  # type: ignore

    def unwrap_or_else(self, func: Callable[[E], T]) -> T:
        """Get the value or compute default from function if failed."""
        return self._value if self._is_ok else func(self._value)  # type: ignore

    def map(self, func: Callable[[T], Any]) -> 'Result[Any, E]':
        """Transform the value if successful."""
        if self._is_ok:
            return Ok(func(self._value))  # type: ignore
        else:
            return self  # type: ignore

    def map_err(self, func: Callable[[E], Any]) -> 'Result[T, Any]':
        """Transform the error if failed."""
        if self._is_err():
            return Err(func(self._value))  # type: ignore
        else:
            return self  # type: ignore

    def and_then(self, func: Callable[[T], 'Result[Any, E]']) -> 'Result[Any, E]':
        """Chain operations that return Results."""
        if self._is_ok:
            return func(self._value)  # type: ignore
        else:
            return self  # type: ignore

    def __repr__(self) -> str:
        if self._is_ok:
            return f"Ok({self._value})"
        else:
            return f"Err({self._value})"


# Convenience functions
Ok = Result.Ok
Err = Result.Err


def try_catch(func: Callable[..., T], *args, **kwargs) -> Result[T, Exception]:
    """Wrap a function call in a Result."""
    try:
        return Ok(func(*args, **kwargs))
    except Exception as e:
        return Err(e)


def safe_divide(numerator: float, denominator: float) -> Result[float, str]:
    """Safely divide two numbers, returning Result."""
    if denominator == 0:
        return Err("Division by zero")
    return Ok(numerator / denominator)
