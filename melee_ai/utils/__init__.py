"""
Common utilities for Melee AI.

This module provides shared utilities and patterns used across
the Melee AI codebase for improved code quality and consistency.
"""

from .result import Result, Ok, Err
from .patterns import guard_clause, safe_divide
from .logging import setup_logging

__all__ = [
    "Result",
    "Ok",
    "Err",
    "guard_clause",
    "safe_divide",
    "setup_logging",
]
