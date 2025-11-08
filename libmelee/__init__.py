"""Expose libmelee as a regular Python package for tooling like mypy."""

from importlib import resources as _resources

__all__ = ["melee", "melee_rust"]
