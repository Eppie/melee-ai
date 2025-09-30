"""
Logging utilities for Melee AI.

This module provides centralized logging setup and utilities
for consistent logging across the codebase.
"""

import logging
import sys
from typing import Optional

from melee_ai.config import Settings


def setup_logging(settings: Settings, name: str = "melee_ai") -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        settings: Configuration settings
        name: Logger name

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Set level
    level = getattr(logging, settings.logging.log_level.upper(), logging.INFO)
    logger.setLevel(level)

    # Create formatter
    if settings.logging.enable_json_logging:
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
        )
    else:
        formatter = logging.Formatter(settings.logging.log_format)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if settings.logging.log_dir:
        import os
        from pathlib import Path

        log_dir = Path(settings.logging.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_dir / f"{name}.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str, settings: Optional[Settings] = None) -> logging.Logger:
    """
    Get a logger with the given name.

    Args:
        name: Logger name
        settings: Optional settings (uses global if None)

    Returns:
        Configured logger
    """
    if settings is None:
        # Try to get from global context or create default
        settings = Settings()

    return setup_logging(settings, name)
