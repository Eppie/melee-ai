from loguru import logger
from pathlib import Path
import sys


def setup_logging():
    """
    Configures Loguru for application-wide logging.

    - Removes default handler to prevent duplicate messages.
    - Adds a console sink to stderr with a colored format.
    - Adds a file sink to 'logs/training.log' with rotation, retention, and compression.
    """
    logger.remove()  # Remove default handler

    # Add console sink
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
    )

    # Ensure logs directory exists
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Add file sink
    logger.add(
        log_dir / "training.log",
        level="DEBUG",
        rotation="10 MB",  # Rotate every 10 MB
        retention="7 days",  # Keep logs for 7 days
        compression="zip",  # Compress rotated logs
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        enqueue=True,  # Use a separate thread for logging to file
    )

    logger.info("Loguru logging configured.")


# The function will be called by config/config.py
