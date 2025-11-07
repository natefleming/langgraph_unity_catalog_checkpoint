"""Logging configuration for Unity Catalog persistence.

This module configures loguru for the package with sensible defaults.
"""

import sys

from loguru import logger

# Remove default handler
logger.remove()

# Add custom handler with nice formatting
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
)


def configure_logging(level: str = "INFO", serialize: bool = False) -> None:
    """Configure logging for the package.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        serialize: If True, output JSON logs instead of formatted text
    """
    logger.remove()

    if serialize:
        # JSON output for production
        logger.add(
            sys.stderr,
            format="{message}",
            level=level,
            serialize=True,
        )
    else:
        # Pretty output for development
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=level,
            colorize=True,
        )


# Default configuration
configure_logging()
