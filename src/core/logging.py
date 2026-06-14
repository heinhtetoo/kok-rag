"""Logging configuration for Kök RAG."""

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger with a structured format.

    Args:
        level: Log level string (e.g. "INFO", "DEBUG", "WARNING").
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Get a named logger instance.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        A configured ``logging.Logger`` instance.
    """
    return logging.getLogger(name)
