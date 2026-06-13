"""
Structured logger for the Adaptive RAG system.

All modules import from here to ensure consistent log formatting
across the application.
"""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger with a consistent format.

    Args:
        name: Module name, typically passed as __name__.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        # Avoid adding duplicate handlers if get_logger is called
        # multiple times for the same module name.
        return logger

    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Prevent log records from propagating to the root logger,
    # which avoids duplicate output when libraries also configure logging.
    logger.propagate = False

    return logger