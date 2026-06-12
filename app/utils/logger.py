"""
app/utils/logger.py

Centralised logging configuration.

Every module obtains a logger via:
    from app.utils.logger import get_logger
    logger = get_logger(__name__)

This keeps log formatting and level consistent across the entire application.
The log level is driven by the LOG_LEVEL environment variable through Settings.
"""

import logging
import sys
from app.utils.config import get_settings


def _configure_root_logger() -> None:
    """
    Configure the root logger once when this module is first imported.

    Format: [timestamp] [level] [module] message
    Output: stdout (container-friendly; log aggregators capture stdout).
    """
    settings = get_settings()
    level = logging.getLevelName(settings.log_level.upper())

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)

    # Avoid duplicate handlers if the module is reloaded during development
    if not root.handlers:
        root.addHandler(handler)


# Configure once at import time
_configure_root_logger()


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger.

    Args:
        name: Typically __name__ of the calling module.

    Returns:
        A configured logging.Logger instance.
    """
    return logging.getLogger(name)