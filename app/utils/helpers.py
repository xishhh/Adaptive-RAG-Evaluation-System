"""
app/utils/helpers.py

Shared utility functions used across the application.

Keep this module small and focused on pure utility logic only.
Business logic belongs in services/, not here.
"""

import time
import uuid
from pathlib import Path
from typing import Any


def generate_id() -> str:
    """
    Generate a unique identifier string.

    Used for chunk IDs, evaluation run IDs, etc.

    Returns:
        A UUID4 hex string (no hyphens).
    """
    return uuid.uuid4().hex


def get_file_extension(filename: str) -> str:
    """
    Extract the lowercase file extension from a filename.

    Args:
        filename: The original filename, e.g. "report.PDF"

    Returns:
        Lowercase extension including the dot, e.g. ".pdf"
        Empty string if no extension is found.
    """
    return Path(filename).suffix.lower()


def is_supported_file_type(filename: str) -> bool:
    """
    Check whether a file type is supported for ingestion.

    Supported types are defined by the project requirements:
    PDF, DOCX, TXT, XLSX.

    Args:
        filename: The filename to check.

    Returns:
        True if the extension is supported, False otherwise.
    """
    supported = {".pdf", ".docx", ".txt", ".xlsx"}
    return get_file_extension(filename) in supported


def timer(func: Any) -> Any:
    """
    Simple decorator that logs the execution time of a function.

    Intended for development profiling; does not affect return values.

    Usage:
        @timer
        def my_function(): ...
    """
    import functools
    from app.utils.logger import get_logger

    logger = get_logger(__name__)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.debug(f"{func.__qualname__} completed in {elapsed:.3f}s")
        return result

    return wrapper