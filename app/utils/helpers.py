from pathlib import Path


def get_file_extension(filename: str) -> str:
    return Path(filename).suffix.lower()
