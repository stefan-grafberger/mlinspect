"""
Some useful utils for the project
"""
from pathlib import Path


def get_project_root() -> Path:
    """Returns the project root folder."""
    return Path(__file__).parent.parent.parent
