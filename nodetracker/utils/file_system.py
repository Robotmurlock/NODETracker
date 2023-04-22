"""
File System custom support functions.
"""
import os
from typing import List


def listdir(path: str) -> List[str]:
    """
    Wrapper for `os.listdir` that ignore `.*` files (like `.DS_Store)

    Args:
        path: Directory path

    Returns:
        Listed files (not hidden)
    """
    return [p for p in os.listdir(path) if not p.startswith('.')]