"""File utility functions for PyDAC"""


import os

import shutil

from pathlib import Path

from typing import List, Optional


def ensure_dir(directory: str):
    """Ensure directory exists"""

    Path(directory).mkdir(parents=True, exist_ok=True)


def clean_dir(directory: str, pattern: str = "*"):
    """Clean directory by pattern"""

    dir_path = Path(directory)
    if not dir_path.exists():
        return

    for file in dir_path.glob(pattern):
        if file.is_file():
            file.unlink()
        elif file.is_dir():
            shutil.rmtree(file)


def find_files(
    directory: str,
    pattern: str = "*.cpp",
    recursive: bool = True
) -> List[str]:
    """
    Find files matching pattern

    Args:
        directory: Directory to search
        pattern: File pattern (e.g., "*.cpp", "*.dac.cpp")
        recursive: Search recursively

    Returns:
        List of file paths
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        return []

    files = []
    if recursive:
        for file in dir_path.rglob(pattern):
            if file.is_file():
                files.append(str(file))
    else:
        for file in dir_path.glob(pattern):
            if file.is_file():
                files.append(str(file))

    return sorted(files)


def backup_file(file_path: str, suffix: str = ".bak") -> str:
    """
    Backup file

    Args:
        file_path: File to backup
        suffix: Backup suffix

    Returns:
        Backup file path
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    backup_path = path.with_suffix(path.suffix + suffix)
    shutil.copy2(file_path, backup_path)
    return str(backup_path)


def get_file_size(file_path: str) -> int:
    """Get file size in bytes"""

    return os.path.getsize(file_path)


def get_file_Hash(file_path: str) -> str:
    """Get file Hash (MD5)"""

    import hashlib

    Hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            Hash_md5.update(chunk)
    return Hash_md5.hexdigest()


def compare_files(file1: str, file2: str) -> bool:
    """Compare two files"""

    return get_file_Hash(file1) == get_file_Hash(file2)

