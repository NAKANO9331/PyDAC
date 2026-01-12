"""Cache management for PyDAC"""


import os

import hashlib

import shutil

from pathlib import Path

from typing import Optional, Dict

import json

import time


class CacheManager:

    """Cache manager for compilation results"""

    def __init__(self, cache_dir: str = ".pydac_cache"):
        """
        Initialize cache manager

        Args:
            cache_dir: Cache directory path
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "metadata.json"
        self._load_metadata()

    def _load_metadata(self):
        """Load cache metadata"""

        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

    def _save_metadata(self):
        """Save cache metadata"""

        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def _get_cache_key(self, source_file: str, flags: list) -> str:
        """Generate cache key"""

        # Read source file
        with open(source_file, 'rb') as f:
            content = f.read()

        # Get file modification time
        mtime = os.path.getmtime(source_file)

        # Generate Hash
        key_data = content + b''.join(flags) + str(mtime).encode()
        return hashlib.md5(key_data).hexdigest()

    def get(self, source_file: str, flags: list) -> Optional[str]:
        """
        Get cached binary path

        Args:
            source_file: Source file path
            flags: Compilation flags

        Returns:
            Cached binary path or None
        """
        cache_key = self._get_cache_key(source_file, flags)
        cached_binary = self.cache_dir / f"{cache_key}.bin"

        if cached_binary.exists():
            # Update access time
            self.metadata[cache_key] = {
                "source_file": source_file,
                "flags": flags,
                "access_time": time.time()
            }
            self._save_metadata()
            return str(cached_binary)

        return None

    def put(self, source_file: str, flags: list, binary_file: str):
        """
        Cache binary file

        Args:
            source_file: Source file path
            flags: Compilation flags
            binary_file: Binary file path
        """
        cache_key = self._get_cache_key(source_file, flags)
        cached_binary = self.cache_dir / f"{cache_key}.bin"

        # Copy binary to cache
        shutil.copy2(binary_file, cached_binary)

        # Update metadata
        self.metadata[cache_key] = {
            "source_file": source_file,
            "flags": flags,
            "access_time": time.time(),
            "size": os.path.getsize(cached_binary)
        }
        self._save_metadata()

    def clear(self, older_than_days: Optional[int] = None):
        """
        Clear cache

        Args:
            older_than_days: Clear entries older than N days (None for all)
        """
        if older_than_days is None:
            # Clear all
            for file in self.cache_dir.glob("*.bin"):
                file.unlink()
            self.metadata = {}
            self._save_metadata()
        else:
            # Clear old entries
            cutoff_time = time.time() - (older_than_days * 24 * 3600)
            to_remove = []

            for cache_key, info in self.metadata.items():
                if info.get("access_time", 0) < cutoff_time:
                    cached_binary = self.cache_dir / f"{cache_key}.bin"
                    if cached_binary.exists():
                        cached_binary.unlink()
                    to_remove.append(cache_key)

            for key in to_remove:
                del self.metadata[key]

            self._save_metadata()

    def get_stats(self) -> Dict:
        """Get cache statistics"""

        total_size = sum(
            os.path.getsize(self.cache_dir / f"{key}.bin")
            for key in self.metadata.keys()
            if (self.cache_dir / f"{key}.bin").exists()
        )

        return {
            "total_entries": len(self.metadata),
            "total_size": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "cache_dir": str(self.cache_dir)
        }

