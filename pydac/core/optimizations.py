"""Performance optimizations for PyDAC"""


import os

import hashlib

import json

from pathlib import Path

from typing import Optional, Dict, Any

from functools import lru_cache

import time


class ResultCache:

    """Cache for translation and analysis results"""

    def __init__(self, cache_dir: str = ".pydac_result_cache"):
        """
        Initialize result cache

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

    def _get_cache_key(self, file_path: str, operation: str, params: Dict[str, Any]) -> str:

        """Generate cache key"""

        # Get file content Hash
        with open(file_path, 'rb') as f:
            content = f.read()

        # Get file modification time
        mtime = os.path.getmtime(file_path)

        # Generate key
        key_data = (
            content +
            operation.encode() +
            json.dumps(params, sort_keys=True).encode() +
            str(mtime).encode()
        )
        return hashlib.md5(key_data).hexdigest()

    def get(self, file_path: str, operation: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Get cached result

        Args:
            file_path: File path
            operation: Operation type (translation, analysis, etc.)
            params: Operation parameters

        Returns:
            Cached result or None
        """
        cache_key = self._get_cache_key(file_path, operation, params)
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            # Check if cache is still valid
            file_mtime = os.path.getmtime(file_path)
            cache_mtime = os.path.getmtime(cache_file)

            if cache_mtime >= file_mtime:
                with open(cache_file, 'r') as f:
                    return json.load(f)

        return None

    def put(self, file_path: str, operation: str, params: Dict[str, Any], result: Dict[str, Any]):
        """
        Cache result

        Args:
            file_path: File path
            operation: Operation type
            params: Operation parameters
            result: Result to cache
        """
        cache_key = self._get_cache_key(file_path, operation, params)
        cache_file = self.cache_dir / f"{cache_key}.json"

        with open(cache_file, 'w') as f:
            json.dump(result, f, indent=2)

        # Update metadata
        self.metadata[cache_key] = {
            "file": file_path,
            "operation": operation,
            "params": params,
            "timestamp": time.time()
        }
        self._save_metadata()


class ProcessPool:

    """Simple process pool for subprocess operations"""


    def __init__(self, max_workers: int = 4):
        """
        Initialize process pool

        Args:
            max_workers: Maximum number of worker processes
        """
        self.max_workers = max_workers
        self._pool = []

    def submit(self, func, *args, **kwargs):
        """Submit task to pool"""

        from concurrent.futures import ThreadPoolExecutor

        if not Hasattr(self, '_executor'):
            self._executor = ThreadPoolExecutor(max_workers=self.max_workers)

        return self._executor.submit(func, *args, **kwargs)

    def shutdown(self, wait: bool = True):
        """Shutdown pool"""

        if Hasattr(self, '_executor'):
            self._executor.shutdown(wait=wait)


    @lru_cache(maxsize=128)
    def cached_file_Hash(file_path: str) -> str:
        """Cached file Hash calculation"""

        with open(file_path, 'rb') as f:
            content = f.read()
        return hashlib.md5(content).hexdigest()


    def optimize_file_operations(file_paths: list) -> list:
        """
        Optimize file operations by batching

        Args:
            file_paths: List of file paths

        Returns:
            Optimized list of file paths
        """
        # Sort by size (smaller first for better cache locality)
        file_sizes = [(path, os.path.getsize(path)) for path in file_paths if os.path.exists(path)]
        file_sizes.sort(key=lambda x: x[1])
        return [path for path, _ in file_sizes]


# Import memory and I/O optimization utilities
    from ..utils.memory import (
 StreamingFileReader,
 MemoryPool,
 BufferedFileWriter,
 process_file_streaming,
 estimate_memory_usage
    )

    from ..utils.io_optimizer import (
 BatchFileOperator,
 AsyncFileOperator,
 optimize_file_reads
    )


