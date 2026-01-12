"""I/O optimization utilities for PyDAC"""


import os

import asyncio

from pathlib import Path

from typing import List, Optional, Callable, Any

from concurrent.futures import ThreadPoolExecutor


class BatchFileOperator:

    """Batch file operations for efficient I/O"""

    def __init__(self, max_workers: int = 4):
        """
        Initialize batch file operator

        Args:
            max_workers: Maximum number of worker threads
        """
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def read_files(self, file_paths: List[str]) -> List[tuple]:
        """
        Read multiple files in parallel

        Args:
            file_paths: List of file paths

        Returns:
            List of (path, content) tuples
        """
        def read_file(path: str):
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    return (path, f.read())
            except Exception as e:
                return (path, None)

        futures = [self._executor.submit(read_file, path) for path in file_paths]
        return [future.result() for future in futures]

    def write_files(self, file_data: List[tuple]) -> List[bool]:
        """
        Write multiple files in parallel

        Args:
            file_data: List of (path, content) tuples

        Returns:
            List of success flags
        """
        def write_file(path: str, content: str):
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True
            except Exception as e:
                return False

        futures = [
            self._executor.submit(write_file, path, content)
            for path, content in file_data
        ]
        return [future.result() for future in futures]

    def process_files(
        self,
        file_paths: List[str],
        processor: Callable[[str, str], Any]
    ) -> List[Any]:
        """
        Process multiple files in parallel

        Args:
            file_paths: List of file paths
            processor: Function to process each file (path, content) -> result

        Returns:
            List of processing results
        """
        def process_file(path: str):
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                return processor(path, content)
            except Exception as e:
                return None

        futures = [self._executor.submit(process_file, path) for path in file_paths]
        return [future.result() for future in futures]

    def shutdown(self):
        """Shutdown executor"""

        self._executor.shutdown(wait=True)


class AsyncFileOperator:

    """Async file operations"""


    @staticmethod
    async def read_file(file_path: str) -> Optional[str]:
        """
        Async read file

        Args:
            file_path: Path to file

        Returns:
            File content or None
        """
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(
                None,
                lambda: Path(file_path).read_text(encoding='utf-8', errors='ignore')
            )
        except Exception:
            return None

    @staticmethod
    async def write_file(file_path: str, content: str) -> bool:
        """
        Async write file

        Args:
            file_path: Path to file
            content: Content to write

        Returns:
            Success flag
        """
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                lambda: Path(file_path).write_text(content, encoding='utf-8')
            )
            return True
        except Exception:
            return False

    @staticmethod
    async def read_files(file_paths: List[str]) -> List[Optional[str]]:
        """
        Async read multiple files

        Args:
            file_paths: List of file paths

        Returns:
            List of file contents
        """
        tasks = [AsyncFileOperator.read_file(path) for path in file_paths]
        return await asyncio.gather(*tasks)

    @staticmethod
    async def write_files(file_data: List[tuple]) -> List[bool]:
        """
        Async write multiple files

        Args:
            file_data: List of (path, content) tuples

        Returns:
            List of success flags
        """
        tasks = [
            AsyncFileOperator.write_file(path, content)
            for path, content in file_data
        ]
        return await asyncio.gather(*tasks)


def optimize_file_reads(file_paths: List[str], batch_size: int = 10) -> List[str]:
    """
    Optimize file read order

    Args:
        file_paths: List of file paths
        batch_size: Size of batches

    Returns:
        Optimized list of file paths
    """
    # Group by directory for better cache locality
    by_dir = {}
    for path in file_paths:
        dir_path = os.path.dirname(path)
        if dir_path not in by_dir:
            by_dir[dir_path] = []
        by_dir[dir_path].append(path)

    # Sort by size within each directory
    optimized = []
    for dir_path, paths in by_dir.items():
        file_sizes = [
            (p, os.path.getsize(p)) for p in paths if os.path.exists(p)
        ]
        file_sizes.sort(key=lambda x: x[1])  # Sort by size
        optimized.extend([p for p, _ in file_sizes])

    return optimized


