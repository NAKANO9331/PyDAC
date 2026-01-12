"""Memory optimization utilities for PyDAC"""


import io

from typing import Iterator, Optional, BinaryIO

from pathlib import Path


class StreamingFileReader:

    """Streaming file reader for large files"""

    def __init__(self, file_path: str, chunk_size: int = 8192):
        """
        Initialize streaming file reader

        Args:
            file_path: Path to file
            chunk_size: Size of chunks to read (bytes)
        """
        self.file_path = Path(file_path)
        self.chunk_size = chunk_size
        self._file: Optional[BinaryIO] = None

    def __enter__(self):
        """Context manager entry"""

        self._file = open(self.file_path, 'rb')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""

        if self._file:
            self._file.close()

    def read_chunks(self) -> Iterator[bytes]:
        """
        Read file in chunks

        Yields:
            Chunks of file content
        """
        if not self._file:
            raise RuntimeError("File not opened. Use as context manager.")

        while True:
            chunk = self._file.read(self.chunk_size)
            if not chunk:
                break
            yield chunk

    def read_lines(self) -> Iterator[str]:
        """
        Read file line by line

        Yields:
            Lines of file content
        """
        if not self._file:
            raise RuntimeError("File not opened. Use as context manager.")

        # Reset to beginning
        self._file.seek(0)

        # Use text mode for line reading
        with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                yield line.rstrip('\n\r')


class MemoryPool:
    """Simple memory pool for reducing allocation overhead"""

    def __init__(self, chunk_size: int = 4096, pool_size: int = 10):
        """
        Initialize memory pool

        Args:
            chunk_size: Size of each chunk (bytes)
            pool_size: Number of chunks in pool
        """
        self.chunk_size = chunk_size
        self.pool_size = pool_size
        self._pool: list = []
        self._allocated = 0

    def get_chunk(self) -> bytearray:
        """
        Get a chunk from pool or allocate new one

        Returns:
            Bytearray chunk
        """
        if self._pool:
            return self._pool.pop()

        self._allocated += 1
        return bytearray(self.chunk_size)

    def return_chunk(self, chunk: bytearray):
        """
        Return chunk to pool

        Args:
            chunk: Chunk to return
        """
        if len(self._pool) < self.pool_size:
            chunk[:] = b'\x00' * len(chunk)  # Clear chunk
            self._pool.append(chunk)

    def clear(self):
        """Clear pool"""

        self._pool.clear()
        self._allocated = 0


class BufferedFileWriter:
    """Buffered file writer for efficient I/O"""

    def __init__(self, file_path: str, buffer_size: int = 8192):
        """
        Initialize buffered file writer

        Args:
            file_path: Path to output file
            buffer_size: Buffer size (bytes)
        """
        self.file_path = Path(file_path)
        self.buffer_size = buffer_size
        self._buffer = io.BytesIO()
        self._file: Optional[BinaryIO] = None

    def __enter__(self):
        """Context manager entry"""

        self._file = open(self.file_path, 'wb')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""

        self.flush()
        if self._file:
            self._file.close()

    def write(self, data: bytes):
        """
        Write data to buffer

        Args:
            data: Data to write
        """
        self._buffer.write(data)
        if self._buffer.tell() >= self.buffer_size:
            self.flush()

    def flush(self):
        """Flush buffer to file"""

        if self._file and self._buffer.tell() > 0:
            self._buffer.seek(0)
            self._file.write(self._buffer.read())
            self._buffer.seek(0)
            self._buffer.truncate(0)


def process_file_streaming(
    file_path: str,
    processor: callable,
    chunk_size: int = 8192
) -> Iterator:
    """
    Process file in streaming mode

    Args:
        file_path: Path to file
        processor: Function to process chunks
        chunk_size: Size of chunks

    Yields:
        Processed results
    """
    with StreamingFileReader(file_path, chunk_size) as reader:
        for chunk in reader.read_chunks():
            yield processor(chunk)


def estimate_memory_usage(file_path: str) -> dict:
    """
    Estimate memory usage for file operations

    Args:
        file_path: Path to file

    Returns:
        Dictionary with memory estimates
    """
    path = Path(file_path)
    if not path.exists():
        return {"error": "File not found"}

    file_size = path.stat().st_size

    return {
        "file_size": file_size,
        "file_size_mb": file_size / (1024 * 1024),
        "recommended_chunk_size": min(8192, file_size // 10) if file_size > 0 else 8192,
        "use_streaming": file_size > 1024 * 1024,  # > 1MB
        "estimated_memory": file_size * 2  # Rough estimate (file + processing)
    }


