"""Performance utilities for PyDAC"""


import time

from functools import wraps

from typing import Callable, Any

from contextlib import contextmanager


def timeit(func: Callable) -> Callable:
    """Decorator to measure function execution time"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        print(f"{func.__name__} took {duration:.2f} seconds")
        return result
    return wrapper


@contextmanager
def timer(description: str = "Operation"):
    """Context manager for timing operations"""

    start = time.time()
    try:
        yield
    finally:
        duration = time.time() - start
        print(f"{description} took {duration:.2f} seconds")


class PerformanceMonitor:
    """Performance monitor for tracking operations"""

    def __init__(self):
        """Initialize performance monitor"""

        self.operations = {}

    def start(self, operation: str):
        """Start timing an operation"""

        self.operations[operation] = {"start": time.time()}

    def stop(self, operation: str):
        """Stop timing an operation"""

        if operation in self.operations:
            self.operations[operation]["duration"] = (
                time.time() - self.operations[operation]["start"]
            )

    def get_stats(self) -> dict:
        """Get performance statistics"""

        stats = {
            "operations": {},
            "total_time": 0.0
        }

        for op, data in self.operations.items():
            if "duration" in data:
                stats["operations"][op] = data["duration"]
                stats["total_time"] += data["duration"]

        return stats

    def print_stats(self):
        """Print performance statistics"""

        stats = self.get_stats()
        print("\nPerformance Statistics:")
        print("-" * 50)
        for op, duration in stats["operations"].items():
            print(f" {op}: {duration:.2f}s")
        print(f" Total: {stats['total_time']:.2f}s")
        print("-" * 50)

