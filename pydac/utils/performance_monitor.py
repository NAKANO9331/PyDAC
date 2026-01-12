"""Advanced performance monitoring for PyDAC"""


import time

import statistics

from typing import Dict, List, Optional, Any

from dataclasses import dataclass, field, asdict

from collections import defaultdict

from datetime import datetime

from pathlib import Path

import json


try:
 import psutil
 HAS_PSUTIL = True
except ImportError:
 HAS_PSUTIL = False


@dataclass
class OperationMetrics:
    """Metrics for a single operation"""

    operation_name: str
    start_time: float
    end_time: float
    duration: float
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    peak_memory_mb: float = 0.0
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceReport:
    """Performance report"""

    total_operations: int
    total_duration: float
    operations: List[Dict[str, Any]]
    statistics: Dict[str, Any]
    timestamp: str


class PerformanceMonitor:

    """Advanced performance monitor for PyDAC operations"""


    def __init__(self, enabled: bool = True):
        """
        Initialize performance monitor

        Args:
            enabled: Whether monitoring is enabled
        """
        self.enabled = enabled
        self.operations: List[OperationMetrics] = []
        self.current_operation: Optional[OperationMetrics] = None
        self._operation_stats: Dict[str, List[float]] = defaultdict(list)

        if HAS_PSUTIL:
            self.process = psutil.Process()
        else:
            self.process = None

    def start_operation(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None):

        """

        Start monitoring an operation

        Args:
        operation_name: Name of the operation
        metadata: Optional metadata
        """
        if not self.enabled:
            return

        if self.process:
            cpu_before = self.process.cpu_percent()
            memory_before = self.process.memory_info().rss / 1024 / 1024
        else:
            cpu_before = 0.0
            memory_before = 0.0

            self.current_operation = OperationMetrics(
            operation_name=operation_name,
            start_time=time.time(),
            end_time=0.0,
            duration=0.0,
            cpu_percent=cpu_before,
            memory_mb=memory_before,
            peak_memory_mb=memory_before,
            metadata=metadata or {}
            )

    def end_operation(self, success: bool = True, error: Optional[str] = None):
        """
        End monitoring current operation

        Args:
            success: Whether operation succeeded
            error: Error message if failed
        """
        if not self.enabled or not self.current_operation:
            return

        end_time = time.time()
        self.current_operation.end_time = end_time
        self.current_operation.duration = end_time - self.current_operation.start_time
        self.current_operation.success = success
        self.current_operation.error = error

        if self.process:
            cpu_after = self.process.cpu_percent()
            memory_after = self.process.memory_info().rss / 1024 / 1024
            self.current_operation.cpu_percent = (self.current_operation.cpu_percent + cpu_after) / 2
            self.current_operation.memory_mb = memory_after
            self.current_operation.peak_memory_mb = max(
                self.current_operation.memory_mb,
                memory_after
            )

        self.operations.append(self.current_operation)
        self._operation_stats[self.current_operation.operation_name].append(
            self.current_operation.duration
        )
        self.current_operation = None

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get performance statistics

        Returns:
            Dictionary with performance statistics
        """
        if not self.operations:
            return {
                "total_operations": 0,
                "total_duration": 0.0,
                "operations": {}
            }

        total_duration = sum(op.duration for op in self.operations)

        # Statistics by operation type
        op_stats = {}
        for op_name, durations in self._operation_stats.items():
            if durations:
                op_stats[op_name] = {
                    "count": len(durations),
                    "total_time": sum(durations),
                    "avg_time": statistics.mean(durations),
                    "min_time": min(durations),
                    "max_time": max(durations),
                    "median_time": statistics.median(durations),
                    "stdev_time": statistics.stdev(durations) if len(durations) > 1 else 0.0
                }

        # Overall statistics
        all_durations = [op.duration for op in self.operations]
        overall_stats = {
            "total_operations": len(self.operations),
            "total_duration": total_duration,
            "avg_duration": statistics.mean(all_durations) if all_durations else 0.0,
            "min_duration": min(all_durations) if all_durations else 0.0,
            "max_duration": max(all_durations) if all_durations else 0.0,
            "median_duration": statistics.median(all_durations) if all_durations else 0.0,
            "success_rate": sum(1 for op in self.operations if op.success) / len(self.operations) if self.operations else 0.0
        }

        # Resource statistics
        if HAS_PSUTIL and self.operations:
            cpu_values = [op.cpu_percent for op in self.operations if op.cpu_percent > 0]
            memory_values = [op.memory_mb for op in self.operations if op.memory_mb > 0]
            peak_memory_values = [op.peak_memory_mb for op in self.operations if op.peak_memory_mb > 0]

            resource_stats = {
                "avg_cpu_percent": statistics.mean(cpu_values) if cpu_values else 0.0,
                "avg_memory_mb": statistics.mean(memory_values) if memory_values else 0.0,
                "peak_memory_mb": max(peak_memory_values) if peak_memory_values else 0.0
            }
        else:
            resource_stats = {
                "avg_cpu_percent": 0.0,
                "avg_memory_mb": 0.0,
                "peak_memory_mb": 0.0
            }

        return {
            "overall": overall_stats,
            "by_operation": op_stats,
            "resources": resource_stats
        }

    def _convert_paths_to_strings(self, obj: Any) -> Any:
        """Recursively convert Path objects to strings for JSON serialization"""

        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_paths_to_strings(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_paths_to_strings(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_paths_to_strings(item) for item in obj)
        else:
            return obj

    def generate_report(self, output_file: Optional[str] = None) -> PerformanceReport:
        """
        Generate performance report

        Args:
            output_file: Optional output file path

        Returns:
            PerformanceReport object
        """
        stats = self.get_statistics()

        report = PerformanceReport(
            total_operations=stats["overall"]["total_operations"],
            total_duration=stats["overall"]["total_duration"],
            operations=[asdict(op) for op in self.operations],
            statistics=stats,
            timestamp=datetime.now().isoformat()
        )

        if output_file:
            report_dict = asdict(report)
            # Convert Path objects to strings for JSON serialization
            report_dict = self._convert_paths_to_strings(report_dict)
            with open(output_file, 'w') as f:
                json.dump(report_dict, f, indent=2)

        return report

    def print_summary(self):

        """Print performance summary"""

        stats = self.get_statistics()

        print("\n" + "=" * 60)
        print("Performance Summary")
        print("=" * 60)

        overall = stats["overall"]
        print(f"Total Operations: {overall['total_operations']}")
        print(f"Total Duration: {overall['total_duration']:.4f}s")
        print(f"Average Duration: {overall['avg_duration']:.4f}s")
        print(f"Success Rate: {overall['success_rate']*100:.1f}%")

        if stats["resources"]["avg_cpu_percent"] > 0:
            print(f"\nResource Usage:")
            print(f" Average CPU: {stats['resources']['avg_cpu_percent']:.2f}%")
            print(f" Average Memory: {stats['resources']['avg_memory_mb']:.2f} MB")
            print(f" Peak Memory: {stats['resources']['peak_memory_mb']:.2f} MB")

        if stats["by_operation"]:
            print(f"\nBy Operation Type:")
        for op_name, op_stats in stats["by_operation"].items():
            print(f" {op_name}:")
            print(f" Count: {op_stats['count']}")
            print(f" Avg: {op_stats['avg_time']:.4f}s")
            print(f" Min: {op_stats['min_time']:.4f}s")
            print(f" Max: {op_stats['max_time']:.4f}s")

            print("=" * 60)

    def reset(self):
        """Reset monitor"""

        self.operations.clear()
        self._operation_stats.clear()
        self.current_operation = None


class PerformanceContext:
    """Context manager for performance monitoring"""

    def __init__(self, monitor: PerformanceMonitor, operation_name: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize performance context

        Args:
            monitor: PerformanceMonitor instance
            operation_name: Name of operation
            metadata: Optional metadata
        """
        self.monitor = monitor
        self.operation_name = operation_name
        self.metadata = metadata
        self.success = True
        self.error = None

    def __enter__(self):
        """Context manager entry"""

        self.monitor.start_operation(self.operation_name, self.metadata)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""

        if exc_type:
            self.success = False
            self.error = str(exc_val)
        self.monitor.end_operation(self.success, self.error)
        return False  # Don't suppress exceptions


