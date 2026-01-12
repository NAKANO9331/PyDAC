"""Performance benchmarking and comparison for PyDAC"""


import os

import subprocess

import time

import statistics

from pathlib import Path

from typing import Dict, List, Optional, Tuple, Any

from dataclasses import dataclass, field

import threading


# Optional dependency for resource monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    # Fallback: create dummy psutil
    class DummyProcess:
        def cpu_percent(self): return 0.0
        def memory_info(self): return type('obj', (object,), {'rss': 0})()

    class DummyPsutil:
        @staticmethod
        def Process():
            return DummyProcess()
    
    psutil = DummyPsutil()


@dataclass
class BenchmarkResult:
    """Benchmark result"""

    operation: str
    method: str  # "direct" or "pydac"
    duration: float
    cpu_percent: float
    memory_mb: float
    success: bool
    error: Optional[str] = None


@dataclass
class ComparisonResult:
    """Performance comparison result"""

    operation: str
    direct_avg: float
    pydac_avg: float
    overhead: float  # Percentage overhead
    overhead_abs: float  # Absolute overhead in seconds
    speedup: float  # pydac_avg / direct_avg
    cpu_diff: float  # CPU usage difference
    memory_diff: float  # Memory usage difference
    samples: int


class PerformanceBenchmark:
    """Performance benchmark for PyDAC vs direct dacpp usage"""

    def __init__(self, translator_path: Optional[str] = None):
        """
        Initialize benchmark

        Args:
            translator_path: Path to translator executable
        """
        self.translator_path = translator_path or self._find_translator()
        self.results: List[BenchmarkResult] = []

    def _find_translator(self) -> str:
        """Find translator executable"""

        # Check environment variable
        if "DACPP_TRANSLATOR" in os.environ:
            path = os.environ["DACPP_TRANSLATOR"]
            if os.path.exists(path):
                return path

        # Check common paths
        common_paths = [
            "./build/bin/translator/translator",
            "/usr/local/bin/translator",
        ]
        for path in common_paths:
            if os.path.exists(path):
                return path

        raise RuntimeError("Cannot find translator executable")

    def _measure_resources(self, func, *args, **kwargs) -> Tuple[float, float, float, Any]:
        """
        Measure function execution with resource monitoring

        Returns:
            (duration, cpu_percent, memory_mb, result)
        """
        if HAS_PSUTIL:
            process = psutil.Process()

            # Initial measurements
            cpu_before = process.cpu_percent()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB

            # Start monitoring thread
            cpu_samples = []
            memory_samples = []
            monitoring = True

            def monitor():
                while monitoring:
                    try:
                        cpu_samples.append(process.cpu_percent())
                        memory_samples.append(process.memory_info().rss / 1024 / 1024)
                    except:
                        pass
                    time.sleep(0.1)

            monitor_thread = threading.Thread(target=monitor, daemon=True)
            monitor_thread.start()
        else:
            cpu_samples = []
            memory_samples = []
            monitoring = False
            monitor_thread = None

        # Execute function
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        finally:
            duration = time.time() - start_time
            if HAS_PSUTIL and monitoring:
                monitoring = False
                if monitor_thread:
                    monitor_thread.join(timeout=1.0)

        # Calculate averages
        if HAS_PSUTIL and cpu_samples:
            cpu_avg = statistics.mean(cpu_samples)
            memory_peak = max(memory_samples) if memory_samples else 0.0
        else:
            cpu_avg = 0.0
            memory_peak = 0.0

        return duration, cpu_avg, memory_peak, (result, success, error)

    def benchmark_direct_translation(
        self,
        input_file: str,
        mode: str = "usm",
        iterations: int = 5
    ) -> List[BenchmarkResult]:
        """
        Benchmark direct dacpp translator usage

        Args:
            input_file: Input file path
            mode: Translation mode
            iterations: Number of iterations

        Returns:
            List of benchmark results
        """
        results = []

        for i in range(iterations):
            def translate():
                cmd = [
                    self.translator_path,
                    input_file,
                    f"--mode={mode}"
                ]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                return result.returncode == 0

            duration, cpu, memory, (_, success, error) = self._measure_resources(translate)

            result = BenchmarkResult(
                operation="translation",
                method="direct",
                duration=duration,
                cpu_percent=cpu,
                memory_mb=memory,
                success=success,
                error=error
            )
            results.append(result)
            self.results.append(result)

        return results

    def benchmark_pydac_translation(
        self,
        input_file: str,
        mode: str = "usm",
        iterations: int = 5
    ) -> List[BenchmarkResult]:
        """
        Benchmark PyDAC translation usage

        Args:
            input_file: Input file path
            mode: Translation mode
            iterations: Number of iterations

        Returns:
            List of benchmark results
        """
        from ..core.translator import PyDAC

        results = []
        translator = PyDAC(verbose=False)

        for i in range(iterations):
            def translate():
                try:
                    result = translator.translate(input_file, mode=mode)
                    return result.success
                except Exception:
                    return False

            duration, cpu, memory, (success, _, error) = self._measure_resources(translate)

            result = BenchmarkResult(
                operation="translation",
                method="pydac",
                duration=duration,
                cpu_percent=cpu,
                memory_mb=memory,
                success=success,
                error=error
            )
            results.append(result)
            self.results.append(result)

        return results

    def benchmark_direct_Compile(
        self,
        source_file: str,
        Compiler: str = "dpcpp",
        flags: Optional[List[str]] = None,
        iterations: int = 3
    ) -> List[BenchmarkResult]:
        """
        Benchmark direct compilation

        Args:
            source_file: Source file path
            Compiler: Compiler name
            flags: Compilation flags
            iterations: Number of iterations

        Returns:
            List of benchmark results
        """
        if flags is None:
            flags = ["-O2", "-std=c++17", "-fsycl"]

        results = []
        output_file = str(Path(source_file).with_suffix('.bin'))

        for i in range(iterations):
            def Compile():
                cmd = [Compiler, source_file, "-o", output_file] + flags
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                return result.returncode == 0

            duration, cpu, memory, (success, _, error) = self._measure_resources(Compile)

            result = BenchmarkResult(
                operation="compilation",
                method="direct",
                duration=duration,
                cpu_percent=cpu,
                memory_mb=memory,
                success=success,
                error=error
            )
            results.append(result)
            self.results.append(result)

        # Clean up
        if os.path.exists(output_file):
            os.remove(output_file)

        return results

    def benchmark_pydac_Compile(
        self,
        source_file: str,
        iterations: int = 3
    ) -> List[BenchmarkResult]:
        """
        Benchmark PyDAC compilation

        Args:
            source_file: Source file path
            iterations: Number of iterations

        Returns:
            List of benchmark results
        """
        from ..core.translator import PyDAC

        results = []
        translator = PyDAC(verbose=False)

        for i in range(iterations):
            def Compile():
                try:
                    result = translator.Compile(source_file)
                    return result.success
                except Exception:
                    return False

            duration, cpu, memory, (success, _, error) = self._measure_resources(Compile)

            result = BenchmarkResult(
                operation="compilation",
                method="pydac",
                duration=duration,
                cpu_percent=cpu,
                memory_mb=memory,
                success=success,
                error=error
            )
            results.append(result)
            self.results.append(result)

        return results

    def compare_translation(
        self,
        input_file: str,
        mode: str = "usm",
        iterations: int = 5
    ) -> ComparisonResult:
        """
        Compare direct vs PyDAC translation performance

        Args:
            input_file: Input file path
            mode: Translation mode
            iterations: Number of iterations

        Returns:
            ComparisonResult
        """
        print(f"Benchmarking translation: {input_file}")
        print(f" Direct dacpp...")
        direct_results = self.benchmark_direct_translation(input_file, mode, iterations)

        print(f" PyDAC...")
        pydac_results = self.benchmark_pydac_translation(input_file, mode, iterations)

        # Calculate statistics
        direct_durations = [r.duration for r in direct_results if r.success]
        pydac_durations = [r.duration for r in pydac_results if r.success]

        if not direct_durations or not pydac_durations:
            raise RuntimeError("Benchmark failed - no successful runs")

        direct_avg = statistics.mean(direct_durations)
        pydac_avg = statistics.mean(pydac_durations)

        overhead_abs = pydac_avg - direct_avg
        overhead = (overhead_abs / direct_avg) * 100 if direct_avg > 0 else 0
        speedup = direct_avg / pydac_avg if pydac_avg > 0 else 0

        direct_cpu_list = [r.cpu_percent for r in direct_results if r.success]
        pydac_cpu_list = [r.cpu_percent for r in pydac_results if r.success]
        direct_cpu = statistics.mean(direct_cpu_list) if direct_cpu_list else 0.0
        pydac_cpu = statistics.mean(pydac_cpu_list) if pydac_cpu_list else 0.0
        cpu_diff = pydac_cpu - direct_cpu

        direct_memory_list = [r.memory_mb for r in direct_results if r.success]
        pydac_memory_list = [r.memory_mb for r in pydac_results if r.success]
        direct_memory = statistics.mean(direct_memory_list) if direct_memory_list else 0.0
        pydac_memory = statistics.mean(pydac_memory_list) if pydac_memory_list else 0.0
        memory_diff = pydac_memory - direct_memory

        return ComparisonResult(
            operation="translation",
            direct_avg=direct_avg,
            pydac_avg=pydac_avg,
            overhead=overhead,
            overhead_abs=overhead_abs,
            speedup=speedup,
            cpu_diff=cpu_diff,
            memory_diff=memory_diff,
            samples=iterations
        )

    def compare_compilation(
        self,
        source_file: str,
        Compiler: str = "dpcpp",
        iterations: int = 3
    ) -> ComparisonResult:
        """
        Compare direct vs PyDAC compilation performance

        Args:
            source_file: Source file path
            Compiler: Compiler name
            iterations: Number of iterations

        Returns:
            ComparisonResult
        """
        print(f"Benchmarking compilation: {source_file}")
        print(f" Direct {Compiler}...")
        direct_results = self.benchmark_direct_Compile(source_file, Compiler, iterations=iterations)

        print(f" PyDAC...")
        pydac_results = self.benchmark_pydac_Compile(source_file, iterations=iterations)

        # Calculate statistics
        direct_durations = [r.duration for r in direct_results if r.success]
        pydac_durations = [r.duration for r in pydac_results if r.success]

        if not direct_durations or not pydac_durations:
            raise RuntimeError("Benchmark failed - no successful runs")

        direct_avg = statistics.mean(direct_durations)
        pydac_avg = statistics.mean(pydac_durations)

        overhead_abs = pydac_avg - direct_avg
        overhead = (overhead_abs / direct_avg) * 100 if direct_avg > 0 else 0
        speedup = direct_avg / pydac_avg if pydac_avg > 0 else 0

        direct_cpu_list = [r.cpu_percent for r in direct_results if r.success]
        pydac_cpu_list = [r.cpu_percent for r in pydac_results if r.success]
        direct_cpu = statistics.mean(direct_cpu_list) if direct_cpu_list else 0.0
        pydac_cpu = statistics.mean(pydac_cpu_list) if pydac_cpu_list else 0.0
        cpu_diff = pydac_cpu - direct_cpu

        direct_memory_list = [r.memory_mb for r in direct_results if r.success]
        pydac_memory_list = [r.memory_mb for r in pydac_results if r.success]
        direct_memory = statistics.mean(direct_memory_list) if direct_memory_list else 0.0
        pydac_memory = statistics.mean(pydac_memory_list) if pydac_memory_list else 0.0
        memory_diff = pydac_memory - direct_memory

        return ComparisonResult(
            operation="compilation",
            direct_avg=direct_avg,
            pydac_avg=pydac_avg,
            overhead=overhead,
            overhead_abs=overhead_abs,
            speedup=speedup,
            cpu_diff=cpu_diff,
            memory_diff=memory_diff,
            samples=iterations
        )

    def print_comparison(self, comparison: ComparisonResult):
        """Print comparison results"""

        print("\n" + "=" * 60)
        print(f"Performance Comparison: {comparison.operation}")
        print("=" * 60)
        print(f"Method | Avg Time (s) | CPU (%) | Memory (MB)")
        print("-" * 60)
        print(f"Direct | {comparison.direct_avg:11.4f} | {comparison.cpu_diff:7.2f} | {comparison.memory_diff:10.2f}")
        print(f"PyDAC | {comparison.pydac_avg:11.4f} | {comparison.cpu_diff:7.2f} | {comparison.memory_diff:10.2f}")
        print("-" * 60)
        print(f"Overhead | {comparison.overhead_abs:11.4f} ({comparison.overhead:+.2f}%)")
        print(f"Speedup | {comparison.speedup:.4f}x")
        print(f"CPU Difference | {comparison.cpu_diff:+.2f}%")
        print(f"Memory Diff | {comparison.memory_diff:+.2f} MB")
        print(f"Samples | {comparison.samples}")
        print("=" * 60)

    def generate_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate benchmark report

        Args:
            output_file: Output file path (None for auto-generate)

        Returns:
            Report content
        """
        import json

        report = {
            "summary": {
                "total_benchmarks": len(self.results),
                "operations": {}
            },
            "results": []
        }

        # Group by operation and method
        for result in self.results:
            op = result.operation
            method = result.method

            if op not in report["summary"]["operations"]:
                report["summary"]["operations"][op] = {}
            if method not in report["summary"]["operations"][op]:
                report["summary"]["operations"][op][method] = {
                    "count": 0,
                    "avg_duration": 0.0,
                    "avg_cpu": 0.0,
                    "avg_memory": 0.0
                }

            stats = report["summary"]["operations"][op][method]
            stats["count"] += 1
            stats["avg_duration"] += result.duration
            stats["avg_cpu"] += result.cpu_percent
            stats["avg_memory"] += result.memory_mb

            report["results"].append({
                "operation": result.operation,
                "method": result.method,
                "duration": result.duration,
                "cpu_percent": result.cpu_percent,
                "memory_mb": result.memory_mb,
                "success": result.success,
                "error": result.error
            })

        # Calculate averages
        for op_stats in report["summary"]["operations"].values():
            for method_stats in op_stats.values():
                count = method_stats["count"]
                if count > 0:
                    method_stats["avg_duration"] /= count
                    method_stats["avg_cpu"] /= count
                    method_stats["avg_memory"] /= count

        report_json = json.dumps(report, indent=2)

        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_json)

        return report_json

