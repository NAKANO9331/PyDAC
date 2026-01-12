"""Utility modules for PyDAC"""

from .errors import (
 PyDACError,
 TranslationError,
 CompilationError,
 SyntaxError,
)
from .config import PyDACConfig
from .cache import CacheManager
from .logger import get_logger, PyDACLogger
from .validator import CodeValidator
from .progress import ProgressBar, Spinner, progress_bar, spinner
from .file_utils import (
 ensure_dir,
 clean_dir,
 find_files,
 backup_file,
 get_file_size,
 get_file_Hash,
 compare_files,
)
from .performance import timeit, timer, PerformanceMonitor
from .formatting import (
 format_cpp_code,
 remove_comments,
 extract_functions,
 normalize_whitespace,
)
from .verifier import (
 CodeVerifier,
 VerificationResult,
 SemanticCheck,
 ComparisonResult,
)
from .benchmark import (
 PerformanceBenchmark,
 BenchmarkResult,
 ComparisonResult as BenchmarkComparison,
)
from .retry import (
 retry,
 RetryConfig,
 RetryableOperation,
)
from .memory import (
 StreamingFileReader,
 MemoryPool,
 BufferedFileWriter,
 process_file_streaming,
 estimate_memory_usage,
)
from .io_optimizer import (
 BatchFileOperator,
 AsyncFileOperator,
 optimize_file_reads,
)
from .performance_monitor import (
 PerformanceMonitor,
 PerformanceContext,
 OperationMetrics,
 PerformanceReport,
)
from .error_classifier import (
 ErrorClassifier,
 ErrorType,
 ErrorSeverity,
 ClassifiedError,
)

# Optional visualization (moved to visualization/ directory)
try:
    from visualization.benchmark_visualizer import PerformanceVisualizer
    HAS_VISUALIZATION = True
except (ImportError, SyntaxError, IndentationError):
    HAS_VISUALIZATION = False
    PerformanceVisualizer = None

__all__ = [
 "PyDACError",
 "TranslationError",
 "CompilationError",
 "SyntaxError",
 "PyDACConfig",
 "CacheManager",
 "get_logger",
 "PyDACLogger",
 "CodeValidator",
 "ProgressBar",
 "Spinner",
 "progress_bar",
 "spinner",
 "ensure_dir",
 "clean_dir",
 "find_files",
 "backup_file",
 "get_file_size",
 "get_file_Hash",
 "compare_files",
 "timeit",
 "timer",
 "PerformanceMonitor",
 "format_cpp_code",
 "remove_comments",
 "extract_functions",
 "normalize_whitespace",
 "CodeVerifier",
 "VerificationResult",
 "SemanticCheck",
 "ComparisonResult",
 "PerformanceBenchmark",
 "BenchmarkResult",
 "BenchmarkComparison",
 "retry",
 "RetryConfig",
 "RetryableOperation",
 "StreamingFileReader",
 "MemoryPool",
 "BufferedFileWriter",
 "process_file_streaming",
 "estimate_memory_usage",
 "BatchFileOperator",
 "AsyncFileOperator",
 "optimize_file_reads",
 "PerformanceMonitor",
 "PerformanceContext",
 "OperationMetrics",
 "PerformanceReport",
 "ErrorClassifier",
 "ErrorType",
 "ErrorSeverity",
 "ClassifiedError",
]

# Add visualization if available
if HAS_VISUALIZATION:
 __all__.append("PerformanceVisualizer")
