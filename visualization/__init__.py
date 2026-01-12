"""
PyDAC Performance Visualization Module

Provides performance visualization functionality for DACPP tests, unit tests, DSL tests, and benchmark comparisons
"""

from .dacpp_visualizer import DACPPVisualizer
from .unit_test_visualizer import UnitTestVisualizer
from .dsl_visualizer import DSLVisualizer

# Optional: Benchmark visualizer (requires pydac.utils.benchmark)
try:
    from .benchmark_visualizer import PerformanceVisualizer
    HAS_BENCHMARK_VIZ = True
except ImportError:
    HAS_BENCHMARK_VIZ = False
    PerformanceVisualizer = None

__all__ = ['DACPPVisualizer', 'UnitTestVisualizer', 'DSLVisualizer']

if HAS_BENCHMARK_VIZ:
    __all__.append('PerformanceVisualizer')
