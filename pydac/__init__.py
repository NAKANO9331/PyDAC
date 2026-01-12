"""
PyDAC - Python interface for DACPP source-to-source translator

A Python interface for converting DAC (Data Associated Computing) C++ code
to various parallel computing backends (SYCL USM, SYCL Buffer, MPI, etc.)
"""

__version__ = "0.1.0"
__author__ = "PyDAC Team"

from .core.translator import PyDAC
from .tensor.tensor import Tensor
from .dsl import Shell, Calc, Expression, ExpressionBuilder
from .generator import CodeGenerator
from .analyzer import CodeAnalyzer, CodeAnalyze

__all__ = [
    "PyDAC",
    "Tensor",
    "Shell",
    "Calc",
    "Expression",
    "ExpressionBuilder",
    "CodeGenerator",
    "CodeAnalyzer",
    "CodeAnalyze",
]
