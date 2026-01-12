"""Code analyzer for PyDAC"""


import re

from typing import List, Dict, Optional, Any, TYPE_CHECKING

from dataclasses import dataclass

from pathlib import Path


if TYPE_CHECKING:
 from ..core.optimizations import ResultCache


@dataclass
class ShellInfo:
    """Shell function information"""

    name: str
    parameters: List[str]
    splits: List[Dict[str, int]]  # [{"name": "sp1", "size": 3, "stride": 1}, ...]
    indices: List[str]
    bindings: Dict[str, str]  # {"sp1": "idx1", ...}
    slices: List[Dict[str, List[str]]]  # [{"tensor": "matIn", "ops": ["sp1", "sp2"]}, ...]


@dataclass
class CalcInfo:
    """Calc function information"""

    name: str
    return_type: str
    parameters: List[Dict[str, str]]  # [{"name": "mat", "type": "double*"}, ...]
    body_lines: List[str]


@dataclass
class ExpressionInfo:
    """Data association expression information"""

    shell_name: str
    calc_name: str
    line_number: int


@dataclass
class CodeAnalyze:
    """Complete code analysis result"""

    shells: List[ShellInfo]
    calcs: List[CalcInfo]
    expressions: List[ExpressionInfo]
    dac_for_loops: List[Dict[str, Any]]


class CodeAnalyzer:
    """Code analyzer for DAC code"""

    def __init__(self, cache: Optional[Any] = None):
        """
        Initialize code analyzer

        Args:
            cache: Optional ResultCache instance for caching analysis results
        """
        self.cache = cache

    def analyze(self, input_file: str) -> CodeAnalyze:
        """
        Analyze DAC code structure

        Args:
            input_file: Input C++ file path

        Returns:
            CodeAnalyze: Analyze result
        """
        # Check cache first
        if self.cache:
            cache_params = {}
            cached_result = self.cache.get(input_file, "analysis", cache_params)
            if cached_result:
                # Reconstruct CodeAnalyze from cached data
                shells = [ShellInfo(**s) for s in cached_result.get("shells", [])]
                calcs = [CalcInfo(**c) for c in cached_result.get("calcs", [])]
                expressions = [ExpressionInfo(**e) for e in cached_result.get("expressions", [])]
                return CodeAnalyze(
                    shells=shells,
                    calcs=calcs,
                    expressions=expressions,
                    dac_for_loops=cached_result.get("dac_for_loops", [])
                )

        with open(input_file, 'r') as f:
            content = f.read()

        shells = self.extract_shells(content)
        calcs = self.extract_calcs(content)
        expressions = self.extract_expressions(content)
        dac_for_loops = self.extract_dac_for_loops(content)

        analysis = CodeAnalyze(
            shells=shells,
            calcs=calcs,
            expressions=expressions,
            dac_for_loops=dac_for_loops
        )

        # Cache result
        if self.cache:
            cache_params = {}
            result_dict = {
                "shells": [
                    {
                        "name": s.name,
                        "parameters": s.parameters,
                        "splits": s.splits,
                        "indices": s.indices,
                        "bindings": s.bindings,
                        "slices": s.slices
                    }
                    for s in shells
                ],
                "calcs": [
                    {
                        "name": c.name,
                        "return_type": c.return_type,
                        "parameters": c.parameters,
                        "body_lines": c.body_lines
                    }
                    for c in calcs
                ],
                "expressions": [
                    {
                        "shell_name": e.shell_name,
                        "calc_name": e.calc_name,
                        "line_number": e.line_number
                    }
                    for e in expressions
                ],
                "dac_for_loops": dac_for_loops
            }
            self.cache.put(input_file, "analysis", cache_params, result_dict)

        return analysis

    def extract_shells(self, code: str) -> List[ShellInfo]:
        """Extract all shell functions"""

        shells = []

        # Pattern: shell dacpp::list functionName(...) { ... }
        pattern = r'shell\s+dacpp::list\s+(\w+)\s*\(([^)]*)\)\s*\{([^}]+)\}'

        for match in re.finditer(pattern, code, re.MULTILINE | re.DOTALL):
            name = match.group(1)
            params_str = match.group(2)
            body = match.group(3)

            # Parse parameters
            parameters = [p.strip() for p in params_str.split(',') if p.strip()]

            # Extract splits
            splits = []
            split_pattern = r'dacpp::split\s+(\w+)\s*\((\d+),\s*(\d+)\)'
            for sp_match in re.finditer(split_pattern, body):
                splits.append({
                    "name": sp_match.group(1),
                    "size": int(sp_match.group(2)),
                    "stride": int(sp_match.group(3))
                })

            # Extract indices
            indices = []
            index_pattern = r'dacpp::index\s+(\w+)'
            for idx_match in re.finditer(index_pattern, body):
                indices.append(idx_match.group(1))

            # Extract bindings
            bindings = {}
            binding_pattern = r'binding\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)'
            for bind_match in re.finditer(binding_pattern, body):
                bindings[bind_match.group(1)] = bind_match.group(2)

            # Extract slices (simplified)
            slices = []
            slice_pattern = r'(\w+)\[([^\]]+)\]'
            for slice_match in re.finditer(slice_pattern, body):
                tensor_name = slice_match.group(1)
                ops_str = slice_match.group(2)
                ops = [op.strip() for op in ops_str.split(',')]
                slices.append({"tensor": tensor_name, "ops": ops})

            shells.append(ShellInfo(
                name=name,
                parameters=parameters,
                splits=splits,
                indices=indices,
                bindings=bindings,
                slices=slices
            ))

        return shells

    def extract_calcs(self, code: str) -> List[CalcInfo]:
        """Extract all calc functions"""

        calcs = []

        # Pattern: calc returnType functionName(...) { ... }
        pattern = r'calc\s+(\w+)\s+(\w+)\s*\(([^)]*)\)\s*\{([^}]+)\}'

        for match in re.finditer(pattern, code, re.MULTILINE | re.DOTALL):
            return_type = match.group(1)
            name = match.group(2)
            params_str = match.group(3)
            body = match.group(4)

            # Parse parameters
            parameters = []
            for param in params_str.split(','):
                param = param.strip()
                if param:
                    # Simple parsing: "type name" or "type* name"
                    parts = param.split()
                    if len(parts) >= 2:
                        param_type = ' '.join(parts[:-1])
                        param_name = parts[-1]
                        parameters.append({"name": param_name, "type": param_type})

            # Parse body lines
            body_lines = [line.strip() for line in body.split('\n') if line.strip()]

            calcs.append(CalcInfo(
                name=name,
                return_type=return_type,
                parameters=parameters,
                body_lines=body_lines
            ))

        return calcs

    def extract_expressions(self, code: str) -> List[ExpressionInfo]:
        """Extract all data association expressions"""

        expressions = []

        # Pattern: shellName(...) <-> calcName;
        pattern = r'(\w+)\s*\([^)]*\)\s*<->\s*(\w+)\s*;'

        lines = code.split('\n')
        for line_num, line in enumerate(lines, 1):
            match = re.search(pattern, line)
            if match:
                expressions.append(ExpressionInfo(
                    shell_name=match.group(1),
                    calc_name=match.group(2),
                    line_number=line_num
                ))

        return expressions

    def extract_dac_for_loops(self, code: str) -> List[Dict[str, Any]]:
        """Extract dac_for loops"""

        loops = []

        # Pattern: dacpp::dac_for(iterations, [&](int step) { ... })
        pattern = r'dacpp::dac_for\s*\(\s*(\w+)\s*,\s*\[&\]\s*\([^)]*\)\s*\{([^}]+)\}\s*\)'

        for match in re.finditer(pattern, code, re.MULTILINE | re.DOTALL):
            iterations = match.group(1)
            body = match.group(2)

            # Check if contains expression
            Has_expression = '<->' in body

            loops.append({
                "iterations": iterations,
                "body": body.strip(),
                "Has_expression": Has_expression
            })

        return loops

    def get_code_statistics(self, input_file: str) -> dict:
        """
        Get code statistics

        Args:
            input_file: Input file path

        Returns:
            Dictionary with statistics
        """
        with open(input_file, 'r') as f:
            content = f.read()

        analysis = self.analyze(input_file)

        stats = {
            "total_lines": len(content.split('\n')),
            "shell_count": len(analysis.shells),
            "calc_count": len(analysis.calcs),
            "expression_count": len(analysis.expressions),
            "dac_for_count": len(analysis.dac_for_loops),
            "Has_main": "int main()" in content,
            "includes": self._extract_includes(content),
            "namespaces": self._extract_namespaces(content)
        }

        return stats

    def _extract_includes(self, code: str) -> List[str]:
        """Extract include statements"""

        includes = []
        pattern = r'#include\s+[<"]([^>"]+)[>"]'
        for match in re.finditer(pattern, code):
            includes.append(match.group(1))
        return includes

    def _extract_namespaces(self, code: str) -> List[str]:
        """Extract namespace declarations"""

        namespaces = []
        pattern = r'namespace\s+(\w+)'
        for match in re.finditer(pattern, code):
            namespaces.append(match.group(1))
        return namespaces

