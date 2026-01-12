"""Code verification utility for PyDAC"""


import re

from typing import List, Tuple, Dict, Any, Optional

from pathlib import Path

from dataclasses import dataclass


@dataclass
class VerificationResult:
    """Result of code verification"""

    is_valid: bool
    errors: List[str]
    warnings: List[str]
    details: Dict[str, Any]


@dataclass
class SemanticCheck:
    """Semantic check result"""

    is_valid: bool
    errors: List[str]
    shell_calc_matches: List[Dict[str, Any]]


@dataclass
class ComparisonResult:
    """Result of code comparison"""

    is_similar: bool
    similarity_score: float
    differences: List[str]
    details: Dict[str, Any]


class CodeVerifier:
    """Code verifier for DAC code"""

    def __init__(self):
        """Initialize verifier"""

        pass

    def verify_syntax(self, code: str) -> VerificationResult:
        """
        Verify syntax correctness

        Args:
            code: C++ code string

        Returns:
            VerificationResult: Verification result
        """
        from .validator import CodeValidator

        validator = CodeValidator()
        is_valid, errors = validator.validate_syntax(code)

        warnings = []
        details = {}

        # Additional checks
        if 'dac_for' in code:
            dac_for_count = len(re.findall(r'dacpp::dac_for', code))
            details['dac_for_count'] = dac_for_count

        # Check for common issues
        if 'namespace dacpp' in code and 'typedef std::vector<std::any> list' not in code:
            warnings.append("dacpp namespace found but list typedef missing")

        return VerificationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            details=details
        )

    def verify_semantics(self, input_file: str) -> SemanticCheck:
        """
        Verify semantic correctness

        Args:
            input_file: Input file path

        Returns:
            SemanticCheck: Semantic check result
        """
        if not Path(input_file).exists():
            return SemanticCheck(
                is_valid=False,
                errors=[f"File not found: {input_file}"],
                shell_calc_matches=[]
            )

        with open(input_file, 'r') as f:
            code = f.read()

        from .validator import CodeValidator
        validator = CodeValidator()
        is_valid, errors = validator.verify_semantics(code)

        # Extract shell-calc matches
        shell_calc_matches = []
        expr_pattern = r'(\w+)\s*\([^)]*\)\s*<->\s*(\w+)\s*;'
        for match in re.finditer(expr_pattern, code):
            shell_calc_matches.append({
                "shell": match.group(1),
                "calc": match.group(2),
                "line": code[:match.start()].count('\n') + 1
            })

        return SemanticCheck(
            is_valid=is_valid,
            errors=errors,
            shell_calc_matches=shell_calc_matches
        )

    def compare_results(
        self,
        original_file: str,
        translated_file: str,
        test_cases: Optional[List[Dict[str, Any]]] = None
    ) -> ComparisonResult:
        """
        Compare original and translated code

        Args:
            original_file: Original DAC file path
            translated_file: Translated SYCL file path
            test_cases: Optional test cases for validation

        Returns:
            ComparisonResult: Comparison result
        """
        from .validator import CodeValidator

        validator = CodeValidator()
        is_similar, comparison_info = validator.compare_results(
            original_file,
            translated_file
        )

        if "error" in comparison_info:
            return ComparisonResult(
                is_similar=False,
                similarity_score=0.0,
                differences=[comparison_info["error"]],
                details=comparison_info
            )

        # Calculate similarity score
        score = 0.0
        differences = []

        if comparison_info.get("Has_sycl_code"):
            score += 0.3
        else:
            differences.append("Missing SYCL code")

        if comparison_info.get("Has_kernel"):
            score += 0.3
        else:
            differences.append("Missing kernel code")

        if comparison_info.get("shells_preserved"):
            score += 0.2
        else:
            differences.append("Some shell functions not preserved")

        if comparison_info.get("calcs_preserved"):
            score += 0.2
        else:
            differences.append("Some calc functions not preserved")

        # Check mode-specific features
        if comparison_info.get("Has_buffer") or comparison_info.get("Has_usm"):
            score += 0.1

        details = comparison_info.copy()
        details["similarity_score"] = score

        return ComparisonResult(
            is_similar=is_similar,
            similarity_score=score,
            differences=differences,
            details=details
        )


