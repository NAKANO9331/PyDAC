"""Error classification and recovery for PyDAC"""


import re

from typing import Dict, List, Optional, Tuple, Any

from dataclasses import dataclass

from enum import Enum


class ErrorType(Enum):
    """Error type enumeration"""

    TRANSLATION_ERROR = "translation_error"
    COMPILATION_ERROR = "compilation_error"
    SYNTAX_ERROR = "syntax_error"
    SEMANTIC_ERROR = "semantic_error"
    FILE_ERROR = "file_error"
    PERMISSION_ERROR = "permission_error"
    TIMEOUT_ERROR = "timeout_error"
    MEMORY_ERROR = "memory_error"
    NETWORK_ERROR = "network_error"
    UNKNOWN_ERROR = "unknown_error"


class ErrorSeverity(Enum):
    """Error severity enumeration"""

    CRITICAL = "critical"  # Cannot recover
    HIGH = "high"  # Difficult to recover
    MEDIUM = "medium"  # May recover with retry
    LOW = "low"  # Easy to recover
    WARNING = "warning"  # Not an error


@dataclass
class ClassifiedError:
    """Classified error information"""

    error_type: ErrorType
    severity: ErrorSeverity
    message: str
    original_error: Optional[str] = None
    recoverable: bool = False
    recovery_strategy: Optional[str] = None
    suggestions: List[str] = None

    def __post_init__(self):
        if self.suggestions is None:
            self.suggestions = []


class ErrorClassifier:
    """Classify and analyze errors"""

    def __init__(self):
        """Initialize error classifier"""

        self._patterns = self._build_patterns()
        self._recovery_strategies = self._build_recovery_strategies()

    def _build_patterns(self) -> Dict[ErrorType, List[re.Pattern]]:
        """Build error pattern matchers"""

        return {
            ErrorType.TRANSLATION_ERROR: [
                re.compile(r"translation.*failed", re.IGNORECASE),
                re.compile(r"translator.*error", re.IGNORECASE),
                re.compile(r"cannot.*translate", re.IGNORECASE),
            ],
            ErrorType.COMPILATION_ERROR: [
                re.compile(r"compilation.*failed", re.IGNORECASE),
                re.compile(r"Compiler.*error", re.IGNORECASE),
                re.compile(r"undefined.*reference", re.IGNORECASE),
                re.compile(r"undefined.*symbol", re.IGNORECASE),
            ],
            ErrorType.SYNTAX_ERROR: [
                re.compile(r"syntax.*error", re.IGNORECASE),
                re.compile(r"parse.*error", re.IGNORECASE),
                re.compile(r"expected.*but.*found", re.IGNORECASE),
            ],
            ErrorType.SEMANTIC_ERROR: [
                re.compile(r"semantic.*error", re.IGNORECASE),
                re.compile(r"type.*mismatch", re.IGNORECASE),
                re.compile(r"parameter.*mismatch", re.IGNORECASE),
            ],
            ErrorType.FILE_ERROR: [
                re.compile(r"file.*not.*found", re.IGNORECASE),
                re.compile(r"no.*such.*file", re.IGNORECASE),
                re.compile(r"cannot.*open.*file", re.IGNORECASE),
            ],
            ErrorType.PERMISSION_ERROR: [
                re.compile(r"permission.*denied", re.IGNORECASE),
                re.compile(r"access.*denied", re.IGNORECASE),
            ],
            ErrorType.TIMEOUT_ERROR: [
                re.compile(r"timeout", re.IGNORECASE),
                re.compile(r"timed.*out", re.IGNORECASE),
            ],
            ErrorType.MEMORY_ERROR: [
                re.compile(r"out.*of.*memory", re.IGNORECASE),
                re.compile(r"memory.*error", re.IGNORECASE),
            ],
        }

    def _build_recovery_strategies(self) -> Dict[ErrorType, Dict[str, Any]]:
        """Build recovery strategies"""

        return {
            ErrorType.TRANSLATION_ERROR: {
                "recoverable": True,
                "strategy": "retry_with_different_mode",
                "suggestions": [
                    "Try a different translation mode",
                    "Check input file syntax",
                    "Verify translator is properly configured"
                ]
            },
            ErrorType.COMPILATION_ERROR: {
                "recoverable": True,
                "strategy": "retry_with_different_flags",
                "suggestions": [
                    "Check Compiler flags",
                    "Verify include paths",
                    "Check for missing dependencies"
                ]
            },
            ErrorType.SYNTAX_ERROR: {
                "recoverable": False,
                "strategy": "fix_syntax",
                "suggestions": [
                    "Fix syntax errors in source code",
                    "Check for missing semicolons or brackets"
                ]
            },
            ErrorType.SEMANTIC_ERROR: {
                "recoverable": False,
                "strategy": "fix_semantics",
                "suggestions": [
                    "Check Shell/Calc parameter matching",
                    "Verify data types",
                    "Check expression consistency"
                ]
            },
            ErrorType.FILE_ERROR: {
                "recoverable": True,
                "strategy": "check_file_path",
                "suggestions": [
                    "Verify file path is correct",
                    "Check file permissions",
                    "Ensure file exists"
                ]
            },
            ErrorType.PERMISSION_ERROR: {
                "recoverable": True,
                "strategy": "check_permissions",
                "suggestions": [
                    "Check file permissions",
                    "Run with appropriate privileges"
                ]
            },
            ErrorType.TIMEOUT_ERROR: {
                "recoverable": True,
                "strategy": "retry_with_longer_timeout",
                "suggestions": [
                    "Increase timeout value",
                    "Check system load",
                    "Try again later"
                ]
            },
            ErrorType.MEMORY_ERROR: {
                "recoverable": True,
                "strategy": "reduce_memory_usage",
                "suggestions": [
                    "Use streaming for large files",
                    "Reduce batch size",
                    "Free up system memory"
                ]
            },
        }

    def classify(self, error_message: str, original_error: Optional[Exception] = None) -> ClassifiedError:
        """
        Classify an error

        Args:
            error_message: Error message
            original_error: Original exception (optional)

        Returns:
            ClassifiedError object
        """
        error_type = ErrorType.UNKNOWN_ERROR
        severity = ErrorSeverity.MEDIUM

        # Try to match error patterns
        for err_type, patterns in self._patterns.items():
            for pattern in patterns:
                if pattern.search(error_message):
                    error_type = err_type
                    break
            if error_type != ErrorType.UNKNOWN_ERROR:
                break

        # Determine severity
        if error_type in [ErrorType.SYNTAX_ERROR, ErrorType.SEMANTIC_ERROR]:
            severity = ErrorSeverity.CRITICAL
        elif error_type in [ErrorType.FILE_ERROR, ErrorType.PERMISSION_ERROR]:
            severity = ErrorSeverity.HIGH
        elif error_type in [ErrorType.TIMEOUT_ERROR, ErrorType.MEMORY_ERROR]:
            severity = ErrorSeverity.MEDIUM
        elif error_type == ErrorType.UNKNOWN_ERROR:
            severity = ErrorSeverity.MEDIUM

        # Get recovery strategy
        strategy_info = self._recovery_strategies.get(error_type, {})
        recoverable = strategy_info.get("recoverable", False)
        recovery_strategy = strategy_info.get("strategy")
        suggestions = strategy_info.get("suggestions", [])

        return ClassifiedError(
            error_type=error_type,
            severity=severity,
            message=error_message,
            original_error=str(original_error) if original_error else None,
            recoverable=recoverable,
            recovery_strategy=recovery_strategy,
            suggestions=suggestions
        )

    def classify_list(self, errors: List[str]) -> List[ClassifiedError]:
        """
        Classify a list of errors

        Args:
            errors: List of error messages

        Returns:
            List of ClassifiedError objects
        """
        return [self.classify(error) for error in errors]

    def get_recovery_actions(self, classified_error: ClassifiedError) -> List[str]:
        """
        Get recovery actions for a classified error

        Args:
            classified_error: ClassifiedError object

        Returns:
            List of recovery actions
        """
        if not classified_error.recoverable:
            return ["Error is not recoverable. Manual intervention required."]

        actions = []

        if classified_error.recovery_strategy == "retry_with_different_mode":
            actions.append("Try different translation mode (usm, buffer, etc.)")
        elif classified_error.recovery_strategy == "retry_with_different_flags":
            actions.append("Try different compilation flags")
        elif classified_error.recovery_strategy == "check_file_path":
            actions.append("Verify file path and permissions")
        elif classified_error.recovery_strategy == "retry_with_longer_timeout":
            actions.append("Increase timeout and retry")
        elif classified_error.recovery_strategy == "reduce_memory_usage":
            actions.append("Use streaming or reduce batch size")

        actions.extend(classified_error.suggestions)

        return actions


