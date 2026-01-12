"""Code formatting utilities for PyDAC"""


import re

from typing import List


def format_cpp_code(code: str, indent: int = 0) -> str:
    """
    Basic C++ code formatting

    Args:
        code: C++ code string
        indent: Base indentation level

    Returns:
        Formatted code
    """
    lines = code.split('\n')
    formatted = []
    current_indent = indent

    for line in lines:
        stripped = line.strip()
        if not stripped:
            formatted.append("")
            continue

        # Decrease indent on closing braces
        if stripped.startswith('}'):
            current_indent = max(0, current_indent - 1)

        # Add indented line
        formatted.append(" " * current_indent + stripped)

        # Increase indent on opening braces
        if stripped.endswith('{'):
            current_indent += 1

    return '\n'.join(formatted)


def remove_comments(code: str) -> str:
    """Remove comments from code"""

    # Remove single-line comments
    code = re.sub(r'//.*', '', code)

    # Remove multi-line comments
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)

    return code


def extract_functions(code: str) -> List[dict]:
    """
    Extract function definitions from code

    Returns:
        List of function information dictionaries
    """
    functions = []

    # Pattern for function definitions
    pattern = r'(\w+)\s+(\w+)\s*\(([^)]*)\)\s*\{'

    for match in re.finditer(pattern, code):
        return_type = match.group(1)
        name = match.group(2)
        params = match.group(3)

        functions.append({
            "name": name,
            "return_type": return_type,
            "parameters": [p.strip() for p in params.split(',') if p.strip()]
        })

    return functions


def normalize_whitespace(code: str) -> str:
    """Normalize whitespace in code"""

    # Replace multiple spaces with single space
    code = re.sub(r' +', ' ', code)

    # Replace multiple newlines with double newline
    code = re.sub(r'\n{3,}', '\n\n', code)

    return code.strip()

