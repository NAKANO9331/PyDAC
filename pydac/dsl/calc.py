"""Calc DSL for PyDAC"""


from typing import List, Tuple


class Calc:

    """Calc function DSL"""

    def __init__(self, name: str, return_type: str = "void"):
        """
        Initialize Calc

        Args:
            name: Calc function name
            return_type: Return type
        """
        self.name = name
        self.return_type = return_type
        self.params: List[Tuple[str, str]] = []  # [(name, type), ...]
        self.body: str = ""

    def add_param(self, name: str, param_type: str) -> 'Calc':
        """Add parameter"""

        self.params.append((name, param_type))
        return self

    def set_body(self, code: str) -> 'Calc':
        """Set function body"""

        self.body = code
        return self

    def to_cpp_code(self) -> str:
        """Generate C++ code"""

        code = f"calc {self.return_type} {self.name}("

        # Generate parameter list
        param_strs = []
        for name, param_type in self.params:
            param_strs.append(f"{param_type} {name}")
        code += ", ".join(param_strs) + ") {\n"

        # Add function body
        for line in self.body.split('\n'):
            code += f" {line}\n"

        code += "}\n"
        return code

