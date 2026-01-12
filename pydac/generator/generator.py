"""Code generator for PyDAC"""


from typing import Dict, Any, Optional, Union, TYPE_CHECKING

from pathlib import Path

import os


if TYPE_CHECKING:
 from ..dsl import Shell, Calc
 from ..tensor import Tensor


class CodeGenerator:

    """Code generator for PyDAC"""

    def __init__(self, template_dir: Optional[str] = None):
        """
        Initialize code generator

        Args:
            template_dir: Template directory path
        """
        if template_dir:
            self.template_dir = Path(template_dir)
        else:
            # Use default template directory
            self.template_dir = Path(__file__).parent / "templates"

        self.template_dir.mkdir(parents=True, exist_ok=True)

    def generate_from_template(
        self,
        template_name: str,
        parameters: Dict[str, Any],
        output_file: Optional[str] = None
    ) -> str:
        """
        Generate code from template

        Args:
            template_name: Template name (stencil/jacobi/matmul etc.)
            parameters: Template parameters
            output_file: Output file path (None to return string)

        Returns:
            Generated code string
        """
        template_file = self.template_dir / f"{template_name}.cpp.j2"

        if not template_file.exists():
            raise FileNotFoundError(f"Template not found: {template_file}")

        try:
            from jinja2 import Template, Environment, FileSystemLoader

            env = Environment(loader=FileSystemLoader(str(self.template_dir)))
            template = env.get_template(f"{template_name}.cpp.j2")
            code = template.render(**parameters)

            if output_file:
                with open(output_file, 'w') as f:
                    f.write(code)

            return code
        except ImportError:
            # Fallback to simple string replacement
            with open(template_file, 'r') as f:
                template_content = f.read()

            # Simple replacement
            code = template_content
            for key, value in parameters.items():
                code = code.replace(f"{{{{ {key} }}}}", str(value))
                code = code.replace(f"${{{key}}}", str(value))

            if output_file:
                with open(output_file, 'w') as f:
                    f.write(code)

            return code

    def generate_complete_code(
        self,
        shell: Union[str, Any],
        calc: Union[str, Any],
        tensors: Dict[str, Any],
        main_logic: Optional[str] = None,
        includes: Optional[list] = None
    ) -> str:
        """
        Generate complete C++ code

        Args:
            shell: Shell definition or C++ code string
            calc: Calc definition or C++ code string
            tensors: Dictionary of Tensor objects
            main_logic: Main function logic
            includes: Additional includes

        Returns:
            Complete C++ code string
        """
        from ..dsl import Shell, Calc
        from ..tensor import Tensor

        code_parts = []

        # Includes
        code_parts.append('#include <iostream>')
        code_parts.append('#include <vector>')
        code_parts.append('#include "ReconTensor.h"')
        if includes:
            for inc in includes:
                code_parts.append(f'#include "{inc}"')
        code_parts.append('')

        # Namespace
        code_parts.append('namespace dacpp {')
        code_parts.append(' typedef std::vector<std::any> list;')
        code_parts.append('}')
        code_parts.append('')

        # Shell code
        if isinstance(shell, Shell):
            code_parts.append(shell.to_cpp_code())
        else:
            code_parts.append(shell)
        code_parts.append('')

        # Calc code
        if isinstance(calc, Calc):
            code_parts.append(calc.to_cpp_code())
        else:
            code_parts.append(calc)
        code_parts.append('')

        # Main function
        code_parts.append('int main() {')

        # Tensor initialization
        for name, tensor in tensors.items():
            cpp_init = tensor.to_cpp_init(name)
            code_parts.append(f" {cpp_init}")

        code_parts.append('')

        # Main logic
        if main_logic:
            for line in main_logic.split('\n'):
                code_parts.append(f" {line}")
        else:
            # Default: call shell <-> calc
            if isinstance(shell, Shell):
                shell_name = shell.name
            else:
                # Extract shell name from code
                shell_name = "shellName"  # Default

            if isinstance(calc, Calc):
                calc_name = calc.name
            else:
                calc_name = "calcName"  # Default

            # Generate default expression
            tensor_names = list(tensors.keys())
            if tensor_names:
                params = ", ".join(tensor_names)
                code_parts.append(f" {shell_name}({params}) <-> {calc_name};")

        code_parts.append('')
        code_parts.append(' return 0;')
        code_parts.append('}')

        return '\n'.join(code_parts)

    def list_templates(self) -> list:
        """List available templates"""

        templates = []
        for file in self.template_dir.glob("*.cpp.j2"):
            templates.append(file.stem.replace(".cpp", ""))
        return templates

    def create_template(
        self,
        template_name: str,
        shell: Union[str, Any],
        calc: Union[str, Any],
        main_template: str
    ):
        """
        Create a new template

        Args:
            template_name: Template name
            shell: Shell definition or code
            calc: Calc definition or code
            main_template: Main function template
        """
        from ..dsl import Shell, Calc

        template_file = self.template_dir / f"{template_name}.cpp.j2"

        # Generate template content
        content_parts = []
        content_parts.append("#include <iostream>")
        content_parts.append('#include "ReconTensor.h"')
        content_parts.append("")
        content_parts.append("namespace dacpp {")
        content_parts.append(" typedef std::vector<std::any> list;")
        content_parts.append("}")
        content_parts.append("")

        # Add shell
        if isinstance(shell, Shell):
            content_parts.append(shell.to_cpp_code())
        else:
            content_parts.append("{{ shell_code }}")

        # Add calc
        if isinstance(calc, Calc):
            content_parts.append(calc.to_cpp_code())
        else:
            content_parts.append("{{ calc_code }}")

        # Add main
        content_parts.append(main_template)

        with open(template_file, 'w') as f:
            f.write('\n'.join(content_parts))

        return str(template_file)

