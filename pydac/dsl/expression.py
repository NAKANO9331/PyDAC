"""Expression DSL for PyDAC"""


from typing import Union, Optional, List, Tuple

from .shell import Shell

from .calc import Calc


class Expression:

    """Data association expression DSL (<->)"""

    def __init__(
        self,
        shell: Union[Shell, str],
        calc: Union[Calc, str],
        arguments: Optional[List[str]] = None
    ):
        """
        Initialize Expression

        Args:
            shell: Shell function or shell name
            calc: Calc function or calc name
            arguments: Arguments for shell function call
        """
        self.shell = shell
        self.calc = calc
        self.arguments = arguments or []

    def to_cpp_code(self) -> str:
        """
        Generate C++ expression code

        Returns:
            C++ code string: shellName(args) <-> calcName;
        """
        # Get shell name
        if isinstance(self.shell, Shell):
            shell_name = self.shell.name
        else:
            shell_name = str(self.shell)

        # Get calc name
        if isinstance(self.calc, Calc):
            calc_name = self.calc.name
        else:
            calc_name = str(self.calc)

        # Generate arguments string
        if self.arguments:
            args_str = ", ".join(self.arguments)
        else:
            # Try to extract arguments from shell if it's a Shell object
            if isinstance(self.shell, Shell):
                args_str = ", ".join([name for name, _ in self.shell.slices])
            else:
                args_str = ""

        # Generate expression
        code = f"{shell_name}({args_str}) <-> {calc_name};"
        return code

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate expression

        Returns:
            (is_valid, errors) tuple
        """
        errors = []

        # Check shell
        if isinstance(self.shell, Shell):
            # Validate shell structure
            if not self.shell.splits and not self.shell.indices:
                errors.append("Shell must have at least one split or index")
            if not self.shell.slices:
                errors.append("Shell must have at least one slice")
        elif not isinstance(self.shell, str):
            errors.append("Shell must be Shell object or string")

        # Check calc
        if isinstance(self.calc, Calc):
            # Validate calc structure
            if not self.calc.params:
                errors.append("Calc must have at least one parameter")
            if not self.calc.body:
                errors.append("Calc must have function body")
        elif not isinstance(self.calc, str):
            errors.append("Calc must be Calc object or string")

        # Check parameter matching if both are objects
        if isinstance(self.shell, Shell) and isinstance(self.calc, Calc):
            # Check if number of slices matches number of calc parameters
            if len(self.shell.slices) != len(self.calc.params):
                errors.append(
                    f"Shell slices count ({len(self.shell.slices)}) "
                    f"does not match Calc parameters count ({len(self.calc.params)})"
                )

        return len(errors) == 0, errors


class ExpressionBuilder:
    """Builder for creating expressions"""

    def __init__(self):
        """Initialize builder"""

        self.shell: Optional[Union[Shell, str]] = None
        self.calc: Optional[Union[Calc, str]] = None
        self.arguments: List[str] = []

    def with_shell(self, shell: Union[Shell, str]) -> 'ExpressionBuilder':
        """Set shell"""

        self.shell = shell
        return self

    def with_calc(self, calc: Union[Calc, str]) -> 'ExpressionBuilder':
        """Set calc"""

        self.calc = calc
        return self

    def with_arguments(self, *args: str) -> 'ExpressionBuilder':
        """Set arguments"""

        self.arguments = list(args)
        return self

    def build(self) -> Expression:
        """Build expression"""

        if self.shell is None:
            raise ValueError("Shell must be set")
        if self.calc is None:
            raise ValueError("Calc must be set")

        return Expression(self.shell, self.calc, self.arguments)

