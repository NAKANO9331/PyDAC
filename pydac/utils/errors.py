"""Error handling for PyDAC"""


class PyDACError(Exception):
    """Base exception for PyDAC"""

    pass


class TranslationError(PyDACError):
    """Translation error"""

    def __init__(self, message: str, stderr: str = "", errors: list = None):
        self.stderr = stderr
        self.errors = errors or []
        super().__init__(message)


class CompilationError(PyDACError):
    """Compilation error"""

    def __init__(self, message: str, stderr: str = "", errors: list = None):
        self.stderr = stderr
        self.errors = errors or []
        super().__init__(message)


class SyntaxError(PyDACError):
    """Syntax error"""

    def __init__(self, message: str, line: int = 0, column: int = 0):
        self.line = line
        self.column = column
        super().__init__(message)

