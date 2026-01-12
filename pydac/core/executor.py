"""Executor for running Compiled binaries"""


import os

import subprocess

import time

from pathlib import Path

from typing import Optional, List, Dict, Any, Tuple

from dataclasses import dataclass

from ..utils.logger import get_logger


@dataclass
class ExecutionResult:
    """Result of execution operation"""

    success: bool
    binary_file: str
    stdout: str
    stderr: str
    return_code: int
    duration: float
    command: str
    args: List[str]


class Executor:

    """Executor for running Compiled binaries"""


    def __init__(self, timeout: Optional[float] = None, verbose: bool = False):
        """
        Initialize executor

        Args:
            timeout: Execution timeout in seconds (None for no timeout)
            verbose: Enable verbose output
        """
        self.timeout = timeout
        self.verbose = verbose
        self.logger = get_logger("pydac.executor", verbose)

    def run(
        self,
        binary_file: str,
        args: Optional[List[str]] = None,
        input_data: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None
    ) -> ExecutionResult:
        """
        Run a Compiled binary

        Args:
            binary_file: Path to binary file
            args: Command line arguments
            input_data: Input data (stdin)
            env: Environment variables
            cwd: Working directory

        Returns:
            ExecutionResult: Execution result

        Raises:
            FileNotFoundError: If binary file doesn't exist
        """
        binary_path = Path(binary_file)

        if not binary_path.exists():
            raise FileNotFoundError(f"Binary file not found: {binary_file}")

        if not os.access(binary_path, os.X_OK):
            raise PermissionError(f"Binary file is not executable: {binary_file}")

        args = args or []
        command = [str(binary_path.absolute())] + args
        full_command = " ".join(command)

        self.logger.debug(f"Running: {full_command}")

        start_time = time.time()

        try:
            # Prepare environment
            exec_env = os.environ.copy()
            if env:
                exec_env.update(env)

            # Run the binary
            process = subprocess.run(
                command,
                input=input_data.encode() if input_data else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout,
                env=exec_env,
                cwd=cwd or str(binary_path.parent)
            )

            duration = time.time() - start_time

            stdout = process.stdout.decode('utf-8', errors='replace')
            stderr = process.stderr.decode('utf-8', errors='replace')

            success = process.returncode == 0

            if self.verbose:
                if success:
                    self.logger.info(f"Execution successful (return code: {process.returncode}, duration: {duration:.2f}s)")
                else:
                    self.logger.warning(f"Execution failed (return code: {process.returncode}, duration: {duration:.2f}s)")
                if stderr:
                    self.logger.warning(f"Error output: {stderr[:200]}")

            return ExecutionResult(
                success=success,
                binary_file=str(binary_path.absolute()),
                stdout=stdout,
                stderr=stderr,
                return_code=process.returncode,
                duration=duration,
                command=full_command,
                args=args
            )

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            self.logger.error(f"Execution timeout after {self.timeout}s")
            return ExecutionResult(
                success=False,
                binary_file=str(binary_path.absolute()),
                stdout="",
                stderr=f"Execution timeout after {self.timeout}s",
                return_code=-1,
                duration=duration,
                command=full_command,
                args=args
            )
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Execution error: {e}")
            return ExecutionResult(
                success=False,
                binary_file=str(binary_path.absolute()),
                stdout="",
                stderr=str(e),
                return_code=-1,
                duration=duration,
                command=full_command,
                args=args
            )

    def run_with_output(
        self,
        binary_file: str,
        args: Optional[List[str]] = None,
        input_data: Optional[str] = None,
        env: Optional[Dict[str, str]] = None
    ) -> Tuple[bool, str, str]:
        """
        Run binary and return (success, stdout, stderr)

        Args:
            binary_file: Path to binary file
            args: Command line arguments
            input_data: Input data (stdin)
            env: Environment variables

        Returns:
            Tuple of (success, stdout, stderr)
        """
        result = self.run(binary_file, args, input_data, env)
        return (result.success, result.stdout, result.stderr)

