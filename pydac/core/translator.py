"""Main PyDAC translator class"""


import os

import tempfile

from typing import Optional, List, Union, Tuple, Dict, Any

from pathlib import Path


from .engine import TranslatorEngine, TranslationResult

from .compiler import CompilerManager, CompilationResult

from .executor import Executor, ExecutionResult

from .compile_db import CompileDBGenerator

from .optimizations import ResultCache, ProcessPool

from ..utils.errors import TranslationError, CompilationError

from ..utils.logger import get_logger

from ..utils.progress import progress_bar

from ..utils.performance_monitor import PerformanceMonitor, PerformanceContext

from ..utils.error_classifier import ErrorClassifier


class PyDAC:

    """Main PyDAC interface class"""

    def __init__(
        self,
        translator_path: Optional[str] = None,
        Compiler: str = "icpx",
        cache_dir: Optional[str] = None,
        verbose: bool = False,
        enable_cache: bool = True
    ):
        """
        Initialize PyDAC instance

        Args:
            translator_path: Path to translator executable (None for auto-detect)
            Compiler: Compiler name (dpcpp/clang++)
            cache_dir: Compilation cache directory
            verbose: Enable verbose output
            enable_cache: Enable result caching
        """
        self.verbose = verbose
        self.logger = get_logger("pydac", verbose)
        self.engine = TranslatorEngine(translator_path, verbose)
        self.Compiler = CompilerManager(Compiler, cache_dir)
        self.executor = Executor(verbose=verbose)
        self.Compile_db_gen = CompileDBGenerator()

        # Initialize cache and process pool
        if enable_cache:
            cache_base = cache_dir or ".pydac_cache"
            result_cache_dir = os.path.join(cache_base, "results")
            self.result_cache = ResultCache(result_cache_dir)
        else:
            self.result_cache = None

        self.process_pool = ProcessPool(max_workers=4)

        # Initialize performance monitor and error classifier
        self.performance_monitor = PerformanceMonitor(enabled=True)
        self.error_classifier = ErrorClassifier()

        if verbose:
            self.logger.info(f"PyDAC initialized (Compiler: {Compiler}, cache: {enable_cache})")

    def translate(
        self,
        input_file: str,
        output_file: Optional[str] = None,
        mode: str = "usm",
        extra_args: Optional[List[str]] = None
    ) -> TranslationResult:
        """
        Translate DAC code to target backend

        Args:
            input_file: Input C++ file path
            output_file: Output file path (None for auto-generate)
            mode: Translation mode (usm/buffer/usm_time/mpi/sycl)
            extra_args: Additional translator arguments

        Returns:
            TranslationResult: Translation result
        """
        self.logger.debug(f"Translating {input_file} to {mode} mode")

        # Performance monitoring
        with PerformanceContext(
            self.performance_monitor,
            "translation",
            {"file": input_file, "mode": mode}
        ):
            # Check cache first
            if self.result_cache:
                cache_params = {"mode": mode, "extra_args": extra_args or []}
                cached_result = self.result_cache.get(input_file, "translation", cache_params)
                if cached_result:
                    self.logger.debug("Using cached translation result")
                    from .engine import TranslationResult
                    return TranslationResult(**cached_result)

            # Check if compile_commands.json exists, generate if not
            source_dir = Path(input_file).parent
            compile_db = source_dir / "compile_commands.json"
            if not compile_db.exists():
                self.logger.info("Generating compile_commands.json...")
                try:
                    db_path = self.Compile_db_gen.generate_for_test_case(input_file)
                    if not Path(db_path).exists():
                        self.logger.warning(
                            f"compile_commands.json generation reported success but file not found at {db_path}. "
                            f"Translation may fail if translator requires compilation database."
                        )
                except Exception as e:
                    self.logger.warning(
                        f"Failed to generate compile_commands.json: {e}. "
                        f"Translation may fail if translator requires compilation database. "
                        f"Please ensure the file exists at {compile_db} or that the translator can work without it."
                    )

            try:
                result = self.engine.translate(input_file, output_file, mode, extra_args)
            except Exception as e:
                # Classify error
                classified = self.error_classifier.classify(str(e), e)
                self.logger.error(f"Translation error ({classified.error_type.value}): {classified.message}")
                if classified.suggestions:
                    self.logger.info(f"Suggestions: {', '.join(classified.suggestions[:2])}")
                raise

            # Cache result if successful
            if self.result_cache and result.success:
                cache_params = {"mode": mode, "extra_args": extra_args or []}
                result_dict = {
                    "success": result.success,
                    "input_file": result.input_file,
                    "output_file": result.output_file,
                    "mode": result.mode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "warnings": result.warnings,
                    "errors": result.errors,
                    "duration": result.duration
                }
                self.result_cache.put(input_file, "translation", cache_params, result_dict)

            if result.success:
                self.logger.info(f"Translation successful: {result.output_file} ({result.duration:.2f}s)")
                if result.warnings:
                    for warning in result.warnings:
                        self.logger.warning(warning)
            else:
                self.logger.error(f"Translation failed: {result.errors}")
                if result.errors:
                    raise TranslationError(
                        f"Translation failed: {result.errors[0]}",
                        stderr=result.stderr,
                        errors=result.errors
                    )

            return result

    def translate_code(
        self,
        code: str,
        mode: str = "usm",
        output_file: Optional[str] = None
    ) -> TranslationResult:
        """
        Translate code string directly

        Args:
            code: C++ code string
            mode: Translation mode
            output_file: Output file path

        Returns:
            TranslationResult: Translation result
        """
        # Create temporary file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.cpp',
            delete=False
        ) as f:
            f.write(code)
            temp_file = f.name

        try:
            return self.translate(temp_file, output_file, mode)
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def Compile(
        self,
        source_file: str,
        output_binary: Optional[str] = None,
        flags: Optional[List[str]] = None
    ) -> CompilationResult:
        """
        Compile translated code

        Args:
            source_file: Source file path
            output_binary: Output binary path
            flags: Compilation flags

        Returns:
            CompilationResult: Compilation result
        """
        self.logger.debug(f"Compiling {source_file}")
        result = self.Compiler.compile(source_file, output_binary, flags)

        if result.success:
            self.logger.info(f"Compilation successful: {result.binary_file} ({result.duration:.2f}s)")
        else:
            self.logger.error(f"Compilation failed: {result.stderr[:200]}")
            if result.stderr:
                raise CompilationError(
                    f"Compilation failed: {result.stderr[:200]}",
                    stderr=result.stderr,
                    errors=self._parse_compilation_errors(result.stderr)
                )

        return result

    def _parse_compilation_errors(self, stderr: str) -> List[str]:
        """
        Parse compilation errors from Compiler output

        Args:
            stderr: Standard error output from Compiler

        Returns:
            List of error messages
        """
        errors = []
        for line in stderr.split('\n'):
            if 'error' in line.lower():
                errors.append(line.strip())
        return errors

    def translate_and_Compile(
        self,
        input_file: str,
        mode: str = "usm",
        Compile: bool = True,
        validate: bool = False
    ) -> Union[TranslationResult, CompilationResult]:
        """
        Translate and Compile (one-step operation)

        Args:
            input_file: Input file path
            mode: Translation mode
            Compile: Whether to Compile after translation
            validate: Whether to validate code before translation

        Returns:
            TranslationResult or CompilationResult
        """
        # Validate if requested
        if validate:
            is_valid, errors = self.validate_file(input_file)
            if not is_valid:
                self.logger.warning(f"Code validation found issues: {errors}")

        # Translate
        trans_result = self.translate(input_file, mode=mode)

        if not trans_result.success:
            return trans_result

        if not Compile:
            return trans_result

        # Compile
        Compile_result = self.Compile(trans_result.output_file)
        return Compile_result

    def run(
        self,
        binary_file: str,
        args: Optional[List[str]] = None,
        input_data: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None
    ) -> ExecutionResult:
        """
        Run a Compiled binary

        Args:
            binary_file: Path to Compiled binary file
            args: Command line arguments
            input_data: Input data (stdin)
            env: Environment variables
            timeout: Execution timeout in seconds

        Returns:
            ExecutionResult: Execution result

        Example:
            >>> translator = PyDAC()
            >>> result = translator.translate_and_Compile("input.dac.cpp", mode="usm")
            >>> if result.success:
            ...     exec_result = translator.run(result.binary_file)
            ...     print(exec_result.stdout)
        """
        if timeout is not None:
            self.executor.timeout = timeout

        return self.executor.run(binary_file, args, input_data, env)

    def translate_Compile_and_run(
        self,
        input_file: str,
        mode: str = "usm",
        run_args: Optional[List[str]] = None,
        input_data: Optional[str] = None,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Translate, Compile, and run in one step

        Args:
            input_file: Input file path
            mode: Translation mode
            run_args: Arguments for running the binary
            input_data: Input data (stdin)
            timeout: Execution timeout

        Returns:
            Dictionary with translation, compilation, and execution results

        Example:
            >>> translator = PyDAC()
            >>> results = translator.translate_Compile_and_run("input.dac.cpp", mode="usm")
            >>> if results["execution"].success:
            ...     print(results["execution"].stdout)
        """
        results = {}

        # Translate
        trans_result = self.translate(input_file, mode=mode)
        results["translation"] = trans_result

        if not trans_result.success:
            results["compilation"] = None
            results["execution"] = None
            return results

        # Compile
        Compile_result = self.Compile(trans_result.output_file)
        results["compilation"] = Compile_result

        if not Compile_result.success:
            results["execution"] = None
            return results

        # Run
        exec_result = self.run(Compile_result.binary_file, run_args, input_data, timeout=timeout)
        results["execution"] = exec_result

        return results

    def translate_batch(
        self,
        input_files: List[str],
        output_dir: Optional[str] = None,
        mode: str = "usm",
        parallel: bool = True
    ) -> List[TranslationResult]:
        """
        Translate multiple files in batch

        Args:
            input_files: List of input file paths
            output_dir: Output directory (None for same as input)
            mode: Translation mode
            parallel: Whether to process in parallel

        Returns:
            List of TranslationResult
        """
        results = []

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        if parallel:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            self.logger.info(f"Batch translating {len(input_files)} files in parallel")

            # Optimize file order for better cache locality
            from .optimizations import optimize_file_operations
            optimized_files = optimize_file_operations(input_files)

            # Use process pool if available
            executor = self.process_pool._executor if Hasattr(self.process_pool, '_executor') else None
            if executor is None:
                executor = ThreadPoolExecutor(max_workers=4)
                use_context = True
            else:
                use_context = False

            if use_context:
                with executor:
                    futures = self._submit_batch_tasks(executor, optimized_files, output_dir, mode)
                    results = self._collect_batch_results(futures, len(optimized_files))
            else:
                futures = self._submit_batch_tasks(executor, optimized_files, output_dir, mode)
                results = self._collect_batch_results(futures, len(optimized_files))
        else:
            with progress_bar(len(input_files), "Translating") as pb:
                for input_file in input_files:
                    output_file = None
                    if output_dir:
                        path = Path(input_file)
                        output_file = os.path.join(output_dir, f"{path.stem}_{mode}.cpp")

                    result = self._translate_without_error(input_file, output_file, mode)
                    results.append(result)
                    pb.update(1)

        return results

    def _submit_batch_tasks(self, executor, input_files, output_dir, mode):
        """Submit batch translation tasks"""

        futures = {}
        for input_file in input_files:
            output_file = None
            if output_dir:
                path = Path(input_file)
                output_file = os.path.join(output_dir, f"{path.stem}_{mode}.cpp")

            future = executor.submit(
                self._translate_without_error,
                input_file,
                output_file,
                mode
            )
            futures[future] = input_file
        return futures

    def _collect_batch_results(self, futures, total_count):
        """Collect batch translation results"""

        from concurrent.futures import as_completed
        results = []

        with progress_bar(total_count, "Translating") as pb:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    pb.update(1)
                except Exception as e:
                    # Create error result
                    results.append(TranslationResult(
                        success=False,
                        input_file=futures[future],
                        output_file="",
                        mode="usm",  # Default mode
                        stdout="",
                        stderr=str(e),
                        warnings=[],
                        errors=[str(e)],
                        duration=0.0
                    ))
                    pb.update(1)
        return results

    def _translate_without_error(
        self,
        input_file: str,
        output_file: Optional[str],
        mode: str
    ) -> TranslationResult:
        """
        Translate without raising exception (for batch processing)

        This method wraps translate() to catch exceptions and return
        a TranslationResult with success=False instead of raising.

        Args:
            input_file: Input file path
            output_file: Output file path (optional)
            mode: Translation mode

        Returns:
            TranslationResult: Result object (may have success=False)
        """
        try:
            return self.engine.translate(input_file, output_file, mode)
        except Exception as e:
            return TranslationResult(
                success=False,
                input_file=input_file,
                output_file=output_file or "",
                mode=mode,
                stdout="",
                stderr=str(e),
                warnings=[],
                errors=[str(e)],
                duration=0.0
            )

    def validate_code(self, code: str) -> Tuple[bool, List[str]]:
        """
        Validate DAC code syntax

        Args:
            code: C++ code string

        Returns:
            (is_valid, errors) tuple
        """
        from ..utils.validator import CodeValidator
        validator = CodeValidator()
        return validator.validate_syntax(code)

    def validate_file(self, file_path: str) -> Tuple[bool, List[str]]:
        """
        Validate DAC file

        Args:
            file_path: File path

        Returns:
            (is_valid, errors) tuple
        """
        # Check cache first
        if self.result_cache:
            cache_params = {}
            cached_result = self.result_cache.get(file_path, "validation", cache_params)
            if cached_result:
                self.logger.debug("Using cached validation result")
                return cached_result.get("is_valid", False), cached_result.get("errors", [])

        from ..utils.validator import CodeValidator
        validator = CodeValidator()
        is_valid, errors = validator.validate_file(file_path)

        # Cache result
        if self.result_cache:
            cache_params = {}
            self.result_cache.put(file_path, "validation", cache_params, {
                "is_valid": is_valid,
                "errors": errors
            })

        return is_valid, errors

    def verify_semantics(self, file_path: str) -> Tuple[bool, List[str]]:
        """
        Verify semantic correctness

        Args:
            file_path: File path

        Returns:
            (is_valid, errors) tuple
        """
        from ..utils.validator import CodeValidator
        validator = CodeValidator()
        is_valid, errors = validator.verify_semantics(
            Path(file_path).read_text() if Path(file_path).exists() else ""
        )
        return is_valid, errors

    def compare_results(
        self,
        original_file: str,
        translated_file: str
    ) -> dict:
        """
        Compare original and translated code

        Args:
            original_file: Original DAC file path
            translated_file: Translated SYCL file path

        Returns:
            Comparison result dictionary
        """
        from ..utils.verifier import CodeVerifier
        verifier = CodeVerifier()
        result = verifier.compare_results(original_file, translated_file)

        return {
            "is_similar": result.is_similar,
            "similarity_score": result.similarity_score,
            "differences": result.differences,
            "details": result.details
        }

    def compare_modes(
        self,
        input_file: str,
        modes: List[str] = ["usm", "buffer", "usm_time"]
    ) -> dict:
        """
        Compare different translation modes

        Args:
            input_file: Input file path
            modes: List of modes to compare

        Returns:
            Dictionary with comparison results
        """
        comparison = {}

        for mode in modes:
            self.logger.info(f"Translating with mode: {mode}")
            result = self._translate_without_error(input_file, None, mode)
            comparison[mode] = {
                "success": result.success,
                "output_file": result.output_file,
                "duration": result.duration,
                "warnings": len(result.warnings),
                "errors": len(result.errors)
            }

        return comparison

    def get_version(self) -> str:
        """
        Get PyDAC version string

        Returns:
            Version string (e.g., "0.1.0")
        """
        from .. import __version__
        return __version__

    def get_info(self) -> dict:
        """
        Get PyDAC instance information

        Returns:
            Dictionary containing:
            - version: PyDAC version
            - translator_path: Path to translator executable
            - Compiler: Compiler name
            - cache_dir: Cache directory path
            - verbose: Verbose mode status
            - cache_enabled: Whether caching is enabled
        """
        return {
            "version": self.get_version(),
            "translator_path": self.engine.translator_path,
            "Compiler": self.Compiler.compiler,
            "cache_dir": str(self.Compiler.cache_dir),
            "verbose": self.verbose,
            "cache_enabled": self.result_cache is not None
        }
