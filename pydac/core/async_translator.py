"""Async support for PyDAC"""


import asyncio

from typing import Optional, List

from pathlib import Path


from .translator import PyDAC

from .engine import TranslationResult

from .compiler import CompilationResult

from ..utils.logger import get_logger


class AsyncPyDAC:

    """Async wrapper for PyDAC"""

    def __init__(self, translator: Optional[PyDAC] = None, **kwargs):
        """
        Initialize async PyDAC wrapper

        Args:
            translator: PyDAC instance (None to create new)
            **kwargs: Arguments for PyDAC if translator is None
        """
        self.translator = translator or PyDAC(**kwargs)
        self.logger = get_logger("pydac.async", self.translator.verbose)

    async def translate(
        self,
        input_file: str,
        output_file: Optional[str] = None,
        mode: str = "usm",
        extra_args: Optional[List[str]] = None
    ) -> TranslationResult:
        """
        Async translate DAC code

        Args:
            input_file: Input C++ file path
            output_file: Output file path
            mode: Translation mode
            extra_args: Additional translator arguments

        Returns:
            TranslationResult: Translation result
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.translator.translate,
            input_file,
            output_file,
 mode,
 extra_args
 )

    async def translate_code(
        self,
        code: str,
        mode: str = "usm",
        output_file: Optional[str] = None
    ) -> TranslationResult:
        """
        Async translate code string

        Args:
            code: C++ code string
            mode: Translation mode
            output_file: Output file path

        Returns:
            TranslationResult: Translation result
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.translator.translate_code,
            code,
            mode,
            output_file
 )

    async def Compile(
        self,
        source_file: str,
        output_binary: Optional[str] = None,
        flags: Optional[List[str]] = None
    ) -> CompilationResult:
        """
        Async Compile translated code

        Args:
            source_file: Source file path
            output_binary: Output binary path
            flags: Compilation flags

        Returns:
            CompilationResult: Compilation result
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.translator.Compile,
            source_file,
            output_binary,
            flags
        )

    async def translate_and_Compile(
        self,
        input_file: str,
        mode: str = "usm",
        Compile: bool = True,
        validate: bool = False
    ):
        """
        Async translate and Compile

        Args:
            input_file: Input file path
            mode: Translation mode
            Compile: Whether to Compile after translation
            validate: Whether to validate code before translation

        Returns:
            TranslationResult or CompilationResult
        """
        trans_result = await self.translate(input_file, mode=mode)

        if not trans_result.success or not Compile:
            return trans_result

        return await self.Compile(trans_result.output_file)

    async def translate_batch(
        self,
        input_files: List[str],
        output_dir: Optional[str] = None,
        mode: str = "usm",
        max_concurrent: int = 4
    ) -> List[TranslationResult]:
        """
        Async batch translate multiple files

        Args:
            input_files: List of input file paths
            output_dir: Output directory
            mode: Translation mode
            max_concurrent: Maximum concurrent translations

        Returns:
            List of TranslationResult
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def translate_with_semaphore(input_file: str):
            async with semaphore:
                output_file = None
                if output_dir:
                    path = Path(input_file)
                    output_file = str(Path(output_dir) / f"{path.stem}_{mode}.cpp")
                return await self.translate(input_file, output_file, mode)

        tasks = [translate_with_semaphore(f) for f in input_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(TranslationResult(
                    success=False,
                    input_file=input_files[i],
                    output_file="",
                    mode=mode,
                    stdout="",
                    stderr=str(result),
                    warnings=[],
                    errors=[str(result)],
                    duration=0.0
                ))
            else:
                final_results.append(result)

        return final_results

    async def validate_file(self, file_path: str):
        """Async validate file"""

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.translator.validate_file,
            file_path
        )


