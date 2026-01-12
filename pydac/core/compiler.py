"""Compiler management for PyDAC"""

import os
import subprocess
import hashlib
import shutil
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass
import time
from ..utils.logger import get_logger


@dataclass
class CompilationResult:
    """Result of compilation operation"""
    success: bool
    source_file: str
    binary_file: str
    stdout: str
    stderr: str
    duration: float


class CompilerManager:
    """Compiler manager implementation"""
    
    def __init__(
        self,
        compiler: str = "icpx",
        cache_dir: Optional[str] = None,
        default_flags: Optional[List[str]] = None
    ):
        """
        Initialize compiler manager
        
        Args:
            compiler: Compiler name (icpx/dpcpp/clang++)
                - icpx: Intel oneAPI C++ compiler (recommended, supports SYCL)
                - dpcpp: Deprecated, use icpx instead
                - clang++: Clang C++ compiler
            cache_dir: Cache directory path
            default_flags: Default compilation flags
        """
        self.compiler = compiler
        self.cache_dir = Path(cache_dir) if cache_dir else Path(".pydac_cache")
        # For icpx, -fsycl is required for SYCL support
        if default_flags is None:
            if compiler == "icpx":
                self.default_flags = ["-O2", "-std=c++17", "-fsycl"]
            elif compiler == "dpcpp":
                # dpcpp already includes SYCL support by default
                self.default_flags = ["-O2", "-std=c++17"]
            else:
                self.default_flags = ["-O2", "-std=c++17"]
        else:
            self.default_flags = default_flags
        self.logger = get_logger("pydac.compiler", False)
        self._setup_cache()
        self._validate_compiler()
    
    def _setup_cache(self):
        """
        Setup compilation cache directory
        
        Creates the cache directory if it doesn't exist
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _validate_compiler(self):
        """Validate compiler is available"""
        compiler_path = shutil.which(self.compiler)
        if not compiler_path:
            available_compilers = []
            for comp in ["dpcpp", "clang++", "g++"]:
                if shutil.which(comp):
                    available_compilers.append(comp)
            
            suggestions = ""
            if available_compilers:
                suggestions = f"\n  Available compilers: {', '.join(available_compilers)}\n  You can use: PyDAC(compiler='{available_compilers[0]}')"
            
            raise RuntimeError(
                f"Compiler '{self.compiler}' not found in PATH.\n"
                f"  Please:\n"
                f"  1. Install {self.compiler} and add it to your PATH\n"
                f"  2. Or use a different compiler by setting: PyDAC(compiler='<compiler_name>')"
                + suggestions
            )
    
    def _get_cache_key(self, source_file: str, flags: List[str]) -> str:
        """
        Generate cache key based on source file content and compilation flags
        
        Args:
            source_file: Source file path
            flags: Compilation flags
            
        Returns:
            MD5 hash string as cache key
        """
        # Read source file content
        with open(source_file, 'rb') as f:
            content = f.read()
        
        # Get file modification time
        mtime = os.path.getmtime(source_file)
        
        # Generate hash (convert flags to bytes)
        flags_bytes = b''.join(f.encode() if isinstance(f, str) else f for f in flags)
        key_data = content + flags_bytes + str(mtime).encode()
        return hashlib.md5(key_data).hexdigest()
    
    def compile(
        self,
        source_file: str,
        output_binary: Optional[str] = None,
        flags: Optional[List[str]] = None
    ) -> CompilationResult:
        """
        Compile source code
        
        Args:
            source_file: Source file path
            output_binary: Output binary path (None for auto-generate)
            flags: Compilation flags
        
        Returns:
            CompilationResult: Compilation result
        """
        if not os.path.exists(source_file):
            raise FileNotFoundError(
                f"Source file not found: {source_file}\n"
                f"  Please check that the file path is correct.\n"
                f"  Current working directory: {os.getcwd()}"
            )
        
        start_time = time.time()
        
        # Check cache
        compile_flags = flags or self.default_flags
        cache_key = self._get_cache_key(source_file, compile_flags)
        cached_binary = self.cache_dir / f"{cache_key}.bin"
        
        if cached_binary.exists():
            # Return cached result
            self.logger.debug(f"Using cached binary: {cached_binary}")
            return CompilationResult(
                success=True,
                source_file=source_file,
                binary_file=str(cached_binary),
                stdout="Using cached binary",
                stderr="",
                duration=0.0
            )
        
        # Build compilation command
        output_binary = output_binary or self._default_binary(source_file)
        cmd = [self.compiler, source_file, "-o", output_binary]
        
        # Add include paths for dacpp headers
        # Use project-internal translator directory
        compiler_file = Path(__file__).resolve()
        # PyDAC/pydac/core -> PyDAC/pydac -> PyDAC
        pydac_root = compiler_file.parent.parent.parent  # PyDAC
        # Use project-internal translator directory
        translator_root = pydac_root / "translator"  # PyDAC/translator
        include_dirs = [
            translator_root / "dpcppLib" / "include",  # DataReconstructor1.h, ParameterGeneration.h
            translator_root / "dacppLib" / "include",  # ReconTensor.h
            translator_root / "rewriter" / "include",  # dacInfo.h
            translator_root / "parser" / "include",  # Parser headers
        ]
        
        # Add all existing include directories
        for include_dir in include_dirs:
            if include_dir.exists():
                    cmd.extend(["-I" + str(include_dir)])
                    self.logger.debug(f"Added include path: {include_dir}")
        
        # Also add source file's directory as include path (for relative includes)
        source_dir = Path(source_file).parent.resolve()
        existing_include_dirs = [Path(d).resolve() for d in include_dirs if Path(d).exists()]
        if source_dir.resolve() not in existing_include_dirs:
            cmd.extend(["-I" + str(source_dir)])
            self.logger.debug(f"Added source directory as include path: {source_dir}")
        
        cmd.extend(compile_flags)
        
        # Execute compilation
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                # Cache compilation result
                shutil.copy2(output_binary, cached_binary)
            
            return CompilationResult(
                success=result.returncode == 0,
                source_file=source_file,
                binary_file=output_binary,
                stdout=result.stdout,
                stderr=result.stderr,
                duration=duration
            )
        except subprocess.TimeoutExpired:
            raise TimeoutError(
                f"Compilation timeout: exceeded 10 minutes for '{source_file}'.\n"
                f"  This may indicate:\n"
                f"  1. The source file is very large or complex\n"
                f"  2. The compiler is experiencing issues\n"
                f"  3. System resources are constrained\n"
                f"  Consider:\n"
                f"  - Optimizing the source code\n"
                f"  - Checking system resources (CPU, memory, disk)\n"
                f"  - Reviewing compilation flags for optimization"
            )
        except Exception as e:
            raise RuntimeError(
                f"Compilation failed for '{source_file}': {e}\n"
                f"  Please check:\n"
                f"  1. The source code compiles correctly\n"
                f"  2. All required headers and libraries are available\n"
                f"  3. Compilation flags are appropriate\n"
                f"  4. The compiler has necessary permissions\n"
                f"  Compiler output:\n{result.stderr if 'result' in locals() else 'N/A'}"
            )
    
    def _default_binary(self, source_file: str) -> str:
        """
        Generate default binary filename based on source file
        
        Args:
            source_file: Source file path
            
        Returns:
            Default binary file path (same directory, same name without extension)
        """
        path = Path(source_file)
        return str(path.parent / path.stem)

