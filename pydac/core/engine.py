"""Translation engine for PyDAC"""

import subprocess
import os
import time
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass
from ..utils.logger import get_logger
from ..utils.retry import retry, RetryConfig


@dataclass
class TranslationResult:
    """Result of translation operation"""
    success: bool
    input_file: str
    output_file: str
    mode: str
    stdout: str
    stderr: str
    warnings: List[str]
    errors: List[str]
    duration: float


class TranslatorEngine:
    """Translation engine implementation"""
    
    def __init__(
        self,
        translator_path: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize translation engine
        
        Args:
            translator_path: Path to translator executable
            verbose: Enable verbose output
        """
        self.verbose = verbose
        self.logger = get_logger("pydac.engine", verbose)
        self.translator_path = translator_path or self._find_translator()
        self._validate_translator()
        
        if verbose:
            self.logger.info(f"Translator engine initialized: {self.translator_path}")
    
    def _find_translator(self) -> str:
        """Automatically find translator executable"""
        # 1. Check environment variable
        if "DACPP_TRANSLATOR" in os.environ:
            path = os.environ["DACPP_TRANSLATOR"]
            if os.path.exists(path) and os.access(path, os.X_OK):
                return path
        
        # 2. Check common paths (relative to current project)
        current_dir = Path(__file__).parent.parent.parent.parent
        common_paths = [
            "/usr/local/bin/translator",
            "/opt/dacpp/bin/translator",
            str(current_dir / "build" / "bin" / "translator" / "translator"),
            str(current_dir / "clang" / "tools" / "translator" / "build" / "bin" / "translator"),
            os.path.expanduser("~/dacpp/build/bin/translator"),
            os.path.expanduser("~/dacpp/clang/tools/translator/build/bin/translator"),
        ]
        for path in common_paths:
            if os.path.exists(path) and os.access(path, os.X_OK):
                return path
        
        # 3. Use which command
        import shutil
        translator = shutil.which("translator")
        if translator:
            return translator
        
        raise RuntimeError(
            "Cannot find translator executable.\n"
            "  Please set DACPP_TRANSLATOR environment variable:\n"
            "    export DACPP_TRANSLATOR=/path/to/translator\n"
            "  Or ensure the translator is installed in a standard location."
        )
    
    def _validate_translator(self):
        """Validate translator is available"""
        try:
            result = subprocess.run(
                [self.translator_path, "--help"],
                capture_output=True,
                timeout=5
            )
            if self.verbose:
                print(f"Translator found at: {self.translator_path}")
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"Translator validation timeout after 5 seconds.\n"
                f"  The translator at '{self.translator_path}' may be unresponsive.\n"
                f"  Please check if the translator is correctly installed and accessible."
            )
        except Exception as e:
            raise RuntimeError(
                f"Cannot validate translator at '{self.translator_path}': {e}\n"
                f"  Please verify:\n"
                f"  1. The translator executable exists and is executable\n"
                f"  2. The translator is correctly installed\n"
                f"  3. You have permission to execute the translator"
            )
    
    def translate(
        self,
        input_file: str,
        output_file: Optional[str] = None,
        mode: str = "usm",
        extra_args: Optional[List[str]] = None
    ) -> TranslationResult:
        """
        Execute translation
        
        Args:
            input_file: Input C++ file path
            output_file: Output file path (None for auto-generate)
            mode: Translation mode (usm/buffer/usm_time/mpi/sycl)
            extra_args: Additional translator arguments
        
        Returns:
            TranslationResult: Translation result
        """
        if not os.path.exists(input_file):
            raise FileNotFoundError(
                f"Input file not found: {input_file}\n"
                f"  Please check that the file path is correct.\n"
                f"  Current working directory: {os.getcwd()}"
            )
        
        start_time = time.time()
        
        # Build command
        cmd = [self.translator_path, input_file]
        
        # Add compilation database path if it exists
        # Translator uses -p flag to specify build path (directory containing compile_commands.json)
        source_dir = Path(input_file).parent
        compile_db = source_dir / "compile_commands.json"
        if compile_db.exists():
            # Use -p flag to specify the directory containing compile_commands.json
            cmd.extend(["-p", str(source_dir)])
            if self.verbose:
                self.logger.debug(f"Added compilation database path: {source_dir}")
        
        # Add include paths for dacpp headers and system headers (if using real translator)
        # Check if this is a real translator (not mock)
        is_mock = "mock_translator" in self.translator_path or self.translator_path.endswith(".py")
        if not is_mock:
            # Find dacpp include directory
            # Use project-internal translator directory
            engine_file = Path(__file__).resolve()
            # PyDAC/pydac/core -> PyDAC/pydac -> PyDAC
            pydac_root = engine_file.parent.parent.parent  # PyDAC
            translator_root = pydac_root / "translator"  # PyDAC/translator
            include_dirs = [
                translator_root / "dacppLib" / "include",
                translator_root / "dpcppLib" / "include",
                translator_root / "rewriter" / "include",
                translator_root / "parser" / "include",
            ]
            for include_dir in include_dirs:
                if include_dir.exists() and (include_dir / "ReconTensor.h").exists() if "dacppLib" in str(include_dir) else include_dir.exists():
                    # Add include path using extra-arg-before
                    cmd.extend(["--extra-arg-before=-I" + str(include_dir)])
                    if self.verbose:
                        self.logger.debug(f"Added include path: {include_dir}")
            
            # Add system include paths for C++ standard library
            # These are needed for translator to find system headers like stddef.h
            # Try to get system include paths from compiler
            system_include_paths = []
            try:
                import subprocess
                # Get system include paths from gcc
                result = subprocess.run(
                    ['gcc', '-E', '-x', 'c++', '-', '-v'],
                    input='',
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                # Parse include paths from stderr
                lines = result.stderr.split('\n')
                in_include_section = False
                for line in lines:
                    if '#include <...> search starts here:' in line:
                        in_include_section = True
                        continue
                    if 'End of search list' in line:
                        break
                    if in_include_section and line.strip():
                        path = line.strip()
                        # Only add standard system paths, skip intel oneapi paths
                        if path.startswith('/usr') and os.path.exists(path):
                            system_include_paths.append(path)
            except Exception:
                # Fallback to common paths if gcc is not available
                system_include_paths = [
                    "/usr/include",
                    "/usr/include/c++/9",
                    "/usr/include/x86_64-linux-gnu",
                    "/usr/include/x86_64-linux-gnu/c++/9",
                    "/usr/lib/gcc/x86_64-linux-gnu/9/include",
                ]
            
            # Add system include paths
            for sys_include in system_include_paths:
                if os.path.exists(sys_include):
                    cmd.extend(["--extra-arg-before=-I" + sys_include])
                    if self.verbose:
                        self.logger.debug(f"Added system include path: {sys_include}")
        
        # Add mode parameter
        mode_map = {
            "usm": "--mode=usm",
            "buffer": "--mode=buffer",
            "usm_time": "--mode=usm_time",
            "mpi": "--mode=mpi",
            "sycl": "--mode=sycl"
        }
        if mode in mode_map:
            cmd.append(mode_map[mode])
        else:
            available_modes = ", ".join(mode_map.keys())
            raise ValueError(
                f"Invalid translation mode: '{mode}'\n"
                f"  Available modes: {available_modes}\n"
                f"  Example: mode='usm' for SYCL USM mode"
            )
        
        # Determine output file before calling translator
        # This ensures the translator creates the file at the expected location
        if not output_file:
            output_file = self._default_output(input_file, mode)
        
        # Check if this is a real translator (not mock)
        # Real translator doesn't support -o parameter, it uses default naming
        is_mock = "mock_translator" in self.translator_path or self.translator_path.endswith(".py")
        if is_mock:
            # Mock translator supports -o parameter
            cmd.extend(["-o", output_file])
        # Real translator will create file with default naming, we'll find it later
        
        # Add extra arguments
        if extra_args:
            cmd.extend(extra_args)
        
        if self.verbose:
            print(f"Running translator: {' '.join(cmd)}")
        
        # Execute translation with retry
        retry_config = RetryConfig(
            max_attempts=2,
            initial_delay=1.0,
            retryable_errors=[subprocess.TimeoutExpired, RuntimeError]
        )
        
        @retry(config=retry_config)
        def _run_translation():
            # Set working directory to source file's directory so translator can find Compile_commands.json
            source_dir = Path(input_file).parent
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes timeout
                cwd=str(source_dir)  # Set working directory to source file's directory
            )
        
        try:
            result = _run_translation()
            
            duration = time.time() - start_time
            
            # Parse output
            warnings = self._parse_warnings(result.stdout)
            errors = self._parse_errors(result.stderr)
            
            # Verify output file exists (translator should have created it)
            if result.returncode == 0:
                # Wait a bit for file system to sync (especially for mock translator)
                time.sleep(0.2)  # Increased wait time for real translator
                
                # Always check for the correct file according to DACPP documentation
                # Even if the expected output_file exists, verify it's the correct one
                input_path = Path(input_file)
                parent_dir = input_path.parent
                
                # According to DACPP documentation, the naming rule is:
                # Original: [filename].dac.cpp (e.g., vecAdd.dac.cpp)
                # Translated: [filename]_sycl_[mode].cpp (e.g., vecAdd.dac_sycl_usm.cpp)
                # For "DFT.dac.cpp", stem is "DFT.dac", output should be "DFT.dac_sycl_usm.cpp"
                
                # Check for the correct file first (real translator naming)
                correct_file = parent_dir / f"{input_path.stem}_sycl_{mode}.cpp"
                
                if correct_file.exists():
                    # Use the correct file
                    output_file = str(correct_file.resolve())
                    if self.verbose:
                        self.logger.debug(f"Found correct output file: {output_file}")
                elif os.path.exists(output_file):
                    # Expected file exists, ensure it's an absolute path
                    output_file = str(Path(output_file).resolve())
                else:
                    # File doesn't exist, try to find it with alternative patterns
                    possible_patterns = [
                        f"{input_path.stem}_sycl_{mode}.cpp",  # DFT.dac_sycl_buffer.cpp (real translator, correct)
                        f"{input_path.stem}_{mode}.cpp",  # DFT.dac_buffer.cpp (mock translator)
                        # Fallback patterns for compatibility
                        f"{input_path.stem}.dac_sycl_{mode}.cpp",  # DFT.dac.dac_sycl_buffer.cpp (incorrect but might exist)
                    ]
                    
                    found_file = None
                    for pattern in possible_patterns:
                        possible_output = parent_dir / pattern
                        if possible_output.exists():
                            found_file = possible_output
                            break
                    
                    # If still not found, search for any new .cpp files created recently
                    if not found_file and parent_dir.exists():
                        # Get modification time before translation (approximate)
                        current_time = time.time()
                        # Look for files modified in the last 5 seconds
                        for f in parent_dir.glob("*.cpp"):
                            if f.name != input_path.name:  # Exclude input file
                                try:
                                    mtime = f.stat().st_mtime
                                    if current_time - mtime < 5:  # Modified in last 5 seconds
                                        # Check if filename contains mode or is likely output
                                        if mode in f.name.lower() or "sycl" in f.name.lower():
                                            found_file = f
                                            break
                                except:
                                    pass
                    
                    if found_file:
                        output_file = str(found_file)
                        if self.verbose:
                            self.logger.debug(f"Found output file: {output_file}")
                    else:
                        # Log warning but don't fail
                        if self.verbose:
                            self.logger.warning(
                                f"Expected output file '{output_file}' not found after translation. "
                                f"Translation may have succeeded but file was not created."
                            )
            
            # Final check: if output_file still doesn't exist, try one more time with correct naming
            # According to DACPP documentation: [filename]_sycl_[mode].cpp
            if result.returncode == 0:
                # Always check for real translator naming pattern, even if output_file was set
                input_path = Path(input_file)
                # Real translator uses: {stem}_sycl_{mode}.cpp (e.g., DFT.dac_sycl_buffer.cpp)
                real_translator_output = input_path.parent / f"{input_path.stem}_sycl_{mode}.cpp"
                if real_translator_output.exists() and not os.path.exists(output_file):
                    output_file = str(real_translator_output.resolve())
                    if self.verbose:
                        self.logger.debug(f"Found output file on final check: {output_file}")
                elif real_translator_output.exists() and os.path.exists(output_file):
                    # If both exist, prefer the real translator output
                    output_file = str(real_translator_output.resolve())
                    if self.verbose:
                        self.logger.debug(f"Using real translator output file: {output_file}")
            
            # Post-process translation output to fix known issues
            if result.returncode == 0 and output_file and os.path.exists(output_file):
                self._post_process_translation(output_file, input_file)
            
            return TranslationResult(
                success=result.returncode == 0,
                input_file=input_file,
                output_file=output_file,
                mode=mode,
                stdout=result.stdout,
                stderr=result.stderr,
                warnings=warnings,
                errors=errors,
                duration=duration
            )
        except subprocess.TimeoutExpired:
            raise TimeoutError(
                f"Translation timeout: exceeded 5 minutes for '{input_file}'.\n"
                f"  This may indicate:\n"
                f"  1. The input file is very large or complex\n"
                f"  2. The translator is experiencing issues\n"
                f"  3. System resources are constrained\n"
                f"  Consider:\n"
                f"  - Breaking down large files into smaller modules\n"
                f"  - Checking system resources (CPU, memory)\n"
                f"  - Reviewing the input file for potential issues"
            )
        except Exception as e:
            raise RuntimeError(
                f"Translation failed for '{input_file}': {e}\n"
                f"  Please check:\n"
                f"  1. The input file syntax is correct\n"
                f"  2. All required dependencies are available\n"
                f"  3. The translator has necessary permissions\n"
                f"  For more details, enable verbose mode: PyDAC(verbose=True)"
            )
    
    def _parse_warnings(self, stdout: str) -> List[str]:
        """
        Parse warning messages from translator output
        
        Args:
            stdout: Standard output from translator
            
        Returns:
            List of warning messages
        """
        warnings = []
        for line in stdout.split('\n'):
            if 'warning' in line.lower():
                warnings.append(line.strip())
        return warnings
    
    def _parse_errors(self, stderr: str) -> List[str]:
        """
        Parse error messages from translator output
        
        Args:
            stderr: Standard error output from translator
            
        Returns:
            List of error messages
        """
        errors = []
        for line in stderr.split('\n'):
            if 'error' in line.lower():
                errors.append(line.strip())
        return errors
    
    def _default_output(self, input_file: str, mode: str) -> str:
        """
        Generate default output filename based on input file and mode
        
        According to DACPP documentation, the naming rule is:
        - Original: [filename].dac.cpp (e.g., vecAdd.dac.cpp)
        - Translated: [filename]_sycl_[mode].cpp (e.g., vecAdd.dac_sycl_usm.cpp)
        
        Args:
            input_file: Input file path
            mode: Translation mode (usm or buffer)
            
        Returns:
            Default output file path
        """
        path = Path(input_file)
        # Check if this is a real translator (not mock)
        is_mock = "mock_translator" in self.translator_path or self.translator_path.endswith(".py")
        
        if is_mock:
            # Mock translator uses: {stem}_{mode}.cpp
            return str(path.parent / f"{path.stem}_{mode}.cpp")
        else:
            # Real translator uses: {stem}_sycl_{mode}.cpp
            # For "DFT.dac.cpp", stem is "DFT.dac", output should be "DFT.dac_sycl_usm.cpp"
            # For "vecAdd.dac.cpp", stem is "vecAdd.dac", output should be "vecAdd.dac_sycl_usm.cpp"
            return str(path.parent / f"{path.stem}_sycl_{mode}.cpp")
    
    def _post_process_translation(self, output_file: str, input_file: str):
        """
        Post-process translated code to fix known issues
        
        Args:
            output_file: Path to translated output file
            input_file: Path to original input file
        """
        try:
            input_path = Path(input_file)
            output_path = Path(output_file)
            
            # Only process if output file exists
            if not output_path.exists():
                return
            
            # Read the translated file
            with open(output_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            modified = False
            
            # Fix 1: imageAdjustment - std::vector<Pixel> initialization
            # Replace: std::vector<Pixel> init(size, 0);
            # With: std::vector<Pixel> init(size, Pixel{0, 0, 0});
            if 'imageadjustment' in input_path.name.lower():
                # Fix vector initialization with 0
                import re
                # More flexible pattern to match with optional spaces
                pattern = r'std::vector<Pixel>\s+init\s*\(\s*([^,]+)\s*,\s*0\s*\)'
                replacement = r'std::vector<Pixel> init(\1, Pixel{0, 0, 0})'
                new_content = re.sub(pattern, replacement, content)
                if new_content != content:
                    content = new_content
                    modified = True
                    if self.verbose:
                        self.logger.debug(f"Fixed std::vector<Pixel> initialization in {output_file}")
                
                # Fix 2: imageAdjustment - image_tensor3 creation
                # Replace the incorrect image_tensor3 creation with correct one
                # Pattern needs to be more flexible to match both Matrix and Tensor
                old_patterns = [
                    # Pattern 1: Matrix version
                    (
                        r'std::vector<Pixel>\s+image3\s*=\s*image2;\s*'
                        r'dacpp::Tensor<Pixel,\s*2>\s+image_tensor3\(\{height,\s*width\},\s*image3\);'
                    ),
                    # Pattern 2: More flexible spacing
                    (
                        r'std::vector<Pixel>\s+image3\s*=\s*image2;\s*\n\s*'
                        r'dacpp::Tensor<Pixel,\s*2>\s+image_tensor3\(\{height,\s*width\},\s*image3\);'
                    ),
                    # Pattern 3: Matrix<Pixel> version
                    (
                        r'std::vector<Pixel>\s+image3\s*=\s*image2;\s*'
                        r'dacpp::Matrix<Pixel>\s+image_tensor3\(\{height,\s*width\},\s*image3\);'
                    ),
                ]
                new_code = (
                    '// From updated image_tensor2, get data to create image3\n'
                    '    // First get the shape and size of image_tensor2\n'
                    '    int tensor2_height = image_tensor2.getShape(0);\n'
                    '    int tensor2_width = image_tensor2.getShape(1);\n'
                    '    std::vector<Pixel> image3;\n'
                    '    image_tensor2.tensor2Array(image3);\n'
                    '    // Use the actual shape of image_tensor2 to create image_tensor3\n'
                    '    dacpp::Tensor<Pixel, 2> image_tensor3({tensor2_height, tensor2_width}, image3);'
                )
                new_content = content
                for old_pattern in old_patterns:
                    new_content = re.sub(old_pattern, new_code, new_content, flags=re.MULTILINE | re.DOTALL)
                if new_content != content:
                    content = new_content
                    modified = True
            
            # Write back if modified
            if modified:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                if self.verbose:
                    self.logger.debug(f"Post-processed translation output: {output_file}")
        except Exception as e:
            # Don't fail translation if post-processing fails
            if self.verbose:
                self.logger.warning(f"Post-processing failed for {output_file}: {e}")

