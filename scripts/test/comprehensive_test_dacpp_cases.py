#!/usr/bin/env python3
"""
Comprehensive DACPP test case testing tool

This tool will:
1. Analyze code structure
2. Translate test cases using PyDAC
3. Validate translation results
4. Compile translated code
5. Run compiled programs 
6. Generate detailed reports
"""

import sys
import json
import argparse
import time
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import os

# Add pydac to path
script_dir = Path(__file__).parent.absolute()
pydac_dir = script_dir.parent.absolute()
dacpp_dir = pydac_dir.parent.absolute()

# Add paths in correct order
if str(dacpp_dir) not in sys.path:
    sys.path.insert(0, str(dacpp_dir))
if str(pydac_dir) not in sys.path:
    sys.path.insert(0, str(pydac_dir))

# Try to import
try:
    from pydac.analyzer import CodeAnalyzer
    from pydac import PyDAC
except ImportError as e:
    print(f"Error: Failed to import PyDAC: {e}")
    print("   Please ensure PyDAC is installed: pip install -e .")
    sys.exit(1)


def find_dacpp_test_cases(test_dir: Path) -> List[Path]:
    """Find all DACPP test cases"""
    test_cases = []
    for test_file in test_dir.rglob("*.dac.cpp"):
        if "tmp" not in str(test_file):
            test_cases.append(test_file)
    return sorted(test_cases)


def test_analysis(analyzer: CodeAnalyzer, test_file: Path) -> Dict[str, Any]:
    """Test code analysis"""
    try:
        analysis = analyzer.analyze(str(test_file))
        return {
            "status": "success",
            "shell_count": len(analysis.shells),
            "calc_count": len(analysis.calcs),
            "expression_count": len(analysis.expressions),
            "shells": [s.name for s in analysis.shells],
            "calcs": [c.name for c in analysis.calcs]
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


def test_translation_direct(test_file: Path, mode: str = "usm", translator_path: Optional[str] = None) -> Dict[str, Any]:
    """Translate using dacpp translator directly (without PyDAC)"""
    if translator_path is None:
        translator_path = os.environ.get("DACPP_TRANSLATOR")
        if not translator_path:
            return {
                "status": "skip",
                "reason": "Translator path not found (DACPP_TRANSLATOR not set)"
            }
    
    if not os.path.exists(translator_path):
        return {
            "status": "error",
            "error": f"Translator not found: {translator_path}"
        }
    
    try:
        # Generate output filename (same naming convention as PyDAC)
        # Actual format: {stem}_sycl_{mode}.cpp (e.g., DFT.dac_sycl_usm.cpp)
        output_file = test_file.parent / f"{test_file.stem}_sycl_{mode}.cpp"
        
        start_time = time.time()
        
        # Build command (simulate PyDAC translation command)
        cmd = [translator_path, str(test_file)]
        
        # Add include paths using project-internal translator directory
        script_file = Path(__file__).resolve()
        # scripts -> PyDAC
        pydac_root = script_file.parent.parent
        translator_root = pydac_root / "translator"  # PyDAC/translator
        include_dirs = [
            translator_root / "dpcppLib" / "include",
            translator_root / "dacppLib" / "include",
            translator_root / "rewriter" / "include",
            translator_root / "parser" / "include",
        ]
        for include_dir in include_dirs:
            if include_dir.exists():
                cmd.extend(["--extra-arg-before=-I" + str(include_dir)])
        
        # Add system include paths (obtained from gcc, same as engine.py)
        system_include_paths = []
        try:
            result = subprocess.run(
                ['gcc', '-E', '-x', 'c++', '-', '-v'],
                input='',
                capture_output=True,
                text=True,
                timeout=5
            )
            # Parse include paths
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
                    # Only add standard system paths
                    if path.startswith('/usr') and os.path.exists(path):
                        system_include_paths.append(path)
        except Exception:
            # Fallback to common paths
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
        
        # Add mode parameter (required!)
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
            # Default to usm
            cmd.append("--mode=usm")
        
        # Execute translation
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        duration = time.time() - start_time
        
        # Check if output file exists
        if result.returncode == 0 and output_file.exists():
            return {
                "status": "success",
                "success": True,
                "output_file": str(output_file),
                "duration": duration,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
        else:
            return {
                "status": "failed",
                "success": False,
                "output_file": None,
                "duration": duration,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
                "error": f"Translation failed with return code {result.returncode}"
            }
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "error": "Translation timeout",
            "error_type": "TimeoutError"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }


def test_translation(translator: Optional[PyDAC], test_file: Path, mode: str = "usm") -> Dict[str, Any]:
    """Test translation"""
    if translator is None:
        return {
            "status": "skip",
            "reason": "Translator not available"
        }
    try:
        result = translator.translate(str(test_file), mode=mode)
        return {
            "status": "success" if result.success else "failed",
            "success": result.success,
            "output_file": str(result.output_file) if result.output_file else None,
            "duration": result.duration,
            "warnings": result.warnings,
            "errors": result.errors,
            "warning_count": len(result.warnings),
            "error_count": len(result.errors)
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }


def test_validation(translator: Optional[PyDAC], test_file: Path) -> Dict[str, Any]:
    """Test validation"""
    if translator is None:
        return {
            "status": "skip",
            "reason": "Translator not available"
        }
    try:
        is_valid, errors = translator.validate_file(str(test_file))
        return {
            "status": "success",
            "is_valid": is_valid,
            "errors": errors,
            "error_count": len(errors)
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


def test_compilation(translator: Optional[PyDAC], translated_file: str) -> Dict[str, Any]:
    """Test compilation"""
    if translator is None:
        return {
            "status": "skip",
            "reason": "Translator not available"
        }
    if not translated_file:
        return {
            "status": "skip",
            "reason": "No translated file"
        }
    try:
        result = translator.Compile(translated_file)
        return {
            "status": "success" if result.success else "failed",
            "success": result.success,
            "binary_file": str(result.binary_file) if result.binary_file else None,
            "duration": result.duration,
            "stderr": result.stderr,
            "error_count": len(result.stderr.split('\n')) if result.stderr else 0
        }
    except TimeoutError as e:
        return {
            "status": "error",
            "error": f"Compilation timeout: {str(e)}",
            "error_type": "TimeoutError"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }


def get_default_input(test_name: str) -> Optional[str]:
    """Get default input data for test cases"""
    # imageAdjustment requires width and height
    # Use smaller values (10x10) to ensure program completes in reasonable time
    # 100x100 would cause program to run too long
    test_lower = test_name.lower()
    if "imageadjustment" in test_lower:
        return "10\n10\n"
    # Other programs that may need input can be added here
    return None


def get_test_timeout(test_name: str, default_timeout: float = 5.0) -> float:
    """Get timeout for test cases"""
    test_lower = test_name.lower()
    
    # Some programs require longer runtime
    # imageAdjustment uses smaller input (10x10), should complete within 30 seconds
    # mdp's Buffer mode may require longer time
    long_running_tests = ["stencil", "waveequation", "imageadjustment", "jacobi", "liuliang", "mdp"]
    for test in long_running_tests:
        if test in test_lower:
            return 30.0  # 30 second timeout
    
    return default_timeout


def should_skip_execution(test_name: str) -> bool:
    """Determine if execution should be skipped"""
    # Some test cases may run too long or have issues, can skip execution
    skip_tests = []  # Currently not skipping any tests
    test_lower = test_name.lower()
    for test in skip_tests:
        if test in test_lower:
            return True
    return False


def test_execution(translator: Optional[PyDAC], binary_file: str, timeout: Optional[float] = None, input_data: Optional[str] = None) -> Dict[str, Any]:
    """Test execution"""
    if translator is None:
        return {
            "status": "skip",
            "reason": "Translator not available"
        }
    if not binary_file:
        return {
            "status": "skip",
            "reason": "No binary file"
        }
    # If no timeout provided, set default timeout (5 seconds) to avoid infinite waiting
    if timeout is None:
        timeout = 5.0
    try:
        result = translator.run(binary_file, timeout=timeout, input_data=input_data)
        return {
            "status": "success" if result.success else "failed",
            "success": result.success,
            "return_code": result.return_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration": result.duration,
            "command": result.command
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }


def get_file_stats(file_path: Path) -> Dict[str, Any]:
    """Get file statistics"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
        
        return {
            "total_lines": len(lines),
            "code_lines": len([l for l in lines if l.strip() and not l.strip().startswith('//')]),
            "comment_lines": len([l for l in lines if l.strip().startswith('//')]),
            "blank_lines": len([l for l in lines if not l.strip()]),
            "file_size_bytes": file_path.stat().st_size,
            "file_size_kb": round(file_path.stat().st_size / 1024, 2)
        }
    except Exception as e:
        return {"error": str(e)}


def comprehensive_test(test_file: Path, analyzer: CodeAnalyzer, translator: Optional[PyDAC], 
                       skip_translation: bool = False, skip_execution: bool = False,
                       timeout: Optional[float] = None, show_output: bool = False,
                       compare_with_direct: bool = False) -> Dict[str, Any]:
    """Comprehensive test for a single test case"""
    result = {
        "file": str(test_file),
        "name": test_file.stem,
        "directory": str(test_file.parent),
        "timestamp": datetime.now().isoformat(),
        "file_stats": get_file_stats(test_file),
        "analysis": None,
        "validation": None,
        "translation_usm": None,
        "translation_buffer": None,
        "compilation_usm": None,
        "compilation_buffer": None,
        "execution_usm": None,
        "execution_buffer": None,
        "summary": {}
    }
    
    # Calculate total steps: analysis(1) + validation(1) + translate USM(1) + compile USM(1) + run USM(1) + translate Buffer(1) + compile Buffer(1) + run Buffer(1) = 8
    if not skip_translation and not skip_execution:
        step_count = 8
    elif skip_execution:
        step_count = 4  # analysis + validation + translate USM + translate Buffer
    else:
        step_count = 2  # analysis + validation
    current_step = 1
    
    # 1. Code analysis
    print(f"  [{current_step}/{step_count}] Analyzing code structure...", end=" ", flush=True)
    current_step += 1
    analysis_result = test_analysis(analyzer, test_file)
    result["analysis"] = analysis_result
    if analysis_result["status"] == "success":
        print("OK")
    else:
        print(f"FAIL: {analysis_result.get('error', 'Unknown error')}")
    
    # 2. Validation
    print(f"  [{current_step}/{step_count}] Validating code...", end=" ", flush=True)
    current_step += 1
    validation_result = test_validation(translator, test_file)
    result["validation"] = validation_result
    if validation_result["status"] == "success":
        if validation_result["is_valid"]:
            print("OK Valid")
        else:
            print(f"WARN: {validation_result['error_count']} errors found")
    else:
        print(f"FAIL: {validation_result.get('error', 'Unknown error')}")
    
    # 3. Translation test
    if not skip_translation:
        print(f"  [{current_step}/{step_count}] Translating (USM)...", end=" ", flush=True)
        current_step += 1
        translation_usm = test_translation(translator, test_file, mode="usm")
        result["translation_usm"] = translation_usm
        
        # If comparison enabled, also translate using direct translator
        translation_direct_usm = None
        if compare_with_direct and translation_usm.get("success"):
            translator_path = os.environ.get("DACPP_TRANSLATOR") or (translator.engine.translator_path if translator and hasattr(translator, 'engine') else None)
            if translator_path:
                translation_direct_usm = test_translation_direct(test_file, mode="usm", translator_path=translator_path)
                result["translation_direct_usm"] = translation_direct_usm
        
        if translation_usm["status"] == "success" and translation_usm.get("success"):
            print("OK")
            if compare_with_direct and translation_direct_usm:
                if translation_direct_usm.get("success"):
                    print(f"    Direct translator: OK")
                    # Compare file contents
                    pydac_file = Path(translation_usm.get("output_file"))
                    direct_file = Path(translation_direct_usm.get("output_file"))
                    if pydac_file.exists() and direct_file.exists():
                        try:
                            with open(pydac_file, 'r', encoding='utf-8') as f:
                                pydac_content = f.read()
                            with open(direct_file, 'r', encoding='utf-8') as f:
                                direct_content = f.read()
                            if pydac_content == direct_content:
                                print(f"    File contents: OK Identical")
                            else:
                                # Calculate differences
                                pydac_lines = pydac_content.split('\n')
                                direct_lines = direct_content.split('\n')
                                diff_count = sum(1 for a, b in zip(pydac_lines, direct_lines) if a != b)
                                print(f"    File contents: WARN Differences ({diff_count} lines differ, PyDAC: {len(pydac_lines)} lines, Direct: {len(direct_lines)} lines)")
                                if show_output:
                                    print(f"    PyDAC output file: {pydac_file}")
                                    print(f"    Direct translator output file: {direct_file}")
                        except Exception as e:
                            print(f"    File comparison error: {e}")
                else:
                    print(f"    Direct translator: FAIL {translation_direct_usm.get('error', 'Failed')}")
            
            # 4. Compile USM
            if not skip_execution:
                print(f"  [{current_step}/{step_count}] Compiling (USM)...", end=" ", flush=True)
                current_step += 1
                output_file = translation_usm.get("output_file")
                # Wait for filesystem sync
                import time
                time.sleep(0.1)
                # Ensure file path is string and file exists
                if output_file and Path(output_file).exists():
                    compilation_usm = test_compilation(translator, str(output_file))
                    result["compilation_usm"] = compilation_usm
                    if compilation_usm.get("success"):
                        print("OK")
                        
                        # 5. Run USM
                        if should_skip_execution(test_file.name):
                            print(f"  [{current_step}/{step_count}] Running (USM)... [SKIP] Skipped (known issue)", flush=True)
                            current_step += 1
                            result["execution_usm"] = {
                                "status": "skip",
                                "reason": "Skipped due to known performance issues"
                            }
                        else:
                            print(f"  [{current_step}/{step_count}] Running (USM)...", end=" ", flush=True)
                            current_step += 1
                            binary_file = compilation_usm.get("binary_file")
                            if binary_file and Path(binary_file).exists():
                                # Provide default input for programs that need it
                                input_data = get_default_input(test_file.name)
                                # If no timeout provided, set appropriate timeout based on test name
                                test_timeout = timeout if timeout is not None else get_test_timeout(test_file.name)
                                execution_usm = test_execution(translator, str(binary_file), test_timeout, input_data)
                                result["execution_usm"] = execution_usm
                                if execution_usm.get("success"):
                                    print(f"OK (return code: {execution_usm.get('return_code')})")
                                    if show_output and execution_usm.get("stdout"):
                                        stdout = execution_usm.get("stdout", "").strip()
                                        if stdout:
                                            print(f"    Output:\n{stdout}")
                                else:
                                    print(f"FAIL (return code: {execution_usm.get('return_code')})")
                                    if show_output:
                                        stdout = execution_usm.get("stdout", "").strip()
                                        stderr = execution_usm.get("stderr", "").strip()
                                        if stdout:
                                            print(f"    Stdout:\n{stdout}")
                                        if stderr:
                                            print(f"    Stderr:\n{stderr}")
                            else:
                                print(f"WARN: Executable file not found: {binary_file}")
                                result["execution_usm"] = {
                                    "status": "skip",
                                    "reason": f"Binary file not found: {binary_file}"
                                }
                    else:
                        print(f"FAIL: {compilation_usm.get('error', 'Compilation failed')}")
                else:
                    print(f"WARN: Translated file not found: {output_file}")
                    result["compilation_usm"] = {
                        "status": "skip",
                        "reason": f"Translated file not found: {output_file}"
                    }
        else:
            print(f"FAIL: {translation_usm.get('error', 'Translation failed')}")
        
        print(f"  [{current_step}/{step_count}] Translating (Buffer)...", end=" ", flush=True)
        current_step += 1
        translation_buffer = test_translation(translator, test_file, mode="buffer")
        result["translation_buffer"] = translation_buffer
        
        # If comparison enabled, also translate using direct translator
        translation_direct_buffer = None
        if compare_with_direct and translation_buffer.get("success"):
            translator_path = os.environ.get("DACPP_TRANSLATOR") or (translator.engine.translator_path if translator and hasattr(translator, 'engine') else None)
            if translator_path:
                translation_direct_buffer = test_translation_direct(test_file, mode="buffer", translator_path=translator_path)
                result["translation_direct_buffer"] = translation_direct_buffer
        
        if translation_buffer["status"] == "success" and translation_buffer.get("success"):
            print("OK")
            if compare_with_direct and translation_direct_buffer:
                if translation_direct_buffer.get("success"):
                    print(f"    Direct translator: OK")
                    # Compare file contents
                    pydac_file = Path(translation_buffer.get("output_file"))
                    direct_file = Path(translation_direct_buffer.get("output_file"))
                    if pydac_file.exists() and direct_file.exists():
                        try:
                            with open(pydac_file, 'r', encoding='utf-8') as f:
                                pydac_content = f.read()
                            with open(direct_file, 'r', encoding='utf-8') as f:
                                direct_content = f.read()
                            if pydac_content == direct_content:
                                print(f"    File contents: OK Identical")
                            else:
                                # Calculate differences
                                pydac_lines = pydac_content.split('\n')
                                direct_lines = direct_content.split('\n')
                                diff_count = sum(1 for a, b in zip(pydac_lines, direct_lines) if a != b)
                                print(f"    File contents: WARN Differences ({diff_count} lines differ, PyDAC: {len(pydac_lines)} lines, Direct: {len(direct_lines)} lines)")
                                if show_output:
                                    print(f"    PyDAC output file: {pydac_file}")
                                    print(f"    Direct translator output file: {direct_file}")
                        except Exception as e:
                            print(f"    File comparison error: {e}")
                else:
                    print(f"    Direct translator: FAIL {translation_direct_buffer.get('error', 'Failed')}")
            
            # Compile Buffer
            if not skip_execution:
                print(f"  [{current_step}/{step_count}] Compiling (Buffer)...", end=" ", flush=True)
                current_step += 1
                output_file = translation_buffer.get("output_file")
                # Wait for filesystem sync
                import time
                time.sleep(0.1)
                # Ensure file path is string and file exists
                if output_file and Path(output_file).exists():
                    compilation_buffer = test_compilation(translator, str(output_file))
                    result["compilation_buffer"] = compilation_buffer
                    if compilation_buffer.get("success"):
                        print("OK")
                        
                        # Run Buffer
                        if should_skip_execution(test_file.name):
                            print(f"  [{current_step}/{step_count}] Running (Buffer)... [SKIP] Skipped (known issue)", flush=True)
                            current_step += 1
                            result["execution_buffer"] = {
                                "status": "skip",
                                "reason": "Skipped due to known performance issues"
                            }
                        else:
                            print(f"  [{current_step}/{step_count}] Running (Buffer)...", end=" ", flush=True)
                            binary_file = compilation_buffer.get("binary_file")
                            if binary_file and Path(binary_file).exists():
                                # Provide default input for programs that need it
                                input_data = get_default_input(test_file.name)
                                # If no timeout provided, set appropriate timeout based on test name
                                test_timeout = timeout if timeout is not None else get_test_timeout(test_file.name)
                                execution_buffer = test_execution(translator, str(binary_file), test_timeout, input_data)
                                result["execution_buffer"] = execution_buffer
                                if execution_buffer.get("success"):
                                    print(f"OK (return code: {execution_buffer.get('return_code')})")
                                    if show_output and execution_buffer.get("stdout"):
                                        stdout = execution_buffer.get("stdout", "").strip()
                                        if stdout:
                                            print(f"    Output:\n{stdout}")
                                else:
                                    print(f"FAIL (return code: {execution_buffer.get('return_code')})")
                                    if show_output:
                                        stdout = execution_buffer.get("stdout", "").strip()
                                        stderr = execution_buffer.get("stderr", "").strip()
                                        if stdout:
                                            print(f"    Stdout:\n{stdout}")
                                        if stderr:
                                            print(f"    Stderr:\n{stderr}")
                            else:
                                print(f"WARN: Executable file not found: {binary_file}")
                                result["execution_buffer"] = {
                                    "status": "skip",
                                    "reason": f"Binary file not found: {binary_file}"
                                }
                    else:
                        print(f"FAIL: {compilation_buffer.get('error', 'Compilation failed')}")
                else:
                    print(f"WARN: Translated file not found: {output_file}")
                    result["compilation_buffer"] = {
                        "status": "skip",
                        "reason": f"Translated file not found: {output_file}"
                    }
        else:
            print(f"FAIL: {translation_buffer.get('error', 'Translation failed')}")
    else:
        print(f"  [{current_step}-{step_count}/{step_count}] Skipping translation tests")
    
    # Generate summary
    result["summary"] = {
        "analysis_success": analysis_result.get("status") == "success",
        "validation_success": validation_result.get("status") == "success" and validation_result.get("is_valid", False),
        "translation_usm_success": result["translation_usm"] and result["translation_usm"].get("success", False) if result["translation_usm"] else None,
        "translation_buffer_success": result["translation_buffer"] and result["translation_buffer"].get("success", False) if result["translation_buffer"] else None,
        "compilation_usm_success": result["compilation_usm"] and result["compilation_usm"].get("success", False) if result["compilation_usm"] else None,
        "compilation_buffer_success": result["compilation_buffer"] and result["compilation_buffer"].get("success", False) if result["compilation_buffer"] else None,
        "execution_usm_success": result["execution_usm"] and result["execution_usm"].get("success", False) if result["execution_usm"] else None,
        "execution_buffer_success": result["execution_buffer"] and result["execution_buffer"].get("success", False) if result["execution_buffer"] else None,
        "overall_success": (
            analysis_result.get("status") == "success" and
            validation_result.get("status") == "success" and
            (skip_translation or (
                (result["translation_usm"] and result["translation_usm"].get("success", False)) or
                (result["translation_buffer"] and result["translation_buffer"].get("success", False))
            ))
        )
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive DACPP test case testing tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--test-dir",
        type=str,
        default=None,
        help="Test case directory (default: auto-detect)"
    )
    
    parser.add_argument(
        "--test",
        type=str,
        default=None,
        help="Test specific test case (e.g., jacobi1.0)"
    )
    
    parser.add_argument(
        "--skip-translation",
        action="store_true",
        help="Skip translation tests (only analysis and validation)"
    )
    
    parser.add_argument(
        "--skip-execution",
        action="store_true",
        help="Skip compilation and execution tests (only translation)"
    )
    
    parser.add_argument(
        "--run-timeout",
        type=float,
        default=None,
        help="Execution timeout (seconds)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON report file path (default: result/test_report.json)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed information"
    )
    
    parser.add_argument(
        "--show-output",
        action="store_true",
        help="Show program execution output (stdout/stderr)"
    )
    
    parser.add_argument(
        "--compare-with-direct",
        action="store_true",
        help="Compare PyDAC translation results with direct dacpp translator results"
    )
    
    
    args = parser.parse_args()
    
    # Determine test directory
    if args.test_dir:
        test_dir = Path(args.test_dir).resolve()
    else:
        script_dir = Path(__file__).parent.absolute()
        pydac_dir = script_dir.parent.absolute()
        dacpp_dir = pydac_dir.parent.absolute()
        # Try multiple possible paths (including PyDAC project translator/tests)
        possible_paths = [
            pydac_dir / "translator" / "tests",  # PyDAC project translator/tests (highest priority)
            dacpp_dir / "dacpp" / "clang" / "tools" / "translator" / "tests",
            dacpp_dir / "clang" / "tools" / "translator" / "tests",
            dacpp_dir / "translator" / "tests",
        ]
        test_dir = None
        for path in possible_paths:
            if path.exists():
                test_dir = path
                break
        
        if test_dir is None:
            # Use the first path as default and show error
            test_dir = possible_paths[0]
            print(f"Error: Test directory does not exist: {test_dir}")
            print(f"   Tried paths:")
            for path in possible_paths:
                print(f"     - {path} ({'exists' if path.exists() else 'not found'})")
            print(f"   Please use --test-dir to specify the correct path")
            sys.exit(1)
    
    # Determine output directory (default: result)
    script_dir = Path(__file__).parent.absolute()
    pydac_dir = script_dir.parent.absolute()
    default_result_dir = pydac_dir / "result"
    default_result_dir.mkdir(parents=True, exist_ok=True)
    
    # Set default output path if not specified
    if args.output is None:
        args.output = str(default_result_dir / "test_report.json")
    else:
        # If relative path is provided, ensure it goes to result directory
        output_path = Path(args.output)
        if not output_path.is_absolute():
            # Relative path - put it in result directory
            args.output = str(default_result_dir / output_path.name)
    
    
    # Initialize
    print("=" * 60)
    print("Comprehensive DACPP Test Case Testing Tool")
    print("=" * 60)
    print(f"Test directory: {test_dir}")
    print(f"Skip translation: {args.skip_translation}")
    print(f"Skip execution: {args.skip_execution}")
    if args.run_timeout:
        print(f"Run timeout: {args.run_timeout} seconds")
    print(f"Output directory: {default_result_dir}")
    print(f"Report will be saved to: {args.output}")
    print()
    
    analyzer = CodeAnalyzer()
    
    # Try to initialize translator (if available)
    translator = None
    try:
        translator = PyDAC(verbose=args.verbose)
        print("Translator available")
    except RuntimeError as e:
        if args.skip_translation:
            print("WARN: Translator not available, but --skip-translation is set, continuing...")
        else:
            print(f"WARN: Translator not available: {e}")
            print("   Use --skip-translation to skip translation tests")
            args.skip_translation = True
    print()
    
    # Find test cases
    if args.test:
        test_file = test_dir / args.test / f"{Path(args.test).stem}.dac.cpp"
        if not test_file.exists():
            test_files = list(test_dir.glob(f"{args.test}/*.dac.cpp"))
            if test_files:
                test_file = test_files[0]
            else:
                print(f"Error: Test case not found: {args.test}")
                sys.exit(1)
        test_cases = [test_file]
    else:
        test_cases = find_dacpp_test_cases(test_dir)
    
    if not test_cases:
        print(f"Error: No test cases found in {test_dir}")
        sys.exit(1)
    
    print(f"Found {len(test_cases)} test cases")
    print()
    
    # Run tests
    results = []
    start_time = time.time()
    
    for i, test_file in enumerate(test_cases, 1):
        print(f"[{i}/{len(test_cases)}] Testing: {test_file.name}")
        print("-" * 60)
        
        result = comprehensive_test(
            test_file, 
            analyzer, 
            translator, 
            skip_translation=args.skip_translation,
            skip_execution=args.skip_execution,
            timeout=args.run_timeout,
            show_output=args.show_output,
            compare_with_direct=args.compare_with_direct
        )
        results.append(result)
        
        # Show summary
        summary = result["summary"]
        print(f"  Summary: ", end="")
        if summary["overall_success"]:
            print("All passed")
        else:
            parts = []
            if summary["analysis_success"]:
                parts.append("Analysis[OK]")
            if summary["validation_success"]:
                parts.append("Validation[OK]")
            if summary["translation_usm_success"]:
                parts.append("USM[OK]")
            if summary["translation_buffer_success"]:
                parts.append("Buffer[OK]")
            print(" ".join(parts) if parts else "Failed")
        print()
    
    total_time = time.time() - start_time
    
    # Generate report
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    total = len(results)
    analysis_success = sum(1 for r in results if r["summary"]["analysis_success"])
    validation_success = sum(1 for r in results if r["summary"]["validation_success"])
    translation_usm_success = sum(1 for r in results if r["translation_usm"] and r["translation_usm"].get("success", False))
    translation_buffer_success = sum(1 for r in results if r["translation_buffer"] and r["translation_buffer"].get("success", False))
    compilation_usm_success = sum(1 for r in results if r["compilation_usm"] and r["compilation_usm"].get("success", False))
    compilation_buffer_success = sum(1 for r in results if r["compilation_buffer"] and r["compilation_buffer"].get("success", False))
    execution_usm_success = sum(1 for r in results if r["execution_usm"] and r["execution_usm"].get("success", False))
    execution_buffer_success = sum(1 for r in results if r["execution_buffer"] and r["execution_buffer"].get("success", False))
    overall_success = sum(1 for r in results if r["summary"]["overall_success"])
    
    print(f"Total: {total} test cases")
    print(f"Code analysis: {analysis_success}/{total} successful")
    print(f"Code validation: {validation_success}/{total} successful")
    if not args.skip_translation:
        print(f"USM translation: {translation_usm_success}/{total} successful")
        print(f"Buffer translation: {translation_buffer_success}/{total} successful")
        if not args.skip_execution:
            print(f"USM compilation: {compilation_usm_success}/{total} successful")
            print(f"Buffer compilation: {compilation_buffer_success}/{total} successful")
            print(f"USM execution: {execution_usm_success}/{total} successful")
            print(f"Buffer execution: {execution_buffer_success}/{total} successful")
    print(f"Overall success: {overall_success}/{total}")
    print(f"Total time: {total_time:.2f} seconds")
    print()
    
    # Output JSON report
    if args.output:
        report = {
            "test_dir": str(test_dir),
            "total_cases": total,
            "timestamp": datetime.now().isoformat(),
            "total_time": total_time,
            "summary": {
                "analysis_success": analysis_success,
                "validation_success": validation_success,
                "translation_usm_success": translation_usm_success,
                "translation_buffer_success": translation_buffer_success,
                "compilation_usm_success": compilation_usm_success if not args.skip_execution else 0,
                "compilation_buffer_success": compilation_buffer_success if not args.skip_execution else 0,
                "execution_usm_success": execution_usm_success if not args.skip_execution else 0,
                "execution_buffer_success": execution_buffer_success if not args.skip_execution else 0,
                "overall_success": overall_success
            },
            "results": results
        }
        
        # Ensure output directory exists
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"Report saved to: {args.output}")

        # For visualization, run: python3 scripts/advanced_visualization.py result/test_report.json


if __name__ == "__main__":
    main()

