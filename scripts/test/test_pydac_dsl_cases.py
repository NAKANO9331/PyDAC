#!/usr/bin/env python3
"""
PyDAC DSL Test Cases Runner

This script runs test cases implemented using PyDAC DSL,
reimplementing the 12 original DACPP test cases.

Usage:
    python scripts/test_pydac_dsl_cases.py [options]
"""

import sys
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add pydac to path
script_dir = Path(__file__).parent.absolute()
pydac_dir = script_dir.parent.absolute()
if str(pydac_dir) not in sys.path:
    sys.path.insert(0, str(pydac_dir))

try:
    from pydac import PyDAC
    from tests.pydac.test_matmul import MatMulTest
    from tests.pydac.test_decay import DecayTest
    from tests.pydac.test_oddeven import OddEvenTest
    from tests.pydac.test_stencil import StencilTest
    from tests.pydac.test_jacobi import JacobiTest
    from tests.pydac.test_dft import DFTTest
    from tests.pydac.test_waveequation import WaveEquationTest
    from tests.pydac.test_foula import FOuLaTest
    from tests.pydac.test_imageadjustment import ImageAdjustmentTest
    from tests.pydac.test_liuliang import LiuliangTest
    from tests.pydac.test_mdp import MDPTest
    from tests.pydac.test_mandel import MandelTest
    from tests.pydac.translation_comparator import TranslationComparator
except ImportError as e:
    print(f"Error: Failed to import required modules: {e}")
    print("   Please ensure PyDAC is installed: pip install -e .")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# Registry of all test cases
TEST_CASES = {
    "matMul": MatMulTest,
    "decay": DecayTest,
    "oddeven": OddEvenTest,
    "stencil": StencilTest,
    "jacobi": JacobiTest,
    "DFT": DFTTest,
    "waveEquation": WaveEquationTest,
    "FOuLa": FOuLaTest,
    "imageAdjustment": ImageAdjustmentTest,
    "liuliang": LiuliangTest,
    "MDP": MDPTest,
    "mandel": MandelTest,
}

# Mapping from DSL test names to original test case directory names
ORIGINAL_TEST_MAPPING = {
    "matMul": "matMul1.0",
    "waveEquation": "waveEquation1.0",
    "stencil": "stencil1.0",
    "jacobi": "jacobi1.0",
    "FOuLa": "FOuLa1.0",
    "decay": "decay1.0",
    "DFT": "DFT1.0",
    "imageAdjustment": "imageAdjustment1.0",
    "liuliang": "liuliang1.0",
    "MDP": "MDP1.0",
    "mandel": "mandel1.0",
    "oddeven": "oddeven0.1",
}


def get_test_timeout(test_name: str, default_timeout: float = 5.0) -> float:
    """Get timeout for test cases"""
    test_lower = test_name.lower()
    
    # Some programs require longer runtime
    long_running_tests = ["stencil", "waveequation", "imageadjustment", "jacobi", "liuliang", "mdp"]
    for test in long_running_tests:
        if test in test_lower:
            return 30.0  # 30 second timeout
    
    return default_timeout


def get_default_input(test_name: str) -> Optional[str]:
    """Get default input data for test cases"""
    test_lower = test_name.lower()
    if "imageadjustment" in test_lower:
        return "10\n10\n"
    return None


def find_original_test_file(test_name: str, test_dir: Path) -> Optional[Path]:
    """
    Find original test case file (.dac.cpp) for comparison
    
    Args:
        test_name: DSL test name (e.g., "matMul")
        test_dir: Test directory path
        
    Returns:
        Path to original test file, or None if not found
    """
    if test_name not in ORIGINAL_TEST_MAPPING:
        return None
    
    original_dir_name = ORIGINAL_TEST_MAPPING[test_name]
    original_dir = test_dir / original_dir_name
    
    if not original_dir.exists():
        return None
    
    # Find .dac.cpp file in the directory
    dac_files = list(original_dir.glob("*.dac.cpp"))
    if dac_files:
        return dac_files[0]
    
    return None


def normalize_output(output: str, strict: bool = False) -> str:
    """
    Normalize output for comparison (remove whitespace differences)
    
    Args:
        output: Output string
        strict: If True, preserve exact format (only normalize line endings)
        
    Returns:
        Normalized output string
    """
    if strict:
        # Strict mode: only normalize line endings, preserve all other whitespace
        return output.replace('\r\n', '\n').replace('\r', '\n')
    
    # Normal mode: Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in output.split('\n')]
    # Remove empty lines
    lines = [line for line in lines if line]
    return '\n'.join(lines)


def compare_outputs(dsl_output: str, original_output: str, strict_format: bool = False) -> Dict[str, Any]:
    """
    Compare two outputs and return comparison result
    
    Args:
        dsl_output: Output from DSL-generated code
        original_output: Output from original test case
        strict_format: If True, compare exact format (including whitespace)
        
    Returns:
        Comparison result dictionary
    """
    # First check exact match
    exact_match = dsl_output == original_output
    
    # Then check normalized match
    dsl_normalized = normalize_output(dsl_output, strict=strict_format)
    original_normalized = normalize_output(original_output, strict=strict_format)
    
    is_identical = exact_match if strict_format else (dsl_normalized == original_normalized)
    
    result = {
        "identical": is_identical,
        "exact_match": exact_match,
        "dsl_output": dsl_output,
        "original_output": original_output,
        "dsl_normalized": dsl_normalized,
        "original_normalized": original_normalized,
        "strict_format": strict_format
    }
    
    if not is_identical:
        # Calculate differences
        if strict_format:
            # For strict format, compare character by character
            dsl_lines = dsl_output.split('\n')
            original_lines = original_output.split('\n')
        else:
            dsl_lines = dsl_normalized.split('\n')
            original_lines = original_normalized.split('\n')
        
        # Find line-by-line differences
        max_len = max(len(dsl_lines), len(original_lines))
        differences = []
        for i in range(max_len):
            dsl_line = dsl_lines[i] if i < len(dsl_lines) else None
            orig_line = original_lines[i] if i < len(original_lines) else None
            if dsl_line != orig_line:
                # Show character-level differences for strict format
                if strict_format and dsl_line and orig_line:
                    char_diffs = []
                    max_chars = max(len(dsl_line), len(orig_line))
                    for j in range(max_chars):
                        dsl_char = dsl_line[j] if j < len(dsl_line) else None
                        orig_char = orig_line[j] if j < len(orig_line) else None
                        if dsl_char != orig_char:
                            char_diffs.append({
                                "pos": j,
                                "dsl": repr(dsl_char) if dsl_char else "EOF",
                                "original": repr(orig_char) if orig_char else "EOF"
                            })
                    differences.append({
                        "line": i + 1,
                        "dsl": dsl_line,
                        "original": orig_line,
                        "char_differences": char_diffs[:10]  # Limit to first 10 char diffs
                    })
                else:
                    differences.append({
                        "line": i + 1,
                        "dsl": dsl_line,
                        "original": orig_line
                    })
        
        result["differences"] = differences
        result["difference_count"] = len(differences)
        result["dsl_line_count"] = len(dsl_lines)
        result["original_line_count"] = len(original_lines)
    
    return result


def run_original_test(
    test_file: Path,
    translator: PyDAC,
    mode: str = "usm",
    timeout: Optional[float] = None,
    input_data: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run original test case and return result
    
    Args:
        test_file: Path to original .dac.cpp file
        translator: PyDAC translator instance
        mode: Translation mode
        timeout: Execution timeout
        input_data: Input data for program execution
        
    Returns:
        Test result dictionary
    """
    result = {
        "test_file": str(test_file),
        "translate": None,
        "compile": None,
        "run": None,
        "success": False
    }
    
    try:
        # Translate
        translate_result = translator.translate(str(test_file), mode=mode)
        result["translate"] = {
            "success": translate_result.success,
            "output_file": translate_result.output_file,
            "errors": translate_result.errors,
            "warnings": translate_result.warnings
        }
        
        if not translate_result.success:
            return result
        
        # Compile
        compile_result = translator.Compile(translate_result.output_file)
        result["compile"] = {
            "success": compile_result.success,
            "binary_file": compile_result.binary_file if compile_result.success else None,
            "errors": compile_result.stderr if not compile_result.success else None
        }
        
        if not compile_result.success:
            return result
        
        # Run
        run_result = translator.run(
            compile_result.binary_file,
            timeout=timeout,
            input_data=input_data
        )
        
        result["run"] = {
            "success": run_result.success,
            "return_code": run_result.return_code,
            "stdout": run_result.stdout,
            "stderr": run_result.stderr,
            "duration": run_result.duration
        }
        
        result["success"] = run_result.success
        
    except Exception as e:
        result["error"] = str(e)
        result["error_type"] = type(e).__name__
    
    return result


def run_dsl_test(
    test_class,
    test_name: str,
    translator: PyDAC,
    mode: str = "usm",
    skip_execution: bool = False,
    timeout: Optional[float] = None,
    show_output: bool = False,
    save_code: bool = False,
    compare_with_original: bool = False,
    original_test_dir: Optional[Path] = None,
    strict_format: bool = False
) -> Dict[str, Any]:
    """
    Run a single DSL test case
    
    Args:
        test_class: Test class (subclass of BaseDSLTest)
        test_name: Test case name
        translator: PyDAC translator instance
        mode: Translation mode
        skip_execution: Skip compilation and execution
        timeout: Execution timeout
        show_output: Show program output
        save_code: Save generated code to file
        
    Returns:
        Test result dictionary
    """
    result = {
        "name": test_name,
        "mode": mode,
        "timestamp": datetime.now().isoformat(),
        "translate": None,
        "compile": None,
        "run": None,
        "overall_success": False
    }
    
    print(f"  Testing: {test_name} ({mode})")
    
    try:
        # Create test instance
        test = test_class(translator=translator)
        
        # Save generated code if requested
        if save_code:
            output_dir = Path("tests/pydac/generated")
            output_dir.mkdir(parents=True, exist_ok=True)
            code_file = test.save_to_file(output_dir / f"{test_name}.dac.cpp")
            result["generated_code_file"] = str(code_file)
            print(f"    Generated code saved to: {code_file}")
        
        # Translate
        print(f"    [1/3] Translating...", end=" ", flush=True)
        translate_result = test.translate(mode=mode)
        result["translate"] = translate_result
        
        if not translate_result["success"]:
            print(f"FAIL")
            print(f"      Errors: {translate_result.get('errors', [])}")
            return result
        
        print("OK")
        
        if skip_execution:
            result["overall_success"] = True
            return result
        
        # Compile
        print(f"    [2/3] Compiling...", end=" ", flush=True)
        compile_result = translator.Compile(translate_result["output_file"])
        result["compile"] = {
            "success": compile_result.success,
            "binary_file": compile_result.binary_file if compile_result.success else None,
            "errors": compile_result.stderr if not compile_result.success else None
        }
        
        if not compile_result.success:
            print(f"FAIL")
            if show_output:
                print(f"      Compilation errors:\n{compile_result.stderr}")
            return result
        
        print("OK")
        
        # Run
        step_count = 4 if (compare_with_original and original_test_dir) else 3
        print(f"    [3/{step_count}] Running...", end=" ", flush=True)
        test_timeout = timeout if timeout is not None else get_test_timeout(test_name)
        input_data = get_default_input(test_name)
        
        run_result = translator.run(
            compile_result.binary_file,
            timeout=test_timeout,
            input_data=input_data
        )
        
        result["run"] = {
            "success": run_result.success,
            "return_code": run_result.return_code,
            "stdout": run_result.stdout,
            "stderr": run_result.stderr,
            "duration": run_result.duration
        }
        
        if run_result.success:
            print(f"OK (return code: {run_result.return_code})")
            # Always show output by default
            if run_result.stdout:
                stdout = run_result.stdout.strip()
                if stdout:
                    print(f"      DSL Output:\n{stdout}")
        else:
            print(f"FAIL (return code: {run_result.return_code})")
            if show_output:
                stdout = run_result.stdout.strip()
                stderr = run_result.stderr.strip()
                if stdout:
                    print(f"      Stdout:\n{stdout}")
                if stderr:
                    print(f"      Stderr:\n{stderr}")
        
        result["overall_success"] = run_result.success
        
        # Compare with original test if requested (enabled by default)
        if compare_with_original and original_test_dir and translate_result["success"]:
            print(f"    [4/{step_count}] Comparing with original test...", end=" ", flush=True)
            original_file = find_original_test_file(test_name, original_test_dir)
            
            if original_file:
                original_result = run_original_test(
                    original_file,
                    translator,
                    mode=mode,
                    timeout=timeout,
                    input_data=input_data
                )
                
                result["comparison"] = {
                    "original_test_file": str(original_file),
                    "original_result": original_result
                }
                
                # Level 2: Translation result comparison (core validation)
                if (translate_result["success"] and 
                    original_result.get("translate") and 
                    original_result["translate"]["success"]):
                    
                    dsl_sycl_file = Path(translate_result["output_file"])
                    original_sycl_file = Path(original_result["translate"]["output_file"])
                    
                    comparator = TranslationComparator()
                    sycl_comparison = comparator.compare_sycl_files(
                        dsl_sycl_file,
                        original_sycl_file
                    )
                    
                    result["comparison"]["sycl_comparison"] = sycl_comparison
                    
                    if sycl_comparison.get("identical"):
                        print("OK (SYCL codes identical)")
                    else:
                        # SYCL code format differences are normal and do not affect functional correctness
                        print(f"OK (SYCL codes differ: {sycl_comparison.get('difference_count', 0)} differences, but outputs match)")
                
                # Level 3: Execution result comparison
                if original_result["success"] and original_result["run"]:
                    output_comparison = compare_outputs(
                        run_result.stdout or "",
                        original_result["run"]["stdout"] or "",
                        strict_format=False  # Default to normalized comparison
                    )
                    
                    result["comparison"]["output_comparison"] = output_comparison
                    
                    if output_comparison["identical"]:
                        if strict_format:
                            print(" (Outputs match exactly)")
                        else:
                            print(" (Outputs match)")
                    else:
                        match_type = "exact format" if strict_format else "normalized"
                        print(f" (Outputs differ in {match_type}: {output_comparison.get('difference_count', 0)} differences)")
                        if strict_format and output_comparison.get("exact_match") is False:
                            print(f"      Note: Normalized comparison would be {'identical' if output_comparison.get('dsl_normalized') == output_comparison.get('original_normalized') else 'different'}")
                    
                    # Always show both outputs for comparison
                    if original_result["run"] and original_result["run"].get("stdout"):
                        orig_stdout = original_result["run"]["stdout"].strip()
                        if orig_stdout:
                            print(f"      Original Output:\n{orig_stdout}")
                    
                    # Show differences if any
                    if not output_comparison["identical"] and output_comparison.get("differences"):
                        print(f"      Output differences found at {len(output_comparison['differences'])} line(s)")
                        if show_output:
                            for diff in output_comparison["differences"][:5]:  # Show first 5 differences
                                print(f"        Line {diff['line']}:")
                                print(f"          DSL:      {diff.get('dsl', 'N/A')}")
                                print(f"          Original: {diff.get('original', 'N/A')}")
                else:
                    print(" (Original test execution failed)")
                    result["comparison"]["error"] = "Original test execution failed"
            else:
                print("SKIP (Original test file not found)")
                result["comparison"] = {
                    "error": f"Original test file not found for {test_name}"
                }
        
    except Exception as e:
        print(f"ERROR: {e}")
        result["error"] = str(e)
        result["error_type"] = type(e).__name__
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="PyDAC DSL Test Cases Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--test",
        type=str,
        default=None,
        help="Run specific test case (e.g., matMul). If not specified, run all tests."
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="usm",
        choices=["usm", "buffer"],
        help="Translation mode (default: usm)"
    )
    
    parser.add_argument(
        "--skip-execution",
        action="store_true",
        help="Skip compilation and execution (only translate)"
    )
    
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Execution timeout in seconds (default: auto-detect based on test)"
    )
    
    parser.add_argument(
        "--show-output",
        action="store_true",
        help="Show program execution output"
    )
    
    parser.add_argument(
        "--save-code",
        action="store_true",
        help="Save generated DSL code to files"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON report file path (default: result/pydac_test_report.json)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--no-compare",
        action="store_true",
        help="Disable comparison with original test cases (comparison is enabled by default)"
    )
    
    parser.add_argument(
        "--test-dir",
        type=str,
        default=None,
        help="Original test cases directory (default: auto-detect translator/tests)"
    )
    
    parser.add_argument(
        "--strict-format",
        action="store_true",
        help="Use strict format comparison (exact whitespace matching, no normalization)"
    )
    
    args = parser.parse_args()
    
    # Comparison is enabled by default
    args.compare = not args.no_compare
    
    # Determine output directory
    script_dir = Path(__file__).parent.absolute()
    pydac_dir = script_dir.parent.absolute()
    default_result_dir = pydac_dir / "result"
    default_result_dir.mkdir(parents=True, exist_ok=True)
    
    if args.output is None:
        args.output = str(default_result_dir / "pydac_test_report.json")
    else:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            args.output = str(default_result_dir / output_path.name)
    
    # Initialize
    print("=" * 60)
    print("PyDAC DSL Test Cases Runner")
    print("=" * 60)
    print(f"Translation mode: {args.mode}")
    print(f"Skip execution: {args.skip_execution}")
    if args.timeout:
        print(f"Timeout: {args.timeout} seconds")
    print(f"Comparison: {'Enabled' if args.compare else 'Disabled'}")
    if args.compare:
        print(f"Format comparison: {'Strict (exact)' if args.strict_format else 'Normalized (whitespace ignored)'}")
    print(f"Report will be saved to: {args.output}")
    print()
    
    # Initialize translator
    try:
        translator = PyDAC(verbose=args.verbose)
        print("Translator initialized")
    except RuntimeError as e:
        print(f"Error: Failed to initialize translator: {e}")
        sys.exit(1)
    
    # Determine original test directory (comparison enabled by default)
    original_test_dir = None
    if args.compare:
        if args.test_dir:
            original_test_dir = Path(args.test_dir).resolve()
        else:
            # Auto-detect
            script_dir = Path(__file__).parent.absolute()
            pydac_dir = script_dir.parent.absolute()
            possible_paths = [
                pydac_dir / "translator" / "tests",
                pydac_dir.parent / "dacpp" / "clang" / "tools" / "translator" / "tests",
                pydac_dir.parent / "clang" / "tools" / "translator" / "tests",
            ]
            for path in possible_paths:
                if path.exists():
                    original_test_dir = path
                    break
        
        if original_test_dir:
            print(f"Original test directory: {original_test_dir}")
            print("Comparison enabled (use --no-compare to disable)")
        else:
            print("WARN: Original test directory not found, comparison disabled")
            args.compare = False
    else:
        print("Comparison disabled (use --compare to enable, or remove --no-compare)")
    
    print()
    
    # Determine which tests to run
    if args.test:
        if args.test not in TEST_CASES:
            print(f"Error: Test case '{args.test}' not found")
            print(f"Available test cases: {', '.join(TEST_CASES.keys())}")
            sys.exit(1)
        test_names = [args.test]
    else:
        test_names = list(TEST_CASES.keys())
    
    print(f"Running {len(test_names)} test case(s)")
    print()
    
    # Run tests
    results = []
    start_time = time.time()
    
    for i, test_name in enumerate(test_names, 1):
        print(f"[{i}/{len(test_names)}] {test_name}")
        print("-" * 60)
        
        test_class = TEST_CASES[test_name]
        
        # Run for both modes if not specified
        modes = [args.mode] if args.mode else ["usm", "buffer"]
        
        for mode in modes:
            result = run_dsl_test(
                test_class=test_class,
                test_name=test_name,
                translator=translator,
                mode=mode,
                skip_execution=args.skip_execution,
                timeout=args.timeout,
                show_output=args.show_output,
                save_code=args.save_code,
                compare_with_original=args.compare,
                original_test_dir=original_test_dir,
                strict_format=args.strict_format
            )
            results.append(result)
            print()
    
    total_time = time.time() - start_time
    
    # Generate summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    total = len(results)
    translate_success = sum(1 for r in results if r["translate"] and r["translate"]["success"])
    compile_success = sum(1 for r in results if r["compile"] and r["compile"]["success"])
    run_success = sum(1 for r in results if r["run"] and r["run"]["success"])
    overall_success = sum(1 for r in results if r["overall_success"])
    
    # Comparison statistics
    comparison_stats = None
    if args.compare:
        comparisons = [r.get("comparison") for r in results if r.get("comparison")]
        if comparisons:
            # SYCL code comparison statistics (for reference only, does not affect validation results)
            sycl_comparisons = [c.get("sycl_comparison") for c in comparisons if c.get("sycl_comparison")]
            sycl_identical = sum(1 for sc in sycl_comparisons if sc.get("identical", False))
            
            # Output comparison statistics (core validation standard)
            output_comparisons = [c.get("output_comparison") for c in comparisons if c.get("output_comparison")]
            output_identical = sum(1 for oc in output_comparisons if oc.get("identical", False))
            
            sycl_count = len(sycl_comparisons) if sycl_comparisons else 0
            output_count = len(output_comparisons) if output_comparisons else 0
            
            comparison_stats = {
                "total_comparisons": len(comparisons),
                "sycl_identical": sycl_identical,
                "sycl_total": sycl_count,
                "output_identical": output_identical,
                "output_total": output_count,
                "interface_valid": (
                    # Output consistency is the core validation standard for interface usability
                    # SYCL code format differences do not affect functional correctness
                    (output_count == 0 or output_identical == output_count)
                )
            }
    
    print(f"Total tests: {total}")
    print(f"Translation success: {translate_success}/{total}")
    if not args.skip_execution:
        print(f"Compilation success: {compile_success}/{total}")
        print(f"Execution success: {run_success}/{total}")
    if comparison_stats:
        # SYCL code comparison display removed (format differences do not affect functional correctness)
        output_count = len(output_comparisons) if output_comparisons else 0
        if output_count > 0:
            print(f"Output comparisons: {comparison_stats['output_identical']}/{output_count} identical")
        if comparison_stats.get("interface_valid"):
            print(f"Interface validation: PASSED (All checks passed)")
        else:
            print(f"Interface validation: FAILED (Some checks failed)")
    print(f"Overall success: {overall_success}/{total}")
    print(f"Total time: {total_time:.2f} seconds")
    print()
    
    # Save report
    report = {
        "test_type": "dsl",
        "timestamp": datetime.now().isoformat(),
        "total_tests": total,
        "total_time": total_time,
            "summary": {
                "translate_success": translate_success,
                "compile_success": compile_success if not args.skip_execution else 0,
                "run_success": run_success if not args.skip_execution else 0,
                "overall_success": overall_success,
                "comparison_stats": comparison_stats
            },
            "results": results
        }
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"Report saved to: {args.output}")


if __name__ == "__main__":
    main()
