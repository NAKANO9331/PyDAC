#!/usr/bin/env python3
"""
PyDAC Performance Visualization Main Entry Script

Generate performance analysis visualization charts from test result JSON files
"""

import sys
import argparse
from pathlib import Path

# Add project path
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from visualization.dacpp_visualizer import DACPPVisualizer
from visualization.unit_test_visualizer import UnitTestVisualizer
from visualization.dsl_visualizer import DSLVisualizer


def main():
    parser = argparse.ArgumentParser(
        description="PyDAC Performance Visualization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Generate PyDAC test visualizations (most important!)
  python visualization/generate_visualizations.py --type dsl --input result/dsl_tests/pydac_test_report.json
  
  # Generate DACPP test visualizations
  python visualization/generate_visualizations.py --type dacpp --input result/dacpp_tests/test_report.json
  
  # Generate unit test visualizations
  python visualization/generate_visualizations.py --type unit --input result/unit_tests/unit_test_results.json
  
  # Auto-detect type
  python visualization/generate_visualizations.py --input result/dsl_tests/pydac_test_report.json
        """
    )
    
    parser.add_argument(
        '--type',
        type=str,
        choices=['dacpp', 'unit', 'dsl', 'auto'],
        default='auto',
        help='Visualization type: dacpp (DACPP tests), unit (unit tests), dsl (DSL tests, most important), auto (auto-detect)'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input JSON test report file path'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: auto-select based on type)'
    )
    
    args = parser.parse_args()
    
    # Check input file
    input_path = Path(args.input)
    
    # If relative path, try multiple possible locations
    if not input_path.is_absolute():
        searched_paths = []
        # Try current directory first
        if input_path.exists():
            input_path = input_path.resolve()
        else:
            searched_paths.append(input_path.resolve())
            # Try relative to project root
            project_input = project_root / input_path
            if project_input.exists():
                input_path = project_input
            else:
                searched_paths.append(project_input)
                # Try in result directory
                result_input = project_root / "result" / input_path.name
                if result_input.exists():
                    input_path = result_input
                else:
                    searched_paths.append(result_input)
                    # Try in result subdirectories (dacpp_tests, unit_tests)
                    for subdir in ["dacpp_tests", "unit_tests"]:
                        subdir_input = project_root / "result" / subdir / input_path.name
                        if subdir_input.exists():
                            input_path = subdir_input
                            break
                        searched_paths.append(subdir_input)
                    else:
                        # Try in scripts/result directory
                        scripts_result_input = project_root / "scripts" / "result" / input_path.name
                        if scripts_result_input.exists():
                            input_path = scripts_result_input
                        else:
                            searched_paths.append(scripts_result_input)
    
    if not input_path.exists():
        print(f"Error: Input file does not exist: {args.input}")
        print(f"   Searched locations:")
        for path in searched_paths:
            print(f"     - {path}")
        sys.exit(1)
    
    # Auto-detect type
    if args.type == 'auto':
        # Try to determine from filename or content
        if 'unit' in input_path.name.lower() or 'pytest' in input_path.name.lower():
            args.type = 'unit'
        elif 'dsl' in input_path.name.lower():
            args.type = 'dsl'
        elif 'dacpp' in input_path.name.lower() or 'test_report' in input_path.name.lower():
            args.type = 'dacpp'
        else:
            # Read JSON content to determine
            import json
            try:
                with open(input_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Check test_type field (most accurate)
                    test_type = data.get('test_type', '')
                    if test_type == 'dsl':
                        args.type = 'dsl'
                    elif test_type == 'unit' or ('tests' in data and isinstance(data['tests'], list)):
                        args.type = 'unit'
                    elif 'results' in data and isinstance(data['results'], list):
                        # Further check: DSL results have mode field
                        if data['results'] and 'mode' in data['results'][0]:
                            args.type = 'dsl'
                        else:
                            args.type = 'dacpp'
                    else:
                        print("Warning: Unable to auto-detect type, defaulting to dsl")
                        args.type = 'dsl'
            except Exception as e:
                print(f"Warning: Failed to read JSON file: {e}, defaulting to dsl")
                args.type = 'dsl'
    
    print("=" * 60)
    print("PyDAC Performance Visualization Tool")
    print("=" * 60)
    print(f"Input file: {input_path}")
    print(f"Visualization type: {args.type}")
    print()
    
    try:
        if args.type == 'dacpp':
            # DACPP test visualization
            output_dir = args.output_dir or "result/dacpp_tests"
            visualizer = DACPPVisualizer(output_dir=output_dir)
            print(f"Output directory: {output_dir}")
            print()
            
            output_paths = visualizer.generate_all_visualizations(str(input_path))
            
            print("\nGenerated charts:")
            for key, path in output_paths.items():
                print(f"  - {key}: {path}")
        
        elif args.type == 'unit':
            # Unit test visualization
            output_dir = args.output_dir or "result/unit_tests"
            visualizer = UnitTestVisualizer(output_dir=output_dir)
            print(f"Output directory: {output_dir}")
            print()
            
            output_paths = visualizer.generate_all_visualizations(str(input_path))
            
            print("\nGenerated charts:")
            for key, path in output_paths.items():
                print(f"  - {key}: {path}")
        
        elif args.type == 'dsl':
            # DSL test visualization (most important test in the project)
            output_dir = args.output_dir or "result/dsl_tests"
            visualizer = DSLVisualizer(output_dir=output_dir)
            print(f"Output directory: {output_dir}")
            print("Warning: This is the most important test visualization in the project!")
            print()
            
            output_paths = visualizer.generate_all_visualizations(str(input_path))
            
            print("\nGenerated charts:")
            for key, path in output_paths.items():
                print(f"  - {key}: {path}")
        
        print("\nVisualization completed!")
        
    except Exception as e:
        print(f"\nError: Failed to generate visualizations: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
