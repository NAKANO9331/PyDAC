# PyDAC Command Reference

## Complete Workflow

### 1. Environment Setup

#### Virtual Environment Setup

```bash
# Create virtual environment
python3 -m venv pydac_env

# Activate virtual environment
source pydac_env/bin/activate  # Linux/macOS
```

#### Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install core dependencies
pip install -e .

# Install optional dependencies (select as needed)
pip install -e .[dev]      # Development tools (pytest, black, flake8, mypy)
pip install -e .[plot]     # Visualization tools (matplotlib, seaborn)
pip install -e .[docs]     # Documentation tools (sphinx)
pip install -e .[all]      # All optional dependencies

# Verify installation
python3 -c "import pydac; print('PyDAC imported successfully')"
pytest --version
```

#### Configure Translator

```bash
# Configure Translator
source scripts/setup/setup_translator.sh set

# Check configuration status
scripts/setup/setup_translator.sh status
```

#### Environment Variables (Optional)

```bash
# Set PYTHONPATH (if needed)
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Add project bin directory to PATH (if CLI is installed)
export PATH="$PATH:$(pwd)/bin"

# Verify environment setup
echo $PYTHONPATH
echo $PATH
```

### 2. Running Tests

#### Unit Tests

```bash
# Run all unit tests
pytest tests/unit/ -v

# Run specific test
pytest tests/test_translator.py -v

# Test specific functionality
pytest tests/ -k "translate" -v

# Generate JSON report for visualization
mkdir -p result/unit_tests
pytest tests/unit/ --json-report --json-report-file=result/unit_tests/unit_test_results.json
```

#### DSL Tests (Recommended, Most Important Tests)

**Test Case List (12 cases)**:
- `matMul` - Matrix multiplication
- `decay` - Decay chain
- `oddeven` - Odd-even sort
- `stencil` - Stencil computation
- `jacobi` - Jacobi iteration
- `DFT` - Discrete Fourier Transform
- `waveEquation` - Wave equation
- `FOuLa` - Fourier transform
- `imageAdjustment` - Image adjustment
- `liuliang` - Flow calculation
- `MDP` - Markov Decision Process
- `mandel` - Mandelbrot set

**Basic Commands**:

```bash
# Run all DSL test cases (12 cases) - Default: runs both USM and Buffer modes
# Output will be saved to result/dsl_tests/ by default
python3 scripts/test/test_pydac_dsl_cases.py

# Run specific test case (both modes by default)
python3 scripts/test/test_pydac_dsl_cases.py --test matMul

# Explicitly specify to run both modes
python3 scripts/test/test_pydac_dsl_cases.py --mode all

# Run only USM mode
python3 scripts/test/test_pydac_dsl_cases.py --mode usm

# Run only Buffer mode
python3 scripts/test/test_pydac_dsl_cases.py --mode buffer

# Translate only (skip compilation and execution)
python3 scripts/test/test_pydac_dsl_cases.py --skip-execution

# Save generated code
python3 scripts/test/test_pydac_dsl_cases.py --save-code

# Show detailed output
python3 scripts/test/test_pydac_dsl_cases.py --show-output --verbose
```

**Complete Parameter List**:

```bash
python3 scripts/test/test_pydac_dsl_cases.py [options]

Options:
  --test TEST_NAME          Specify test case name (e.g., matMul, jacobi, waveEquation, etc.)
  --mode {usm,buffer,all}   Translation mode: 'usm', 'buffer', or 'all' for both modes (default: 'all' - runs both USM and Buffer)
  --skip-execution          Skip compilation and execution, only translate
  --show-output             Show test execution output
  --save-code               Save generated code to tests/pydac/generated/
  --no-compare              Disable comparison with original test cases (comparison enabled by default)
  --test-dir PATH           Specify original test case directory (auto-detected by default)
  --strict-format           Use strict format comparison (exact format match, including whitespace)
  --verbose                 Show detailed logs
  --timeout SECONDS         Set timeout (seconds)
  --output FILE             Specify output report file (default: result/dsl_tests/pydac_test_report.json)
```

**Note**: 
- By default, the script runs tests in **both USM and Buffer modes** for comprehensive testing
- Each test will show:
  - **PyDAC Output**: Results from code generated via PyDAC DSL
  - **Direct Translator Output**: Results from directly calling the translator on original test cases
  - **Comparison results**: Side-by-side comparison of both outputs
- Use `--mode usm` or `--mode buffer` to run only a single mode

#### DACPP Tests

```bash
# Run all test cases
# Output will be saved to result/dacpp_tests/ by default
python3 scripts/test/comprehensive_test_dacpp_cases.py

# Test specific case
python3 scripts/test/comprehensive_test_dacpp_cases.py --test imageAdjustment1.0

# Analyze only (no translation)
python3 scripts/test/comprehensive_test_dacpp_cases.py --skip-translation

# Translate and compile only (no execution)
python3 scripts/test/comprehensive_test_dacpp_cases.py --skip-execution

# Show output
python3 scripts/test/comprehensive_test_dacpp_cases.py --show-output

# Compare with direct translator
python3 scripts/test/comprehensive_test_dacpp_cases.py --compare-with-direct

# Clear cache
rm -rf .pydac_cache
```

### 3. Visualization Analysis

#### DSL Test Visualization

```bash
# Generate DSL test visualization charts
python3 visualization/generate_visualizations.py --type dsl --input result/dsl_tests/pydac_test_report.json
```

#### DACPP Test Visualization

```bash
# Generate DACPP test visualization charts
python3 visualization/generate_visualizations.py --type dacpp --input result/dacpp_tests/test_report.json
```

#### Unit Test Visualization

```bash
# First generate JSON report (if not already generated)
pytest tests/unit/ --json-report --json-report-file=result/unit_tests/unit_test_results.json

# Generate visualization charts
python3 visualization/generate_visualizations.py --type unit --input result/unit_tests/unit_test_results.json
```

### 4. File Management

```bash
# View result files
ls -la result/

# Check file sizes
du -sh result/

# Manually clean old files (execute as needed)
find result/ -name "*.json" -mtime +7 -delete  # Delete JSON files older than 7 days
find result/ -name "*.png" -mtime +30 -delete  # Delete image files older than 30 days

# Clean test cache
rm -rf .pytest_cache
rm -rf .pydac_cache
```