# PyDAC

Python interface for DACPP source-to-source translator.

## Introduction

PyDAC is a Python interface for the DACPP (Data Associated Computing) source-to-source translator. It enables developers to convert DAC (Data Associated Computing) C++ code to various parallel computing backends, including SYCL USM, SYCL Buffer, MPI, and other modes, through a concise Python API.

> **Important Note**: PyDAC is the Python interface for DACPP. **This project includes a pre-built translator executable** (located in the `translator/bin/` directory) that can be used directly without additional setup or building.

## Project Structure

```
PyDAC/
├── pydac/                    # Main code directory
│   ├── core/                 # Core module (translator, compiler, engine)
│   ├── dsl/                  # DSL module (Shell, Calc, Expression)
│   ├── analyzer/             # Code analysis module
│   ├── tensor/               # Tensor module
│   └── utils/                # Utility module
├── translator/               # Translator related
│   ├── bin/                  # Translator executable
│   └── tests/                # DACPP test cases
├── tests/                    # Test cases
│   ├── unit/                 # Unit tests
│   └── pydac/                # PyDAC DSL test cases (12 test cases)
├── scripts/                  # Utility scripts
│   ├── test/                 # Test scripts
│   │   ├── test_pydac_dsl_cases.py      # DSL test runner (recommended)
│   │   └── comprehensive_test_dacpp_cases.py  # DACPP test runner
│   └── setup/                # Setup scripts
│       └── setup_translator.sh           # Translator setup script
├── visualization/            # Visualization modules (all unified here)
│   ├── dsl_visualizer.py     # DSL test visualization
│   ├── dacpp_visualizer.py   # DACPP test visualization
│   ├── unit_test_visualizer.py # Unit test visualization
│   ├── benchmark_visualizer.py # Performance benchmark visualization
│   └── generate_visualizations.py # Main visualization script
├── result/                   # Test results and visualizations (organized by type)
│   ├── dsl_tests/            # DSL test results and charts
│   ├── dacpp_tests/         # DACPP test results and charts
│   └── unit_tests/          # Unit test results and charts
├── pyproject.toml            # Project configuration and dependencies
└── README.md                 # This file
```

## Quick Links

- **[Command Reference](COMMAND_REFERENCE.md)** - Complete installation guide, usage instructions, and all available commands

## Requirements

- Python 3.8+
- NumPy >= 1.20.0 (core dependency)
- Optional: matplotlib, seaborn (for visualization)
- Optional: pytest, pytest-asyncio (for testing)

## Installation

### Quick Start

```bash
# 1. Clone the repository
git clone <repository-url>
cd PyDAC-main

# 2. Create and activate virtual environment
python3 -m venv pydac_env
source pydac_env/bin/activate  # Linux/macOS
# or
pydac_env\Scripts\activate     # Windows

# 3. Install PyDAC
pip install --upgrade pip
pip install -e .

# 4. Install optional dependencies (as needed)
pip install -e .[dev]      # Development tools (pytest, black, etc.)
pip install -e .[plot]     # Visualization tools (matplotlib, seaborn)
pip install -e .[docs]     # Documentation tools (sphinx)
pip install -e .[all]       # All optional dependencies

# 5. Configure translator
source scripts/setup/setup_translator.sh set
```

### Installation Options

The project uses `pyproject.toml` for dependency management. You can install:

- **Core package**: `pip install -e .` (includes numpy)
- **With dev tools**: `pip install -e .[dev]` (pytest, black, flake8, mypy)
- **With visualization**: `pip install -e .[plot]` (matplotlib, seaborn)
- **With documentation**: `pip install -e .[docs]` (sphinx, sphinx-rtd-theme)
- **Everything**: `pip install -e .[all]`

## Performance Optimization

PyDAC includes multiple performance optimization mechanisms:

- **Result Caching**: 4.68x - 5.87x speedup
- **Batch Processing**: 5.87x speedup
- **I/O Optimization**: 2.19x speedup
- **Async Operations**: Non-blocking concurrent processing
- **Memory Optimization**: Streaming processing for large files

## Testing and Visualization

### Test Coverage

PyDAC includes comprehensive test coverage:

- **Unit Tests**: Independent module testing (config, DSL, tensor, translator)
- **DSL Tests** (Recommended, Most Important): 12 test cases reimplemented using PyDAC DSL
  - Tests both USM and Buffer modes by default
  - Compares DSL-generated code with original test cases
  - Output saved to `result/dsl_tests/`
- **DACPP Tests**: Direct testing of original DACPP test cases
  - Code analysis, translation, compilation, and execution
  - Output saved to `result/dacpp_tests/`

### Visualization

Comprehensive performance analysis charts are available:

- All visualization modules unified in `visualization/` directory
- Results organized in `result/` subdirectories by test type
- Supports DSL, DACPP, and unit test visualizations

For detailed testing instructions and complete command reference, see [Command Reference](COMMAND_REFERENCE.md).

## Contact

For questions, please contact: nakano9331@gmail.com
