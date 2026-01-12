# PyDAC

Python interface for DACPP source-to-source translator.

## Introduction

PyDAC is a Python interface for the DACPP (Data Associated Computing) source-to-source translator. It enables developers to convert DAC (Data Associated Computing) C++ code to various parallel computing backends, including SYCL USM, SYCL Buffer, MPI, and other modes, through a concise Python API.

> **Important Note**: PyDAC is the Python interface for DACPP. **This project includes a pre-built translator executable** (located in the `translator/bin/` directory) that can be used directly without additional setup or building.

## Core Features

- **Concise Python API**: Easy-to-use interface, lowering the barrier to entry
- **Multi-Mode Translation**: Support for USM, Buffer, USM Time, MPI, and Standard SYCL modes
- **Code Generation**: DSL support (Shell and Calc definitions) and template-based code generation
- **Tensor Abstraction**: Seamless NumPy integration for data conversion
- **Automatic Compilation**: Intelligent compilation management with caching mechanism
- **Code Analysis**: Automatic structure analysis and verification
- **Performance Optimization**: Result caching (4.68x - 5.87x speedup), async operations, batch processing
- **Complete Toolchain**: From code translation to testing and visualization

## Project Structure

```
PyDAC/
├── pydac/                    # Main code directory
│   ├── core/                 # Core module (translator, compiler, engine)
│   ├── dsl/                  # DSL module (Shell, Calc, Expression)
│   ├── generator/            # Code generation module
│   ├── analyzer/             # Code analysis module
│   ├── tensor/               # Tensor module
│   └── utils/                # Utility module
├── translator/               # Translator related
│   ├── bin/                  # Translator executable
│   └── tests/                # DACPP test cases
├── tests/                    # Unit tests
│   ├── unit/                 # Unit tests
│   └── pydac/                # PyDAC DSL test cases
├── scripts/                  # Utility scripts
│   ├── test/                 # Test scripts
│   │   ├── test_pydac_dsl_cases.py      # PyDAC DSL test runner
│   │   └── comprehensive_test_dacpp_cases.py  # DACPP test runner
│   └── setup/                # Setup scripts
│       └── setup_translator.sh           # Translator setup script
├── visualization/            # Visualization modules
│   ├── dsl_visualizer.py     # DSL test visualization
│   ├── dacpp_visualizer.py   # DACPP test visualization
│   ├── unit_test_visualizer.py # Unit test visualization
│   └── generate_visualizations.py # Main visualization script
├── pyproject.toml            # Project configuration and dependencies
└── README.md                 # This file
```

## Documentation

For detailed usage instructions and reference documentation, please see:

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

## Performance

PyDAC includes multiple performance optimization mechanisms:

- **Result Caching**: 4.68x - 5.87x speedup
- **Batch Processing**: 5.87x speedup
- **I/O Optimization**: 2.19x speedup
- **Async Operations**: Non-blocking concurrent processing
- **Memory Optimization**: Streaming processing for large files

## Testing

PyDAC includes comprehensive test coverage:

- **28 test functions**: Covering all major features
- **Unit Tests**: Independent module testing (config, DSL, tensor, translator)
- **Integration Tests**: Complete workflow testing
- **Performance Tests**: Performance optimization verification

## Contact

For questions, please contact: nakano9331@gmail.com
