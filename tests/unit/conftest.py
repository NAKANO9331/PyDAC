"""

Pytest configuration and fixtures for PyDAC tests

"""


import pytest

import os

import tempfile

import shutil

from pathlib import Path


# Get the mock translator path
MOCK_TRANSLATOR = Path(__file__).parent / "mock_translator.py"


def _find_real_translator():
    """Find real translator if available"""

    # Check environment variable
    translator_path = os.environ.get("DACPP_TRANSLATOR")
    if translator_path and os.path.exists(translator_path) and os.access(translator_path, os.X_OK):
        return translator_path

    # Check common paths
    common_paths = [
        "/usr/local/bin/translator",
        "/opt/dacpp/bin/translator",
    ]
    for path in common_paths:
        if os.path.exists(path) and os.access(path, os.X_OK):
            return path

    # Check which
    translator = shutil.which("translator")
    if translator:
        return translator

    return None


@pytest.fixture(scope="session", autouse=True)
def setup_mock_translator():

    """Setup mock translator for all tests if real translator is not available"""

    real_translator = _find_real_translator()

    if real_translator:
        # Real translator exists, use it
        yield
        return

    # No real translator found, use mock
    if MOCK_TRANSLATOR.exists() and MOCK_TRANSLATOR.is_file():
        # Set environment variable to use mock translator
        original_env = os.environ.get("DACPP_TRANSLATOR")
        os.environ["DACPP_TRANSLATOR"] = str(MOCK_TRANSLATOR.absolute())

        yield

        # Restore original environment
        if original_env:
            os.environ["DACPP_TRANSLATOR"] = original_env
        elif "DACPP_TRANSLATOR" in os.environ:
            del os.environ["DACPP_TRANSLATOR"]
    else:
        # Mock translator not found, but don't skip - let tests handle it
        yield


@pytest.fixture
def mock_translator_path():

    """Return path to mock translator or real translator"""

    # Check if user wants to force mock translator
    force_mock = os.environ.get("PYDAC_USE_MOCK", "").lower() in ("1", "true", "yes")

    if not force_mock:
        real_translator = _find_real_translator()
        if real_translator:
            return real_translator

    if MOCK_TRANSLATOR.exists() and MOCK_TRANSLATOR.is_file():
        return str(MOCK_TRANSLATOR.absolute())

    pytest.skip("No translator available (neither real nor mock)")


@pytest.fixture
def temp_test_file():

    """Create a temporary test file"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.dac.cpp', delete=False) as f:
        f.write("""
#include "ReconTensor.h"
#include <vector>
        namespace dacpp { typedef std::vector<std::any> list; }


        shell dacpp::list testShell(const dacpp::Vector<float>& x) {

        dacpp::index i;
        return dacpp::list{x[i]};
        }


        calc void testCalc(float* x) {

        x[0] = 1.0;
        }


        int main() {

        // Use std::vector to initialize dacpp::Vector (correct syntax)
        std::vector<float> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
        dacpp::Vector<float> x(data);
        testShell(x) <-> testCalc;
        return 0;
        }

        """)

        temp_file = f.name

        yield temp_file

    # Cleanup
    if os.path.exists(temp_file):
        os.unlink(temp_file)

    # Also clean up any generated output files
    base_name = os.path.splitext(temp_file)[0]
    for mode in ['usm', 'buffer', 'sycl']:
        output_file = f"{base_name}_{mode}.cpp"
        if os.path.exists(output_file):
            os.unlink(output_file)
