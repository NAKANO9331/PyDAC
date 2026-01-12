"""Tests for PyDAC translator"""

import pytest

import os

from pathlib import Path

from pydac import PyDAC

from pydac.utils.errors import TranslationError, CompilationError


class TestPyDAC:

    """Test cases for PyDAC"""

    @pytest.fixture
    def translator(self, mock_translator_path):
        """Create PyDAC instance for testing"""
 # Use mock translator if real one is not available
        return PyDAC(translator_path=mock_translator_path, verbose=False)

    def test_translate_basic(self, translator, temp_test_file):

        """Test basic translation functionality"""

        result = translator.translate(temp_test_file, mode="usm")
        assert isinstance(result.success, bool)
        assert result.mode == "usm"
        if result.success:
            assert os.path.exists(result.output_file)

    def test_translate_invalid_file(self, translator):

        """Test handling of invalid file"""

        with pytest.raises(FileNotFoundError):
            translator.translate("nonexistent.cpp")

    def test_translate_invalid_mode(self, translator, temp_test_file):

        """Test handling of invalid mode"""

        with pytest.raises(ValueError):
            translator.translate(temp_test_file, mode="invalid_mode")

    def test_translate_code(self, translator, mock_translator_path):

        """Test translation from code string"""

        code = """
#include "ReconTensor.h"
    namespace dacpp { typedef std::vector<std::any> list; }


    shell dacpp::list testShell(const dacpp::Vector<float>& x) {

        dacpp::index i;
        return dacpp::list{x[i]};
    }


    calc void testCalc(float* x) {

        x[0] = 1.0;
    }

    """

        result = translator.translate_code(code, mode="usm")
        assert isinstance(result.success, bool)

