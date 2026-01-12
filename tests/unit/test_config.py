"""Tests for configuration management"""


import pytest

import json

import os

import tempfile

from pathlib import Path

from unittest.mock import patch


from pydac.utils.config import PyDACConfig


class TestPyDACConfig:

    """Test PyDACConfig class"""

    def test_default_config(self):
        """Test default configuration"""

        config = PyDACConfig.default()

        assert config.Compiler == "dpcpp"
        assert config.cache_dir == "./.pydac_cache"
        assert config.default_mode == "usm"
        assert config.enable_cache is True
        assert config.parallel_jobs == 4
        assert len(config.default_flags) > 0

    def test_config_from_file_json(self):

        """Test loading configuration from JSON file"""

        config_data = {
        "translator_path": "/path/to/translator",
        "Compiler": "clang++",
        "cache_dir": "/tmp/cache",
        "default_mode": "buffer",
        "default_flags": ["-O3", "-std=c++17"],
        "output_dir": "/tmp/output",
        "keep_intermediate": True,
        "enable_cache": False,
        "parallel_jobs": 8
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            config = PyDACConfig.from_file(temp_path)

            assert config.translator_path == "/path/to/translator"
            assert config.Compiler == "clang++"
            assert config.cache_dir == "/tmp/cache"
            assert config.default_mode == "buffer"
            assert config.default_flags == ["-O3", "-std=c++17"]
            assert config.output_dir == "/tmp/output"
            assert config.keep_intermediate is True
            assert config.enable_cache is False
            assert config.parallel_jobs == 8
        finally:
            os.unlink(temp_path)

    def test_config_from_file_not_found(self):

        """Test loading configuration from non-existent file"""

        with pytest.raises(FileNotFoundError):
            PyDACConfig.from_file("/nonexistent/config.json")

    def test_config_from_file_unsupported_format(self):

        """Test loading configuration from unsupported format"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("test")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported config file format"):
                PyDACConfig.from_file(temp_path)
        finally:
            os.unlink(temp_path)

    def test_config_save_to_file_json(self):

        """Test saving configuration to JSON file"""

        config = PyDACConfig(
        translator_path="/path/to/translator",
        Compiler="dpcpp",
        cache_dir="/tmp/cache",
        default_mode="usm"
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            config.save_to_file(temp_path)

            # Verify file was created and contains correct data
            assert Path(temp_path).exists()

            with open(temp_path, 'r') as f:
                loaded_data = json.load(f)

            assert loaded_data["translator_path"] == "/path/to/translator"
            assert loaded_data["Compiler"] == "dpcpp"
            assert loaded_data["cache_dir"] == "/tmp/cache"
            assert loaded_data["default_mode"] == "usm"
        finally:
            os.unlink(temp_path)

    def test_config_from_env(self):

        """Test loading configuration from environment variables"""

        env_vars = {
        "DACPP_TRANSLATOR": "/env/translator",
        "DACPP_COMPILER": "clang++",
        "DACPP_CACHE_DIR": "/env/cache",
        "DACPP_DEFAULT_MODE": "buffer"
        }

        with patch.dict(os.environ, env_vars):
            config = PyDACConfig.from_env()

            assert config.translator_path == "/env/translator"
            assert config.Compiler == "clang++"
            assert config.cache_dir == "/env/cache"
            assert config.default_mode == "buffer"

    def test_config_from_env_partial(self):

        """Test loading configuration from partial environment variables"""

        env_vars = {
        "DACPP_TRANSLATOR": "/env/translator",
        "DACPP_COMPILER": "clang++"
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = PyDACConfig.from_env()

            assert config.translator_path == "/env/translator"
            assert config.Compiler == "clang++"
 # Should use defaults for other values
            assert config.cache_dir == "./.pydac_cache"
            assert config.default_mode == "usm"

    def test_config_from_env_empty(self):

        """Test loading configuration with no environment variables"""

        with patch.dict(os.environ, {}, clear=True):
            config = PyDACConfig.from_env()

 # Should use all defaults
            assert config.translator_path is None
            assert config.Compiler == "dpcpp"
            assert config.cache_dir == "./.pydac_cache"
            assert config.default_mode == "usm"

    def test_config_toml_not_installed(self):

        """Test TOML support when tomli is not installed"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write("[project]\nname = 'test'")
            temp_path = f.name

        try:
            with patch('pydac.utils.config.tomli', None):
                with pytest.raises(ImportError, match="tomli package required"):
                    PyDACConfig.from_file(temp_path)
        finally:
            os.unlink(temp_path)

    def test_config_save_toml_not_installed(self):

        """Test saving TOML when tomli_w is not installed"""

        config = PyDACConfig()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            temp_path = f.name

        try:
            with patch('pydac.utils.config.tomli_w', None):
                with pytest.raises(ImportError, match="tomli-w package required"):
                    config.save_to_file(temp_path)
        finally:
            if Path(temp_path).exists():
                os.unlink(temp_path)

    def test_config_custom_values(self):

        """Test configuration with custom values"""

        config = PyDACConfig(
        translator_path="/custom/translator",
        Compiler="g++",
        cache_dir="/custom/cache",
        default_mode="mpi",
        default_flags=["-O0", "-g"],
        output_dir="/custom/output",
        keep_intermediate=True,
        enable_cache=False,
        parallel_jobs=16
        )

        assert config.translator_path == "/custom/translator"
        assert config.Compiler == "g++"
        assert config.cache_dir == "/custom/cache"
        assert config.default_mode == "mpi"
        assert config.default_flags == ["-O0", "-g"]
        assert config.output_dir == "/custom/output"
        assert config.keep_intermediate is True
        assert config.enable_cache is False
        assert config.parallel_jobs == 16


