"""Configuration management for PyDAC"""


import os

from pathlib import Path

from typing import Optional, List

from dataclasses import dataclass, field


# Import tomli modules for testing/mocking (optional imports)
try:
 import tomli
except ImportError:
 tomli = None

try:
 import tomli_w
except ImportError:
 tomli_w = None


@dataclass
class PyDACConfig:
    """PyDAC configuration"""

    # Path configuration
    translator_path: Optional[str] = None
    Compiler: str = "dpcpp"
    cache_dir: str = "./.pydac_cache"

    # Default options
    default_mode: str = "usm"
    default_flags: List[str] = field(default_factory=lambda: ["-O2", "-std=c++17", "-fsycl"])

    # Output configuration
    output_dir: str = "./output"
    keep_intermediate: bool = False

    # Performance configuration
    enable_cache: bool = True
    parallel_jobs: int = 4

    @classmethod
    def from_file(cls, config_file: str) -> 'PyDACConfig':
        """Load configuration from file"""

        import json

        path = Path(config_file)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")

        if path.suffix == '.json':
            with open(path, 'r') as f:
                data = json.load(f)
        elif path.suffix == '.toml':
            if tomli is None:
                raise ImportError(
                    "tomli package required for TOML support. "
                    "Install with: pip install tomli"
                )
            with open(path, 'rb') as f:
                data = tomli.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")

        return cls(**data)

    def save_to_file(self, config_file: str):
        """Save configuration to file"""

        import json

        path = Path(config_file)
        data = {
            "translator_path": self.translator_path,
            "Compiler": self.Compiler,
            "cache_dir": self.cache_dir,
            "default_mode": self.default_mode,
            "default_flags": self.default_flags,
            "output_dir": self.output_dir,
            "keep_intermediate": self.keep_intermediate,
            "enable_cache": self.enable_cache,
            "parallel_jobs": self.parallel_jobs,
        }

        if path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        elif path.suffix == '.toml':
            if tomli_w is None:
                raise ImportError(
                    "tomli-w package required for TOML support. "
                    "Install with: pip install tomli-w"
                )
            with open(path, 'wb') as f:
                tomli_w.dump(data, f)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")

    @classmethod
    def from_env(cls) -> 'PyDACConfig':
        """Load configuration from environment variables"""

        config = cls()

        if "DACPP_TRANSLATOR" in os.environ:
            config.translator_path = os.environ["DACPP_TRANSLATOR"]

        if "DACPP_COMPILER" in os.environ:
            config.Compiler = os.environ["DACPP_COMPILER"]

        if "DACPP_CACHE_DIR" in os.environ:
            config.cache_dir = os.environ["DACPP_CACHE_DIR"]

        if "DACPP_DEFAULT_MODE" in os.environ:
            config.default_mode = os.environ["DACPP_DEFAULT_MODE"]

        return config

    @classmethod
    def default(cls) -> 'PyDACConfig':
        """Get default configuration"""

        return cls()

