"""Logging utility for PyDAC"""


import logging

import sys

from typing import Optional

from pathlib import Path


class PyDACLogger:
    """Logger for PyDAC"""

    _logger: Optional[logging.Logger] = None

    @classmethod
    def get_logger(
        cls,
        name: str = "pydac",
        level: int = logging.INFO,
        log_file: Optional[str] = None
    ) -> logging.Logger:
        """
        Get or create logger

        Args:
            name: Logger name
            level: Logging level
            log_file: Log file path (None for console only)

        Returns:
            Logger instance
        """
        if cls._logger is None:
            cls._logger = logging.getLogger(name)
            cls._logger.setLevel(level)

            # Remove existing handlers
            cls._logger.handlers.clear()

            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_format = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(console_format)
            cls._logger.addHandler(console_handler)

            # File handler (if specified)
            if log_file:
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(logging.DEBUG)
                file_format = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
                )
                file_handler.setFormatter(file_format)
                cls._logger.addHandler(file_handler)

        return cls._logger

    @classmethod
    def set_level(cls, level: int):
        """Set logging level"""

        if cls._logger:
            cls._logger.setLevel(level)
            for handler in cls._logger.handlers:
                handler.setLevel(level)


def get_logger(name: str = "pydac", verbose: bool = False) -> logging.Logger:
    """Get logger instance"""

    level = logging.DEBUG if verbose else logging.INFO
    return PyDACLogger.get_logger(name, level)

