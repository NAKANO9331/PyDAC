"""Retry mechanism for PyDAC operations"""


import time

from typing import Callable, TypeVar, Optional, List, Tuple

from functools import wraps

from ..utils.logger import get_logger


T = TypeVar('T')

logger = get_logger("pydac.retry")


class RetryConfig:

    """Configuration for retry mechanism"""

    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        retryable_errors: Optional[List[type]] = None
    ):
        """
        Initialize retry configuration

        Args:
            max_attempts: Maximum number of retry attempts
            initial_delay: Initial delay between retries (seconds)
            max_delay: Maximum delay between retries (seconds)
            exponential_base: Base for exponential backoff
            retryable_errors: List of exception types that should be retried
        """
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.retryable_errors = retryable_errors or [Exception]


def retry(
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[Exception, int], None]] = None
):
    """
    Decorator for retrying operations

    Args:
        config: Retry configuration
        on_retry: Callback function called on each retry
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(1, config.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    # Check if error is retryable
                    if not any(isinstance(e, err_type) for err_type in config.retryable_errors):
                        raise

                    # Check if we've exhausted attempts
                    if attempt >= config.max_attempts:
                        logger.warning(
                            f"{func.__name__} failed after {attempt} attempts: {e}"
                        )
                        raise

                    # Calculate delay with exponential backoff
                    delay = min(
                        config.initial_delay * (config.exponential_base ** (attempt - 1)),
                        config.max_delay
                    )

                    logger.debug(
                        f"{func.__name__} failed (attempt {attempt}/{config.max_attempts}), "
                        f"retrying in {delay:.2f}s: {e}"
                    )

                    if on_retry:
                        on_retry(e, attempt)

                    time.sleep(delay)

            # Should never reach here, but just in case
            if last_exception:
                raise last_exception

            raise RuntimeError("Retry mechanism failed unexpectedly")

        return wrapper

    return decorator


class RetryableOperation:
    """Wrapper for retryable operations"""

    def __init__(self, config: Optional[RetryConfig] = None):
        """
        Initialize retryable operation

        Args:
            config: Retry configuration
        """
        self.config = config or RetryConfig()

    def execute(
        self,
        func: Callable[..., T],
        *args,
        **kwargs
    ) -> T:
        """
        Execute function with retry logic

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result
        """
        last_exception = None

        for attempt in range(1, self.config.max_attempts + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e

                # Check if error is retryable
                if not any(isinstance(e, err_type) for err_type in self.config.retryable_errors):
                    raise

                # Check if we've exhausted attempts
                if attempt >= self.config.max_attempts:
                    logger.warning(
                        f"Operation failed after {attempt} attempts: {e}"
                    )
                    raise

                # Calculate delay with exponential backoff
                delay = min(
                    self.config.initial_delay * (self.config.exponential_base ** (attempt - 1)),
                    self.config.max_delay
                )

                logger.debug(
                    f"Operation failed (attempt {attempt}/{self.config.max_attempts}), "
                    f"retrying in {delay:.2f}s: {e}"
                )

                time.sleep(delay)

        # Should never reach here
        if last_exception:
            raise last_exception

        raise RuntimeError("Retry mechanism failed unexpectedly")


