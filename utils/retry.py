"""
Shared retry/backoff policy for flaky I/O (LLM calls, HTTP requests).
"""
from __future__ import annotations

from typing import Callable, TypeVar

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from utils.logger import get_logger

logger = get_logger("retry")
T = TypeVar("T")

RETRYABLE_EXCEPTIONS = (ConnectionError, TimeoutError, OSError)


def with_retry(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 10.0,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Retry a function with exponential backoff on transient network errors."""
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=min_wait, max=max_wait),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        reraise=True,
    )
