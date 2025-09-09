"""
Retry utilities specifically for multimodal processing.
Implements per-item retry logic with exponential backoff for sequential processing.
"""

import asyncio
import random
from typing import Any, Callable, TypeVar
from .exceptions import APIError, InferenceError
from .util.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


async def run_with_retries(
    coro_func: Callable[..., Any],
    item: Any,
    retries: int = 3,
    base_delay: float = 5.0,  # Increased to 5 seconds for provider outages
    max_delay: float = 30.0,
    jitter: bool = True,
    timeout: float = 30.0,
) -> str:
    """
    Execute an async function with retry logic for multimodal processing.

    Args:
        coro_func: Async function to execute
        item: Item to process (passed as first argument to coro_func)
        retries: Maximum number of retry attempts
        base_delay: Base delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        jitter: Whether to add random jitter to delays
        timeout: Timeout for each individual attempt

    Returns:
        Result string from successful function execution

    Raises:
        The last exception encountered if all retries fail
    """
    last_exception = None

    for attempt in range(retries):
        try:
            logger.debug(
                f"🔄 Retry attempt {attempt + 1}/{retries} for {coro_func.__name__}"
            )

            # Apply timeout to each attempt
            result = await asyncio.wait_for(coro_func(item), timeout=timeout)

            if attempt > 0:
                logger.info(
                    f"✅ Function {coro_func.__name__} succeeded on attempt {attempt + 1}"
                )

            return result

        except Exception as e:
            last_exception = e

            # Check if error is retryable
            if not _is_retryable_error(e):
                logger.debug(f"❌ Non-retryable error in {coro_func.__name__}: {e}")
                break

            if attempt == retries - 1:
                logger.error(
                    f"❌ All {retries} retry attempts failed for {coro_func.__name__}"
                )
                break

            delay = _calculate_delay(attempt, base_delay, max_delay, jitter)
            logger.warning(
                f"⚠️ Attempt {attempt + 1} failed for {coro_func.__name__}: {e}. "
                f"Retrying in {delay:.2f}s..."
            )

            await asyncio.sleep(delay)

    # All retries exhausted - return error message instead of raising
    error_msg = f"Failed after {retries} attempts: {str(last_exception)}"
    logger.error(f"❌ {coro_func.__name__} failed completely: {error_msg}")

    # Return a canonical failure string for aggregation
    item_name = _get_item_name(item)
    return f"[{item_name}] ❌ Processing failed after {retries} attempts ({type(last_exception).__name__})"


def _is_retryable_error(error: Exception) -> bool:
    """Determine if an error is retryable."""
    # Network and API errors are retryable
    if isinstance(error, (APIError, InferenceError, ConnectionError, TimeoutError)):
        return True

    # Check error message for retryable patterns
    error_str = str(error).lower()
    retryable_patterns = [
        "502",
        "503",
        "504",
        "500",  # HTTP server errors
        "provider returned error",
        "internal server error",
        "bad gateway",
        "service unavailable",
        "gateway timeout",
        "too many requests",
        "connection error",
        "timeout",
        "temporary failure",
    ]

    return any(pattern in error_str for pattern in retryable_patterns)


def _calculate_delay(
    attempt: int, base_delay: float, max_delay: float, jitter: bool
) -> float:
    """Calculate delay for exponential backoff with jitter."""
    delay = base_delay * (2**attempt)
    delay = min(delay, max_delay)

    if jitter:
        # Add jitter to prevent thundering herd
        jitter_range = delay * 0.1
        delay += random.uniform(-jitter_range, jitter_range)

    return max(0, delay)


def _get_item_name(item: Any) -> str:
    """Extract a human-readable name from an item for error messages."""
    if hasattr(item, "payload"):
        payload = item.payload
        if hasattr(payload, "filename"):
            return payload.filename
        elif isinstance(payload, str):
            # URL - extract domain or filename
            if "/" in payload:
                return payload.split("/")[-1] or payload.split("/")[-2]
            return payload[:50] + "..." if len(payload) > 50 else payload

    return str(type(item).__name__)
