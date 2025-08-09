"""
Retry utilities for handling transient errors with exponential backoff.
Implements robust error handling patterns for external API calls.
"""
import asyncio
import logging
import random
from typing import Any, Callable, List, Type, Union, Optional
from functools import wraps
from .exceptions import APIError, InferenceError
import httpx

logger = logging.getLogger(__name__)

class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Optional[List[Type[Exception]]] = None,
        retryable_status_codes: Optional[List[int]] = None
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions or [
            APIError, InferenceError, ConnectionError, TimeoutError
        ]
        self.retryable_status_codes = retryable_status_codes or [
            500, 502, 503, 504, 429  # Server errors and rate limiting
        ]

def is_retryable_error(error: Exception, config: RetryConfig) -> bool:
    """
    Determine if an error is retryable based on configuration.
    
    Args:
        error: The exception that occurred
        config: Retry configuration
        
    Returns:
        True if the error should be retried, False otherwise
    """
    error_str = str(error).lower()

    # 1) Direct type match
    if any(isinstance(error, exc_type) for exc_type in config.retryable_exceptions):
        return True

    # 2) HTTP status codes present in message (e.g., '429 Too Many Requests')
    if any(str(code) in error_str for code in config.retryable_status_codes):
        return True

    # 3) Common transient error patterns
    transient_patterns = [
        'internal server error',
        'bad gateway',
        'service unavailable',
        'gateway timeout',
        'too many requests',
        'provider returned error',
        'connection error',
        'timeout',
        'timed out',
        'no choices returned',
        'no choices in response',
        'empty response from api'
    ]
    if any(pattern in error_str for pattern in transient_patterns):
        return True

    return False

def calculate_delay(attempt: int, config: RetryConfig) -> float:
    """
    Calculate delay for exponential backoff with jitter.
    
    Args:
        attempt: Current attempt number (0-based)
        config: Retry configuration
        
    Returns:
        Delay in seconds
    """
    delay = config.base_delay * (config.exponential_base ** attempt)
    delay = min(delay, config.max_delay)
    
    if config.jitter:
        # Add jitter to prevent thundering herd
        jitter_range = delay * 0.1
        delay += random.uniform(-jitter_range, jitter_range)
    
    return max(0, delay)

async def retry_async(
    func: Callable,
    config: RetryConfig,
    *args,
    **kwargs
) -> Any:
    """
    Execute an async function with retry logic.
    
    Args:
        func: Async function to execute
        config: Retry configuration
        *args: Arguments to pass to func
        **kwargs: Keyword arguments to pass to func
        
    Returns:
        Result of successful function execution
        
    Raises:
        The last exception encountered if all retries fail
    """
    last_exception = None
    
    for attempt in range(config.max_attempts):
        try:
            logger.debug(f"🔄 Retry attempt {attempt + 1}/{config.max_attempts} for {func.__name__}")
            result = await func(*args, **kwargs)
            
            if attempt > 0:
                logger.info(f"✅ Function {func.__name__} succeeded on attempt {attempt + 1}")
            
            return result
            
        except Exception as e:
            last_exception = e
            
            if not is_retryable_error(e, config):
                logger.debug(f"❌ Non-retryable error in {func.__name__}: {e}")
                raise e
            
            if attempt == config.max_attempts - 1:
                logger.error(f"❌ All {config.max_attempts} retry attempts failed for {func.__name__}")
                break
            
            # Base exponential backoff
            delay = calculate_delay(attempt, config)
            # If the exception carries a Retry-After hint, respect it within bounds [REH][PA]
            retry_after_hint = getattr(e, "retry_after_seconds", None)
            extra_note = ""
            try:
                if retry_after_hint is not None:
                    ra = float(retry_after_hint)
                    if ra > 0:
                        # Respect server-provided cooldown but do not exceed configured max_delay
                        bounded_ra = min(ra, config.max_delay)
                        delay = max(delay, bounded_ra)
                        extra_note = f" (respecting Retry-After={bounded_ra:.2f}s)"
            except Exception:
                pass
            logger.warning(
                f"⚠️ Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                f"Retrying in {delay:.2f}s...{extra_note}"
            )
            
            await asyncio.sleep(delay)
    
    # All retries exhausted
    raise last_exception

def with_retry(config: Optional[RetryConfig] = None):
    """
    Decorator to add retry logic to async functions.
    
    Args:
        config: Retry configuration. If None, uses default config.
        
    Returns:
        Decorated function with retry logic
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await retry_async(func, config, *args, **kwargs)
        return wrapper
    
    return decorator

# Predefined retry configurations for common scenarios
VISION_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    base_delay=5.0,  # Increased to 5 seconds for provider outages
    max_delay=30.0,
    exponential_base=2.0,
    jitter=True,
    retryable_exceptions=[APIError, InferenceError, ConnectionError, TimeoutError, httpx.HTTPStatusError],
    retryable_status_codes=[500, 502, 503, 504, 429]
)

API_RETRY_CONFIG = RetryConfig(
    max_attempts=2,
    base_delay=1.0,
    max_delay=8.0,
    exponential_base=1.5,
    jitter=True,
    retryable_exceptions=[APIError, ConnectionError, TimeoutError, httpx.HTTPStatusError],
    retryable_status_codes=[500, 502, 503, 504, 429]
)

QUICK_RETRY_CONFIG = RetryConfig(
    max_attempts=2,
    base_delay=0.5,
    max_delay=5.0,
    exponential_base=2.0,
    jitter=True,
    retryable_exceptions=[APIError, ConnectionError, TimeoutError, httpx.HTTPStatusError],
    retryable_status_codes=[500, 502, 503, 504, 429]
)
