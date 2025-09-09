"""
Optimized OpenRouter Client with Connection Pooling, Circuit Breaker, and Intelligent Retries.
Implements PA (Performance Awareness) and REH (Robust Error Handling) rules.
"""

import asyncio
import json
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional, AsyncGenerator
from aiohttp import ClientTimeout, ClientSession, TCPConnector

from .phase_constants import PhaseConstants as PC
from .phase_timing import get_timing_manager, PipelineTracker
from ..utils.logging import get_logger

logger = get_logger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics and state."""

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: float = 0
    last_success_time: float = 0
    total_requests: int = 0
    total_failures: int = 0

    def record_success(self):
        """Record successful request."""
        self.failure_count = 0
        self.last_success_time = time.time()
        self.total_requests += 1
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            logger.info("✅ Circuit breaker CLOSED - service recovered")

    def record_failure(self):
        """Record failed request and update state."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        self.total_requests += 1
        self.total_failures += 1

        # Check if should open circuit [REH]
        if (
            self.state == CircuitState.CLOSED
            and self.failure_count >= PC.OR_BREAKER_FAILURE_WINDOW
        ):
            self.state = CircuitState.OPEN
            logger.warning(
                f"⚠️ Circuit breaker OPEN - {self.failure_count} consecutive failures"
            )

    def should_attempt_request(self) -> bool:
        """Check if request should be attempted based on circuit state."""
        current_time = time.time()

        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            # Check if cooldown period has passed
            if current_time - self.last_failure_time >= (PC.OR_BREAKER_OPEN_MS / 1000):
                self.state = CircuitState.HALF_OPEN
                logger.info("🔄 Circuit breaker HALF_OPEN - testing recovery")
                return True
            return False
        elif self.state == CircuitState.HALF_OPEN:
            # Allow limited testing
            import random

            return random.random() < PC.OR_BREAKER_HALFOPEN_PROB

        return False


@dataclass
class ModelConfig:
    """Configuration for specific models."""

    name: str
    max_tokens: int = PC.OR_MAX_TOKENS_SIMPLE
    temperature: float = 0.7
    timeout_ms: int = PC.OR_TOTAL_DEADLINE_MS
    fallback_model: Optional[str] = None


class OptimizedOpenRouterClient:
    """High-performance OpenRouter client with circuit breaker and connection pooling."""

    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.session: Optional[ClientSession] = None

        # Circuit breaker per model [REH]
        self.circuit_breakers: Dict[str, CircuitBreakerStats] = {}

        # Connection pool metrics
        self.pool_stats = {
            "connections_created": 0,
            "connections_reused": 0,
            "requests_total": 0,
            "requests_failed": 0,
            "avg_response_time_ms": 0,
        }

        # Model configurations [CMV]
        self.model_configs = {
            "gpt-4": ModelConfig(
                "gpt-4", max_tokens=1000, fallback_model="gpt-3.5-turbo"
            ),
            "gpt-3.5-turbo": ModelConfig("gpt-3.5-turbo", max_tokens=800),
            "deepseek/deepseek-chat-v3-0324:free": ModelConfig(
                "deepseek/deepseek-chat-v3-0324:free",
                max_tokens=PC.OR_MAX_TOKENS_SIMPLE,
                fallback_model="gpt-3.5-turbo",
            ),
        }

        logger.info("🔧 OptimizedOpenRouterClient initialized")

    async def _ensure_session(self):
        """Ensure HTTP session is created with optimized settings [PA]."""
        if self.session is None or self.session.closed:
            # Create optimized TCP connector [PA]
            connector = TCPConnector(
                limit=PC.OR_POOL_MAX_CONNECTIONS,
                limit_per_host=PC.OR_POOL_MAX_CONNECTIONS // 2,
                keepalive_timeout=PC.OR_POOL_KEEPALIVE_SECS,
                enable_cleanup_closed=True,
                force_close=False,
                # auto_decompress not needed for aiohttp TCPConnector
            )

            # Optimized timeouts [REH]
            timeout = ClientTimeout(
                total=PC.OR_TOTAL_DEADLINE_MS / 1000,
                connect=PC.OR_CONNECT_TIMEOUT_MS / 1000,
                sock_read=PC.OR_READ_TIMEOUT_MS / 1000,
            )

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "DiscordBot-OptimizedClient/1.0",
            }

            self.session = ClientSession(
                connector=connector,
                timeout=timeout,
                headers=headers,
                json_serialize=json.dumps,  # Faster JSON serialization
            )

            self.pool_stats["connections_created"] += 1
            logger.debug("🔗 New HTTP session created with optimized settings")
        else:
            self.pool_stats["connections_reused"] += 1

    def _get_circuit_breaker(self, model: str) -> CircuitBreakerStats:
        """Get or create circuit breaker for model."""
        if model not in self.circuit_breakers:
            self.circuit_breakers[model] = CircuitBreakerStats()
        return self.circuit_breakers[model]

    def _get_model_config(self, model: str) -> ModelConfig:
        """Get configuration for model with fallback."""
        return self.model_configs.get(model, ModelConfig(model))

    async def _retry_with_backoff(
        self, model: str, request_func, max_retries: int = None
    ):
        """Execute request with exponential backoff retry logic [REH]."""
        if max_retries is None:
            max_retries = PC.OR_MAX_RETRIES

        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                result = await request_func()
                return result
            except Exception as e:
                last_exception = e

                # Check if error is retryable [REH]
                if not self._is_retryable_error(e):
                    logger.debug(f"❌ Non-retryable error for {model}: {str(e)}")
                    raise

                if attempt < max_retries:
                    # Exponential backoff with jitter
                    delay = (1.5**attempt) * (PC.OR_RETRY_JITTER_MS / 1000)
                    jitter = delay * 0.1 * (0.5 - asyncio.get_event_loop().time() % 1)
                    wait_time = delay + jitter

                    logger.debug(
                        f"🔄 Retry {attempt + 1}/{max_retries} for {model} in {wait_time:.2f}s"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"❌ All {max_retries} retries exhausted for {model}")

        raise last_exception

    def _is_retryable_error(self, error: Exception) -> bool:
        """Determine if error should trigger retry [REH]."""
        error_str = str(error).lower()

        # Retryable: network, timeout, server errors
        retryable_patterns = [
            "timeout",
            "connection",
            "502",
            "503",
            "504",
            "provider returned error",
            "temporary",
            "rate limit",
        ]

        # Non-retryable: auth, client errors
        non_retryable_patterns = [
            "401",
            "403",
            "404",
            "invalid api key",
            "unauthorized",
        ]

        for pattern in non_retryable_patterns:
            if pattern in error_str:
                return False

        for pattern in retryable_patterns:
            if pattern in error_str:
                return True

        # Default: retry on unknown errors
        return True

    async def _make_request_with_fallback(
        self, model: str, messages: list, **kwargs
    ) -> Dict[str, Any]:
        """Make request with automatic model fallback [REH]."""
        model_config = self._get_model_config(model)

        # Try primary model first
        try:
            return await self._make_single_request(model, messages, **kwargs)
        except Exception as e:
            logger.warning(f"⚠️ Primary model {model} failed: {str(e)}")

            # Try fallback model if available
            if model_config.fallback_model:
                logger.info(f"🔄 Falling back to {model_config.fallback_model}")
                try:
                    result = await self._make_single_request(
                        model_config.fallback_model, messages, **kwargs
                    )
                    # Mark as fallback in response
                    result["fallback_used"] = True
                    result["original_model"] = model
                    return result
                except Exception as fallback_error:
                    logger.error(
                        f"❌ Fallback model also failed: {str(fallback_error)}"
                    )
                    raise
            else:
                raise

    async def _make_single_request(
        self, model: str, messages: list, **kwargs
    ) -> Dict[str, Any]:
        """Make single request to OpenRouter API."""
        circuit_breaker = self._get_circuit_breaker(model)

        # Check circuit breaker state [REH]
        if not circuit_breaker.should_attempt_request():
            raise Exception(f"Circuit breaker OPEN for model {model}")

        await self._ensure_session()
        model_config = self._get_model_config(model)

        # Prepare payload with model-specific settings [PA]
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", model_config.max_tokens),
            "temperature": kwargs.get("temperature", model_config.temperature),
            **{
                k: v
                for k, v in kwargs.items()
                if k not in ["max_tokens", "temperature"]
            },
        }

        start_time = time.time()

        try:
            async with self.session.post(
                f"{self.base_url}/chat/completions", json=payload
            ) as response:
                response_time_ms = int((time.time() - start_time) * 1000)

                # Update pool stats [PA]
                self.pool_stats["requests_total"] += 1
                old_avg = self.pool_stats["avg_response_time_ms"]
                self.pool_stats["avg_response_time_ms"] = (
                    old_avg * (self.pool_stats["requests_total"] - 1) + response_time_ms
                ) / self.pool_stats["requests_total"]

                if response.status == 200:
                    data = await response.json()
                    circuit_breaker.record_success()

                    # Add timing and metadata
                    data["response_time_ms"] = response_time_ms
                    data["pool_reused"] = self.pool_stats["connections_reused"] > 0
                    data["model_used"] = model

                    # Check for slow response warning [REH]
                    if response_time_ms > PC.OR_WARN_SLOW_MS:
                        logger.warning(
                            f"⚠️ Slow OpenRouter response: {response_time_ms}ms (model: {model})"
                        )

                    return data
                else:
                    error_text = await response.text()
                    circuit_breaker.record_failure()
                    self.pool_stats["requests_failed"] += 1
                    raise Exception(f"HTTP {response.status}: {error_text}")

        except Exception:
            circuit_breaker.record_failure()
            self.pool_stats["requests_failed"] += 1
            raise

    @asynccontextmanager
    async def chat_completion(
        self,
        model: str,
        messages: list,
        tracker: Optional[PipelineTracker] = None,
        **kwargs,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Create chat completion with timing and error handling."""
        timing_manager = get_timing_manager()

        if tracker:
            async with timing_manager.track_phase(
                tracker, PC.PHASE_LLM_CALL, model=model, message_count=len(messages)
            ) as phase_metric:
                try:
                    result = await self._retry_with_backoff(
                        model,
                        lambda: self._make_request_with_fallback(
                            model, messages, **kwargs
                        ),
                    )

                    # Add metrics to phase
                    phase_metric.metadata.update(
                        {
                            "response_time_ms": result.get("response_time_ms", 0),
                            "model_used": result.get("model_used", model),
                            "pool_reused": result.get("pool_reused", False),
                            "fallback_used": result.get("fallback_used", False),
                        }
                    )

                    yield result
                except Exception as e:
                    phase_metric.metadata["error_type"] = type(e).__name__
                    raise
        else:
            # Direct call without tracking
            result = await self._retry_with_backoff(
                model,
                lambda: self._make_request_with_fallback(model, messages, **kwargs),
            )
            yield result

    async def close(self):
        """Clean up resources [RM]."""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.debug("🔒 HTTP session closed")

    def get_stats(self) -> Dict[str, Any]:
        """Get client performance statistics."""
        circuit_stats = {}
        for model, cb in self.circuit_breakers.items():
            circuit_stats[model] = {
                "state": cb.state.value,
                "total_requests": cb.total_requests,
                "total_failures": cb.total_failures,
                "failure_rate": cb.total_failures / max(cb.total_requests, 1),
            }

        return {
            "pool_stats": self.pool_stats.copy(),
            "circuit_breakers": circuit_stats,
            "active_sessions": 0 if not self.session or self.session.closed else 1,
        }


# Global client instance for reuse [PA]
_client_instance: Optional[OptimizedOpenRouterClient] = None


async def get_openrouter_client(
    api_key: str, base_url: str = "https://openrouter.ai/api/v1"
) -> OptimizedOpenRouterClient:
    """Get or create optimized OpenRouter client instance."""
    global _client_instance

    if _client_instance is None:
        _client_instance = OptimizedOpenRouterClient(api_key, base_url)
        logger.info("🚀 Global OptimizedOpenRouterClient created")

    return _client_instance


async def close_global_client():
    """Close global client for cleanup [RM]."""
    global _client_instance
    if _client_instance:
        await _client_instance.close()
        _client_instance = None
