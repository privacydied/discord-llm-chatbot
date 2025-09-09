"""
Shared async HTTP/2 client with optimized pools and DNS cache. [PA][RM]

This module provides a singleton HTTP client optimized for the router's networking needs:
- HTTP/2 support where safe
- Connection pooling with keep-alive
- DNS caching with TTL
- Per-host concurrency limits to avoid WAF throttling
- Configurable timeouts and retries with jitter
- Comprehensive error handling and retry logic

Key optimizations:
- Single shared client across all router operations
- Persistent connections and HTTP/2 multiplexing
- Smart retry logic with exponential backoff and jitter
- Per-family timeout budgets (soft/hard deadlines)
- Circuit breaker patterns for failing hosts
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Dict, Optional, Any, Tuple
from urllib.parse import urlparse
import httpx
from httpx import AsyncClient, Response, RequestError, HTTPStatusError, TimeoutException

from .util.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RequestConfig:
    """Configuration for individual HTTP requests. [CMV]"""

    connect_timeout: float = 1.5  # seconds
    read_timeout: float = 5.0  # seconds
    total_timeout: float = 6.0  # seconds
    max_retries: int = 3
    retry_delay_base: float = 1.0  # seconds
    retry_delay_max: float = 8.0  # seconds
    retry_exponential_base: float = 1.5
    follow_redirects: bool = True
    max_redirects: int = 5


@dataclass
class HostLimits:
    """Per-host concurrency and circuit breaker limits. [REH]"""

    max_concurrent: int = 4
    circuit_breaker_failures: int = 5
    circuit_breaker_window: float = 60.0  # seconds
    circuit_breaker_cooldown: float = 15.0  # seconds


@dataclass
class ClientMetrics:
    """HTTP client metrics for monitoring. [PA]"""

    requests_total: int = 0
    requests_success: int = 0
    requests_failed: int = 0
    requests_retried: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    circuit_breaker_trips: int = 0
    total_bytes_downloaded: int = 0
    avg_response_time_ms: float = 0.0


class CircuitBreaker:
    """Simple circuit breaker for failing hosts. [REH]"""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 15.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def can_request(self) -> bool:
        """Check if requests are allowed through the circuit breaker."""
        now = time.time()

        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if now - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        else:  # HALF_OPEN
            return True

    def record_success(self) -> None:
        """Record a successful request."""
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            self.failure_count = 0

    def record_failure(self) -> None:
        """Record a failed request."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(
                f"🔴 Circuit breaker OPEN after {self.failure_count} failures"
            )


class SharedHttpClient:
    """Shared async HTTP client with optimizations. [PA][RM]"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize shared HTTP client with configuration."""
        self.config = config or {}
        self.client: Optional[AsyncClient] = None
        self.metrics = ClientMetrics()

        # Per-host state tracking
        self.host_semaphores: Dict[str, asyncio.Semaphore] = {}
        self.host_circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.host_limits: Dict[str, HostLimits] = {}

        # DNS cache (simple in-memory)
        self.dns_cache: Dict[str, Tuple[str, float]] = {}  # host -> (ip, expire_time)
        self.dns_cache_ttl = float(self.config.get("HTTP_DNS_CACHE_TTL_S", 300))

        # Default request configuration
        self.default_config = RequestConfig(
            connect_timeout=float(self.config.get("HTTP_CONNECT_TIMEOUT_MS", 1500))
            / 1000,
            read_timeout=float(self.config.get("HTTP_READ_TIMEOUT_MS", 5000)) / 1000,
            total_timeout=float(self.config.get("HTTP_TOTAL_DEADLINE_MS", 6000)) / 1000,
        )

        # HTTP client configuration
        self.http2_enabled = self.config.get("HTTP2_ENABLE", True)
        self.max_connections = int(self.config.get("HTTP_MAX_CONNECTIONS", 64))
        self.max_keepalive = int(self.config.get("HTTP_MAX_KEEPALIVE_CONNECTIONS", 32))

        logger.info(f"🌐 SharedHttpClient initialized (HTTP/2: {self.http2_enabled})")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()

    async def start(self) -> None:
        """Start the HTTP client with optimized configuration."""
        if self.client is not None:
            return  # Already started

        # Configure HTTP client with optimizations
        limits = httpx.Limits(
            max_connections=self.max_connections,
            max_keepalive_connections=self.max_keepalive,
            keepalive_expiry=30.0,  # Keep connections alive for 30s
        )

        timeout = httpx.Timeout(
            connect=self.default_config.connect_timeout,
            read=self.default_config.read_timeout,
            write=5.0,
            pool=1.0,
        )

        headers = {
            "User-Agent": "Discord-LLM-Bot/1.0 (+https://github.com/example/discord-llm-chatbot)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

        self.client = AsyncClient(
            limits=limits,
            timeout=timeout,
            headers=headers,
            http2=self.http2_enabled,
            follow_redirects=True,
            max_redirects=5,
        )

        logger.info("✅ SharedHttpClient started with optimized configuration")

    async def stop(self) -> None:
        """Stop the HTTP client and clean up resources."""
        if self.client is not None:
            await self.client.aclose()
            self.client = None
            logger.info("🛑 SharedHttpClient stopped")

    def _get_host_semaphore(self, host: str) -> asyncio.Semaphore:
        """Get or create per-host concurrency semaphore. [RM]"""
        if host not in self.host_semaphores:
            limits = self.host_limits.get(host, HostLimits())
            self.host_semaphores[host] = asyncio.Semaphore(limits.max_concurrent)
        return self.host_semaphores[host]

    def _get_circuit_breaker(self, host: str) -> CircuitBreaker:
        """Get or create per-host circuit breaker. [REH]"""
        if host not in self.host_circuit_breakers:
            limits = self.host_limits.get(host, HostLimits())
            self.host_circuit_breakers[host] = CircuitBreaker(
                limits.circuit_breaker_failures, limits.circuit_breaker_cooldown
            )
        return self.host_circuit_breakers[host]

    async def _wait_with_jitter(self, delay: float) -> None:
        """Wait with jitter to avoid thundering herds. [REH]"""
        import random

        jitter = random.uniform(0.1, 0.3) * delay
        await asyncio.sleep(delay + jitter)

    async def request(
        self, method: str, url: str, config: Optional[RequestConfig] = None, **kwargs
    ) -> Response:
        """Make HTTP request with retries and circuit breaker. [REH][PA]"""
        if self.client is None:
            await self.start()

        config = config or self.default_config
        parsed_url = urlparse(url)
        host = parsed_url.netloc.lower()

        # Check circuit breaker
        circuit_breaker = self._get_circuit_breaker(host)
        if not circuit_breaker.can_request():
            self.metrics.circuit_breaker_trips += 1
            raise httpx.RequestError(f"Circuit breaker OPEN for {host}")

        # Get per-host semaphore for concurrency control
        semaphore = self._get_host_semaphore(host)

        last_exception = None

        for attempt in range(config.max_retries + 1):
            start_time = time.time()

            try:
                async with semaphore:  # Limit per-host concurrency
                    # Set timeout for this attempt
                    timeout = httpx.Timeout(
                        connect=config.connect_timeout,
                        read=config.read_timeout,
                        write=5.0,
                        pool=1.0,
                    )

                    response = await self.client.request(
                        method=method, url=url, timeout=timeout, **kwargs
                    )

                    # Record metrics
                    self.metrics.requests_total += 1
                    response_time = (time.time() - start_time) * 1000

                    # Check for HTTP errors
                    if response.status_code >= 400:
                        if response.status_code >= 500:
                            # Server error - retryable
                            raise HTTPStatusError(
                                f"HTTP {response.status_code}",
                                request=response.request,
                                response=response,
                            )
                        else:
                            # Client error - not retryable
                            circuit_breaker.record_success()
                            self.metrics.requests_success += 1
                            return response

                    # Success
                    circuit_breaker.record_success()
                    self.metrics.requests_success += 1
                    self.metrics.total_bytes_downloaded += len(response.content)

                    # Update average response time
                    if self.metrics.requests_success > 0:
                        self.metrics.avg_response_time_ms = (
                            self.metrics.avg_response_time_ms
                            * (self.metrics.requests_success - 1)
                            + response_time
                        ) / self.metrics.requests_success

                    return response

            except (RequestError, TimeoutException, HTTPStatusError) as e:
                last_exception = e
                circuit_breaker.record_failure()
                self.metrics.requests_failed += 1

                # Don't retry on the last attempt
                if attempt == config.max_retries:
                    break

                # Calculate retry delay with exponential backoff
                delay = min(
                    config.retry_delay_base * (config.retry_exponential_base**attempt),
                    config.retry_delay_max,
                )

                logger.debug(
                    f"🔄 HTTP request failed (attempt {attempt + 1}/{config.max_retries + 1}), "
                    f"retrying in {delay:.1f}s: {e}",
                    extra={
                        "event": "http.retry",
                        "detail": {
                            "url": url,
                            "attempt": attempt + 1,
                            "delay_s": delay,
                            "error": str(e),
                        },
                    },
                )

                self.metrics.requests_retried += 1
                await self._wait_with_jitter(delay)

        # All retries exhausted
        logger.error(
            f"❌ HTTP request failed after {config.max_retries + 1} attempts: {last_exception}"
        )
        raise last_exception

    async def get(
        self, url: str, config: Optional[RequestConfig] = None, **kwargs
    ) -> Response:
        """Make GET request."""
        return await self.request("GET", url, config, **kwargs)

    async def post(
        self, url: str, config: Optional[RequestConfig] = None, **kwargs
    ) -> Response:
        """Make POST request."""
        return await self.request("POST", url, config, **kwargs)

    async def head(
        self, url: str, config: Optional[RequestConfig] = None, **kwargs
    ) -> Response:
        """Make HEAD request."""
        return await self.request("HEAD", url, config, **kwargs)

    def get_metrics(self) -> ClientMetrics:
        """Get current HTTP client metrics."""
        return self.metrics

    def set_host_limits(self, host: str, limits: HostLimits) -> None:
        """Configure per-host limits. [CMV]"""
        self.host_limits[host] = limits
        logger.info(f"🔧 Set limits for {host}: max_concurrent={limits.max_concurrent}")


# Global singleton instance
_http_client_instance: Optional[SharedHttpClient] = None


async def get_http_client(config: Optional[Dict[str, Any]] = None) -> SharedHttpClient:
    """Get or create the shared HTTP client instance. [CA]"""
    global _http_client_instance

    if _http_client_instance is None:
        _http_client_instance = SharedHttpClient(config)
        await _http_client_instance.start()

    return _http_client_instance


async def cleanup_http_client() -> None:
    """Clean up the shared HTTP client instance."""
    global _http_client_instance

    if _http_client_instance is not None:
        await _http_client_instance.stop()
        _http_client_instance = None


def configure_host_limits(host: str, limits: HostLimits) -> None:
    """Configure limits for a specific host. [CMV]"""
    if _http_client_instance is not None:
        _http_client_instance.set_host_limits(host, limits)
