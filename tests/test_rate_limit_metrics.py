import asyncio
import os
import time
from typing import Optional, Dict

from bot.retry_utils import RetryConfig, retry_async
from bot.exceptions import APIError
from bot.x_api_client import XApiClient
from bot.metrics.prometheus_metrics import PrometheusMetrics
from bot.router import Router
from bot.util.logging import init_logging
import httpx


class _FakeBot:
    def __init__(self, metrics):
        self.metrics = metrics
        self.config: Dict[str, object] = {}
        self.tts_manager = None
        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            self.loop = None


async def _always_fails_with_retry_after():
    # Raise APIError with a large Retry-After hint
    err = APIError("synthetic error for retry test")
    setattr(err, "retry_after_seconds", 5.0)  # 5s hint (will be bounded by config.max_delay)
    raise err


async def test_retry_respects_retry_after_bound():
    cfg = RetryConfig(
        max_attempts=2,      # 1 failure + 1 retry
        base_delay=0.01,     # base backoff is tiny
        max_delay=0.2,       # bound any retry-after to 0.2s
        exponential_base=2.0,
        jitter=False,
        retryable_exceptions=[APIError, httpx.HTTPStatusError]
    )

    start = time.perf_counter()
    try:
        await retry_async(_always_fails_with_retry_after, cfg)
    except Exception as e:
        # Expected to fail after retries exhausted
        elapsed = time.perf_counter() - start
        # We expect at least ~max_delay because Retry-After=5 was bounded to 0.2 and max(delay, 0.2) applies
        assert elapsed >= 0.18, f"elapsed {elapsed:.3f}s did not respect bounded Retry-After"
    else:
        raise AssertionError("retry_async unexpectedly succeeded")


async def test_x_api_client_parses_retry_after_header():
    client = XApiClient(bearer_token=None, timeout_ms=500)
    try:
        # Build a synthetic 429 response with Retry-After header
        req = httpx.Request("GET", "https://api.twitter.com/2/tweets/123")
        resp = httpx.Response(
            status_code=429,
            headers={"retry-after": "17"},
            content=b"{}",
            request=req,
        )
        try:
            await client._raise_for_status(resp)
        except APIError as err:
            ra = getattr(err, "retry_after_seconds", None)
            assert ra is not None, "retry_after_seconds not attached on APIError"
            assert abs(float(ra) - 17.0) < 0.001, f"unexpected retry_after_seconds {ra}"
        else:
            raise AssertionError("_raise_for_status did not raise on 429")
    finally:
        await client.aclose()


def test_metrics_and_router_increment():
    # Disable HTTP server to avoid port binding during tests
    metrics = PrometheusMetrics(port=0, enable_http_server=False)

    # Define counters exactly as bot.core.bot.setup_hook does
    metrics.define_counter("x.photo_to_vl.enabled", "X photos routed to VL (feature enabled)")
    metrics.define_counter("x.photo_to_vl.no_urls", "X photo routing: no photo URLs available")
    metrics.define_counter("x.photo_to_vl.skipped", "X photo routing skipped", labels=["enabled"]) 
    metrics.define_counter("x.photo_to_vl.attempt", "X photo routing attempts", labels=["idx"]) 
    metrics.define_counter("x.photo_to_vl.success", "X photo routing success", labels=["idx"]) 
    metrics.define_counter("x.photo_to_vl.failure", "X photo routing failure", labels=["idx"]) 

    # Basic presence checks (internal map)
    for name in [
        "x.photo_to_vl.enabled",
        "x.photo_to_vl.no_urls",
        "x.photo_to_vl.skipped",
        "x.photo_to_vl.attempt",
        "x.photo_to_vl.success",
        "x.photo_to_vl.failure",
    ]:
        assert name in metrics._counters, f"counter {name} not registered"

    # Build a minimal Router with a fake bot and increment metrics via helper
    bot = _FakeBot(metrics)
    router = Router(bot)

    # These should not raise
    router._metric_inc("x.photo_to_vl.enabled", None)
    router._metric_inc("x.photo_to_vl.skipped", {"enabled": "false"})
    router._metric_inc("x.photo_to_vl.attempt", {"idx": "1"})


async def _amain():
    # Configure logging per user preference (Rich + JSONL sinks)
    os.environ.setdefault("LOG_LEVEL", "DEBUG")
    os.environ.setdefault("LOG_JSONL_PATH", "./logs/test.jsonl")
    init_logging()

    await test_retry_respects_retry_after_bound()
    await test_x_api_client_parses_retry_after_header()
    test_metrics_and_router_increment()
    print("OK: tests completed")


if __name__ == "__main__":
    asyncio.run(_amain())
