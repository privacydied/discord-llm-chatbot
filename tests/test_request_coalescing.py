"""Tests for request coalescing module.

[PA] Deduplication for expensive operations
[RM] Memory cleanup
"""

import asyncio
from typing import Never

import pytest

from bot.request_coalescing import (
    RequestCoalescer,
    _CoalescedEntry,
    get_url_processing_coalescer,
    get_vl_image_coalescer,
)


class TestRequestCoalescer:
    @pytest.fixture
    def coalescer(self):
        return RequestCoalescer[str](name="test", result_ttl_s=0.1, cleanup_interval_s=0.05)

    @pytest.mark.asyncio
    async def test_single_request_executes(self, coalescer) -> None:
        """A single request should execute normally."""

        async def operation() -> str:
            return "result"

        result = await coalescer.execute("key", operation)
        assert result == "result"

    @pytest.mark.asyncio
    async def test_concurrent_requests_share_result(self, coalescer) -> None:
        """Multiple concurrent requests for same key should share one execution."""
        call_count = 0

        async def slow_operation() -> str:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.05)
            return f"result_{call_count}"

        # Launch multiple concurrent requests
        tasks = [coalescer.execute("same_key", slow_operation) for _ in range(5)]

        results = await asyncio.gather(*tasks)

        # All results should be the same (shared execution)
        assert all(r == results[0] for r in results)
        # But only one actual execution occurred
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_sequential_requests_re_execute(self, coalescer) -> None:
        """Sequential requests should execute separately (after TTL)."""
        call_count = 0

        async def operation() -> str:
            nonlocal call_count
            call_count += 1
            return f"result_{call_count}"

        # First request
        result1 = await coalescer.execute("key", operation)
        assert result1 == "result_1"

        # Wait for TTL to expire
        await asyncio.sleep(0.15)

        # Second request should execute again
        result2 = await coalescer.execute("key", operation)
        assert result2 == "result_2"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_error_propagation(self, coalescer) -> None:
        """Errors should be propagated to all waiters."""

        async def failing_operation() -> Never:
            msg = "test error"
            raise ValueError(msg)

        # Multiple concurrent requests
        tasks = [coalescer.execute("error_key", failing_operation) for _ in range(3)]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should receive the same error
        for r in results:
            assert isinstance(r, ValueError)
            assert str(r) == "test error"

    @pytest.mark.asyncio
    async def test_timeout(self, coalescer) -> None:
        """Timeout should be enforced."""

        async def slow_operation() -> str:
            await asyncio.sleep(10)  # Very slow
            return "result"

        with pytest.raises(asyncio.TimeoutError):
            await coalescer.execute("timeout_key", slow_operation, timeout=0.01)

    @pytest.mark.asyncio
    async def test_different_keys_independent(self, coalescer) -> None:
        """Different keys should execute independently."""
        results = []

        async def make_operation(key):
            async def operation() -> str:
                results.append(key)
                return f"result_{key}"

            return operation

        # Execute different keys concurrently
        tasks = [coalescer.execute(f"key_{i}", await make_operation(i)) for i in range(5)]

        await asyncio.gather(*tasks)

        # All should have executed
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached_result(self, coalescer) -> None:
        """Quick re-requests should use cached result."""
        call_count = 0

        async def operation() -> str:
            nonlocal call_count
            call_count += 1
            return "cached_value"

        # First request
        r1 = await coalescer.execute("cache_key", operation)
        assert r1 == "cached_value"
        assert call_count == 1

        # Immediate second request (within TTL) - should use cache
        r2 = await coalescer.execute("cache_key", operation)
        assert r2 == "cached_value"
        assert call_count == 1  # No additional call


class TestGlobalCoalescers:
    def test_get_url_processing_coalescer(self) -> None:
        """URL processing coalescer should be a singleton."""
        c1 = get_url_processing_coalescer()
        c2 = get_url_processing_coalescer()
        assert c1 is c2
        assert c1.name == "url_proc"

    def test_get_vl_image_coalescer(self) -> None:
        """VL image coalescer should be a singleton."""
        c1 = get_vl_image_coalescer()
        c2 = get_vl_image_coalescer()
        assert c1 is c2
        assert c1.name == "vl_image"


class TestCoalescedEntry:
    def test_entry_creation(self) -> None:
        """Entry should be created with proper defaults."""
        entry = _CoalescedEntry(key="test_key", future=asyncio.Future())
        assert entry.key == "test_key"
        assert entry.result is None
        assert entry.error is None
        assert entry.completed is False
