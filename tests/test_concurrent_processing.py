"""Tests for concurrent processing module.

[PA] Bounded async concurrency
[REH] Timeout handling
"""

import asyncio
from typing import Never
from unittest.mock import Mock

import pytest


# Mock InputModality for tests
class MockInputModality:
    TEXT_ONLY = Mock(name="TEXT_ONLY")
    SINGLE_IMAGE = Mock(name="SINGLE_IMAGE")
    MULTI_IMAGE = Mock(name="MULTI_IMAGE")
    VIDEO_URL = Mock(name="VIDEO_URL")
    PDF_DOCUMENT = Mock(name="PDF_DOCUMENT")
    GENERAL_URL = Mock(name="GENERAL_URL")
    AUDIO_VIDEO_FILE = Mock(name="AUDIO_VIDEO_FILE")

    def __init__(self, name) -> None:
        self.name = name


class TestBatchConfig:
    def test_default_values(self) -> None:
        from bot.concurrent_processing import BatchConfig

        config = BatchConfig()
        assert config.network_timeout == 30.0
        assert config.heavy_timeout == 120.0
        assert config.enable_coalescing is True

    def test_custom_values(self) -> None:
        from bot.concurrent_processing import BatchConfig

        config = BatchConfig(
            network_timeout=60.0,
            heavy_timeout=300.0,
            enable_coalescing=False,
        )
        assert config.network_timeout == 60.0
        assert config.heavy_timeout == 300.0
        assert config.enable_coalescing is False


class TestProcessItemWithBudget:
    @pytest.mark.asyncio
    async def test_successful_processing(self) -> None:
        from bot.concurrent_processing import _process_item_with_budget

        item = Mock()
        modality = Mock()
        modality.name = "TEST"

        async def handler(item, message=None) -> str:
            return "processed_result"

        result = await _process_item_with_budget(item, modality, handler, timeout=10.0, message=None)

        assert result.success is True
        assert result.result_text == "processed_result"
        assert result.modality == modality

    @pytest.mark.asyncio
    async def test_timeout_handling(self) -> None:
        from bot.concurrent_processing import _process_item_with_budget

        item = Mock()
        modality = Mock()
        modality.name = "TEST"

        async def slow_handler(item, message=None) -> str:
            await asyncio.sleep(10)  # Will timeout
            return "never"

        result = await _process_item_with_budget(item, modality, slow_handler, timeout=0.01, message=None)

        assert result.success is False
        assert "Timed out" in result.result_text
        assert result.duration < 1.0  # Should fail fast

    @pytest.mark.asyncio
    async def test_exception_handling(self) -> None:
        from bot.concurrent_processing import _process_item_with_budget

        item = Mock()
        modality = Mock()
        modality.name = "TEST"

        async def failing_handler(item, message=None) -> Never:
            msg = "processing error"
            raise ValueError(msg)

        result = await _process_item_with_budget(item, modality, failing_handler, timeout=10.0, message=None)

        assert result.success is False
        assert "Failed" in result.result_text


class TestProcessIndependentItemsConcurrently:
    @pytest.mark.asyncio
    async def test_empty_items(self) -> None:
        from bot.concurrent_processing import (
            BatchConfig,
            process_independent_items_concurrently,
        )

        results = await process_independent_items_concurrently(items=[], message=None, config=BatchConfig())

        assert results == []

    @pytest.mark.asyncio
    async def test_single_item(self) -> None:
        from bot.concurrent_processing import (
            BatchConfig,
            process_independent_items_concurrently,
        )

        item = Mock()
        modality = Mock()
        modality.name = "TEST"

        async def handler(item, message=None) -> str:
            return "single_result"

        results = await process_independent_items_concurrently(
            items=[(item, modality, handler)],
            message=None,
            config=BatchConfig(),
        )

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].result_text == "single_result"

    @pytest.mark.asyncio
    async def test_concurrent_processing(self) -> None:
        from bot.concurrent_processing import (
            BatchConfig,
            process_independent_items_concurrently,
        )

        execution_order = []

        async def make_handler(idx):
            async def handler(item, message=None) -> str:
                execution_order.append(("start", idx))
                await asyncio.sleep(0.05)
                execution_order.append(("end", idx))
                return f"result_{idx}"

            return handler

        # Create items that would take 0.15s sequentially
        # but should run concurrently
        modality = Mock()
        modality.name = "GENERAL_URL"

        items = []
        for i in range(3):
            item = Mock()
            item.source_type = "url"
            item.payload = f"http://example.com/{i}"
            items.append((item, modality, await make_handler(i)))

        start = asyncio.get_event_loop().time()
        results = await process_independent_items_concurrently(items=items, message=None, config=BatchConfig())
        duration = asyncio.get_event_loop().time() - start

        # Should complete much faster than sequential (0.15s)
        assert duration < 0.2  # Allow some overhead
        assert len(results) == 3
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_partial_success_preserved(self) -> None:
        from bot.concurrent_processing import (
            BatchConfig,
            process_independent_items_concurrently,
        )

        async def success_handler(item, message=None) -> str:
            return "success"

        async def fail_handler(item, message=None) -> Never:
            msg = "failure"
            raise ValueError(msg)

        modality = Mock()
        modality.name = "TEST"

        items = [
            (Mock(), modality, success_handler),
            (Mock(), modality, fail_handler),
            (Mock(), modality, success_handler),
        ]

        results = await process_independent_items_concurrently(items=items, message=None, config=BatchConfig())

        assert len(results) == 3
        assert results[0].success is True
        assert results[1].success is False
        assert results[2].success is True

    @pytest.mark.asyncio
    async def test_progress_callback(self) -> None:
        from bot.concurrent_processing import (
            BatchConfig,
            process_independent_items_concurrently,
        )

        progress_calls = []

        def progress_logger(current, total, desc) -> None:
            progress_calls.append((current, total, desc))

        async def handler(item, message=None) -> str:
            return "done"

        modality = Mock()
        modality.name = "TEST"

        items = [(Mock(), modality, handler) for _ in range(3)]

        await process_independent_items_concurrently(
            items=items,
            message=None,
            config=BatchConfig(),
            progress_logger=progress_logger,
        )

        assert len(progress_calls) == 3


class TestProcessItemsSequentialWithTimeout:
    @pytest.mark.asyncio
    async def test_sequential_order(self) -> None:
        from bot.concurrent_processing import (
            process_items_sequential_with_timeout,
        )

        execution_order = []

        async def make_handler(idx):
            async def handler(item, message=None) -> str:
                execution_order.append(idx)
                return f"result_{idx}"

            return handler

        modality = Mock()
        modality.name = "TEST"

        items = []
        for i in range(5):
            items.append((Mock(), modality, await make_handler(i)))

        results = await process_items_sequential_with_timeout(items=items, message=None, timeout_per_item=10.0)

        # Should execute in order
        assert execution_order == [0, 1, 2, 3, 4]
        assert len(results) == 5


class TestUrlNormalization:
    def test_normalization_strips_www(self) -> None:
        from bot.concurrent_processing import _normalize_url_for_dedup

        result = _normalize_url_for_dedup("https://www.example.com/path")
        assert "www." not in result
        assert "example.com" in result

    def test_normalization_lowercases_domain(self) -> None:
        from bot.concurrent_processing import _normalize_url_for_dedup

        result = _normalize_url_for_dedup("https://EXAMPLE.COM/Path")
        assert "example.com" in result.lower()

    def test_normalization_removes_tracking_params(self) -> None:
        from bot.concurrent_processing import _normalize_url_for_dedup

        result = _normalize_url_for_dedup("https://example.com/page?utm_source=google&fbclid=123")
        assert "utm_source" not in result
        assert "fbclid" not in result

    def test_normalization_preserves_important_params(self) -> None:
        from bot.concurrent_processing import _normalize_url_for_dedup

        result = _normalize_url_for_dedup("https://example.com/page?id=123&sort=asc")
        assert "id=123" in result
        assert "sort=asc" in result
