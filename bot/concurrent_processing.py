"""
Bounded async concurrency for multimodal item processing. [PA][RM]

This module provides utilities for parallelizing independent item processing
with clear timeout budgets and controlled concurrency.

Key features:
- Work-stealing for independent items (images, PDFs, URLs processed in parallel)
- Clear timeout budget per batch
- Preserves partial success behavior
- No locks that can stall the bot
- Request coalescing for duplicate URLs
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Coroutine,
    List,
    Optional,
    Tuple,
    TYPE_CHECKING,
)

from .utils.logging import get_logger

if TYPE_CHECKING:
    from discord import Message

logger = get_logger(__name__)


# Concurrency limits from environment/config
MAX_CONCURRENT_NETWORK = int(__import__("os").environ.get("ROUTER_MAX_CONCURRENCY_NETWORK", "4"))
MAX_CONCURRENT_HEAVY = int(__import__("os").environ.get("ROUTER_MAX_CONCURRENCY_HEAVY", "2"))


@dataclass
class ProcessedResult:
    """Result from processing a single item."""

    item: Any  # InputItem
    modality: Any  # InputModality
    result_text: str
    success: bool
    duration: float
    attempts: int = 1


@dataclass
class BatchConfig:
    """Configuration for concurrent batch processing."""

    # Timeout budgets (in seconds)
    network_timeout: float = 30.0  # HTTP/API calls
    heavy_timeout: float = 120.0  # OCR, STT, ffmpeg

    # Concurrency limits
    max_network_concurrency: int = MAX_CONCURRENT_NETWORK
    max_heavy_concurrency: int = MAX_CONCURRENT_HEAVY

    # Enable coalescing
    enable_coalescing: bool = True


def _normalize_url_for_dedup(url: str) -> str:
    """Normalize URL for coalescing (strip query params, lowercase domain)."""
    try:
        from urllib.parse import urlparse, urlunparse

        p = urlparse(url.strip())
        # Normalize: lowercase domain, remove www, keep path (case sensitive for path)
        netloc = p.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        # Remove common tracking params
        from urllib.parse import parse_qsl, urlencode

        qs_list = parse_qsl(p.query)
        filtered = [(k, v) for k, v in qs_list if k.lower() not in {"utm_source", "utm_medium", "utm_campaign", "fbclid", "gclid"}]
        new_qs = urlencode(filtered)
        return urlunparse((p.scheme, netloc, p.path, "", new_qs, ""))
    except Exception:
        # Fallback: strip whitespace only
        return url.strip()


def _get_url_key(item: Any, modality: Any) -> Optional[str]:
    """Extract coalescing key from item if it's URL-based."""
    from .modality import InputModality

    # URL-based modalities that benefit from coalescing
    url_modalities = {
        InputModality.GENERAL_URL,
        InputModality.SCREENSHOT_URL,
        InputModality.VIDEO_URL,
        InputModality.SINGLE_IMAGE,  # URL images
        InputModality.PDF_DOCUMENT,  # URL PDFs
    }

    if modality not in url_modalities:
        return None

    # Get URL from item
    try:
        source_type = getattr(item, "source_type", None)
        payload = getattr(item, "payload", None)
        if source_type == "url" and payload and isinstance(payload, str):
            return _normalize_url_for_dedup(payload)
    except Exception:
        pass

    return None


async def _process_item_with_budget(
    item: Any,  # InputItem
    modality: Any,  # InputModality
    handler_fn: Callable[..., Coroutine[Any, Any, str]],
    timeout: float,
    message: Optional["Message"],
) -> ProcessedResult:
    """Process a single item with clear timeout budget. [PA][REH]"""
    import time

    start_time = time.time()

    try:
        # Wrap handler with timeout
        result_text = await asyncio.wait_for(
            handler_fn(item, message=message),
            timeout=timeout,
        )

        duration = time.time() - start_time
        return ProcessedResult(
            item=item,
            modality=modality,
            result_text=result_text or "",
            success=True,
            duration=duration,
            attempts=1,
        )

    except asyncio.TimeoutError:
        duration = time.time() - start_time
        logger.warning(f"process_item.timeout | modality={modality.name} timeout={timeout}s")
        return ProcessedResult(
            item=item,
            modality=modality,
            result_text=f"❌ Timed out after {timeout}s",
            success=False,
            duration=duration,
            attempts=1,
        )

    except Exception as e:
        duration = time.time() - start_time
        logger.warning(f"process_item.error | mod={modality.name} error={e}")
        return ProcessedResult(
            item=item,
            modality=modality,
            result_text=f"❌ Failed: {e}",
            success=False,
            duration=duration,
            attempts=1,
        )


async def _process_item_with_coalescing(
    item: Any,
    modality: Any,
    handler_fn: Callable[..., Coroutine[Any, Any, str]],
    timeout: float,
    message: Optional["Message"],
    config: BatchConfig,
) -> ProcessedResult:
    """Process item with optional request coalescing."""

    # Check if coalescing applies
    if not config.enable_coalescing:
        return await _process_item_with_budget(item, modality, handler_fn, timeout, message)

    url_key = _get_url_key(item, modality)
    if url_key is None:
        # Not URL-based, process normally
        return await _process_item_with_budget(item, modality, handler_fn, timeout, message)

    # Use coalescing for URL-based operations
    try:
        from .request_coalescing import get_url_processing_coalescer

        coalescer = get_url_processing_coalescer()
    except ImportError:
        # Coalescer not available, process without dedup
        return await _process_item_with_budget(item, modality, handler_fn, timeout, message)

    async def _do_process() -> str:
        return await handler_fn(item, message=message)

    try:
        start_time = __import__("time").time()
        result_text = await coalescer.execute(url_key, _do_process, timeout=timeout)
        duration = __import__("time").time() - start_time
        return ProcessedResult(
            item=item,
            modality=modality,
            result_text=result_text or "",
            success=True,
            duration=duration,
            attempts=1,
        )
    except asyncio.TimeoutError:
        duration = __import__("time").time() - start_time
        return ProcessedResult(
            item=item,
            modality=modality,
            result_text=f"❌ Timed out after {timeout}s",
            success=False,
            duration=duration,
            attempts=1,
        )
    except Exception as e:
        duration = __import__("time").time() - start_time
        return ProcessedResult(
            item=item,
            modality=modality,
            result_text=f"❌ Failed: {e}",
            success=False,
            duration=duration,
            attempts=1,
        )


async def process_independent_items_concurrently(
    items: List[Tuple[Any, Any, Callable[..., Coroutine[Any, Any, str]]]],
    message: Optional["Message"],
    config: Optional[BatchConfig] = None,
    progress_logger: Optional[Callable[[int, int, str], None]] = None,
) -> List[ProcessedResult]:
    """
    Process independent items concurrently with bounded concurrency.

    Args:
        items: List of (item, modality, handler_fn) tuples
        message: Discord message for context
        config: Batch configuration for timeouts/concurrency
        progress_logger: Optional callback(item_num, total, item_desc)

    Returns:
        List of ProcessedResult in input order

    Notes:
        - Operations are independent; order doesn't matter for final result
        - Results are returned in input order for consistency
        - Partial success is preserved (failures don't cancel other items)
        - Clear timeout budget: network=30s, heavy=120s
    """
    if not items:
        return []

    from .modality import InputModality

    cfg = config or BatchConfig()

    # Categorize items by work type
    network_items: List[Tuple[int, Tuple]] = []  # (index, (item, modality, handler))
    heavy_items: List[Tuple[int, Tuple]] = []  # (index, (item, modality, handler))

    for i, (item, modality, handler) in enumerate(items):
        if modality in (InputModality.SINGLE_IMAGE, InputModality.MULTI_IMAGE):
            # VL inference is network-bound (API call)
            network_items.append((i, (item, modality, handler)))
        elif modality in (
            InputModality.VIDEO_URL,
            InputModality.AUDIO_VIDEO_FILE,
            InputModality.PDF_DOCUMENT,
            InputModality.PDF_OCR,
        ):
            # These are heavy (download + process)
            heavy_items.append((i, (item, modality, handler)))
        else:
            # URLs, screenshots are network-bound
            network_items.append((i, (item, modality, handler)))

    results: List[Optional[ProcessedResult]] = [None] * len(items)

    async def _process_with_semaphore(
        index: int,
        item: Any,
        modality: Any,
        handler_fn: Callable,
        sem: asyncio.Semaphore,
        timeout: float,
    ) -> Tuple[int, ProcessedResult]:
        """Process item with semaphore-controlled concurrency."""
        async with sem:
            # Log progress before processing
            if progress_logger:
                desc = f"{modality.name}"
                try:
                    if hasattr(item, "source_type") and hasattr(item, "payload"):
                        if item.source_type == "url":
                            url = str(item.payload)[:40]
                            desc = f"{modality.name} ({url}...)"
                        elif item.source_type == "attachment":
                            fname = getattr(item.payload, "filename", "attachment")
                            desc = f"{modality.name} ({fname})"
                except Exception:
                    pass
                progress_logger(index + 1, len(items), desc)

            result = await _process_item_with_coalescing(item, modality, handler_fn, timeout, message, cfg)
            return index, result

    # Create semaphores for bounded concurrency
    network_sem = asyncio.Semaphore(cfg.max_network_concurrency)
    heavy_sem = asyncio.Semaphore(cfg.max_heavy_concurrency)

    # Build task list
    tasks: List[Coroutine] = []

    for index, (item, modality, handler) in network_items:
        tasks.append(_process_with_semaphore(index, item, modality, handler, network_sem, cfg.network_timeout))

    for index, (item, modality, handler) in heavy_items:
        tasks.append(_process_with_semaphore(index, item, modality, handler, heavy_sem, cfg.heavy_timeout))

    # Process all concurrently
    completed = await asyncio.gather(*tasks, return_exceptions=True)

    # Collect results
    for res in completed:
        if isinstance(res, Exception):
            logger.error(f"batch_process.exception | error={res}")
            continue
        index, result = res
        results[index] = result

    return [r for r in results if r is not None]


async def process_items_sequential_with_timeout(
    items: List[Tuple[Any, Any, Callable[..., Coroutine[Any, Any, str]]]],
    message: Optional["Message"],
    timeout_per_item: float = 30.0,
    progress_logger: Optional[Callable[[int, int, str], None]] = None,
) -> List[ProcessedResult]:
    """
    Process items sequentially with per-item timeout (original behavior).

    Use this when order matters or when items must be processed one at a time.
    Preserves original sequential semantics.
    """
    results: List[ProcessedResult] = []

    for i, (item, modality, handler_fn) in enumerate(items):
        # Log progress
        if progress_logger:
            desc = f"{modality.name}"
            progress_logger(i + 1, len(items), desc)

        result = await _process_item_with_budget(item, modality, handler_fn, timeout_per_item, message)
        results.append(result)

    return results
