"""
Factory to create search providers and manage shared HTTP client.
[CA][RM][CMV]
"""
from __future__ import annotations

import asyncio
from typing import Optional

import httpx

from bot.util.logging import get_logger
from bot.config import load_config
from .types import SearchQueryParams, SearchResults
from .base import SearchProvider

logger = get_logger(__name__)

_client_lock = asyncio.Lock()
_client: Optional[httpx.AsyncClient] = None


def _build_client(max_connections: int) -> httpx.AsyncClient:
    limits = httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections)
    return httpx.AsyncClient(limits=limits, timeout=None)


async def get_search_client() -> httpx.AsyncClient:
    global _client
    if _client is not None:
        return _client
    async with _client_lock:
        if _client is None:
            cfg = load_config()
            _client = _build_client(cfg.get("SEARCH_POOL_MAX_CONNECTIONS", 10))
            logger.debug("Created shared httpx.AsyncClient for search providers")
    return _client


async def close_search_client() -> None:
    global _client
    if _client is not None:
        try:
            await _client.aclose()
        except Exception as e:  # [REH]
            logger.debug(f"Error closing search HTTP client: {e}")
        finally:
            _client = None


def get_search_provider() -> SearchProvider:
    cfg = load_config()
    provider = cfg.get("SEARCH_PROVIDER", "ddg").lower()

    if provider == "ddg":
        from .providers.ddg import DDGSearchProvider  # local import to avoid cycle
        return DDGSearchProvider()
    elif provider == "custom":
        from .providers.custom import CustomSearchProvider  # type: ignore
        return CustomSearchProvider()  # pragma: no cover (stubbed unless provided)
    else:
        logger.warning(f"Unknown SEARCH_PROVIDER '{provider}', falling back to ddg")
        from .providers.ddg import DDGSearchProvider
        return DDGSearchProvider()
