from .types import SafeSearch, SearchQueryParams, SearchResult, SearchResults
from .factory import get_search_provider, get_search_client, close_search_client

# Backward-compatibility shims for legacy imports
# Some older modules referenced functions directly from the package-level namespace
# e.g. `from bot.search import web_search, search_memories`. Provide minimal shims
# to prevent ImportError and guide callers toward the provider abstraction.

import os
from typing import List


async def web_search(query: str, max_results: int = 5) -> List[SearchResult]:
    """Compatibility: provider-backed web search.

    New code should construct SearchQueryParams and call provider.search(params) directly.
    """
    provider = get_search_provider()
    # Resolve SafeSearch from env (default: moderate)
    safe_env = os.getenv("SEARCH_SAFE", SafeSearch.MODERATE.value).lower()
    safe_level = SafeSearch(safe_env) if safe_env in {s.value for s in SafeSearch} else SafeSearch.MODERATE

    try:
        timeout_ms = int(os.getenv("DDG_TIMEOUT_MS", "5000"))
    except ValueError:
        timeout_ms = 5000

    params = SearchQueryParams(
        query=query,
        max_results=max_results,
        safesearch=safe_level,
        locale=os.getenv("SEARCH_LOCALE") or None,
        timeout_ms=timeout_ms,
    )

    return await provider.search(params)


async def search_memories(*_, **__):  # type: ignore[no-untyped-def]
    """Deprecated: legacy memory search (no longer supported here).

    Returns an empty list to preserve call sites from crashing. Migrate to RAG memory systems.
    """
    return []


async def search_files(*_, **__):  # type: ignore[no-untyped-def]
    """Deprecated: legacy file search (no longer supported here)."""
    return []


async def search_all(query: str, *_, **__) -> dict:  # type: ignore[no-untyped-def]
    """Deprecated aggregate shim: returns only provider web results under 'web' key."""
    results = await web_search(query)
    return {"web": results, "memories": [], "files": []}
