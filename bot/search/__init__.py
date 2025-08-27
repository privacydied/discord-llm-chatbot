from .types import SearchResult  # expose dataclass used by tests

# Backward-compatibility shims for legacy imports
# Some older modules referenced functions directly from the package-level namespace
# e.g. `from bot.search import web_search, search_memories`. Provide minimal shims
# to prevent ImportError and keep tests deterministic by mocking aiohttp.

import os
from typing import List
import aiohttp
from bs4 import BeautifulSoup


async def web_search(query: str, max_results: int = 5) -> List[SearchResult]:
    """Simple DuckDuckGo HTML search used by legacy tests.

    This implementation uses aiohttp and BeautifulSoup so unit tests can
    patch the HTTP call and provide deterministic HTML.
    """
    search_url = "https://html.duckduckgo.com/html/"
    params = {"q": query, "kl": os.getenv("SEARCH_LOCALE", "us-en")}
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
    }

    results: List[SearchResult] = []
    async with aiohttp.ClientSession() as session:
        async with session.post(search_url, data=params, headers=headers) as resp:
            if resp.status != 200:
                return []
            html = await resp.text()

    soup = BeautifulSoup(html, "html.parser")
    items = soup.select(".result")
    for result in items[: max_results if max_results else len(items)]:
        title_elem = result.select_one(".result__a")
        if not title_elem:
            continue
        title = title_elem.get_text(strip=True)
        url = title_elem.get("href", "")
        # unwrap DDG redirect if present
        if url.startswith("//duckduckgo.com/l/"):
            raw = url.replace("//duckduckgo.com/l/?uddg=", "").split("&", 1)[0]
            import urllib.parse

            url = urllib.parse.unquote(raw)
        snippet_elem = result.select_one(".result__snippet")
        snippet = snippet_elem.get_text(strip=True) if snippet_elem else None
        results.append(SearchResult(title=title, url=url, snippet=snippet))

    return results


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
