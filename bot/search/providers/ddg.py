"""
DuckDuckGo search provider using the official duckduckgo_search (ddgs) client.
Falls back to legacy HTML parsing if ddgs is unavailable.
[CA][REH][IV][PA]
"""
from __future__ import annotations

import asyncio
from typing import List, Optional, Set
from urllib.parse import urlencode

import httpx
from bs4 import BeautifulSoup  # beautifulsoup4 is in requirements
import importlib

from bot.util.logging import get_logger
from bot.config import load_config
from ..types import SearchQueryParams, SearchResult, SearchResults, SafeSearch
from ..factory import get_search_client

logger = get_logger(__name__)


class DDGSearchProvider:
    def __init__(self) -> None:
        self.cfg = load_config()
        self.endpoint = self.cfg.get("DDG_API_ENDPOINT", "https://duckduckgo.com/html/")

    async def search(self, params: SearchQueryParams) -> SearchResults:
        """Execute a web search using ddgs with fallback to HTML parsing. [REH]"""
        query = params.query.strip()
        if not query:
            return []

        timeout_s = max(0.001, params.timeout_ms / 1000.0)
        safesearch = self._map_safesearch(params.safesearch)
        region = self._map_locale(params.locale or self.cfg.get("SEARCH_LOCALE") or None)

        # Try official client first (non-blocking via thread pool)
        try:
            async def _ddgs_call() -> List[dict]:
                def _worker() -> List[dict]:
                    # Prefer 'ddgs' package first
                    try:
                        mod = importlib.import_module("ddgs")
                        DDGS_cls = getattr(mod, "DDGS", None)
                        if DDGS_cls is not None:
                            with DDGS_cls() as client:
                                gen = client.text(
                                    query,
                                    region=region or "wt-wt",
                                    safesearch=safesearch,
                                    timelimit=None,
                                    max_results=params.max_results,
                                )
                                return list(gen)
                    except Exception:
                        # Try next fallback
                        pass

                    # Fallback to 'duckduckgo_search' package
                    mod = importlib.import_module("duckduckgo_search")
                    DDGS_cls = getattr(mod, "DDGS", None)
                    if DDGS_cls is None:
                        raise ImportError("duckduckgo_search.DDGS not found")
                    with DDGS_cls() as client:
                        gen = client.text(
                            query,
                            region=region or "wt-wt",
                            safesearch=safesearch,
                            timelimit=None,
                            max_results=params.max_results,
                        )
                        return list(gen)

                return await asyncio.to_thread(_worker)

            raw: List[dict] = await asyncio.wait_for(_ddgs_call(), timeout=timeout_s)

            seen: Set[str] = set()
            results: List[SearchResult] = []
            for item in raw:
                title = (item.get("title") or item.get("text") or "").strip()
                url = (item.get("href") or item.get("url") or "").strip()
                if not url or url in seen:
                    continue
                seen.add(url)
                snippet = (item.get("body") or item.get("snippet") or None)
                if isinstance(snippet, str):
                    snippet = snippet.strip() or None
                results.append(SearchResult(title=title or url, url=url, snippet=snippet))
                if len(results) >= params.max_results:
                    break
            return results

        except ImportError:
            # Fallback to HTML provider if ddgs not installed
            logger.warning("duckduckgo_search not installed; falling back to HTML parsing provider. Install 'duckduckgo_search' for better results.")
            return await self._search_via_html(query, params, timeout_s)
        except asyncio.TimeoutError:
            logger.warning("DDGS call timed out")
            return []
        except Exception as e:
            logger.debug(f"DDGS error: {type(e).__name__}: {e}")
            # Fallback path to HTML parsing for resilience
            return await self._search_via_html(query, params, timeout_s)

    def _map_safesearch(self, ss: SafeSearch) -> str:
        """Map SafeSearch enum to ddgs expected string."""
        try:
            if isinstance(ss, SafeSearch):
                return ss.value
            # Defensive: allow string passthrough
            s = str(ss).lower()
            return s if s in {"off", "moderate", "strict"} else "moderate"
        except Exception:
            return "moderate"

    def _map_locale(self, locale: Optional[str]) -> Optional[str]:
        """Map locale to ddgs region code if possible. Defaults to world (wt-wt)."""
        if not locale:
            return None
        loc = locale.replace("_", "-")
        return loc

    async def _search_via_html(self, query: str, params: SearchQueryParams, timeout_s: float) -> SearchResults:
        """Legacy HTML parsing fallback using duckduckgo.com/html. [REH]"""
        # Build query params. Keep minimal for robustness.
        q = {"q": query}
        url = self.endpoint
        if not url.endswith("?"):
            url = url.rstrip("/") + "/?"
        url = url + urlencode(q)

        client: httpx.AsyncClient = await get_search_client()
        try:
            resp = await client.get(url, timeout=timeout_s, headers={
                "User-Agent": "Mozilla/5.0 (compatible; LLMDiscordBot/1.0; +https://example.invalid)",
            })
            resp.raise_for_status()
        except httpx.TimeoutException:
            logger.warning("DDG HTML request timed out")
            return []
        except httpx.HTTPError as e:
            logger.warning(f"DDG HTML HTTP error: {e}")
            return []

        results: List[SearchResult] = []
        try:
            soup = BeautifulSoup(resp.text, "html.parser")
            for a in soup.find_all("a", class_="result__a"):
                title = a.get_text(strip=True)
                href = a.get("href")
                if not href:
                    continue
                snippet = None
                parent = a.find_parent("div", class_="result__body")
                if parent:
                    sn = parent.find("a", class_="result__snippet") or parent.find("div", class_="result__snippet")
                    if sn:
                        snippet = sn.get_text(strip=True)
                results.append(SearchResult(title=title, url=href, snippet=snippet))
                if len(results) >= params.max_results:
                    break
        except Exception as e:
            logger.debug(f"DDG HTML parse error: {e}")
            return []

        return results
