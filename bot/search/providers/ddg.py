"""
DuckDuckGo HTML search provider (no official API).
Parses DDG HTML results for basic title/url/snippet extraction.
[CA][REH][IV][PA]
"""
from __future__ import annotations

from typing import List
from urllib.parse import urlencode

import httpx
from bs4 import BeautifulSoup  # beautifulsoup4 is in requirements

from bot.util.logging import get_logger
from bot.config import load_config
from ..types import SearchQueryParams, SearchResult, SearchResults
from ..factory import get_search_client

logger = get_logger(__name__)


class DDGSearchProvider:
    def __init__(self) -> None:
        self.cfg = load_config()
        self.endpoint = self.cfg.get("DDG_API_ENDPOINT", "https://duckduckgo.com/html/")

    async def search(self, params: SearchQueryParams) -> SearchResults:
        query = params.query.strip()
        if not query:
            return []

        # Build query params. We deliberately keep this minimal for robustness. [KBT]
        q = {"q": query}
        # locale and safesearch knobs may be mapped later when verified
        url = self.endpoint
        if not url.endswith("?"):
            url = url.rstrip("/") + "/?"
        url = url + urlencode(q)

        timeout_s = max(0.001, params.timeout_ms / 1000.0)

        client: httpx.AsyncClient = await get_search_client()
        try:
            resp = await client.get(url, timeout=timeout_s, headers={
                "User-Agent": "Mozilla/5.0 (compatible; LLMDiscordBot/1.0; +https://example.invalid)",
            })
            resp.raise_for_status()
        except httpx.TimeoutException:
            logger.warning("DDG request timed out")
            return []
        except httpx.HTTPError as e:
            logger.warning(f"DDG HTTP error: {e}")
            return []

        results: List[SearchResult] = []
        try:
            soup = BeautifulSoup(resp.text, "html.parser")
            # Try common DDG HTML structure
            for a in soup.find_all("a", class_="result__a"):
                title = a.get_text(strip=True)
                href = a.get("href")
                if not href:
                    continue
                # attempt to get snippet
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
            logger.debug(f"DDG parse error: {e}")
            return []

        return results
