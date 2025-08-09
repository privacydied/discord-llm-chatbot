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
import logging

from bot.util.logging import get_logger
from bot.config import load_config
from ..types import SearchQueryParams, SearchResult, SearchResults, SafeSearch, SearchCategory
from ..factory import get_search_client

logger = get_logger(__name__)


class DDGSearchProvider:
    def __init__(self) -> None:
        self.cfg = load_config()
        self.endpoint = self.cfg.get("DDG_API_ENDPOINT", "https://duckduckgo.com/html/")
        # Suppress verbose logs from external ddgs libs to avoid channel noise
        try:
            logging.getLogger("ddgs").setLevel(logging.WARNING)
            logging.getLogger("duckduckgo_search").setLevel(logging.WARNING)
        except Exception:
            pass

    async def search(self, params: SearchQueryParams) -> SearchResults:
        """Execute a web search using ddgs with fallback to HTML parsing. [REH]"""
        query = params.query.strip()
        if not query:
            return []

        timeout_s = max(0.001, params.timeout_ms / 1000.0)
        safesearch = self._map_safesearch(params.safesearch)
        region = self._map_locale(params.locale or self.cfg.get("SEARCH_LOCALE") or None)
        category = params.category or SearchCategory.TEXT

        # If explicitly requested, use legacy HTML endpoint only. [CMV]
        try:
            force_html = bool(self.cfg.get("DDG_FORCE_HTML"))
        except Exception:
            force_html = False
        if force_html or (isinstance(self.endpoint, str) and "html.duckduckgo.com" in self.endpoint):
            return await self._search_via_html(query, params, timeout_s)

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
                                method_name = self._map_category_to_method(category)
                                fn = getattr(client, method_name, None) or getattr(client, "text")
                                return list(self._invoke_ddgs(fn, query, region, safesearch, params.max_results))
                    except Exception:
                        # Try next fallback
                        pass

                    # Fallback to 'duckduckgo_search' package
                    mod = importlib.import_module("duckduckgo_search")
                    DDGS_cls = getattr(mod, "DDGS", None)
                    if DDGS_cls is None:
                        raise ImportError("duckduckgo_search.DDGS not found")
                    with DDGS_cls() as client:
                        method_name = self._map_category_to_method(category)
                        fn = getattr(client, method_name, None) or getattr(client, "text")
                        return list(self._invoke_ddgs(fn, query, region, safesearch, params.max_results))

                return await asyncio.to_thread(_worker)

            raw: List[dict] = await asyncio.wait_for(_ddgs_call(), timeout=timeout_s)

            # Normalize, deduplicate, and rank
            prelim: List[SearchResult] = []
            for item in raw:
                title, url, snippet = self._extract_item(item)
                if not url:
                    continue
                norm_url, _ = self._normalize_url(url)
                title = title or norm_url or url
                prelim.append(SearchResult(title=title, url=norm_url or url, snippet=(snippet.strip() or None) if isinstance(snippet, str) else snippet))

            return self._dedup_and_rank(prelim, params.max_results)

        except ImportError:
            # Fallback to HTML provider if ddgs not installed
            logger.warning("duckduckgo_search not installed; falling back to HTML parsing provider. Install 'duckduckgo_search' for better results.")
            return await self._search_via_html(query, params, timeout_s)
        except asyncio.TimeoutError:
            logger.warning("DDGS call timed out")
            return []
        except Exception as e:
            name = type(e).__name__
            # Compact log for known ddgs exceptions (e.g., DDGSException)
            if "DDGSException" in name or "DDGSException" in repr(e):
                logger.warning("DDGSException from provider; falling back to HTML parsing")
            else:
                logger.debug(f"DDGS error: {name}: {e}")
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

    def _map_category_to_method(self, cat: SearchCategory) -> str:
        """Map category to DDGS method name. Defaults to 'text'. [CMV]"""
        try:
            mapping = {
                SearchCategory.TEXT: "text",
                SearchCategory.NEWS: "news",
                SearchCategory.IMAGES: "images",
                SearchCategory.VIDEOS: "videos",
            }
            return mapping.get(cat, "text")
        except Exception:
            return "text"

    def _invoke_ddgs(self, fn, query: str, region: Optional[str], safesearch: str, max_results: int):
        """Invoke a DDGS method with tolerant signature handling. [REH]"""
        # Try full signature, then progressively simpler fallbacks to avoid TypeError across versions.
        try:
            return fn(
                query,
                region=region or "wt-wt",
                safesearch=safesearch,
                timelimit=None,
                max_results=max_results,
            )
        except TypeError:
            try:
                return fn(
                    query,
                    region=region or "wt-wt",
                    safesearch=safesearch,
                    max_results=max_results,
                )
            except TypeError:
                try:
                    return fn(
                        query,
                        safesearch=safesearch,
                        max_results=max_results,
                    )
                except TypeError:
                    return fn(query, max_results=max_results)

    def _extract_item(self, item: dict) -> tuple[str, str, Optional[str]]:
        """Best-effort field extraction across ddgs variants. [IV][REH]"""
        title = (item.get("title") or item.get("text") or item.get("name") or "").strip()
        # Try common URL keys
        url = (
            item.get("href")
            or item.get("url")
            or item.get("link")
            or item.get("content_url")
            or item.get("source")
            or ""
        )
        url = url.strip() if isinstance(url, str) else ""
        snippet = (
            item.get("body")
            or item.get("snippet")
            or item.get("description")
            or item.get("content")
            or item.get("excerpt")
            or None
        )
        if isinstance(snippet, str):
            snippet = snippet.strip() or None
        return title, url, snippet

    def _normalize_url(self, url: str) -> tuple[Optional[str], Optional[str]]:
        """Normalize URL and produce a deduplication key. [IV][CMV]"""
        try:
            from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

            if not url or not isinstance(url, str):
                return None, None

            u = urlparse(url)
            if not u.scheme or not u.netloc:
                return None, None

            scheme = u.scheme.lower()
            netloc = u.netloc.lower()
            # Strip default ports
            if netloc.endswith(":80") and scheme == "http":
                netloc = netloc[:-3]
            if netloc.endswith(":443") and scheme == "https":
                netloc = netloc[:-4]

            path = u.path or "/"
            # Remove trailing slash except root
            if path != "/" and path.endswith("/"):
                path = path[:-1]

            # Remove tracking params
            TRACKERS = {
                "utm_source",
                "utm_medium",
                "utm_campaign",
                "utm_term",
                "utm_content",
                "utm_id",
                "gclid",
                "fbclid",
                "igshid",
                "ved",
                "ei",
                "oq",
                "sxsrf",
            }
            q_pairs = [(k, v) for k, v in parse_qsl(u.query, keep_blank_values=False) if k not in TRACKERS]
            q_pairs.sort()
            query = urlencode(q_pairs)

            fragment = ""  # strip fragment
            norm = urlunparse((scheme, netloc, path, "", query, fragment))

            # Dedup key ignores scheme and leading www.
            netloc_key = netloc[4:] if netloc.startswith("www.") else netloc
            dedup_key = f"{netloc_key}{path}?{query}" if query else f"{netloc_key}{path}"
            return norm, dedup_key
        except Exception:
            return url, url

    def _score(self, r: SearchResult) -> int:
        """Simple heuristic ranking score. [PA]"""
        try:
            score = 0
            if r.snippet:
                score += 2
            if r.url.startswith("https://"):
                score += 1
            # Prefer reputable TLDs a tiny bit
            for tld in (".edu", ".gov", ".org"):
                if r.url.endswith(tld):
                    score += 1
                    break
            # Penalize very long URLs
            score -= min(len(r.url) // 120, 2)
            return score
        except Exception:
            return 0

    def _dedup_and_rank(self, items: List[SearchResult], max_results: int) -> SearchResults:
        """Deduplicate by normalized key and rank by heuristic while keeping stability. [CA][REH]"""
        seen: Set[str] = set()
        pruned: List[SearchResult] = []
        keys: List[str] = []
        for r in items:
            norm_url, key = self._normalize_url(r.url)
            if key and key not in seen:
                seen.add(key)
                if norm_url and norm_url != r.url:
                    r = SearchResult(title=r.title, url=norm_url, snippet=r.snippet, favicon=r.favicon)
                pruned.append(r)
                keys.append(key)
        # Rank with stable sort
        indexed = list(enumerate(pruned))
        indexed.sort(key=lambda kv: (-self._score(kv[1]), kv[0]))
        final = [kv[1] for kv in indexed][:max_results]
        return final

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
                norm_url, _ = self._normalize_url(href)
                results.append(SearchResult(title=title or (norm_url or href), url=(norm_url or href), snippet=snippet))
        except Exception as e:
            logger.debug(f"DDG HTML parse error: {e}")
            return []

        return self._dedup_and_rank(results, params.max_results)
