from __future__ import annotations

import asyncio
import json
import os
import re
import html
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx
from bs4 import BeautifulSoup

from .utils.logging import get_logger

logger = get_logger(__name__)

# Defaults and env-driven budgets (additive; no renames) [CMV]
TIER_A_TIMEOUT_S = float(os.getenv("WEBEX_TIER_A_TIMEOUT_S", "6.0"))
TIER_B_TIMEOUT_S = float(os.getenv("WEBEX_TIER_B_TIMEOUT_S", "12.0"))
USER_AGENT = os.getenv(
    "WEBEX_UA_DESKTOP",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
)
ENABLE_TIER_B = os.getenv("WEBEX_ENABLE_TIER_B", "1").strip() not in {
    "0",
    "false",
    "False",
}
ACCEPT_LANGUAGE = os.getenv("WEBEX_ACCEPT_LANGUAGE", "en-US,en;q=0.9")


@dataclass
class ExtractionResult:
    success: bool
    tier_used: str
    canonical_url: Optional[str] = None
    text: Optional[str] = None
    author: Optional[str] = None
    raw_json_present: bool = False
    error: Optional[str] = None

    def to_message(self) -> str:
        if not self.success:
            return f"⚠️ Extraction failed ({self.tier_used}): {self.error or 'unknown error'}"
        text_snippet = (self.text or "").strip()
        if len(text_snippet) > 800:
            text_snippet = text_snippet[:800] + "…"
        parts = []
        if self.canonical_url:
            parts.append(f"URL: {self.canonical_url}")
        if self.author:
            parts.append(f"Author: {self.author}")
        if text_snippet:
            parts.append(f"Text: {text_snippet}")
        if not parts:
            return "🔍 No textual content found."
        return "\n".join(parts)


class WebExtractionService:
    """Tiered web extractor with fast HTTPX path and optional Playwright. [PA][REH]"""

    def __init__(self) -> None:
        self._client: Optional[httpx.AsyncClient] = None
        # Runtime gate for Tier B; auto-disables on fatal env errors [REH]
        self._tier_b_available: bool = ENABLE_TIER_B

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": ACCEPT_LANGUAGE,
                },
                follow_redirects=True,
                timeout=TIER_A_TIMEOUT_S,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def extract(self, url: str) -> ExtractionResult:
        # Tier A: HTTPX fast path [IV]
        try:
            res = await self._tier_a_httpx(url)
            if res.success:
                return res
            logger.info(f"Tier A failed for {url}: {res.error}")
        except httpx.HTTPStatusError as e:  # expected client/server responses
            status = e.response.status_code if getattr(e, "response", None) else "?"
            logger.info(f"Tier A HTTP {status} for {url}: {str(e)[:160]}")
        except (
            httpx.RequestError,
            httpx.TimeoutException,
            httpx.TooManyRedirects,
        ) as e:
            logger.info(f"Tier A network error for {url}: {str(e)[:160]}")
        except Exception as e:  # unexpected
            logger.debug(f"Tier A exception for {url}: {str(e)[:200]}")

        # Tier B: Playwright (optional, runtime-disabled on fatal errors)
        if self._tier_b_available:
            try:
                res_b = await self._tier_b_playwright(url)
                if res_b and res_b.success:
                    return res_b
            except Exception as e:
                logger.info(f"Tier B exception for {url}: {str(e)[:200]}")
                # Auto-disable Tier B on missing shared libs or launch failures
                emsg = str(e).lower()
                if (
                    "error while loading shared libraries" in emsg
                    or "browser has been closed" in emsg
                    or "target page, context or browser has been closed" in emsg
                ):
                    self._tier_b_available = False
                    logger.warning(
                        "🛑 Disabling Tier B (Playwright) due to missing system libraries/launch failure."
                    )

        return ExtractionResult(
            success=False, tier_used="none", error="all tiers failed"
        )

    async def _tier_a_httpx(self, url: str) -> ExtractionResult:
        client = await self._get_client()
        r = await client.get(url)
        r.raise_for_status()
        canonical_url = str(r.url)
        content_type = r.headers.get("content-type", "")
        if "text/html" not in content_type:
            return ExtractionResult(
                success=False,
                tier_used="A",
                error=f"unsupported content-type: {content_type}",
            )
        html = r.text
        parsed = self._parse_html_for_text(html, canonical_url)
        if parsed.get("text"):
            return ExtractionResult(
                success=True,
                tier_used="A",
                canonical_url=canonical_url,
                text=parsed.get("text"),
                author=parsed.get("author"),
                raw_json_present=parsed.get("raw_json_present", False),
            )
        return ExtractionResult(success=False, tier_used="A", error="no text extracted")

    async def _tier_b_playwright(self, url: str) -> Optional[ExtractionResult]:
        try:
            from playwright.async_api import async_playwright
        except Exception:
            return None
        timeout_ms = int(TIER_B_TIMEOUT_S * 1000)
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                context = await browser.new_context(
                    user_agent=USER_AGENT, java_script_enabled=True
                )
                page = await context.new_page()
                await page.route(
                    "**/*",
                    lambda route: asyncio.create_task(route.continue_())
                    if route.request.resource_type
                    in {"document", "xhr", "fetch", "script"}
                    else asyncio.create_task(route.abort()),
                )
                await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                # Try to read meta/og after DOM loaded
                html = await page.content()
                parsed = self._parse_html_for_text(
                    html, await page.evaluate("() => document.location.href")
                )
                if parsed.get("text"):
                    return ExtractionResult(
                        success=True,
                        tier_used="B",
                        canonical_url=parsed.get("canonical_url"),
                        text=parsed.get("text"),
                        author=parsed.get("author"),
                        raw_json_present=parsed.get("raw_json_present", False),
                    )
                return ExtractionResult(
                    success=False, tier_used="B", error="no text extracted"
                )
            finally:
                await browser.close()

    # --- Parsers --- [CSD]
    @staticmethod
    def _normalize_tweet_text(text: str) -> str:
        """Normalize common Twitter/X OG wrapping and whitespace."""
        if not text:
            return ""
        # Unescape HTML entities and trim quotes/wrappers
        t = html.unescape(text).strip()
        # Remove leading/trailing Unicode quotes often used in OG
        t = t.strip("\u201c\u201d\"'")
        # Collapse whitespace
        t = re.sub(r"\s+", " ", t).strip()
        return t

    @staticmethod
    def _parse_html_for_text(html: str, url: str) -> Dict[str, Any]:
        soup = BeautifulSoup(html, "html.parser")
        text_candidates = []
        author = None
        raw_json_present = False

        # Twitter/X specific heuristics [PA]
        if re.search(r"https?://(www\.)?(twitter|x)\.com/", url):
            # Try OpenGraph and Twitter cards first (highest precision for visible text)
            og_desc = soup.find("meta", attrs={"property": "og:description"})
            tw_desc = soup.find("meta", attrs={"name": "twitter:description"})
            for m in (og_desc, tw_desc):
                if m and m.get("content"):
                    norm = WebExtractionService._normalize_tweet_text(
                        m["content"]
                    )  # often contains tweet text
                    if norm:
                        text_candidates.append(norm)

            # Author signals
            og_title = soup.find("meta", attrs={"property": "og:title"})
            tw_creator = soup.find("meta", attrs={"name": "twitter:creator"})
            for m in (tw_creator, og_title):
                if author:
                    break
                if m and m.get("content"):
                    author = (m["content"] or "").strip() or author

            # Detect presence of photos via OG to inform logs (no change in success semantics)
            og_image = soup.find("meta", attrs={"property": "og:image"})
            if og_image and og_image.get("content"):
                logger.debug(
                    "OG image detected for Twitter URL",
                    extra={"event": "webex.twitter.og_image", "detail": {"url": url}},
                )

            # Look for embedded JSON in script tags (prefer ld+json / __NEXT_DATA__)
            keys_preference = (
                "legacy.full_text",
                "full_text",
                "text",
                "articleBody",
                "description",
            )
            for script in soup.find_all("script"):
                t = script.string or script.text or ""
                if not t:
                    continue
                is_ld = script.get("type") == "application/ld+json"
                if (
                    is_ld
                    or "__NEXT_DATA__" in t
                    or "__INITIAL_STATE__" in t
                    or "hydrate" in t
                ):
                    try:
                        raw_json_present = True
                        # Attempt to parse a JSON object within the script content
                        start = t.find("{")
                        end = t.rfind("}")
                        if 0 <= start < end:
                            obj = json.loads(t[start : end + 1])
                            for k in keys_preference:
                                v = WebExtractionService._deep_get(obj, k)
                                if isinstance(v, str):
                                    norm = WebExtractionService._normalize_tweet_text(v)
                                    if norm:
                                        text_candidates.append(norm)
                                        break
                    except Exception:
                        # best-effort only
                        pass
        else:
            # Generic site extraction via meta
            og_desc = soup.find("meta", attrs={"property": "og:description"})
            if og_desc and og_desc.get("content"):
                text_candidates.append(og_desc["content"])
            desc = soup.find("meta", attrs={"name": "description"})
            if desc and desc.get("content"):
                text_candidates.append(desc["content"])

        # Fallback main text: take largest paragraph block
        if not text_candidates:
            paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
            paragraphs = [p for p in paragraphs if len(p) >= 40]
            if paragraphs:
                text_candidates.append(max(paragraphs, key=len))

        # Deduplicate while preserving order
        seen = set()
        uniq_candidates = []
        for c in text_candidates:
            if c not in seen:
                seen.add(c)
                uniq_candidates.append(c)

        text = None
        for cand in uniq_candidates:
            cand = (cand or "").strip()
            if cand:
                text = cand
                break

        return {
            "canonical_url": url,
            "text": text,
            "author": author,
            "raw_json_present": raw_json_present,
        }

    @staticmethod
    def _deep_get(obj: Any, key: str) -> Optional[Any]:
        if isinstance(obj, dict):
            if key in obj:
                return obj[key]
            for v in obj.values():
                r = WebExtractionService._deep_get(v, key)
                if r is not None:
                    return r
        elif isinstance(obj, list):
            for v in obj:
                r = WebExtractionService._deep_get(v, key)
                if r is not None:
                    return r
        return None


# Singleton instance (lightweight) [CA]
web_extractor = WebExtractionService()
