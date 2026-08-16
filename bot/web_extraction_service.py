from __future__ import annotations

import contextlib
import html
import json
import os
import re
from dataclasses import dataclass
from typing import Any

import httpx
from bs4 import BeautifulSoup

from .utils.logging import get_logger
from .utils.playwright_helpers import connect_browser as _pw_connect_browser

logger = get_logger(__name__)

# Defaults and env-driven budgets (additive; no renames) [CMV]
TIER_A_TIMEOUT_S = float(os.getenv("WEBEX_TIER_A_TIMEOUT_S", "6.0"))
TIER_B_TIMEOUT_S = float(os.getenv("WEBEX_TIER_B_TIMEOUT_S", "12.0"))
USER_AGENT = os.getenv(
    "WEBEX_UA_DESKTOP",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
)
ENABLE_TIER_B = os.getenv("WEBEX_ENABLE_TIER_B", "1").strip() not in {
    "0",
    "false",
    "False",
}
ACCEPT_LANGUAGE = os.getenv("WEBEX_ACCEPT_LANGUAGE", "en-US,en;q=0.9")

# Tier C: server-side reader proxy that bypasses soft paywalls and returns
# clean article text. Default r.jina.ai; override WEBEX_TIER_C_READER to point
# at a self-hosted reader (e.g. a local r.jina.ai or mercure instance).
# This is the headless-safe substitute for a browser paywall-bypass extension,
# which the Playwright/Chrome-for-Testing 145 build cannot load via
# --load-extension. [PAY]
ENABLE_TIER_C = os.getenv("WEBEX_ENABLE_TIER_C", "1").strip() not in {
    "0",
    "false",
    "False",
}
TIER_C_TIMEOUT_S = float(os.getenv("WEBEX_TIER_C_TIMEOUT_S", "15.0"))
TIER_C_READER_BASE = os.getenv(
    "WEBEX_TIER_C_READER",
    "https://r.jina.ai/",
).rstrip("/") + "/"
# Reader proxies can be slow/flaky; only use them for clearly article-like URLs.
TIER_C_MAX_TEXT_CHARS = int(os.getenv("WEBEX_TIER_C_MAX_CHARS", "60000"))
# Below this many chars, a Tier A (httpx) result is treated as a thin teaser
# (typical paywall behavior: HTTP 200 with only a headline + prompt) and the
# extraction cascades to Tier B/C instead of being returned as "success".
TIER_A_MIN_CHARS = int(os.getenv("WEBEX_TIER_A_MIN_CHARS", "800"))


@dataclass
class ExtractionResult:
    success: bool
    tier_used: str
    canonical_url: str | None = None
    text: str | None = None
    author: str | None = None
    raw_json_present: bool = False
    error: str | None = None

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
    """Tiered web extractor with fast HTTPX path and optional Playwright. [PA][REH]."""

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None
        # Runtime gate for Tier B; auto-disables on fatal env errors [REH]
        self._tier_b_available: bool = ENABLE_TIER_B

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": ("text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf;q=0.9,*/*;q=0.8"),
                    "Accept-Language": ACCEPT_LANGUAGE,
                    "Accept-Encoding": "gzip, deflate, br",
                    "DNT": "1",
                    "Upgrade-Insecure-Requests": "1",
                },
                follow_redirects=True,
                timeout=TIER_A_TIMEOUT_S,
            )
        return self._client

    @staticmethod
    def _is_playwright_fatal_error(message: str) -> bool:
        """Return True when the error indicates the browser tier is
        unrecoverable for this process lifetime (e.g. missing system
        libraries, missing binary).

        Version-mismatch (428) and transient connection errors are NOT
        fatal -- they should be re-attempted once the version is aligned
        or the server comes back.
        """
        m = (message or "").lower()
        fatal_markers = (
            "error while loading shared libraries",
            "executable doesn't exist",
            "executable doesn't exist at",
            "failed to launch",
            "browserType.launch:",
            "chromium.launch:",
            "playwright install",
            "cannot find module",
            "no such file or directory",
        )
        return any(tok in m for tok in fatal_markers)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def extract(self, url: str) -> ExtractionResult:
        from bot.url_safety import (
            UrlSafetyError,
            validate_url_with_dns,
        )

        # SSRF / URL safety validation before any fetch
        try:
            await validate_url_with_dns(url)
        except UrlSafetyError as exc:
            logger.warning("URL safety blocked extraction: %s", exc)
            return ExtractionResult(success=False, tier_used="none", error=f"URL blocked: {exc}")

        last_error: str | None = None
        last_tier = "none"

        try:
            res = await self._tier_a_httpx(url)
            if res.success:
                # A paywalled page often returns HTTP 200 with only a short
                # teaser (e.g. 122 chars). Treat a thin Tier A result as a soft
                # failure so we cascade to Tier B (Playwright) and Tier C
                # (reader proxy) instead of handing the LLM an empty article.
                from bot.news import thin_content

                if thin_content.assess(res.text, min_chars=TIER_A_MIN_CHARS).is_thin:
                    logger.info(
                        f"Tier A success but thin ({len(res.text or '')} chars) for {url}; cascading to B/C"
                    )
                    res = ExtractionResult(
                        success=False,
                        tier_used="A",
                        error=f"thin_content:{len(res.text or '')}",
                    )
                else:
                    return res
            last_error = res.error
            last_tier = "A"
            logger.info(f"Tier A failed for {url}: {res.error}")
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if getattr(e, "response", None) else "?"
            last_error = f"http_status:{status}"
            last_tier = "A"
            logger.info(f"Tier A HTTP {status} for {url}: {str(e)[:160]}")
        except (
            httpx.RequestError,
            httpx.TimeoutException,
            httpx.TooManyRedirects,
        ) as e:
            last_error = f"network_error:{e.__class__.__name__}"
            last_tier = "A"
            logger.info(f"Tier A network error for {url}: {str(e)[:160]}")
        except Exception as e:
            last_error = f"exception:{e.__class__.__name__}"
            last_tier = "A"
            logger.debug(f"Tier A exception for {url}: {str(e)[:200]}")

        # Tier C first: the reader proxy is the paywall-bypass specialist. A
        # paywalled page renders fine in headless Chromium (Tier B) but still
        # shows the wall, so returning Tier B's result would block the bypass.
        # Prefer Tier C whenever it yields real article text; fall back to
        # Tier B (Playwright) for URLs the reader proxy can't handle. [PAY]
        if ENABLE_TIER_C:
            try:
                res_c = await self._tier_c_reader(url)
                if res_c is not None and res_c.success:
                    from bot.news import thin_content

                    if not thin_content.assess(res_c.text, min_chars=TIER_A_MIN_CHARS).is_thin:
                        return res_c
                    last_error = res_c.error or last_error
                    last_tier = "C"
                    logger.info(f"Tier C returned thin content for {url}; trying Tier B")
            except Exception as e:
                last_error = f"exception:{e.__class__.__name__}"
                last_tier = "C"
                logger.info(f"Tier C exception for {url}: {str(e)[:200]}")

        if self._tier_b_available:
            try:
                res_b = await self._tier_b_playwright(url)
                if res_b is not None:
                    if res_b.success:
                        return res_b
                    last_error = res_b.error or last_error
                    last_tier = "B"
            except Exception as e:
                last_error = f"exception:{e.__class__.__name__}"
                last_tier = "B"
                logger.info(f"Tier B exception for {url}: {str(e)[:200]}")
                if self._is_playwright_fatal_error(str(e)):
                    self._tier_b_available = False
                    logger.warning("🛑 Disabling Tier B (Playwright) due to runtime/launch failure.")

        return ExtractionResult(
            success=False,
            tier_used=last_tier,
            error=last_error or "all tiers failed",
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

    async def _tier_b_playwright(self, url: str) -> ExtractionResult | None:
        try:
            from playwright.async_api import async_playwright
        except Exception as exc:
            logger.debug(f"Playwright import failed: {exc}")
            return None
        timeout_ms = int(TIER_B_TIMEOUT_S * 1000)

        try:
            async with async_playwright() as p:
                browser = await _pw_connect_browser(p.chromium)
                if browser is None:
                    # No PW_SERVER_URL configured; browser tier is unavailable.
                    return None
                context = None
                try:
                    context = await browser.new_context(user_agent=USER_AGENT, java_script_enabled=True)
                    page = await context.new_page()
                    page.set_default_timeout(timeout_ms)

                    async def _route_handler(route, request) -> None:
                        try:
                            if request.resource_type in {
                                "document",
                                "xhr",
                                "fetch",
                                "script",
                            }:
                                await route.continue_()
                            else:
                                await route.abort()
                        except Exception as exc:
                            logger.debug(f"Route handler failed: {exc}")
                            try:
                                await route.abort()
                            except Exception as exc2:
                                logger.debug(f"Route abort failed: {exc2}")

                    await page.route("**/*", _route_handler)
                    await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                    html_doc = await page.content()
                    final_url = await page.evaluate("() => document.location.href")
                    parsed = self._parse_html_for_text(html_doc, final_url)
                    if parsed.get("text"):
                        return ExtractionResult(
                            success=True,
                            tier_used="B",
                            canonical_url=parsed.get("canonical_url"),
                            text=parsed.get("text"),
                            author=parsed.get("author"),
                            raw_json_present=parsed.get("raw_json_present", False),
                        )
                    return ExtractionResult(success=False, tier_used="B", error="no text extracted")
                finally:
                    if context is not None:
                        with contextlib.suppress(Exception):
                            await context.close()
        except Exception as exc:
            logger.warning(f"Tier B Playwright failed for {url}: {exc}")
            raise

    async def _tier_c_reader(self, url: str) -> ExtractionResult | None:
        """Tier C: fetch the article through a server-side reader proxy.

        The proxy (default r.jina.ai) strips paywalls and returns cleaned
        article text. This is the headless-safe stand-in for a browser
        paywall-bypass extension, which the automation Chromium build cannot
        load via --load-extension. [PAY]
        """
        # Build the reader URL. Two common jina.ai shapes exist depending on
        # egress/anti-abuse: the bare form  r.jina.ai/<url>  (pass the original
        # https:// URL through) and the  r.jina.ai/http/<url>  form (which
        # requires the target scheme to be http://). Detect which the configured
        # base expects so we don't get 422/403 from a mismatched scheme.
        from urllib.parse import urlsplit, urlunsplit

        parts = urlsplit(url)
        if TIER_C_READER_BASE.rstrip("/").endswith("/http"):
            target = urlunsplit(("http", parts.netloc, parts.path, parts.query, parts.fragment))
        else:
            target = url
        reader_url = f"{TIER_C_READER_BASE}{target}"
        timeout = httpx.Timeout(TIER_C_TIMEOUT_S, connect=10.0)
        # Fresh client for the reader hop. NOTE: send NO custom User-Agent and
        # NO restrictive Accept header -- jina.ai returns 403 when it sees a
        # browser-like UA (e.g. Chrome) or an Accept of text/plain/markdown.
        # The default python-httpx UA is accepted. [PAY]
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=timeout,
        ) as rc:
            r = await rc.get(reader_url)
        r.raise_for_status()
        raw = r.text or ""
        if not raw.strip():
            return ExtractionResult(success=False, tier_used="C", error="empty reader response")
        text = self._strip_reader_wrapper(raw)
        text = text.strip()[:TIER_C_MAX_TEXT_CHARS]
        if len(text) < 40:
            return ExtractionResult(success=False, tier_used="C", error="reader returned no article body")
        return ExtractionResult(
            success=True,
            tier_used="C",
            canonical_url=url,
            text=text,
            author=None,
            raw_json_present=False,
        )

    @staticmethod
    def _strip_reader_wrapper(raw: str) -> str:
        """Extract the article body from a reader-proxy response.

        r.jina.ai returns markdown that begins with metadata lines ('Title:',
        'URL Source:', 'Published Time:') and often a 'Markdown Content:'
        label, followed by the rendered page (which still includes the site's
        nav menu using '## ' H2 links). The actual article starts at its H1
        title ('# Heading'). So we trim from the first single-'# ' heading line
        (after any nav), which is the article body -- this works whether or not
        the 'Markdown Content:' label is present. Line-ending tolerant.
        """
        # First single-'# ' H1 heading line = article start (nav uses '## ').
        for i in range(len(raw)):
            if raw[i] in ("\n", "\r") and raw[i + 1 : i + 3] == "# ":
                return raw[i + 1:].strip()
        # Fallback: drop everything up to and including a 'Markdown Content:' label.
        marker = "Markdown Content:"
        idx = raw.find(marker)
        if idx != -1:
            return raw[idx + len(marker):].strip()
        return raw.strip()

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
        return re.sub(r"\s+", " ", t).strip()

    @staticmethod
    def _parse_html_for_text(html: str, url: str) -> dict[str, Any]:
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
                    norm = WebExtractionService._normalize_tweet_text(m["content"])  # often contains tweet text
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
                if is_ld or "__NEXT_DATA__" in t or "__INITIAL_STATE__" in t or "hydrate" in t:
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
                    except Exception as exc:
                        # best-effort only
                        logger.debug(f"tweet text extraction failed: {exc}")
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
    def _deep_get(obj: Any, key: str) -> Any | None:
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
