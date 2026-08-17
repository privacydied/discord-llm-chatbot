from __future__ import annotations

import contextlib
import asyncio
import html
import json
import os
import re
from dataclasses import dataclass
from typing import Any

import httpx
from bs4 import BeautifulSoup
from urllib.parse import urlsplit, urlunsplit

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
# jina.ai is per-IP rate-limited: rapid repeated requests push it into a
# sustained "paywall landing page" phase, which is WORSE than a single request.
# So we do few retries with spacing -- hammering defeats the purpose. The bare
# r.jina.ai/<url> form is the reliable one; the /http/ form 422s from this host
# and is intentionally not used. [PAY]
TIER_C_RETRIES = int(os.getenv("WEBEX_TIER_C_RETRIES", "1"))
TIER_C_RETRY_DELAY_S = float(os.getenv("WEBEX_TIER_C_RETRY_DELAY_S", "2.0"))
# Minimum word count for a reader-proxy response to count as a real article.
# Landing pages / teasers have very few words of body and fail this gate so the
# retry/fallback chain can recover the actual article. [PAY]
TIER_C_MIN_WORDS = int(os.getenv("WEBEX_TIER_C_MIN_WORDS", "250"))
# Reader proxies can be slow/flaky; only use them for clearly article-like URLs.
TIER_C_MAX_TEXT_CHARS = int(os.getenv("WEBEX_TIER_C_MAX_CHARS", "60000"))
# Below this many chars, a Tier A (httpx) result is treated as a thin teaser
# (typical paywall behavior: HTTP 200 with only a headline + prompt) and the
# extraction cascades to Tier B/C instead of being returned as "success".
TIER_A_MIN_CHARS = int(os.getenv("WEBEX_TIER_A_MIN_CHARS", "800"))

# ---------------------------------------------------------------------------
# Bot-wall / challenge-page detection [PAY][REH]
# ---------------------------------------------------------------------------
# Substrings (lowercased) that identify a "security check" / CAPTCHA interstitial
# rather than article content. When any tier fetches a body containing one of
# these, the page is a bot-wall and there is no point launching heavier tiers
# (Playwright) or retrying -- skip straight to a clear failure message. [PAY]
BOT_WALL_MARKERS: tuple[str, ...] = (
    "one more step",
    "complete the security check",
    "verify you are a human",
    "just a moment",
    "checking your browser",
    "enable javascript and cookies to continue",
    "please verify you are a human",
)

# Known capture / archive mirrors that hard-block automated access from this
# server's egress IP (HTTP 429 + "One more step" interstitial). Surfacing a
# specific, actionable message beats the generic extraction-failure text. [PAY]
BLOCKED_HOST_SUFFIXES: tuple[str, ...] = (
    "archive.is",
    "archive.ph",
    "archive.today",
    "archive.fo",
    "archive.li",
    "archive.vn",
)

# Human-facing messages for bot-wall failures. [PAY]
BOT_WALL_GENERIC_MSG = (
    "This page is behind a bot-check (security challenge / CAPTCHA) that this "
    "server can't pass automatically."
)
BOT_WALL_BLOCKED_HOST_MSG = (
    "this capture host blocks automated access from this server; try the "
    "original source URL or web.archive.org if a public snapshot exists"
)

# Optional Wayback fallback: when the tiered extractor fails and archive.org has
# a public snapshot of the target, fetch that instead. Gated by env so it can be
# disabled; only ever runs on the failure path (one availability API call + one
# fetch, both bounded). [PAY]
ENABLE_WAYBACK_FALLBACK = os.getenv("WEBEX_ENABLE_WAYBACK_FALLBACK", "1").strip() not in {
    "0",
    "false",
    "False",
}
WAYBACK_AVAILABILITY_URL = "https://archive.org/wayback/available"
WAYBACK_FETCH_TIMEOUT_S = float(os.getenv("WEBEX_WAYBACK_TIMEOUT_S", "10.0"))


def is_bot_wall(text: str | None) -> str | None:
    """Return the matched challenge marker (lowercased) if ``text`` looks like a
    bot-wall interstitial, else None. Cheap, body-only, no network. [PAY]"""
    if not text:
        return None
    lowered = (text or "").lower()
    for marker in BOT_WALL_MARKERS:
        if marker in lowered:
            return marker
    return None


def _host_of(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower()
    except Exception:
        return ""


def is_blocked_host(url: str) -> bool:
    """True when ``url`` points at a known bot-wall capture host. [PAY]"""
    host = _host_of(url)
    return any(host == s or host.endswith("." + s) for s in BLOCKED_HOST_SUFFIXES)


def _bot_wall_marker_from_error(error: str | None) -> str | None:
    if not error or not error.startswith("bot_wall:"):
        return None
    return error[len("bot_wall:"):] or "challenge"


async def _wayback_snapshot(url: str) -> str | None:
    """Return the closest public Wayback snapshot URL for ``url``, or None.

    Uses the archive.org availability API (one bounded GET). Never raises. [PAY]
    """
    api = f"{WAYBACK_AVAILABILITY_URL}?url={url}"
    try:
        async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as c:
            r = await c.get(api)
            r.raise_for_status()
            data = r.json()
        snap = (data.get("archived_snapshots") or {}).get("closest") or {}
        if snap.get("available") and snap.get("url"):
            return str(snap["url"])
    except Exception as exc:  # noqa: BLE001 - best-effort fallback
        logger.debug(f"Wayback availability lookup failed for {url}: {exc!r}")
    return None


@dataclass
class ExtractionResult:
    success: bool
    tier_used: str
    canonical_url: str | None = None
    text: str | None = None
    author: str | None = None
    raw_json_present: bool = False
    error: str | None = None
    # Set when the failure is a bot-wall/challenge page; carries the matched
    # marker so callers can surface a specific message instead of the generic
    # extraction-failure text. [PAY]
    bot_wall_marker: str | None = None

    def to_message(self) -> str:
        if not self.success:
            if self.bot_wall_marker is not None:
                if is_blocked_host(self.canonical_url or ""):
                    return (
                        f"⚠️ Extraction failed ({self.tier_used}): "
                        f"{BOT_WALL_BLOCKED_HOST_MSG}"
                    )
                return (
                    f"⚠️ Extraction failed ({self.tier_used}): {BOT_WALL_GENERIC_MSG}"
                )
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

    @staticmethod
    def _bot_wall_result(url: str, tier: str, marker: str) -> ExtractionResult:
        """Build a bot-wall failure result for ``url`` (host is the original
        target, so the blocklist message logic keys off it). [PAY]"""
        return ExtractionResult(
            success=False,
            tier_used=tier,
            canonical_url=url,
            error=f"bot_wall:{marker}",
            bot_wall_marker=marker,
        )

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

        res = await self._extract_via(url)

        # Optional Wayback fallback: only on a hard failure and only when a public
        # snapshot exists. One bounded availability API call + one bounded fetch;
        # if the snapshot extractor also fails we return the ORIGINAL result so the
        # user still gets the (specific) bot-wall message rather than a Wayback error. [PAY]
        if not res.success and ENABLE_WAYBACK_FALLBACK and not res.bot_wall_marker:
            snap = await _wayback_snapshot(url)
            if snap:
                logger.info(f"🌐 Wayback snapshot found for {url}: {snap}; trying it")
                snap_res = await self._extract_via(snap, original_url=url)
                if snap_res.success:
                    logger.info(f"🌐 Wayback fallback succeeded for {url}")
                    return snap_res
                # Snapshot was also unreadable -- keep the original failure so the
                # message reflects the requested URL, not the snapshot.
                logger.info(f"🌐 Wayback fallback also failed for {url}: {snap_res.error}")

        return res

    async def _extract_via(self, url: str, original_url: str | None = None) -> ExtractionResult:
        """Run the tier cascade (A -> C -> B) for ``url``.

        ``original_url`` is the user-facing target (kept on bot-wall results so
        the blocklist message keys off the requested host, not a Wayback mirror).
        Fast-fails the whole cascade the moment any tier returns a bot-wall
        interstitial instead of spending ~26s on a doomed Playwright launch. [PAY]
        """
        last_error: str | None = None
        last_tier = "none"

        try:
            res = await self._tier_a_httpx(url)
            marker = is_bot_wall(res.text)
            if marker is not None:
                logger.info(f"🛑 Tier A returned a bot-wall interstitial ({marker}) for {url}; skipping B/C")
                return self._bot_wall_result(original_url or url, "A", marker)
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
                marker = is_bot_wall(res_c.text if res_c is not None else None)
                if marker is not None:
                    logger.info(f"🛑 Tier C returned a bot-wall interstitial ({marker}) for {url}; skipping B")
                    return self._bot_wall_result(original_url or url, "C", marker)
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
                    marker = is_bot_wall(res_b.text)
                    if marker is not None:
                        logger.info(f"🛑 Tier B returned a bot-wall interstitial ({marker}) for {url}")
                        return self._bot_wall_result(original_url or url, "B", marker)
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
            canonical_url=original_url or url,
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
        load via --load-extension. jina.ai is flaky per egress IP, so we try
        the primary endpoint, then a fallback endpoint, and reject responses
        that still look like the paywalled landing page. [PAY]
        """
        from urllib.parse import urlsplit

        parts = urlsplit(url)
        # Two jina.ai shapes exist: bare r.jina.ai/<url> (pass https:// through)
        # and r.jina.ai/http/<url> (target scheme must be http://). Build the
        # target per-base so we don't get 422/403 from a mismatched scheme.
        def _target_for(base: str) -> str:
            if base.rstrip("/").endswith("/http"):
                return urlunsplit(("http", parts.netloc, parts.path, parts.query, parts.fragment))
            return url

        timeout = httpx.Timeout(TIER_C_TIMEOUT_S, connect=10.0)
        # Fresh client for the reader hop. NOTE: send NO custom User-Agent and
        # NO restrictive Accept header -- jina.ai returns 403 when it sees a
        # browser-like UA (e.g. Chrome) or an Accept of text/plain/markdown.
        # The default python-httpx UA is accepted. [PAY]

        async def _try(base: str) -> ExtractionResult | None:
            reader_url = f"{base}{_target_for(base)}"
            async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as rc:
                r = await rc.get(reader_url)
            r.raise_for_status()
            raw = r.text or ""
            if not raw.strip():
                return ExtractionResult(success=False, tier_used="C", error="empty reader response")
            text = self._strip_reader_wrapper(raw).strip()[:TIER_C_MAX_TEXT_CHARS]
            if len(text) < 40:
                return ExtractionResult(success=False, tier_used="C", error="reader returned no article body")
            # A reader-proxy response is only useful if it carries a real article
            # body. jina.ai intermittently returns the paywalled landing page
            # (teaser + nav + "[Subscribe now]" CTA) instead of the article; that
            # response has very few words of body. Gate success on a minimum word
            # count so landing pages fail and the retry/fallback chain can recover
            # the real article. Real articles (which may end with a subscribe CTA)
            # easily clear the threshold. [PAY]
            if len(text.split()) < TIER_C_MIN_WORDS:
                return ExtractionResult(
                    success=False, tier_used="C",
                    error="reader returned paywall/teaser (insufficient body)",
                )
            return ExtractionResult(success=True, tier_used="C", canonical_url=url, text=text, author=None, raw_json_present=False)

        # jina.ai is flaky per egress IP: it intermittently returns the paywalled
        # landing page instead of the article. Retry the endpoint set a few times
        # (the article usually comes through on a later attempt). [PAY]
        last_err = "reader proxy not attempted"
        for attempt in range(max(1, int(TIER_C_RETRIES))):
            try:
                res = await _try(TIER_C_READER_BASE)
                if res is not None and res.success:
                    return res
                last_err = (res.error if res else "no response") or last_err
                logger.info(f"Tier C endpoint failed (attempt {attempt + 1}) for {url}: {last_err}")
            except Exception as e:
                last_err = f"exception:{e.__class__.__name__}"
                logger.info(f"Tier C endpoint exception (attempt {attempt + 1}) for {url}: {str(e)[:160]}")
            if attempt < max(1, int(TIER_C_RETRIES)) - 1:
                await asyncio.sleep(TIER_C_RETRY_DELAY_S)
        return ExtractionResult(success=False, tier_used="C", error=last_err)

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
