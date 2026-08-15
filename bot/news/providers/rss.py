"""Publisher syndication-feed provider.
[CA][REH][SFT][CMV][PA].

Many publishers syndicate article text in their own RSS/Atom feeds -- content
they publish deliberately for downstream readers. This provider discovers the
feed a page declares, then looks the article up in it by canonical URL.

Every fetched URL is SSRF-validated, because feed hrefs come from third-party
page markup and are therefore untrusted. [SFT]
"""

from __future__ import annotations

from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from bot.http_client import RequestConfig, get_http_client
from bot.utils.logging import get_logger

from ..base import AbstractNewsProvider
from ..feed_parse import find_entry, parse_feed
from ..types import NewsArticle

logger = get_logger(__name__)

# Feed MIME types announced via <link rel="alternate">. [CMV]
FEED_TYPES: frozenset[str] = frozenset({"application/rss+xml", "application/atom+xml", "application/feed+json", "text/xml", "application/xml"})

# Conventional feed locations, tried when a page declares none. [CMV]
WELL_KNOWN_PATHS: tuple[str, ...] = ("/feed", "/rss", "/rss.xml", "/feed.xml", "/index.xml", "/atom.xml")

# Bound the work per article so a hostile page cannot fan out fetches. [PA][CMV]
MAX_FEEDS_TRIED = 4

# Refuse absurd feed payloads before parsing them. [SFT][CMV]
MAX_FEED_BYTES = 5 * 1024 * 1024

_REQUEST = RequestConfig(connect_timeout=2.0, read_timeout=6.0, total_timeout=8.0, max_retries=1)

_BROWSER_UA = "Mozilla/5.0 (compatible; DiscordBot/1.0; +https://discord.com)"


async def _safe_get(url: str) -> bytes | None:
    """GET a URL after SSRF validation. Returns None on any failure. [REH][SFT]"""
    from bot.url_safety import UrlSafetyError, validate_url_with_dns

    try:
        await validate_url_with_dns(url)
    except UrlSafetyError as exc:
        logger.debug("rss.blocked url=%s reason=%s", url[:120], exc)
        return None
    except Exception as exc:  # [REH]
        logger.debug("rss.validate_failed url=%s error=%s", url[:120], exc)
        return None

    try:
        client = await get_http_client()
        response = await client.get(url, config=_REQUEST, headers={"User-Agent": _BROWSER_UA})
    except Exception as exc:  # [REH]
        logger.debug("rss.fetch_failed url=%s error=%s", url[:120], exc)
        return None

    if response.status_code != 200:
        return None
    content = response.content
    return content if content and len(content) <= MAX_FEED_BYTES else None


def discover_feeds(html: str, base_url: str) -> list[str]:
    """Return feed URLs declared by a page, most-specific first."""
    if not html:
        return []
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception as exc:  # [REH]
        logger.debug("rss.discover_parse_failed: %s", exc)
        return []

    found: list[str] = []
    for link in soup.find_all("link"):
        rels = {r.lower() for r in (link.get("rel") or [])}
        if "alternate" not in rels:
            continue
        if (link.get("type") or "").lower() not in FEED_TYPES:
            continue
        href = (link.get("href") or "").strip()
        if href:
            resolved = urljoin(base_url, href)
            if resolved not in found:
                found.append(resolved)
    return found


def well_known_feeds(url: str) -> list[str]:
    """Return conventional feed locations for the URL's origin."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return []
    if not parsed.scheme or not parsed.netloc:
        return []
    origin = f"{parsed.scheme}://{parsed.netloc}"
    return [f"{origin}{path}" for path in WELL_KNOWN_PATHS]


class RssNewsProvider(AbstractNewsProvider):
    """Resolves article text from a publisher's own syndication feed."""

    name = "rss"

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    def supports(self, url: str) -> bool:
        return self._enabled and url.startswith(("http://", "https://"))

    async def fetch(self, url: str) -> NewsArticle | None:
        """Return the article from a declared feed, or None. Never raises. [REH]"""
        if not self.supports(url):
            return None
        try:
            return await self._resolve(url)
        except Exception as exc:  # [REH] provider failure must not break the chain
            logger.debug("rss.fetch failed url=%s error=%s", url[:120], exc)
            return None

    async def _candidate_feeds(self, url: str) -> list[str]:
        page = await _safe_get(url)
        declared = discover_feeds(page.decode("utf-8", errors="replace"), url) if page else []
        candidates = declared + [f for f in well_known_feeds(url) if f not in declared]
        return candidates[:MAX_FEEDS_TRIED]

    async def _resolve(self, url: str) -> NewsArticle | None:
        for feed_url in await self._candidate_feeds(url):
            raw = await _safe_get(feed_url)
            if not raw:
                continue
            entry = find_entry(parse_feed(raw), url)
            if entry and entry.body.strip():
                logger.info("rss.hit feed=%s url=%s chars=%d", feed_url[:120], url[:120], len(entry.body))
                return NewsArticle(
                    url=url,
                    body=entry.body,
                    title=entry.title,
                    author=entry.author,
                    published=entry.published,
                    provider=self.name,
                )
        return None
