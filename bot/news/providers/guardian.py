"""Guardian Open Platform content provider.
[CA][REH][CMV][SFT].

The Guardian licenses full article text through its public content API. An
article's API id is exactly its web URL path, so a link resolves to an exact
item lookup -- no search heuristics needed.

Requires a free developer key in ``GUARDIAN_API_KEY``; the provider disables
itself cleanly when the key is absent.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

from bot.http_client import RequestConfig, get_http_client
from bot.utils.logging import get_logger

from ..base import AbstractNewsProvider
from ..types import NewsArticle

logger = get_logger(__name__)

# Fixed, trusted endpoint -- not user-controlled, so no SSRF surface. [SFT][CMV]
API_ROOT = "https://content.guardianapis.com"

# Hosts whose URL path maps directly onto a Guardian content id. [CMV]
SUPPORTED_HOSTS: frozenset[str] = frozenset(
    {
        "theguardian.com",
        "www.theguardian.com",
        "amp.theguardian.com",
        "guardian.co.uk",
        "www.guardian.co.uk",
    }
)

# Fields requested from the API. [CMV]
SHOW_FIELDS = "bodyText,headline,byline,firstPublicationDate"

# Trailing path segments that are presentation variants, not part of the id. [CMV]
_STRIP_SUFFIXES = ("/amp", "/index.html")

_TIMEOUT = RequestConfig(connect_timeout=2.0, read_timeout=6.0, total_timeout=8.0, max_retries=2)

_SLASH_RE = re.compile(r"/{2,}")


def content_id_from_url(url: str) -> str | None:
    """Map a Guardian article URL onto its content API id.

    Returns None when the URL is not a Guardian article path (e.g. a section
    front, which has no body to retrieve).
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return None
    if (parsed.hostname or "").lower() not in SUPPORTED_HOSTS:
        return None

    path = _SLASH_RE.sub("/", parsed.path or "")
    for suffix in _STRIP_SUFFIXES:
        if path.endswith(suffix):
            path = path[: -len(suffix)]
    path = path.strip("/")

    # Article ids look like <section>/<yyyy>/<mon>/<dd>/<slug>; anything with
    # fewer segments is a listing page rather than an article.
    if path.count("/") < 3:
        return None
    return path


class GuardianNewsProvider(AbstractNewsProvider):
    """Retrieves full article text via the Guardian Open Platform."""

    name = "guardian"

    def __init__(self, api_key: str | None) -> None:
        self._api_key = (api_key or "").strip() or None

    @property
    def enabled(self) -> bool:
        return self._api_key is not None

    def supports(self, url: str) -> bool:
        return self.enabled and content_id_from_url(url) is not None

    async def fetch(self, url: str) -> NewsArticle | None:
        """Return the article body, or None on any failure. Never raises. [REH]"""
        content_id = content_id_from_url(url)
        if not content_id or not self._api_key:
            return None
        try:
            payload = await self._request(content_id)
        except Exception as exc:  # [REH] provider failure must not break the chain
            logger.debug("guardian.fetch failed for %s: %s", content_id, exc)
            return None
        return self._to_article(payload, url) if payload else None

    async def _request(self, content_id: str) -> dict[str, Any] | None:
        client = await get_http_client()
        response = await client.get(
            f"{API_ROOT}/{content_id}",
            config=_TIMEOUT,
            params={
                "api-key": self._api_key,
                "show-fields": SHOW_FIELDS,
                "format": "json",
            },
        )
        if response.status_code != 200:
            logger.debug("guardian.http_status status=%s id=%s", response.status_code, content_id)
            return None
        return response.json()

    def _to_article(self, payload: dict[str, Any], url: str) -> NewsArticle | None:
        response = (payload or {}).get("response") or {}
        if response.get("status") != "ok":
            return None
        content = response.get("content") or {}
        fields = content.get("fields") or {}
        body = (fields.get("bodyText") or "").strip()
        if not body:
            return None
        return NewsArticle(
            url=content.get("webUrl") or url,
            body=body,
            title=fields.get("headline") or content.get("webTitle"),
            author=fields.get("byline"),
            published=fields.get("firstPublicationDate") or content.get("webPublicationDate"),
            provider=self.name,
        )
