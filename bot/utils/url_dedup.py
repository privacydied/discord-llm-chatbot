"""URL deduplication helper for message processing.

Prevents the same logical URL from being processed multiple times when
it appears in different message surfaces (content text, embeds, attachments).
Phase 17.
"""

from __future__ import annotations

from urllib.parse import urlparse


def _normalize_url(url: str) -> str:
    """Strip query/fragment, lowercase host, dedupe trailing slash."""
    try:
        parsed = urlparse(url)
        path = parsed.path.rstrip("/") or "/"
        # Collapse tracking params (t, utm_*, ref, etc.)
        return f"{parsed.scheme.lower()}://{parsed.hostname.lower()}{path}"
    except Exception:
        return url.strip().lower()


def deduplicate_urls(urls: list[str]) -> list[str]:
    """Return URLs in original order, keeping only the first occurrence of
    each normalized URL.

    E.g. if a user sends "https://example.com/page?t=1" + Discord creates
    an embed for the same URL, only the original will be processed.
    """
    seen: set[str] = set()
    result: list[str] = []
    for url in urls:
        key = _normalize_url(url)
        if key and key not in seen:
            seen.add(key)
            result.append(url)
    return result
