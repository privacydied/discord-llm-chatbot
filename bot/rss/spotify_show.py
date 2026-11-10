"""Spotify podcast shows as RSS-poller feeds.

Some shows (e.g. a patron feed linked into Spotify) have no public RSS. The show
page itself is JavaScript-only, but ``open.spotify.com/embed/show/<id>`` is
server-rendered and carries the newest episode (``__NEXT_DATA__``). The poller
runs every few minutes, so seeing only the latest episode is enough to post each
new one; an episode published between two polls alongside another is the only
one that can be missed.
"""

from __future__ import annotations

import json
import re

from .feeds import FeedItem, FeedParseError

SHOW_URL_RE = re.compile(r"^https?://open\.spotify\.com/(?:intl-[a-z]{2}/)?(?:embed/)?show/(?P<id>[A-Za-z0-9]{22})(?:[/?#]|$)")
EPISODE_URI_PREFIX = "spotify:episode:"
_NEXT_DATA_RE = re.compile(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.S)


def is_spotify_show(url: str) -> bool:
    return SHOW_URL_RE.match(url) is not None


def embed_url(url: str) -> str:
    """Server-rendered embed URL for a show URL."""
    match = SHOW_URL_RE.match(url)
    if match is None:
        raise ValueError(f"not a Spotify show URL: {url!r}")
    return f"https://open.spotify.com/embed/show/{match['id']}"


def parse_embed(data: bytes) -> tuple[str, list[FeedItem]]:
    """(show title, [latest episode]) from an embed page; FeedParseError when it has none."""
    found = _NEXT_DATA_RE.search(data.decode("utf-8", "replace"))
    if found is None:
        raise FeedParseError("Spotify embed has no __NEXT_DATA__")
    try:
        entity = json.loads(found.group(1))["props"]["pageProps"]["state"]["data"]["entity"]
    except (ValueError, KeyError, TypeError) as exc:
        raise FeedParseError(f"Spotify embed JSON unexpected: {exc!r}") from exc
    uri = str(entity.get("uri") or "")
    if entity.get("type") != "episode" or not uri.startswith(EPISODE_URI_PREFIX):
        raise FeedParseError("Spotify embed has no episode")
    release = entity.get("releaseDate")
    pub_date = str(release.get("isoString") or "") if isinstance(release, dict) else ""
    episode_id = uri[len(EPISODE_URI_PREFIX) :]
    item = FeedItem(
        title=str(entity.get("title") or entity.get("name") or "").strip(),
        pub_date=pub_date,
        description="",
        enclosure_url="",
        link=f"https://open.spotify.com/episode/{episode_id}",
        guid=uri,
    )
    return str(entity.get("subtitle") or ""), [item]
