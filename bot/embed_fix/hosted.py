"""Publish fixable posts on the bot's own embed host instead of a proxy link.

Discord plays video inside a card only for an unfurled page with og:video,
so the bot downloads the post, stores it on ``bot.embed_host`` and replies
with that page's URL. Used only while the host has proven its public URL
reaches it (``EmbedHost.ready``); otherwise callers keep the proxy path.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from bot.utils.logging import get_logger

from . import instagram

if TYPE_CHECKING:
    from bot.embed_host import EmbedHost

logger = get_logger(__name__)

HOSTED_SITES = frozenset({"instagram"})
INSTAGRAM_COLOUR = "#E1306C"


async def _none() -> None:
    return None


async def publish_instagram(host: EmbedHost, post_url: str) -> str | None:
    """Download *post_url*, store it on *host*; the page URL, or None on any failure."""
    from bot.embed_host.store import NewMedia, StoredPost

    post = await instagram.fetch_post(post_url, max_bytes=host.settings.max_file_bytes)
    if post is None or not post.files:
        return None
    meta = post.meta
    has_video = any(not f.is_image for f in post.files)
    thumb, avatar = await asyncio.gather(
        instagram.fetch_thumbnail(meta.thumbnail_url) if has_video else _none(),
        instagram.fetch_thumbnail(meta.avatar_url, max_bytes=1024 * 1024),
    )
    record = StoredPost(
        token="",
        source_url=post.url,
        site_name="Instagram",
        colour=INSTAGRAM_COLOUR,
        author_name=meta.display_name,
        author_handle=meta.username,
        author_url=f"https://www.instagram.com/{meta.username}/" if meta.username else None,
        avatar_url=meta.avatar_url,
        text=meta.caption,
        stats=instagram.stats_line(meta),
        published=meta.published.isoformat().replace("+00:00", "Z") if meta.published else None,
    )
    media = [NewMedia(f.filename, f.data) for f in post.files]
    thumbnail = NewMedia(thumb.filename, thumb.data) if thumb else None
    avatar_file = NewMedia(avatar.filename, avatar.data) if avatar else None
    try:
        stored = await asyncio.to_thread(host.store.put, record, media, thumbnail, avatar_file)
    except OSError as exc:
        logger.warning(f"embed_fix.hosted.store_failed | url={post_url} error={exc}")
        return None
    logger.info(f"embed_fix.hosted.published | url={post_url} token={stored.token} files={len(media)}")
    return host.page_url(stored.token)
