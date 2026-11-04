"""Preview cards for rumble.com video links.

Rumble serves Discord's crawler a 307 redirect loop (Cloudflare) and no
rewrite proxy exists (fxrumble.com is "coming soon"), so the page has no
usable Open Graph tags. Rumble's public oEmbed endpoint still answers, so the
bot builds its own embed (title, channel, thumbnail) linking to the video.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

import aiohttp
import discord

from bot.utils.logging import get_logger

logger = get_logger(__name__)

RUMBLE_OEMBED = "https://rumble.com/api/Media/oembed.json?url={url}"
RUMBLE_TIMEOUT_S = 6.0
RUMBLE_COLOR = 0x85C742
_USER_AGENT = "Mozilla/5.0 (compatible; discord-llm-chatbot embed-fix)"
_FIELD_MAX = 256


def build_embed(data: Mapping[str, Any], page_url: str) -> discord.Embed | None:
    """Embed from an oEmbed record, or None when it has no title."""
    title = str(data.get("title") or "").strip()
    if not title:
        return None
    embed = discord.Embed(title=title[:_FIELD_MAX], url=page_url, color=RUMBLE_COLOR)
    author = str(data.get("author_name") or "").strip()
    if author:
        author_url = str(data.get("author_url") or "")
        embed.set_author(name=author[:_FIELD_MAX], url=author_url if author_url.startswith("https://") else None)
    embed.set_footer(text="Rumble")
    thumb = str(data.get("thumbnail_url") or "")
    if thumb.startswith("https://"):
        embed.set_image(url=thumb)
    return embed


async def _fetch(session: aiohttp.ClientSession, page_url: str) -> Mapping[str, Any] | None:
    try:
        async with session.get(RUMBLE_OEMBED.format(url=quote(page_url, safe=""))) as resp:
            if resp.status != 200:
                logger.debug(f"embed_fix.rumble.api_status | url={page_url} status={resp.status}")
                return None
            payload = await resp.json(content_type=None)
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
        logger.debug(f"embed_fix.rumble.api_error | url={page_url} error={exc}")
        return None
    return payload if isinstance(payload, dict) else None


async def build_cards(page_urls: list[str]) -> list[discord.Embed]:
    """Fetch and build embeds for rumble video URLs; failures are dropped."""
    if not page_urls:
        return []
    timeout = aiohttp.ClientTimeout(total=RUMBLE_TIMEOUT_S)
    async with aiohttp.ClientSession(timeout=timeout, headers={"User-Agent": _USER_AGENT}) as session:
        records = await asyncio.gather(*(_fetch(session, url) for url in page_urls))
    cards = [card for rec, url in zip(records, page_urls, strict=True) if rec and (card := build_embed(rec, url))]
    if len(cards) < len(page_urls):
        logger.info(f"embed_fix.rumble.lookup_failed | requested={len(page_urls)} built={len(cards)}")
    return cards
