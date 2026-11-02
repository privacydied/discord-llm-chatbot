"""Preview cards for Spotify links that FxSpotify cannot serve.

FxSpotify only proxies track pages (album, playlist, episode, show and artist
URLs 404). For those kinds the bot reads the public open.spotify.com page's
Open Graph tags and replies with its own embed that links to the Spotify page.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

import aiohttp
import discord

from bot.utils.logging import get_logger

from .instagram import parse_og, read_capped

logger = get_logger(__name__)

SPOTIFY_TIMEOUT_S = 6.0
SPOTIFY_COLOR = 0x1DB954
# Spotify only ships og: tags to crawler user agents; a browser UA gets a bare JS shell.
_USER_AGENT = "Mozilla/5.0 (compatible; Discordbot/2.0; +https://discordapp.com)"
_FIELD_MAX = 256
_DESCRIPTION_MAX = 4096
_MAX_PAGE_BYTES = 512_000


def build_embed(og: Mapping[str, str], url: str) -> discord.Embed | None:
    """Embed from a page's Open Graph tags, or None when there is no title."""
    title = (og.get("og:title") or "").strip()
    if not title:
        return None
    embed = discord.Embed(
        title=title[:_FIELD_MAX],
        description=(og.get("og:description") or "")[:_DESCRIPTION_MAX] or None,
        url=url,
        color=SPOTIFY_COLOR,
    )
    embed.set_footer(text="Spotify")
    image = og.get("og:image")
    if image and image.startswith("https://"):
        embed.set_image(url=image)
    return embed


async def _fetch_og(session: aiohttp.ClientSession, url: str) -> dict[str, str] | None:
    try:
        async with session.get(url) as resp:
            if resp.status != 200:
                logger.debug(f"embed_fix.spotify.page_status | url={url} status={resp.status}")
                return None
            raw = await read_capped(resp, _MAX_PAGE_BYTES)
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        logger.debug(f"embed_fix.spotify.page_failed | url={url} error={exc}")
        return None
    return parse_og(raw.decode("utf-8", errors="replace"))


async def build_cards(urls: list[str]) -> list[discord.Embed]:
    """Fetch and build embeds for Spotify page URLs; failures are dropped."""
    if not urls:
        return []
    timeout = aiohttp.ClientTimeout(total=SPOTIFY_TIMEOUT_S)
    async with aiohttp.ClientSession(timeout=timeout, headers={"User-Agent": _USER_AGENT}) as session:
        pages = await asyncio.gather(*(_fetch_og(session, url) for url in urls))
    cards = [card for og, url in zip(pages, urls, strict=True) if og and (card := build_embed(og, url))]
    if len(cards) < len(urls):
        logger.info(f"embed_fix.spotify.lookup_failed | requested={len(urls)} built={len(cards)}")
    return cards
