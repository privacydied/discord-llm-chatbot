"""Preview cards for ppv.* stream links.

The ppv sites (ppv.st and its mirrors) are a client-side Nuxt app: the HTML
has no Open Graph tags, so Discord shows no preview at all. Instead of
rewriting the URL, the bot looks the event up on the site's public API and
replies with its own embed (title, poster, category, start time) that links
back to the page the user posted. The stream itself is not extracted; Discord
can't play these streams inline anyway.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from typing import Any

import aiohttp
import discord

from bot.utils.logging import get_logger

from .sites import page_slug

logger = get_logger(__name__)

PPV_API_BASES: tuple[str, ...] = ("https://api.ppv.st/api", "https://api.ppv.cx/api")
PPV_TIMEOUT_S = 6.0
PPV_COLOR = 0xE11D48
_USER_AGENT = "Mozilla/5.0 (compatible; discord-llm-chatbot embed-fix)"
_FIELD_MAX = 256


async def fetch_stream(session: aiohttp.ClientSession, slug: str) -> Mapping[str, Any] | None:
    """Return the stream record for *slug*, trying each API base in turn."""
    for base in PPV_API_BASES:
        try:
            async with session.get(f"{base}/streams/{slug}") as resp:
                if resp.status != 200:
                    logger.debug(f"embed_fix.ppv.api_status | base={base} slug={slug} status={resp.status}")
                    continue
                payload = await resp.json(content_type=None)
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
            logger.debug(f"embed_fix.ppv.api_error | base={base} slug={slug} error={exc}")
            continue
        data = payload.get("data") if isinstance(payload, dict) and payload.get("success") else None
        if isinstance(data, dict) and data.get("name"):
            return data
    return None


def _status(data: Mapping[str, Any], now: float) -> str:
    start, end = int(data.get("start_timestamp") or 0), int(data.get("end_timestamp") or 0)
    if data.get("always_live") or not start:
        return "🔴 24/7"
    if now < start:
        return f"⏳ Starts <t:{start}:R>"
    if not end or now < end:
        return "🔴 LIVE"
    return "⏹️ Ended"


def _clip(value: Any) -> str:
    return str(value)[:_FIELD_MAX]


def build_embed(data: Mapping[str, Any], page_url: str, *, now: float | None = None) -> discord.Embed:
    """Turn an API stream record into a Discord embed linking to *page_url*."""
    now = time.time() if now is None else now
    embed = discord.Embed(title=_clip(data["name"]), url=page_url, color=PPV_COLOR, description=_status(data, now))
    if data.get("tag"):
        embed.add_field(name="Category", value=_clip(data["tag"]), inline=True)
    if data.get("source_tag"):
        embed.add_field(name="Source", value=_clip(data["source_tag"]), inline=True)
    start = int(data.get("start_timestamp") or 0)
    if start and not data.get("always_live"):
        embed.add_field(name="Start", value=f"<t:{start}:F>", inline=False)
    if isinstance(data.get("viewers"), int) and data["viewers"] > 0:
        embed.add_field(name="Viewers", value=f"{data['viewers']:,}", inline=True)
    poster = data.get("poster")
    if isinstance(poster, str) and poster.startswith("https://"):
        embed.set_image(url=poster)
    return embed


async def build_cards(page_urls: list[str]) -> list[discord.Embed]:
    """Fetch and build embeds for ppv page URLs (see sites.py); failures are dropped."""
    items = [(slug, url) for url in page_urls if (slug := page_slug(url))]
    if not items:
        return []
    timeout = aiohttp.ClientTimeout(total=PPV_TIMEOUT_S)
    async with aiohttp.ClientSession(timeout=timeout, headers={"User-Agent": _USER_AGENT}) as session:
        records = await asyncio.gather(*(fetch_stream(session, slug) for slug, _ in items))
    cards = [build_embed(rec, url) for rec, (_, url) in zip(records, items, strict=True) if rec]
    if len(cards) < len(items):
        logger.info(f"embed_fix.ppv.lookup_failed | requested={len(items)} built={len(cards)}")
    return cards
