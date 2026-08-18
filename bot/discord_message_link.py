"""Resolve Discord message links (jump URLs) through the Discord API. [CA][SFT]

``https://discord.com/channels/<guild>/<channel>/<message>`` is not a scrapeable
web page: discord.com serves a JS-only shell to anonymous clients, so the generic
web extractor burns ~30s on Tier A/B/C and finally reports ``empty_or_js_only``.

The bot is already a Discord client, so the correct move is to fetch the message
over the gateway/REST API instead of scraping it.

Safety: the requesting user must themselves be able to read the target channel,
otherwise a link is an oracle for private channels. Content is wrapped as
untrusted (it is arbitrary user text and may contain prompt injection). [SFT]
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any

import discord

from .url_safety import wrap_untrusted_content
from .utils.logging import get_logger


logger = get_logger(__name__)

# --- Constants [CMV] ---------------------------------------------------------
MESSAGE_LINK_RE = re.compile(
    r"https?://(?:(?:canary|ptb)\.)?discord(?:app)?\.com/channels/"
    r"(?P<guild>@me|\d{15,25})/(?P<channel>\d{15,25})/(?P<message>\d{15,25})",
    re.IGNORECASE,
)
DEFAULT_FETCH_TIMEOUT_S = 10.0
DEFAULT_MAX_CHARS = 4000
MAX_ATTACHMENTS_RENDERED = 10
MAX_EMBEDS_RENDERED = 5
MAX_EMBED_DESC_CHARS = 500
TRUNCATION_SUFFIX = "\n…[truncated]"

REASON_NOT_A_LINK = "not_a_message_link"
REASON_DM_LINK = "dm_link"
REASON_CHANNEL_UNAVAILABLE = "channel_unavailable"
REASON_FORBIDDEN_REQUESTER = "requester_no_access"
REASON_FORBIDDEN_BOT = "bot_no_access"
REASON_NOT_FOUND = "message_not_found"
REASON_TIMEOUT = "timeout"

_USER_MSG = {
    REASON_DM_LINK: "that link points at a DM conversation, which I can't read",
    REASON_CHANNEL_UNAVAILABLE: "I can't see that channel (not a server I'm in, or I lack access)",
    REASON_FORBIDDEN_REQUESTER: "you don't have access to that channel, so I won't read it out",
    REASON_FORBIDDEN_BOT: "I don't have permission to read that channel's history",
    REASON_NOT_FOUND: "that message no longer exists (deleted, or the link is wrong)",
    REASON_TIMEOUT: "Discord took too long to return that message",
}


@dataclass(frozen=True)
class MessageLinkRef:
    """Parsed components of a Discord message jump URL."""

    guild_id: int | None  # None when the link is a DM ("@me")
    channel_id: int
    message_id: int
    url: str

    @property
    def is_dm(self) -> bool:
        return self.guild_id is None


@dataclass(frozen=True)
class LinkResolution:
    """Outcome of resolving a message link.

    ``text`` is always prompt-safe: rendered message content on success, a short
    human explanation on failure.
    """

    ok: bool
    text: str
    reason: str = ""


def parse_message_link(url: str) -> MessageLinkRef | None:
    """Parse a Discord message jump URL; return None when it isn't one. [IV]"""
    if not url or "discord" not in url.lower():
        return None
    match = MESSAGE_LINK_RE.search(url)
    if match is None:
        return None
    raw_guild = match.group("guild")
    return MessageLinkRef(
        guild_id=None if raw_guild == "@me" else int(raw_guild),
        channel_id=int(match.group("channel")),
        message_id=int(match.group("message")),
        url=match.group(0),
    )


async def _resolve_channel(bot: Any, channel_id: int, timeout_s: float) -> Any | None:
    """Return the channel object for ``channel_id`` from cache, else fetch it. [REH]"""
    cached = None
    with_cache = getattr(bot, "get_channel", None)
    if callable(with_cache):
        cached = with_cache(channel_id)
    if cached is not None:
        return cached
    fetch = getattr(bot, "fetch_channel", None)
    if not callable(fetch):
        return None
    try:
        return await asyncio.wait_for(fetch(channel_id), timeout=timeout_s)
    except (TimeoutError, asyncio.TimeoutError, discord.NotFound, discord.Forbidden, discord.HTTPException):
        return None
    except Exception as exc:  # defensive: never let link resolution crash the router
        logger.debug(f"discord_link.channel_fetch_failed id={channel_id} error={exc}")
        return None


async def _as_member(channel: Any, requester: Any, timeout_s: float = DEFAULT_FETCH_TIMEOUT_S) -> Any | None:
    """Upgrade a User to the guild Member for permission checks, when possible.

    Falls back to a REST member fetch: without the members intent the cache is
    often empty, and denying on a cache miss would reject legitimate askers.
    """
    if requester is None:
        return None
    if isinstance(requester, discord.Member):
        return requester
    guild = getattr(channel, "guild", None)
    user_id = getattr(requester, "id", 0)
    get_member = getattr(guild, "get_member", None)
    cached = get_member(user_id) if callable(get_member) else None
    if cached is not None:
        return cached
    fetch_member = getattr(guild, "fetch_member", None)
    if not callable(fetch_member):
        return None
    try:
        return await asyncio.wait_for(fetch_member(user_id), timeout=timeout_s)
    except (TimeoutError, asyncio.TimeoutError, discord.NotFound, discord.Forbidden, discord.HTTPException):
        return None
    except Exception as exc:  # defensive: permission checks must never crash routing
        logger.debug(f"discord_link.member_fetch_failed id={user_id} error={exc}")
        return None


def _member_can_read(channel: Any, member: Any) -> bool:
    """True only if ``member`` may read the target channel. [SFT]"""
    if member is None:
        return False
    perms_for = getattr(channel, "permissions_for", None)
    if not callable(perms_for):
        return False
    try:
        perms = perms_for(member)
    except Exception as exc:
        logger.debug(f"discord_link.perms_failed error={exc}")
        return False
    return bool(getattr(perms, "view_channel", False) and getattr(perms, "read_message_history", False))


async def requester_can_read(channel: Any, requester: Any, timeout_s: float = DEFAULT_FETCH_TIMEOUT_S) -> bool:
    """True only if the *requesting user* may read the target channel. [SFT]

    Without this gate a jump URL is an oracle: anyone could make the bot read
    out a private channel they have no access to.
    """
    member = await _as_member(channel, requester, timeout_s)
    return _member_can_read(channel, member)


def _render_attachments(msg: Any) -> list[str]:
    lines: list[str] = []
    attachments = list(getattr(msg, "attachments", None) or [])[:MAX_ATTACHMENTS_RENDERED]
    for att in attachments:
        name = getattr(att, "filename", "file")
        url = getattr(att, "url", "")
        lines.append(f"- attachment: {name} ({url})")
    stickers = list(getattr(msg, "stickers", None) or [])
    lines.extend(f"- sticker: {getattr(s, 'name', 'sticker')}" for s in stickers)
    return lines


def _render_embeds(msg: Any) -> list[str]:
    lines: list[str] = []
    for emb in list(getattr(msg, "embeds", None) or [])[:MAX_EMBEDS_RENDERED]:
        title = (getattr(emb, "title", None) or "").strip()
        desc = (getattr(emb, "description", None) or "").strip()[:MAX_EMBED_DESC_CHARS]
        url = (getattr(emb, "url", None) or "").strip()
        parts = [p for p in (title, url, desc) if p]
        if parts:
            lines.append("- embed: " + " | ".join(parts))
    return lines


def render_message(msg: Any, *, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """Render a discord.Message into compact plain text for the prompt."""
    author = getattr(msg, "author", None)
    author_name = getattr(author, "display_name", None) or getattr(author, "name", None) or "unknown"
    channel = getattr(msg, "channel", None)
    channel_name = getattr(channel, "name", None) or "unknown-channel"
    guild_name = getattr(getattr(msg, "guild", None), "name", None) or "DM"
    created = getattr(msg, "created_at", None)
    when = created.isoformat() if created is not None else "unknown time"

    body = (getattr(msg, "clean_content", None) or getattr(msg, "content", "") or "").strip()
    lines = [f"Author: {author_name} · #{channel_name} in {guild_name} · {when}", ""]
    lines.append(body if body else "(no text content)")
    extras = _render_attachments(msg) + _render_embeds(msg)
    if extras:
        lines.extend(["", *extras])

    rendered = "\n".join(lines)
    if len(rendered) > max_chars:
        rendered = rendered[: max(0, max_chars - len(TRUNCATION_SUFFIX))] + TRUNCATION_SUFFIX
    return rendered


def _failure(reason: str) -> LinkResolution:
    return LinkResolution(ok=False, text=f"[Linked Discord message could not be read: {_USER_MSG[reason]}.]", reason=reason)


async def _fetch_linked_message(channel: Any, message_id: int, timeout_s: float) -> tuple[Any | None, str]:
    """Fetch the linked message; return (message, failure_reason). [REH]"""
    fetch = getattr(channel, "fetch_message", None)
    if not callable(fetch):
        return None, REASON_CHANNEL_UNAVAILABLE
    try:
        return await asyncio.wait_for(fetch(message_id), timeout=timeout_s), ""
    except (TimeoutError, asyncio.TimeoutError):
        return None, REASON_TIMEOUT
    except discord.NotFound:
        return None, REASON_NOT_FOUND
    except discord.Forbidden:
        return None, REASON_FORBIDDEN_BOT
    except discord.HTTPException as exc:
        logger.debug(f"discord_link.fetch_http_error id={message_id} error={exc}")
        return None, REASON_NOT_FOUND


async def resolve_message_link(
    bot: Any,
    url: str,
    *,
    requester: Any = None,
    timeout_s: float = DEFAULT_FETCH_TIMEOUT_S,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> LinkResolution | None:
    """Resolve a Discord jump URL to its message text.

    Returns None when ``url`` is not a Discord message link (caller should carry
    on with normal URL handling).
    """
    ref = parse_message_link(url)
    if ref is None:
        return None
    if ref.is_dm:
        return _failure(REASON_DM_LINK)

    channel = await _resolve_channel(bot, ref.channel_id, timeout_s)
    if channel is None:
        return _failure(REASON_CHANNEL_UNAVAILABLE)
    if not await requester_can_read(channel, requester, timeout_s):
        return _failure(REASON_FORBIDDEN_REQUESTER)

    msg, reason = await _fetch_linked_message(channel, ref.message_id, timeout_s)
    if msg is None:
        return _failure(reason or REASON_NOT_FOUND)

    rendered = render_message(msg, max_chars=max_chars)
    wrapped = wrap_untrusted_content(rendered, source=ref.url)
    return LinkResolution(ok=True, text=f"Linked Discord message ({ref.url}):\n{wrapped}", reason="ok")


def link_budget(config: Any) -> tuple[float, int]:
    """Read (timeout_s, max_chars) for link resolution from config. [CMV][REH]"""
    getter = getattr(config, "get", None)
    if not callable(getter):
        return DEFAULT_FETCH_TIMEOUT_S, DEFAULT_MAX_CHARS
    try:
        return (
            float(getter("DISCORD_LINK_TIMEOUT_S", DEFAULT_FETCH_TIMEOUT_S)),
            int(getter("DISCORD_LINK_MAX_CHARS", DEFAULT_MAX_CHARS)),
        )
    except (TypeError, ValueError):
        return DEFAULT_FETCH_TIMEOUT_S, DEFAULT_MAX_CHARS
