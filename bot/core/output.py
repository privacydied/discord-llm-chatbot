"""Explicit safe outbound Discord helpers.

All bot-to-Discord text/embed output should flow through these helpers so that
sanitization *and* Discord's 2000-character content limit are enforced explicitly
rather than relying on monkey-patches or on each call site remembering to truncate.

Content over the limit is split with the shared splitter and sent as ordered
multi-part messages: each part is fully awaited before the next is issued, because
Discord assigns snowflakes in receipt order and concurrent sends can land a short
trailing part ahead of the long part it continues. Embeds, files and views ride on
the FIRST part only, so a caller's attachment is never duplicated per part.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from bot.public_output import (
    sanitize_embed_collection_for_public,
    sanitize_embed_for_public,
    sanitize_public_text,
)

from .text_chunking import DISCORD_MAX_CONTENT_LEN, render_chunks_for_discord, split_for_discord

if TYPE_CHECKING:
    from collections.abc import Sequence

    import discord

logger = logging.getLogger(__name__)

# Matches the dispatch path's inter-part gap: Discord allows 5 messages per 5s per
# channel, and continuation parts must not trip that mid-response. [CMV][REH]
_INTER_PART_DELAY_S = 0.3

# Discord's hard per-field embed limits. Enforced here -- the actual outbound
# boundary -- so a caller that forgets to truncate still can't trip HTTP 400 /
# error 50035 ("Invalid Form Body"). [CMV][REH]
_EMBED_TITLE_LIMIT = 256
_EMBED_DESCRIPTION_LIMIT = 4096
_EMBED_FIELD_NAME_LIMIT = 256
_EMBED_FIELD_VALUE_LIMIT = 1024
_EMBED_FOOTER_LIMIT = 2048
_EMBED_AUTHOR_NAME_LIMIT = 256
_EMBED_MAX_FIELDS = 25
# Discord also caps the SUM of every text field on the embed, independent of the
# per-field limits above -- a 20-field embed can be individually-legal per field
# and still trip 50035 on the aggregate. [CMV]
_EMBED_TOTAL_LIMIT = 6000


async def safe_send(
    destination: discord.abc.Messageable,
    content: str | None = None,
    *,
    embed: discord.Embed | None = None,
    embeds: Sequence[discord.Embed] | None = None,
    file: discord.File | None = None,
    files: Sequence[discord.File] | None = None,
    view: discord.ui.View | None = None,
    **kwargs,
) -> discord.Message:
    """Sanitize, split to Discord's limit, then send in order.

    Returns the LAST message sent, so callers that keep a handle to edit or react
    still get a usable reference for multi-part output.
    """
    if content is not None:
        content = _maybe_sanitize_text(content)

    embed, embeds = _sanitize_embeds(embed, embeds)

    async def _send(part: str | None, *, first: bool) -> discord.Message:
        # Files and views are forwarded without modification, first part only.
        if embeds is not None and first:
            return await destination.send(
                part,
                embeds=embeds,
                file=file,
                files=files,
                view=view,
                **kwargs,
            )
        if first:
            return await destination.send(
                part,
                embed=embed,
                file=file,
                files=files,
                view=view,
                **kwargs,
            )
        return await destination.send(part, **kwargs)

    return await _send_in_order(content, _send)


async def safe_reply(
    message: discord.Message,
    content: str | None = None,
    *,
    embed: discord.Embed | None = None,
    embeds: Sequence[discord.Embed] | None = None,
    file: discord.File | None = None,
    files: Sequence[discord.File] | None = None,
    view: discord.ui.View | None = None,
    mention_author: bool = False,
    **kwargs,
) -> discord.Message:
    """Sanitize, split to Discord's limit, then reply in order.

    Only the first part is an actual reply; continuations go to the channel so the
    user's message isn't pinged once per part.
    """
    if content is not None:
        content = _maybe_sanitize_text(content)

    embed, embeds = _sanitize_embeds(embed, embeds)

    async def _send(part: str | None, *, first: bool) -> discord.Message:
        if first:
            return await message.reply(
                part,
                embed=embed,
                embeds=embeds,
                file=file,
                files=files,
                view=view,
                mention_author=mention_author,
                **kwargs,
            )
        return await message.channel.send(part, **kwargs)

    return await _send_in_order(content, _send)


async def safe_edit(
    message: discord.Message,
    content: str | None = None,
    *,
    embed: discord.Embed | None = None,
    embeds: Sequence[discord.Embed] | None = None,
    view: discord.ui.View | None = None,
    **kwargs,
) -> discord.Message:
    """Sanitize, split to Discord's limit, then edit in place.

    An oversize payload edits the target with the first part and appends the rest as
    ordered follow-up messages in the same channel. Returns the EDITED message (not
    the last follow-up), since callers hold that handle to edit again later.
    """
    if content is not None:
        content = _maybe_sanitize_text(content)

    embed, embeds = _sanitize_embeds(embed, embeds)

    parts = render_chunks_for_discord(_split_content(content))
    edited = await message.edit(
        content=parts[0] if parts else content,
        embed=embed,
        embeds=embeds,  # type: ignore[arg-type]
        view=view,
        **kwargs,
    )

    for part in parts[1:]:
        await asyncio.sleep(_INTER_PART_DELAY_S)
        await message.channel.send(part)

    return edited


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _split_content(content: str | None) -> list[str]:
    """Parts to send for this content; empty when there is no text to send."""
    if not content or len(content) <= DISCORD_MAX_CONTENT_LEN:
        return [content] if content else []
    return split_for_discord(content)


async def _send_in_order(content: str | None, send) -> discord.Message:
    """Send each part strictly sequentially, returning the last message.

    ``send(part, first=...)`` performs one send. Each await completes before the
    next begins: Discord orders messages by receipt, so overlapping sends can put a
    short trailing part ahead of the long part it continues. [REH]

    Parts are fence-wrapped for rendering (``render_chunks_for_discord``) right
    before this, after ``content`` was already sanitized as a whole -- so a code
    block split across parts still renders as code in every part, without
    changing what ``_split_content`` itself guarantees byte-exact. [REH]
    """
    parts = render_chunks_for_discord(_split_content(content))
    if not parts:
        # No text at all -- still one send, so embeds/files/views go out.
        return await send(content, first=True)

    last = None
    for index, part in enumerate(parts):
        if index:
            await asyncio.sleep(_INTER_PART_DELAY_S)
        last = await send(part, first=index == 0)
    return last


def _maybe_sanitize_text(text: str) -> str:
    """Apply sanitization only when text is non-empty and non-whitespace."""
    if not text or not text.strip():
        return text
    return sanitize_public_text(text)


def _truncate(text: str | None, limit: int) -> str | None:
    """Clamp text to `limit` chars, appending an ellipsis marker when cut."""
    if text is None or len(text) <= limit:
        return text
    if limit <= 1:
        return text[:limit]
    return text[: limit - 1] + "…"


def _embed_total_len(embed: discord.Embed) -> int:
    total = len(embed.title or "") + len(embed.description or "")
    if embed.footer and embed.footer.text:
        total += len(embed.footer.text)
    if embed.author and embed.author.name:
        total += len(embed.author.name)
    for f in embed.fields or []:
        total += len(f.name or "") + len(f.value or "")
    return total


def _enforce_embed_total_limit(embed: discord.Embed) -> None:
    """Bring the embed's aggregate text length under Discord's 6000-char cap.

    Per-field truncation alone doesn't prevent a many-field embed from still
    tripping 50035 on the SUM. Drop trailing fields first (least information
    loss for the parts a reader sees first), then shave the description as a
    last resort. [REH]
    """
    overflow = _embed_total_len(embed) - _EMBED_TOTAL_LIMIT
    if overflow <= 0:
        return

    fields = list(embed.fields or [])
    kept = list(fields)
    while kept and overflow > 0:
        dropped = kept.pop()
        overflow -= len(dropped.name or "") + len(dropped.value or "")

    if len(kept) != len(fields):
        embed._fields = []  # noqa: SLF001 - same pattern sanitize_embed_for_public uses to rebuild fields
        for f in kept:
            embed.add_field(name=f.name, value=f.value, inline=f.inline)

    if overflow > 0 and embed.description:
        keep_len = max(0, len(embed.description) - overflow)
        embed.description = _truncate(embed.description, keep_len) if keep_len else ""


def _enforce_embed_limits(embed: discord.Embed | None) -> discord.Embed | None:
    """Clamp an embed to Discord's outbound size limits. Runs AFTER sanitization,
    at the actual send boundary, so a caller that forgets to truncate still can't
    produce an HTTP 400 / error 50035 payload. [REH][CMV]
    """
    if embed is None:
        return None

    before = _embed_total_len(embed)

    if embed.title:
        embed.title = _truncate(embed.title, _EMBED_TITLE_LIMIT)
    if embed.description:
        embed.description = _truncate(embed.description, _EMBED_DESCRIPTION_LIMIT)
    if embed.footer and embed.footer.text:
        embed.set_footer(
            text=_truncate(embed.footer.text, _EMBED_FOOTER_LIMIT),
            icon_url=embed.footer.icon_url or None,
        )
    if embed.author and embed.author.name:
        embed.set_author(
            name=_truncate(embed.author.name, _EMBED_AUTHOR_NAME_LIMIT),
            url=embed.author.url or None,
            icon_url=embed.author.icon_url or None,
        )

    if embed.fields:
        fields = list(embed.fields)[:_EMBED_MAX_FIELDS]
        embed._fields = []  # noqa: SLF001 - same pattern sanitize_embed_for_public uses to rebuild fields
        for f in fields:
            embed.add_field(
                name=_truncate(f.name, _EMBED_FIELD_NAME_LIMIT) or "\u200b",
                value=_truncate(f.value, _EMBED_FIELD_VALUE_LIMIT) or "\u200b",
                inline=f.inline,
            )

    _enforce_embed_total_limit(embed)

    if _embed_total_len(embed) != before:
        logger.debug(f"embed truncated to fit Discord limits: {before} -> {_embed_total_len(embed)} chars")

    return embed


def _sanitize_embeds(
    embed: discord.Embed | None,
    embeds: Sequence[discord.Embed] | None,
) -> tuple[discord.Embed | None, list[discord.Embed] | None]:
    """Sanitize embed(s), enforce Discord's size limits, and normalize to
    exactly one representation.
    """
    sanitized_embed = _enforce_embed_limits(sanitize_embed_for_public(embed)) if embed is not None else None
    sanitized_embeds = [_enforce_embed_limits(e) for e in sanitize_embed_collection_for_public(list(embeds or []))]

    # Discord rejects payloads with both embed and embeds set.
    if sanitized_embeds and not sanitized_embed:
        return None, sanitized_embeds
    if sanitized_embed and not sanitized_embeds:
        return sanitized_embed, None
    if sanitized_embed and sanitized_embeds:
        # Prefer plural representation when both are present.
        return None, [sanitized_embed, *sanitized_embeds]
    return None, None
