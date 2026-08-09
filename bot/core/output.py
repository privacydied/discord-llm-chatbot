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
from typing import TYPE_CHECKING

from bot.public_output import (
    sanitize_embed_collection_for_public,
    sanitize_embed_for_public,
    sanitize_public_text,
)

from .text_chunking import DISCORD_MAX_CONTENT_LEN, split_for_discord

if TYPE_CHECKING:
    from collections.abc import Sequence

    import discord

# Matches the dispatch path's inter-part gap: Discord allows 5 messages per 5s per
# channel, and continuation parts must not trip that mid-response. [CMV][REH]
_INTER_PART_DELAY_S = 0.3


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

    parts = _split_content(content)
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
    """
    parts = _split_content(content)
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


def _sanitize_embeds(
    embed: discord.Embed | None,
    embeds: Sequence[discord.Embed] | None,
) -> tuple[discord.Embed | None, list[discord.Embed] | None]:
    """Sanitize embed(s) and normalize to exactly one representation."""
    sanitized_embed = sanitize_embed_for_public(embed) if embed is not None else None
    sanitized_embeds = sanitize_embed_collection_for_public(list(embeds or []))

    # Discord rejects payloads with both embed and embeds set.
    if sanitized_embeds and not sanitized_embed:
        return None, sanitized_embeds
    if sanitized_embed and not sanitized_embeds:
        return sanitized_embed, None
    if sanitized_embed and sanitized_embeds:
        # Prefer plural representation when both are present.
        return None, [sanitized_embed, *sanitized_embeds]
    return None, None
