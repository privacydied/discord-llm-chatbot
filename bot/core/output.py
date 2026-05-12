"""Explicit Discord output helper with built-in sanitization.

Wraps all public sends/replies/edits through sanitize_public_text and
sanitize_embed_for_public so that no unsanitized content can reach Discord.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import discord

from bot.public_output import (
    sanitize_embed_collection_for_public,
    sanitize_public_text,
)

from .logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


async def safe_send(
    channel: discord.abc.Messageable,
    content: Optional[str] = None,
    *,
    embed: Optional[discord.Embed] = None,
    embeds: Optional[Sequence[discord.Embed]] = None,
    file: Optional[discord.File] = None,
    files: Optional[Sequence[discord.File]] = None,
    **send_kwargs: Any,
) -> discord.Message:
    """Send a message with automatic content sanitization."""
    content = sanitize_public_text(content) if content else None
    embeds = _collect_embeds(embed, embeds)
    embeds = sanitize_embed_collection_for_public(embeds)

    # Only pass non-None embeds to discord; empty list is fine
    kw: Dict[str, Any] = dict(send_kwargs)
    if content:
        kw["content"] = content
    if embeds:
        kw["embeds"] = embeds
    if file:
        kw["file"] = file
    if files:
        kw["files"] = files

    return await channel.send(**kw)


async def safe_reply(
    message: discord.Message,
    content: Optional[str] = None,
    *,
    embed: Optional[discord.Embed] = None,
    embeds: Optional[Sequence[discord.Embed]] = None,
    file: Optional[discord.File] = None,
    files: Optional[Sequence[discord.File]] = None,
    mention: bool = True,
    **reply_kwargs: Any,
) -> discord.Message:
    """Reply to a message with automatic content sanitization."""
    content = sanitize_public_text(content) if content else None
    embeds = _collect_embeds(embed, embeds)
    embeds = sanitize_embed_collection_for_public(embeds)

    kw: Dict[str, Any] = dict(reply_kwargs)
    kw["mention"] = mention
    if content:
        kw["content"] = content
    if embeds:
        kw["embeds"] = embeds
    if file:
        kw["file"] = file
    if files:
        kw["files"] = files

    return await message.reply(**kw)


async def safe_edit(
    message: discord.Message,
    content: Optional[str] = None,
    *,
    embed: Optional[discord.Embed] = None,
    embeds: Optional[Sequence[discord.Embed]] = None,
    **edit_kwargs: Any,
) -> discord.Message:
    """Edit a message with automatic content sanitization."""
    content = sanitize_public_text(content) if content else content
    embeds = _collect_embeds(embed, embeds)
    embeds = sanitize_embed_collection_for_public(embeds)

    kw: Dict[str, Any] = dict(edit_kwargs)
    if content is not None:  # allow empty string edit
        kw["content"] = content
    if embeds is not None:
        kw["embeds"] = embeds

    return await message.edit(**kw)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _collect_embeds(
    embed: Optional[discord.Embed],
    embeds: Optional[Sequence[discord.Embed]],
) -> Optional[List[discord.Embed]]:
    """Merge single embed + embed list into one list."""
    out: List[discord.Embed] = []
    if embed is not None:
        out.append(embed)
    if embeds:
        out.extend(embeds)
    return out if out else None
