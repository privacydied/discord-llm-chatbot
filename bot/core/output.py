"""Explicit safe outbound Discord helpers.

All bot-to-Discord text/embed output should flow through these helpers
so that sanitization is explicit rather than relying on monkey-patches.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bot.public_output import (
    sanitize_embed_collection_for_public,
    sanitize_embed_for_public,
    sanitize_public_text,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import discord


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
    """Sanitize then send a message to any Messageable channel."""
    if content is not None:
        content = _maybe_sanitize_text(content)

    embed, embeds = _sanitize_embeds(embed, embeds)

    # Files and views are forwarded without modification.
    if embeds is not None:
        return await destination.send(
            content,
            embeds=embeds,
            file=file,
            files=files,
            view=view,
            **kwargs,
        )
    return await destination.send(
        content,
        embed=embed,
        file=file,
        files=files,
        view=view,
        **kwargs,
    )


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
    """Sanitize then reply to a Discord message."""
    if content is not None:
        content = _maybe_sanitize_text(content)

    embed, embeds = _sanitize_embeds(embed, embeds)

    return await message.reply(
        content,
        embed=embed,
        embeds=embeds,
        file=file,
        files=files,
        view=view,
        mention_author=mention_author,
        **kwargs,
    )


async def safe_edit(
    message: discord.Message,
    content: str | None = None,
    *,
    embed: discord.Embed | None = None,
    embeds: Sequence[discord.Embed] | None = None,
    view: discord.ui.View | None = None,
    **kwargs,
) -> discord.Message:
    """Sanitize then edit a Discord message."""
    if content is not None:
        content = _maybe_sanitize_text(content)

    embed, embeds = _sanitize_embeds(embed, embeds)

    return await message.edit(
        content=content,
        embed=embed,
        embeds=embeds,  # type: ignore[arg-type]
        view=view,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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
