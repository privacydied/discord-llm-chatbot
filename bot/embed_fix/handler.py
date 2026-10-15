"""Discord side of link-embed fixing (SaucyBot SiteManager/MessageValidator port).

Flow per message: cheap prefilter -> opt-out tags -> match sites -> per-guild
toggle -> channel permissions -> NSFW gate -> reply with fixed links ->
suppress the original message's embeds when the bot may manage messages.

The hook is a side effect. It never consumes the message, so the normal
router / LLM path still sees every link post.
"""

from __future__ import annotations

import asyncio
import io
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any
from urllib.parse import urlsplit

import discord

from bot.core.output import safe_reply
from bot.server_features import is_server_feature_enabled
from bot.utils.logging import get_logger

from . import instagram, ppv, translate
from .sites import DEFAULT_MAX_LINKS, EmbedFix, find_fixes, is_ignored, parse_domain_overrides, parse_site_list

logger = get_logger(__name__)

FEATURE_NAME = "embed_fix"
_DISCORD_TIMEOUT_S = 10.0
_DISCORD_ERRORS = (discord.HTTPException, asyncio.TimeoutError)
_MAX_EMBEDS = 10  # Discord's per-message embed cap
# Uploads take longer than a text reply (several MB from the NAS).
_UPLOAD_TIMEOUT_S = 60.0
# Sites whose media the bot downloads and attaches instead of linking a proxy.
UPLOAD_SITES = frozenset({"instagram"})
_MAX_UPLOAD_POSTS = 4
# Re-check points (seconds between checks) for the late-unfurl race; total ~15s.
SUPPRESS_RECHECK_DELAYS_S: tuple[float, ...] = (2.0, 4.0, 9.0)


@dataclass(frozen=True, slots=True)
class EmbedFixSettings:
    """Config snapshot, read per message so hot-reload applies at once."""

    enabled: bool = True
    disabled_sites: frozenset[str] = frozenset()
    domain_overrides: Mapping[str, str] | None = None
    max_links: int = DEFAULT_MAX_LINKS
    suppress_original: bool = True
    restrict_nsfw: bool = True
    translate_to: str | None = translate.DEFAULT_TARGET_LANG
    upload_media: bool = True

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> EmbedFixSettings:
        cfg = config or {}
        return cls(
            enabled=bool(cfg.get("EMBED_FIX_ENABLED", True)),
            disabled_sites=parse_site_list(cfg.get("EMBED_FIX_DISABLED_SITES")),
            domain_overrides=parse_domain_overrides(cfg.get("EMBED_FIX_DOMAIN_OVERRIDES")),
            max_links=int(cfg.get("EMBED_FIX_MAX_LINKS", DEFAULT_MAX_LINKS) or 0),
            suppress_original=bool(cfg.get("EMBED_FIX_SUPPRESS_ORIGINAL", True)),
            restrict_nsfw=bool(cfg.get("EMBED_FIX_RESTRICT_NSFW", True)),
            translate_to=translate.parse_target(cfg.get("EMBED_FIX_TRANSLATE_TO")),
            upload_media=bool(cfg.get("EMBED_FIX_UPLOAD_MEDIA", False)),
        )


def might_have_fixable_link(message: discord.Message) -> bool:
    """Very cheap check so the bot does not schedule a task per chat line."""
    content = getattr(message, "content", "") or ""
    return getattr(message, "guild", None) is not None and "http" in content.lower()


def _bot_permissions(message: discord.Message) -> discord.Permissions | None:
    channel, guild = message.channel, message.guild
    me = getattr(guild, "me", None)
    permissions_for = getattr(channel, "permissions_for", None)
    if me is None or permissions_for is None:
        return None
    return permissions_for(me)


def can_create_embed(message: discord.Message, perms: discord.Permissions | None) -> bool:
    """SaucyBot ChannelExtensions.CanCreateEmbed: guild text channels and threads only."""
    if perms is None or isinstance(message.channel, (discord.DMChannel, discord.GroupChannel)):
        return False
    if not (perms.send_messages and perms.embed_links and perms.read_message_history):
        return False
    if isinstance(message.channel, discord.Thread):
        return bool(perms.send_messages_in_threads)
    return True


def channel_is_nsfw(channel: Any) -> bool:
    """Threads inherit NSFW from the parent (discord.py Thread.is_nsfw does this)."""
    is_nsfw = getattr(channel, "is_nsfw", None)
    try:
        return bool(is_nsfw()) if callable(is_nsfw) else False
    except (AttributeError, TypeError):
        return False


def filter_nsfw(fixes: list[EmbedFix], channel: Any, *, restrict: bool) -> list[EmbedFix]:
    if not restrict or channel_is_nsfw(channel):
        return fixes
    return [fix for fix in fixes if not fix.nsfw]


async def _guild_allows(guild_id: int) -> bool:
    # Profile lookup can hit disk on a cache miss; keep it off the event loop.
    try:
        return await asyncio.to_thread(is_server_feature_enabled, guild_id, FEATURE_NAME)
    except Exception as exc:  # profile store failure must not break chat
        logger.warning(f"embed_fix.toggle_lookup_failed | guild={guild_id} error={exc}")
        return True


@dataclass(slots=True)
class _Rendered:
    content: str | None
    cards: list[discord.Embed]
    files: list[instagram.MediaFile]
    sent: list[EmbedFix]


def _can_upload(message: discord.Message, perms: discord.Permissions | None, settings: EmbedFixSettings) -> bool:
    return settings.upload_media and perms is not None and bool(perms.attach_files) and message.guild is not None


@dataclass(slots=True)
class _Uploads:
    files: list[instagram.MediaFile]
    cards: list[discord.Embed]
    positions: set[int]


async def _fetch_uploads(fixes: list[EmbedFix], max_bytes: int) -> _Uploads:
    """Download media (+ a styled card) for upload-site fixes within one shared byte budget.

    Video posts oginstagram can embed are skipped on purpose: a bot embed can't
    hold video, but Discord's unfurl of the oginstagram link plays it inside
    the card. Those, and fixes that got no media, go out as the proxy link.
    """
    out = _Uploads([], [], set())
    for fix in [f for f in fixes if f.site in UPLOAD_SITES][:_MAX_UPLOAD_POSTS]:
        room = max_bytes - sum(len(f.data) for f in out.files)
        slots = instagram.MAX_ITEMS - len(out.files)
        if room <= 0 or slots <= 0:
            break
        if urlsplit(fix.rewritten).hostname == instagram.OG_HOST and await instagram.has_link_video(fix.original):
            # Discord's unfurl of the oginstagram link puts the video inside the card.
            continue
        post = await instagram.fetch_post(fix.original, max_bytes=room)
        if post is None or not post.files:
            continue
        post = replace(post, files=post.files[:slots])
        out.files.extend(post.files)
        out.cards.append(instagram.build_card(post))
        out.positions.add(fix.position)
    return out


async def _render(fixes: list[EmbedFix], *, upload_budget: int = 0) -> _Rendered:
    """Split fixes into uploaded media (+ cards), reply text (rewritten links) and bot-built cards."""
    uploads = await _fetch_uploads(fixes, upload_budget) if upload_budget > 0 else _Uploads([], [], set())
    links = [fix for fix in fixes if not fix.card and fix.position not in uploads.positions]
    card_fixes = [fix for fix in fixes if fix.card]
    room = _MAX_EMBEDS - len(uploads.cards)
    cards = await ppv.build_cards([fix.rewritten for fix in card_fixes[:room]]) if card_fixes and room > 0 else []
    sent = [fix for fix in fixes if fix.position in uploads.positions] + links + (card_fixes if cards else [])
    sent.sort(key=lambda fix: fix.position)
    return _Rendered("\n".join(fix.rewritten for fix in links) or None, uploads.cards + cards, uploads.files, sent)


async def _send_fixes(message: discord.Message, fixes: list[EmbedFix], *, upload_budget: int = 0) -> list[EmbedFix]:
    rendered = await _render(fixes, upload_budget=upload_budget)
    if not rendered.content and not rendered.cards and not rendered.files:
        return []
    files = [discord.File(io.BytesIO(media.data), filename=media.filename) for media in rendered.files]
    try:
        await asyncio.wait_for(
            safe_reply(
                message,
                rendered.content,
                embeds=rendered.cards or None,
                files=files or None,
                mention_author=False,
                allowed_mentions=discord.AllowedMentions.none(),
                # Persistent marker (flags.suppress_notifications) the router
                # gate uses to tell embed-fix posts apart from chat replies.
                silent=True,
            ),
            timeout=_UPLOAD_TIMEOUT_S if files else _DISCORD_TIMEOUT_S,
        )
    except discord.NotFound:
        logger.debug(f"embed_fix.original_deleted | msg_id={message.id}")
        return []
    except _DISCORD_ERRORS as exc:
        logger.warning(f"embed_fix.reply_failed | msg_id={message.id} error={exc}")
        return []
    if files:
        logger.info(f"embed_fix.uploaded | msg_id={message.id} files={len(files)} bytes={sum(len(m.data) for m in rendered.files)}")
    return rendered.sent


async def _suppress_once(message: discord.Message) -> bool:
    # Raw edit is fine here: flags only, no text crosses the output boundary.
    try:
        await asyncio.wait_for(message.edit(suppress=True), timeout=_DISCORD_TIMEOUT_S)
    except discord.NotFound:
        logger.debug(f"embed_fix.suppress_skipped | reason=deleted msg_id={message.id}")
        return False
    except _DISCORD_ERRORS as exc:
        logger.warning(f"embed_fix.suppress_failed | msg_id={message.id} error={exc}")
        return False
    return True


async def _refetch(message: discord.Message) -> discord.Message | None:
    try:
        return await asyncio.wait_for(message.channel.fetch_message(message.id), timeout=_DISCORD_TIMEOUT_S)
    except discord.NotFound:
        return None
    except _DISCORD_ERRORS as exc:
        logger.debug(f"embed_fix.refetch_failed | msg_id={message.id} error={exc}")
        return None


def _embeds_visible(current: discord.Message) -> bool:
    return bool(current.embeds) or not current.flags.suppress_embeds


async def _suppress_original(message: discord.Message) -> None:
    """Hide the original link preview, and keep it hidden.

    Discord unfurls links asynchronously after the message is created. When
    the suppress PATCH lands before the unfurl finishes, the unfurl can attach
    the embed afterwards, so a single immediate suppress works only sometimes.
    Re-check on a short schedule and re-suppress when the embed came back.
    """
    if not await _suppress_once(message):
        return
    for delay in SUPPRESS_RECHECK_DELAYS_S:
        await asyncio.sleep(delay)
        current = await _refetch(message)
        if current is None:
            return
        if _embeds_visible(current):
            logger.info(f"embed_fix.suppress_reapplied | msg_id={message.id} after_s={delay}")
            if not await _suppress_once(message):
                return


def _collect_fixes(message: discord.Message, settings: EmbedFixSettings) -> list[EmbedFix]:
    content = message.content or ""
    if is_ignored(content):
        return []
    return find_fixes(
        content,
        disabled=settings.disabled_sites,
        overrides=settings.domain_overrides,
        limit=settings.max_links,
    )


async def maybe_fix_embeds(message: discord.Message, config: Mapping[str, Any] | None) -> list[EmbedFix]:
    """Reply with embed-friendly rewrites of supported links. Returns what was sent."""
    settings = EmbedFixSettings.from_config(config)
    if not settings.enabled or getattr(message.author, "bot", False) or not might_have_fixable_link(message):
        return []
    fixes = _collect_fixes(message, settings)
    if not fixes or not await _guild_allows(message.guild.id):
        return []
    perms = _bot_permissions(message)
    if not can_create_embed(message, perms):
        logger.debug(f"embed_fix.skip | reason=missing_permissions msg_id={message.id}")
        return []
    fixes = filter_nsfw(fixes, message.channel, restrict=settings.restrict_nsfw)
    if not fixes:
        return []
    budget = instagram.upload_budget(getattr(message.guild, "filesize_limit", None)) if _can_upload(message, perms, settings) else 0
    fixes = await _send_fixes(message, await translate.apply(fixes, settings.translate_to), upload_budget=budget)
    if not fixes:
        return []
    logger.info(
        f"embed_fix.sent | msg_id={message.id} sites={','.join(sorted({f.site for f in fixes}))} count={len(fixes)}",
        extra={"event": "embed_fix.sent", "msg_id": message.id, "count": len(fixes)},
    )
    if settings.suppress_original and perms.manage_messages:
        await _suppress_original(message)
    elif settings.suppress_original:
        logger.info(f"embed_fix.suppress_skipped | reason=no_manage_messages channel={getattr(message.channel, 'id', None)} msg_id={message.id}")
    return fixes
