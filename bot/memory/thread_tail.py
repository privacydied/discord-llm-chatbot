from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import discord

from bot.utils.logging import get_logger
from .mention_context import MentionContextBlock, PackagedItem  # reuse packaging schema

logger = get_logger(__name__)


def _is_thread_channel(ch: discord.abc.GuildChannel | discord.Thread | Any) -> bool:
    try:
        # Forum posts are Threads; include all thread channel types
        if isinstance(ch, discord.Thread):
            return True
        t = getattr(ch, "type", None)
        return t in (
            discord.ChannelType.public_thread,
            discord.ChannelType.private_thread,
            discord.ChannelType.news_thread,
        )
    except Exception:
        return False


def _sanitize_mentions(text: str, guild: Optional[discord.Guild]) -> str:
    """Lightweight mention sanitizer to keep alignment with existing behavior."""
    if not text:
        return ""
    import re

    def repl(m):
        uid = int(m.group(1))
        name = None
        try:
            if guild:
                mem = guild.get_member(uid)
                if mem:
                    name = mem.display_name
        except Exception:
            name = None
        return f"@{name or uid}"

    try:
        s = re.sub(r"<@!?(\d+)>", repl, text)
        s = s.strip()
        s = re.sub(r"\n{3,}", "\n\n", s)
        return s
    except Exception:
        return text.strip()


def _format_joined_text(items: List[PackagedItem]) -> str:
    n = len(items)
    parts: List[str] = []
    for i, it in enumerate(items, start=1):
        ts = it.created_at_iso
        try:
            if ts:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                ts = dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            pass
        header = f"[{i}/{n}] {it.author_name} – {ts}" if ts else f"[{i}/{n}] {it.author_name}"
        parts.append(header)
        parts.append(it.text_plain)
        parts.append("")
    return "\n".join(parts).strip()


async def resolve_thread_reply_target(
    bot: discord.Client, message: discord.Message, cfg: Dict[str, Any]
) -> Tuple[Optional[discord.Message], str]:
    """
    Resolve the reply target for thread messages at send-time.
    Returns (reply_target_message_or_None, reason)
    Rules:
    - Prefer absolute newest message in the thread.
    - If newest is our bot, and the directly previous is human, target that human.
    - If no human messages exist, return None (send to thread without reply reference).
    Fallbacks: on any error, return (message, "fallback_current").
    """
    try:
        ch = getattr(message, "channel", None)
        if not _is_thread_channel(ch):
            return message, "not_thread"

        # Fetch the last few messages to apply tie-breakers
        # newest_first by default; fetch up to 3 to be safe
        # IMPORTANT: exclude the triggering message itself
        msgs: List[discord.Message] = []
        try:
            async for m in ch.history(limit=3, before=message):
                msgs.append(m)
        except Exception:
            msgs = []

        if not msgs:
            return message, "empty_history"

        newest = msgs[0]
        # Determine if newest is ours
        try:
            is_bot_newest = bool(getattr(newest.author, "bot", False)) and int(
                getattr(newest.author, "id", 0)
            ) == int(getattr(bot.user, "id", 0))
        except Exception:
            is_bot_newest = False

        if is_bot_newest and len(msgs) >= 2:
            prev = msgs[1]
            is_prev_human = not bool(getattr(prev.author, "bot", False))
            if is_prev_human:
                return prev, "prev_human"

        # If newest is human or previous not suitable, reply to newest
        if not bool(getattr(newest.author, "bot", False)):
            return newest, "newest_human"
        else:
            # Newest is bot (not necessarily ours) and no immediate human before
            # If no human messages exist in the window, send without reply reference
            only_bots = all(bool(getattr(m.author, "bot", False)) for m in msgs)
            if only_bots:
                return None, "no_humans"
            # Otherwise find the latest human in window
            for m in msgs:
                if not bool(getattr(m.author, "bot", False)):
                    return m, "latest_human_window"
            return None, "no_humans"
    except Exception:
        return message, "fallback_current"

async def resolve_implicit_anchor(
    bot: discord.Client, message: discord.Message, cfg: Dict[str, Any]
) -> Tuple[Optional[discord.Message], str]:
    """
    For LONE_CASE (non-thread), when the message @mentions the bot and carries no substantive content,
    choose an implicit anchor: the most recent human message above within a small look-back window.
    Returns (anchor_or_None, reason).
    """
    try:
        ch = getattr(message, "channel", None)
        if _is_thread_channel(ch):
            return None, "not_applicable_in_thread"
        # Look-back limits (small, cheap)
        try:
            k = int(cfg.get("THREAD_CONTEXT_TAIL_COUNT", 5))
        except Exception:
            k = 5
        lookback = max(3, min(10, k + 2))

        # Optional recency guard using MEM_MAX_AGE_MIN
        try:
            max_age_min = int(cfg.get("MEM_MAX_AGE_MIN", 240))
        except Exception:
            max_age_min = 240
        now = datetime.now(timezone.utc)

        async for m in ch.history(limit=lookback, before=message):
            try:
                if (now - (m.created_at or now)).total_seconds() > max_age_min * 60:
                    continue
            except Exception:
                pass
            if not bool(getattr(m.author, "bot", False)):
                return m, "nearest_human"
        return None, "no_recent_human"
    except Exception:
        return None, "exception"

async def collect_implicit_anchor_context(
    bot: discord.Client,
    message: discord.Message,
    anchor: Optional[discord.Message],
    cfg: Dict[str, Any],
) -> Optional[Tuple[str, MentionContextBlock]]:
    """
    Build a bounded tail context before the implicit anchor in non-thread channels.
    Returns (joined_text, MentionContextBlock) with source='discord-implicit-anchor'.
    """
    try:
        ch = getattr(message, "channel", None)
        if _is_thread_channel(ch) or not anchor:
            return None

        try:
            k = int(cfg.get("THREAD_CONTEXT_TAIL_COUNT", 5))
        except Exception:
            k = 5
        k = max(0, min(k, 40))

        tail: List[discord.Message] = []
        try:
            async for m in ch.history(limit=k * 3, before=anchor):
                is_bot = bool(getattr(m.author, "bot", False))
                is_ours = int(getattr(m.author, "id", 0)) == int(getattr(bot.user, "id", 0))
                if is_bot and not is_ours:
                    continue
                tail.append(m)
                if len(tail) >= k:
                    break
        except Exception:
            tail = []

        tail = list(reversed(tail))

        guild = getattr(message, "guild", None)
        items: List[PackagedItem] = []
        for i, m in enumerate(tail, start=1):
            author_name = getattr(m.author, "display_name", getattr(m.author, "name", f"User({getattr(m.author, 'id', 'unknown')})"))
            txt = _sanitize_mentions(m.content or "", guild)
            items.append(
                PackagedItem(
                    idx=i,
                    id=str(getattr(m, "id", "")),
                    author_name=author_name,
                    author_id=str(getattr(m.author, "id", "")),
                    created_at_iso=(getattr(m, "created_at", None) or datetime.now(timezone.utc)).isoformat(),
                    text_plain=txt,
                    has_attachments=bool(getattr(m, "attachments", None)),
                    attachment_summaries_if_available=None,
                    jump_url=getattr(m, "jump_url", ""),
                )
            )

        joined_text = _format_joined_text(items) if items else ""
        anc = {
            "id": str(getattr(anchor, "id", "")),
            "author": getattr(anchor.author, "display_name", getattr(anchor.author, "name", "")),
            "created_at_iso": (getattr(anchor, "created_at", None) or datetime.now(timezone.utc)).isoformat(),
            "jump_url": getattr(anchor, "jump_url", ""),
        }
        block = MentionContextBlock(
            source="discord-implicit-anchor",
            conversation_id=str(getattr(anchor, "id", "")),
            anchor=anc,
            count=len(items),
            items=items,
            joined_text=joined_text,
            truncated=False,
        )

        try:
            logger.info(
                "anchor_ok",
                extra={
                    "subsys": "mem.ctx",
                    "event": "anchor_ok",
                    "guild_id": getattr(guild, "id", None),
                    "user_id": getattr(getattr(message, "author", None), "id", None),
                    "msg_id": getattr(message, "id", None),
                    "detail": {"case": "LONE", "msgs": len(items), "reply_target": anc.get("id")},
                },
            )
        except Exception:
            pass

        return joined_text, block
    except Exception as e:
        try:
            logger.info(
                "anchor_none",
                extra={
                    "subsys": "mem.ctx",
                    "event": "anchor_none",
                    "guild_id": getattr(getattr(message, "guild", None), "id", None),
                    "user_id": getattr(getattr(message, "author", None), "id", None),
                    "msg_id": getattr(message, "id", None),
                    "detail": {"reason": str(e)[:120]},
                },
            )
        except Exception:
            pass
        return None

async def collect_thread_tail_context(
    bot: discord.Client,
    message: discord.Message,
    reply_target: Optional[discord.Message],
    cfg: Dict[str, Any],
) -> Optional[Tuple[str, MentionContextBlock]]:
    """
    Collect up to K previous messages in the same thread before the reply_target (or current message if None),
    ordered oldest → newest. Include humans + our bot; exclude other bots/system messages.
    Package as MentionContextBlock with source='discord-thread-tail' and return (joined_text, block).
    On any error, return None gracefully.
    """
    try:
        ch = getattr(message, "channel", None)
        if not _is_thread_channel(ch):
            return None

        # Hard-enabled per design: always on in thread channels
        enabled = True

        try:
            k = int(cfg.get("THREAD_CONTEXT_TAIL_COUNT", 5))
        except Exception:
            k = 5
        k = max(0, min(k, 40))  # cap to a reasonable bound

        # Determine anchor for collection (reply_target or current message)
        anchor = reply_target or message

        # Fetch messages strictly before anchor
        tail: List[discord.Message] = []
        try:
            # newest_first; we'll reverse later
            async for m in ch.history(limit=k * 3, before=anchor):
                # Exclude other bots; include our bot and humans
                is_bot = bool(getattr(m.author, "bot", False))
                is_ours = int(getattr(m.author, "id", 0)) == int(getattr(bot.user, "id", 0))
                if is_bot and not is_ours:
                    continue
                tail.append(m)
                if len(tail) >= k:
                    break
        except Exception:
            # Fallback: nothing collected
            tail = []

        # Reverse to oldest → newest
        tail = list(reversed(tail))

        # Build items
        guild = getattr(message, "guild", None)
        items: List[PackagedItem] = []
        for i, m in enumerate(tail, start=1):
            author_name = getattr(m.author, "display_name", getattr(m.author, "name", f"User({getattr(m.author, 'id', 'unknown')})"))
            txt = _sanitize_mentions(m.content or "", guild)
            it = PackagedItem(
                idx=i,
                id=str(getattr(m, "id", "")),
                author_name=author_name,
                author_id=str(getattr(m.author, "id", "")),
                created_at_iso=(getattr(m, "created_at", None) or datetime.now(timezone.utc)).isoformat(),
                text_plain=txt,
                has_attachments=bool(getattr(m, "attachments", None)),
                attachment_summaries_if_available=None,
                jump_url=getattr(m, "jump_url", ""),
            )
            items.append(it)

        joined_text = _format_joined_text(items) if items else ""

        # Package block
        anc = {
            "id": str(getattr(anchor, "id", "")),
            "author": getattr(anchor.author, "display_name", getattr(anchor.author, "name", "")),
            "created_at_iso": (getattr(anchor, "created_at", None) or datetime.now(timezone.utc)).isoformat(),
            "jump_url": getattr(anchor, "jump_url", ""),
        }
        block = MentionContextBlock(
            source="discord-thread-tail",
            conversation_id=str(getattr(ch, "id", "")),
            anchor=anc,
            count=len(items),
            items=items,
            joined_text=joined_text,
            truncated=False,
        )

        # Logging (quiet)
        try:
            logger.info(
                "tail_ok",
                extra={
                    "subsys": "mem.ctx",
                    "event": "tail_ok",
                    "guild_id": getattr(guild, "id", None),
                    "user_id": getattr(getattr(message, "author", None), "id", None),
                    "msg_id": getattr(message, "id", None),
                    "detail": {"k": k, "reply_target": anc.get("id"), "count": len(items)},
                },
            )
        except Exception:
            pass

        return joined_text, block
    except Exception as e:
        try:
            logger.info(
                "tail_fallback",
                extra={
                    "subsys": "mem.ctx",
                    "event": "tail_fallback",
                    "guild_id": getattr(getattr(message, "guild", None), "id", None),
                    "user_id": getattr(getattr(message, "author", None), "id", None),
                    "msg_id": getattr(message, "id", None),
                    "detail": {"reason": str(e)[:120]},
                },
            )
        except Exception:
            pass
        return None
