"""Mention context utilities for Discord message processing.

Handles anchor resolution, thread/reply context collection, and text sanitization.
"""

import asyncio
import contextlib
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import discord

from ..utils.logging import get_logger


logger = get_logger(__name__)


# Case classification constants
THREAD_CASE = "thread"
REPLY_CASE = "reply"
LONE_CASE = "lone"


@dataclass
class PackagedItem:
    idx: int
    id: str
    author_name: str
    author_id: str
    created_at_iso: str
    text_plain: str
    has_attachments: bool
    attachment_summaries_if_available: list[str] | None
    jump_url: str


@dataclass
class MentionContextBlock:
    source: str  # discord-thread | discord-reply-chain
    conversation_id: str  # thread_id or anchor_message_id
    anchor: dict[str, Any]
    count: int
    items: list[PackagedItem]
    joined_text: str
    truncated: bool


def _is_thread_channel(ch: discord.abc.Messageable) -> bool:
    """Return True if channel is a thread type (including Forum post threads)."""
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
    except (AttributeError, TypeError):
        return False


def _now_utc() -> datetime:
    return datetime.now(UTC)


def classify_message_case(message: discord.Message) -> str:
    try:
        if _is_thread_channel(message.channel):
            return THREAD_CASE
        if getattr(message, "reference", None) is not None:
            return REPLY_CASE
        return LONE_CASE
    except (AttributeError, TypeError):
        return LONE_CASE


async def _fetch_message_safely(channel: discord.abc.Messageable, mid: int, timeout_s: float) -> discord.Message | None:
    try:
        return await asyncio.wait_for(channel.fetch_message(mid), timeout=timeout_s)
    except (asyncio.TimeoutError, discord.NotFound, discord.Forbidden, discord.HTTPException):
        return None


async def resolve_anchor(
    message: discord.Message,
    case: str,
    timeout_s: float,
) -> discord.Message | None:
    if case == LONE_CASE:
        return None

    if case == THREAD_CASE:
        # Best-effort: earliest message in the thread (acts as thread starter)
        try:
            hist = message.channel.history(limit=200, oldest_first=True)

            async def _first() -> discord.Message | None:
                async for m in hist:
                    return m
                return None

            return await asyncio.wait_for(_first(), timeout=timeout_s)
        except (asyncio.TimeoutError, asyncio.CancelledError, discord.NotFound, discord.Forbidden, discord.HTTPException):
            return None

    # REPLY_CASE
    try:
        if getattr(message, "reference", None) is None:
            return None
        # Prefer immediate parent as a reliable anchor
        ref = message.reference
        parent = getattr(ref, "resolved", None)
        if parent is None:
            ref_id = getattr(ref, "message_id", None)
            if ref_id:
                parent = await _fetch_message_safely(message.channel, ref_id, timeout_s)
        if parent is None:
            # No resolvable parent; fail gracefully
            return None

        # Optionally walk up to the root, but never lose the parent fallback
        cur = parent
        seen = set()
        while getattr(cur, "reference", None) is not None:
            r = cur.reference
            rid = getattr(r, "message_id", None)
            if rid is None or rid in seen:
                break
            seen.add(rid)
            p2 = await _fetch_message_safely(message.channel, rid, timeout_s)
            if not p2:
                break
            cur = p2
        return cur or parent
    except (asyncio.TimeoutError, asyncio.CancelledError, discord.NotFound, discord.Forbidden, discord.HTTPException, AttributeError, TypeError):
        # Best-effort fallback: use parent if any
        try:
            ref = getattr(message, "reference", None)
            return getattr(ref, "resolved", None)
        except (AttributeError, TypeError):
            return None


def _sanitize_mentions(text: str, guild: discord.Guild | None) -> str:
    if not text:
        return ""

    def repl(m) -> str:
        uid = int(m.group(1))
        name = None
        try:
            if guild:
                mem = guild.get_member(uid)
                if mem:
                    name = mem.display_name
        except (AttributeError, TypeError, ValueError):
            name = None
        if not name:
            try:
                # Fallback to global cache
                from discord import utils as dutils  # local import

                u = dutils.get(guild.members, id=uid) if guild else None
                if u:
                    name = getattr(u, "display_name", None)
            except (AttributeError, TypeError):
                name = None
        return f"@{name or uid}"

    try:
        s = re.sub(r"<@!?(\d+)>", repl, text)
        # Normalize whitespace: strip, collapse many newlines
        s = s.strip()
        return re.sub(r"\n{3,}", "\n\n", s)
    except (re.error, AttributeError, TypeError):
        return text.strip()


def _format_joined_text(items: list[PackagedItem]) -> str:
    n = len(items)
    parts: list[str] = []
    for i, it in enumerate(items, start=1):
        ts = it.created_at_iso
        # Prefer friendly UTC timestamp (YYYY-MM-DD HH:MM UTC)
        try:
            if ts:
                dt = datetime.fromisoformat(ts)
                ts = dt.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")
        except (ValueError, AttributeError, TypeError) as e:
            logger.debug(f"Failed to parse timestamp {ts}: {e}")
        header = f"[{i}/{n}] {it.author_name} \u2013 {ts}" if ts else f"[{i}/{n}] {it.author_name}"
        parts.append(header)
        parts.append(it.text_plain)
        parts.append("")
    return "\n".join(parts).strip()


def _include_message(msg: discord.Message, bot_user_id: int) -> bool:
    try:
        return not (getattr(msg.author, "bot", False) and int(getattr(msg.author, "id", 0)) != int(bot_user_id))
    except (AttributeError, TypeError, ValueError) as e:
        logger.debug(f"Failed to check message inclusion: {e}")
        return True


def _within_age(msg: discord.Message, max_age_min: int, now: datetime) -> bool:
    try:
        return (now - (msg.created_at or now)) <= timedelta(minutes=max_age_min)
    except (AttributeError, TypeError, ValueError) as e:
        logger.debug(f"Failed to check message age: {e}")
        return True


async def _collect_thread_context(
    bot: discord.Client,
    message: discord.Message,
    anchor: discord.Message | None,
    *,
    max_msgs: int,
    max_chars: int,
    max_age_min: int,
    timeout_s: float,
) -> tuple[list[discord.Message], bool]:
    """Collect recent thread messages in chronological order with caps. Returns (messages, truncated)."""
    now = _now_utc()
    collected: list[discord.Message] = []
    total_chars = 0
    truncated = False

    async def _walk():
        async for m in message.channel.history(limit=max_msgs * 3, oldest_first=True):
            yield m

    walker = _walk()
    # Guard the iteration with a soft time limit by chunking
    start_time = time.monotonic()
    try:
        async for m in walker:
            if time.monotonic() - start_time > max(timeout_s * 2, 5):
                # Safety guard: don't overrun
                truncated = True
                break
            if not _include_message(m, int(getattr(bot.user, "id", 0) or 0)):
                continue
            if not _within_age(m, max_age_min, now):
                # Allow the root/anchor even if older
                if anchor and getattr(m, "id", None) == getattr(anchor, "id", None):
                    pass
                else:
                    continue
            txt = m.content or ""
            if len(collected) + 1 > max_msgs or (total_chars + len(txt)) > max_chars:
                truncated = True
                break
            collected.append(m)
            total_chars += len(txt)
    except (AttributeError, TypeError, ValueError):
        # On any error, return empty to trigger fallback
        return [], False

    # Ensure anchor and current are included
    ids = {getattr(m, "id", None) for m in collected}
    if anchor and getattr(anchor, "id", None) not in ids:
        collected.insert(0, anchor)
    if getattr(message, "id", None) not in ids:
        collected.append(message)

    # Deduplicate preserving order
    seen = set()
    uniq: list[discord.Message] = []
    for m in collected:
        mid = getattr(m, "id", None)
        if mid in seen:
            continue
        seen.add(mid)
        uniq.append(m)

    return uniq, truncated


async def _collect_reply_chain(
    bot: discord.Client,
    message: discord.Message,
    anchor: discord.Message | None,
    *,
    max_msgs: int,
    max_chars: int,
    max_age_min: int,
    timeout_s: float,
) -> tuple[list[discord.Message], bool]:
    """Collect a bounded tail of the reply chain nearest to the triggering message.
    Returns (messages, truncated) where messages are in chronological order (oldest first).
    """
    now = _now_utc()
    collected: list[discord.Message] = []
    total_chars = 0
    truncated = False

    try:
        parent = anchor
        if parent is None:
            ref = getattr(message, "reference", None)
            if ref:
                resolved = getattr(ref, "resolved", None)
                if isinstance(resolved, discord.Message):
                    parent = resolved
                else:
                    ref_id = getattr(ref, "message_id", None)
                    if ref_id:
                        parent = await _fetch_message_safely(message.channel, ref_id, timeout_s)

        if parent is not None:
            collected.append(parent)
            total_chars += len(parent.content or "")

        cur = message
        while getattr(cur, "reference", None) is not None:
            if len(collected) >= max_msgs:
                truncated = True
                break
            ref = cur.reference
            rid = getattr(ref, "message_id", None)
            if rid is None:
                break
            next_msg = await _fetch_message_safely(message.channel, rid, timeout_s)
            if not next_msg:
                break
            if not _include_message(next_msg, int(getattr(bot.user, "id", 0) or 0)):
                break
            if not _within_age(next_msg, max_age_min, now):
                break
            collected.append(next_msg)
            total_chars += len(next_msg.content or "")
            cur = next_msg
    except (AttributeError, TypeError, ValueError, asyncio.TimeoutError, asyncio.CancelledError, discord.NotFound, discord.Forbidden, discord.HTTPException):
        # Best-effort fallback
        pass

    collected.reverse()  # Reverse to chronological order
    return collected, truncated


def _package(
    bot: discord.Client,
    message: discord.Message,
    case: str,
    anchor: discord.Message | None,
    messages: list[discord.Message],
) -> MentionContextBlock:
    guild = message.guild
    items: list[PackagedItem] = []
    for i, m in enumerate(messages, start=1):
        author_name = getattr(
            m.author,
            "display_name",
            getattr(m.author, "name", f"User({getattr(m.author, 'id', 'unknown')})"),
        )
        txt = _sanitize_mentions(m.content or "", guild)
        it = PackagedItem(
            idx=i,
            id=str(getattr(m, "id", "")),
            author_name=author_name,
            author_id=str(getattr(m.author, "id", "")),
            created_at_iso=(m.created_at or _now_utc()).isoformat(),
            text_plain=txt,
            has_attachments=bool(getattr(m, "attachments", None)),
            attachment_summaries_if_available=None,
            jump_url=getattr(m, "jump_url", ""),
        )
        items.append(it)

    joined_text = _format_joined_text(items)
    conv_id = str(getattr(message.channel, "id", "")) if case == THREAD_CASE else str(getattr(anchor, "id", ""))
    anc = None
    if anchor:
        anc = {
            "id": str(getattr(anchor, "id", "")),
            "author": getattr(anchor.author, "display_name", getattr(anchor.author, "name", "")),
            "created_at_iso": (anchor.created_at or _now_utc()).isoformat(),
            "jump_url": getattr(anchor, "jump_url", ""),
        }
    return MentionContextBlock(
        source=("discord-thread" if case == THREAD_CASE else "discord-reply-chain"),
        conversation_id=conv_id,
        anchor=anc or {},
        count=len(items),
        items=items,
        joined_text=joined_text,
        truncated=False,  # caller may update
    )


async def maybe_build_mention_context(
    bot: discord.Client,
    message: discord.Message,
    cfg: dict[str, Any],
) -> tuple[str, MentionContextBlock] | None:
    """Build mention-aware context when feature is enabled and message mentions the bot.
    Returns (joined_text, block) or None on fallback/no-op.
    """
    # Hard-enabled feature: always on regardless of env/config
    enabled = True
    if not enabled:
        return None

    # Only trigger on @mention
    try:
        mentioned = bot.user in (getattr(message, "mentions", None) or [])
    except (AttributeError, TypeError):
        mentioned = False
    if not mentioned:
        return None

    # Caps and limits
    try:
        # Tail size K for reply/thread context collection
        tail_k = int(cfg.get("THREAD_CONTEXT_TAIL_COUNT", 5))
    except (ValueError, TypeError):
        tail_k = 5
    # Global caps (chars/timeouts)
    try:
        max_chars = int(cfg.get("MEM_MAX_CHARS", 8000))
        max_age_min = int(cfg.get("MEM_MAX_AGE_MIN", 240))
        timeout_s = float(cfg.get("MEM_FETCH_TIMEOUT_S", 5))
        subsys = str(cfg.get("MEM_LOG_SUBSYS", "mem.ctx"))
    except (ValueError, TypeError):
        max_chars, max_age_min, timeout_s, subsys = 8000, 240, 5.0, "mem.ctx"
    # Enforce sane bounds
    tail_k = max(0, min(tail_k, 40))

    t0 = time.perf_counter()
    case = classify_message_case(message)

    # Anchor resolution
    try:
        anchor = await resolve_anchor(message, case, timeout_s)
    except (asyncio.TimeoutError, asyncio.CancelledError, discord.NotFound, discord.Forbidden, discord.HTTPException, AttributeError, TypeError) as e:
        with contextlib.suppress(Exception):
            logger.info(
                "collect_fallback",
                extra={
                    "subsys": subsys,
                    "event": "collect_fallback",
                    "guild_id": getattr(getattr(message, "guild", None), "id", None),
                    "user_id": getattr(getattr(message, "author", None), "id", None),
                    "msg_id": getattr(message, "id", None),
                    "detail": {
                        "reason": f"anchor_resolution:{e.__class__.__name__}",
                        "case": case,
                    },
                },
            )
        return None

    # LONE case: no special context
    if case == LONE_CASE or anchor is None:
        return None

    # Collect context
    try:
        if case == THREAD_CASE:
            messages, truncated = await _collect_thread_context(
                bot,
                message,
                anchor,
                max_msgs=tail_k,
                max_chars=max_chars,
                max_age_min=max_age_min,
                timeout_s=timeout_s,
            )
        else:
            messages, _truncated = await _collect_reply_chain(
                bot,
                message,
                anchor,
                max_msgs=tail_k,
                max_chars=max_chars,
                max_age_min=max_age_min,
                timeout_s=timeout_s,
            )
    except TimeoutError:
        with contextlib.suppress(Exception):
            logger.info(
                "collect_fallback",
                extra={
                    "subsys": subsys,
                    "event": "collect_fallback",
                    "guild_id": getattr(getattr(message, "guild", None), "id", None),
                    "user_id": getattr(getattr(message, "author", None), "id", None),
                    "msg_id": getattr(message, "id", None),
                    "detail": {"reason": "timeout", "case": case},
                },
            )
        return None
    except (asyncio.TimeoutError, asyncio.CancelledError, discord.NotFound, discord.Forbidden, discord.HTTPException, AttributeError, TypeError, ValueError) as e:
        with contextlib.suppress(Exception):
            logger.info(
                "collect_fallback",
                extra={
                    "subsys": subsys,
                    "event": "collect_fallback",
                    "guild_id": getattr(getattr(message, "guild", None), "id", None),
                    "user_id": getattr(getattr(message, "author", None), "id", None),
                    "msg_id": getattr(message, "id", None),
                    "detail": {
                        "reason": f"collect_error:{e.__class__.__name__}",
                        "case": case,
                    },
                },
            )
        return None

    if not messages:
        return None

    block = _package(bot, message, case, anchor, messages)
    block.truncated = bool(block.truncated or (len(messages) >= tail_k))

    # Telemetry
    try:
        dt_ms = int((time.perf_counter() - t0) * 1000)
        tot_chars = sum(len(it.text_plain or "") for it in block.items)
        logger.info(
            "collect_ok",
            extra={
                "subsys": subsys,
                "event": "collect_ok",
                "guild_id": getattr(getattr(message, "guild", None), "id", None),
                "user_id": getattr(getattr(message, "author", None), "id", None),
                "msg_id": getattr(message, "id", None),
                "detail": {
                    "case": case,
                    "msgs": block.count,
                    "chars": tot_chars,
                    "ms": dt_ms,
                },
            },
        )
        if block.truncated:
            logger.info(
                "collect_truncated",
                extra={
                    "subsys": subsys,
                    "event": "collect_truncated",
                    "guild_id": getattr(getattr(message, "guild", None), "id", None),
                    "user_id": getattr(getattr(message, "author", None), "id", None),
                    "msg_id": getattr(message, "id", None),
                    "detail": {"reason": "cap_hit", "msgs": block.count},
                },
            )
    except (AttributeError, TypeError, ValueError) as e:
        logger.debug(f"Failed to log truncation breadcrumb: {e}")

    return block.joined_text, block


async def build_mention_context(
    bot: discord.Client,
    message: discord.Message,
    *,
    max_msgs: int = 50,
    max_chars: int = 12000,
    max_age_min: int = 60,
    timeout_s: float = 3.0,
) -> str:
    """Build a sanitized context string for a mention-triggered response.

    Includes thread/reply history with caps on messages, chars, and age.
    """
    case = classify_message_case(message)

    if case == THREAD_CASE:
        anchor = await resolve_anchor(message, case, timeout_s)
        collected, truncated = await _collect_thread_context(
            bot,
            message,
            anchor,
            max_msgs=max_msgs,
            max_chars=max_chars,
            max_age_min=max_age_min,
            timeout_s=timeout_s,
        )
    elif case == REPLY_CASE:
        anchor = await resolve_anchor(message, case, timeout_s)
        collected, truncated = await _collect_reply_chain(
            bot,
            message,
            anchor,
            max_msgs=max_msgs,
            max_chars=max_chars,
            max_age_min=max_age_min,
            timeout_s=timeout_s,
        )
    else:  # LONE_CASE
        collected = [message]
        truncated = False

    # Package into typed items
    items: list[PackagedItem] = []
    for m in collected:
        author_name = m.author.display_name if m.author else "Unknown"
        text = m.content or ""
        ts = m.created_at.isoformat() if m.created_at else ""
        items.append(PackagedItem(author_name=author_name, text_plain=text, created_at_iso=ts))

    joined = _format_joined_text(items)
    if truncated:
        joined += "\n\n[Context truncated due to length/age limits]"
    return _sanitize_mentions(joined, message.guild)
