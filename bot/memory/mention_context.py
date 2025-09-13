from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import discord

from bot.utils.logging import get_logger


logger = get_logger(__name__)


# Cases
THREAD_CASE = "THREAD"
REPLY_CASE = "REPLY"
LONE_CASE = "LONE"


@dataclass
class PackagedItem:
    idx: int
    id: str
    author_name: str
    author_id: str
    created_at_iso: str
    text_plain: str
    has_attachments: bool
    attachment_summaries_if_available: Optional[List[str]]
    jump_url: str


@dataclass
class MentionContextBlock:
    source: str  # discord-thread | discord-reply-chain
    conversation_id: str  # thread_id or anchor_message_id
    anchor: Dict[str, Any]
    count: int
    items: List[PackagedItem]
    joined_text: str
    truncated: bool


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


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def classify_message_case(message: discord.Message) -> str:
    try:
        if _is_thread_channel(message.channel):
            return THREAD_CASE
        if getattr(message, "reference", None) is not None:
            return REPLY_CASE
        return LONE_CASE
    except Exception:
        return LONE_CASE


async def _fetch_message_safely(channel: discord.abc.Messageable, mid: int, timeout_s: float) -> Optional[discord.Message]:
    try:
        return await asyncio.wait_for(channel.fetch_message(mid), timeout=timeout_s)
    except Exception:
        return None


async def resolve_anchor(message: discord.Message, case: str, timeout_s: float) -> Optional[discord.Message]:
    if case == LONE_CASE:
        return None

    if case == THREAD_CASE:
        # Best-effort: earliest message in the thread (acts as thread starter)
        try:
            hist = message.channel.history(limit=200, oldest_first=True)
            # Guard the first fetch with timeout
            first: Optional[discord.Message] = None
            async def _first() -> Optional[discord.Message]:
                async for m in hist:
                    first = m
                    return first
                return None
            return await asyncio.wait_for(_first(), timeout=timeout_s)
        except Exception:
            return None

    # REPLY_CASE
    try:
        cur = message
        seen = set()
        while getattr(cur, "reference", None) is not None:
            ref = cur.reference
            ref_id = getattr(ref, "message_id", None)
            if ref_id is None or ref_id in seen:
                break
            seen.add(ref_id)
            parent = await _fetch_message_safely(message.channel, ref_id, timeout_s)
            if not parent:
                break
            cur = parent
        return cur if cur is not message else (await _fetch_message_safely(message.channel, getattr(message.reference, "message_id", 0), timeout_s))
    except Exception:
        return None


def _sanitize_mentions(text: str, guild: Optional[discord.Guild]) -> str:
    if not text:
        return ""
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
        if not name:
            try:
                # Fallback to global cache
                from discord import utils as dutils  # local import
                u = dutils.get(guild.members, id=uid) if guild else None
                if u:
                    name = getattr(u, "display_name", None)
            except Exception:
                name = None
        return f"@{name or uid}"
    try:
        s = re.sub(r"<@!?(\d+)>", repl, text)
        # Normalize whitespace: strip, collapse many newlines
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
        # Prefer friendly UTC timestamp (YYYY-MM-DD HH:MM UTC)
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


def _include_message(msg: discord.Message, bot_user_id: int) -> bool:
    try:
        if getattr(msg.author, "bot", False) and int(getattr(msg.author, "id", 0)) != int(bot_user_id):
            return False
        return True
    except Exception:
        return True


def _within_age(msg: discord.Message, max_age_min: int, now: datetime) -> bool:
    try:
        return (now - (msg.created_at or now)) <= timedelta(minutes=max_age_min)
    except Exception:
        return True


async def _collect_thread_context(
    bot: discord.Client,
    message: discord.Message,
    anchor: Optional[discord.Message],
    *,
    max_msgs: int,
    max_chars: int,
    max_age_min: int,
    timeout_s: float,
) -> Tuple[List[discord.Message], bool]:
    """Collect recent thread messages in chronological order with caps. Returns (messages, truncated)."""
    now = _now_utc()
    collected: List[discord.Message] = []
    total_chars = 0
    truncated = False

    try:
        async def _walk():
            async for m in message.channel.history(limit=max_msgs * 3, oldest_first=True):
                yield m
        walker = _walk()
        # Guard the iteration with a soft time limit by chunking
        start_time = time.monotonic()
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
            txt = (m.content or "")
            if len(collected) + 1 > max_msgs or (total_chars + len(txt)) > max_chars:
                truncated = True
                break
            collected.append(m)
            total_chars += len(txt)
    except Exception:
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
    uniq: List[discord.Message] = []
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
    anchor: Optional[discord.Message],
    *,
    max_msgs: int,
    max_chars: int,
    max_age_min: int,
    timeout_s: float,
) -> Tuple[List[discord.Message], bool]:
    """Collect linear reply chain around anchor (root) until current message. Returns (messages, truncated)."""
    if not anchor:
        return [], False

    now = _now_utc()
    collected: List[discord.Message] = []
    total_chars = 0
    truncated = False

    # Upward chain (anchor path) — we already have anchor at root; reconstruct path if available
    try:
        path: List[discord.Message] = []
        cur = message
        # Build path to root (excluding current) then reverse
        while getattr(cur, "reference", None) is not None:
            ref_id = getattr(cur.reference, "message_id", None)
            if ref_id is None:
                break
            parent = await _fetch_message_safely(message.channel, ref_id, timeout_s)
            if not parent:
                break
            path.append(parent)
            if getattr(parent, "reference", None) is None:
                break
            cur = parent
        # Ensure the top of path is the true root aka anchor
        if path:
            if getattr(path[-1], "id", None) != getattr(anchor, "id", None):
                path.append(anchor)
        else:
            path = [anchor]
    except Exception:
        path = [anchor]

    # Start collected with upward chain (oldest -> newest)
    for p in reversed(path):
        if not _include_message(p, int(getattr(bot.user, "id", 0) or 0)):
            continue
        if not _within_age(p, max_age_min, now) and getattr(p, "id", None) != getattr(anchor, "id", None):
            continue
        txt = (p.content or "")
        if len(collected) + 1 > max_msgs or (total_chars + len(txt)) > max_chars:
            truncated = True
            break
        collected.append(p)
        total_chars += len(txt)

    # Downward: follow replies to any already-included message, until we reach the triggering message
    try:
        included_ids = {getattr(m, "id", None) for m in collected}
        reached_trigger = getattr(message, "id", None) in included_ids
        async def _walk():
            async for m in message.channel.history(limit=max_msgs * 4, oldest_first=True, after=anchor.created_at):
                yield m
        if not reached_trigger:
            walker = _walk()
            start_time = time.monotonic()
            async for m in walker:
                if time.monotonic() - start_time > max(timeout_s * 2, 5):
                    truncated = True
                    break
                if m.created_at > message.created_at:
                    # Stop at the point we pass the trigger
                    break
                if not _include_message(m, int(getattr(bot.user, "id", 0) or 0)):
                    continue
                if not _within_age(m, max_age_min, now) and getattr(m, "id", None) != getattr(anchor, "id", None):
                    continue
                ref = getattr(m, "reference", None)
                if not ref or getattr(ref, "message_id", None) not in included_ids:
                    continue
                txt = (m.content or "")
                if len(collected) + 1 > max_msgs or (total_chars + len(txt)) > max_chars:
                    truncated = True
                    break
                collected.append(m)
                total_chars += len(txt)
                included_ids.add(getattr(m, "id", None))
                if getattr(m, "id", None) == getattr(message, "id", None):
                    reached_trigger = True
                    break
    except Exception:
        # ignore; best-effort
        pass

    # Ensure current message is included
    ids = {getattr(m, "id", None) for m in collected}
    if getattr(message, "id", None) not in ids:
        collected.append(message)

    # Deduplicate preserving order
    seen = set()
    uniq: List[discord.Message] = []
    for m in collected:
        mid = getattr(m, "id", None)
        if mid in seen:
            continue
        seen.add(mid)
        uniq.append(m)

    return uniq, truncated


def _package(
    bot: discord.Client,
    message: discord.Message,
    case: str,
    anchor: Optional[discord.Message],
    messages: List[discord.Message],
) -> MentionContextBlock:
    guild = message.guild
    items: List[PackagedItem] = []
    for i, m in enumerate(messages, start=1):
        author_name = getattr(m.author, "display_name", getattr(m.author, "name", f"User({getattr(m.author, 'id', 'unknown')})"))
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
    conv_id = (
        str(getattr(message.channel, "id", "")) if case == THREAD_CASE else str(getattr(anchor, "id", ""))
    )
    anc = None
    if anchor:
        anc = {
            "id": str(getattr(anchor, "id", "")),
            "author": getattr(anchor.author, "display_name", getattr(anchor.author, "name", "")),
            "created_at_iso": (anchor.created_at or _now_utc()).isoformat(),
            "jump_url": getattr(anchor, "jump_url", ""),
        }
    block = MentionContextBlock(
        source=("discord-thread" if case == THREAD_CASE else "discord-reply-chain"),
        conversation_id=conv_id,
        anchor=anc or {},
        count=len(items),
        items=items,
        joined_text=joined_text,
        truncated=False,  # caller may update
    )
    return block


async def maybe_build_mention_context(
    bot: discord.Client,
    message: discord.Message,
    cfg: Dict[str, Any],
) -> Optional[Tuple[str, MentionContextBlock]]:
    """
    Build mention-aware context when feature is enabled and message mentions the bot.
    Returns (joined_text, block) or None on fallback/no-op.
    """
    # Hard-enabled feature: always on regardless of env/config
    enabled = True
    if not enabled:
        return None

    # Only trigger on @mention
    try:
        mentioned = bot.user in (getattr(message, "mentions", None) or [])
    except Exception:
        mentioned = False
    if not mentioned:
        return None

    # Caps and limits
    try:
        max_msgs = int(cfg.get("MEM_MAX_MSGS", 40))
        max_chars = int(cfg.get("MEM_MAX_CHARS", 8000))
        max_age_min = int(cfg.get("MEM_MAX_AGE_MIN", 240))
        timeout_s = float(cfg.get("MEM_FETCH_TIMEOUT_S", 5))
        subsys = str(cfg.get("MEM_LOG_SUBSYS", "mem.ctx"))
    except Exception:
        max_msgs, max_chars, max_age_min, timeout_s, subsys = 40, 8000, 240, 5.0, "mem.ctx"

    t0 = time.perf_counter()
    case = classify_message_case(message)

    # Anchor resolution
    try:
        anchor = await resolve_anchor(message, case, timeout_s)
    except Exception as e:
        try:
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
        except Exception:
            pass
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
                max_msgs=max_msgs,
                max_chars=max_chars,
                max_age_min=max_age_min,
                timeout_s=timeout_s,
            )
        else:
            messages, truncated = await _collect_reply_chain(
                bot,
                message,
                anchor,
                max_msgs=max_msgs,
                max_chars=max_chars,
                max_age_min=max_age_min,
                timeout_s=timeout_s,
            )
    except asyncio.TimeoutError:
        try:
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
        except Exception:
            pass
        return None
    except Exception as e:
        try:
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
        except Exception:
            pass
        return None

    if not messages:
        return None

    block = _package(bot, message, case, anchor, messages)
    block.truncated = bool(block.truncated or (len(messages) >= max_msgs))

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
    except Exception:
        pass

    return block.joined_text, block
