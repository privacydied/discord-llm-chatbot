"""Replies to allowlisted bots, with a loop guard.
[SFT][REH][CMV].

Bots are ignored by default. A bot whose user ID is in BOT_REPLY_ALLOWLIST
gets a reply only when it addresses this bot (@mention or reply to one of our
messages) -- never ambient replies or reacts. Two bots that ping each other
would otherwise loop forever, so each channel allows at most
BOT_REPLY_MAX_PER_WINDOW bot-triggered replies per BOT_REPLY_WINDOW_S.
"""

from __future__ import annotations

import collections
import time
from typing import Any

from bot.router_components.gating import is_reply_to_bot, mentions_bot
from bot.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_MAX_PER_WINDOW = 3
DEFAULT_WINDOW_S = 300


def parse_allowlist(raw: Any) -> frozenset[int]:
    ids = set()
    for part in str(raw or "").replace(";", ",").split(","):
        part = part.strip().strip("<@!>")
        if part.isdigit():
            ids.add(int(part))
    return frozenset(ids)


class BotReplyBudget:
    """Sliding-window count of bot-triggered replies per channel."""

    def __init__(self) -> None:
        self._events: dict[int, collections.deque[float]] = {}

    def try_consume(self, channel_id: int, limit: int, window_s: float, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        events = self._events.setdefault(channel_id, collections.deque())
        while events and now - events[0] >= window_s:
            events.popleft()
        if len(events) >= limit:
            return False
        events.append(now)
        return True


_budget = BotReplyBudget()


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def is_addressed_peer_bot(message: Any, cfg: dict[str, Any], own_id: int | None) -> bool:
    """Allowlisted other bot that @mentions or replies to us. No budget change."""
    author = getattr(message, "author", None)
    author_id = getattr(author, "id", None)
    if author_id is None or own_id is None or author_id == own_id:
        return False
    if author_id not in parse_allowlist(cfg.get("BOT_REPLY_ALLOWLIST")):
        return False
    return bool(mentions_bot(message, own_id) or is_reply_to_bot(message, own_id))


def accept_peer_bot_message(message: Any, cfg: dict[str, Any], own_id: int | None) -> bool:
    """True if a message from another bot should be answered. Consumes budget."""
    author_id = getattr(getattr(message, "author", None), "id", None)
    if not is_addressed_peer_bot(message, cfg, own_id):
        if author_id in parse_allowlist(cfg.get("BOT_REPLY_ALLOWLIST")):
            logger.info(f"peer_bot.skip reason=not_addressed author={author_id} msg_id={getattr(message, 'id', None)}")
        return False
    channel_id = getattr(getattr(message, "channel", None), "id", 0) or 0
    limit = _int(cfg.get("BOT_REPLY_MAX_PER_WINDOW"), DEFAULT_MAX_PER_WINDOW)
    window = _int(cfg.get("BOT_REPLY_WINDOW_S"), DEFAULT_WINDOW_S)
    if not _budget.try_consume(channel_id, limit, window):
        logger.info(f"peer_bot.skip reason=loop_guard author={author_id} channel={channel_id} limit={limit}/{window}s")
        return False
    logger.info(f"peer_bot.accept author={author_id} msg_id={getattr(message, 'id', None)}")
    return True
