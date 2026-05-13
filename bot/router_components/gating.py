"""Gating helper utilities extracted from Router."""

from __future__ import annotations

import re
from typing import Any


def mentions_bot(message: Any, bot_user_id: int | None) -> bool:
    """Return True if the message explicitly mentions the bot user id."""
    if bot_user_id is None:
        return False
    try:
        mentions = getattr(message, "mentions", None) or []
        for user in mentions:
            if getattr(user, "id", None) == bot_user_id:
                return True
    except Exception:
        return False
    return False


def is_reply_to_bot(message: Any, bot_user_id: int | None) -> bool:
    """Return True if the message is a reply to a bot-authored message."""
    if bot_user_id is None:
        return False
    try:
        reference = getattr(message, "reference", None)
        if not reference or not getattr(reference, "message_id", None):
            return False
        ref_msg = getattr(reference, "resolved", None) or getattr(reference, "cached_message", None)
        if not ref_msg:
            return False
        author_id = getattr(getattr(ref_msg, "author", None), "id", None)
        return author_id == bot_user_id
    except Exception:
        return False


def strip_leading_bot_mention(content: str | None, bot_user_id: int | None) -> str:
    """Strip `<@id>` / `<@!id>` prefix if present."""
    text = (content or "").strip()
    if not text or bot_user_id is None:
        return text
    pattern = rf"^<@!?{bot_user_id}>\s*"
    try:
        return re.sub(pattern, "", text).strip()
    except Exception:
        return text
