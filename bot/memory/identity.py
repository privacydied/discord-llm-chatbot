"""Resolve the subject user for memory recall.

This module ensures that "me / myself" queries are always resolved to
message.author.id -- never to a mentioned user, replied-to author, previous
speaker, or any other heuristic.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import discord


# Natural-language patterns that indicate "about me / about myself" recall.
_SELF_RECALL_PATTERNS = [
    r"\b(tell|say)\b\s+me\b.*\b(about\s+myself|about\s+me)\b",
    r"\b(tell|say)\b\s+me\b\s+something\b.*\b(about\s+myself|about\s+me)\b",
    r"\bshow\s+me\b.*\babout\s+myself\b",
    r"\bwhat\s+(do|can)\s+(you|ya)\s+(know|remember|have)\b.*\babout\s+me\b",
    r"\bwhat\s+about\s+me\b",
    r"\bwhat\s+do\s+you\s+know\s+about\s+myself\b",
    r"\bwhat\s+are\s+my\b.*\bmemor",
    r"\bwhat\s+do\s+you\s+remember\s+about\s+me\b",
    r"\bwhat\s+do\s+you\s+know\s+about\s+me\b",
    r"\b(remind)\s+me\s+(who|what)\b",
]

_COMPILED_SELF_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _SELF_RECALL_PATTERNS]


def is_self_recall_intent(content: str) -> bool:
    """Return True when the message is asking about the author's own memories."""
    content = (content or "").strip()
    if not content:
        return False
    return any(p.search(content) for p in _COMPILED_SELF_PATTERNS)


def resolve_memory_subject_user_id(
    message: discord.Message,
) -> str:
    """Resolve the user whose memories should be consulted.

    For all normal chat messages, the subject is always message.author.id.
    This function exists to prevent accidental overrides elsewhere in the
    pipeline (e.g. replied-to author, mention, last speaker).
    """
    return str(message.author.id)
