"""Tool: read earlier messages from the current channel.
[CA][REH][IV][SFT][PA].

Reads via ``channel.history(before=message)``, which is exact -- it matches
what a human sees scrolling back. The SQLite archive is deliberately not used:
it excludes bot messages and soft-deleted rows by default, so its positional
indices drift from the real channel.

Message text retrieved here is OTHER USERS' input flowing back into the model,
so it is mention-sanitised and wrapped as untrusted content. [SFT]
"""

from __future__ import annotations

import re
from typing import Any

from bot.utils.logging import get_logger

from ..types import ToolContext, ToolResult, ToolSpec

logger = get_logger(__name__)

# Bounds. posts_ago is capped because Discord paginates history at 100/request,
# so deep lookbacks cost multiple round trips. [CMV][PA]
MAX_POSTS_AGO = 200
MAX_COUNT = 10
MAX_CHARS_PER_MESSAGE = 500

# Collapse pings so retrieved text cannot re-ping anyone via the model. [SFT]
_MENTION_RE = re.compile(r"<@!?(\d+)>|<@&(\d+)>|@everyone|@here")

PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "posts_ago": {
            "type": "integer",
            "description": ("How many messages back to start, counting from the message being replied to. 1 means the message immediately before it."),
            "minimum": 1,
            "maximum": MAX_POSTS_AGO,
        },
        "count": {
            "type": "integer",
            "description": (
                "How many consecutive messages to read. The block starts at posts_ago and extends FURTHER BACK in time. "
                "Example: posts_ago=3 with count=3 returns the messages 3, 4 and 5 posts ago. "
                "To cover a range such as 'between 3 and 5 posts ago', set posts_ago to the SMALLER number (3) "
                "and count to the size of the range (3)."
            ),
            "minimum": 1,
            "maximum": MAX_COUNT,
            "default": 1,
        },
    },
    "required": ["posts_ago"],
}

DESCRIPTION = (
    "Read earlier messages from this Discord channel by position. Use when the user refers to something said a specific number of messages ago "
    "(for example 'what did I say 20 posts ago', 'scroll back 5 messages'). Returns the author, timestamp and text of each message."
)


def _coerce_int(value: Any, *, default: int | None = None) -> int | None:
    """Accept the ints and numeric strings models actually emit. [IV]"""
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return default
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return default


def _sanitize(text: str) -> str:
    cleaned = _MENTION_RE.sub("[mention]", text or "")
    cleaned = cleaned.strip()
    if len(cleaned) > MAX_CHARS_PER_MESSAGE:
        cleaned = cleaned[:MAX_CHARS_PER_MESSAGE] + "…"
    return cleaned


def _author_name(msg: Any) -> str:
    author = getattr(msg, "author", None)
    return str(getattr(author, "display_name", None) or getattr(author, "name", None) or "unknown")


def _render(messages: list[Any], start_offset: int) -> str:
    lines: list[str] = []
    for index, msg in enumerate(messages):
        position = start_offset + index
        stamp = getattr(msg, "created_at", None)
        when = stamp.strftime("%Y-%m-%d %H:%M UTC") if stamp else "unknown time"
        body = _sanitize(getattr(msg, "content", "") or "")
        if not body:
            body = "[no text content — may be an attachment or embed]"
        lines.append(f"[{position} posts ago] {_author_name(msg)} — {when}\n{body}")
    return "\n\n".join(lines)


def _validate(arguments: dict[str, Any]) -> tuple[int, int] | str:
    """Return (posts_ago, count) or an error string. [IV]"""
    posts_ago = _coerce_int(arguments.get("posts_ago"))
    if posts_ago is None:
        return "posts_ago is required and must be an integer"
    if posts_ago < 1 or posts_ago > MAX_POSTS_AGO:
        return f"posts_ago must be between 1 and {MAX_POSTS_AGO}"

    # Distinguish "omitted" from an explicit bad value: `or 1` would silently
    # accept count=0 as 1 rather than reporting it. [IV]
    raw_count = arguments.get("count")
    count = 1 if raw_count is None else _coerce_int(raw_count)
    if count is None:
        return "count must be an integer"
    if count < 1 or count > MAX_COUNT:
        return f"count must be between 1 and {MAX_COUNT}"
    return posts_ago, count


async def read_channel_history(ctx: ToolContext, arguments: dict[str, Any]) -> ToolResult:
    """Fetch `count` messages starting `posts_ago` back. Never raises. [REH]"""
    validated = _validate(arguments)
    if isinstance(validated, str):
        return ToolResult.failure(validated)
    posts_ago, count = validated

    channel = ctx.channel
    if channel is None or not hasattr(channel, "history"):
        return ToolResult.failure("no channel available to read history from")

    limit = posts_ago + count - 1
    try:
        collected = [msg async for msg in channel.history(limit=limit, before=ctx.message)]
    except Exception as exc:  # [REH] permissions, rate limits, gateway issues
        name = type(exc).__name__
        if name == "Forbidden":
            return ToolResult.failure("missing permission to read message history in this channel")
        logger.warning("tool.history.failed error=%s", exc)
        return ToolResult.failure(f"could not read channel history ({name})")

    # history() is newest-first, so index 0 is 1 post ago.
    window = collected[posts_ago - 1 : posts_ago - 1 + count]
    if not window:
        return ToolResult.failure(f"channel history does not go back {posts_ago} messages (found {len(collected)})")

    logger.info("tool.history.ok posts_ago=%d count=%d returned=%d", posts_ago, count, len(window))

    from bot.url_safety import wrap_untrusted_content

    rendered = _render(window, posts_ago)
    return ToolResult.success(wrap_untrusted_content(rendered, source="discord-channel-history"))


SPEC = ToolSpec(
    name="read_channel_history",
    description=DESCRIPTION,
    parameters=PARAMETERS,
    handler=read_channel_history,
)
