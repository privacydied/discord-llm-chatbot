"""Tool: search this server's message archive.
[CA][SFT][IV][PA].

Answers "what did dre say about the knicks last week" from the SQLite server
archive (full-text indexed), not just the last N messages.

Privacy: the archive holds every channel the bot can read, including ones the
person asking cannot. Each result is kept only if the requester can view that
channel and read its history, so the tool never reveals more than they could
scroll to themselves. Results are mention-sanitised and wrapped as untrusted.
"""

from __future__ import annotations

from typing import Any

from bot.utils.logging import get_logger

from ..types import ToolContext, ToolResult, ToolSpec
from .history import _sanitize

logger = get_logger(__name__)

DEFAULT_RESULTS = 5
MAX_RESULTS = 8
# Over-fetch before the permission filter; the store caps one query at 10. [CMV]
FETCH_LIMIT = 10
MAX_QUERY_CHARS = 200
MAX_AUTHOR_CHARS = 64

DESCRIPTION = (
    "Search this Discord server's past messages by keywords, optionally only from one person. "
    "Use for questions about what someone said, when something was discussed, or finding an old link or take "
    "-- anything older than the recent scrollback. Returns dated messages with author and channel."
)

PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Keywords to find, e.g. 'knicks trade'. Not a full sentence."},
        "author": {"type": "string", "description": "Optional: name or username of the person who said it."},
        "max_results": {"type": "integer", "minimum": 1, "maximum": MAX_RESULTS, "default": DEFAULT_RESULTS},
    },
    "required": ["query"],
}


def _visible_to(guild: Any, member: Any, result: Any) -> bool:
    """True if ``member`` could read the channel or thread ``result`` came from."""
    channel_id = result.thread_id or result.channel_id
    try:
        channel = guild.get_channel_or_thread(int(channel_id))
    except (TypeError, ValueError):
        return False
    if channel is None:
        return False
    try:
        perms = channel.permissions_for(member)
    except (AttributeError, TypeError, ValueError):
        return False
    return bool(perms.view_channel and perms.read_message_history)


def _render(results: list[Any]) -> str:
    lines = []
    for r in results:
        date = (r.created_at or "")[:10]
        where = f"#{r.channel_name}" if r.channel_name else "a channel"
        lines.append(f"- [{date}] {_sanitize(r.author_name or 'someone')} in {where}: {_sanitize(r.snippet)}")
    return "\n".join(lines)


async def search_chat_history(ctx: ToolContext, arguments: dict[str, Any]) -> ToolResult:
    """Full-text search of the server archive, filtered to what the asker can see. Never raises."""
    message = ctx.message
    guild = getattr(message, "guild", None)
    member = getattr(message, "author", None)
    if guild is None or member is None:
        return ToolResult.failure("chat history search only works inside a server")
    query = " ".join(str(arguments.get("query") or "").split())[:MAX_QUERY_CHARS]
    if not query:
        return ToolResult.failure("query is required")
    author = " ".join(str(arguments.get("author") or "").split())[:MAX_AUTHOR_CHARS] or None
    try:
        limit = max(1, min(MAX_RESULTS, int(arguments.get("max_results") or DEFAULT_RESULTS)))
    except (TypeError, ValueError):
        limit = DEFAULT_RESULTS

    from bot.server_archive import search_archive

    found = await search_archive(query, guild_id=str(guild.id), author_name=author, limit=FETCH_LIMIT)
    own_id = str(getattr(message, "id", ""))
    visible = [r for r in found if r.message_id != own_id and _visible_to(guild, member, r)][:limit]
    logger.info("tool.search_chat_history query=%s author=%s found=%d visible=%d", query[:60], author, len(found), len(visible))
    if not visible:
        who = f" from {author}" if author else ""
        return ToolResult.success(f'No messages{who} matching "{query}". Try fewer or different keywords.')

    from bot.url_safety import wrap_untrusted_content

    return ToolResult.success(wrap_untrusted_content(_render(visible), source="server chat archive"))


SPEC = ToolSpec(
    name="search_chat_history",
    description=DESCRIPTION,
    parameters=PARAMETERS,
    handler=search_chat_history,
    timeout_s=15.0,
)
