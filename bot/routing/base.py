"""Base routing infrastructure for the Discord bot.

Defines the shared types and contracts that all route handlers implement.
Extraction follows the compatibility-shell protocol:

1. Identify one router concern
2. Move logic into one handler module
3. Keep bot/router.py delegating to the new handler
4. Preserve imports outside bot/router.py
5. Add handler-level tests
6. Run full tests
7. Merge
8. Repeat
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import discord


# ------------------------------------------------------------------ #
#  RouteContext — carries all resolved values a handler might need
# ------------------------------------------------------------------ #


@dataclass(frozen=True)
class RouteContext:
    """Context passed to every route handler.

    Attributes
    ----------
    message : Optional[discord.Message]
        The raw Discord message that triggered this route.  May be empty
        for non-message events (e.g. scheduled tasks).
    author_id : int
        Discord user ID of the message author (0 when unknown).
    guild_id : Optional[int]
        Guild ID where the message originated, or None for DMs.
    channel_id : Optional[int]
        Text channel ID.
    source_type : str
        Origin label from the ``InputItem`` (``"url"``, ``"attachment"``,
        ``"text"``, etc.).
    payload : object
        The raw payload passed to the handler (URL string, ``Attachment``,
        etc.).
    correlation_id : str
        Unique identifier for this request, useful for logging and tracing.
    """

    message: Optional[discord.Message] = None
    author_id: int = 0
    guild_id: Optional[int] = None
    channel_id: Optional[int] = None
    source_type: str = "text"
    payload: object = None
    correlation_id: str = ""

    @classmethod
    def from_discord_message(
        cls,
        message: discord.Message,
        *,
        source_type: str = "text",
        payload: object = None,
        correlation_id: str = "",
    ) -> "RouteContext":
        """Build a RouteContext from a Discord Message."""
        return cls(
            message=message,
            author_id=message.author.id,
            guild_id=getattr(message.guild, "id", None),
            channel_id=message.channel.id,
            source_type=source_type,
            payload=payload or message,
            correlation_id=correlation_id or str(message.id),
        )


# ------------------------------------------------------------------ #
#  RouteResult — what a handler returns to the router
# ------------------------------------------------------------------ #


@dataclass
class RouteResult:
    """Result returned by a route handler.

    Attributes
    ----------
    text : Optional[str]
        Plain text response to send.
    embed : Optional[discord.Embed]
        Embed to include in the response.
    files : list[discord.File]
        Files to attach.
    handled : bool
        False means the handler declined; the router should try the next one.
    """

    text: Optional[str] = None
    embed: Optional[discord.Embed] = None
    files: list[discord.File] = field(default_factory=list)
    handled: bool = True

    @classmethod
    def text_only(cls, text: str) -> "RouteResult":
        return cls(text=text)

    @classmethod
    def declined(cls) -> "RouteResult":
        return cls(handled=False)


# ------------------------------------------------------------------ #
#  RouteHandler — handler protocol
# ------------------------------------------------------------------ #


class RouteHandler:
    """Base class for route handlers.

    Subclasses implement two methods:

    * ``can_handle(ctx) -> bool`` — quick synchronous check
    * ``async handle(ctx) -> RouteResult`` — actual processing
    """

    async def can_handle(self, ctx: RouteContext) -> bool:
        """Return True if this handler can process *ctx*.

        Default implementation always returns True (catch-all).
        """
        return True

    async def handle(self, ctx: RouteContext) -> RouteResult:  # pragma: no cover
        """Process *ctx* and return the result.

        Must be overridden by subclasses.
        """
        raise NotImplementedError
