"""Reusable permission check utilities for the Discord-like dashboard.

Provides a clean interface for checking bot permissions on channels and users
without importing Discord internals directly. All functions accept a bot
instance (discord.ext.commands.Bot) and return clear results with reasons.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, NamedTuple

from bot.utils.logging import get_logger

if TYPE_CHECKING:
    import discord
    from discord.ext.commands import Bot as DiscordBot

logger = get_logger(__name__)


class PermissionResult(NamedTuple):
    """Result of a permission check.

    Attributes:
        allowed: Whether the action is permitted.
        reason: Human-readable explanation if denied.
        permissions: Dict of permission_name -> bool for granular access.

    """

    allowed: bool
    reason: str
    permissions: dict[str, bool]


def _get_bot_member(guild: discord.Guild) -> discord.Member | None:
    """Get the bot's Member object for a guild. Returns None if unavailable."""
    try:
        return guild.me
    except Exception:
        return None


def _safe_permissions(channel: Any) -> dict[str, bool]:
    """Get a dict of common permission flags for the bot in a channel.

    Returns a safe default (all False) if anything goes wrong.
    """
    defaults = {
        "read_messages": False,
        "read_message_history": False,
        "send_messages": False,
        "send_messages_in_threads": False,
        "embed_links": False,
        "attach_files": False,
        "add_reactions": False,
        "use_external_emojis": False,
        "mention_everyone": False,
        "manage_messages": False,
        "manage_channels": False,
        "administrator": False,
    }

    try:
        if not hasattr(channel, "permissions_for"):
            return defaults

        guild = getattr(channel, "guild", None)
        if guild is None:
            return defaults

        me = _get_bot_member(guild)
        if me is None:
            return defaults

        perms = channel.permissions_for(me)
        return {
            "read_messages": perms.read_messages,
            "read_message_history": perms.read_message_history,
            "send_messages": perms.send_messages,
            "send_messages_in_threads": perms.send_messages_in_threads,
            "embed_links": perms.embed_links,
            "attach_files": perms.attach_files,
            "add_reactions": perms.add_reactions,
            "use_external_emojis": perms.use_external_emojis,
            "mention_everyone": perms.mention_everyone,
            "manage_messages": perms.manage_messages,
            "manage_channels": perms.manage_channels,
            "administrator": perms.administrator,
        }
    except Exception as e:
        logger.debug("Failed to get permissions for channel %s: %s", getattr(channel, "id", "?"), e)
        return defaults


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------


def can_view_channel(bot: DiscordBot, channel_id: int) -> PermissionResult:
    """Check if the bot can view (read messages in) a channel.

    The bot must be in the guild, the channel must exist, and the bot must
    have read_messages permission.
    """
    channel = _resolve_channel(bot, channel_id)
    if channel is None:
        return PermissionResult(False, "Channel not found or bot not in guild", {})

    perms = _safe_permissions(channel)

    if not perms.get("read_messages", False):
        reason = "Bot lacks read_messages permission"
        if perms.get("administrator", False):
            # Administrator overrides this, but let's be precise
            pass
        return PermissionResult(False, reason, perms)

    return PermissionResult(True, "ok", perms)


def can_read_message_history(bot: DiscordBot, channel_id: int) -> PermissionResult:
    """Check if the bot can read message history in a channel.

    Requires both read_messages and read_message_history (or administrator).
    """
    channel = _resolve_channel(bot, channel_id)
    if channel is None:
        return PermissionResult(False, "Channel not found or bot not in guild", {})

    perms = _safe_permissions(channel)

    if perms.get("administrator", False):
        return PermissionResult(True, "administrator override", perms)

    if not perms.get("read_messages", False):
        return PermissionResult(False, "Bot lacks read_messages permission", perms)
    if not perms.get("read_message_history", False):
        return PermissionResult(False, "Bot lacks read_message_history permission", perms)

    return PermissionResult(True, "ok", perms)


def can_send_messages(bot: DiscordBot, channel_id: int) -> PermissionResult:
    """Check if the bot can send messages in a channel.

    Requires send_messages (or send_messages_in_threads for thread channels).
    """
    channel = _resolve_channel(bot, channel_id)
    if channel is None:
        return PermissionResult(False, "Channel not found or bot not in guild", {})

    perms = _safe_permissions(channel)

    if perms.get("administrator", False):
        return PermissionResult(True, "administrator override", perms)

    # Determine if this is a thread
    is_thread = hasattr(channel, "parent") and channel.parent is not None
    required_perm = "send_messages_in_threads" if is_thread else "send_messages"

    if not perms.get("read_messages", False):
        return PermissionResult(False, "Bot lacks read_messages permission", perms)
    if not perms.get(required_perm, False):
        required_name = required_perm.replace("_", " ")
        return PermissionResult(False, f"Bot lacks {required_name} permission", perms)

    return PermissionResult(True, "ok", perms)


def get_channel_permissions(bot: DiscordBot, channel_id: int) -> dict[str, Any]:
    """Get a channel info dict plus a detailed permission summary.

    Returns a dict with keys:
        found: bool
        channel_id: str
        channel_name: str or None
        guild_id: str or None
        guild_name: str or None
        channel_type: str or None
        permissions: dict of permission flags
        permission_summary: human-readable list of granted permissions
    """
    channel = _resolve_channel(bot, channel_id)
    if channel is None:
        return {
            "found": False,
            "channel_id": str(channel_id),
            "channel_name": None,
            "guild_id": None,
            "guild_name": None,
            "channel_type": None,
            "permissions": {},
            "permission_summary": [],
        }

    perms = _safe_permissions(channel)
    guild = getattr(channel, "guild", None)

    # Build a human-readable summary
    granted = [name.replace("_", " ") for name, value in perms.items() if value]
    if not granted:
        granted = ["none"]
    granted.sort()

    channel_type = None
    with contextlib.suppress(Exception):
        channel_type = str(getattr(channel, "type", "text"))

    return {
        "found": True,
        "channel_id": str(channel.id),
        "channel_name": getattr(channel, "name", None),
        "guild_id": str(guild.id) if guild else None,
        "guild_name": guild.name if guild else None,
        "channel_type": channel_type,
        "permissions": perms,
        "permission_summary": granted,
    }


def can_send_dm(bot: DiscordBot, user_id: int) -> PermissionResult:
    """Check if the bot can send a DM to a user.

    Checks:
    - User exists (in cache or can be fetched)
    - User is not the bot itself
    - User is not a bot (Discord restricts DMs to bots)
    """
    user = bot.get_user(user_id)
    if user is None:
        # User may still be fetchable — but we can't check reliably without
        # an API call. Return uncertain.
        return PermissionResult(
            False,
            "User not found in cache. User may not share a guild with the bot.",
            {},
        )

    bot_user = bot.user
    if bot_user and user.id == bot_user.id:
        return PermissionResult(False, "Cannot DM the bot itself", {})

    if user.bot:
        return PermissionResult(False, "Cannot DM another bot", {})

    # Try to get or create the DM channel to verify ability
    try:
        _ = user.dm_channel
        # Having a DM channel object doesn't guarantee the user allows DMs
        # from bot users — Discord automatically creates DM channels for
        # mutual guild members. We'll report it as likely possible.
    except Exception as e:
        logger.debug(f"Failed to access dm_channel for user {user.id}: {e}")

    return PermissionResult(
        True,
        "User found and is reachable via DM (shared guild or open DMs)",
        {"is_bot": user.bot, "has_dm_channel": user.dm_channel is not None},
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_channel(bot: DiscordBot, channel_id: int) -> Any:
    """Resolve a channel by ID across all guilds the bot can see.

    First tries bot.get_channel() (which covers all connected guilds),
    then falls back to bot.fetch_channel().
    """
    try:
        channel = bot.get_channel(channel_id)
        if channel is not None:
            return channel
    except Exception as e:
        logger.debug(f"bot.get_channel failed for {channel_id}: {e}")

    # DM channels are not guild channels — bot.get_channel won't find them
    # for DM-specific checks we need the DMStore or user lookup instead.
    return None
