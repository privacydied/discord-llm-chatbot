"""Centralized admin permission checks for the Discord bot.

Provides a single source of truth for admin/gating logic that supports:
- Prefix commands (ctx.author) with normal reply/DM for denials
- Slash commands (interaction.user) with ephemeral=True for denials
- Bot owner checks
- Configured guild admin role checks
- Configured admin user ID lists

Usage:
    # Prefix command decorator (replies normally on denial):
    from bot.core.permissions import admin_only_prefix
    @commands.command()
    @admin_only_prefix()
    async def my_admin_cmd(ctx): ...

    # Slash command decorator (ephemeral denial):
    from bot.core.permissions import admin_only_slash
    @bot.tree.command(...)
    @admin_only_slash()
    async def my_admin_slash(interaction): ...

    # Programmatic check (for message-based flows):
    from bot.core.permissions import check_admin_async
    if not await check_admin_async(message.author, bot):
        return False
"""

from __future__ import annotations

import functools
from typing import Optional, Set

import discord
from discord.ext import commands

from bot.config import load_config
from bot.exceptions import PermissionDeniedError
from bot.utils.logging import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------ #
#  Core admin resolution                                                #
# ------------------------------------------------------------------ #


def _get_configured_admin_ids(config: Optional[dict] = None) -> Set[int]:
    """Collect all configured admin/owner user IDs from config."""
    if config is None:
        config = load_config()

    ids: set[int] = set()

    # OWNER_IDS
    owner_ids = config.get("OWNER_IDS") or config.get("owner_ids") or []
    if isinstance(owner_ids, str):
        owner_ids = owner_ids.split(",")
    for oid in owner_ids:
        oid_str = str(oid).strip()
        if oid_str:
            try:
                ids.add(int(oid_str))
            except ValueError:
                pass

    # ALERT_ADMIN_USER_IDS
    alert_admin = config.get("ALERT_ADMIN_USER_IDS") or config.get(
        "alert_admin_user_ids", ""
    )
    if isinstance(alert_admin, str):
        for aid in alert_admin.split(","):
            aid_str = aid.strip()
            if aid_str:
                try:
                    ids.add(int(aid_str))
                except ValueError:
                    pass

    return ids


async def _resolve_bot_owner_ids(bot) -> Set[int]:
    """Resolve bot owner IDs from discord client attributes."""
    ids: set[int] = set()

    # owner_ids (application.team owners)
    if hasattr(bot, "owner_ids") and bot.owner_ids:
        ids.update(bot.owner_ids)

    # owner_id (non-team app)
    if hasattr(bot, "owner_id") and bot.owner_id:
        ids.add(bot.owner_id)

    # config fallback
    try:
        config = load_config()
        for oid in _get_configured_admin_ids(config):
            ids.add(oid)
    except Exception:
        pass

    return ids


async def is_admin_user(
    user: discord.abc.User | discord.Member,
    bot,
    *,
    require_guild: bool = False,
) -> bool:
    """Check whether *user* is an admin.

    Supports both **users** (DM context, no guild_permissions) and
    **members** (guild context, has guild_permissions).

    Admin = any of:
    1. User is a bot owner (owner_id / owner_ids)
    2. User ID is in configured admin lists (OWNER_IDS, ALERT_ADMIN_USER_IDS)
    3. User has guild administrator permission (if Member, not require_guild)

    Args:
        user: discord.User or discord.Member
        bot: discord.Bot instance
        require_guild: If True, guild administrator permission is required
            (configured IDs and owners alone are insufficient).
    """
    user_id = user.id

    # Check bot owners
    owner_ids = await _resolve_bot_owner_ids(bot)
    if user_id in owner_ids:
        return True

    # Check configured admin IDs
    try:
        config_ids = _get_configured_admin_ids()
        if user_id in config_ids:
            return True
    except Exception:
        pass

    # Guild administrator check (Members only)
    if isinstance(user, discord.Member) and user.guild_permissions.administrator:
        return True

    if require_guild:
        # User passed owner/config check but not guild admin
        return False

    return False


# ------------------------------------------------------------------ #
#  Prefix-command decorator (normal reply on denial)                    #
# ------------------------------------------------------------------ #


def admin_only_prefix(
    *,
    message: str = "You do not have permission to use this command.",
) -> callable:
    """Decorator for prefix commands.

    Denial is sent as a normal channel reply (no ephemeral).
    """

    def decorator(fn):
        @commands.check
        @functools.wraps(fn)
        async def wrapper(ctx: commands.Context, *args, **kwargs):
            if not await is_admin_user(ctx.author, ctx.bot):
                await ctx.send(message)
                raise commands.CheckFailure(message)
            return await fn(ctx, *args, **kwargs)

        return wrapper

    return decorator


# ------------------------------------------------------------------ #
#  Slash-command decorator (ephemeral denial)                            #
# ------------------------------------------------------------------ #


def admin_only_slash(
    *,
    message: str = "You do not have permission to use this command.",
) -> callable:
    """Decorator for slash commands / interactions.

    Denial is sent as an ephemeral reply.
    """

    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(interaction: discord.Interaction, *args, **kwargs):
            if not await is_admin_user(interaction.user, interaction.client):
                await interaction.response.send_message(message, ephemeral=True)
                raise PermissionDeniedError(message)
            return await fn(interaction, *args, **kwargs)

        return wrapper

    return decorator


# ------------------------------------------------------------------ #
#  Low-level helpers                                                    #
# ------------------------------------------------------------------ #


async def check_admin_async(
    author: discord.abc.User | discord.Member,
    bot,
    *,
    reply_channel=None,
    ephemeral: bool = False,
) -> bool:
    """Inline admin check for use inside command bodies.

    If *reply_channel* is given and the check fails, a denial message
    is sent.  For :class:`discord.Interaction` use ephemeral=True.
    For prefix :class:`commands.Context`, use ephemeral=False (default).

    Returns True if authorized, False otherwise.
    """
    if await is_admin_user(author, bot):
        return True

    if reply_channel is not None:
        if ephemeral and isinstance(reply_channel, discord.Interaction):
            try:
                await reply_channel.response.send_message(
                    "Permission denied.", ephemeral=True
                )
            except discord.InteractionResponded:
                pass
        else:
            await reply_channel.send("Permission denied.")

    return False
