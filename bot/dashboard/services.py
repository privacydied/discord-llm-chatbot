"""Service layer: narrow interfaces between dashboard and bot internals.

Provides:
- Summary collection with coalescing and short TTL cache
- Guild inventory
- DM sending with permission checks, rate limits, audit logging
- Guild message sending with permission verification
- Reply to guild messages with MessageReference support
- Message store archiving
- Rate limiter for send actions
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import defaultdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import discord

from bot.utils.logging import get_logger

from .audit_store import (
    EVENT_DASHBOARD_REPLY_FAILURE,
    EVENT_DASHBOARD_REPLY_REQUESTED,
    EVENT_DASHBOARD_REPLY_SUCCESS,
    EVENT_DASHBOARD_SEND_FAILURE,
    EVENT_DASHBOARD_SEND_REQUESTED,
    EVENT_DASHBOARD_SEND_SUCCESS,
)

if TYPE_CHECKING:
    from discord.ext.commands import Bot as DiscordBot

    from .audit_store import AuditStore
    from .backfill import BackfillService
    from .config import DashboardConfig
    from .dm_store import DMStore
    from .message_store import MessageStore

logger = get_logger(__name__)


class _RateLimiter:
    """Token-bucket rate limiter per (user_id, target) pair."""

    def __init__(self, sends_per_minute: int) -> None:
        self._sends_per_minute = sends_per_minute
        self._windows: dict[str, list[float]] = defaultdict(list)
        self._lock = asyncio.Lock()

    def _bucket_key(self, actor_id: int | None, target_id: int | None) -> str:
        return f"{actor_id}:{target_id}"

    async def check_and_consume(self, actor_id: int | None, target_id: int | None) -> tuple[bool, str]:
        """Check rate limit and consume a token if available. Returns (allowed, reason)."""
        key = self._bucket_key(actor_id, target_id)
        async with self._lock:
            now = time.monotonic()
            window = 60.0
            # Remove expired entries
            self._windows[key] = [t for t in self._windows[key] if now - t < window]

            if len(self._windows[key]) >= self._sends_per_minute:
                oldest = self._windows[key][0]
                retry_after = window - (now - oldest)
                return False, f"Rate limited: {self._sends_per_minute} sends/min. Retry after {retry_after:.0f}s"

            self._windows[key].append(now)
            return True, "ok"


class DashboardServices:
    """Narrow service interface for dashboard operations."""

    def __init__(
        self,
        bot: DiscordBot | None,
        config: DashboardConfig,
        audit_store: AuditStore,
        dm_store: DMStore,
        message_store: MessageStore | None = None,
        backfill_service: BackfillService | None = None,
    ) -> None:
        self._bot = bot
        self._config = config
        self._audit_store = audit_store
        self._dm_store = dm_store
        self._message_store = message_store
        self._backfill_service = backfill_service
        self._rate_limiter = _RateLimiter(config.rate_limit_sends_per_minute)
        self._summary_cache: dict[str, Any] | None = None
        self._summary_cache_time: float = 0
        self._summary_lock = asyncio.Lock()

    @property
    def bot(self) -> DiscordBot | None:
        return self._bot

    @property
    def message_store(self) -> MessageStore | None:
        return self._message_store

    @property
    def backfill_service(self) -> BackfillService | None:
        return self._backfill_service

    def get_message_store(self) -> MessageStore | None:
        """Get the message store instance."""
        return self._message_store

    async def get_summary(self) -> dict[str, Any]:
        """Get bot summary with short TTL cache and request coalescing."""
        now = time.monotonic()
        if self._summary_cache and (now - self._summary_cache_time) < self._config.summary_ttl_seconds:
            return self._summary_cache

        async with self._summary_lock:
            # Double-check after acquiring lock
            if self._summary_cache and (now - self._summary_cache_time) < self._config.summary_ttl_seconds:
                return self._summary_cache

            summary = await self._collect_summary()
            self._summary_cache = summary
            self._summary_cache_time = now
            return summary

    async def _collect_summary(self) -> dict[str, Any]:
        """Collect summary from bot internals. Best-effort — never raises."""
        bot = self._bot
        if bot is None:
            return {"status": "not_ready", "error": "Bot not connected"}

        try:
            bot_user = bot.user
            uptime = _uptime(bot)
            guild_count = len(bot.guilds) if bot.guilds else 0

            # Estimate total visible users
            total_users = 0
            for g in bot.guilds:
                with contextlib.suppress(Exception):
                    total_users += g.member_count or 0

            # Cog count
            cog_count = len(bot.cogs) if bot.cogs else 0

            # Latency
            latency = round(bot.latency * 1000, 1) if bot.latency else 0

            # Channel count
            channel_count = 0
            for g in bot.guilds:
                with contextlib.suppress(Exception):
                    channel_count += len(list(g.channels)) if g.channels else 0

            # Audit event count
            try:
                audit_result = await self._audit_store.query(page=1, page_size=1)
                audit_count = audit_result.get("total", 0)
            except (AttributeError, TypeError, ValueError, RuntimeError):
                audit_count = 0

            # Message store count
            archived_count = 0
            try:
                if self._message_store:
                    result = await self._message_store.search_messages(query="", page=1, page_size=1)
                    archived_count = result.get("total", 0)
            except (AttributeError, TypeError, ValueError, RuntimeError) as e:
                logger.debug(f"Failed to get archived message count: {e}")

            # DM count
            dm_count = 0
            try:
                if self._dm_store:
                    dm_result = await self._dm_store.list_threads(page=1, page_size=1)
                    dm_count = dm_result.get("total", 0)
            except (AttributeError, TypeError, ValueError, RuntimeError) as e:
                logger.debug(f"Failed to get DM count: {e}")

            # Feature flags
            feature_flags = {
                "dm_archive_enabled": self._config.dm_archive_enabled,
                "guild_archive_enabled": self._config.guild_archive_enabled,
                "show_message_previews": self._config.show_message_previews,
                "backfill_enabled": self._config.backfill_enabled,
                "redact_secrets": self._config.redact_secrets,
            }

            # Cog names
            cogs = list(bot.cogs.keys()) if bot.cogs else []

            # Bot avatar URL
            avatar_url = None
            if bot_user and bot_user.avatar:
                avatar_url = str(bot_user.avatar.url)

            return {
                "status": "ready",
                "bot_username": bot_user.display_name if bot_user else "unknown",
                "bot_id": str(bot_user.id) if bot_user else None,
                "bot_avatar_url": avatar_url,
                "uptime_seconds": uptime,
                "uptime_human": _format_uptime(uptime),
                "guild_count": guild_count,
                "channel_count": channel_count,
                "dm_count": dm_count,
                "archived_message_count": archived_count,
                "audit_event_count": audit_count,
                "total_users_estimate": total_users,
                "cog_count": cog_count,
                "cogs": cogs,
                "latency_ms": latency,
                "feature_flags": feature_flags,
                "loaded_at": _iso_now(),
            }
        except (AttributeError, TypeError, ValueError, RuntimeError) as e:
            logger.warning("Dashboard summary collection failed: %s", e)
            return {"status": "error", "error": str(e)}

    async def get_guilds(
        self,
        page: int = 1,
        page_size: int = 50,
        max_page_size: int = 200,
        search: str | None = None,
    ) -> dict[str, Any]:
        """Get guild inventory from bot cache."""
        bot = self._bot
        if bot is None:
            return {"guilds": [], "total": 0, "page": page, "page_size": page_size, "total_pages": 0}

        all_guilds = list(bot.guilds) if bot.guilds else []

        # Filter by search
        if search:
            search_lower = search.lower()
            all_guilds = [g for g in all_guilds if search_lower in (g.name or "").lower() or search_lower in str(g.id)]

        total = len(all_guilds)
        page = max(1, page)
        page_size = min(max(1, page_size), max_page_size)
        offset = (page - 1) * page_size
        slice_guilds = all_guilds[offset : offset + page_size]

        guilds = []
        for g in slice_guilds:
            try:
                channel_count = len(list(g.channels)) if g.channels else 0
                text_channel_count = len(list(g.text_channels)) if g.text_channels else 0
                voice_channel_count = len(list(g.voice_channels)) if g.voice_channels else 0
                owner_id = str(g.owner_id) if g.owner_id else None
                member_count = g.member_count
                joined_at = g.me.joined_at.strftime("%Y-%m-%dT%H:%M:%SZ") if g.me and g.me.joined_at else None

                # Bot permissions in a representative channel
                perm_summary = "unknown"
                if g.text_channels:
                    try:
                        ch = next(iter(g.text_channels))
                        perms = ch.permissions_for(g.me)
                        features = []
                        if perms.send_messages:
                            features.append("send")
                        if perms.read_messages:
                            features.append("read")
                        if perms.embed_links:
                            features.append("embed")
                        if perms.attach_files:
                            features.append("attach")
                        if perms.administrator:
                            features.append("admin")
                        perm_summary = ",".join(features) if features else "none"
                    except (AttributeError, TypeError, ValueError, discord.Forbidden) as e:
                        logger.debug(f"Failed to get permissions for guild {g.id}: {e}")

                guilds.append(
                    {
                        "id": str(g.id),
                        "name": g.name,
                        "owner_id": owner_id,
                        "member_count": member_count,
                        "channel_count": channel_count,
                        "text_channel_count": text_channel_count,
                        "voice_channel_count": voice_channel_count,
                        "joined_at": joined_at,
                        "permissions": perm_summary,
                        "features": list(g.features) if g.features else [],
                        "icon_url": str(g.icon.url) if g.icon else None,
                        "banner_url": str(g.banner.url) if g.banner else None,
                    },
                )
            except (AttributeError, TypeError, ValueError, discord.HTTPException) as e:
                logger.warning("Failed to collect guild info for %s: %s", g.id, e)

        return {
            "guilds": guilds,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": max(1, (total + page_size - 1) // page_size),
        }

    async def send_dm(
        self,
        target_user_id: int,
        content: str,
        actor_id: int | None = None,
        source_ip: str | None = None,
        user_agent: str | None = None,
    ) -> dict[str, Any]:
        """Send a DM as the bot. Owner-only, rate-limited, audited."""
        # Audit: requested
        await self._audit_store.record(
            event_type=EVENT_DASHBOARD_SEND_REQUESTED,
            result="pending",
            actor_user_id=actor_id,
            actor_source_ip=source_ip,
            actor_user_agent=user_agent,
            target_user_id=target_user_id,
            content_preview=content[:200],
        )

        # Rate limit check
        allowed, reason = await self._rate_limiter.check_and_consume(actor_id, target_user_id)
        if not allowed:
            await self._audit_store.record(
                event_type=EVENT_DASHBOARD_SEND_FAILURE,
                result="rate_limited",
                actor_user_id=actor_id,
                actor_source_ip=source_ip,
                actor_user_agent=user_agent,
                target_user_id=target_user_id,
                content_preview=content[:100],
                metadata={"reason": reason},
            )
            return {"success": False, "error": reason, "status": "rate_limited"}

        # Validate content length
        if len(content) > self._config.max_message_chars:
            return {
                "success": False,
                "error": f"Content exceeds {self._config.max_message_chars} characters",
                "status": "too_long",
            }

        bot = self._bot
        if bot is None:
            return {"success": False, "error": "Bot not ready", "status": "not_ready"}

        try:
            # Resolve user
            user = bot.get_user(target_user_id)
            if user is None:
                try:
                    user = await asyncio.wait_for(bot.fetch_user(target_user_id), timeout=10.0)
                except (discord.NotFound, discord.HTTPException, discord.Forbidden, asyncio.TimeoutError) as e:
                    error_msg = f"User not found: {e}"
                    await self._audit_store.record(
                        event_type=EVENT_DASHBOARD_SEND_FAILURE,
                        result="failed",
                        actor_user_id=actor_id,
                        actor_source_ip=source_ip,
                        actor_user_agent=user_agent,
                        target_user_id=target_user_id,
                        error_code="user_not_found",
                        content_preview=content[:200],
                    )
                    return {"success": False, "error": error_msg, "status": "user_not_found"}

            # Send DM
            try:
                msg = await asyncio.wait_for(user.send(content), timeout=30.0)

                # Archive if DM archive is enabled
                if self._config.dm_archive_enabled:
                    await self._dm_store.upsert_user(
                        user_id=user.id,
                        username=user.name,
                        global_name=user.global_name,
                        display_name=user.display_name,
                        is_bot=user.bot,
                    )
                    await self._dm_store.add_message(
                        message_id=msg.id,
                        channel_id=msg.channel.id,
                        author_id=bot.user.id if bot.user else 0,
                        content=msg.content,
                        clean_content=msg.clean_content,
                        is_bot_author=True,
                        jump_url=msg.jump_url,
                    )

                # Also archive in unified MessageStore if available
                if self._message_store:
                    try:
                        await self._message_store.insert_message(
                            discord_message_id=msg.id,
                            channel_id=msg.channel.id,
                            content=msg.content or "",
                            author_id=bot.user.id if bot.user else 0,
                            author_username=bot.user.name if bot.user else "bot",
                            author_display_name=bot.user.display_name if bot.user else "Bot",
                            author_is_bot=True,
                            is_own_bot=True,
                            direction="outbound",
                            channel_type="private",
                            reply_to_message_id=msg.reference.message_id if msg.reference and msg.reference.message_id else None,
                            metadata={"jump_url": msg.jump_url} if msg.jump_url else None,
                        )
                    except (AttributeError, TypeError, ValueError, discord.HTTPException) as e:
                        logger.debug("Failed to archive sent DM in MessageStore: %s", e)

                await self._audit_store.record(
                    event_type=EVENT_DASHBOARD_SEND_SUCCESS,
                    result="success",
                    actor_user_id=actor_id,
                    actor_source_ip=source_ip,
                    actor_user_agent=user_agent,
                    target_user_id=target_user_id,
                    message_id=msg.id,
                    content_preview=content[:200],
                    metadata={"message_id": str(msg.id)},
                )
                return {"success": True, "status": "sent", "message_id": str(msg.id)}

            except discord.Forbidden:
                await self._audit_store.record(
                    event_type=EVENT_DASHBOARD_SEND_FAILURE,
                    result="failed",
                    actor_user_id=actor_id,
                    actor_source_ip=source_ip,
                    actor_user_agent=user_agent,
                    target_user_id=target_user_id,
                    error_code="forbidden",
                    content_preview=content[:200],
                )
                return {"success": False, "error": "Cannot DM this user (blocked or privacy settings)", "status": "forbidden"}

            except (discord.HTTPException, discord.NotFound, discord.Forbidden, AttributeError, TypeError, ValueError) as e:
                await self._audit_store.record(
                    event_type=EVENT_DASHBOARD_SEND_FAILURE,
                    result="failed",
                    actor_user_id=actor_id,
                    actor_source_ip=source_ip,
                    actor_user_agent=user_agent,
                    target_user_id=target_user_id,
                    error_code="send_failed",
                    content_preview=content[:200],
                    metadata={"error": str(e)},
                )
                return {"success": False, "error": f"Failed to send: {e}", "status": "send_failed"}

        except (AttributeError, TypeError, ValueError, RuntimeError) as e:
            return {"success": False, "error": f"Unexpected error: {e}", "status": "error"}

    async def send_guild_message(
        self,
        guild_id: int,
        channel_id: int,
        content: str,
        actor_id: int | None = None,
        source_ip: str | None = None,
        user_agent: str | None = None,
    ) -> dict[str, Any]:
        """Send a message to a guild channel where bot has permission."""
        # Audit: requested
        await self._audit_store.record(
            event_type=EVENT_DASHBOARD_SEND_REQUESTED,
            result="pending",
            actor_user_id=actor_id,
            actor_source_ip=source_ip,
            actor_user_agent=user_agent,
            target_guild_id=guild_id,
            target_channel_id=channel_id,
            content_preview=content[:200],
        )

        allowed, reason = await self._rate_limiter.check_and_consume(actor_id, channel_id)
        if not allowed:
            await self._audit_store.record(
                event_type=EVENT_DASHBOARD_SEND_FAILURE,
                result="rate_limited",
                actor_user_id=actor_id,
                actor_source_ip=source_ip,
                actor_user_agent=user_agent,
                target_guild_id=guild_id,
                target_channel_id=channel_id,
                content_preview=content[:100],
                metadata={"reason": reason},
            )
            return {"success": False, "error": reason, "status": "rate_limited"}

        if len(content) > self._config.max_message_chars:
            return {
                "success": False,
                "error": f"Content exceeds {self._config.max_message_chars} characters",
                "status": "too_long",
            }

        bot = self._bot
        if bot is None:
            return {"success": False, "error": "Bot not ready", "status": "not_ready"}

        try:
            guild = bot.get_guild(guild_id)
            if guild is None:
                return {"success": False, "error": "Guild not found", "status": "guild_not_found"}

            channel = guild.get_channel(channel_id)
            if channel is None:
                return {"success": False, "error": "Channel not found", "status": "channel_not_found"}

            # Permission check
            if not hasattr(channel, "permissions_for"):
                return {"success": False, "error": "Not a text channel", "status": "invalid_channel"}

            try:
                perms = channel.permissions_for(guild.me)
            except (AttributeError, TypeError, discord.Forbidden):
                return {"success": False, "error": "Cannot check permissions", "status": "perm_check_failed"}

            if not perms.send_messages:
                return {"success": False, "error": "Bot lacks send_messages permission", "status": "permission_denied"}

            if not perms.read_message_history:
                return {"success": False, "error": "Bot lacks read_message_history permission", "status": "permission_denied"}

            try:
                msg = await asyncio.wait_for(channel.send(content), timeout=30.0)

                # Archive in MessageStore if available
                if self._message_store and self._config.guild_archive_enabled:
                    try:
                        bot_user = bot.user
                        channel_name = getattr(channel, "name", None)
                        await self._message_store.insert_message(
                            discord_message_id=msg.id,
                            channel_id=channel.id,
                            guild_id=guild.id,
                            content=msg.content or "",
                            channel_name=channel_name,
                            channel_type=str(getattr(channel, "type", "text")),
                            author_id=bot_user.id if bot_user else 0,
                            author_username=bot_user.name if bot_user else "bot",
                            author_display_name=bot_user.display_name if bot_user else "Bot",
                            author_is_bot=True,
                            is_own_bot=True,
                            direction="outbound",
                            metadata={"jump_url": msg.jump_url} if msg.jump_url else None,
                        )
                    except (AttributeError, TypeError, ValueError, discord.HTTPException) as e:
                        logger.debug("Failed to archive guild message in MessageStore: %s", e)

                await self._audit_store.record(
                    event_type=EVENT_DASHBOARD_SEND_SUCCESS,
                    result="success",
                    actor_user_id=actor_id,
                    actor_source_ip=source_ip,
                    actor_user_agent=user_agent,
                    target_guild_id=guild_id,
                    target_channel_id=channel_id,
                    message_id=msg.id,
                    content_preview=content[:200],
                    metadata={"message_id": str(msg.id)},
                )
                return {"success": True, "status": "sent", "message_id": str(msg.id)}
            except discord.Forbidden:
                await self._audit_store.record(
                    event_type=EVENT_DASHBOARD_SEND_FAILURE,
                    result="failed",
                    actor_user_id=actor_id,
                    actor_source_ip=source_ip,
                    actor_user_agent=user_agent,
                    target_guild_id=guild_id,
                    target_channel_id=channel_id,
                    error_code="forbidden",
                    content_preview=content[:200],
                )
                return {"success": False, "error": "Bot lacks permission to send", "status": "forbidden"}
            except (discord.HTTPException, discord.NotFound, discord.Forbidden, AttributeError, TypeError, ValueError) as e:
                await self._audit_store.record(
                    event_type=EVENT_DASHBOARD_SEND_FAILURE,
                    result="failed",
                    actor_user_id=actor_id,
                    actor_source_ip=source_ip,
                    actor_user_agent=user_agent,
                    target_guild_id=guild_id,
                    target_channel_id=channel_id,
                    error_code="send_failed",
                    content_preview=content[:200],
                    metadata={"error": str(e)},
                )
                return {"success": False, "error": f"Failed to send: {e}", "status": "send_failed"}

        except (AttributeError, TypeError, ValueError, RuntimeError) as e:
            return {"success": False, "error": f"Unexpected error: {e}", "status": "error"}

    async def reply_dm(
        self,
        channel_id: int,
        content: str,
        actor_id: int | None = None,
        source_ip: str | None = None,
        user_agent: str | None = None,
    ) -> dict[str, Any]:
        """Reply within a DM channel (channel_id is the DM channel's ID)."""
        bot = self._bot
        if bot is None:
            return {"success": False, "error": "Bot not ready", "status": "not_ready"}

        # Find the DM channel
        channel = bot.get_channel(channel_id)
        if channel is None:
            # If not found directly, try to find it by iterating private channels
            try:
                for private_ch in bot.private_channels:
                    if private_ch.id == channel_id:
                        channel = private_ch
                        break
            except (AttributeError, TypeError) as e:
                logger.debug(f"Failed to iterate private channels: {e}")

        if channel is None or not hasattr(channel, "recipient"):
            return {"success": False, "error": "DM channel not found", "status": "channel_not_found"}

        recipient = channel.recipient
        if recipient is None:
            return {"success": False, "error": "DM recipient not available", "status": "recipient_not_found"}

        # Delegate to send_dm using recipient's user ID
        return await self.send_dm(
            target_user_id=recipient.id,
            content=content,
            actor_id=actor_id,
            source_ip=source_ip,
            user_agent=user_agent,
        )

    async def reply_guild_message(
        self,
        message_id: int,
        channel_id: int,
        content: str,
        actor_id: int | None = None,
        source_ip: str | None = None,
        user_agent: str | None = None,
    ) -> dict[str, Any]:
        """Reply to a guild message with MessageReference if possible."""
        # Audit: requested
        await self._audit_store.record(
            event_type=EVENT_DASHBOARD_REPLY_REQUESTED,
            result="pending",
            actor_user_id=actor_id,
            actor_source_ip=source_ip,
            actor_user_agent=user_agent,
            target_channel_id=channel_id,
            content_preview=content[:200],
        )

        allowed, reason = await self._rate_limiter.check_and_consume(actor_id, channel_id)
        if not allowed:
            await self._audit_store.record(
                event_type=EVENT_DASHBOARD_REPLY_FAILURE,
                result="rate_limited",
                actor_user_id=actor_id,
                actor_source_ip=source_ip,
                actor_user_agent=user_agent,
                target_channel_id=channel_id,
                content_preview=content[:100],
                metadata={"reason": reason},
            )
            return {"success": False, "error": reason, "status": "rate_limited"}

        if len(content) > self._config.max_message_chars:
            return {
                "success": False,
                "error": f"Content exceeds {self._config.max_message_chars} characters",
                "status": "too_long",
            }

        bot = self._bot
        if bot is None:
            return {"success": False, "error": "Bot not ready", "status": "not_ready"}

        try:
            # Find the channel
            channel = bot.get_channel(channel_id)
            if channel is None:
                return {"success": False, "error": "Channel not found", "status": "channel_not_found"}

            guild = getattr(channel, "guild", None)
            if guild is None:
                return {"success": False, "error": "Not a guild channel", "status": "invalid_channel"}

            # Check permissions
            if not hasattr(channel, "permissions_for"):
                return {"success": False, "error": "Not a text channel", "status": "invalid_channel"}

            try:
                perms = channel.permissions_for(guild.me)
            except (AttributeError, TypeError, discord.Forbidden):
                return {"success": False, "error": "Cannot check permissions", "status": "perm_check_failed"}

            if not perms.send_messages:
                return {"success": False, "error": "Bot lacks send_messages permission", "status": "permission_denied"}

            # Try to fetch the original message for MessageReference
            original_msg = None
            with contextlib.suppress(Exception):
                original_msg = await asyncio.wait_for(channel.fetch_message(message_id), timeout=10.0)

            # Build reply
            if original_msg:
                ref = original_msg.to_reference()
                msg = await asyncio.wait_for(channel.send(content, reference=ref), timeout=30.0)
            else:
                # Fallback: send without reference
                msg = await asyncio.wait_for(channel.send(content), timeout=30.0)

            # Archive in MessageStore
            if self._message_store and self._config.guild_archive_enabled:
                try:
                    bot_user = bot.user
                    channel_name = getattr(channel, "name", None)
                    await self._message_store.insert_message(
                        discord_message_id=msg.id,
                        channel_id=channel.id,
                        guild_id=guild.id,
                        content=msg.content or "",
                        channel_name=channel_name,
                        channel_type=str(getattr(channel, "type", "text")),
                        author_id=bot_user.id if bot_user else 0,
                        author_username=bot_user.name if bot_user else "bot",
                        author_display_name=bot_user.display_name if bot_user else "Bot",
                        author_is_bot=True,
                        is_own_bot=True,
                        direction="outbound",
                        reply_to_message_id=message_id,
                        metadata={"jump_url": msg.jump_url} if msg.jump_url else None,
                    )
                except (AttributeError, TypeError, ValueError, discord.HTTPException) as e:
                    logger.debug("Failed to archive reply in MessageStore: %s", e)

            await self._audit_store.record(
                event_type=EVENT_DASHBOARD_REPLY_SUCCESS,
                result="success",
                actor_user_id=actor_id,
                actor_source_ip=source_ip,
                actor_user_agent=user_agent,
                target_channel_id=channel_id,
                target_guild_id=guild.id if guild else None,
                message_id=msg.id,
                content_preview=content[:200],
                metadata={"message_id": str(msg.id), "reply_to_message_id": message_id},
            )
            return {"success": True, "status": "sent", "message_id": str(msg.id)}

        except discord.Forbidden:
            await self._audit_store.record(
                event_type=EVENT_DASHBOARD_REPLY_FAILURE,
                result="failed",
                actor_user_id=actor_id,
                actor_source_ip=source_ip,
                actor_user_agent=user_agent,
                target_channel_id=channel_id,
                error_code="forbidden",
                content_preview=content[:200],
            )
            return {"success": False, "error": "Bot lacks permission to send", "status": "forbidden"}
        except (discord.HTTPException, discord.NotFound, discord.Forbidden, AttributeError, TypeError, ValueError) as e:
            await self._audit_store.record(
                event_type=EVENT_DASHBOARD_REPLY_FAILURE,
                result="failed",
                actor_user_id=actor_id,
                actor_source_ip=source_ip,
                actor_user_agent=user_agent,
                target_channel_id=channel_id,
                error_code="send_failed",
                content_preview=content[:200],
                metadata={"error": str(e)},
            )
            return {"success": False, "error": f"Failed to send reply: {e}", "status": "send_failed"}

    async def live_channel_messages(
        self,
        channel_id: int,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Fetch live messages from Discord for a channel, inserting into MessageStore."""
        bot = self._bot
        if bot is None:
            return {"messages": [], "error": "bot not ready"}

        try:
            channel = bot.get_channel(channel_id)
            if channel is None:
                return {"messages": [], "error": "channel not found"}

            # Check permissions
            guild = getattr(channel, "guild", None)
            if guild:
                try:
                    perms = channel.permissions_for(guild.me)
                    if not perms.read_message_history:
                        return {"messages": [], "error": "no read_message_history permission"}
                except (AttributeError, TypeError, discord.Forbidden) as e:
                    logger.debug(f"Failed to check permissions for channel {channel.id}: {e}")

            messages = []
            try:

                async def _fetch_history_coro():
                    return [m async for m in channel.history(limit=limit, oldest_first=False)]

                history = await asyncio.wait_for(_fetch_history_coro(), timeout=15.0)
            except discord.Forbidden:
                return {"messages": [], "error": "forbidden"}
            except (AttributeError, TypeError, ValueError, discord.HTTPException) as e:
                return {"messages": [], "error": str(e)}

            # Process in reverse (oldest first for display, but we store in order)
            for msg in reversed(history):
                author = msg.author
                attachments = []
                if msg.attachments:
                    attachments = [{"id": str(a.id), "filename": a.filename, "url": a.url, "size": a.size} for a in msg.attachments]
                embeds = []
                if msg.embeds:
                    for e in msg.embeds[:5]:
                        embeds.append(
                            {
                                "type": str(e.type),
                                "title": e.title,
                                "description": e.description[:300] if e.description else None,
                                "url": e.url,
                                "color": e.color.value if e.color else None,
                            },
                        )

                msg_dict = {
                    "id": str(msg.id),
                    "discord_message_id": str(msg.id),
                    "channel_id": str(channel_id),
                    "guild_id": str(guild.id) if guild else None,
                    "channel_name": getattr(channel, "name", None),
                    "channel_type": str(getattr(channel, "type", "text")),
                    "author_id": str(author.id),
                    "author_username": author.name,
                    "author_display_name": author.display_name,
                    "author_avatar_url": str(author.display_avatar.url) if author.display_avatar else None,
                    "author_is_bot": author.bot,
                    "is_own_bot": bot.user and author.id == bot.user.id,
                    "direction": "outbound" if bot.user and author.id == bot.user.id else "inbound",
                    "content": msg.content or "",
                    "created_at": msg.created_at.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                    "edited_at": msg.edited_at.strftime("%Y-%m-%dT%H:%M:%S.%fZ") if msg.edited_at else None,
                    "reply_to_message_id": str(msg.reference.message_id) if msg.reference and msg.reference.message_id else None,
                    "attachments": attachments,
                    "embeds": embeds,
                    "jump_url": msg.jump_url,
                }
                messages.append(msg_dict)

                # Archive in MessageStore
                if self._message_store and self._config.guild_archive_enabled:
                    try:
                        await self._message_store.insert_message(
                            discord_message_id=msg.id,
                            channel_id=channel.id,
                            guild_id=guild.id if guild else None,
                            content=msg.content or "",
                            channel_name=getattr(channel, "name", None),
                            channel_type=str(getattr(channel, "type", "text")),
                            author_id=author.id,
                            author_username=author.name,
                            author_display_name=author.display_name,
                            author_avatar_url=str(author.display_avatar.url) if author.display_avatar else None,
                            author_is_bot=author.bot,
                            is_own_bot=bot.user and author.id == bot.user.id,
                            direction="outbound" if bot.user and author.id == bot.user.id else "inbound",
                            created_at=msg.created_at.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                            edited_at=msg.edited_at.strftime("%Y-%m-%dT%H:%M:%S.%fZ") if msg.edited_at else None,
                            reply_to_message_id=msg.reference.message_id if msg.reference and msg.reference.message_id else None,
                            attachments=attachments,
                            embeds=embeds,
                            metadata={"jump_url": msg.jump_url} if msg.jump_url else None,
                        )
                    except (AttributeError, TypeError, ValueError, discord.HTTPException) as e:
                        logger.debug("Failed to archive live message: %s", e)

            return {"messages": messages, "count": len(messages)}
        except (AttributeError, TypeError, ValueError, RuntimeError, discord.HTTPException) as e:
            logger.warning("live_channel_messages failed: %s", e)
            return {"messages": [], "error": str(e)}

    async def live_dm_messages(
        self,
        channel_id: int,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Fetch live DM messages from Discord."""
        bot = self._bot
        if bot is None:
            return {"messages": [], "error": "bot not ready"}

        try:
            channel = bot.get_channel(channel_id)
            if channel is None:
                # Try private channels
                for pc in bot.private_channels:
                    if pc.id == channel_id:
                        channel = pc
                        break
            if channel is None:
                return {"messages": [], "error": "DM channel not found"}

            messages = []
            try:

                async def _fetch_history_coro():
                    return [m async for m in channel.history(limit=limit, oldest_first=False)]

                history = await asyncio.wait_for(_fetch_history_coro(), timeout=15.0)
            except (discord.Forbidden, discord.HTTPException, asyncio.TimeoutError, AttributeError, TypeError) as e:
                return {"messages": [], "error": str(e)}

            for msg in reversed(history):
                author = msg.author
                attachments = []
                if msg.attachments:
                    attachments = [{"id": str(a.id), "filename": a.filename, "url": a.url, "size": a.size} for a in msg.attachments]
                embeds = []
                if msg.embeds:
                    for e in msg.embeds[:5]:
                        embeds.append(
                            {
                                "type": str(e.type),
                                "title": e.title,
                                "description": e.description[:300] if e.description else None,
                                "url": e.url,
                            },
                        )

                msg_dict = {
                    "id": str(msg.id),
                    "discord_message_id": str(msg.id),
                    "channel_id": str(channel_id),
                    "author_id": str(author.id),
                    "author_username": author.name,
                    "author_display_name": author.display_name,
                    "author_avatar_url": str(author.display_avatar.url) if author.display_avatar else None,
                    "author_is_bot": author.bot,
                    "is_own_bot": bot.user and author.id == bot.user.id,
                    "direction": "outbound" if bot.user and author.id == bot.user.id else "inbound",
                    "content": msg.content or "",
                    "channel_type": "private",
                    "created_at": msg.created_at.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                    "edited_at": msg.edited_at.strftime("%Y-%m-%dT%H:%M:%S.%fZ") if msg.edited_at else None,
                    "reply_to_message_id": str(msg.reference.message_id) if msg.reference and msg.reference.message_id else None,
                    "attachments": attachments,
                    "embeds": embeds,
                    "jump_url": msg.jump_url,
                }
                messages.append(msg_dict)

                # Archive
                if self._config.dm_archive_enabled:
                    try:
                        await self._dm_store.upsert_user(
                            user_id=author.id,
                            username=author.name,
                            global_name=getattr(author, "global_name", None),
                            display_name=author.display_name,
                            is_bot=author.bot,
                        )
                        await self._dm_store.add_message(
                            message_id=msg.id,
                            channel_id=msg.channel.id,
                            author_id=author.id,
                            content=msg.content or "",
                            clean_content=msg.clean_content if hasattr(msg, "clean_content") else msg.content,
                            is_bot_author=(bot.user and author.id == bot.user.id),
                            reply_to_message_id=msg.reference.message_id if msg.reference else None,
                            jump_url=msg.jump_url,
                        )
                    except (AttributeError, TypeError, ValueError, discord.HTTPException) as e:
                        logger.debug("Failed to archive live DM: %s", e)

                    if self._message_store:
                        try:
                            await self._message_store.insert_message(
                                discord_message_id=msg.id,
                                channel_id=msg.channel.id,
                                content=msg.content or "",
                                channel_type="private",
                                author_id=author.id,
                                author_username=author.name,
                                author_display_name=author.display_name,
                                author_avatar_url=str(author.display_avatar.url) if author.display_avatar else None,
                                author_is_bot=author.bot,
                                is_own_bot=bot.user and author.id == bot.user.id,
                                direction="outbound" if bot.user and author.id == bot.user.id else "inbound",
                                created_at=msg.created_at.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                                reply_to_message_id=msg.reference.message_id if msg.reference and msg.reference.message_id else None,
                                attachments=attachments,
                                embeds=embeds,
                                metadata={"jump_url": msg.jump_url} if msg.jump_url else None,
                            )
                            await self._message_store.upsert_dm_thread(
                                dm_channel_id=msg.channel.id,
                                user_id=author.id,
                                username=author.name,
                                display_name=author.display_name,
                                avatar_url=str(author.display_avatar.url) if author.display_avatar else None,
                                last_message_id=msg.id,
                                last_message_at=msg.created_at.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                            )
                        except (AttributeError, TypeError, ValueError, discord.HTTPException) as e:
                            logger.debug("Failed to archive live DM in MessageStore: %s", e)

            return {"messages": messages, "count": len(messages)}
        except (AttributeError, TypeError, ValueError, RuntimeError, discord.HTTPException) as e:
            logger.warning("live_dm_messages failed: %s", e)
            return {"messages": [], "error": str(e)}

    async def record_dm_message(self, message) -> None:
        """Record an incoming DM to the bot (called from on_message event)."""
        if not self._config.dm_archive_enabled:
            return
        # Only archive DMs where bot is involved
        is_dm = hasattr(message.channel, "recipient") or (hasattr(message.channel, "type") and str(message.channel.type) == "private")
        if not is_dm:
            return
        try:
            bot_user = self._bot.user if self._bot else None
            if bot_user is None:
                logger.debug("DM archive skipped: bot user not ready")
                return
            if message.author.id == bot_user.id:
                return  # Skip bot's own messages for incoming archive

            await self._dm_store.upsert_user(
                user_id=message.author.id,
                username=message.author.name,
                global_name=getattr(message.author, "global_name", None),
                display_name=message.author.display_name,
                is_bot=message.author.bot,
            )
            await self._dm_store.add_message(
                message_id=message.id,
                channel_id=message.channel.id,
                author_id=message.author.id,
                content=message.content,
                clean_content=message.clean_content,
                is_bot_author=False,
                reply_to_message_id=message.reference.message_id if message.reference else None,
                has_attachments=bool(message.attachments),
                has_embeds=bool(message.embeds),
                jump_url=message.jump_url,
            )

            # Also archive in unified MessageStore
            if self._message_store:
                try:
                    attachments = []
                    if message.attachments:
                        attachments = [{"id": str(a.id), "filename": a.filename, "url": a.url, "size": a.size} for a in message.attachments]
                    embeds = []
                    if message.embeds:
                        for e in message.embeds[:5]:
                            embeds.append(
                                {
                                    "type": str(e.type),
                                    "title": e.title,
                                    "url": e.url,
                                },
                            )
                    await self._message_store.insert_message(
                        discord_message_id=message.id,
                        channel_id=message.channel.id,
                        content=message.content or "",
                        channel_type="private",
                        author_id=message.author.id,
                        author_username=message.author.name,
                        author_display_name=message.author.display_name,
                        author_avatar_url=str(message.author.display_avatar.url) if message.author.display_avatar else None,
                        author_is_bot=message.author.bot,
                        is_own_bot=False,
                        direction="inbound",
                        created_at=message.created_at.strftime("%Y-%m-%dT%H:%M:%S.%fZ") if message.created_at else None,
                        edited_at=message.edited_at.strftime("%Y-%m-%dT%H:%M:%S.%fZ") if message.edited_at else None,
                        reply_to_message_id=message.reference.message_id if message.reference and message.reference.message_id else None,
                        attachments=attachments,
                        embeds=embeds,
                        metadata={"jump_url": message.jump_url} if message.jump_url else None,
                    )

                    # Upsert DM thread
                    await self._message_store.upsert_dm_thread(
                        dm_channel_id=message.channel.id,
                        user_id=message.author.id,
                        username=message.author.name,
                        display_name=message.author.display_name,
                        avatar_url=str(message.author.display_avatar.url) if message.author.display_avatar else None,
                        last_message_id=message.id,
                        last_message_at=message.created_at.strftime("%Y-%m-%dT%H:%M:%S.%fZ") if message.created_at else None,
                        increment_count=True,
                    )
                except (AttributeError, TypeError, ValueError, discord.HTTPException) as e:
                    logger.debug("Failed to archive DM in MessageStore: %s", e)

        except (AttributeError, TypeError, ValueError, RuntimeError) as e:
            logger.warning("Failed to archive DM message: %s", e)

    async def record_guild_message(self, message) -> None:
        """Record a guild message into the unified MessageStore."""
        if not self._config.guild_archive_enabled:
            return
        if not self._message_store:
            return

        # Only guild channels (not DMs)
        is_dm = hasattr(message.channel, "recipient") or (hasattr(message.channel, "type") and str(message.channel.type) == "private")
        if is_dm:
            return

        try:
            bot_user = self._bot.user if self._bot else None
            is_own = bot_user is not None and message.author.id == bot_user.id
            direction = "outbound" if is_own else "inbound"

            guild = getattr(message.channel, "guild", None)
            guild_id = guild.id if guild else None
            channel_name = getattr(message.channel, "name", None)
            channel_type = str(getattr(message.channel, "type", "text"))

            attachments = []
            if message.attachments:
                attachments = [
                    {
                        "id": str(a.id),
                        "filename": a.filename,
                        "url": a.url,
                        "size": a.size,
                        "content_type": a.content_type,
                    }
                    for a in message.attachments
                ]

            embeds = []
            if message.embeds:
                for e in message.embeds[:5]:
                    embeds.append(
                        {
                            "type": str(e.type),
                            "title": e.title,
                            "description": e.description[:200] if e.description else None,
                            "url": e.url,
                        },
                    )

            await self._message_store.insert_message(
                discord_message_id=message.id,
                channel_id=message.channel.id,
                guild_id=guild_id,
                content=message.content or "",
                channel_name=channel_name,
                channel_type=channel_type,
                author_id=message.author.id,
                author_username=message.author.name,
                author_display_name=message.author.display_name,
                author_avatar_url=str(message.author.display_avatar.url) if message.author.display_avatar else None,
                author_is_bot=message.author.bot,
                is_own_bot=is_own,
                direction=direction,
                created_at=message.created_at.strftime("%Y-%m-%dT%H:%M:%S.%fZ") if message.created_at else None,
                edited_at=message.edited_at.strftime("%Y-%m-%dT%H:%M:%S.%fZ") if message.edited_at else None,
                reply_to_message_id=message.reference.message_id if message.reference and message.reference.message_id else None,
                attachments=attachments,
                embeds=embeds,
                metadata={"jump_url": message.jump_url} if message.jump_url else None,
            )
        except (AttributeError, TypeError, ValueError, discord.HTTPException) as e:
            logger.debug("Failed to archive guild message: %s", e)


def _uptime(bot) -> int:
    """Calculate bot uptime in seconds."""
    if bot and hasattr(bot, "ready_at") and bot.ready_at:
        return int((datetime.now(UTC) - bot.ready_at).total_seconds())
    return 0


def _format_uptime(seconds: int) -> str:
    """Format uptime as human-readable string."""
    if seconds < 0:
        return "0s"
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def _iso_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
