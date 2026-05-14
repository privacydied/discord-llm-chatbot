"""Service layer: narrow interfaces between dashboard and bot internals.

Provides:
- Summary collection with coalescing and short TTL cache
- Guild inventory
- DM sending with permission checks, rate limits, audit logging
- Guild message sending with permission verification
- Rate limiter for send actions
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

from bot.utils.logging import get_logger

if TYPE_CHECKING:
    from discord.ext.commands import Bot as DiscordBot

    from .audit_store import AuditStore
    from .config import DashboardConfig
    from .dm_store import DMStore

logger = get_logger(__name__)


class _RateLimiter:
    """Token-bucket rate limiter per (user_id, target) pair."""

    def __init__(self, sends_per_minute: int) -> None:
        self._sends_per_minute = sends_per_minute
        self._windows: dict[str, list[float]] = defaultdict(list)
        self._lock = asyncio.Lock()

    def _bucket_key(self, actor_id: Optional[int], target_id: Optional[int]) -> str:
        return f"{actor_id}:{target_id}"

    async def check_and_consume(self, actor_id: Optional[int], target_id: Optional[int]) -> tuple[bool, str]:
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
        bot: Optional["DiscordBot"],
        config: "DashboardConfig",
        audit_store: "AuditStore",
        dm_store: "DMStore",
    ) -> None:
        self._bot = bot
        self._config = config
        self._audit_store = audit_store
        self._dm_store = dm_store
        self._rate_limiter = _RateLimiter(config.rate_limit_sends_per_minute)
        self._summary_cache: Optional[dict[str, Any]] = None
        self._summary_cache_time: float = 0
        self._summary_lock = asyncio.Lock()

    @property
    def bot(self) -> Optional["DiscordBot"]:
        return self._bot

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
                try:
                    total_users += g.member_count or 0
                except Exception:
                    pass

            # Cog count
            cog_count = len(bot.cogs) if bot.cogs else 0

            # Latency
            latency = round(bot.latency * 1000, 1) if bot.latency else 0

            # Audit event count
            try:
                audit_result = await self._audit_store.query(page=1, page_size=1)
                audit_count = audit_result.get("total", 0)
            except Exception:
                audit_count = 0

            return {
                "status": "ready",
                "bot_username": bot_user.display_name if bot_user else "unknown",
                "bot_id": str(bot_user.id) if bot_user else None,
                "uptime_seconds": uptime,
                "uptime_human": _format_uptime(uptime),
                "guild_count": guild_count,
                "total_users_estimate": total_users,
                "cog_count": cog_count,
                "latency_ms": latency,
                "audit_event_count": audit_count,
                "dm_archive_enabled": self._config.dm_archive_enabled,
                "show_message_previews": self._config.show_message_previews,
                "loaded_at": _iso_now(),
            }
        except Exception as e:
            logger.warning("Dashboard summary collection failed: %s", e)
            return {"status": "error", "error": str(e)}

    async def get_guilds(
        self,
        page: int = 1,
        page_size: int = 50,
        max_page_size: int = 200,
        search: Optional[str] = None,
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
                owner_id = str(g.owner_id) if g.owner_id else None
                member_count = g.member_count
                joined_at = g.me.joined_at.strftime("%Y-%m-%dT%H:%M:%SZ") if g.me and g.me.joined_at else None

                # Bot permissions in a representative channel
                perm_summary = "unknown"
                if g.text_channels:
                    try:
                        ch = list(g.text_channels)[0]
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
                    except Exception:
                        pass

                guilds.append(
                    {
                        "id": str(g.id),
                        "name": g.name,
                        "owner_id": owner_id,
                        "member_count": member_count,
                        "channel_count": channel_count,
                        "text_channel_count": text_channel_count,
                        "joined_at": joined_at,
                        "permissions": perm_summary,
                        "features": list(g.features) if g.features else [],
                        "icon_url": str(g.icon.url) if g.icon else None,
                    }
                )
            except Exception as e:
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
        actor_id: Optional[int] = None,
        source_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> dict[str, Any]:
        """Send a DM as the bot. Owner-only, rate-limited, audited."""
        # Rate limit check
        allowed, reason = await self._rate_limiter.check_and_consume(actor_id, target_user_id)
        if not allowed:
            await self._audit_store.record(
                event_type="dashboard.send.dm",
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

        # Audit: pending
        await self._audit_store.record(
            event_type="dashboard.send.dm",
            result="pending",
            actor_user_id=actor_id,
            actor_source_ip=source_ip,
            actor_user_agent=user_agent,
            target_user_id=target_user_id,
            content_preview=content[:200],
        )

        bot = self._bot
        if bot is None:
            return {"success": False, "error": "Bot not ready", "status": "not_ready"}

        try:
            # Resolve user
            user = bot.get_user(target_user_id)
            if user is None:
                try:
                    user = await asyncio.wait_for(bot.fetch_user(target_user_id), timeout=10.0)
                except Exception as e:
                    error_msg = f"User not found: {e}"
                    await self._audit_store.record(
                        event_type="dashboard.send.dm",
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

                await self._audit_store.record(
                    event_type="dashboard.send.dm",
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
                    event_type="dashboard.send.dm",
                    result="failed",
                    actor_user_id=actor_id,
                    actor_source_ip=source_ip,
                    actor_user_agent=user_agent,
                    target_user_id=target_user_id,
                    error_code="forbidden",
                    content_preview=content[:200],
                )
                return {"success": False, "error": "Cannot DM this user (blocked or privacy settings)", "status": "forbidden"}

            except Exception as e:
                await self._audit_store.record(
                    event_type="dashboard.send.dm",
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

        except Exception as e:
            return {"success": False, "error": f"Unexpected error: {e}", "status": "error"}

    async def send_guild_message(
        self,
        guild_id: int,
        channel_id: int,
        content: str,
        actor_id: Optional[int] = None,
        source_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> dict[str, Any]:
        """Send a message to a guild channel where bot has permission."""
        allowed, reason = await self._rate_limiter.check_and_consume(actor_id, channel_id)
        if not allowed:
            await self._audit_store.record(
                event_type="dashboard.send.guild_message",
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

        await self._audit_store.record(
            event_type="dashboard.send.guild_message",
            result="pending",
            actor_user_id=actor_id,
            actor_source_ip=source_ip,
            actor_user_agent=user_agent,
            target_guild_id=guild_id,
            target_channel_id=channel_id,
            content_preview=content[:200],
        )

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
            except Exception:
                return {"success": False, "error": "Cannot check permissions", "status": "perm_check_failed"}

            if not perms.send_messages:
                return {"success": False, "error": "Bot lacks send_messages permission", "status": "permission_denied"}

            if not perms.read_message_history:
                return {"success": False, "error": "Bot lacks read_message_history permission", "status": "permission_denied"}

            try:
                msg = await asyncio.wait_for(channel.send(content), timeout=30.0)
                await self._audit_store.record(
                    event_type="dashboard.send.guild_message",
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
                    event_type="dashboard.send.guild_message",
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
            except Exception as e:
                await self._audit_store.record(
                    event_type="dashboard.send.guild_message",
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

        except Exception as e:
            return {"success": False, "error": f"Unexpected error: {e}", "status": "error"}

    async def record_dm_message(self, message) -> None:
        """Record an incoming DM to the bot (called from on_message event)."""
        if not self._config.dm_archive_enabled:
            return
        # Only archive DMs where bot is involved
        if not hasattr(message.channel, "recipient") and not (hasattr(message.channel, "type") and str(message.channel.type) == "private"):
            return
        try:
            bot_user = self._bot.user if self._bot else None
            if bot_user is None or message.author.id == bot_user.id:
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
        except Exception as e:
            logger.warning("Failed to archive DM message: %s", e)


def _uptime(bot) -> int:
    """Calculate bot uptime in seconds."""
    if bot and hasattr(bot, "ready_at") and bot.ready_at:
        return int((datetime.now(timezone.utc) - bot.ready_at).total_seconds())
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
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# Import discord inside functions to avoid circular imports
import discord
