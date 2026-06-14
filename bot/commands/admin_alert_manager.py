"""Admin alert manager — session lifecycle, reaction queue, discovery, and broadcasting."""

import asyncio
import time
from typing import Any

import discord

from bot.commands.admin_alert_models import (
    AlertDestination,
    AlertSession,
    AlertSessionStatus,
)
from bot.config import load_config
from bot.utils.logging import get_logger


class AdminAlertManager:
    def __init__(self, bot) -> None:
        self.bot = bot
        self.config = load_config()
        self.logger = get_logger(f"{__name__}.AdminAlertManager")
        self.sessions: dict[int, AlertSession] = {}
        self._sessions_lock = asyncio.Lock()
        self.reaction_queues: dict[int, list] = {}  # Per-message reaction queues

        self.enabled = self.config.get("ALERT_ENABLE", "false").lower() == "true"
        self.admin_user_ids = self._build_authorized_user_ids()
        self.session_timeout = int(self.config.get("ALERT_SESSION_TIMEOUT_S", "1800"))

        self.logger.info(f"Admin alert system initialized: enabled={self.enabled}")

    async def _queue_reaction_operation(self, message, emoji: str, operation: str, user) -> None:
        """Queue reaction add/remove operations with spacing to prevent rate limits."""
        message_id = message.id
        if message_id not in self.reaction_queues:
            self.reaction_queues[message_id] = []

        queue = self.reaction_queues[message_id]
        for queued_op in queue:
            if queued_op["emoji"] == emoji and queued_op["operation"] == operation:
                return

        queue.append({"emoji": emoji, "operation": operation, "user": user, "message": message})
        if len(queue) == 1:
            self.logger.debug(f"Starting reaction queue processing for message {message_id}")
            await self._process_reaction_queue(message_id)

    async def _process_reaction_queue(self, message_id: int) -> None:
        """Process queued reactions with spacing."""
        queue = self.reaction_queues.get(message_id, [])
        while queue:
            op = queue.pop(0)
            try:
                if op["operation"] == "add":
                    existing = [r.emoji for r in op["message"].reactions]
                    if op["emoji"] not in [str(r) for r in existing]:
                        await op["message"].add_reaction(op["emoji"])
                elif op["operation"] == "remove":
                    await op["message"].remove_reaction(op["emoji"], op["user"])
                if queue:
                    await asyncio.sleep(0.25)
            except (discord.HTTPException, discord.NotFound, discord.Forbidden) as e:
                self.logger.warning(f"Reaction queue operation failed: {e}")

        if message_id in self.reaction_queues and not self.reaction_queues[message_id]:
            del self.reaction_queues[message_id]

    def _parse_admin_user_ids(self) -> set[int]:
        try:
            admin_ids_str = self.config.get("ALERT_ADMIN_USER_IDS", "")
            if not admin_ids_str:
                return set()
            return {int(s.strip()) for s in admin_ids_str.split(",") if s.strip()}
        except (ValueError, AttributeError, TypeError) as e:
            self.logger.exception(f"Failed to parse admin user IDs: {e}")
            return set()

    def _build_authorized_user_ids(self) -> set[int]:
        explicit = self._parse_admin_user_ids()
        if explicit:
            return set(explicit)
        authorized: set[int] = set()
        try:
            owners = self.config.get("OWNER_IDS", [])
            if isinstance(owners, list):
                for owner_id in owners:
                    try:
                        authorized.add(int(owner_id))
                    except (ValueError, TypeError) as e:
                        self.logger.debug(f"Invalid owner ID {owner_id}: {e}")
        except (ValueError, AttributeError, TypeError) as e:
            self.logger.debug(f"Failed to parse OWNER_IDS: {e}")
        return authorized

    def refresh_config(self) -> None:
        try:
            self.config = load_config()
        except (FileNotFoundError, ValueError, TypeError, OSError) as e:
            self.logger.debug(f"Failed to reload config: {e}")
            return
        try:
            self.enabled = self.config.get("ALERT_ENABLE", "false").lower() == "true"
        except (AttributeError, TypeError) as e:
            self.logger.debug(f"Failed to parse ALERT_ENABLE: {e}")
        try:
            self.admin_user_ids = self._build_authorized_user_ids()
        except (ValueError, AttributeError, TypeError) as e:
            self.logger.debug(f"Failed to build authorized user IDs: {e}")
        try:
            self.session_timeout = int(self.config.get("ALERT_SESSION_TIMEOUT_S", "1800"))
        except (ValueError, TypeError) as e:
            self.logger.debug(f"Failed to parse ALERT_SESSION_TIMEOUT_S: {e}")

    def is_admin_user(self, user_id: int) -> bool:
        return user_id in self.admin_user_ids

    def is_dm_channel(self, channel) -> bool:
        return isinstance(channel, (discord.DMChannel, discord.GroupChannel))

    async def create_session(self, user_id: int) -> AlertSession:
        session_id = f"alert_{user_id}_{int(time.time())}"
        now = time.time()
        session = AlertSession(
            user_id=user_id,
            session_id=session_id,
            status=AlertSessionStatus.COMPOSING,
            created_at=now,
            expires_at=now + self.session_timeout,
        )
        if user_id in self.sessions:
            old = self.sessions[user_id]
            self.logger.info(f"Replacing session {old.session_id}")
        self.sessions[user_id] = session
        self.logger.info(f"Created alert session {session_id}")
        return session

    def get_session(self, user_id: int) -> AlertSession | None:
        session = self.sessions.get(user_id)
        if not session:
            return None
        if time.time() > session.expires_at:
            self.logger.info(f"Session {session.session_id} expired")
            session.status = AlertSessionStatus.EXPIRED
            del self.sessions[user_id]
            return None
        return session

    def _validate_embed_limits(self, embed: discord.Embed) -> discord.Embed:
        """Validate and truncate embed to stay within Discord limits."""
        if embed.title and len(embed.title) > 256:
            embed.title = embed.title[:253] + "..."
        if embed.description and len(embed.description) > 4096:
            embed.description = embed.description[:4093] + "..."
        if len(embed.fields) > 25:
            truncated = len(embed.fields) - 24
            embed.fields = embed.fields[:24]
            embed.add_field(name="Truncated", value=f"...and {truncated} more fields", inline=False)
        for field in embed.fields:
            if len(field.name) > 256:
                field.name = field.name[:253] + "..."
            if len(field.value) > 1024:
                field.value = field.value[:1021] + "..."
        if embed.footer and len(embed.footer.text) > 2048:
            embed.set_footer(text=embed.footer.text[:2045] + "...")
        total = len(embed.title or "") + len(embed.description or "")
        for f in embed.fields:
            total += len(f.name) + len(f.value)
        if embed.footer:
            total += len(embed.footer.text)
        if total > 6000:
            self.logger.warning(f"Embed exceeds 6000 chars ({total}), may cause errors")
        return embed

    def _discover_available_destinations(self, invoking_user_id: int) -> list[AlertDestination]:
        """Cache-based, permission-aware discovery of available guilds/channels."""
        destinations: list[AlertDestination] = []
        guilds_shown = 0
        channels_shown = 0
        max_guilds = 10
        max_channels_per_guild = 3
        try:
            for guild in self.bot.guilds:
                if guilds_shown >= max_guilds:
                    break
                bot_member = guild.get_member(self.bot.user.id)
                if not bot_member:
                    continue
                invoking_member = guild.get_member(invoking_user_id)
                if not invoking_member:
                    continue
                eligible = []
                for channel in guild.text_channels:
                    if len(eligible) >= max_channels_per_guild:
                        break
                    if channel.permissions_for(bot_member).send_messages:
                        eligible.append(channel)
                        channels_shown += 1
                if eligible:
                    guilds_shown += 1
                    for ch in eligible:
                        destinations.append(
                            AlertDestination(
                                guild_id=guild.id,
                                channel_id=ch.id,
                                channel_name=ch.name,
                                guild_name=guild.name,
                            ),
                        )
            total_guilds = len(self.bot.guilds)
            total_channels = sum(len(g.text_channels) for g in self.bot.guilds)
            self.logger.info(f"alert:discovery guilds={total_guilds} channels={total_channels} shown_guilds={guilds_shown} shown_channels={channels_shown}")
            return destinations
        except Exception as e:
            self.logger.exception(f"Discovery failed: {e}")
            return []

    async def build_composer_embed(self, session: AlertSession) -> discord.Embed:
        embed = discord.Embed(
            title="Admin Alert Composer",
            color=0x1F8B4C,
            timestamp=discord.utils.utcnow(),
        )
        step_map = {
            "select_channels": "1. Select Channels",
            "compose_content": "2. Compose Content",
            "preview_alert": "3. Preview & Send",
            "confirm_send": "4. Confirm Send",
        }
        current = step_map.get(session.current_step, "Unknown")
        embed.add_field(name="Current Step", value=current, inline=False)

        if session.destinations:
            parts = []
            for d in session.destinations[:5]:
                parts.append(f"#{d.channel_name or 'unknown'}")
            if len(session.destinations) > 5:
                parts.append(f"... and {len(session.destinations) - 5} more")
            embed.add_field(
                name=f"Destinations ({len(session.destinations)})",
                value="\n".join(parts),
                inline=True,
            )
        if session.content or session.embed_title:
            preview = session.content[:100] if session.content else ""
            if session.embed_title:
                preview = f"**{session.embed_title}**\n{preview}"
            if len(preview) > 150:
                preview = preview[:147] + "..."
            embed.add_field(name="Content Preview", value=preview or "*No content*", inline=True)

        remaining = int(session.expires_at - time.time())
        embed.add_field(name="Session Info", value=f"Expires: {remaining // 60}m {remaining % 60}s", inline=True)

        desc_map = {
            "select_channels": "React with the channel icon to select channels.",
            "compose_content": "React with the compose icon to compose content.",
            "preview_alert": "React with the preview icon to review before sending.",
            "confirm_send": "Final confirmation required.",
        }
        embed.description = desc_map.get(session.current_step, "")
        embed.set_footer(text=f"Session: {session.session_id}")
        return self._validate_embed_limits(embed)

    async def get_accessible_channels(self) -> list[discord.TextChannel]:
        accessible: list[discord.TextChannel] = []
        for guild in self.bot.guilds:
            member = getattr(guild, "me", None) or guild.get_member(self.bot.user.id)
            if member is None:
                continue
            for channel in guild.text_channels:
                perms = channel.permissions_for(member)
                if perms.send_messages and perms.read_messages:
                    accessible.append(channel)
        accessible.sort(key=lambda c: (c.guild.name.lower(), c.guild.id, c.position, c.name.lower()))
        return accessible

    async def send_alert(self, session: AlertSession) -> dict[str, Any]:
        self.logger.info(f"Sending alert from session {session.session_id}")
        session.status = AlertSessionStatus.POSTING
        MAX_CAP = 20
        capped = len(session.destinations) > MAX_CAP
        destinations = session.destinations[:MAX_CAP] if capped else session.destinations
        if capped:
            self.logger.warning(f"alert:send_alert:recipient_cap total={len(session.destinations)} cap={MAX_CAP}")
        results = {"total_destinations": len(session.destinations), "successful_sends": 0, "failed_sends": 0, "send_results": []}
        alert_content = session.content
        embed = None
        if session.embed_title or session.embed_description:
            embed = discord.Embed(
                title=session.embed_title,
                description=session.embed_description,
                color=0x1F8B4C,
                timestamp=discord.utils.utcnow(),
            )
            embed.set_footer(text="Admin Alert")
        for dest in destinations:
            try:
                channel = self.bot.get_channel(dest.channel_id)
                if not channel:
                    results["send_results"].append({"channel_id": dest.channel_id, "success": False, "error": "Channel not found"})
                    results["failed_sends"] += 1
                    continue
                if embed:
                    msg = await channel.send(content=alert_content, embed=embed)
                else:
                    msg = await channel.send(content=alert_content)
                results["send_results"].append({"channel_id": dest.channel_id, "success": True, "message_id": msg.id})
                results["successful_sends"] += 1
            except Exception as e:
                self.logger.exception(f"Failed to send to {dest.channel_name}: {e}")
                results["send_results"].append({"channel_id": dest.channel_id, "success": False, "error": str(e)})
                results["failed_sends"] += 1
        session.status = AlertSessionStatus.COMPLETED
        return results
