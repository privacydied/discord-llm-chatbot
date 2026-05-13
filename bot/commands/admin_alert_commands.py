"""
Admin DM Alert System - Secure, DM-only broadcast alerting with emoji-driven composer.
"""

import asyncio
import time
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum

import discord
from discord.ext import commands

from bot.config import load_config
from bot.utils.logging import get_logger
from bot.public_output import sanitize_public_text

logger = get_logger(__name__)


class AlertSessionStatus(Enum):
    COMPOSING = "composing"
    READY = "ready"
    POSTING = "posting"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


@dataclass
class AlertDestination:
    guild_id: Optional[int]
    channel_id: Optional[int]
    channel_name: Optional[str]
    guild_name: Optional[str] = None
    permissions_valid: bool = True
    permission_issues: List[str] = field(default_factory=list)


@dataclass
class AlertSession:
    user_id: int
    session_id: str
    status: AlertSessionStatus
    created_at: float
    expires_at: float
    content: str = ""
    embed_title: str = ""
    embed_description: str = ""
    destinations: List[AlertDestination] = field(default_factory=list)
    mention_everyone: bool = False
    current_step: str = "select_channels"
    composer_message_id: Optional[int] = None
    composer_ready: bool = False
    # Guild navigation pagination (NEW)
    guild_page: int = 0
    selected_guild_id: Optional[int] = None
    channel_page: int = 0
    guilds_list: List = field(default_factory=list)
    selection_message_id: Optional[int] = None
    channel_message_id: Optional[int] = None


class AdminAlertManager:
    def __init__(self, bot):
        self.bot = bot
        self.config = load_config()
        self.logger = get_logger(f"{__name__}.AdminAlertManager")
        self.sessions: Dict[int, AlertSession] = {}
        self._sessions_lock = asyncio.Lock()
        self.reaction_queues: Dict[int, List] = {}  # Per-message reaction queues

        self.enabled = self.config.get("ALERT_ENABLE", "false").lower() == "true"
        self.admin_user_ids = self._build_authorized_user_ids()
        self.session_timeout = int(self.config.get("ALERT_SESSION_TIMEOUT_S", "1800"))

        self.logger.info(f"🚨 Admin alert system initialized: enabled={self.enabled}")

    async def _queue_reaction_operation(
        self, message, emoji: str, operation: str, user
    ):
        """Queue reaction add/remove operations with spacing to prevent rate limits."""

        message_id = message.id
        if message_id not in self.reaction_queues:
            self.reaction_queues[message_id] = []

        queue = self.reaction_queues[message_id]

        # Check for duplicates
        for queued_op in queue:
            if queued_op["emoji"] == emoji and queued_op["operation"] == operation:
                return  # Already queued

        queue.append(
            {"emoji": emoji, "operation": operation, "user": user, "message": message}
        )

        # If this is the first item, start processing
        if len(queue) == 1:
            self.logger.debug(
                f"🎯 Starting reaction queue processing for message {message_id}"
            )
            await self._process_reaction_queue(message_id)

    async def _process_reaction_queue(self, message_id: int) -> None:
        """Process queued reactions with spacing."""
        import asyncio

        queue = self.reaction_queues.get(message_id, [])

        while queue:
            op = queue.pop(0)
            try:
                if op["operation"] == "add":
                    # Check if reaction already exists
                    existing_reactions = [r.emoji for r in op["message"].reactions]
                    if op["emoji"] not in [str(r) for r in existing_reactions]:
                        await op["message"].add_reaction(op["emoji"])
                elif op["operation"] == "remove":
                    await op["message"].remove_reaction(op["emoji"], op["user"])

                # Wait between operations to prevent rate limits
                if queue:  # Only wait if more operations pending
                    await asyncio.sleep(0.25)  # 250ms spacing

            except Exception as e:
                self.logger.warning(f"⚠️ Reaction queue operation failed: {e}")

        # Clean up empty queue
        if message_id in self.reaction_queues and not self.reaction_queues[message_id]:
            del self.reaction_queues[message_id]
            self.logger.debug(f"🎯 Reaction queue drained for message {message_id}")

    def _parse_admin_user_ids(self) -> Set[int]:
        try:
            admin_ids_str = self.config.get("ALERT_ADMIN_USER_IDS", "")
            if not admin_ids_str:
                return set()

            admin_ids = set()
            for id_str in admin_ids_str.split(","):
                id_str = id_str.strip()
                if id_str:
                    admin_ids.add(int(id_str))
            return admin_ids
        except Exception as e:
            self.logger.error(f"❌ Failed to parse admin user IDs: {e}")
            return set()

    def _build_authorized_user_ids(self) -> Set[int]:
        explicit = self._parse_admin_user_ids()
        if explicit:
            return set(explicit)

        authorized: Set[int] = set()
        try:
            owners = self.config.get("OWNER_IDS", [])
            if isinstance(owners, list):
                for owner_id in owners:
                    try:
                        authorized.add(int(owner_id))
                    except Exception:
                        continue
        except Exception:
            pass
        return authorized

    def refresh_config(self) -> None:
        try:
            self.config = load_config()
        except Exception:
            return

        try:
            self.enabled = self.config.get("ALERT_ENABLE", "false").lower() == "true"
        except Exception:
            pass

        try:
            self.admin_user_ids = self._build_authorized_user_ids()
        except Exception:
            pass

        try:
            self.session_timeout = int(
                self.config.get("ALERT_SESSION_TIMEOUT_S", "1800")
            )
        except Exception:
            pass

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
            old_session = self.sessions[user_id]
            self.logger.info(f"♻️ Replacing session {old_session.session_id}")

        self.sessions[user_id] = session
        self.logger.info(f"🚀 Created alert session {session_id}")
        return session

    def get_session(self, user_id: int) -> Optional[AlertSession]:
        session = self.sessions.get(user_id)
        if not session:
            return None

        if time.time() > session.expires_at:
            self.logger.info(f"⏰ Session {session.session_id} expired")
            session.status = AlertSessionStatus.EXPIRED
            del self.sessions[user_id]
            return None

        return session

    def _validate_embed_limits(self, embed: discord.Embed) -> discord.Embed:
        """Validate and truncate embed to stay within Discord limits."""

        # Title limit: 256 characters
        if embed.title and len(embed.title) > 256:
            embed.title = embed.title[:253] + "..."

        # Description limit: 4096 characters
        if embed.description and len(embed.description) > 4096:
            embed.description = embed.description[:4093] + "..."

        # Fields limit: 25 fields max
        if len(embed.fields) > 25:
            truncated_count = len(embed.fields) - 24
            embed.fields = embed.fields[:24]
            embed.add_field(
                name="Truncated",
                value=f"...and {truncated_count} more fields",
                inline=False,
            )

        # Individual field limits
        for embed_field in embed.fields:
            if len(embed_field.name) > 256:
                embed_field.name = embed_field.name[:253] + "..."
            if len(embed_field.value) > 1024:
                embed_field.value = embed_field.value[:1021] + "..."

        # Footer limit: 2048 characters
        if embed.footer and len(embed.footer.text) > 2048:
            embed.set_footer(text=embed.footer.text[:2045] + "...")

        # Total embed size check: 6000 characters max
        total_length = 0
        total_length += len(embed.title or "")
        total_length += len(embed.description or "")
        for embed_field in embed.fields:
            total_length += len(embed_field.name) + len(embed_field.value)
        if embed.footer:
            total_length += len(embed.footer.text)

        if total_length > 6000:
            self.logger.warning(
                f"⚠️ Embed exceeds 6000 chars ({total_length}), may cause errors"
            )

        return embed

    def _discover_available_destinations(
        self, invoking_user_id: int
    ) -> List[AlertDestination]:
        """Cache-based, permission-aware discovery of available guilds/channels with strict bounds."""
        destinations: List[AlertDestination] = []
        guilds_shown = 0
        channels_shown = 0
        max_guilds = 10  # Hard limit per spec
        max_channels_per_guild = 3  # Hard limit per spec

        try:
            for guild in self.bot.guilds:
                if guilds_shown >= max_guilds:
                    break

                # Check if bot has send permissions in at least one text channel
                bot_member = guild.get_member(self.bot.user.id)
                if not bot_member:
                    continue

                # Check if invoking user is a member (optional constraint)
                invoking_member = guild.get_member(invoking_user_id)
                if not invoking_member:
                    continue

                # Find eligible channels
                eligible_channels = []
                for channel in guild.text_channels:
                    if len(eligible_channels) >= max_channels_per_guild:
                        break

                    # Check bot permissions
                    perms = channel.permissions_for(bot_member)
                    if perms.send_messages:
                        eligible_channels.append(channel)
                        channels_shown += 1

                if eligible_channels:
                    guilds_shown += 1
                    for channel in eligible_channels:
                        destinations.append(
                            AlertDestination(
                                guild_id=guild.id,
                                channel_id=channel.id,
                                channel_name=channel.name,
                                guild_name=guild.name,
                            )
                        )

            # Log discovery results per spec
            total_guilds = len(self.bot.guilds)
            total_channels = sum(len(g.text_channels) for g in self.bot.guilds)
            truncated = guilds_shown < total_guilds or channels_shown < total_channels

            self.logger.info(
                f"alert:discovery guilds={total_guilds} channels={total_channels} shown_guilds={guilds_shown} shown_channels={channels_shown} truncated={truncated}"
            )

            return destinations

        except Exception as e:
            self.logger.error(f"❌ Discovery failed: {e}")
            return []

    async def build_composer_embed(self, session: AlertSession) -> discord.Embed:
        embed = discord.Embed(
            title="🚨 Admin Alert Composer",
            color=0x1F8B4C,
            timestamp=discord.utils.utcnow(),
        )

        step_indicators = {
            "select_channels": "📋 **1. Select Channels**",
            "compose_content": "✏️ **2. Compose Content**",
            "preview_alert": "👁️ **3. Preview & Send**",
            "confirm_send": "📤 **4. Confirm Send**",
        }

        current_step = step_indicators.get(session.current_step, "❓ Unknown")
        embed.add_field(name="Current Step", value=current_step, inline=False)

        if session.destinations:
            dest_text = []
            for dest in session.destinations[:5]:
                status_emoji = "✅" if dest.permissions_valid else "⚠️"
                channel_display = dest.channel_name or "unknown-channel"
                dest_text.append(f"{status_emoji} #{channel_display}")

            if len(session.destinations) > 5:
                dest_text.append(f"... and {len(session.destinations) - 5} more")

            embed.add_field(
                name=f"📋 Destinations ({len(session.destinations)})",
                value="\n".join(dest_text) if dest_text else "None selected",
                inline=True,
            )

        if session.content or session.embed_title:
            content_preview = session.content[:100] if session.content else ""
            if session.embed_title:
                content_preview = f"**{session.embed_title}**\n{content_preview}"
            if len(content_preview) > 150:
                content_preview = content_preview[:147] + "..."

            embed.add_field(
                name="✏️ Content Preview",
                value=content_preview or "*No content yet*",
                inline=True,
            )

        time_remaining = int(session.expires_at - time.time())
        embed.add_field(
            name="⏰ Session Info",
            value=f"Expires in: {time_remaining // 60}m {time_remaining % 60}s",
            inline=True,
        )

        if session.current_step == "select_channels":
            embed.description = "React with 📋 to select channels for broadcasting."
        elif session.current_step == "compose_content":
            embed.description = "React with ✏️ to compose your alert content."
        elif session.current_step == "preview_alert":
            embed.description = "React with 👁️ to preview your alert before sending."
        elif session.current_step == "confirm_send":
            embed.description = (
                "⚠️ **Final confirmation required** - React with 📤 to send alert."
            )

        embed.set_footer(text=f"Session: {session.session_id}")
        return self._validate_embed_limits(embed)

    async def get_accessible_channels(self) -> List[discord.TextChannel]:
        accessible_channels: List[discord.TextChannel] = []

        for guild in self.bot.guilds:
            member = getattr(guild, "me", None) or guild.get_member(self.bot.user.id)
            if member is None:
                continue

            for channel in guild.text_channels:
                perms = channel.permissions_for(member)
                if perms.send_messages and perms.read_messages:
                    accessible_channels.append(channel)

        # Provide a stable ordering so the numbered list matches follow-up selections
        accessible_channels.sort(
            key=lambda channel: (
                channel.guild.name.lower(),
                channel.guild.id,
                channel.position,
                channel.name.lower(),
            )
        )

        return accessible_channels

    async def send_alert(self, session: AlertSession) -> Dict[str, Any]:
        self.logger.info(f"📤 Sending alert from session {session.session_id}")
        session.status = AlertSessionStatus.POSTING

        # Hard recipient cap: MAX 20 sends per alert [CAP]
        MAX_CAP = 20
        capped = len(session.destinations) > MAX_CAP
        destinations_iter = session.destinations[:MAX_CAP] if capped else session.destinations

        if capped:
            self.logger.warning(
                f"alert:send_alert:recipient_cap total={len(session.destinations)} "
                f"cap={MAX_CAP} session_id={session.session_id}"
            )

        results = {
            "total_destinations": len(session.destinations),
            "successful_sends": 0,
            "failed_sends": 0,
            "send_results": [],
        }

        # Build alert content
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

        for dest in destinations_iter:
            try:
                channel = self.bot.get_channel(dest.channel_id)
                if not channel:
                    results["send_results"].append(
                        {
                            "channel_id": dest.channel_id,
                            "success": False,
                            "error": "Channel not found",
                        }
                    )
                    results["failed_sends"] += 1
                    continue

                if embed:
                    message = await channel.send(content=alert_content, embed=embed)
                else:
                    message = await channel.send(content=alert_content)

                results["send_results"].append(
                    {
                        "channel_id": dest.channel_id,
                        "success": True,
                        "message_id": message.id,
                    }
                )
                results["successful_sends"] += 1

            except Exception as e:
                self.logger.error(f"❌ Failed to send to {dest.channel_name}: {e}")
                results["send_results"].append(
                    {"channel_id": dest.channel_id, "success": False, "error": str(e)}
                )
                results["failed_sends"] += 1

        session.status = AlertSessionStatus.COMPLETED
        return results


class AdminAlertCommands(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.config = load_config()
        self.logger = get_logger(f"{__name__}.AdminAlertCommands")
        self.alert_manager = AdminAlertManager(bot)
        self.logger.info("🚨 Admin Alert Commands loaded")

    @commands.command(name="alert")
    @commands.cooldown(2, 300, type=commands.BucketType.user)
    async def alert_command(
        self, ctx: commands.Context, *, message: str = None
    ) -> None:
        """
        Admin broadcast command.

        Usage:
          !alert <message>   - Direct broadcast to all servers
          !alert             - Start interactive composer session (DM-only)
        """
        self.alert_manager.refresh_config()

        # CRITICAL: Admin check BEFORE any routing or side effects [REH][SFT]
        if not self.alert_manager.is_admin_user(ctx.author.id):
            self.logger.warning(
                f"alert:unauthorized user_id={ctx.author.id} channel_type={'DM' if self.alert_manager.is_dm_channel(ctx.channel) else 'guild'}"
            )
            await ctx.send(
                "🚫 Access denied. You are not authorized to use the alert system."
            )
            return

        if not self.alert_manager.enabled:
            await ctx.send("❌ Alert system is disabled.")
            return

        # DIRECT BROADCAST MODE: !alert <message> [REH][CA]
        if message is not None:
            content = message.strip()
            if not content:
                await ctx.send("Usage: !alert <message>")
                return
            await self._handle_direct_broadcast(ctx, content)
            return

        # COMPOSER MODE: !alert (no message) - DM-only interactive workflow
        if not self.alert_manager.is_dm_channel(ctx.channel):
            await ctx.send(
                "🔒 Interactive alert composer can only be used in DMs. Use `!alert <message>` for direct broadcast."
            )
            return

        # Prevent concurrent alert sessions per user [REH][CA]
        existing = self.alert_manager.get_session(ctx.author.id)
        if existing is not None:
            await ctx.send(
                "⚠️ An alert session is already active. Use the composer or react with ❌ to cancel."
            )
            return

        try:
            session = await self.alert_manager.create_session(ctx.author.id)
            embed = await self.alert_manager.build_composer_embed(session)
            composer_msg = await ctx.send(embed=embed)

            session.composer_message_id = composer_msg.id

            # Add reaction controls with queuing
            reactions = ["📋", "✏️", "👁️", "📤", "❌"]
            for emoji in reactions:
                await self.alert_manager._queue_reaction_operation(
                    composer_msg, emoji, "add", ctx.author
                )

            # Mark composer as ready after all setup is complete
            session.composer_ready = True

            self.logger.info(f"🚀 Alert session started for user {ctx.author.id}")

        except Exception as e:
            self.logger.error(f"❌ Failed to create alert session: {e}")
            await ctx.send("❌ Failed to start alert session. Please try again.")

    async def _handle_direct_broadcast(self, ctx, content: str) -> None:
        """
        Direct broadcast mode: send message to all eligible guilds immediately.

        [REH] Non-fatal per-guild errors; continues to other guilds.
        [CA] Reuses existing permission checking logic.
        """
        self.logger.info(
            f"alert:direct_broadcast:start user_id={ctx.author.id} content_len={len(content)}"
        )

        # Sanitize broadcast content before sending [REH]
        content = sanitize_public_text(content)

        guilds_targeted = 0
        guilds_success = 0
        guilds_skipped = 0
        guilds_failed = 0

        # Hard recipient cap: MAX 20 sends per alert [CAP]
        MAX_RECIPIENTS = 20

        truncated_guilds = []
        for guild in self.bot.guilds:
            truncated_guilds.append(guild)

        if len(truncated_guilds) > MAX_RECIPIENTS:
            self.logger.warning(
                f"alert:direct_broadcast:recipient_cap guilds={len(truncated_guilds)} cap={MAX_RECIPIENTS}"
            )
            await ctx.send(
                f"⚠️ Alert recipient cap reached ({MAX_RECIPIENTS}). "
                f"Broadcast truncated from {len(truncated_guilds)} to {MAX_RECIPIENTS} recipients."
            )
            truncated_guilds = truncated_guilds[:MAX_RECIPIENTS]

        for guild in truncated_guilds:
            guilds_targeted += 1

            # Get bot member to check permissions
            bot_member = guild.get_member(self.bot.user.id)
            if not bot_member:
                guilds_skipped += 1
                self.logger.debug(
                    f"alert:skip guild_id={guild.id} reason=bot_not_member"
                )
                continue

            # Find target channel: system channel > first writable text channel
            target_channel = None

            # Priority 1: Guild's system channel if bot can send there
            if guild.system_channel:
                perms = guild.system_channel.permissions_for(bot_member)
                if perms.send_messages and perms.read_messages:
                    target_channel = guild.system_channel

            # Priority 2: First text channel bot can write to
            if not target_channel:
                for channel in guild.text_channels:
                    perms = channel.permissions_for(bot_member)
                    if perms.send_messages and perms.read_messages:
                        target_channel = channel
                        break

            if not target_channel:
                guilds_skipped += 1
                self.logger.debug(
                    f"alert:skip guild_id={guild.id} reason=no_writable_channel"
                )
                continue

            # Send alert to this guild [REH]
            try:
                await target_channel.send(content)
                guilds_success += 1
                self.logger.debug(
                    f"alert:sent guild_id={guild.id} channel_id={target_channel.id}"
                )
            except Exception as e:
                guilds_failed += 1
                self.logger.warning(
                    f"alert:failed guild_id={guild.id} channel_id={target_channel.id} error={e}"
                )

        # Log summary [REH]
        self.logger.info(
            f"alert:direct_broadcast:complete user_id={ctx.author.id} "
            f"targeted={guilds_targeted} success={guilds_success} "
            f"skipped={guilds_skipped} failed={guilds_failed}"
        )

        # Report to invoking user
        if guilds_success > 0:
            await ctx.send(
                f"✅ Alert sent to {guilds_success} server(s). "
                f"(Skipped: {guilds_skipped}, Failed: {guilds_failed})"
            )
        else:
            await ctx.send(
                f"⚠️ Alert could not be delivered to any servers. "
                f"(Skipped: {guilds_skipped}, Failed: {guilds_failed})"
            )

    @commands.Cog.listener()
    async def on_reaction_add(
        self, reaction: discord.Reaction, user: discord.User
    ) -> None:
        """Handle emoji reactions on composer cards."""
        if user.bot:
            return

        session = self.alert_manager.get_session(user.id)
        if not session:
            return

        # Allow reactions on composer, guild selection, or channel selection messages
        valid_message_ids = [
            session.composer_message_id,
            getattr(session, "selection_message_id", None),
            getattr(session, "channel_message_id", None),
        ]
        if reaction.message.id not in [m for m in valid_message_ids if m]:
            return

        # Ready gate: ignore reactions until composer is fully initialized (only for composer message)
        if (
            session.composer_message_id == reaction.message.id
            and not session.composer_ready
        ):
            try:
                await reaction.remove(user)
            except Exception:
                pass  # Ignore removal failures
            return

        emoji = str(reaction.emoji)

        try:
            if emoji == "📋":
                await self._handle_channel_selection(reaction, user, session)
            elif emoji == "✏️":
                await self._handle_content_composition(reaction, user, session)
            elif emoji == "👁️":
                await self._handle_preview(reaction, user, session)
            elif emoji == "📤":
                await self._handle_send_confirmation(reaction, user, session)
            elif emoji == "❌":
                await self._handle_cancel(reaction, user, session)

            # Guild selection navigation (on selection_message_id)
            elif reaction.message.id == getattr(session, "selection_message_id", None):
                channels = await self.alert_manager.get_accessible_channels()
                if not channels:
                    return

                guild_map: Dict[int, List[discord.TextChannel]] = {}
                for ch in channels:
                    gid = ch.guild.id
                    if gid not in guild_map:
                        guild_map[gid] = []
                    guild_map[gid].append(ch)
                for gid in guild_map:
                    guild_map[gid].sort(key=lambda c: (c.position, c.name.lower()))

                # Rebuild sorted_guilds from current guild_map
                sorted_guilds = sorted(
                    guild_map.keys(),
                    key=lambda g: next(
                        (c.guild.name.lower() for c in channels if c.guild.id == g), ""
                    ),
                )
                session.guilds_list = sorted_guilds

                if emoji == "⬆️" and session.guild_page > 0:
                    session.guild_page -= 1
                    await self._show_guild_selection(
                        user, session, guild_map, sorted_guilds
                    )
                elif emoji == "⬇️":
                    total_pages = (len(sorted_guilds) + 7) // 8
                    if session.guild_page < total_pages - 1:
                        session.guild_page += 1
                        await self._show_guild_selection(
                            user, session, guild_map, sorted_guilds
                        )
                try:
                    await reaction.remove(user)
                except discord.HTTPException:
                    pass  # Bot may lack MANAGE_MESSAGES permission; skip silently

            # Channel selection navigation (on channel_message_id)
            elif reaction.message.id == getattr(session, "channel_message_id", None):
                channels = await self.alert_manager.get_accessible_channels()
                guild_map: Dict[int, List[discord.TextChannel]] = {}
                for ch in channels:
                    gid = ch.guild.id
                    if gid not in guild_map:
                        guild_map[gid] = []
                    guild_map[gid].append(ch)
                for gid in guild_map:
                    guild_map[gid].sort(key=lambda c: (c.position, c.name.lower()))

                if emoji == "⬅️" and session.channel_page > 0:
                    session.channel_page -= 1
                    await self._show_channel_selection_for_guild(
                        user, session, session.selected_guild_id, guild_map
                    )
                elif emoji == "➡️":
                    guild_channels = guild_map.get(session.selected_guild_id, [])
                    total_pages = (len(guild_channels) + 9) // 10
                    if session.channel_page < total_pages - 1:
                        session.channel_page += 1
                        await self._show_channel_selection_for_guild(
                            user, session, session.selected_guild_id, guild_map
                        )
                elif emoji == "🏠":
                    # Turn the current channel-selection card back into the guild picker
                    # so the user sees the message they clicked actually change back.
                    session.selected_guild_id = None
                    session.channel_page = 0
                    session.selection_message_id = reaction.message.id
                    session.channel_message_id = None
                    try:
                        await reaction.message.clear_reactions()
                    except Exception:
                        pass
                    sorted_guilds = session.guilds_list if session.guilds_list else []
                    await self._show_guild_selection(
                        user, session, guild_map, sorted_guilds
                    )
                elif emoji == "❌":
                    await self._handle_cancel(reaction, user, session)
                try:
                    await reaction.remove(user)
                except discord.HTTPException:
                    pass  # Bot may lack MANAGE_MESSAGES permission; skip silently

        except discord.HTTPException as e:
            # Structured logging for 50035 diagnostics [REH]
            response_text = (
                getattr(e.response, "text", "N/A") if hasattr(e, "response") else "N/A"
            )
            self.logger.error(
                f"❌ Discord API error handling reaction {emoji}: status={e.status}, code={e.code}, response_length={len(str(response_text))}"
            )
            await user.send("❌ An error occurred. Please try again.")
        except Exception as e:
            self.logger.error(f"❌ Error handling reaction {emoji}: {e}")
            await user.send("❌ An error occurred. Please try again.")

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        """Handle DM replies during an active alert session.

        Supports:
        - Selecting channels by sending comma-separated indices (e.g., 1,3,5) when on step 'select_channels'.
        - Composing content and optional embed fields when on step 'compose_content'.
        """
        try:
            # Ignore bot messages
            if message.author.bot:
                return
            # Only process DMs/Groups
            if not self.alert_manager.is_dm_channel(message.channel):
                return

            # Only consider messages from admins with active sessions
            session = self.alert_manager.get_session(message.author.id)
            if not session:
                return
            if not self.alert_manager.is_admin_user(message.author.id):
                return

            content = (message.content or "").strip()
            if not content:
                return

            # STEP: Select Channels (guild-first navigation)
            if session.current_step == "select_channels":
                # Handle "done" message
                if content.lower() == "done":
                    if session.destinations:
                        session.current_step = "compose_content"
                        await message.channel.send(
                            f"✅ Using {len(session.destinations)} selected channel(s).\n\n"
                            f"React with ✏️ to compose content, or send your alert content now."
                        )
                        await self._update_composer_embed(message, session)
                    else:
                        await message.channel.send(
                            "❌ No channels selected yet. Reply with a guild number first."
                        )
                    return

                # Handle "back" message to return to guild list
                if content.lower() == "back":
                    session.selected_guild_id = None
                    session.channel_page = 0
                    channels = await self.alert_manager.get_accessible_channels()
                    if not channels:
                        await message.channel.send("❌ No channels found.")
                        return
                    # Rebuild guild map
                    guild_map: Dict[int, List[discord.TextChannel]] = {}
                    for ch in channels:
                        gid = ch.guild.id
                        if gid not in guild_map:
                            guild_map[gid] = []
                        guild_map[gid].append(ch)
                    for gid in guild_map:
                        guild_map[gid].sort(key=lambda c: (c.position, c.name.lower()))
                    # Reuse stored guilds_list
                    sorted_guilds = session.guilds_list if session.guilds_list else []
                    await self._show_guild_selection(
                        message.author, session, guild_map, sorted_guilds
                    )
                    return

                indices = self._extract_indices(content)
                if not indices:
                    await message.channel.send(
                        "⚠️ Please send numbers like `1,3,5` corresponding to the list, or type `back` to return."
                    )
                    return

                channels = await self.alert_manager.get_accessible_channels()
                if not channels:
                    await message.channel.send("❌ No accessible channels found.")
                    return

                # Build guild index
                guild_map: Dict[int, List[discord.TextChannel]] = {}
                for ch in channels:
                    gid = ch.guild.id
                    if gid not in guild_map:
                        guild_map[gid] = []
                    guild_map[gid].append(ch)

                for gid in guild_map:
                    guild_map[gid].sort(key=lambda c: (c.position, c.name.lower()))

                # Restore guilds_list if empty
                if not session.guilds_list:
                    sorted_guilds = sorted(
                        guild_map.keys(),
                        key=lambda g: next(
                            (c.guild.name.lower() for c in channels if c.guild.id == g),
                            "",
                        ),
                    )
                    session.guilds_list = sorted_guilds

                sorted_guilds = session.guilds_list

                # Case 1: No guild selected - interpret as guild selection
                if session.selected_guild_id is None:
                    if len(indices) != 1:
                        await message.channel.send(
                            f"⚠️ Please send **one** guild number (1-{len(sorted_guilds)})."
                        )
                        return

                    guild_idx = indices[0] - 1
                    if guild_idx < 0 or guild_idx >= len(sorted_guilds):
                        await message.channel.send(
                            f"❌ Invalid guild number. Please choose 1-{len(sorted_guilds)}."
                        )
                        return

                    selected_guild_id = sorted_guilds[guild_idx]
                    await self._show_channel_selection_for_guild(
                        message.author, session, selected_guild_id, guild_map
                    )
                    return

                # Case 2: Guild selected - interpret as channel selection
                guild_channels = guild_map.get(session.selected_guild_id, [])
                if not guild_channels:
                    await message.channel.send(
                        "❌ Could not find channels for that guild."
                    )
                    return

                selected: List[AlertDestination] = []
                invalid: List[int] = []

                for idx in indices:
                    if 1 <= idx <= len(guild_channels):
                        ch = guild_channels[idx - 1]
                        selected.append(
                            AlertDestination(
                                guild_id=ch.guild.id,
                                channel_id=ch.id,
                                channel_name=ch.name,
                                guild_name=ch.guild.name,
                            )
                        )
                    else:
                        invalid.append(idx)

                if not selected:
                    await message.channel.send(
                        f"❌ No valid selections. Please choose 1-{len(guild_channels)}."
                    )
                    return

                session.destinations.extend(selected)
                session.selected_guild_id = None  # Reset to allow multi-guild selection

                self.logger.info(
                    f"✅ User {message.author.id} selected {len(selected)} channel(s); total={len(session.destinations)}"
                )

                if invalid:
                    await message.channel.send(
                        f"⚠️ Ignored out-of-range: {', '.join(map(str, invalid))}"
                    )

                names = ", ".join([f"#{d.channel_name}" for d in selected])
                await message.channel.send(
                    f"✅ Added {len(selected)}: {names}\n\n"
                    f"Total selected: {len(session.destinations)} channel(s)\n"
                    f"Reply `back` for more guilds, `done` to finish selection."
                )
                await self._update_composer_embed(message, session)
                return

            # STEP: Compose Content
            if session.current_step == "compose_content":
                title, desc, body = self._parse_content_fields(content)
                # Update session fields if provided
                if title is not None:
                    session.embed_title = title
                if desc is not None:
                    session.embed_description = desc
                if body:
                    session.content = body

                # Acknowledge and guide next step
                parts = []
                if title is not None:
                    parts.append("title")
                if desc is not None:
                    parts.append("description")
                if body:
                    parts.append("content")
                changed = ", ".join(parts) if parts else "(no changes)"

                await message.channel.send(
                    f"✅ Updated {changed}. React with 👁️ to preview or 📤 to send."
                )

                # Keep step as compose until they preview/send
                await self._update_composer_embed(message, session)
                return

        except Exception as e:
            self.logger.error(f"❌ Error handling DM message: {e}")
            try:
                await message.channel.send(
                    "❌ Error processing your input. Please try again."
                )
            except Exception:
                pass

    def _extract_indices(self, text: str) -> List[int]:
        """Parse comma/space separated integers from user input."""
        indices: List[int] = []
        for token in text.replace("\n", ",").split(","):
            token = token.strip()
            if not token:
                continue
            try:
                indices.append(int(token))
            except ValueError:
                # allow space separated paths too
                for sub in token.split():
                    try:
                        indices.append(int(sub))
                    except ValueError:
                        continue
        return indices

    def _parse_content_fields(
        self, text: str
    ) -> tuple[Optional[str], Optional[str], str]:
        """Extract TITLE: and DESC: lines; return (title|None, desc|None, body_text)."""
        title: Optional[str] = None
        desc: Optional[str] = None
        body_lines: List[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("title:"):
                title = stripped.split(":", 1)[1].strip()
            elif stripped.lower().startswith("desc:"):
                desc = stripped.split(":", 1)[1].strip()
            else:
                body_lines.append(line)
        return title, desc, "\n".join(body_lines).strip()

    async def _update_composer_embed(
        self, source_message: discord.Message, session: AlertSession
    ) -> None:
        """Refresh the composer embed message based on current session state."""
        try:
            if not session.composer_message_id:
                return
            composer_embed = await self.alert_manager.build_composer_embed(session)
            composer_embed = self.alert_manager._validate_embed_limits(composer_embed)
            dm_msg = await source_message.channel.fetch_message(
                session.composer_message_id
            )
            await dm_msg.edit(embed=composer_embed)
        except discord.HTTPException as e:
            getattr(e.response, "text", "N/A") if hasattr(e, "response") else "N/A"
            self.logger.error(f"❌ Embed edit failed: status={e.status}, code={e.code}")
        except Exception as e:
            self.logger.error(f"❌ Failed to update composer embed: {e}")

    async def _handle_channel_selection(
        self, reaction: discord.Reaction, user: discord.User, session: "AlertSession"
    ) -> None:
        """Present guild-based channel selection with scrollable guild list."""
        session.current_step = "select_channels"

        channels = await self.alert_manager.get_accessible_channels()
        if not channels:
            await user.send(
                "❌ I couldn't find any text channels I can send messages to. "
                "Check the bot permissions and try again."
            )
            return

        # Build guild index from accessible channels
        guild_map: Dict[int, List[discord.TextChannel]] = {}
        for ch in channels:
            gid = ch.guild.id
            if gid not in guild_map:
                guild_map[gid] = []
            guild_map[gid].append(ch)

        # Sort channels within each guild by position
        for gid in guild_map:
            guild_map[gid].sort(key=lambda c: (c.position, c.name.lower()))

        # Sort guilds by name (case-insensitive)
        sorted_guilds = sorted(
            guild_map.keys(),
            key=lambda g: next(
                (c.guild.name.lower() for c in channels if c.guild.id == g), ""
            ),
        )
        session.guilds_list = sorted_guilds

        await self._show_guild_selection(user, session, guild_map, sorted_guilds)

        try:
            await reaction.remove(user)
        except discord.HTTPException:
            pass

    async def _show_guild_selection(
        self,
        user: discord.User,
        session: "AlertSession",
        guild_map: Dict[int, List[discord.TextChannel]],
        sorted_guilds: List[int],
    ) -> None:
        """Display paginated guild list with scroll indicators."""
        GUILDS_PER_PAGE = 8

        page = getattr(session, "guild_page", 0)
        total_pages = max(
            1, (len(sorted_guilds) + GUILDS_PER_PAGE - 1) // GUILDS_PER_PAGE
        )
        page = max(0, min(page, total_pages - 1))
        session.guild_page = page

        start_idx = page * GUILDS_PER_PAGE
        end_idx = min(start_idx + GUILDS_PER_PAGE, len(sorted_guilds))

        lines = []
        for i, gid in enumerate(sorted_guilds[start_idx:end_idx], start=start_idx + 1):
            g_channels = guild_map[gid]
            guild_name = g_channels[0].guild.name if g_channels else "Unknown"
            channel_count = len(g_channels)
            lines.append(f"`{i:>2}` {guild_name} ({channel_count} channels)")

        embed = discord.Embed(
            title="📋 Select a Guild",
            description="Use ⬆️/⬇️ to scroll, then reply with a guild number to browse its channels.",
            color=0x5865F2,
        )

        embed.add_field(
            name=f"Guilds (page {page + 1}/{total_pages})",
            value="\n".join(lines) if lines else "No guilds available",
            inline=False,
        )

        nav_text = "⬆️ Previous | ⬇️ Next | ❌ Cancel"
        embed.add_field(name="Navigation", value=nav_text, inline=False)
        embed.set_footer(
            text=f"Total: {len(sorted_guilds)} guilds | Reply with a number (1-{len(sorted_guilds)})"
        )

        # Edit existing message if available, otherwise send new
        existing_msg_id = getattr(session, "selection_message_id", None)
        if existing_msg_id:
            try:
                existing_msg = await user.fetch_message(existing_msg_id)
                await existing_msg.edit(embed=embed)
                return
            except discord.NotFound:
                pass  # Message was deleted, send new one
            except discord.HTTPException:
                pass  # Other error, send new one

        selection_msg = await user.send(embed=embed)
        session.selection_message_id = selection_msg.id

        if total_pages > 1:
            await selection_msg.add_reaction("⬆️")
            await selection_msg.add_reaction("⬇️")

    async def _show_channel_selection_for_guild(
        self,
        user: discord.User,
        session: "AlertSession",
        guild_id: int,
        guild_map: Dict[int, List[discord.TextChannel]],
    ) -> None:
        """Show channels within a selected guild with pagination."""
        CHANNELS_PER_PAGE = 10

        channels = guild_map.get(guild_id, [])
        if not channels:
            await user.send("❌ Could not find channels for that guild.")
            return

        guild_name = channels[0].guild.name if channels[0].guild else "Unknown Guild"

        page = getattr(session, "channel_page", 0)
        total_pages = max(
            1, (len(channels) + CHANNELS_PER_PAGE - 1) // CHANNELS_PER_PAGE
        )
        page = max(0, min(page, total_pages - 1))
        session.channel_page = page
        session.selected_guild_id = guild_id

        start_idx = page * CHANNELS_PER_PAGE
        end_idx = min(start_idx + CHANNELS_PER_PAGE, len(channels))

        lines = []
        for i, ch in enumerate(channels[start_idx:end_idx], start=start_idx + 1):
            lines.append(f"`{i:>2}` #{ch.name}")

        embed = discord.Embed(
            title=f"📋 Select Channels from {guild_name}",
            description="Reply with channel numbers (e.g., `1,3,5`) or use navigation.",
            color=0x5865F2,
        )

        embed.add_field(
            name=f"Channels (page {page + 1}/{total_pages})",
            value="\n".join(lines) if lines else "No channels",
            inline=False,
        )

        nav_parts = []
        if total_pages > 1:
            nav_parts.extend(["⬅️ Prev", "➡️ Next"])
        nav_parts.extend(["🏠 Back to guilds", "❌ Cancel"])
        embed.add_field(name="Navigation", value=" | ".join(nav_parts), inline=False)
        embed.set_footer(text="Or reply with channel numbers to select (e.g., 1,2,3)")

        msg = await user.send(embed=embed)
        session.channel_message_id = msg.id

        if total_pages > 1:
            await msg.add_reaction("⬅️")
            await msg.add_reaction("➡️")
        await msg.add_reaction("🏠")
        await msg.add_reaction("❌")

    async def _handle_content_composition(
        self, reaction: discord.Reaction, user: discord.User, session: "AlertSession"
    ) -> None:
        session.current_step = "compose_content"

        await user.send(
            "✏️ **Step 3: Compose Content**\n\n"
            "Reply with your alert content. You can include:\n"
            "• Message text\n"
            "• Embed title (prefix with `TITLE: `)\n"
            "• Embed description (prefix with `DESC: `)\n\n"
            "Example:\n"
            "```\n"
            "TITLE: Server Maintenance\n"
            "DESC: Scheduled maintenance tonight\n"
            "Please save your work.\n"
            "```"
        )

        # Update composer (embed-only, no components from reaction)
        try:
            composer_embed = await self.alert_manager.build_composer_embed(session)
            composer_embed = self.alert_manager._validate_embed_limits(composer_embed)
            # Fetch full message to avoid partial message edit failures [REH]
            full_message = await reaction.message.channel.fetch_message(
                reaction.message.id
            )
            await full_message.edit(embed=composer_embed)
        except discord.HTTPException as e:
            self.logger.error(
                f"❌ Failed to update composer embed in content composition: status={e.status}, code={e.code}"
            )
            raise

    async def _handle_preview(
        self, reaction: discord.Reaction, user: discord.User, session: "AlertSession"
    ) -> None:
        if not session.destinations:
            await user.send("❌ Please select destinations first (📋).")
            return

        if not session.content and not session.embed_title:
            await user.send("❌ Please compose content first (✏️).")
            return

        session.current_step = "preview_alert"

        preview_embed = discord.Embed(
            title="👁️ Alert Preview",
            description="This is how your alert will appear:",
            color=0x5865F2,
        )

        dest_list = [
            f"• #{dest.channel_name or 'unknown-channel'}"
            for dest in session.destinations[:10]
        ]
        if len(session.destinations) > 10:
            dest_list.append(f"• ... and {len(session.destinations) - 10} more")

        preview_embed.add_field(
            name=f"📋 Destinations ({len(session.destinations)})",
            value="\n".join(dest_list),
            inline=False,
        )

        await user.send(embed=preview_embed)

        # Show preview
        if session.embed_title or session.embed_description:
            alert_embed = discord.Embed(
                title=session.embed_title,
                description=session.embed_description,
                color=0x1F8B4C,
            )
            await user.send(
                content=f"**PREVIEW:** {session.content}", embed=alert_embed
            )
        else:
            await user.send(f"**PREVIEW:** {session.content}")

        session.current_step = "confirm_send"
        try:
            composer_embed = await self.alert_manager.build_composer_embed(session)
            composer_embed = self.alert_manager._validate_embed_limits(composer_embed)
            # Fetch full message to avoid partial message edit failures [REH]
            full_message = await reaction.message.channel.fetch_message(
                reaction.message.id
            )
            await full_message.edit(embed=composer_embed)
        except discord.HTTPException as e:
            self.logger.error(
                f"❌ Failed to update composer embed in preview: status={e.status}, code={e.code}"
            )
            raise

    async def _handle_send_confirmation(
        self, reaction: discord.Reaction, user: discord.User, session: "AlertSession"
    ) -> None:
        if session.current_step != "confirm_send":
            await user.send("❌ Please complete all steps before sending.")
            return

        confirm_embed = discord.Embed(
            title="⚠️ Final Confirmation",
            description=f"Send alert to **{len(session.destinations)} channels**?",
            color=0xFF9500,
        )

        confirm_message = await user.send(embed=confirm_embed)
        await confirm_message.add_reaction("✅")
        await confirm_message.add_reaction("❌")

        def check(reaction_check, user_check):
            return (
                user_check == user
                and reaction_check.message.id == confirm_message.id
                and str(reaction_check.emoji) in ["✅", "❌"]
            )

        try:
            reaction_result, _ = await self.bot.wait_for(
                "reaction_add", timeout=60.0, check=check
            )

            if str(reaction_result.emoji) == "✅":
                await user.send("📤 Sending alert...")
                results = await self.alert_manager.send_alert(session)

                result_embed = discord.Embed(
                    title="📋 Alert Send Results",
                    color=0x00FF00 if results["failed_sends"] == 0 else 0xFF9500,
                )

                result_embed.add_field(
                    name="📊 Summary",
                    value=f"✅ Successful: {results['successful_sends']}\n❌ Failed: {results['failed_sends']}",
                    inline=False,
                )

                await user.send(embed=result_embed)
                del self.alert_manager.sessions[user.id]

            else:
                await user.send("❌ Alert send cancelled.")

        except asyncio.TimeoutError:
            await user.send("⏰ Confirmation timeout. Alert cancelled.")

    async def _handle_cancel(
        self, reaction: discord.Reaction, user: discord.User, session: "AlertSession"
    ) -> None:
        session.status = AlertSessionStatus.CANCELLED
        del self.alert_manager.sessions[user.id]
        await user.send("❌ Alert session cancelled.")


async def setup(bot) -> None:
    await bot.add_cog(AdminAlertCommands(bot))
