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


class AdminAlertManager:
    def __init__(self, bot):
        self.bot = bot
        self.config = load_config()
        self.logger = get_logger(f"{__name__}.AdminAlertManager")
        self.sessions: Dict[int, AlertSession] = {}
        self.reaction_queues: Dict[int, List] = {}  # Per-message reaction queues

        self.enabled = self.config.get("ALERT_ENABLE", "false").lower() == "true"
        self.admin_user_ids = self._parse_admin_user_ids()
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

    async def _process_reaction_queue(self, message_id: int):
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

        # Send to each destination
        for dest in session.destinations:
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
    async def alert_command(self, ctx):
        """Start an admin alert composition session (DM-only)."""
        if not self.alert_manager.enabled:
            await ctx.send("❌ Alert system is disabled.")
            return

        if not self.alert_manager.is_dm_channel(ctx.channel):
            await ctx.send("🔒 Alert command can only be used in DMs for security.")
            return

        if not self.alert_manager.is_admin_user(ctx.author.id):
            await ctx.send(
                "🚫 Access denied. You are not authorized to use the alert system."
            )
            self.logger.warning(f"🚫 Unauthorized alert access: {ctx.author.id}")
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
            message = await ctx.send(embed=embed)

            session.composer_message_id = message.id

            # Add reaction controls with queuing
            reactions = ["📋", "✏️", "👁️", "📤", "❌"]
            for emoji in reactions:
                await self.alert_manager._queue_reaction_operation(
                    message, emoji, "add", ctx.author
                )

            # Mark composer as ready after all setup is complete
            session.composer_ready = True

            self.logger.info(f"🚀 Alert session started for user {ctx.author.id}")

        except Exception as e:
            self.logger.error(f"❌ Failed to create alert session: {e}")
            await ctx.send("❌ Failed to start alert session. Please try again.")

    @commands.Cog.listener()
    async def on_reaction_add(self, reaction, user):
        """Handle emoji reactions on composer cards."""
        if user.bot:
            return

        session = self.alert_manager.get_session(user.id)
        if not session or session.composer_message_id != reaction.message.id:
            return

        # Ready gate: ignore reactions until composer is fully initialized
        if not session.composer_ready:
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
    async def on_message(self, message: discord.Message):
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

            # STEP: Select Channels
            if session.current_step == "select_channels":
                indices = self._extract_indices(content)
                if not indices:
                    await message.channel.send(
                        "⚠ Please send numbers like `1,3,5` corresponding to the list."
                    )
                    return

                channels = await self.alert_manager.get_accessible_channels()
                max_count = min(20, len(channels))
                selected: List[AlertDestination] = []
                invalid: List[int] = []

                for idx in indices:
                    if 1 <= idx <= max_count:
                        ch = channels[idx - 1]
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
                        "❌ No valid selections. Please choose indices from the provided list."
                    )
                    return

                session.destinations = selected
                session.current_step = "compose_content"
                self.logger.info(
                    f"✅ User {message.author.id} selected {len(selected)} destination(s); invalid={invalid}"
                )

                if invalid:
                    await message.channel.send(
                        f"⚠ Ignored out-of-range indices: {', '.join(map(str, invalid))}"
                    )

                names = ", ".join(
                    [
                        f"#{d.channel_name or 'unknown-channel'} "
                        f"({d.guild_name or 'Unknown Guild'})"
                        for d in selected
                    ]
                )
                await message.channel.send(
                    f"✅ Destinations set: {names}\n\nNow send your alert content, and optionally include `TITLE:` and `DESC:` lines, or react with ✏️."
                )

                # Update composer embed
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

    async def _handle_channel_selection(self, reaction, user, session):
        """Present the admin with a numbered list of accessible channels."""
        session.current_step = "select_channels"

        channels = await self.alert_manager.get_accessible_channels()
        if not channels:
            await user.send(
                "❌ I couldn't find any text channels I can send messages to. "
                "Check the bot permissions and try again."
            )
            return

        max_count = min(20, len(channels))
        lines = []
        for idx, channel in enumerate(channels[:max_count], start=1):
            guild_name = channel.guild.name if channel.guild else "Unknown Guild"
            lines.append(
                f"`{idx}` • {channel.mention} ({guild_name})"
            )

        embed = discord.Embed(
            title="📋 Select Alert Destinations",
            description=(
                "Reply with the numbers of the channels you want to alert. "
                "Use commas or spaces, for example `1,3,5`."
            ),
            color=0x5865F2,
        )

        embed.add_field(
            name=(
                f"Available Channels (showing {max_count} of {len(channels)})"
                if len(channels) > max_count
                else "Available Channels"
            ),
            value="\n".join(lines),
            inline=False,
        )

        if len(channels) > max_count:
            embed.add_field(
                name="ℹ️ Tip",
                value=(
                    "Only the first 20 channels are shown. Adjust the bot's "
                    "permissions to narrow the list if needed."
                ),
                inline=False,
            )

        embed.set_footer(
            text="You can react with ❌ at any time to cancel the alert session."
        )

        await user.send(embed=embed)

        try:
            composer_embed = await self.alert_manager.build_composer_embed(session)
            composer_embed = self.alert_manager._validate_embed_limits(composer_embed)
            await reaction.message.edit(embed=composer_embed, view=None)
        except discord.HTTPException as e:
            self.logger.error(
                f"❌ Failed to update composer embed during channel selection: "
                f"status={e.status}, code={e.code}"
            )
            raise

        try:
            await reaction.remove(user)
        except discord.HTTPException:
            # Ignore failures (e.g. missing permissions); user can still remove manually
            pass

    async def _handle_content_composition(self, reaction, user, session):
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
            await reaction.message.edit(
                embed=composer_embed, view=None
            )  # Explicitly remove components
        except discord.HTTPException as e:
            self.logger.error(
                f"❌ Failed to update composer embed in content composition: status={e.status}, code={e.code}"
            )
            raise

    async def _handle_preview(self, reaction, user, session):
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
            await reaction.message.edit(
                embed=composer_embed, view=None
            )  # Explicitly remove components
        except discord.HTTPException as e:
            self.logger.error(
                f"❌ Failed to update composer embed in preview: status={e.status}, code={e.code}"
            )
            raise

    async def _handle_send_confirmation(self, reaction, user, session):
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

    async def _handle_cancel(self, reaction, user, session):
        session.status = AlertSessionStatus.CANCELLED
        del self.alert_manager.sessions[user.id]
        await user.send("❌ Alert session cancelled.")


async def setup(bot):
    await bot.add_cog(AdminAlertCommands(bot))
