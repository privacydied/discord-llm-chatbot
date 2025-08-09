"""
Admin DM Alert System - Secure, DM-only broadcast alerting with emoji-driven composer.
"""

import asyncio
import time
from typing import Dict, List, Optional, Set, Any, Union
from dataclasses import dataclass, field
from enum import Enum

import discord
from discord.ext import commands

from bot.config import load_config
from bot.util.logging import get_logger

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
    channel_id: int
    channel_name: str
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


class AdminAlertManager:
    def __init__(self, bot):
        self.bot = bot
        self.config = load_config()
        self.logger = get_logger(f"{__name__}.AdminAlertManager")
        self.sessions: Dict[int, AlertSession] = {}
        
        self.enabled = self.config.get('ALERT_ENABLE', 'false').lower() == 'true'
        self.admin_user_ids = self._parse_admin_user_ids()
        self.session_timeout = int(self.config.get('ALERT_SESSION_TIMEOUT_S', '1800'))
        
        self.logger.info(f"🚨 Admin alert system initialized: enabled={self.enabled}")
    
    def _parse_admin_user_ids(self) -> Set[int]:
        try:
            admin_ids_str = self.config.get('ALERT_ADMIN_USER_IDS', '')
            if not admin_ids_str:
                return set()
            
            admin_ids = set()
            for id_str in admin_ids_str.split(','):
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
            expires_at=now + self.session_timeout
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
    
    async def build_composer_embed(self, session: AlertSession) -> discord.Embed:
        embed = discord.Embed(
            title="🚨 Admin Alert Composer",
            color=0x1f8b4c,
            timestamp=discord.utils.utcnow()
        )
        
        step_indicators = {
            "select_channels": "📋 **1. Select Channels**",
            "compose_content": "✏️ **2. Compose Content**", 
            "preview_alert": "👁️ **3. Preview & Send**",
            "confirm_send": "📤 **4. Confirm Send**"
        }
        
        current_step = step_indicators.get(session.current_step, "❓ Unknown")
        embed.add_field(name="Current Step", value=current_step, inline=False)
        
        if session.destinations:
            dest_text = []
            for dest in session.destinations[:5]:
                status_emoji = "✅" if dest.permissions_valid else "⚠️"
                dest_text.append(f"{status_emoji} #{dest.channel_name}")
            
            if len(session.destinations) > 5:
                dest_text.append(f"... and {len(session.destinations) - 5} more")
            
            embed.add_field(
                name=f"📋 Destinations ({len(session.destinations)})",
                value="\n".join(dest_text) if dest_text else "None selected",
                inline=True
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
                inline=True
            )
        
        time_remaining = int(session.expires_at - time.time())
        embed.add_field(
            name="⏰ Session Info",
            value=f"Expires in: {time_remaining // 60}m {time_remaining % 60}s",
            inline=True
        )
        
        if session.current_step == "select_channels":
            embed.description = "React with 📋 to select channels for broadcasting."
        elif session.current_step == "compose_content":
            embed.description = "React with ✏️ to compose your alert content."
        elif session.current_step == "preview_alert":
            embed.description = "React with 👁️ to preview your alert before sending."
        elif session.current_step == "confirm_send":
            embed.description = "⚠️ **Final confirmation required** - React with 📤 to send alert."
        
        embed.set_footer(text=f"Session: {session.session_id}")
        return embed
    
    async def get_accessible_channels(self) -> List[discord.TextChannel]:
        accessible_channels = []
        
        for guild in self.bot.guilds:
            for channel in guild.channels:
                if isinstance(channel, discord.TextChannel):
                    perms = channel.permissions_for(guild.me)
                    if perms.send_messages and perms.read_messages:
                        accessible_channels.append(channel)
        
        return accessible_channels
    
    async def send_alert(self, session: AlertSession) -> Dict[str, Any]:
        self.logger.info(f"📤 Sending alert from session {session.session_id}")
        session.status = AlertSessionStatus.POSTING
        
        results = {
            "total_destinations": len(session.destinations),
            "successful_sends": 0,
            "failed_sends": 0,
            "send_results": []
        }
        
        # Build alert content
        alert_content = session.content
        embed = None
        
        if session.embed_title or session.embed_description:
            embed = discord.Embed(
                title=session.embed_title,
                description=session.embed_description,
                color=0x1f8b4c,
                timestamp=discord.utils.utcnow()
            )
            embed.set_footer(text="Admin Alert")
        
        # Send to each destination
        for dest in session.destinations:
            try:
                channel = self.bot.get_channel(dest.channel_id)
                if not channel:
                    results["send_results"].append({
                        "channel_id": dest.channel_id,
                        "success": False,
                        "error": "Channel not found"
                    })
                    results["failed_sends"] += 1
                    continue
                
                if embed:
                    message = await channel.send(content=alert_content, embed=embed)
                else:
                    message = await channel.send(content=alert_content)
                
                results["send_results"].append({
                    "channel_id": dest.channel_id,
                    "success": True,
                    "message_id": message.id
                })
                results["successful_sends"] += 1
                
            except Exception as e:
                self.logger.error(f"❌ Failed to send to {dest.channel_name}: {e}")
                results["send_results"].append({
                    "channel_id": dest.channel_id,
                    "success": False,
                    "error": str(e)
                })
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
    
    @commands.command(name='alert')
    async def alert_command(self, ctx):
        """Start an admin alert composition session (DM-only)."""
        if not self.alert_manager.enabled:
            await ctx.send("❌ Alert system is disabled.")
            return
        
        if not self.alert_manager.is_dm_channel(ctx.channel):
            await ctx.send("🔒 Alert command can only be used in DMs for security.")
            return
        
        if not self.alert_manager.is_admin_user(ctx.author.id):
            await ctx.send("🚫 Access denied. You are not authorized to use the alert system.")
            self.logger.warning(f"🚫 Unauthorized alert access: {ctx.author.id}")
            return
        
        # Prevent concurrent alert sessions per user [REH][CA]
        existing = self.alert_manager.get_session(ctx.author.id)
        if existing is not None:
            await ctx.send("⚠️ An alert session is already active. Use the composer or react with ❌ to cancel.")
            return
        
        try:
            session = await self.alert_manager.create_session(ctx.author.id)
            embed = await self.alert_manager.build_composer_embed(session)
            message = await ctx.send(embed=embed)
            
            session.composer_message_id = message.id
            
            # Add reaction controls
            reactions = ['📋', '✏️', '👁️', '📤', '❌']
            for emoji in reactions:
                await message.add_reaction(emoji)
            
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
        
        emoji = str(reaction.emoji)
        
        try:
            if emoji == '📋':
                await self._handle_channel_selection(reaction, user, session)
            elif emoji == '✏️':
                await self._handle_content_composition(reaction, user, session)
            elif emoji == '👁️':
                await self._handle_preview(reaction, user, session)
            elif emoji == '📤':
                await self._handle_send_confirmation(reaction, user, session)
            elif emoji == '❌':
                await self._handle_cancel(reaction, user, session)
            
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
                    await message.channel.send("⚠ Please send numbers like `1,3,5` corresponding to the list.")
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
                                channel_id=ch.id,
                                channel_name=f"#{ch.name}",
                                guild_name=ch.guild.name,
                            )
                        )
                    else:
                        invalid.append(idx)

                if not selected:
                    await message.channel.send("❌ No valid selections. Please choose indices from the provided list.")
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

                names = ", ".join([f"{d.channel_name} ({d.guild_name})" for d in selected])
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
                await message.channel.send("❌ Error processing your input. Please try again.")
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

    def _parse_content_fields(self, text: str) -> tuple[Optional[str], Optional[str], str]:
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

    async def _update_composer_embed(self, source_message: discord.Message, session: AlertSession) -> None:
        """Refresh the composer embed message based on current session state."""
        try:
            if not session.composer_message_id:
                return
            composer_embed = await self.alert_manager.build_composer_embed(session)
            dm_msg = await source_message.channel.fetch_message(session.composer_message_id)
            await dm_msg.edit(embed=composer_embed)
        except Exception as e:
            self.logger.error(f"❌ Failed to update composer embed: {e}")
    
    async def _handle_channel_selection(self, reaction, user, session):
        session.current_step = "select_channels"
        
        channels = await self.alert_manager.get_accessible_channels()
        
        if not channels:
            await user.send("❌ No accessible channels found.")
            return
        
        embed = discord.Embed(
            title="📋 Select Alert Destinations",
            description="Available channels:",
            color=0x5865f2
        )
        
        channel_text = []
        for i, channel in enumerate(channels[:20]):  # Limit to 20
            channel_text.append(f"{i+1}. #{channel.name} ({channel.guild.name})")
        
        embed.add_field(
            name="Channels",
            value="\n".join(channel_text),
            inline=False
        )
        
        await user.send(embed=embed)
        await user.send("Send the numbers of channels you want to alert (e.g., `1,3,5`):")
        
        # Update composer
        composer_embed = await self.alert_manager.build_composer_embed(session)
        await reaction.message.edit(embed=composer_embed)
    
    async def _handle_content_composition(self, reaction, user, session):
        session.current_step = "compose_content"
        
        await user.send(
            "✏️ **Compose Alert Content**\n\n"
            "Please provide your alert content. You can include:\n"
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
        
        composer_embed = await self.alert_manager.build_composer_embed(session)
        await reaction.message.edit(embed=composer_embed)
    
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
            color=0x5865f2
        )
        
        dest_list = [f"• #{dest.channel_name}" for dest in session.destinations[:10]]
        if len(session.destinations) > 10:
            dest_list.append(f"• ... and {len(session.destinations) - 10} more")
        
        preview_embed.add_field(
            name=f"📋 Destinations ({len(session.destinations)})",
            value="\n".join(dest_list),
            inline=False
        )
        
        await user.send(embed=preview_embed)
        
        # Show preview
        if session.embed_title or session.embed_description:
            alert_embed = discord.Embed(
                title=session.embed_title,
                description=session.embed_description,
                color=0x1f8b4c
            )
            await user.send(content=f"**PREVIEW:** {session.content}", embed=alert_embed)
        else:
            await user.send(f"**PREVIEW:** {session.content}")
        
        session.current_step = "confirm_send"
        composer_embed = await self.alert_manager.build_composer_embed(session)
        await reaction.message.edit(embed=composer_embed)
    
    async def _handle_send_confirmation(self, reaction, user, session):
        if session.current_step != "confirm_send":
            await user.send("❌ Please complete all steps before sending.")
            return
        
        confirm_embed = discord.Embed(
            title="⚠️ Final Confirmation",
            description=f"Send alert to **{len(session.destinations)} channels**?",
            color=0xff9500
        )
        
        confirm_message = await user.send(embed=confirm_embed)
        await confirm_message.add_reaction('✅')
        await confirm_message.add_reaction('❌')
        
        def check(reaction_check, user_check):
            return (user_check == user and 
                   reaction_check.message.id == confirm_message.id and
                   str(reaction_check.emoji) in ['✅', '❌'])
        
        try:
            reaction_result, _ = await self.bot.wait_for('reaction_add', timeout=60.0, check=check)
            
            if str(reaction_result.emoji) == '✅':
                await user.send("📤 Sending alert...")
                results = await self.alert_manager.send_alert(session)
                
                result_embed = discord.Embed(
                    title="📋 Alert Send Results",
                    color=0x00ff00 if results["failed_sends"] == 0 else 0xff9500
                )
                
                result_embed.add_field(
                    name="📊 Summary",
                    value=f"✅ Successful: {results['successful_sends']}\n❌ Failed: {results['failed_sends']}",
                    inline=False
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
