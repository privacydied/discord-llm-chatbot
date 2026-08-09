"""Admin DM Alert System - Secure, DM-only broadcast alerting with emoji-driven composer.

This module contains ONLY the Discord command cog and event handlers.
Business logic is in admin_alert_manager.py, models in admin_alert_models.py.
"""

import contextlib

import discord
from discord.ext import commands

from bot.commands.admin_alert_manager import AdminAlertManager
from bot.commands.admin_alert_models import AlertDestination, AlertSessionStatus
from bot.config import load_config
from bot.core.output import safe_send
from bot.public_output import sanitize_public_text
from bot.utils.logging import get_logger

logger = get_logger(__name__)


class AdminAlertCommands(commands.Cog):
    def __init__(self, bot) -> None:
        self.bot = bot
        self.config = load_config()
        self.logger = get_logger(f"{__name__}.AdminAlertCommands")
        self.alert_manager = AdminAlertManager(bot)
        self.logger.info("Admin Alert Commands loaded")

    @commands.command(name="alert")
    @commands.cooldown(2, 300, type=commands.BucketType.user)
    async def alert_command(self, ctx: commands.Context, *, message: str | None = None) -> None:
        """Admin broadcast command.

        Usage:
          !alert <message>   - Direct broadcast to all servers
          !alert             - Start interactive composer session (DM-only)
        """
        self.alert_manager.refresh_config()

        # CRITICAL: Admin check BEFORE any routing or side effects [REH][SFT]
        if not self.alert_manager.is_admin_user(ctx.author.id):
            self.logger.warning(f"alert:unauthorized user_id={ctx.author.id} channel_type={'DM' if self.alert_manager.is_dm_channel(ctx.channel) else 'guild'}")
            await ctx.send("Access denied. You are not authorized to use the alert system.")
            return

        if not self.alert_manager.enabled:
            await ctx.send("Alert system is disabled.")
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
            await ctx.send("Interactive alert composer can only be used in DMs. Use `!alert <message>` for direct broadcast.")
            return

        # Prevent concurrent alert sessions per user [REH][CA]
        existing = self.alert_manager.get_session(ctx.author.id)
        if existing is not None:
            await ctx.send("An alert session is already active. Use the composer or react with X to cancel.")
            return

        try:
            session = await self.alert_manager.create_session(ctx.author.id)
            embed = await self.alert_manager.build_composer_embed(session)
            composer_msg = await ctx.send(embed=embed)

            session.composer_message_id = composer_msg.id

            # Add reaction controls with queuing
            reactions = ["📋", "✏️", "👁️", "📤", "❌"]
            for emoji in reactions:
                await self.alert_manager._queue_reaction_operation(composer_msg, emoji, "add", ctx.author)

            # Mark composer as ready after all setup is complete
            session.composer_ready = True

            self.logger.info(f"Alert session started for user {ctx.author.id}")

        except Exception as e:
            self.logger.exception(f"Failed to create alert session: {e}")
            await ctx.send("Failed to start alert session. Please try again.")

    async def _handle_direct_broadcast(self, ctx, content: str) -> None:
        """Direct broadcast mode: send message to all eligible guilds immediately.

        [REH] Non-fatal per-guild errors; continues to other guilds.
        [CA] Reuses existing permission checking logic.
        """
        self.logger.info(f"alert:direct_broadcast:start user_id={ctx.author.id} content_len={len(content)}")

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
            self.logger.warning(f"alert:direct_broadcast:recipient_cap guilds={len(truncated_guilds)} cap={MAX_RECIPIENTS}")
            await ctx.send(f"Alert recipient cap reached ({MAX_RECIPIENTS}). Broadcast truncated from {len(truncated_guilds)} to {MAX_RECIPIENTS} recipients.")
            truncated_guilds = truncated_guilds[:MAX_RECIPIENTS]

        for guild in truncated_guilds:
            guilds_targeted += 1

            # Get bot member to check permissions
            bot_member = guild.get_member(self.bot.user.id)
            if not bot_member:
                guilds_skipped += 1
                self.logger.debug(f"alert:skip guild_id={guild.id} reason=bot_not_member")
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
                self.logger.debug(f"alert:skip guild_id={guild.id} reason=no_writable_channel")
                continue

            # Send alert to this guild [REH]
            try:
                await safe_send(target_channel, content)
                guilds_success += 1
                self.logger.debug(f"alert:sent guild_id={guild.id} channel_id={target_channel.id}")
            except (discord.HTTPException, discord.Forbidden, discord.NotFound) as e:
                guilds_failed += 1
                self.logger.warning(f"alert:failed guild_id={guild.id} channel_id={target_channel.id} error={e}")

        # Log summary [REH]
        self.logger.info(f"alert:direct_broadcast:complete user_id={ctx.author.id} targeted={guilds_targeted} success={guilds_success} skipped={guilds_skipped} failed={guilds_failed}")

        # Report to invoking user
        if guilds_success > 0:
            await ctx.send(f"Alert sent to {guilds_success} server(s). (Skipped: {guilds_skipped}, Failed: {guilds_failed})")
        else:
            await ctx.send(f"Alert could not be delivered to any servers. (Skipped: {guilds_skipped}, Failed: {guilds_failed})")

    @commands.Cog.listener()
    async def on_reaction_add(self, reaction: discord.Reaction, user: discord.User) -> None:
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

        # Ready gate: ignore reactions until composer is fully initialized
        if session.composer_message_id == reaction.message.id and not session.composer_ready:
            try:
                await reaction.remove(user)
            except (discord.HTTPException, discord.NotFound, discord.Forbidden):
                self.logger.debug("Failed to remove reaction", exc_info=True)
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
            if reaction.message.id == getattr(session, "selection_message_id", None):
                channels = await self.alert_manager.get_accessible_channels()
                if not channels:
                    return

                guild_map: dict[int, list[discord.TextChannel]] = {}
                for ch in channels:
                    gid = ch.guild.id
                    if gid not in guild_map:
                        guild_map[gid] = []
                    guild_map[gid].append(ch)
                for gid in guild_map:
                    guild_map[gid].sort(key=lambda c: (c.position, c.name.lower()))

                sorted_guilds = sorted(
                    guild_map.keys(),
                    key=lambda g: next((c.guild.name.lower() for c in channels if c.guild.id == g), ""),
                )
                session.guilds_list = sorted_guilds

                if emoji == "⬆️" and session.guild_page > 0:
                    session.guild_page -= 1
                    await self._show_guild_selection(user, session, guild_map, sorted_guilds)
                elif emoji == "⬇️":
                    total_pages = (len(sorted_guilds) + 7) // 8
                    if session.guild_page < total_pages - 1:
                        session.guild_page += 1
                        await self._show_guild_selection(user, session, guild_map, sorted_guilds)
                with contextlib.suppress(discord.HTTPException):
                    await reaction.remove(user)

            # Channel selection navigation (on channel_message_id)
            if reaction.message.id == getattr(session, "channel_message_id", None):
                channels = await self.alert_manager.get_accessible_channels()
                guild_map: dict[int, list[discord.TextChannel]] = {}
                for ch in channels:
                    gid = ch.guild.id
                    if gid not in guild_map:
                        guild_map[gid] = []
                    guild_map[gid].append(ch)
                for gid in guild_map:
                    guild_map[gid].sort(key=lambda c: (c.position, c.name.lower()))

                if emoji == "⬅️" and session.channel_page > 0:
                    session.channel_page -= 1
                    await self._show_channel_selection_for_guild(user, session, session.selected_guild_id, guild_map)
                elif emoji == "➡️":
                    guild_channels = guild_map.get(session.selected_guild_id, [])
                    total_pages = (len(guild_channels) + 9) // 10
                    if session.channel_page < total_pages - 1:
                        session.channel_page += 1
                        await self._show_channel_selection_for_guild(user, session, session.selected_guild_id, guild_map)
                elif emoji == "🏠":
                    session.selected_guild_id = None
                    session.channel_page = 0
                    session.selection_message_id = reaction.message.id
                    session.channel_message_id = None
                    try:
                        await reaction.message.clear_reactions()
                    except (discord.HTTPException, discord.NotFound, discord.Forbidden):
                        self.logger.debug("Failed to clear reactions", exc_info=True)
                    sorted_guilds = session.guilds_list or []
                    await self._show_guild_selection(user, session, guild_map, sorted_guilds)
                elif emoji == "❌":
                    await self._handle_cancel(reaction, user, session)
                with contextlib.suppress(discord.HTTPException):
                    await reaction.remove(user)

        except discord.HTTPException as e:
            self.logger.exception(f"Discord API error handling reaction {emoji}: status={e.status}, code={e.code}")
            await user.send("An error occurred. Please try again.")
        except Exception as e:
            self.logger.exception(f"Error handling reaction {emoji}: {e}")
            await user.send("An error occurred. Please try again.")

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        """Handle DM replies during an active alert session."""
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
                if content.lower() == "done":
                    if session.destinations:
                        session.current_step = "compose_content"
                        await message.channel.send(f"Using {len(session.destinations)} selected channel(s).\n\nReact to compose content, or send your alert content now.")
                        await self._update_composer_embed(message, session)
                    else:
                        await message.channel.send("No channels selected yet. Reply with a guild number first.")
                    return

                if content.lower() == "back":
                    session.selected_guild_id = None
                    session.channel_page = 0
                    channels = await self.alert_manager.get_accessible_channels()
                    if not channels:
                        await message.channel.send("No channels found.")
                        return
                    guild_map: dict[int, list[discord.TextChannel]] = {}
                    for ch in channels:
                        gid = ch.guild.id
                        if gid not in guild_map:
                            guild_map[gid] = []
                        guild_map[gid].append(ch)
                    for gid in guild_map:
                        guild_map[gid].sort(key=lambda c: (c.position, c.name.lower()))
                    sorted_guilds = session.guilds_list or []
                    await self._show_guild_selection(message.author, session, guild_map, sorted_guilds)
                    return

                indices = self._extract_indices(content)
                if not indices:
                    await message.channel.send("Please send numbers like `1,3,5` corresponding to the list, or type `back` to return.")
                    return

                channels = await self.alert_manager.get_accessible_channels()
                if not channels:
                    await message.channel.send("No accessible channels found.")
                    return

                guild_map: dict[int, list[discord.TextChannel]] = {}
                for ch in channels:
                    gid = ch.guild.id
                    if gid not in guild_map:
                        guild_map[gid] = []
                    guild_map[gid].append(ch)

                for gid in guild_map:
                    guild_map[gid].sort(key=lambda c: (c.position, c.name.lower()))

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

                # Case 1: No guild selected
                if session.selected_guild_id is None:
                    if len(indices) != 1:
                        await message.channel.send(f"Please send **one** guild number (1-{len(sorted_guilds)}).")
                        return

                    guild_idx = indices[0] - 1
                    if guild_idx < 0 or guild_idx >= len(sorted_guilds):
                        await message.channel.send(f"Invalid guild number. Please choose 1-{len(sorted_guilds)}.")
                        return

                    selected_guild_id = sorted_guilds[guild_idx]
                    await self._show_channel_selection_for_guild(message.author, session, selected_guild_id, guild_map)
                    return

                # Case 2: Guild selected - channel selection
                guild_channels = guild_map.get(session.selected_guild_id, [])
                if not guild_channels:
                    await message.channel.send("Could not find channels for that guild.")
                    return

                selected: list[AlertDestination] = []
                invalid: list[int] = []

                for idx in indices:
                    if 1 <= idx <= len(guild_channels):
                        ch = guild_channels[idx - 1]
                        selected.append(
                            AlertDestination(
                                guild_id=ch.guild.id,
                                channel_id=ch.id,
                                channel_name=ch.name,
                                guild_name=ch.guild.name,
                            ),
                        )
                    else:
                        invalid.append(idx)

                if not selected:
                    await message.channel.send(f"No valid selections. Please choose 1-{len(guild_channels)}.")
                    return

                session.destinations.extend(selected)
                session.selected_guild_id = None

                self.logger.info(f"User {message.author.id} selected {len(selected)} channel(s); total={len(session.destinations)}")

                if invalid:
                    await message.channel.send(f"Ignored out-of-range: {', '.join(map(str, invalid))}")

                names = ", ".join([f"#{d.channel_name}" for d in selected])
                await message.channel.send(f"Added {len(selected)}: {names}\n\nTotal selected: {len(session.destinations)} channel(s)\nReply `back` for more guilds, `done` to finish selection.")
                await self._update_composer_embed(message, session)
                return

            # STEP: Compose Content
            if session.current_step == "compose_content":
                title, desc, body = self._parse_content_fields(content)
                if title is not None:
                    session.embed_title = title
                if desc is not None:
                    session.embed_description = desc
                if body:
                    session.content = body

                parts = []
                if title is not None:
                    parts.append("title")
                if desc is not None:
                    parts.append("description")
                if body:
                    parts.append("content")
                changed = ", ".join(parts) if parts else "(no changes)"

                await message.channel.send(f"Updated {changed}. React with preview to preview or send to send.")
                await self._update_composer_embed(message, session)
                return

        except Exception as e:
            self.logger.exception(f"Error handling DM message: {e}")
            try:
                await message.channel.send("Error processing your input. Please try again.")
            except (discord.HTTPException, discord.NotFound, discord.Forbidden):
                self.logger.debug("Failed to send error message to user", exc_info=True)

    def _extract_indices(self, text: str) -> list[int]:
        """Parse comma/space separated integers from user input."""
        indices: list[int] = []
        for token in text.replace("\n", ",").split(","):
            token = token.strip()
            if not token:
                continue
            try:
                indices.append(int(token))
            except ValueError:
                for sub in token.split():
                    try:
                        indices.append(int(sub))
                    except ValueError:
                        continue
        return indices

    def _parse_content_fields(self, text: str) -> tuple[str | None, str | None, str]:
        """Extract TITLE: and DESC: lines; return (title|None, desc|None, body_text)."""
        title: str | None = None
        desc: str | None = None
        body_lines: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("title:"):
                title = stripped.split(":", 1)[1].strip()
            elif stripped.lower().startswith("desc:"):
                desc = stripped.split(":", 1)[1].strip()
            else:
                body_lines.append(line)
        return title, desc, "\n".join(body_lines).strip()

    async def _update_composer_embed(self, source_message: discord.Message, session) -> None:
        """Refresh the composer embed message based on current session state."""
        try:
            if not session.composer_message_id:
                return
            composer_embed = await self.alert_manager.build_composer_embed(session)
            composer_embed = self.alert_manager._validate_embed_limits(composer_embed)
            dm_msg = await source_message.channel.fetch_message(session.composer_message_id)
            await dm_msg.edit(embed=composer_embed)
        except discord.HTTPException as e:
            self.logger.exception(f"Embed edit failed: status={e.status}, code={e.code}")
        except Exception as e:
            self.logger.exception(f"Failed to update composer embed: {e}")

    async def _handle_channel_selection(self, reaction: discord.Reaction, user: discord.User, session) -> None:
        """Present guild-based channel selection with scrollable guild list."""
        session.current_step = "select_channels"

        channels = await self.alert_manager.get_accessible_channels()
        if not channels:
            await user.send("I couldn't find any text channels I can send messages to. Check the bot permissions and try again.")
            return

        guild_map: dict[int, list[discord.TextChannel]] = {}
        for ch in channels:
            gid = ch.guild.id
            if gid not in guild_map:
                guild_map[gid] = []
            guild_map[gid].append(ch)

        for gid in guild_map:
            guild_map[gid].sort(key=lambda c: (c.position, c.name.lower()))

        sorted_guilds = sorted(
            guild_map.keys(),
            key=lambda g: next((c.guild.name.lower() for c in channels if c.guild.id == g), ""),
        )
        session.guilds_list = sorted_guilds

        await self._show_guild_selection(user, session, guild_map, sorted_guilds)

        with contextlib.suppress(discord.HTTPException):
            await reaction.remove(user)

    async def _show_guild_selection(
        self,
        user: discord.User,
        session,
        guild_map: dict[int, list[discord.TextChannel]],
        sorted_guilds: list[int],
    ) -> None:
        """Display paginated guild list with scroll indicators."""
        GUILDS_PER_PAGE = 8

        page = getattr(session, "guild_page", 0)
        total_pages = max(1, (len(sorted_guilds) + GUILDS_PER_PAGE - 1) // GUILDS_PER_PAGE)
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
            title="Select a Guild",
            description="Use arrows to scroll, then reply with a guild number to browse its channels.",
            color=0x5865F2,
        )

        embed.add_field(
            name=f"Guilds (page {page + 1}/{total_pages})",
            value="\n".join(lines) if lines else "No guilds available",
            inline=False,
        )

        nav_text = "Previous | Next | Cancel"
        embed.add_field(name="Navigation", value=nav_text, inline=False)
        embed.set_footer(text=f"Total: {len(sorted_guilds)} guilds | Reply with a number (1-{len(sorted_guilds)})")

        existing_msg_id = getattr(session, "selection_message_id", None)
        if existing_msg_id:
            try:
                existing_msg = await user.fetch_message(existing_msg_id)
                await existing_msg.edit(embed=embed)
                return
            except discord.NotFound:
                pass
            except discord.HTTPException:
                pass

        selection_msg = await user.send(embed=embed)
        session.selection_message_id = selection_msg.id

        if total_pages > 1:
            await selection_msg.add_reaction("⬆️")
            await selection_msg.add_reaction("⬇️")

    async def _show_channel_selection_for_guild(
        self,
        user: discord.User,
        session,
        guild_id: int,
        guild_map: dict[int, list[discord.TextChannel]],
    ) -> None:
        """Show channels within a selected guild with pagination."""
        CHANNELS_PER_PAGE = 10

        channels = guild_map.get(guild_id, [])
        if not channels:
            await user.send("Could not find channels for that guild.")
            return

        guild_name = channels[0].guild.name if channels[0].guild else "Unknown Guild"

        page = getattr(session, "channel_page", 0)
        total_pages = max(1, (len(channels) + CHANNELS_PER_PAGE - 1) // CHANNELS_PER_PAGE)
        page = max(0, min(page, total_pages - 1))
        session.channel_page = page
        session.selected_guild_id = guild_id

        start_idx = page * CHANNELS_PER_PAGE
        end_idx = min(start_idx + CHANNELS_PER_PAGE, len(channels))

        lines = []
        for i, ch in enumerate(channels[start_idx:end_idx], start=start_idx + 1):
            lines.append(f"`{i:>2}` #{ch.name}")

        embed = discord.Embed(
            title=f"Select Channels from {guild_name}",  # nosec B608
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
            nav_parts.extend(["Prev", "Next"])
        nav_parts.extend(["Back to guilds", "Cancel"])
        embed.add_field(name="Navigation", value=" | ".join(nav_parts), inline=False)
        embed.set_footer(text="Or reply with channel numbers to select (e.g., 1,2,3)")

        msg = await user.send(embed=embed)
        session.channel_message_id = msg.id

        if total_pages > 1:
            await msg.add_reaction("⬅️")
            await msg.add_reaction("➡️")
        await msg.add_reaction("🏠")
        await msg.add_reaction("❌")

    async def _handle_content_composition(self, reaction: discord.Reaction, user: discord.User, session) -> None:
        session.current_step = "compose_content"

        await user.send(
            "**Step 3: Compose Content**\n\n"
            "Reply with your alert content. You can include:\n"
            "- Message text\n"
            "- Embed title (prefix with `TITLE: `)\n"
            "- Embed description (prefix with `DESC: `)\n\n"
            "Example:\n"
            "```\n"
            "TITLE: Server Maintenance\n"
            "DESC: Scheduled maintenance tonight\n"
            "Please save your work.\n"
            "```",
        )

        try:
            composer_embed = await self.alert_manager.build_composer_embed(session)
            composer_embed = self.alert_manager._validate_embed_limits(composer_embed)
            full_message = await reaction.message.channel.fetch_message(reaction.message.id)
            await full_message.edit(embed=composer_embed)
        except discord.HTTPException as e:
            self.logger.exception(f"Failed to update composer embed in content composition: status={e.status}, code={e.code}")
            raise

    async def _handle_preview(self, reaction: discord.Reaction, user: discord.User, session) -> None:
        if not session.destinations:
            await user.send("Please select destinations first.")
            return

        if not session.content and not session.embed_title:
            await user.send("Please compose content first.")
            return

        session.current_step = "preview_alert"

        preview_embed = discord.Embed(
            title="Alert Preview",
            description="This is how your alert will appear:",
            color=0x5865F2,
        )

        dest_list = [f"#{dest.channel_name or 'unknown-channel'}" for dest in session.destinations[:10]]
        if len(session.destinations) > 10:
            dest_list.append(f"... and {len(session.destinations) - 10} more")

        preview_embed.add_field(
            name=f"Destinations ({len(session.destinations)})",
            value="\n".join(dest_list),
            inline=False,
        )

        await user.send(embed=preview_embed)

        if session.embed_title or session.embed_description:
            alert_embed = discord.Embed(
                title=session.embed_title,
                description=session.embed_description,
                color=0x1F8B4C,
            )
            await safe_send(user, f"**PREVIEW:** {session.content}", embed=alert_embed)
        else:
            await safe_send(user, f"**PREVIEW:** {session.content}")

        session.current_step = "confirm_send"
        try:
            composer_embed = await self.alert_manager.build_composer_embed(session)
            composer_embed = self.alert_manager._validate_embed_limits(composer_embed)
            full_message = await reaction.message.channel.fetch_message(reaction.message.id)
            await full_message.edit(embed=composer_embed)
        except discord.HTTPException as e:
            self.logger.exception(f"Failed to update composer embed in preview: status={e.status}, code={e.code}")
            raise

    async def _handle_send_confirmation(self, reaction: discord.Reaction, user: discord.User, session) -> None:
        if session.current_step != "confirm_send":
            await user.send("Please complete all steps before sending.")
            return

        confirm_embed = discord.Embed(
            title="Final Confirmation",
            description=f"Send alert to **{len(session.destinations)} channels**?",
            color=0xFF9500,
        )

        confirm_message = await user.send(embed=confirm_embed)
        await confirm_message.add_reaction("✅")
        await confirm_message.add_reaction("❌")

        def check(reaction_check, user_check):
            return user_check == user and reaction_check.message.id == confirm_message.id and str(reaction_check.emoji) in ["✅", "❌"]

        try:
            reaction_result, _ = await self.bot.wait_for("reaction_add", timeout=60.0, check=check)

            if str(reaction_result.emoji) == "✅":
                await user.send("Sending alert...")
                results = await self.alert_manager.send_alert(session)

                result_embed = discord.Embed(
                    title="Alert Send Results",
                    color=0x00FF00 if results["failed_sends"] == 0 else 0xFF9500,
                )

                result_embed.add_field(
                    name="Summary",
                    value=f"Successful: {results['successful_sends']}\nFailed: {results['failed_sends']}",
                    inline=False,
                )

                await user.send(embed=result_embed)
                del self.alert_manager.sessions[user.id]

            else:
                await user.send("Alert send cancelled.")

        except TimeoutError:
            await user.send("Confirmation timeout. Alert cancelled.")

    async def _handle_cancel(self, reaction: discord.Reaction, user: discord.User, session) -> None:
        session.status = AlertSessionStatus.CANCELLED
        del self.alert_manager.sessions[user.id]
        await user.send("Alert session cancelled.")


async def setup(bot) -> None:
    await bot.add_cog(AdminAlertCommands(bot))
