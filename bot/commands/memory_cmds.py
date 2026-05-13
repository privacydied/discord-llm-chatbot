"""
Memory-related commands for the Discord bot.

This module provides commands to manage user and server memories.
"""

import asyncio
import logging

import discord
from discord.ext import commands

# Import bot modules
from ..config import load_config
from ..memory import (
    add_explicit_memory,
    delete_memory,
    get_memory_distiller,
    get_profile,
    get_server_profile,
    list_user_memories,
    save_profile,
    save_server_profile,
    search_user_memories,
    wipe_user_memories,
)
from bot.logger import log_command

logger = logging.getLogger(__name__)


class MemoryCommands(commands.Cog):
    """Commands for managing user and server memories."""

    def __init__(self, bot) -> None:
        self.bot = bot
        self.config = load_config()
        self.router = bot.router
        self.prefix = self.config.get("COMMAND_PREFIX", "!")
        self._distill_once_tasks: dict[str, asyncio.Task] = {}

    async def _add_curated_memory(
        self,
        ctx,
        *,
        content: str,
        context_type: str = "user_preference",
    ):
        content = (content or "").strip()
        if not content:
            await ctx.send("❌ Memory content cannot be empty.")
            return

        try:
            record = await add_explicit_memory(
                user_id=str(ctx.author.id),
                text=content,
                guild_id=str(ctx.guild.id) if getattr(ctx, "guild", None) else None,
                channel_id=str(ctx.channel.id)
                if getattr(ctx, "channel", None)
                else None,
                thread_id=str(ctx.channel.id)
                if isinstance(ctx.channel, discord.Thread)
                else None,
                source_message_id=str(getattr(ctx.message, "id", None))
                if getattr(ctx, "message", None)
                else None,
                context_type=context_type,
                source="explicit_memory_command",
                metadata={
                    "command": ctx.command.qualified_name
                    if getattr(ctx, "command", None)
                    else "memory-add"
                },
            )
        except Exception as exc:
            logger.error("Failed to persist curated memory: %s", exc, exc_info=True)
            await ctx.send("❌ Failed to save memory. Please try again.")
            return

        await ctx.send(f"✅ Memory saved. ID: `{record.memory_id[:8]}`")

    async def _show_curated_memories(self, ctx, limit: int = 5):
        limit = min(max(1, int(limit)), 20)
        records = await list_user_memories(str(ctx.author.id), limit=limit)
        if not records:
            await ctx.send(
                "You don't have any durable memories yet. Use `!memory-add <content>` to add one!"
            )
            return

        embed = discord.Embed(
            title=f"Your Durable Memories ({len(records)})",
            color=discord.Color.blurple(),
        )
        for idx, record in enumerate(records, 1):
            embed.add_field(
                name=f"{idx}. {record.memory_id[:8]} · {record.context_type}",
                value=(record.summary or record.text)[:1024],
                inline=False,
            )
        await ctx.send(embed=embed)

    async def _delete_curated_memory(self, ctx, query: str):
        query = (query or "").strip()
        if not query:
            await ctx.send("❌ Usage: `!memory-del <id or search>`")
            return

        if await delete_memory(query, owner_id=str(ctx.author.id)):
            await ctx.send(f"✅ Deleted memory `{query[:8]}`.")
            return

        matches = await search_user_memories(str(ctx.author.id), query, limit=5)
        if not matches:
            await ctx.send("❌ No matching memory found.")
            return

        target = matches[0]
        if await delete_memory(target.memory_id, owner_id=str(ctx.author.id)):
            await ctx.send(
                f"✅ Deleted memory `{target.memory_id[:8]}`: {target.summary}"
            )
        else:
            await ctx.send("❌ Failed to delete memory. Please try again.")

    async def _wipe_curated_memories(self, ctx):
        try:
            count = await wipe_user_memories(str(ctx.author.id))
            await ctx.send(f"✅ Wiped {count} durable memories.")
        except Exception as exc:
            logger.error("Failed to wipe curated memories: %s", exc, exc_info=True)
            await ctx.send("❌ Failed to wipe memories. Please try again.")

    async def _search_curated_memories(self, ctx, query: str, limit: int = 5):
        query = (query or "").strip()
        if not query:
            await ctx.send("❌ Usage: `!memory-search <query>`")
            return

        records = await search_user_memories(
            str(ctx.author.id), query, limit=min(max(1, limit), 10)
        )
        if not records:
            await ctx.send("No matching durable memories found.")
            return

        embed = discord.Embed(
            title=f"Memory Search: {query}", color=discord.Color.gold()
        )
        for idx, record in enumerate(records, 1):
            embed.add_field(
                name=f"{idx}. {record.memory_id[:8]} · {record.context_type}",
                value=(record.summary or record.text)[:1024],
                inline=False,
            )
        await ctx.send(embed=embed)

    @commands.command(name="memory-add")
    async def memory_add_direct(self, ctx, *, content: str):
        await self._add_curated_memory(
            ctx, content=content, context_type="user_preference"
        )

    @commands.command(name="memory-show")
    async def memory_show_direct(self, ctx, limit: int = 5):
        await self._show_curated_memories(ctx, limit=limit)

    @commands.command(name="memory-del")
    async def memory_del_direct(self, ctx, *, query: str):
        await self._delete_curated_memory(ctx, query=query)

    @commands.command(name="memory-wipe")
    @commands.cooldown(1, 120, commands.BucketType.user)
    async def memory_wipe_direct(self, ctx):
        await self._wipe_curated_memories(ctx)

    @commands.command(name="memory-search")
    async def memory_search_direct(self, ctx, *, query: str):
        await self._search_curated_memories(ctx, query=query)

    @commands.command(name="memory-distill-status", aliases=["memory-distil-status"])
    @commands.guild_only()
    @commands.has_permissions(administrator=True)
    async def memory_distill_status(self, ctx):
        distiller = await get_memory_distiller(self.bot)
        status = await distiller.get_status(guild_id=str(ctx.guild.id))
        running_task = self._distill_once_tasks.get(str(ctx.guild.id))
        running = bool(running_task and not running_task.done())
        last_run = status.get("last_run") or {}
        scanned_count = int(last_run.get("scanned_count", 0) or 0)
        candidate_count = int(last_run.get("candidate_count", 0) or 0)
        rejected_count = int(last_run.get("rejected_count", 0) or 0)
        accepted_count = int(last_run.get("accepted_count", 0) or 0)
        merged_count = int(last_run.get("merged_count", 0) or 0)
        skipped_early_count = max(0, scanned_count - candidate_count)
        candidate_rejected_count = max(0, rejected_count - skipped_early_count)
        embed = discord.Embed(
            title="Memory Distiller Status",
            color=discord.Color.blue()
            if status.get("enabled")
            else discord.Color.dark_grey(),
        )
        embed.add_field(name="Enabled", value=str(status.get("enabled")), inline=True)
        embed.add_field(name="Dry run", value=str(status.get("dry_run")), inline=True)
        embed.add_field(name="Started", value=str(status.get("started")), inline=True)
        embed.add_field(name="Running", value=str(running), inline=True)
        embed.add_field(name="Backlog", value=str(status.get("backlog")), inline=True)
        embed.add_field(
            name="Batch size", value=str(status.get("batch_size")), inline=True
        )
        embed.add_field(
            name="Interval", value=f"{status.get('interval_seconds')}s", inline=True
        )
        embed.add_field(name="Scanned", value=str(scanned_count), inline=True)
        embed.add_field(
            name="Skipped early", value=str(skipped_early_count), inline=True
        )
        embed.add_field(
            name="Candidate rejected", value=str(candidate_rejected_count), inline=True
        )
        embed.add_field(name="Accepted", value=str(accepted_count), inline=True)
        embed.add_field(name="Merged", value=str(merged_count), inline=True)
        if last_run:
            embed.add_field(
                name="Last run",
                value=(
                    f"candidate_count={candidate_count} total_rejected={rejected_count}"
                ),
                inline=False,
            )
        await ctx.send(embed=embed)

    @commands.command(name="memory-distill-once", aliases=["memory-distil-once"])
    @commands.guild_only()
    @commands.has_permissions(administrator=True)
    async def memory_distill_once(self, ctx):
        guild_id = str(ctx.guild.id)
        existing = self._distill_once_tasks.get(guild_id)
        if existing and not existing.done():
            await ctx.send("⏳ Memory distillation is already running for this server.")
            return
        if existing and existing.done():
            self._distill_once_tasks.pop(guild_id, None)

        distiller = await get_memory_distiller(self.bot)
        start_embed = discord.Embed(
            title="Memory Distillation Started",
            description="Running `!memory-distill-once` in the background.",
            color=discord.Color.gold(),
        )
        start_embed.add_field(name="Mode", value="background", inline=True)
        start_embed.add_field(name="Status", value="running", inline=True)
        await ctx.send(embed=start_embed)

        async def _run_and_report() -> None:
            try:
                result = await distiller.run_once()
                done_embed = discord.Embed(
                    title="Memory Distillation Complete",
                    color=discord.Color.green(),
                )
                done_embed.add_field(
                    name="Scanned",
                    value=str(result.get("scanned_count", 0)),
                    inline=True,
                )
                done_embed.add_field(
                    name="Candidates",
                    value=str(result.get("candidate_count", 0)),
                    inline=True,
                )
                done_embed.add_field(
                    name="Accepted",
                    value=str(result.get("accepted_count", 0)),
                    inline=True,
                )
                done_embed.add_field(
                    name="Rejected",
                    value=str(result.get("rejected_count", 0)),
                    inline=True,
                )
                done_embed.add_field(
                    name="Merged", value=str(result.get("merged_count", 0)), inline=True
                )
                done_embed.add_field(
                    name="Dry run", value=str(result.get("dry_run", True)), inline=True
                )
                await ctx.send(embed=done_embed)
            except Exception as exc:
                logger.exception("Background memory distillation failed")
                await ctx.send(f"❌ Memory distillation failed: `{exc}`")
            finally:
                task = self._distill_once_tasks.get(guild_id)
                if task is not None and task.done():
                    self._distill_once_tasks.pop(guild_id, None)

        task = asyncio.create_task(
            _run_and_report(), name=f"memory-distill-once-{guild_id}"
        )
        self._distill_once_tasks[guild_id] = task

        def _cleanup(_task: asyncio.Task) -> None:
            current = self._distill_once_tasks.get(guild_id)
            if current is _task:
                self._distill_once_tasks.pop(guild_id, None)

        task.add_done_callback(_cleanup)

    @commands.command(name="memory-distill-dryrun")
    @commands.guild_only()
    @commands.has_permissions(administrator=True)
    async def memory_distill_dryrun(self, ctx, mode: str):
        mode = (mode or "").strip().lower()
        if mode not in {"on", "off"}:
            await ctx.send("❌ Usage: `!memory-distill-dryrun <on|off>`")
            return
        distiller = await get_memory_distiller(self.bot)
        distiller.set_dry_run(mode == "on")
        await ctx.send(f"✅ Memory distiller dry-run set to `{distiller.dry_run}`.")

    @commands.group(name="memory", invoke_without_command=True)
    async def memory_group(self, ctx):
        """Memory management commands.

        Usage:
        !memory add <content> - Add a new memory
        !memory list [limit] - List your recent memories (default: 5)
        !memory clear - Clear all your memories
        """
        if ctx.invoked_subcommand is None:
            await ctx.send_help(ctx.command)

    @memory_group.command(name="add")
    @commands.cooldown(1, 10, commands.BucketType.user)  # [BUGFIX] Rate limit added
    async def add_memory_cmd(self, ctx, *, content: str):
        """Add a memory to your profile.

        Example:
        !memory add I prefer to be called by my nickname, not my full name
        """
        try:
            # Input validation [REH][SFT]
            if not content or not content.strip():
                await ctx.send("❌ Memory content cannot be empty.")
                return

            content = content.strip()

            # Length validation to prevent memory abuse [SFT]
            MAX_MEMORY_LENGTH = 2000  # Discord message limit is 2000 chars
            if len(content) > MAX_MEMORY_LENGTH:
                await ctx.send(
                    f"❌ Memory too long. Maximum length is {MAX_MEMORY_LENGTH} characters."
                )
                return

            # Content validation - basic safety checks [SFT]
            prohibited_patterns = ["<script", "javascript:", "data:", "vbscript:"]
            content_lower = content.lower()
            if any(pattern in content_lower for pattern in prohibited_patterns):
                await ctx.send("❌ Memory contains prohibited content.")
                return

            # Get user's profile
            profile = get_profile(str(ctx.author.id), str(ctx.author))

            # Add the memory with sanitized content
            memory = {
                "content": content,
                "timestamp": discord.utils.utcnow().isoformat(),
                "context": f"Added via command in {ctx.channel.name if hasattr(ctx.channel, 'name') else 'DM'}",
            }

            if "memories" not in profile:
                profile["memories"] = []

            profile["memories"].append(memory)

            # Enforce memory limit
            if len(profile["memories"]) > self.config["MAX_MEMORIES"]:
                profile["memories"] = profile["memories"][
                    -self.config["MAX_MEMORIES"] :
                ]

            # Save the profile
            if save_profile(profile, caller_id=str(ctx.author.id)):
                await ctx.send(
                    f"✅ Memory added! You now have {len(profile['memories'])} memories."
                )
                log_command(ctx, f"Added memory: {content[:50]}...")
            else:
                await ctx.send("❌ Failed to save memory. Please try again.")
                logging.error(f"Failed to save memory for user {ctx.author.id}")

        except Exception as e:
            logging.error(f"Error in add_memory_cmd: {str(e)}", exc_info=True)
            await ctx.send("❌ An error occurred while adding the memory.")
            log_command(ctx, "memory_add_error", {"error": str(e)}, success=False)

    @memory_group.command(name="list")
    async def list_memories_cmd(self, ctx, limit: int = 5):
        """List your recent memories.

        Args:
            limit: Number of memories to show (default: 5, max: 20)

        Example:
        !memory list 3 - Show your 3 most recent memories
        """
        try:
            # Enforce a reasonable limit
            limit = min(max(1, limit), 20)

            profile = get_profile(str(ctx.author.id))

            if not profile or "memories" not in profile or not profile["memories"]:
                await ctx.send(
                    "You don't have any memories yet. Use `!memory add <content>` to add one!"
                )
                return

            # Limit the number of memories to show
            memories = profile["memories"][-limit:]

            if not memories:
                await ctx.send("No memories found.")
                return

            # Create an embed to display memories
            embed = discord.Embed(
                title=f"Your Recent Memories (Last {len(memories)} of {len(profile['memories'])})",
                color=discord.Color.blue(),
            )

            for i, memory in enumerate(reversed(memories), 1):
                timestamp = memory.get("timestamp", "Unknown")
                context = memory.get("context", "No context")
                embed.add_field(
                    name=f"Memory #{len(profile['memories']) - len(memories) + i}",
                    value=f"{memory['content']}\n*{context} - {timestamp}*",
                    inline=False,
                )

            await ctx.send(embed=embed)
            log_command(ctx, "Listed memories")

        except Exception as e:
            logging.error(f"Error in list_memories_cmd: {str(e)}", exc_info=True)
            await ctx.send("❌ An error occurred while retrieving memories.")
            log_command(ctx, "memory_list_error", {"error": str(e)}, success=False)

    @memory_group.command(name="clear")
    @commands.cooldown(1, 120, commands.BucketType.user)
    async def clear_memories_cmd(self, ctx):
        """Clear all your memories after confirmation."""
        try:
            # Ask for confirmation
            confirm_msg = await ctx.send(
                "⚠️ Are you sure you want to delete ALL your memories? This cannot be undone. Type `yes` to confirm."
            )

            def check(m):
                return (
                    m.author == ctx.author
                    and m.channel == ctx.channel
                    and m.content.lower() == "yes"
                )

            try:
                await ctx.bot.wait_for("message", check=check, timeout=30.0)
            except asyncio.TimeoutError:
                await confirm_msg.edit(content="Memory clear cancelled due to timeout.")
                return

            # Get and clear memories
            profile = get_profile(str(ctx.author.id))
            if not profile:
                await ctx.send("No profile found to clear.")
                return

            if "memories" in profile and profile["memories"]:
                memory_count = len(profile["memories"])
                profile["memories"] = []

                if save_profile(profile, caller_id=str(ctx.author.id)):
                    await ctx.send(f"✅ Successfully cleared {memory_count} memories.")
                    log_command(ctx, f"Cleared {memory_count} memories")
                else:
                    await ctx.send("❌ Failed to clear memories. Please try again.")
            else:
                await ctx.send("No memories found to clear.")

        except Exception as e:
            logging.error(f"Error in clear_memories_cmd: {str(e)}", exc_info=True)
            await ctx.send("❌ An error occurred while clearing memories.")
        finally:
            # Reset cooldown if command failed
            self.clear_memories_cmd.reset_cooldown(ctx)

    @commands.group(
        name="server-memory",
        description="Manage server memories (Admin only)",
        invoke_without_command=True,
    )
    @commands.guild_only()
    @commands.has_permissions(administrator=True)
    async def server_memory_group(self, ctx):
        """Manage server memories (Admin only).

        Subcommands:
        add <content> - Add a server memory
        list - List all server memories
        clear - Clear all server memories
        """
        if ctx.invoked_subcommand is None:
            await ctx.send_help(ctx.command)

    @server_memory_group.command(name="add")
    @commands.has_permissions(administrator=True)
    async def server_memory_add(self, ctx, *, content: str):
        """Add a memory to the server's profile."""
        try:
            # Input validation [REH][SFT] — mirror add_memory_cmd safety
            if not content or not content.strip():
                await ctx.send("❌ Memory content cannot be empty.")
                return

            content = content.strip()

            # Length validation to prevent memory abuse [SFT]
            MAX_MEMORY_LENGTH = 2000
            if len(content) > MAX_MEMORY_LENGTH:
                await ctx.send(
                    f"❌ Memory too long. Maximum length is {MAX_MEMORY_LENGTH} characters."
                )
                return

            # Content validation - basic safety checks [SFT]
            prohibited_patterns = ["<script", "javascript:", "data:", "vbscript:"]
            content_lower = content.lower()
            if any(pattern in content_lower for pattern in prohibited_patterns):
                await ctx.send("❌ Memory contains prohibited content.")
                return

            # Get server profile
            server_id = str(ctx.guild.id)
            profile = get_server_profile(server_id, ctx.guild.name)

            # Add the memory
            memory = {
                "content": content,
                "timestamp": discord.utils.utcnow().isoformat(),
                "added_by": str(ctx.author),
                "context": f"Added in #{ctx.channel.name if hasattr(ctx.channel, 'name') else 'unknown'}",
            }

            if "memories" not in profile:
                profile["memories"] = []

            profile["memories"].append(memory)

            # Enforce memory limit
            if len(profile["memories"]) > self.config["MAX_SERVER_MEMORIES"]:
                profile["memories"] = profile["memories"][
                    -self.config["MAX_SERVER_MEMORIES"] :
                ]

            # Save the profile
            if save_server_profile(server_id, profile):
                await ctx.send(
                    f"✅ Server memory added! There are now {len(profile['memories'])} server memories."
                )
                log_command(ctx, f"Added server memory: {content[:50]}...")
            else:
                await ctx.send("❌ Failed to save server memory. Please try again.")

        except Exception as e:
            logging.error(f"Error adding server memory: {str(e)}", exc_info=True)
            await ctx.send("❌ An error occurred while adding the server memory.")

    @server_memory_group.command(name="list")
    @commands.has_permissions(administrator=True)
    async def server_memory_list(self, ctx):
        """List all server memories."""
        try:
            profile = get_server_profile(str(ctx.guild.id))

            if not profile or "memories" not in profile or not profile["memories"]:
                await ctx.send(
                    "No server memories found. Use `!server-memory add <content>` to add one!"
                )
                return

            # Create an embed to display memories
            embed = discord.Embed(
                title=f"Server Memories ({len(profile['memories'])} total)",
                color=discord.Color.green(),
            )

            for i, memory in enumerate(reversed(profile["memories"]), 1):
                added_by = memory.get("added_by", "Unknown")
                timestamp = memory.get("timestamp", "Unknown")
                context = memory.get("context", "No context")

                embed.add_field(
                    name=f"Memory #{i}",
                    value=f"{memory['content']}\n*Added by {added_by} - {context} - {timestamp}*",
                    inline=False,
                )

                # Discord has a limit of 25 fields per embed
                if i >= 25:
                    embed.set_footer(
                        text=f"Showing 25 most recent of {len(profile['memories'])} memories."
                    )
                    break

            await ctx.send(embed=embed)
            log_command(ctx, "Listed server memories")

        except Exception as e:
            logging.error(f"Error listing server memories: {str(e)}", exc_info=True)
            await ctx.send("❌ An error occurred while retrieving server memories.")

    @server_memory_group.command(name="clear")
    @commands.has_permissions(administrator=True)
    @commands.cooldown(1, 60, commands.BucketType.guild)
    async def server_memory_clear(self, ctx):
        """Clear all server memories after confirmation."""
        try:
            # Ask for confirmation
            confirm_msg = await ctx.send(
                "⚠️ Are you sure you want to delete ALL server memories? This cannot be undone. Type `yes` to confirm."
            )

            def check(m):
                return (
                    m.author == ctx.author
                    and m.channel == ctx.channel
                    and m.content.lower() == "yes"
                )

            try:
                await ctx.bot.wait_for("message", check=check, timeout=30.0)
            except asyncio.TimeoutError:
                await confirm_msg.edit(
                    content="Server memory clear cancelled due to timeout."
                )
                return

            # Clear server memories
            profile = get_server_profile(str(ctx.guild.id))
            if not profile or "memories" not in profile or not profile["memories"]:
                await ctx.send("No server memories found to clear.")
                return

            memory_count = len(profile["memories"])
            profile["memories"] = []

            if save_server_profile(str(ctx.guild.id), profile):
                await ctx.send(
                    f"✅ Successfully cleared {memory_count} server memories."
                )
                log_command(ctx, f"Cleared {memory_count} server memories")
            else:
                await ctx.send("❌ Failed to clear server memories. Please try again.")

        except Exception as e:
            logging.error(f"Error clearing server memories: {str(e)}", exc_info=True)
            await ctx.send("❌ An error occurred while clearing server memories.")
        finally:
            # Reset cooldown if command failed
            self.server_memory_clear.reset_cooldown(ctx)


async def setup(bot) -> None:
    """Add memory commands to the bot."""
    if not bot.get_cog("MemoryCommands"):
        await bot.add_cog(MemoryCommands(bot))
    else:
        logger.warning("'MemoryCommands' cog already loaded, skipping setup.")
