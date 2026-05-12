"""
Additional memory control commands for the Discord bot.
Extends memory_cmds.py with !memory-status, !memory-review, !memory-forget,
!memory-disable, !memory-enable, !memory-export
"""

import io
import logging

import discord
from discord.ext import commands

from ..config import load_config
from ..memory import get_memory_service

logger = logging.getLogger(__name__)


class ExtendedMemoryCommands(commands.Cog):
    """Extended memory management commands."""

    def __init__(self, bot) -> None:
        self.bot = bot
        self.config = load_config()

    @commands.command(name="memory-status", aliases=["mem-status"])
    async def memory_status(self, ctx):
        """Show memory service status and queue depth.

        Owner/admin only for detailed diagnostics.
        """
        try:
            from bot.core.permissions import is_admin_user

            if not await is_admin_user(ctx.author, self.bot):
                await ctx.send(
                    "🔒 Memory status is only available to owners and admins."
                )
                return

            service = await get_memory_service(self.bot)
            enabled = service.enabled if service else False

            embed = discord.Embed(
                title="Memory Service Status",
                color=discord.Color.green() if enabled else discord.Color.red(),
            )
            embed.add_field(name="Enabled", value=str(enabled), inline=True)

            if enabled and service:
                queue_size = service.queue.qsize()
                embed.add_field(name="Queue Depth", value=str(queue_size), inline=True)

                try:
                    chroma_status = (
                        "ready"
                        if service.semantic_store._collection
                        else "initializing"
                    )
                except Exception:
                    chroma_status = "unknown"
                embed.add_field(name="Vector Store", value=chroma_status, inline=True)

                try:
                    store_status = (
                        "connected" if service.store._conn else "disconnected"
                    )
                except Exception:
                    store_status = "unknown"
                embed.add_field(name="SQLite Store", value=store_status, inline=True)
            else:
                embed.add_field(
                    name="Status",
                    value="Memory service disabled or not initialized",
                    inline=False,
                )

            await ctx.send(embed=embed)

        except Exception as e:
            logger.error(f"Error in memory-status: {e}", exc_info=True)
            await ctx.send("❌ Failed to retrieve memory status.")

    @commands.command(name="memory-review", aliases=["mem-review"])
    async def memory_review(self, ctx, limit: int = 10):
        """Review your recent curated memories with context.

        Args:
            limit: Number of memories to review (default: 10, max: 50)
        """
        try:
            limit = min(max(1, int(limit)), 50)
            service = await get_memory_service(self.bot)

            if not service or not service.enabled:
                await ctx.send("Memory service is not enabled.")
                return

            user_id = str(ctx.author.id)
            records = await service.list_user_memories(user_id, limit=limit)

            if not records:
                await ctx.send("You don't have any curated memories yet.")
                return

            embed = discord.Embed(
                title=f"Your Curated Memories ({len(records)} shown)",
                color=discord.Color.blue(),
            )

            for idx, record in enumerate(records, 1):
                summary = record.summary or record.text or ""
                if len(summary) > 500:
                    summary = summary[:497] + "..."

                field_value = (
                    f"**Type:** {record.context_type}\n"
                    f"**Confidence:** {record.confidence:.2f}\n"
                    f"**Summary:** {summary}\n"
                    f"**Created:** {record.created_at[:10] if record.created_at else 'N/A'}"
                )

                embed.add_field(
                    name=f"{idx}. `{record.memory_id[:8]}`",
                    value=field_value,
                    inline=False,
                )

            await ctx.send(embed=embed)

        except Exception as e:
            logger.error(f"Error in memory-review: {e}", exc_info=True)
            await ctx.send("❌ Failed to retrieve memories for review.")

    @commands.command(name="memory-forget", aliases=["mem-forget"])
    async def memory_forget(self, ctx, *, memory_id: str):
        """Forget a specific memory by ID.

        Args:
            memory_id: The memory ID (first 8 chars OK) or search query
        """
        try:
            memory_id = (memory_id or "").strip()
            if not memory_id:
                await ctx.send("❌ Usage: `!memory-forget <id or search>`")
                return

            service = await get_memory_service(self.bot)
            if not service or not service.enabled:
                await ctx.send("Memory service is not enabled.")
                return

            deleted = await service.delete_memory(memory_id)

            if not deleted:
                matches = await service.search_user_memories(
                    str(ctx.author.id), memory_id, limit=5
                )
                if not matches:
                    await ctx.send("❌ No matching memory found.")
                    return

                target = matches[0]
                deleted = await service.delete_memory(target.memory_id)

                if deleted:
                    await ctx.send(
                        f"✅ Deleted memory `{target.memory_id[:8]}`: {target.summary}"
                    )
                else:
                    await ctx.send("❌ Failed to delete memory.")
            else:
                await ctx.send(f"✅ Deleted memory `{memory_id[:8]}`.")

        except Exception as e:
            logger.error(f"Error in memory-forget: {e}", exc_info=True)
            await ctx.send("❌ Failed to forget memory.")

    @commands.command(name="memory-disable", aliases=["mem-disable"])
    async def memory_disable(self, ctx):
        """Disable memory ingestion for your profile (owner/admin only in guilds).

        In DMs, any user can disable their own memory.
        In guilds, only owner/admin can disable.
        """
        try:
            from bot.core.permissions import is_admin_user

            if ctx.guild and not await is_admin_user(ctx.author, self.bot):
                await ctx.send("🔒 In guilds, only owners/admins can disable memory.")
                return

            await ctx.send(
                "✅ Memory ingestion preference noted. "
                "Note: This is a placeholder - full implementation requires user preference persistence."
            )

        except Exception as e:
            logger.error(f"Error in memory-disable: {e}", exc_info=True)
            await ctx.send("❌ Failed to disable memory.")

    @commands.command(name="memory-enable", aliases=["mem-enable"])
    async def memory_enable(self, ctx):
        """ "Re-enable memory ingestion for your profile.

        Same permission rules as memory-disable.
        """
        try:
            from bot.core.permissions import is_admin_user

            if ctx.guild and not await is_admin_user(ctx.author, self.bot):
                await ctx.send("🔒 In guilds, only owners/admins can enable memory.")
                return

            # Get or create user profile
            from bot.memory.profiles import get_profile, save_profile

            profile = get_profile(str(ctx.author.id))

            # Set memory preference
            if "preferences" not in profile:
                profile["preferences"] = {}
            profile["preferences"]["memory_enabled"] = True

            # Save profile
            if save_profile(profile, caller_id=str(ctx.author.id)):
                await ctx.send(
                    "✅ Memory ingestion has been re-enabled for your profile."
                )
            else:
                await ctx.send("❌ Failed to save memory preference.")

        except Exception as e:
            logger.error(f"Error in memory-enable: {e}", exc_info=True)
            await ctx.send("❌ Failed to enable memory.")

    @commands.command(name="memory-export", aliases=["mem-export"])
    async def memory_export(self, ctx, format: str = "text"):
        """Export your memories in a specified format.

        Args:
            format: Export format ('text', 'json') - default: text
        """
        try:
            format = (format or "text").strip().lower()
            if format not in ("text", "json"):
                await ctx.send("❌ Format must be 'text' or 'json'.")
                return

            service = await get_memory_service(self.bot)
            if not service or not service.enabled:
                await ctx.send("Memory service is not enabled.")
                return

            user_id = str(ctx.author.id)
            records = await service.list_user_memories(user_id, limit=50)

            if not records:
                await ctx.send("You don't have any memories to export.")
                return

            if format == "json":
                import json

                data = [
                    {
                        "id": r.memory_id,
                        "summary": r.summary,
                        "text": r.text,
                        "context_type": r.context_type,
                        "created_at": r.created_at,
                        "confidence": r.confidence,
                    }
                    for r in records
                ]
                json_str = json.dumps(data, indent=2, default=str)
                if len(json_str) > 1900:
                    file = discord.File(
                        fp=io.BytesIO(json_str.encode()), filename="memories.json"
                    )
                    await ctx.send(
                        f"📄 Your memories ({len(records)} records):", file=file
                    )
                else:
                    await ctx.send(f"```\n{json_str}\n```")
            else:
                lines = [f"=== Your Memories ({len(records)}) ==="]
                for idx, r in enumerate(records, 1):
                    lines.append(
                        f"\n{idx}. [{r.memory_id[:8]}] ({r.context_type})\n"
                        f"   {r.summary or r.text}\n"
                        f"   Created: {r.created_at[:10] if r.created_at else 'N/A'}"
                    )

                text = "\n".join(lines)
                if len(text) > 1900:
                    file = discord.File(
                        fp=io.BytesIO(text.encode()), filename="memories.txt"
                    )
                    await ctx.send(
                        f"📄 Your memories ({len(records)} records):", file=file
                    )
                else:
                    await ctx.send(f"```\n{text}\n```")

        except Exception as e:
            logger.error(f"Error in memory-export: {e}", exc_info=True)
            await ctx.send("❌ Failed to export memories.")


async def setup(bot) -> None:
    """Add extended memory commands to the bot."""
    if not bot.get_cog("ExtendedMemoryCommands"):
        await bot.add_cog(ExtendedMemoryCommands(bot))
    else:
        logger.warning("'ExtendedMemoryCommands' cog already loaded, skipping setup.")
