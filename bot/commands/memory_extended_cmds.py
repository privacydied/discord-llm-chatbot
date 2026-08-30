"""Additional memory control commands for the Discord bot.
Extends memory_cmds.py with !memory-status, !memory-review, !memory-forget,
!memory-disable, !memory-enable, !memory-export.
"""

import asyncio
import io
import logging

import discord
from discord.ext import commands

from bot.config import load_config
from bot.core.output import safe_send
from bot.memory import get_memory_service

logger = logging.getLogger(__name__)


class ExtendedMemoryCommands(commands.Cog):
    """Extended memory management commands."""

    def __init__(self, bot) -> None:
        self.bot = bot
        self.config = load_config()

    @commands.command(name="memory-status", aliases=["mem-status"])
    async def memory_status(self, ctx) -> None:
        """Show memory service status and queue depth.

        Owner/admin only for detailed diagnostics.
        """
        try:
            from bot.core.permissions import is_admin_user

            if not await is_admin_user(ctx.author, self.bot):
                await ctx.send("🔒 Memory status is only available to owners and admins.")
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
                    chroma_status = "ready" if service.semantic_store._collection else "initializing"
                except AttributeError:
                    chroma_status = "unknown"
                embed.add_field(name="Vector Store", value=chroma_status, inline=True)

                try:
                    store_status = "connected" if service.store._conn else "disconnected"
                except AttributeError:
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
    async def memory_review(self, ctx, limit: int = 10) -> None:
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

                field_value = f"**Type:** {record.context_type}\n**Confidence:** {record.confidence:.2f}\n**Summary:** {summary}\n**Created:** {record.created_at[:10] if record.created_at else 'N/A'}"

                embed.add_field(
                    name=f"{idx}. `{record.memory_id[:8]}`",
                    value=field_value,
                    inline=False,
                )

            await ctx.send(embed=embed)

        except Exception as e:
            logger.error(f"Error in memory-review: {e}", exc_info=True)
            await ctx.send("❌ Failed to retrieve memories for review.")

    @staticmethod
    def _parse_memory_ids(raw: str) -> list[str]:
        """Split whitespace-separated IDs, preserving order and de-duping."""
        seen: dict[str, None] = {}
        for token in (raw or "").split():
            seen.setdefault(token, None)
        return list(seen)

    @staticmethod
    def _resolve_memory_target(owned: list, raw_id: str) -> tuple[object | None, str | None]:
        """Resolve one ID/prefix against owned memories.

        Returns (record, None) on a clean match, or (None, error_text).
        """
        exact_matches = [r for r in owned if r.memory_id == raw_id]
        candidates = exact_matches if len(exact_matches) == 1 else [r for r in owned if r.memory_id.startswith(raw_id)]

        if not candidates:
            return None, f"No memory owned by you matches ID prefix `{raw_id}`."
        if len(candidates) > 1:
            ids = ", ".join(f"`{r.memory_id[:8]}`" for r in candidates)
            return None, f"Ambiguous prefix `{raw_id}` matches {len(candidates)} memories: {ids}. Provide more characters."
        return candidates[0], None

    def _resolve_memory_targets(self, owned: list, raw_ids: list[str]) -> tuple[list, list[str]]:
        """Resolve every requested ID, deduping by resolved memory_id."""
        targets: dict[str, object] = {}
        errors: list[str] = []
        for raw_id in raw_ids:
            record, error = self._resolve_memory_target(owned, raw_id)
            if error:
                errors.append(error)
            else:
                targets[record.memory_id] = record
        return list(targets.values()), errors

    @staticmethod
    def _format_target_preview(target) -> str:
        summary = target.summary or target.text or ""
        if len(summary) > 200:
            summary = summary[:197] + "..."
        return f"**ID:** `{target.memory_id}`\n**Preview:** {summary}\n**Type:** {target.context_type}  **Confidence:** {target.confidence:.2f}"

    async def _confirm_forget(self, ctx, targets: list) -> bool:
        """Show a single confirmation for one or many memories; wait for a reaction."""
        previews = "\n\n".join(self._format_target_preview(t) for t in targets)
        plural = "y" if len(targets) == 1 else "ies"
        header = f"🧠 Found {len(targets)} matching memor{plural}. React with ✅ to delete, or ignore.\n\n"
        await safe_send(ctx, header + previews)

        def check(reaction, user):
            return user.id == ctx.author.id and reaction.message.channel == ctx.channel and str(reaction.emoji) == "✅"

        try:
            await self.bot.wait_for("reaction_add", timeout=30.0, check=check)
        except asyncio.TimeoutError:
            await safe_send(ctx, "⏱️ Delete cancelled (timed out).")
            return False
        return True

    async def _delete_targets(self, ctx, service, requester_id: str, targets: list) -> None:
        lines = []
        for target in targets:
            deleted = await service.delete_memory(target.memory_id, owner_id=requester_id)
            mark = "✅ Deleted" if deleted else "❌ Failed to delete"
            lines.append(f"{mark} `{target.memory_id[:8]}`.")
        await safe_send(ctx, "\n".join(lines))

    @commands.command(name="memory-forget", aliases=["mem-forget"])
    async def memory_forget(self, ctx, *, memory_id: str) -> None:
        """Forget one or more memories by ID.

        Accepts one or more full UUIDs or unambiguous prefixes, space
        separated, e.g. `!memory-forget ca5a45ec 8e41dd1b`. Never uses
        fuzzy/semantic matching for deletion — only exact or prefix match
        within the requesting user's own memories.
        """
        try:
            raw_ids = self._parse_memory_ids(memory_id)
            if not raw_ids:
                await safe_send(ctx, "❌ Usage: `!memory-forget <id> [id2] [id3] ...`")
                return

            service = await get_memory_service(self.bot)
            if not service or not service.enabled:
                await safe_send(ctx, "Memory service is not enabled.")
                return

            requester_id = str(ctx.author.id)
            owned = await service.list_user_memories(requester_id, limit=500)
            targets, errors = self._resolve_memory_targets(owned, raw_ids)

            if errors:
                await safe_send(ctx, "❌ " + "\n".join(errors))
            if not targets:
                return

            if not await self._confirm_forget(ctx, targets):
                return

            await self._delete_targets(ctx, service, requester_id, targets)

        except Exception as e:
            logger.error(f"Error in memory-forget: {e}", exc_info=True)
            await safe_send(ctx, "❌ Failed to forget memory.")

    @commands.command(name="memory-disable", aliases=["mem-disable"])
    async def memory_disable(self, ctx) -> None:
        """Disable memory ingestion for your profile (owner/admin only in guilds).

        In DMs, any user can disable their own memory.
        In guilds, only owner/admin can disable.
        """
        try:
            from bot.core.permissions import is_admin_user

            if ctx.guild and not await is_admin_user(ctx.author, self.bot):
                await ctx.send("🔒 In guilds, only owners/admins can disable memory.")
                return

            await ctx.send("✅ Memory ingestion preference noted. Note: This is a placeholder - full implementation requires user preference persistence.")

        except Exception as e:
            logger.error(f"Error in memory-disable: {e}", exc_info=True)
            await ctx.send("❌ Failed to disable memory.")

    @commands.command(name="memory-enable", aliases=["mem-enable"])
    async def memory_enable(self, ctx) -> None:
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
                await ctx.send("✅ Memory ingestion has been re-enabled for your profile.")
            else:
                await ctx.send("❌ Failed to save memory preference.")

        except Exception as e:
            logger.error(f"Error in memory-enable: {e}", exc_info=True)
            await ctx.send("❌ Failed to enable memory.")

    @commands.command(name="memory-export", aliases=["mem-export"])
    async def memory_export(self, ctx, format: str = "text") -> None:
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
                    file = discord.File(fp=io.BytesIO(json_str.encode()), filename="memories.json")
                    await ctx.send(f"📄 Your memories ({len(records)} records):", file=file)
                else:
                    await ctx.send(f"```\n{json_str}\n```")
            else:
                lines = [f"=== Your Memories ({len(records)}) ==="]
                for idx, r in enumerate(records, 1):
                    lines.append(f"\n{idx}. [{r.memory_id[:8]}] ({r.context_type})\n   {r.summary or r.text}\n   Created: {r.created_at[:10] if r.created_at else 'N/A'}")

                text = "\n".join(lines)
                if len(text) > 1900:
                    file = discord.File(fp=io.BytesIO(text.encode()), filename="memories.txt")
                    await ctx.send(f"📄 Your memories ({len(records)} records):", file=file)
                else:
                    await ctx.send(f"```\n{text}\n```")

        except Exception as e:
            logger.error(f"Error in memory-export: {e}", exc_info=True)
            await ctx.send("❌ Failed to export memories.")

    @commands.command(name="memories-show", aliases=["memories", "mem-show"])
    async def memories_show(self, ctx, limit: int = 5) -> None:
        """Show your most recent stored memories.

        Args:
            limit: Number of memories to show (default: 5, max: 50)

        """
        try:
            limit = min(max(1, int(limit)), 50)
            service = await get_memory_service(self.bot)
            if not service or not service.enabled:
                await ctx.send("❌ Memory service is not enabled.")
                return

            user_id = str(ctx.author.id)
            records = await service.list_user_memories(user_id, limit=limit)

            if not records:
                await ctx.send("🧠 No memories stored yet. Keep chatting!")
                return

            embed = discord.Embed(
                title=f"🧠 Your Memories ({len(records)} total, showing {limit})",
                color=discord.Color.purple(),
            )

            for idx, record in enumerate(records, 1):
                summary = record.summary or record.text or ""
                if len(summary) > 300:
                    summary = summary[:297] + "..."

                embed.add_field(
                    name=f"{idx}. {record.memory_id[:8]}",
                    value=f"{summary}\n*{record.context_type} • {record.created_at[:10] if record.created_at else 'N/A'}*",
                    inline=False,
                )

            embed.set_footer(text="Use !memories-show <n> for more, !memory-forget <id> to delete")
            await ctx.send(embed=embed)

        except Exception as e:
            logger.error(f"Error in memories-show: {e}", exc_info=True)
            await ctx.send("❌ Failed to show memories.")


async def setup(bot) -> None:
    """Add extended memory commands to the bot."""
    if not bot.get_cog("ExtendedMemoryCommands"):
        await bot.add_cog(ExtendedMemoryCommands(bot))
    else:
        logger.warning("'ExtendedMemoryCommands' cog already loaded, skipping setup.")
