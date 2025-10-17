"""
Discord commands for manual janitor operations.
"""

import discord
from discord.ext import commands

from ..janitor import manual_clean
from ..utils.logging import get_logger

logger = get_logger(__name__)


class JanitorCommands(commands.Cog):
    """Commands for manual cache and log cleanup operations."""

    def __init__(self, bot):
        self.bot = bot

    @commands.command(name="clean", aliases=["cleanup", "janitor"])
    @commands.has_permissions(administrator=True)
    async def clean(self, ctx):
        """Manually trigger cache and log cleanup (Admin only)."""
        try:
            logger.info(
                f"🧹 Manual clean requested by {ctx.author.id} in {ctx.guild.id if ctx.guild else 'DM'}",
                extra={
                    "subsys": "janitor",
                    "event": "janitor.manual_clean_requested",
                    "detail": {
                        "user_id": ctx.author.id,
                        "guild_id": ctx.guild.id if ctx.guild else None,
                    },
                },
            )

            # Show typing indicator
            async with ctx.typing():
                result = await manual_clean()

            # Format response based on result
            if result.get("success"):
                files_deleted = result.get("total_files_deleted", 0)
                bytes_freed = result.get("total_bytes_freed", 0)
                logs_compressed = result.get("logs_compressed", 0)
                dirs_processed = result.get("directories_processed", 0)

                # Convert bytes to human-readable format
                if bytes_freed < 1024:
                    size_str = f"{bytes_freed} B"
                elif bytes_freed < 1024 * 1024:
                    size_str = f"{bytes_freed / 1024:.1f} KB"
                elif bytes_freed < 1024 * 1024 * 1024:
                    size_str = f"{bytes_freed / (1024 * 1024):.1f} MB"
                else:
                    size_str = f"{bytes_freed / (1024 * 1024 * 1024):.2f} GB"

                embed = discord.Embed(
                    title="🧹 Manual Cleanup Complete",
                    color=discord.Color.green(),
                    timestamp=discord.utils.utcnow(),
                )

                embed.add_field(
                    name="📁 Directories Processed",
                    value=f"{dirs_processed}",
                    inline=True,
                )

                embed.add_field(
                    name="🗑️ Files Deleted",
                    value=f"{files_deleted}",
                    inline=True,
                )

                embed.add_field(
                    name="💾 Space Freed",
                    value=size_str,
                    inline=True,
                )

                if logs_compressed > 0:
                    embed.add_field(
                        name="📦 Logs Compressed",
                        value=f"{logs_compressed}",
                        inline=True,
                    )

                if files_deleted == 0 and logs_compressed == 0:
                    embed.description = "✨ Everything is already clean! No files needed cleanup."
                else:
                    embed.description = "✅ Cache and log cleanup completed successfully."

                embed.set_footer(
                    text="Automatic cleanup runs every 60 minutes in the background"
                )

                await ctx.reply(embed=embed, mention_author=False)
                logger.info(f"✅ Manual clean completed for {ctx.author.id}")

            else:
                error = result.get("error", "Unknown error")
                embed = discord.Embed(
                    title="❌ Cleanup Failed",
                    description=f"Failed to complete cleanup: {error}",
                    color=discord.Color.red(),
                )
                await ctx.reply(embed=embed, mention_author=False)
                logger.error(f"❌ Manual clean failed: {error}")

        except commands.MissingPermissions:
            await ctx.reply(
                "❌ You need administrator permissions to run cleanup commands.",
                mention_author=False,
            )
        except Exception as e:
            logger.error(f"❌ Manual clean command failed: {e}", exc_info=True)
            await ctx.reply(
                "❌ Failed to run cleanup. Check logs for details.",
                mention_author=False,
            )

    @commands.command(name="clean-status", aliases=["cleanup-status", "janitor-status"])
    @commands.has_permissions(administrator=True)
    async def clean_status(self, ctx):
        """Show janitor configuration and status (Admin only)."""
        try:
            from ..janitor import (
                JANITOR_INTERVAL_MINUTES,
                HOLD_OFF_MINUTES,
                LOG_RETENTION_DAYS,
                LOG_TOTAL_CAP_MB,
            )

            embed = discord.Embed(
                title="🧹 Janitor Status",
                description="Automatic cache and log cleanup configuration",
                color=discord.Color.blue(),
                timestamp=discord.utils.utcnow(),
            )

            embed.add_field(
                name="⏰ Schedule",
                value=f"Every {JANITOR_INTERVAL_MINUTES} minutes\n(±5 min jitter)",
                inline=True,
            )

            embed.add_field(
                name="🛡️ Safety Hold-off",
                value=f"{HOLD_OFF_MINUTES} minutes",
                inline=True,
            )

            embed.add_field(
                name="📋 Log Retention",
                value=f"{LOG_RETENTION_DAYS} days\n{LOG_TOTAL_CAP_MB} MB cap",
                inline=True,
            )

            # Directory policies
            policies_text = []
            policies_text.append("**Logs**: 7d, 256 MB")
            policies_text.append("**Video/Audio**: 3d, 2 GB")
            policies_text.append("**STT Cache**: 24h, 1 GB")
            policies_text.append("**TTS Cache**: 24h, 512 MB")
            policies_text.append("**HTTP Cache**: 24h, 512 MB")
            policies_text.append("**Temp**: 6h")

            embed.add_field(
                name="📁 Directory Policies",
                value="\n".join(policies_text),
                inline=False,
            )

            embed.add_field(
                name="🔧 Operations",
                value="• Compress logs older than 1 hour\n"
                "• Prune by age and size\n"
                "• Skip files modified in last 30 min\n"
                "• Max 500 files per run per directory",
                inline=False,
            )

            embed.set_footer(text="Use !clean to manually trigger cleanup now")

            await ctx.reply(embed=embed, mention_author=False)

        except commands.MissingPermissions:
            await ctx.reply(
                "❌ You need administrator permissions to view janitor status.",
                mention_author=False,
            )
        except Exception as e:
            logger.error(f"❌ Clean status command failed: {e}", exc_info=True)
            await ctx.reply(
                "❌ Failed to retrieve janitor status.", mention_author=False
            )

    @commands.command(name="clean-help", aliases=["cleanup-help", "janitor-help"])
    async def clean_help(self, ctx):
        """Show help information about janitor/cleanup commands."""
        embed = discord.Embed(
            title="🧹 Janitor Commands Help",
            description="Commands for cache and log cleanup",
            color=discord.Color.green(),
        )

        embed.add_field(
            name="!clean",
            value="Manually trigger cache and log cleanup now\n*Requires: Administrator*",
            inline=False,
        )

        embed.add_field(
            name="!clean-status",
            value="Show janitor configuration and directory policies\n*Requires: Administrator*",
            inline=False,
        )

        embed.add_field(
            name="🤖 Automatic Cleanup",
            value="The janitor runs automatically every 60 minutes:\n"
            "• Rotates and compresses logs\n"
            "• Prunes old cache files by age\n"
            "• Enforces size caps on cache directories\n"
            "• Never touches files modified in last 30 minutes",
            inline=False,
        )

        embed.add_field(
            name="🛡️ Safety Features",
            value="• Hold-off window protects in-flight files\n"
            "• Never deletes active log files\n"
            "• Batch processing (max 500 files/run)\n"
            "• Cross-platform compatible\n"
            "• Comprehensive error handling",
            inline=False,
        )

        embed.set_footer(text="Janitor keeps disk usage under control for long-running bots")

        await ctx.reply(embed=embed, mention_author=False)


async def setup(bot):
    """Set up the janitor commands cog."""
    await bot.add_cog(JanitorCommands(bot))
    logger.info("✅ JanitorCommands cog loaded")
