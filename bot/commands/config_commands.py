"""
Discord commands for configuration management and dynamic reloading.
"""

import discord
from discord.ext import commands

from ..config_reload import (
    manual_reload_command,
    get_config_for_debug,
    get_config_version,
)
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ConfigCommands(commands.Cog):
    """Commands for configuration management and dynamic reloading."""

    def __init__(self, bot):
        self.bot = bot

    @commands.command(name="reload-config", aliases=["reload_config", "config_reload"])
    @commands.has_permissions(administrator=True)
    async def reload_config(self, ctx):
        """Manually reload configuration from .env file (Admin only)."""
        try:
            logger.info(
                f"🔄 Manual config reload requested by {ctx.author.id} in {ctx.guild.id if ctx.guild else 'DM'}"
            )

            # Show typing indicator
            async with ctx.typing():
                result_message = manual_reload_command()

            await ctx.reply(result_message, mention_author=False)
            logger.info(f"✅ Manual config reload completed for {ctx.author.id}")

        except commands.MissingPermissions:
            await ctx.reply(
                "❌ You need administrator permissions to reload configuration.",
                mention_author=False,
            )
        except Exception as e:
            logger.error(f"❌ Manual config reload failed: {e}")
            await ctx.reply(
                "❌ Failed to reload configuration. Check logs for details.",
                mention_author=False,
            )

    @commands.command(name="config-status", aliases=["config_status", "config_info"])
    @commands.has_permissions(administrator=True)
    async def config_status(self, ctx):
        """Show current configuration status and version (Admin only)."""
        try:
            version = get_config_version()
            config_debug = get_config_for_debug()

            # Create embed with config info
            embed = discord.Embed(
                title="🔧 Configuration Status",
                color=discord.Color.blue(),
                timestamp=discord.utils.utcnow(),
            )

            embed.add_field(name="📋 Version", value=f"`{version}`", inline=True)

            embed.add_field(
                name="📊 Total Settings",
                value=f"{len(config_debug)} variables",
                inline=True,
            )

            # Show some key non-sensitive settings
            key_settings = []
            important_keys = [
                "TEXT_BACKEND",
                "OPENAI_TEXT_MODEL",
                "OLLAMA_MODEL",
                "TTS_BACKEND",
                "STT_ENGINE",
                "LOG_LEVEL",
                "MAX_USER_MEMORY",
                "MAX_SERVER_MEMORY",
                "HISTORY_WINDOW",
            ]

            for key in important_keys:
                if key in config_debug:
                    value = config_debug[key]
                    if value is not None:
                        key_settings.append(f"`{key}`: {value}")

            if key_settings:
                embed.add_field(
                    name="🔑 Key Settings",
                    value="\n".join(
                        key_settings[:10]
                    ),  # Limit to avoid embed size issues
                    inline=False,
                )

            embed.set_footer(text="Use !reload-config to reload from .env file")

            await ctx.reply(embed=embed, mention_author=False)
            logger.info(f"✅ Config status shown to {ctx.author.id}")

        except commands.MissingPermissions:
            await ctx.reply(
                "❌ You need administrator permissions to view configuration status.",
                mention_author=False,
            )
        except Exception as e:
            logger.error(f"❌ Config status command failed: {e}")
            await ctx.reply(
                "❌ Failed to retrieve configuration status.", mention_author=False
            )

    @commands.command(name="config-help", aliases=["config_help"])
    async def config_help(self, ctx):
        """Show help information about configuration commands."""
        embed = discord.Embed(
            title="🔧 Configuration Commands Help",
            description="Commands for managing bot configuration",
            color=discord.Color.green(),
        )

        embed.add_field(
            name="!reload-config",
            value="Manually reload configuration from .env file\n*Requires: Administrator*",
            inline=False,
        )

        embed.add_field(
            name="!config-status",
            value="Show current configuration version and key settings\n*Requires: Administrator*",
            inline=False,
        )

        embed.add_field(
            name="🔄 Automatic Reloading",
            value="Configuration automatically reloads when:\n"
            "• .env file is modified (file watcher)\n"
            "• SIGHUP signal is sent (Unix systems)\n"
            "• Manual reload command is used",
            inline=False,
        )

        embed.add_field(
            name="📋 Hot-Reload Support",
            value="Most settings take effect immediately:\n"
            "• Model settings (TEXT_MODEL, etc.)\n"
            "• Memory limits (MAX_USER_MEMORY, etc.)\n"
            "• TTS/STT engine settings\n"
            "• Log levels and debug flags",
            inline=False,
        )

        embed.set_footer(
            text="Changes are logged with before/after values (sensitive data redacted)"
        )

        await ctx.reply(embed=embed, mention_author=False)


async def setup(bot):
    """Set up the config commands cog."""
    await bot.add_cog(ConfigCommands(bot))
    logger.info("✅ ConfigCommands cog loaded")
