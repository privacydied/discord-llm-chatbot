"""
Context management commands for the enhanced contextual conversation manager.
"""

import discord
from discord.ext import commands

from bot.utils.logging import get_logger
from bot.contextual_brain import create_context_command_handler

logger = get_logger(__name__)


class ContextCommands(commands.Cog):
    """Commands for managing conversation context and privacy settings."""

    def __init__(self, bot) -> None:
        self.bot = bot
        self.context_handlers = create_context_command_handler(bot)
        logger.info("✔ Context commands loaded")

    @commands.command(name="context_reset", aliases=["reset_context", "clear_context"])
    async def context_reset(self, ctx: commands.Context) -> None:
        """Reset the conversation context for this channel/DM."""
        try:
            response = await self.context_handlers["context_reset"](ctx.message)
            await ctx.reply(response, mention_author=False)
            logger.info(f"✔ Context reset requested by {ctx.author.id} in {ctx.channel.id}")
        except Exception as e:
            logger.error(f"❌ Context reset failed: {e}")
            await ctx.reply("❌ Failed to reset context.", mention_author=False)

    @commands.command(name="context_stats", aliases=["ctx_stats"])
    async def context_stats(self, ctx: commands.Context) -> None:
        """Show context manager statistics."""
        try:
            response = await self.context_handlers["context_stats"](ctx.message)
            await ctx.reply(response, mention_author=False)
            logger.info(f"✔ Context stats requested by {ctx.author.id}")
        except Exception as e:
            logger.error(f"❌ Context stats failed: {e}")
            await ctx.reply("❌ Failed to get context stats.", mention_author=False)

    @commands.command(name="privacy_optout", aliases=["opt_out", "no_context"])
    async def privacy_optout(self, ctx: commands.Context) -> None:
        """Opt out of conversation context tracking."""
        try:
            response = await self.context_handlers["privacy_optout"](ctx.message)
            await ctx.reply(response, mention_author=False)
            logger.info(f"✔ Privacy opt-out by {ctx.author.id}")
        except Exception as e:
            logger.error(f"❌ Privacy opt-out failed: {e}")
            await ctx.reply("❌ Failed to process privacy opt-out.", mention_author=False)

    @commands.command(name="privacy_optin", aliases=["opt_in", "enable_context"])
    async def privacy_optin(self, ctx: commands.Context) -> None:
        """Opt back into conversation context tracking."""
        try:
            response = await self.context_handlers["privacy_optin"](ctx.message)
            await ctx.reply(response, mention_author=False)
            logger.info(f"✔ Privacy opt-in by {ctx.author.id}")
        except Exception as e:
            logger.error(f"❌ Privacy opt-in failed: {e}")
            await ctx.reply("❌ Failed to process privacy opt-in.", mention_author=False)

    @commands.command(name="context_help", aliases=["ctx_help"])
    async def context_help(self, ctx: commands.Context) -> None:
        """Show help for context management commands."""
        help_text = """
**🧠 Enhanced Context Management Commands**

**Context Control:**
• `!context_reset` - Reset conversation history for this channel/DM
• `!context_stats` - Show context manager statistics

**Privacy Controls:**
• `!privacy_optout` - Opt out of conversation context tracking
• `!privacy_optin` - Opt back into conversation context tracking

**Features:**
✔ **Multi-user conversations** - Tracks context across users in shared channels
✔ **Privacy-first** - DMs are never saved to disk, opt-out available
✔ **Encrypted storage** - All content encrypted at rest
✔ **Smart context** - Includes relevant conversation history in responses
✔ **Configurable** - History window controlled by `HISTORY_WINDOW` env var

**Environment Variables:**
• `HISTORY_WINDOW` - Max messages per user/channel (default: 10)
• `USE_ENHANCED_CONTEXT` - Enable enhanced context (default: true)
• `IN_MEMORY_CONTEXT_ONLY` - Keep context in memory only (default: false)
        """

        embed = discord.Embed(
            title="🧠 Enhanced Context Management",
            description=help_text.strip(),
            color=0x00FF00,
        )

        await ctx.reply(embed=embed, mention_author=False)


async def setup(bot) -> None:
    """Set up context commands."""
    await bot.add_cog(ContextCommands(bot))
    logger.info("✔ Context commands cog loaded")
