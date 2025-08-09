"""
Discord commands for explicit screenshot capture and analysis.

Implements the `!ss` command which is strictly command-gated to
capture a screenshot of a URL using the configured external API and
then analyze it via the existing VL flow.

Privacy/Security:
- No hardcoded secrets; relies entirely on environment variables used by
  `bot.utils.external_api.external_screenshot`.
- No automatic screenshot fallback anywhere; this command is the only
  intentional trigger for screenshots.
"""

from __future__ import annotations

import re
from typing import Optional, TYPE_CHECKING

import discord
from discord.ext import commands

from ..util.logging import get_logger
from ..modality import InputItem

if TYPE_CHECKING:  # type hints only
    from bot.core.bot import LLMBot

logger = get_logger(__name__)

# Simple URL pattern (consistent with modality URL extraction) [IV]
_URL_PATTERN = re.compile(r"https?://[^\s<>\"'\[\]{}|\\^`]+")


class ScreenshotCommands(commands.Cog):
    """Commands for explicit screenshot capture and VL analysis."""

    def __init__(self, bot: "LLMBot") -> None:
        self.bot = bot
        logger.info("📸 ScreenshotCommands cog initialized")

    def _extract_first_url(self, text: str) -> Optional[str]:
        """Extract the first URL from the provided text. [IV]"""
        if not text:
            return None
        m = _URL_PATTERN.search(text)
        return m.group(0) if m else None

    @commands.command(name="ss", aliases=["screenshot"])
    async def screenshot_cmd(self, ctx: commands.Context, url: Optional[str] = None) -> None:
        """
        Take a screenshot of the given URL and analyze it with the vision model.

        Usage:
            !ss <url>
            !screenshot <url>
        """
        try:
            # Resolve URL from argument or message content [IV]
            if not url:
                url = self._extract_first_url(ctx.message.content)

            if not url:
                await ctx.reply(
                    "❌ Please provide a valid URL to screenshot.\n"
                    "**Usage:** `!ss <url>`\n"
                    "**Example:** `!ss https://example.com`"
                )
                return

            # Validate we have an initialized router
            router = getattr(self.bot, "router", None)
            if router is None:
                await ctx.reply("⚠️ Router not initialized. Please try again shortly.")
                logger.error("Router is None when invoking !ss")
                return

            # Construct InputItem and delegate to Router's explicit screenshot handler
            item = InputItem(source_type="url", payload=url, order_index=0)

            # Provide quick processing notice
            processing_msg = await ctx.reply(
                f"📸 Capturing screenshot for: {url}\n"
                f"🔒 Privacy: explicit command-gated; no background screenshots."
            )

            # Show typing indicator while processing
            async with ctx.typing():
                result_text = await router._handle_screenshot_url(item)

            # Build a compact embed for better UX
            embed = discord.Embed(
                title="📸 Screenshot Result",
                description=result_text[:4000],  # Discord limit guard [PA]
                color=0x00AEEF,
            )
            embed.set_footer(text="Screenshots are opt-in via !ss · No secrets stored")

            await processing_msg.edit(content=None, embed=embed)

        except Exception as e:
            # Robust error handling with friendly message [REH]
            logger.error(f"❌ !ss command failed: {e}", exc_info=True)
            await ctx.reply(
                "❌ Failed to capture or analyze the screenshot. "
                "This might be a temporary service issue. Please try again."
            )


async def setup(bot: "LLMBot") -> None:
    """Set up the screenshot commands cog."""
    await bot.add_cog(ScreenshotCommands(bot))
    logger.info("✅ ScreenshotCommands cog loaded")
