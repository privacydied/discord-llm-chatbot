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
from typing import Optional, TYPE_CHECKING, Dict, Any

import discord
from discord.ext import commands

from ..utils.logging import get_logger
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
    async def screenshot_cmd(
        self, ctx: commands.Context, url: Optional[str] = None
    ) -> None:
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

            # Construct InputItem
            item = InputItem(source_type="url", payload=url, order_index=0)

            # Streaming card config
            cfg = getattr(self.bot, "config", {}) or {}
            streaming_on = bool(cfg.get("STREAMING_ENABLE", True))
            style = str(cfg.get("STREAMING_EMBED_STYLE", "compact"))
            tick_ms = int(cfg.get("STREAMING_TICK_MS", 750))

            # Prepare initial embed message
            def make_embed(
                stage_idx: int, stages: Dict[int, Dict[str, Any]]
            ) -> discord.Embed:
                title = "📸 Screenshot"
                color = 0x00AEEF
                if stage_idx >= 6:
                    title = "📸 Screenshot Complete"
                    color = 0x10C080
                desc_lines = []
                for i in range(1, 7):
                    meta = stages[i]
                    status = meta["status"]
                    label = meta["label"]
                    if status == "done":
                        icon = "✔"
                    elif status == "active":
                        icon = "⏳"
                    else:
                        icon = "•"
                    if style == "detailed":
                        desc_lines.append(f"{icon} {i}/6 {label}")
                    else:
                        desc_lines.append(f"{icon} {label}")
                embed = discord.Embed(
                    title=title,
                    description="\n".join(desc_lines)[:4000],
                    color=color,
                )
                embed.set_footer(
                    text="Screenshots are opt-in via !ss · No secrets stored"
                )
                return embed

            # Initialize stages map
            stages: Dict[int, Dict[str, Any]] = {
                1: {"key": "validate", "label": "Validate URL", "status": "queued"},
                2: {"key": "prepare", "label": "Prepare capture", "status": "queued"},
                3: {"key": "capture", "label": "Capture page", "status": "queued"},
                4: {"key": "saved", "label": "Save to cache", "status": "queued"},
                5: {
                    "key": "analyze",
                    "label": "Analyze screenshot",
                    "status": "queued",
                },
                6: {"key": "done", "label": "Complete", "status": "queued"},
            }

            # Send initial message
            processing_msg = await ctx.reply(
                content=f"🔒 Privacy: explicit command-gated; no background screenshots.\nURL: {url}",
                embed=make_embed(0, stages),
            )

            last_edit = 0.0

            async def update_stage(stage_num: int):
                nonlocal last_edit
                # monotonic progression
                for i in range(1, stage_num):
                    if stages[i]["status"] != "done":
                        stages[i]["status"] = "done"
                # set active for current
                if stage_num <= 6:
                    if stages[stage_num]["status"] != "done":
                        stages[stage_num]["status"] = "active"
                # throttle edits
                now = discord.utils.utcnow().timestamp()
                if (now - last_edit) * 1000.0 < tick_ms:
                    return
                last_edit = now
                try:
                    await processing_msg.edit(embed=make_embed(stage_num, stages))
                except Exception as edit_err:
                    logger.debug(f"Streaming edit skipped: {edit_err}")

            # Progress callback for router
            async def progress_cb(phase: str, step: int) -> None:
                mapping = {
                    "validate": 1,
                    "prepare": 2,
                    "capture": 3,
                    "saved": 4,
                    "analyze": 5,
                    "done": 6,
                }
                stage_num = mapping.get(phase, step if 1 <= step <= 6 else 1)
                await update_stage(stage_num)

            # If streaming disabled, fall back to one-shot UX
            if not streaming_on:
                async with ctx.typing():
                    result_text = await router._handle_screenshot_url(item)
                embed = discord.Embed(
                    title="📸 Screenshot Result",
                    description=result_text[:4000],
                    color=0x00AEEF,
                )
                embed.set_footer(
                    text="Screenshots are opt-in via !ss · No secrets stored"
                )
                await processing_msg.edit(content=None, embed=embed)
                return

            # Streaming path
            await update_stage(1)
            await update_stage(2)
            async with ctx.typing():
                result_text = await router._handle_screenshot_url(
                    item, progress_cb=progress_cb
                )
            await update_stage(6)

            # Finalize with result embed
            final_embed = discord.Embed(
                title="📸 Screenshot Result",
                description=result_text[:4000],
                color=0x10C080,
            )
            final_embed.set_footer(
                text="Screenshots are opt-in via !ss · No secrets stored"
            )
            await processing_msg.edit(embed=final_embed)

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
