"""
Discord commands for video URL ingestion and transcription.
Supports YouTube and TikTok URL processing through STT pipeline.
"""

import re
from typing import Optional
import discord
from discord.ext import commands

from ..util.logging import get_logger
from ..exceptions import InferenceError
from ..hear import hear_infer_from_url

logger = get_logger(__name__)

# URL validation patterns
YOUTUBE_PATTERNS = [
    r"https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+",
    r"https?://youtu\.be/[\w-]+",
]

TIKTOK_PATTERNS = [
    r"https?://(?:www\.)?tiktok\.com/@[\w.-]+/video/\d+",
    r"https?://(?:www\.)?tiktok\.com/t/[\w-]+",
    r"https?://(?:m|vm)\.tiktok\.com/[\w-]+",
]

ALL_PATTERNS = YOUTUBE_PATTERNS + TIKTOK_PATTERNS


class VideoCommands(commands.Cog):
    """Commands for video URL processing and transcription."""

    def __init__(self, bot):
        self.bot = bot
        logger.info("🎥 VideoCommands cog initialized")

    def _extract_url_from_message(self, content: str) -> Optional[str]:
        """Extract first supported URL from message content."""
        for pattern in ALL_PATTERNS:
            match = re.search(pattern, content)
            if match:
                return match.group(0)
        return None

    def _get_url_type(self, url: str) -> str:
        """Determine URL type (YouTube or TikTok)."""
        if any(re.match(pattern, url) for pattern in YOUTUBE_PATTERNS):
            return "YouTube"
        elif any(re.match(pattern, url) for pattern in TIKTOK_PATTERNS):
            return "TikTok"
        else:
            return "Unknown"

    @commands.command(name="watch", aliases=["transcribe", "listen"])
    async def watch_video(self, ctx, url: str = None, *, options: str = ""):
        """
        Transcribe audio from YouTube or TikTok video.

        Usage:
            !watch <url> [--speed 1.5] [--force]
            !transcribe https://youtu.be/example
            !listen https://tiktok.com/@user/video/123 --speed 2.0
        """
        try:
            # Extract URL from command or message content
            if not url:
                url = self._extract_url_from_message(ctx.message.content)
                if not url:
                    await ctx.reply(
                        "❌ Please provide a YouTube or TikTok URL.\n"
                        "**Usage:** `!watch <url>` or `!transcribe <url>`\n"
                        "**Example:** `!watch https://youtu.be/dQw4w9WgXcQ`"
                    )
                    return

            # Parse options
            speedup = 1.5  # default
            force_refresh = False

            if "--speed" in options:
                try:
                    speed_match = re.search(r"--speed\s+([\d.]+)", options)
                    if speed_match:
                        speedup = float(speed_match.group(1))
                        if speedup < 0.5 or speedup > 3.0:
                            await ctx.reply("❌ Speed must be between 0.5x and 3.0x")
                            return
                except ValueError:
                    await ctx.reply("❌ Invalid speed value. Use format: `--speed 1.5`")
                    return

            if "--force" in options:
                force_refresh = True

            url_type = self._get_url_type(url)

            # Send initial processing message
            processing_msg = await ctx.reply(
                f"🎥 Processing {url_type} video...\n"
                f"📊 Speed: {speedup}x | Cache: {'Refresh' if force_refresh else 'Enabled'}\n"
                f"⏳ This may take a moment..."
            )

            # Show typing indicator
            async with ctx.typing():
                # Process the video URL
                result = await hear_infer_from_url(url, speedup, force_refresh)

                transcription = result["transcription"]
                metadata = result["metadata"]

                # Create rich response embed
                embed = discord.Embed(
                    title="🎥 Video Transcription Complete",
                    description=f"**{metadata['title']}**",
                    color=0x00FF00,
                    url=metadata["url"],
                )

                # Add metadata fields
                embed.add_field(
                    name="📊 Details",
                    value=f"**Source:** {metadata['source'].title()}\n"
                    f"**Uploader:** {metadata['uploader']}\n"
                    f"**Duration:** {metadata['original_duration_s']:.1f}s → {metadata['processed_duration_s']:.1f}s\n"
                    f"**Speed:** {metadata['speedup_factor']}x\n"
                    f"**Cache:** {'Hit' if metadata['cache_hit'] else 'Miss'}",
                    inline=True,
                )

                # Add transcription (truncated if too long)
                transcription_preview = transcription
                if len(transcription) > 1000:
                    transcription_preview = transcription[:1000] + "..."

                embed.add_field(
                    name="📝 Transcription",
                    value=f"```\n{transcription_preview}\n```",
                    inline=False,
                )

                # Add footer with processing info
                embed.set_footer(text=f"Processed at {metadata['timestamp'][:19]} UTC")

                # Edit the processing message with results
                await processing_msg.edit(content=None, embed=embed)

                # Log successful transcription
                logger.info(
                    "✅ Video transcription completed",
                    extra={
                        "subsys": "video_ingest",
                        "event": "transcription_complete",
                        "user_id": ctx.author.id,
                        "guild_id": ctx.guild.id if ctx.guild else None,
                        "url": url,
                        "source": metadata["source"],
                        "duration": metadata["original_duration_s"],
                        "cache_hit": metadata["cache_hit"],
                    },
                )

        except InferenceError as e:
            # User-friendly error message
            await processing_msg.edit(content=f"❌ **Transcription Failed**\n{str(e)}")
            logger.warning(
                f"Video transcription failed: {e}",
                extra={
                    "subsys": "video_ingest",
                    "event": "transcription_error",
                    "user_id": ctx.author.id,
                    "url": url,
                },
            )

        except Exception as e:
            # Unexpected error
            await processing_msg.edit(
                content="❌ **Unexpected Error**\n"
                "An unexpected error occurred while processing the video. "
                "Please try again or contact support."
            )
            logger.error(
                f"Unexpected video transcription error: {e}",
                exc_info=True,
                extra={
                    "subsys": "video_ingest",
                    "event": "transcription_error_unexpected",
                    "user_id": ctx.author.id,
                    "url": url,
                },
            )

    @commands.command(name="video-help", aliases=["watch-help"])
    async def video_help(self, ctx):
        """Show help for video transcription commands."""
        embed = discord.Embed(
            title="🎥 Video Transcription Commands",
            description="Transcribe audio from YouTube and TikTok videos",
            color=0x0099FF,
        )

        embed.add_field(
            name="📋 Commands",
            value="`!watch <url>` - Transcribe video audio\n"
            "`!transcribe <url>` - Same as watch\n"
            "`!listen <url>` - Same as watch",
            inline=False,
        )

        embed.add_field(
            name="⚙️ Options",
            value="`--speed <number>` - Set playback speed (0.5x to 3.0x)\n"
            "`--force` - Force re-download (ignore cache)",
            inline=False,
        )

        embed.add_field(
            name="🌐 Supported Sites",
            value="• YouTube (youtube.com, youtu.be)\n"
            "• TikTok (tiktok.com, tiktok.com/t, m.tiktok.com, vm.tiktok.com)",
            inline=False,
        )

        embed.add_field(
            name="📝 Examples",
            value="`!watch https://youtu.be/dQw4w9WgXcQ`\n"
            "`!transcribe https://tiktok.com/@user/video/123 --speed 2.0`\n"
            "`!listen https://youtu.be/example --force`",
            inline=False,
        )

        embed.add_field(
            name="⚠️ Limitations",
            value="• Maximum video length: 10 minutes\n"
            "• Rate limited to prevent abuse\n"
            "• Cached results expire after 7 days",
            inline=False,
        )

        await ctx.reply(embed=embed)

    @commands.command(name="video-cache", aliases=["watch-cache"])
    @commands.has_permissions(administrator=True)
    async def video_cache_info(self, ctx):
        """Show video cache information (Admin only)."""
        try:
            from ..video_ingest import video_manager
            import json

            # Read cache index
            if not video_manager.cache_index_path.exists():
                await ctx.reply("📁 Video cache is empty.")
                return

            with open(video_manager.cache_index_path, "r") as f:
                index = json.load(f)

            if not index:
                await ctx.reply("📁 Video cache is empty.")
                return

            # Calculate cache statistics
            total_entries = len(index)
            total_duration = sum(
                entry.get("duration_seconds", 0) for entry in index.values()
            )

            # Create cache info embed
            embed = discord.Embed(title="📁 Video Cache Information", color=0x0099FF)

            embed.add_field(
                name="📊 Statistics",
                value=f"**Entries:** {total_entries}\n"
                f"**Total Duration:** {total_duration / 60:.1f} minutes\n"
                f"**Cache Directory:** `{video_manager.cache_dir}`",
                inline=False,
            )

            # Show recent entries
            recent_entries = sorted(
                index.items(), key=lambda x: x[1].get("cached_at", ""), reverse=True
            )[:5]

            if recent_entries:
                recent_list = []
                for cache_key, entry in recent_entries:
                    title = entry.get("title", "Unknown")[:30]
                    source = entry.get("source_type", "unknown").title()
                    duration = entry.get("duration_seconds", 0)
                    recent_list.append(f"• **{title}** ({source}, {duration:.0f}s)")

                embed.add_field(
                    name="🕒 Recent Entries", value="\n".join(recent_list), inline=False
                )

            await ctx.reply(embed=embed)

        except Exception as e:
            await ctx.reply(f"❌ Error reading cache information: {str(e)}")
            logger.error(f"Error reading video cache info: {e}", exc_info=True)


async def setup(bot):
    """Set up the video commands cog."""
    await bot.add_cog(VideoCommands(bot))
    logger.info("✅ VideoCommands cog loaded")
