"""
Event handlers for the Discord bot.
"""
import asyncio
from pathlib import Path

import discord
from discord.ext import commands

from bot.util.logging import get_logger
from bot.router import get_router
from bot.config import load_config
from .hear import hear_infer

logger = get_logger(__name__)

# --- Minimal image attachment utilities for tests and multimodal routing ---
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff"}

def _attachment_is_image(att) -> bool:
    """Return True if a Discord attachment looks like an image.

    Duck-typed to work with unittest mocks in tests.
    """
    try:
        ctype = getattr(att, "content_type", None)
        if isinstance(ctype, str) and ctype.lower().startswith("image/"):
            return True
        name = getattr(att, "filename", "") or ""
        import os
        _, ext = os.path.splitext(name.lower())
        return ext in IMAGE_EXTS
    except Exception:
        return False

def has_image_attachments(message) -> bool:
    """Whether the message has at least one image attachment.

    Accepts a mock with an `attachments` list (as used in tests).
    """
    atts = getattr(message, "attachments", None) or []
    return any(_attachment_is_image(att) for att in atts)

def get_image_urls(message) -> list:
    """Return URLs for image attachments on the message.

    Only includes attachments detected as images.
    """
    urls = []
    for att in getattr(message, "attachments", None) or []:
        if _attachment_is_image(att):
            url = getattr(att, "url", None)
            if url:
                urls.append(url)
    return urls

class BotEventHandler(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.router = get_router()
        self.config = load_config()

    def _is_relevant(self, message: discord.Message) -> bool:
        """Determines if a message is relevant for the bot to process."""
        # Ignore messages from bots
        if message.author.bot:
            return False

        # Process all DMs
        if isinstance(message.channel, discord.DMChannel):
            return True

        # In guilds, only process if mentioned
        if self.bot.user.mentioned_in(message):
            return True

        return False

    # NOTE:
    # This file's message handling is obsolete; routing is handled in bot.py (on_message).
    # The previous top-level await-based block has been removed to avoid SyntaxError.

    async def _is_admin(self, user: discord.Member, guild: discord.Guild) -> bool:
        """Check if user has admin permissions."""
        if guild is None:
            return False

        # Check Discord permissions
        if user.guild_permissions.administrator:
            return True
        
        return False
    
    async def handle_hybrid_reply(self, ctx, text_response: str, mode: str = "both"):
        """Hybrid inference pipeline for text and TTS responses."""
        # For TTS modes, synthesize and send audio
        if mode in ("tts", "both"):
            if not self.bot.tts_manager.is_available():
                await ctx.send("🔄 TTS initializing—please retry in a moment.")
                # Fall back to text if TTS is not available
                if mode == "both":
                    await ctx.send(text_response)
            else:
                try:
                    # Generate OGG/Opus asynchronously via TTSManager
                    audio_path, mime_type = await self.bot.tts_manager.generate_tts(
                        text_response,
                        output_format="ogg",
                    )

                    if audio_path and Path(audio_path).exists():
                        # Send only voice response when TTS is active
                        await ctx.send(file=discord.File(str(audio_path), filename="tts.ogg"))
                    else:
                        logger.error(f"TTS synthesis produced no file for: {text_response}")
                        # Fall back to text on TTS failure
                        await ctx.send(text_response)
                except Exception as e:
                    logger.error(f"TTS synthesis error: {e}", exc_info=True)
                    await ctx.send("⚠️ TTS synthesis failed")
                    # Fall back to text on error
                    await ctx.send(text_response)
        
        # For text-only mode, send text response
        elif mode == "text":
            await ctx.send(text_response)
    
    async def _generate_bot_reply(self, message: discord.Message) -> str:
        """Generate bot reply using AI backend."""
        from .brain import brain_infer
        from .exceptions import InferenceError
        
        try:
            return await brain_infer(message.content)
        except InferenceError as e:
            logger.error(f"Brain inference failed: {str(e)}")
            return "⚠️ An error occurred while generating the response"

# Background task for cache maintenance
async def cache_maintenance_task(bot: commands.Bot):
    """Background task to clean up old cache files."""
    logger.info("Starting TTS cache maintenance task [subsys: tts_cache, event: task_start]")
    while True:
        try:
            logger.debug("Running TTS cache maintenance... [subsys: tts_cache, event: task_run]")
            # Get stats before cleanup
            stats_before = bot.tts_manager.get_cache_stats()
            logger.debug(f"Cache stats before cleanup: {stats_before} [subsys: tts_cache, event: stats_before]")
            
            # Run the cleanup
            bot.tts_manager.purge_old_cache()  # This is a synchronous method
            
            # Get stats after cleanup
            stats_after = bot.tts_manager.get_cache_stats()
            logger.info(f"TTS cache maintenance completed. Stats before: {stats_before}, after: {stats_after} [subsys: tts_cache, event: task_success]")
            
            # Wait for 24 hours before next run
            logger.debug("Next TTS cache maintenance in 24 hours... [subsys: tts_cache, event: task_sleep]")
            await asyncio.sleep(86400)  # Run daily
            
        except Exception as e:
            logger.error(f"TTS cache maintenance failed: {str(e)} [subsys: tts_cache, event: task_fail]", exc_info=True)
            logger.info("Retrying TTS cache maintenance in 1 hour... [subsys: tts_cache, event: task_retry]")
            await asyncio.sleep(3600)  # Retry in 1 hour

async def setup(bot) -> None:
    """Set up event handlers."""
    if 'BotEventHandler' not in bot.cogs:
        await bot.add_cog(BotEventHandler(bot))
        logger.info(f"Event handlers loaded. [subsys: core, event: cog_load_success, cog: {__name__}]")
    else:
        logger.debug("BotEventHandler already loaded.")
