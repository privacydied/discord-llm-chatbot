"""
Event handlers for the Discord bot.
"""
import asyncio
import asyncio
import re
import sys
from pathlib import Path
from typing import Optional

import discord
from discord.ext import commands

from bot.logger import get_logger
from bot.router import get_router
from bot.tts_manager import tts_manager
from bot.config import load_config

logger = get_logger(__name__)

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

    def _get_log_extra(self, message: discord.Message) -> dict:
        """Helper to create the extra dictionary for structured logging."""
        return {
            'subsys': 'events',
            'guild_id': message.guild.id if message.guild else None,
            'user_id': message.author.id,
            'msg_id': message.id
        }

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        """Main message handler that gates and dispatches messages."""
        if not self._is_relevant(message):
            return

        extra = self._get_log_extra(message)
        logger.info(
            f"[WIND][EVENT] Relevant message received. Type: {'DM' if isinstance(message.channel, discord.DMChannel) else 'Mention'}",
            extra={**extra, 'event': 'message_received'}
        )

        # Let the commands extension handle actual commands first
        # This prevents the router from processing a message that is also a command
        ctx = await self.bot.get_context(message)
        if ctx.valid:
            logger.info(
                f"[WIND][EVENT] Message is a valid command, letting command processor handle it: {ctx.command.name}",
                extra={**extra, 'event': 'command_dispatch'}
            )
            return

        # If not a command, dispatch to the router
        try:
            # The router now handles all message types, including attachments.
            # We just need to strip the mention and pass the raw content.
            if self.bot.user.mentioned_in(message):
                # Strip the mention from the content to get the raw input
                raw_input = message.content.replace(f'<@!{self.bot.user.id}>', '', 1).replace(f'<@{self.bot.user.id}>', '', 1).strip()
            else:
                # For DMs, the content is the raw input
                raw_input = message.content

            await self.router.handle(message, raw_input)

            logger.info("[WIND][EVENT] Message dispatched to router successfully.", extra={**extra, 'event': 'router_dispatch_success'})

        except Exception as e:
            logger.error(
                f"[WIND][EVENT] Error processing message: {e}",
                exc_info=True,
                extra={**extra, 'event': 'router_fail'}
            )

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
            if not tts_manager.is_available():
                await ctx.send("🔄 TTS initializing—please retry in a moment.")
                # Fall back to text if TTS is not available
                if mode == "both":
                    await ctx.send(text_response)
            else:
                try:
                    # Create temporary audio file
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                        audio_path = Path(tmpfile.name)
                    
                    # Synthesize audio asynchronously
                    audio_path = await tts_manager.synthesize_async(text_response, audio_path)
                    
                    if audio_path and audio_path.exists():
                        # Send only voice response when TTS is active
                        await ctx.send(file=discord.File(str(audio_path), filename="tts.ogg"))
                    else:
                        logging.error(f"TTS synthesis failed for: {text_response}")
                        # Fall back to text on TTS failure
                        await ctx.send(text_response)
                except Exception as e:
                    logging.error(f"TTS synthesis error: {e}")
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
            logging.error(f"Brain inference failed: {str(e)}")
            return "⚠️ An error occurred while generating the response"

# Background task for cache maintenance
async def cache_maintenance_task():
    """Background task to clean up old cache files."""
    logger.info("Starting TTS cache maintenance task", extra={'subsys': 'tts_cache', 'event': 'task_start'})
    while True:
        try:
            logger.debug("Running TTS cache maintenance...", extra={'subsys': 'tts_cache', 'event': 'task_run'})
            # Get stats before cleanup
            stats_before = tts_manager.get_cache_stats()
            logger.debug(f"Cache stats before cleanup: {stats_before}", extra={'subsys': 'tts_cache', 'event': 'stats_before'})
            
            # Run the cleanup
            tts_manager.purge_old_cache()  # This is a synchronous method
            
            # Get stats after cleanup
            stats_after = tts_manager.get_cache_stats()
            logger.info(f"TTS cache maintenance completed. Stats before: {stats_before}, after: {stats_after}", extra={'subsys': 'tts_cache', 'event': 'task_success'})
            
            # Wait for 24 hours before next run
            logger.debug("Next TTS cache maintenance in 24 hours...", extra={'subsys': 'tts_cache', 'event': 'task_sleep'})
            await asyncio.sleep(86400)  # Run daily
            
        except Exception as e:
            logger.error(f"TTS cache maintenance failed: {str(e)}", exc_info=True, extra={'subsys': 'tts_cache', 'event': 'task_fail'})
            logger.info("Retrying TTS cache maintenance in 1 hour...", extra={'subsys': 'tts_cache', 'event': 'task_retry'})
            await asyncio.sleep(3600)  # Retry in 1 hour

async def setup(bot) -> None:
    """Set up event handlers."""
    await bot.add_cog(BotEventHandler(bot))
    logger.info("Event handlers loaded.", extra={'subsys': 'core', 'event': 'cog_load_success', 'cog': __name__})
