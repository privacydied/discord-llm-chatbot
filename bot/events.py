"""
Event handlers for the Discord bot.
"""
import asyncio
import logging
import re
import sys
from pathlib import Path
from typing import Optional

import discord
from discord.ext import commands

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

from bot.router import setup_router, get_router
from bot.tts_manager import tts_manager
from bot.tts_state import tts_state
from bot.config import load_config

class BotEventHandler(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.config = load_config()
        self.router = setup_router(bot)
        self.prefix = self.config.get('COMMAND_PREFIX', '!')
    
    def _strip_prefix(self, content: str) -> str:
        """Strip command prefix from message content."""
        if content.startswith(self.prefix):
            return content[len(self.prefix):].strip()
        return content.strip()
    
    def _strip_mention(self, content: str) -> str:
        """Remove bot mention from message content."""
        # Remove user-style mention
        content = re.sub(rf'<@{self.bot.user.id}>', '', content)
        # Remove nickname-style mention
        content = re.sub(rf'<@!{self.bot.user.id}>', '', content)
        return content.strip()
    
    async def dispatch_message(self, message: discord.Message) -> None:
        """
        Process a message through the appropriate pipeline.
        Handles image attachments automatically, otherwise uses the router.
        """
        # Skip bot messages
        if message.author.bot:
            return
        
        # Check if message should be processed
        is_dm = isinstance(message.channel, discord.DMChannel)
        is_mentioned = self.bot.user.mentioned_in(message)
        has_prefix = message.content.startswith(self.prefix)
        has_image = any(att.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')) 
                      for att in message.attachments)
        has_document = any(att.filename.lower().endswith(('.pdf', '.txt', '.md', '.docx'))
                         for att in message.attachments)
        has_audio = any(att.filename.lower().endswith(('.wav', '.mp3', '.ogg'))
                       for att in message.attachments)
        
        # Skip if no trigger and no attachments that can be auto-processed
        if not (has_prefix or is_mentioned or is_dm or has_image or has_document or has_audio):
            return
        
        # Get clean input (strip prefix/mention)
        content = message.content
        if has_prefix:
            content = self._strip_prefix(content)
        elif is_mentioned:
            content = self._strip_mention(content)
        
        # Check if this is a Discord command that will be processed by command handlers
        raw_content = message.content.strip()
        is_discord_command = False
        
        # In DMs: check for !command
        if is_dm and raw_content.startswith('!'):
            command_part = raw_content[1:].split()[0] if raw_content[1:] else ''
            is_discord_command = command_part in ['speak', 'say', 'tts', 'help']
        
        # In guilds: check for <@bot> !command pattern
        elif not is_dm and is_mentioned:
            # Remove mention and check for !command
            clean_content = self._strip_mention(raw_content)
            if clean_content.startswith('!'):
                command_part = clean_content[1:].split()[0] if clean_content[1:] else ''
                is_discord_command = command_part in ['speak', 'say', 'tts', 'help']
        
        if is_discord_command:
            logger.debug(f"🎯 Skipping router for Discord command: {command_part}")
            return  # Let Discord's command system handle this
        
        # Handle document uploads
        if has_document and not (has_prefix or is_mentioned):
            try:
                doc_attachment = next(
                    att for att in message.attachments 
                    if att.filename.lower().endswith(('.pdf', '.txt', '.md', '.docx'))
                )
                
                # Process with document mode
                await self.router.handle(message, f"--mode=text {content}" if content else "--mode=text")
                return
                
            except Exception as e:
                logger.error(f"Error processing document: {str(e)}", exc_info=True)
                await message.channel.send("⚠️ An error occurred while processing the document")
                return
        
        # Handle audio uploads
        if has_audio and not (has_prefix or is_mentioned):
            try:
                # Process with STT mode
                await self.router.handle(message, "--mode=stt")
                return
                
            except Exception as e:
                logger.error(f"Error processing audio: {str(e)}", exc_info=True)
                await message.channel.send("⚠️ An error occurred while processing the audio")
                return
        
        # If there's an image without explicit commands, only auto-process in DMs
        # In guilds, require mention or prefix for image processing
        if has_image and not (has_prefix or is_mentioned) and is_dm:
            logger.info(f"📷 Auto-processing image attachment in DM through router")
            try:
                # Let the router handle image processing with proper TTS integration
                await self.router.handle(message, content or "What's in this image?")
                return
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}", exc_info=True)
                await message.channel.send("⚠️ An error occurred while processing the image")
                return
        
        # For non-image messages or explicit commands, use the router
        try:
            await self.router.handle(message, content)
        except Exception as e:
            logger.error(f"Error in router.handle: {str(e)}", exc_info=True)
            await message.channel.send("⚠️ An error occurred while processing your message")
    
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        """Main message handler that dispatches to the router."""
        print(f"\n📨 Message received from {message.author} in {message.channel}:"
              f"\n  Content: {message.content}"
              f"\n  Attachments: {len(message.attachments)}")
        
        # Log message type and context
        is_dm = isinstance(message.channel, discord.DMChannel)
        is_mentioned = self.bot.user.mentioned_in(message)
        has_prefix = message.content.startswith(self.prefix)
        
        print(f"  Context - DM: {is_dm}, Mentioned: {is_mentioned}, Has Prefix: {has_prefix}")
        
        try:
            await self.dispatch_message(message)
            print("✅ Message processed successfully")
        except Exception as e:
            print(f"❌ Error processing message: {str(e)}")
            import traceback
            traceback.print_exc()
    
    async def _is_admin(self, user: discord.Member, guild: discord.Guild) -> bool:
        """Check if user has admin permissions."""
        if guild is None:
            return False
        
        # Check cached admin status
        if tts_state.is_admin(user.id):
            return True
        
        # Check Discord permissions
        if user.guild_permissions.administrator:
            tts_state.add_admin(user.id)
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
    logger.info("🚀 Starting TTS cache maintenance task")
    while True:
        try:
            logger.debug("🔄 Running TTS cache maintenance...")
            # Get stats before cleanup
            stats_before = tts_manager.get_cache_stats()
            logger.debug(f"📊 Cache stats before cleanup: {stats_before}")
            
            # Run the cleanup
            tts_manager.purge_old_cache()  # This is a synchronous method
            
            # Get stats after cleanup
            stats_after = tts_manager.get_cache_stats()
            logger.info(f"✅ TTS cache maintenance completed. Stats before: {stats_before}, after: {stats_after}")
            
            # Wait for 24 hours before next run
            logger.debug("⏳ Next TTS cache maintenance in 24 hours...")
            await asyncio.sleep(86400)  # Run daily
            
        except Exception as e:
            logger.error(f"❌ TTS cache maintenance failed: {str(e)}", exc_info=True)
            logger.info("⏳ Retrying TTS cache maintenance in 1 hour...")
            await asyncio.sleep(3600)  # Retry in 1 hour

async def setup(bot) -> None:
    """Set up event handlers."""
    await bot.add_cog(BotEventHandler(bot))
    logger.info("✅ Event handlers loaded")
    logger.info("Event handlers registered")
