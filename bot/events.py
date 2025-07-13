"""
Event handlers for the Discord bot.
"""
import asyncio
import logging
import sys
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
        """Strip bot mention from message content."""
        # Remove <@!USER_ID> or <@USER_ID> mentions
        mention = f'<@!{self.bot.user.id}>'
        alt_mention = f'<@{self.bot.user.id}>'
        
        if content.startswith(mention):
            return content[len(mention):].strip()
        elif content.startswith(alt_mention):
            return content[len(alt_mention):].strip()
        return content
    
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
        
        # If there's an image, process it through VL and text models
        if has_image and not (has_prefix or is_mentioned):
            try:
                # Process the first image attachment
                image_attachment = next(
                    att for att in message.attachments 
                    if att.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))
                )
                
                # Download the image
                from pathlib import Path
                import tempfile
                import os
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(image_attachment.filename).suffix) as tmp:
                    await image_attachment.save(tmp.name)
                    image_path = Path(tmp.name)
                
                try:
                    # Process image with VL model
                    from bot.see import see_infer
                    from bot.brain import brain_infer
                    
                    logger.info(f"👁️ Processing image: {image_attachment.filename}")
                    
                    # Get VL model description
                    vl_result = await see_infer(image_path, content or "What's in this image?")
                    logger.debug(f"VL model output: {vl_result[:200]}...")
                    
                    # Create enhanced prompt for text model
                    enhanced_prompt = f"""Based on this image description:
                    
                    {vl_result}
                    
                    {content if content else 'What can you tell me about this image?'}
                    """
                    
                    # Get text model response
                    logger.info("🤖 Getting text model response...")
                    response = await brain_infer(enhanced_prompt)
                    
                    # Send the final response
                    await message.channel.send(response)
                    
                finally:
                    # Clean up the temporary file
                    try:
                        os.unlink(image_path)
                    except Exception as e:
                        logger.warning(f"Failed to delete temp file {image_path}: {e}")
                
                # Return early since we've handled the image
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
            else:
                try:
                    # Create temporary audio file
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                        audio_path = Path(tmpfile.name)
                    
                    # Synthesize audio asynchronously
                    audio_path = await tts_manager.synthesize_async(text_response, audio_path)
                    
                    if audio_path and audio_path.exists():
                        # Fix: Add missing closing parenthesis
                        await ctx.send(file=discord.File(str(audio_path), filename="tts.ogg"))
                    else:
                        logging.error(f"TTS synthesis failed for: {text_response}")
                except Exception as e:
                    logging.error(f"TTS synthesis error: {e}")
                    await ctx.send("⚠️ TTS synthesis failed")
        
        # For text modes, send text response
        if mode in ("text", "both"):
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
