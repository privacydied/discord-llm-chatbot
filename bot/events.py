"""
Event handlers for Discord bot - extracted from main.py business logic.
"""
import os
import re
import logging
from typing import Optional

import discord
from discord.ext import commands

from .context import get_conversation_context
from .logs import log_message, log_command
from .memory import get_profile, get_server_profile
# CRITICAL FIX: Use unified backend router instead of direct ollama import
from .ai_backend import generate_response, generate_vl_response
from .utils import send_chunks
from .web import get_url_preview

logger = logging.getLogger(__name__)


def has_image_attachments(message: discord.Message) -> bool:
    """
    Utility function to detect image uploads in Discord messages.
    CHANGE: Enhanced image detection with comprehensive debugging for hybrid multimodal logic.
    """
    logger.debug(f"🔍 has_image_attachments: checking {len(message.attachments)} attachments")
    
    if not message.attachments:
        logger.debug("🔍 No attachments found")
        return False
    
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'}
    image_mimes = {'image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/webp', 'image/bmp'}
    
    for i, attachment in enumerate(message.attachments):
        logger.debug(f"🔍 Checking attachment {i}: filename='{attachment.filename}'")
        
        # Check by file extension
        filename_lower = attachment.filename.lower()
        for ext in image_extensions:
            if filename_lower.endswith(ext):
                logger.info(f"✅ Image detected by extension: {attachment.filename} (extension: {ext})")
                return True
        
        # Check by content type if available
        if hasattr(attachment, 'content_type') and attachment.content_type:
            logger.debug(f"🔍 Content type: {attachment.content_type}")
            if attachment.content_type in image_mimes:
                logger.info(f"✅ Image detected by MIME type: {attachment.filename} (MIME: {attachment.content_type})")
                return True
        else:
            logger.debug(f"⚠️  No content_type available for {attachment.filename}")
    
    logger.debug("🔍 No images detected in attachments")
    return False


def get_image_urls(message: discord.Message) -> list:
    """
    Extract image URLs from Discord message attachments.
    CHANGE: Added image URL extraction for VL model processing.
    """
    image_urls = []
    if not message.attachments:
        return image_urls
    
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'}
    image_mimes = {'image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/webp', 'image/bmp'}
    
    for attachment in message.attachments:
        is_image = False
        
        # Check by file extension
        if any(attachment.filename.lower().endswith(ext) for ext in image_extensions):
            is_image = True
        
        # Check by content type if available
        if hasattr(attachment, 'content_type') and attachment.content_type in image_mimes:
            is_image = True
            
        if is_image:
            image_urls.append(attachment.url)
    
    return image_urls


class EventHandlers(commands.Cog):
    """Event handlers for message processing and AI responses."""
    
    def __init__(self, bot: commands.Bot):
        self.bot = bot
    
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        """Process incoming messages."""
        # CHANGE: Added ultra-verbose debug logging for all message processing
        logger.debug(f"📨 on_message triggered: author={message.author.name}#{message.author.discriminator}, content_length={len(message.content)}")
        logger.debug(f"📎 Attachments count: {len(message.attachments)}")
        
        # CHANGE: Log every attachment for debugging
        for i, attachment in enumerate(message.attachments):
            logger.debug(f"📎 Attachment {i}: filename='{attachment.filename}', content_type='{getattr(attachment, 'content_type', 'UNKNOWN')}', url='{attachment.url}'")
        
        # CHANGE: Fixed AttributeError by using self.bot.user instead of self.user
        # Ignore messages from the bot itself
        if message.author == self.bot.user:
            logger.debug("🤖 Ignoring message from bot itself")
            return
        
        # Ignore messages from other bots
        if message.author.bot:
            logger.debug("🤖 Ignoring message from other bot")
            return
        
        # Process commands first
        await self.bot.process_commands(message)
        
        # Log the message
        log_message(message)
        
        # Process URLs in the message
        await self.process_urls(message)
        
        # CHANGE: Enhanced trigger condition logging
        is_mentioned = self.bot.user.mentioned_in(message)
        is_dm = isinstance(message.channel, discord.DMChannel)
        logger.debug(f"🎯 Trigger conditions: mentioned={is_mentioned}, is_dm={is_dm}")
        
        # Process AI responses if the bot is mentioned or in a DM
        if is_mentioned or is_dm:
            logger.info(f"🚀 Triggering AI response for message from {message.author.name}")
            await self.generate_ai_response(message)
        else:
            logger.debug("🚫 AI response not triggered - bot not mentioned and not in DM")
    
    async def process_urls(self, message: discord.Message) -> None:
        """Process URLs in a message."""
        try:
            # Simple URL regex pattern
            url_pattern = r'https?://[^\s]+'
            urls = re.findall(url_pattern, message.content)
            
            for url in urls:
                try:
                    # Get URL preview
                    preview = await get_url_preview(url)
                    if preview:
                        # Send the preview as an embed
                        embed = discord.Embed(
                            title=preview.get('title', 'Link Preview'),
                            description=preview.get('description', '')[:500],
                            url=url,
                            color=discord.Color.blue()
                        )
                        
                        if preview.get('image'):
                            embed.set_image(url=preview['image'])
                        
                        await message.channel.send(embed=embed)
                
                except Exception as e:
                    logger.error(f"Error processing URL {url}: {e}")
        
        except Exception as e:
            logger.error(f"Error in process_urls: {e}", exc_info=True)
    
    async def generate_ai_response(self, message: discord.Message) -> None:
        """Generate an AI response to a message."""
        try:
            # Get user and guild IDs
            user_id = str(message.author.id)
            guild_id = str(message.guild.id) if message.guild else None
            
            # Get conversation context
            conversation = get_conversation_context(user_id, guild_id)
            
            # Get user profile for TTS preference
            user_profile = get_profile(user_id)
            use_tts = user_profile.get('tts_enabled', False) if user_profile else False
            
            # Get server profile for TTS preference
            if guild_id:
                server_profile = get_server_profile(guild_id)
                if server_profile and server_profile.get('tts_enabled', False):
                    use_tts = True
            
            # Clean up the message content
            content = message.content
            if content.startswith(f'<@!{self.bot.user.id}>'):
                content = content[len(f'<@!{self.bot.user.id}>'):].strip()
            elif f'<@{self.bot.user.id}>' in content:
                content = content.replace(f'<@{self.bot.user.id}>', '').strip()
            
            # Check if this is a search query
            is_search = any(word in content.lower() for word in ['search', 'look up', 'find', 'what is'])
            
            try:
                # Show typing indicator
                async with message.channel.typing():
                    # CHANGE: Enhanced hybrid multimodal logic with comprehensive debugging
                    logger.debug("🤖 Starting AI response generation")
                    logger.debug(f"🤖 User prompt: '{content[:100]}{'...' if len(content) > 100 else ''}'")
                    
                    # Check if message contains images
                    has_images = has_image_attachments(message)
                    logger.info(f"🖼️  Image detection result: {has_images}")
                    
                    if has_images:
                        logger.info("🎨 ENTERING VL INFERENCE BRANCH")
                        logger.debug(f"📁 VL_PROMPT_FILE = {os.getenv('VL_PROMPT_FILE')}")
                        logger.debug(f"🤖 VL_MODEL = {os.getenv('VL_MODEL')}")
                        
                        # Get all image URLs from the message
                        image_urls = get_image_urls(message)
                        logger.debug(f"🖼️  Extracted {len(image_urls)} image URLs: {image_urls}")
                        
                        # Process the first image with VL model
                        if image_urls:
                            try:
                                logger.info(f"🎨 Processing image with VL model: {image_urls[0][:100]}...")
                                
                                # Step 1: Use VL model with VL_PROMPT_FILE
                                vl_response = await generate_vl_response(
                                    image_url=image_urls[0],  # Process first image
                                    user_prompt=content,  # Include user's text as additional context
                                    user_id=user_id,
                                    guild_id=guild_id,
                                    temperature=0.7
                                )
                                
                                vl_output = vl_response.get('text', '').strip()
                                logger.info(f"✅ VL model returned {len(vl_output)} characters")
                                logger.debug(f"🎨 VL model output preview: {vl_output[:200]}{'...' if len(vl_output) > 200 else ''}")
                                
                                if not vl_output:
                                    logger.warning("⚠️  VL model returned empty output!")
                                
                                # Step 2: Feed VL output into text model with original context
                                enhanced_context = f"{conversation}\n\nVision analysis:\n{vl_output}"
                                logger.info("🔗 Chaining VL output into text model")
                                logger.debug(f"🔗 Enhanced context length: {len(enhanced_context)} chars")
                                
                                # Generate final response using text model
                                response = await generate_response(
                                    prompt=content,
                                    context=enhanced_context,
                                    user_id=user_id,
                                    guild_id=guild_id,
                                    max_tokens=1000,
                                    temperature=0.7
                                )
                                
                                logger.info("🎉 Hybrid multimodal processing completed successfully")
                                
                            except Exception as vl_error:
                                logger.error(f"❌ VL processing failed: {vl_error}", exc_info=True)
                                # Fallback to text-only processing
                                logger.warning("🔄 Falling back to text-only processing")
                                response = await generate_response(
                                    prompt=f"[Note: Image uploaded but vision processing failed] {content}",
                                    context=conversation,
                                    user_id=user_id,
                                    guild_id=guild_id,
                                    max_tokens=1000,
                                    temperature=0.7
                                )
                        else:
                            # No valid images found, fallback to text
                            logger.warning("⚠️  No valid image URLs extracted despite has_image_attachments=True")
                            logger.info("🔄 Falling back to text-only processing")
                            response = await generate_response(
                                prompt=content,
                                context=conversation,
                                user_id=user_id,
                                guild_id=guild_id,
                                max_tokens=1000,
                                temperature=0.7
                            )
                    else:
                        # No images - standard text processing with PROMPT_FILE
                        logger.info("📝 ENTERING STANDARD TEXT PROCESSING BRANCH")
                        logger.debug(f"📁 PROMPT_FILE = {os.getenv('PROMPT_FILE')}")
                        response = await generate_response(
                            prompt=content,
                            context=conversation,
                            user_id=user_id,
                            guild_id=guild_id,
                            max_tokens=1000,
                            temperature=0.7
                        )
                    
                    # Send the response
                    response_text = response.get('text', '').strip()
                    if not response_text:
                        return
                    
                    # Split long messages into chunks
                    await send_chunks(message.channel, response_text)
                    
                    # Send TTS if enabled
                    if use_tts:
                        from .tts import generate_tts
                        tts_file = await generate_tts(response_text, user_id=user_id)
                        if tts_file and tts_file.exists():
                            await message.channel.send(
                                file=discord.File(tts_file, filename="tts_response.wav"),
                                reference=message
                            )
                            # Clean up the temporary file
                            try:
                                tts_file.unlink()
                            except Exception as e:
                                logger.warning(f"Failed to delete TTS file: {e}")
                
                # Log the command
                log_command(
                    user_id=user_id,
                    guild_id=guild_id,
                    command="ai_response",
                    success=True,
                    message=content[:100]  # Log first 100 chars
                )
                
            except Exception as e:
                logger.error(f"Error generating AI response: {e}", exc_info=True)
                
                # Send an error message
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                await message.channel.send(error_msg)
                
                # Log the error
                log_command(
                    user_id=user_id,
                    guild_id=guild_id,
                    command="ai_response",
                    success=False,
                    message=f"Error: {str(e)}"
                )
        
        except Exception as e:
            logger.error(f"Error in generate_ai_response: {e}", exc_info=True)


async def setup(bot: commands.Bot) -> None:
    """Setup function to add the event handlers cog."""
    await bot.add_cog(EventHandlers(bot))
