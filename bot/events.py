"""
Event handlers for Discord bot - extracted from main.py business logic.
"""
import re
import logging
from typing import Optional

import discord
from discord.ext import commands

from .context import get_conversation_context
from .logs import log_message, log_command
from .memory import get_profile, get_server_profile
from .ollama import generate_response
from .utils import send_chunks
from .web import get_url_preview

logger = logging.getLogger(__name__)


class EventHandlers(commands.Cog):
    """Event handlers for message processing and AI responses."""
    
    def __init__(self, bot: commands.Bot):
        self.bot = bot
    
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        """Process incoming messages."""
        # Ignore messages from the bot itself
        if message.author == self.user:
            return
        
        # Ignore messages from other bots
        if message.author.bot:
            return
        
        # Process commands first
        await self.bot.process_commands(message)
        
        # Log the message
        log_message(message)
        
        # Process URLs in the message
        await self.process_urls(message)
        
        # Process AI responses if the bot is mentioned or in a DM
        if self.bot.user.mentioned_in(message) or isinstance(message.channel, discord.DMChannel):
            await self.generate_ai_response(message)
    
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
                    # Generate the response
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
