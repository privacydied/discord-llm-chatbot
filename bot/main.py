"""
Main entry point for the Discord bot.
"""
import os
import sys
import logging
import signal
import asyncio
from typing import Optional, Dict, Any, List

import discord
from discord.ext import commands

# Import bot modules
from .config import load_config, setup_logging
from .memory import (
    load_all_profiles, save_all_profiles,
    load_all_server_profiles, save_all_server_profiles,
    save_profile, save_server_profile, 
    get_profile, get_server_profile,
    user_profiles, server_profiles,
    user_profiles_last_saved, server_profiles_last_saved
)
from .context import ConversationContext, get_conversation_context, reset_conversation_context
from .tts import setup_tts, cleanup_tts, tts_enabled, toggle_tts, toggle_global_tts
from .search import search_all, SearchResult
from .web import process_url, get_url_preview
from .pdf_utils import pdf_processor, PDFProcessor
from .ollama import ollama_client, OllamaAPIError, generate_response
from .commands import setup_commands
from .logs import setup_logging as setup_logging_utils, log_command, log_message
from .utils import send_chunks, download_file, is_text_file

# Load configuration
config = load_config()

# Bot configuration
TOKEN = config["DISCORD_TOKEN"]
PREFIX = config.get("COMMAND_PREFIX", "!")
OWNER_IDS = config.get("OWNER_IDS", [])
INTENTS = discord.Intents.default()
INTENTS.message_content = True
INTENTS.members = True
INTENTS.guilds = True

# Set up logging
logger = setup_logging()
setup_logging_utils()

class LLMBot(commands.Bot):
    """Custom bot class with additional functionality."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.startup_time = None
        self.commands_loaded = False
        self.config = config
        
    async def setup_hook(self) -> None:
        """Run when the bot starts up."""
        self.startup_time = discord.utils.utcnow()
        
        # Load all profiles
        await self.load_profiles()
        
        # Set up TTS
        await setup_tts()
        
        # Load commands
        await setup_commands(self)
        self.commands_loaded = True
        
        logger.info(f"Logged in as {self.user} (ID: {self.user.id})")
        logger.info(f"Connected to {len(self.guilds)} guilds")
        
        # Set presence
        activity = discord.Activity(
            type=discord.ActivityType.listening,
            name=f"{PREFIX}help"
        )
        await self.change_presence(activity=activity)
    
    async def load_profiles(self) -> None:
        """Load all user and server profiles."""
        logger.info("Loading user profiles...")
        load_all_profiles()
        
        logger.info("Loading server profiles...")
        load_all_server_profiles()
        
        logger.info(f"Loaded {len(user_profiles)} user profiles and {len(server_profiles)} server profiles")
    
    async def on_ready(self) -> None:
        """Run when the bot is ready."""
        if not hasattr(self, 'startup_time'):
            self.startup_time = discord.utils.utcnow()
        
        logger.info(f"Bot is ready! Logged in as {self.user} (ID: {self.user.id})")
        logger.info(f"Connected to {len(self.guilds)} guilds")
    
    async def on_message(self, message: discord.Message) -> None:
        """Process incoming messages."""
        # Ignore messages from the bot itself
        if message.author == self.user:
            return
        
        # Ignore messages from other bots
        if message.author.bot:
            return
        
        # Process commands first
        await self.process_commands(message)
        
        # Log the message
        log_message(message)
        
        # Process URLs in the message
        await self.process_urls(message)
        
        # Process AI responses if the bot is mentioned or in a DM
        if self.user.mentioned_in(message) or isinstance(message.channel, discord.DMChannel):
            await self.generate_ai_response(message)
    
    async def process_urls(self, message: discord.Message) -> None:
        """Process URLs in a message."""
        # Simple URL regex pattern
        import re
        url_pattern = re.compile(r'https?://\S+')
        urls = url_pattern.findall(message.content)
        
        if not urls:
            return
        
        # Process each URL
        for url in urls:
            try:
                # Skip common false positives
                if any(ext in url.lower() for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']):
                    continue
                
                # Create and send the URL preview
                embed = await get_url_preview(url)
                if embed:
                    await message.channel.send(embed=embed)
            
            except Exception as e:
                logger.error(f"Error processing URL {url}: {e}", exc_info=True)
    
    async def generate_ai_response(self, message: discord.Message) -> None:
        """Generate an AI response to a message."""
        # Get the conversation context
        ctx = await self.get_context(message)
        conversation = get_conversation_context(message.channel.id)
        
        # Check if we should respond with TTS
        user_id = str(message.author.id)
        guild_id = str(message.guild.id) if message.guild else None
        
        use_tts = False
        if guild_id:
            server_profile = get_server_profile(guild_id)
            global_tts = server_profile.get('preferences', {}).get('tts_enabled', False)
            user_prefs = get_profile(user_id).get('preferences', {})
            use_tts = global_tts and user_prefs.get('tts_enabled', False)
        
        # Remove the bot mention if present
        content = message.content
        if f'<@!{self.user.id}>' in content:
            content = content.replace(f'<@!{self.user.id}>', '').strip()
        elif f'<@{self.user.id}>' in content:
            content = content.replace(f'<@{self.user.id}>', '').strip()
        
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
    
    async def close(self) -> None:
        """Clean up before shutting down."""
        # Save all profiles
        save_all_profiles()
        save_all_server_profiles()
        
        # Clean up TTS
        await cleanup_tts()
        
        # Close the Ollama client
        await ollama_client.close()
        
        # Call the parent class close method
        await super().close()

def run_bot() -> None:
    """Run the Discord bot."""
    if not TOKEN:
        logger.error("No Discord token provided. Please set the DISCORD_TOKEN environment variable.")
        sys.exit(1)
    
    # Set up signal handlers for graceful shutdown
    def handle_signal(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        # Save all profiles before exiting
        save_all_profiles()
        save_all_server_profiles()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    # Create and run the bot
    bot = LLMBot(
        command_prefix=commands.when_mentioned_or(PREFIX),
        intents=INTENTS,
        owner_ids=set(OWNER_IDS)
    )
    
    # Run the bot
    try:
        bot.run(TOKEN)
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Ensure all profiles are saved before exiting
        save_all_profiles()
        save_all_server_profiles()

if __name__ == "__main__":
    run_bot()
