"""
Contains bot startup and pre-flight check logic.
"""
import discord
from discord.ext import commands
import hashlib
import os

from bot.config import ConfigurationError
from bot.util.logging import get_logger

def run_pre_flight_checks(config: dict) -> None:
    """Runs all mandatory startup checks as per the Windsurf spec."""
    logger = get_logger(__name__)
    logger.info("--- Running Pre-Flight Checklist ---")

    # 1. Bot Token Check
    token = config.get("DISCORD_TOKEN")
    if not token:
        logger.critical("DISCORD_TOKEN is missing. Bot cannot start.")
        raise ConfigurationError("DISCORD_TOKEN not found in environment.")
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    logger.info(f"[WIND][INIT] Token hash={token_hash} validated")

    # 2. Intents Check
    intents = create_bot_intents()
    required_intents = {
        'message_content': intents.message_content,
        'guilds': intents.guilds,
        'guild_messages': intents.messages, # This covers guild messages
        'dm_messages': intents.messages, # and DM messages
        'guild_voice_states': intents.voice_states,
    }

    all_intents_ok = True
    for intent_name, is_enabled in required_intents.items():
        if not is_enabled:
            all_intents_ok = False
            logger.error(f"Required intent '{intent_name}' is disabled.")
    
    if all_intents_ok:
        logger.info("[WIND][INIT] Intents verified")
    else:
        logger.critical("One or more required intents are missing. Bot may not function correctly.")
        # Depending on strictness, you might raise an error here.

    # 3. Voice Gateway Version
    logger.info(f"[WIND][INIT] Discord.py Version: {discord.__version__}")
    logger.info("--- Pre-Flight Checklist Complete ---")

def create_bot_intents() -> discord.Intents:
    """Create Discord intents with all required permissions."""
    intents = discord.Intents.default()
    intents.message_content = True
    intents.guilds = True
    intents.members = True  # Required for many user-related operations
    intents.voice_states = True # Required for TTS
    return intents

def get_prefix(bot, message: discord.Message) -> list[str]:
    """Dynamically get the command prefix for the bot."""
    base_prefixes = os.getenv('BOT_PREFIX', '!').split(',')
    return commands.when_mentioned_or(*base_prefixes)(bot, message)
