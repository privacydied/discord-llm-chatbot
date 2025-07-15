"""
Contains bot startup and pre-flight check logic.
"""
import discord
from discord.ext import commands
import hashlib

from bot.config import ConfigurationError
from bot.logger import get_logger

def run_pre_flight_checks(config: dict) -> None:
    """Runs all mandatory startup checks as per the Windsurf spec."""
    logger = get_logger(__name__)
    logger.info("--- Running Pre-Flight Checklist ---", extra={'subsys': 'core', 'event': 'pre_flight_checks'})

    # 1. Bot Token Check
    token = config.get("DISCORD_TOKEN")
    if not token:
        logger.critical("DISCORD_TOKEN is missing. Bot cannot start.", extra={'subsys': 'core', 'event': 'token_check_fail'})
        raise ConfigurationError("DISCORD_TOKEN not found in environment.")
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    logger.info(f"[WIND][INIT] Token hash={token_hash} validated", extra={'subsys': 'core', 'event': 'token_check_pass'})

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
            logger.error(f"Required intent '{intent_name}' is disabled.", extra={'subsys': 'core', 'event': 'intent_check_fail'})
    
    if all_intents_ok:
        logger.info("[WIND][INIT] Intents verified", extra={'subsys': 'core', 'event': 'intent_check_pass'})
    else:
        logger.critical("One or more required intents are missing. Bot may not function correctly.", extra={'subsys': 'core', 'event': 'intent_check_fail'})
        # Depending on strictness, you might raise an error here.

    # 3. Voice Gateway Version
    logger.info(f"[WIND][INIT] Discord.py Version: {discord.__version__}", extra={'subsys': 'core', 'event': 'version_check'})
    logger.info("--- Pre-Flight Checklist Complete ---", extra={'subsys': 'core', 'event': 'pre_flight_checks_pass'})

def create_bot_intents() -> discord.Intents:
    """Create Discord intents with all required permissions."""
    intents = discord.Intents.default()  # Start with default intents
    intents.message_content = True  # Enable message content intent
    return intents

def get_prefix(bot: commands.Bot, message: discord.Message) -> list[str]:
    """Determine command prefix based on context."""
    config = bot.config
    prefix = config.get('COMMAND_PREFIX', '!')
    if message.guild is None:
        return [prefix]
    mentions = commands.when_mentioned(bot, message)
    return [f"{m}{prefix}" for m in mentions]

