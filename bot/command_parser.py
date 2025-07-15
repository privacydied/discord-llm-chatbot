"""
Parses raw Discord messages to identify commands and extract clean content, 
enforcing strict context rules for guilds vs. DMs.
"""
import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import discord
from discord.ext import commands

logger = logging.getLogger(__name__)

class Command(Enum):
    """Enumeration of all supported bot commands."""
    TTS = auto()
    TTS_ALL = auto()
    SPEAK = auto()
    SAY = auto()
    GENERAL = auto()  # Default for any non-command message

@dataclass
class ParsedCommand:
    """Represents a parsed command with its type and cleaned content."""
    command: Command
    cleaned_content: str

# Maps the raw command string to the Command enum
COMMAND_MAP = {
    "!tts": Command.TTS,
    "!tts-all": Command.TTS_ALL,
    "!speak": Command.SPEAK,
    "!say": Command.SAY,
}

def parse_command(message: discord.Message, bot: commands.Bot) -> Optional[ParsedCommand]:
    """
    Parses a message to determine the command and extracts clean content based on context.

    Args:
        message: The discord.Message object to parse.
        bot: The bot instance, used to check for mentions.

    Returns:
        A ParsedCommand object if the message is a valid command, otherwise None.
    """
    content = message.content
    is_dm = isinstance(message.channel, discord.DMChannel)

    if is_dm:
        # In DMs, commands must start with '!', no mention needed.
        if not content.startswith('!'):
            return ParsedCommand(Command.GENERAL, content.strip())
        
        parts = content.split(maxsplit=1)
        command_str = parts[0]
        cleaned_content = parts[1] if len(parts) > 1 else ""
        command = COMMAND_MAP.get(command_str, Command.GENERAL)
        # If it started with ! but wasn't a known command, treat as general.
        if command == Command.GENERAL:
            return ParsedCommand(Command.GENERAL, content.strip())
        return ParsedCommand(command, cleaned_content.strip())

    else: # Guild message
        # In guilds, the bot must be mentioned.
        if not bot.user.mentioned_in(message):
            return None # Ignore unmentioned messages in guilds

        # Remove the mention to get the actual content
        mention_pattern = fr'<@!?{bot.user.id}>\s*'
        cleaned_content = discord.utils.remove_markdown(content)
        cleaned_content = discord.utils.remove_mentions(cleaned_content).strip()

        if not cleaned_content.startswith('!'):
            return ParsedCommand(Command.GENERAL, cleaned_content)

        parts = cleaned_content.split(maxsplit=1)
        command_str = parts[0]
        cleaned_content = parts[1] if len(parts) > 1 else ""
        command = COMMAND_MAP.get(command_str, Command.GENERAL)
        
        # If it started with ! but wasn't a known command, treat as general.
        if command == Command.GENERAL:
             return ParsedCommand(Command.GENERAL, cleaned_content.strip())

        return ParsedCommand(command, cleaned_content.strip())