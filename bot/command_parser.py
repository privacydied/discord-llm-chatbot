"""
Parses raw Discord messages to identify commands and extract clean content,
enforcing strict context rules for guilds vs. DMs.
"""
import logging
import re
from typing import Optional

import discord
from discord.ext import commands

from .types import Command, ParsedCommand

logger = logging.getLogger(__name__)

# Maps the raw command string to the Command enum
COMMAND_MAP = {
    "!ping": Command.PING,
    "!tts": Command.TTS,
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
    content = message.content.strip()
    is_dm = isinstance(message.channel, discord.DMChannel)

    # In guilds, the bot must be mentioned. If not, ignore the message.
    if not is_dm and not bot.user.mentioned_in(message):
        return None

    # Remove the mention from guild messages to get the actual content
    if not is_dm:
        # This regex handles both <@USER_ID> and <@!USER_ID> mentions
        mention_pattern = fr'^<@!?{bot.user.id}>\s*'
        content = re.sub(mention_pattern, '', content)

    # If there's no content left, it's not a command
    if not content:
        return None

    # Check if the message starts with a known command prefix
    if content.startswith('!'):
        parts = content.split(maxsplit=1)
        command_str = parts[0]
        remaining_content = parts[1] if len(parts) > 1 else ""
        
        command = COMMAND_MAP.get(command_str)
        if command:
            return ParsedCommand(command=command, cleaned_content=remaining_content.strip())

    # If no specific command was matched, treat it as a general chat command.
    return ParsedCommand(command=Command.CHAT, cleaned_content=content)