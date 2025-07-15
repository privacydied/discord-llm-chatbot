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
    "!help": Command.HELP,
    "!chat": Command.CHAT,
    "!tts": Command.TTS,
    "!say": Command.SAY,
    "!memory-add": Command.MEMORY_ADD,
    "!memory-del": Command.MEMORY_DEL,
    "!memory-show": Command.MEMORY_SHOW,
    "!memory-wipe": Command.MEMORY_WIPE,
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

    # In DMs, we process the message directly.
    # In guilds, the bot must be mentioned.
    if is_dm:
        pass  # Proceed with parsing
    elif bot.user.mentioned_in(message):
        # Remove the mention from guild messages to get the actual content
        mention_pattern = fr'^<@!?{bot.user.id}>\s*'
        content = re.sub(mention_pattern, '', content)
    else:
        # Not a DM and no mention, so ignore.
        return None

    # If there's no content left but has attachments, treat as CHAT command
    if not content:
        if message.attachments:
            # Empty content with attachments is valid for processing
            attachment_info = f"{len(message.attachments)} attachment(s): {message.attachments[0].filename}"
            logging.debug(f"📎 Processing empty content message with {attachment_info}", 
                        extra={'subsys': 'parser', 'event': 'empty_with_attachment'})
            return ParsedCommand(command=Command.CHAT, cleaned_content="")
        return None

    # If the message starts with a command prefix, try to parse it as a command.
    if content.startswith('!'):
        parts = content.split(maxsplit=1)
        command_str = parts[0]
        remaining_content = parts[1] if len(parts) > 1 else ""

        command = COMMAND_MAP.get(command_str)

        if command:
            # It's a known command.
            return ParsedCommand(command=command, cleaned_content=remaining_content.strip())
        else:
            # It's an unknown command starting with '!', so we ignore it.
            return None

    # If it doesn't start with '!', it's a general chat message.
    return ParsedCommand(command=Command.CHAT, cleaned_content=content)