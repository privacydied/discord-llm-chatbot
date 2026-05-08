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
    "!status": Command.STATUS,
    "!feature": Command.FEATURE,
    "!index": Command.INDEX,
    "!chat": Command.CHAT,
    "!search": Command.SEARCH,
    "!tts": Command.TTS,
    "!speak": Command.SPEAK,
    "!say": Command.SAY,
    "!memory-add": Command.MEMORY_ADD,
    "!memory-del": Command.MEMORY_DEL,
    "!memory-show": Command.MEMORY_SHOW,
    "!memory-wipe": Command.MEMORY_WIPE,
    "!rag": Command.RAG,
    "!rag bootstrap": Command.RAG_BOOTSTRAP,
    "!rag search": Command.RAG_SEARCH,
    "!rag status": Command.RAG_STATUS,
    "!alert": Command.ALERT,
    "!img": Command.IMG,
}


def parse_command(
    message: discord.Message, bot: commands.Bot
) -> Optional[ParsedCommand]:
    """
    Parses a message to determine if it's an explicit command.

    This function ONLY identifies commands that start with '!'. It does not handle
    routing logic for DMs or mentions; that is the responsibility of the Router.

    Args:
        message: The discord.Message object to parse.
        bot: The bot instance (currently unused, kept for API consistency).

    Returns:
        A ParsedCommand object if a known '!' command is found, otherwise None.
    """
    raw_content = (message.content or "").strip()

    # Remove bot mention from the beginning of the message to isolate the command
    mention_pattern = rf"^<@!?{bot.user.id}>\s*"
    mentioned_me = bool(re.match(mention_pattern, raw_content))
    content = re.sub(mention_pattern, "", raw_content)

    is_dm = isinstance(message.channel, discord.DMChannel)

    # Ignore empty content after mention stripping
    if not content:
        return None

    # Guild messages must explicitly mention the bot to trigger command parsing
    if not is_dm and not mentioned_me:
        return None

    if not content.startswith("!"):
        # Treat plain DM messages or direct mentions as chat commands for a smoother UX
        return ParsedCommand(command=Command.CHAT, cleaned_content=content)

    parts = content.split(maxsplit=1)
    command_str = parts[0]
    remaining_content = parts[1] if len(parts) > 1 else ""

    command = COMMAND_MAP.get(command_str)

    if command:
        # A known command was found.
        logger.debug(
            f"Parsed command: {command.name} with content: '{remaining_content[:50]}...'",
            extra={"subsys": "parser", "event": "command.found"},
        )
        return ParsedCommand(command=command, cleaned_content=remaining_content.strip())

    # An unknown '!' command was found, ignore it.
    logger.debug(
        f"Ignoring unknown command: {command_str}",
        extra={"subsys": "parser", "event": "command.unknown"},
    )
    return None
