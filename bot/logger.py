import logging
from typing import Optional, Dict, Any

from discord.ext import commands

# Get a logger instance for this module
logger = logging.getLogger(__name__)


def log_command(
    ctx: commands.Context,
    event: str,
    detail: Optional[Dict[str, Any]] = None,
    success: bool = True,
):
    """
    Logs a command execution with structured context.

    Args:
        ctx: The command context from discord.py.
        event: A string describing the event (e.g., 'add_memory').
        detail: An optional dictionary for additional structured details.
        success: A boolean indicating if the command was successful.
    """
    guild_id = ctx.guild.id if ctx.guild else "DM"
    user_id = ctx.author.id

    level = logging.INFO if success else logging.ERROR
    status_icon = "✔" if success else "✖"

    # Format the core log message with embedded context
    log_message = f"{status_icon} CMD [guild: {guild_id}, user: {user_id}, cmd: {ctx.command.name}] {event}"

    # If additional details are provided, append them for structured logging
    if detail:
        detail_str = ", ".join([f"{k}: {v}" for k, v in detail.items()])
        log_message += f" ({detail_str})"

    extra_context = {
        "subsys": "command",
        "guild_id": guild_id,
        "user_id": user_id,
        "msg_id": ctx.message.id,
        "event": event,
        "detail": detail or {},
    }

    logger.log(level, log_message, extra=extra_context)
