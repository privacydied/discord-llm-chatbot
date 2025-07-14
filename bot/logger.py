"""
Dual-sink logger for the Windsurf bot.

Implements the specification for both human-readable, "pretty" console logs
and structured, machine-readable JSONL file logs.
"""
import logging
import sys
from pathlib import Path

import discord
from discord.ext import commands
from pythonjsonlogger import jsonlogger
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

from bot.config import load_config

# --- Pretty Console Logger Setup ---

# Define custom theme for pretty logger
CUSTOM_THEME = Theme({
    "info": "green",
    "warning": "yellow",
    "error": "bold red",
    "critical": "bold magenta",
    "debug": "dim cyan",
})

# Define level-based icons
LOG_LEVEL_ICONS = {
    logging.INFO: "✔",
    logging.WARNING: "⚠",
    logging.ERROR: "✖",
    logging.CRITICAL: "🔥",
    logging.DEBUG: "⚙️",
}



class PrettyConsoleHandler(logging.Handler):
    """A RichHandler that formats logs with level-based icons and colors."""

    def __init__(self, console: Console, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.console = console
        self.log_time_format = "%Y-%m-%d %H:%M:%S,%f"

    def emit(self, record: logging.LogRecord) -> None:
        """Renders the log message with a custom table format."""
        from datetime import datetime
        from rich.table import Table
        from rich.text import Text

        table = Table.grid(padding=(0, 1), expand=False)
        table.add_column(style="dim")  # For icon
        table.add_column(style="dim")  # For timestamp
        table.add_column()  # For message

        icon = Text(LOG_LEVEL_ICONS.get(record.levelno, '•'))
        log_time = datetime.fromtimestamp(record.created).strftime(self.log_time_format)

        # Get the rich-formatted message
        message_renderable = self.format(record)

        # Define a direct mapping from log level to color name
        level_to_color = {
            logging.DEBUG: "dim cyan",
            logging.INFO: "green",
            logging.WARNING: "yellow",
            logging.ERROR: "bold red",
            logging.CRITICAL: "bold magenta",
        }
        # Apply color based on the level
        color = level_to_color.get(record.levelno)
        if color:
            icon.stylize(color)

        table.add_row(icon, Text(f"[{log_time}]"), message_renderable)
        self.console.print(table)

# --- Structured JSON Logger Setup ---

class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """A custom JSON formatter to add context to log records."""

    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        log_record['level'] = record.levelname
        log_record['name'] = record.name
        log_record['timestamp'] = self.formatTime(record, self.datefmt)
        if 'message' in log_record:
            log_record['detail'] = log_record.pop('message')

# --- Main Logging Setup ---

def setup_logging() -> None:
    """
    Configures the root logger with a dual-sink setup:
    1. A "pretty" console logger using rich.
    2. A structured JSON file logger.
    """
    config = load_config()
    log_level = config.get("LOG_LEVEL", "INFO").upper()
    log_file_path = Path(config.get("LOG_FILE", "logs/bot.jsonl"))

    # Ensure the log directory exists
    log_file_path.parent.mkdir(exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Remove any existing handlers to prevent duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # 1. Add Pretty Console Handler
    pretty_handler = PrettyConsoleHandler(console=Console())
    # We need a formatter to handle the message part
    pretty_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(pretty_handler)

    # 2. Add Structured JSON File Handler
    format_str = '%(timestamp)s %(level)s %(name)s %(detail)s %(subsys)s %(guild_id)s %(user_id)s %(msg_id)s %(event)s'
    json_formatter = CustomJsonFormatter(format_str)
    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setFormatter(json_formatter)
    logger.addHandler(file_handler)

    # Assert that both handlers are active
    assert len(logger.handlers) == 2, "Logger must have exactly two handlers (pretty and jsonl)."
    logger.info(f"[WIND] Dual-sink logging initialized. Level: {log_level}. JSON logs at: {log_file_path}", extra={'subsys': 'core', 'event': 'init'})

    # Suppress noisy third-party loggers
    logging.getLogger('discord').setLevel(logging.WARNING)
    logging.getLogger('websockets').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)

def get_logger(name: str) -> logging.Logger:
    """Returns a logger instance with the specified name."""
    return logging.getLogger(name)

def log_command(ctx: commands.Context, event: str, extra_info: dict = None, success: bool = True) -> None:
    """
    Logs a command execution with structured context.
    """
    logger = get_logger('command_logger')
    context = get_event_context(ctx.message)
    context['event'] = event
    context['command'] = ctx.command.qualified_name if ctx.command else 'unknown'

    if extra_info:
        context.update(extra_info)

    if success:
        logger.info(f"Command '{context['command']}' executed.", extra=context)
    else:
        logger.error(f"Command '{context['command']}' failed.", extra=context)

def get_event_context(message: discord.Message) -> dict:
    """
    Creates a dictionary with context from a discord.Message for logging.
    This ensures consistent context across all event logs.
    """
    return {
        'guild_id': message.guild.id if message.guild else 'DM',
        'user_id': message.author.id,
        'msg_id': message.id,
    }
