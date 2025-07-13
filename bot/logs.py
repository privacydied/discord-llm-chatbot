"""
Logging utilities for the Discord bot.
"""
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

# Import config
from .config import load_config

# Load configuration
config = load_config()

# Configure root logger
def setup_logging():
    """Configure the root logger with file and console handlers."""
    # Ensure logs directory exists
    config["USER_LOGS_DIR"].mkdir(parents=True, exist_ok=True)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Set up file handler for all logs
    log_file = config["USER_LOGS_DIR"] / 'bot.log'
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Set up console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add our handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Disable discord.py's noisy debug logging
    logging.getLogger('discord').setLevel(logging.WARNING)
    logging.getLogger('websockets').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)

def log_user_message(message) -> str:
    """
    Log a user message to their personal log file.
    
    Returns the path to the log file.
    """
    try:
        user_id = str(message.author.id)
        log_dir = config["USER_LOGS_DIR"] / user_id
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a log file for each day
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = log_dir / f"{today}.log"
        
        # Format the log entry
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if message.guild:
            guild_name = message.guild.name
            channel_name = message.channel.name if hasattr(message.channel, 'name') else 'DM'
            log_entry = f"{timestamp} [Server: {guild_name}] [#{channel_name}] {message.author}: {message.content}\n"
        else:
            log_entry = f"{timestamp} [DM] {message.author}: {message.content}\n"
        
        # Write to the log file
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
            
        return str(log_file)
        
    except Exception as e:
        logging.error(f"Error logging user message: {e}", exc_info=True)
        return ""

def log_dm_message(message) -> str:
    """
    Log a direct message to the DM log file.
    
    Returns the path to the log file.
    """
    try:
        user_id = str(message.author.id)
        log_dir = config["USER_LOGS_DIR"] / "dms"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a log file for each day
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = log_dir / f"{user_id}_{today}.log"
        
        # Format the log entry
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp} {message.author}: {message.content}\n"
        
        # Write to the log file
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
            
        return str(log_file)
        
    except Exception as e:
        logging.error(f"Error logging DM: {e}", exc_info=True)
        return ""

def log_message(message) -> None:
    """
    Log a message from Discord, handling both DM and server messages.
    This function determines the appropriate logging method based on message type.
    """
    try:
        if isinstance(message.channel, discord.DMChannel):
            # Log as DM message
            log_dm_message(message)
        else:
            # Log as user message in server
            log_user_message(message)
    except Exception as e:
        logging.error(f"Error logging message: {e}", exc_info=True)

def log_command(ctx, command_name: str, args: dict, success: bool = True, error: Optional[str] = None):
    """Log a command execution."""
    try:
        user_id = str(ctx.author.id)
        username = str(ctx.author)
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'username': username,
            'command': command_name,
            'args': args,
            'success': success,
            'error': error,
            'channel': str(ctx.channel) if hasattr(ctx, 'channel') else 'DM',
            'guild': str(ctx.guild) if hasattr(ctx, 'guild') and ctx.guild else 'DM'
        }
        
        logging.info(f"Command executed: {log_entry}")
        
    except Exception as e:
        logging.error(f"Error logging command: {e}", exc_info=True)

# Initialize logging when module is imported
setup_logging()
