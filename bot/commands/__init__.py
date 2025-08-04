"""
Command handlers for the Discord bot.
"""
from typing import Dict, Any
import logging

# Command modules are loaded via the setup_commands function below.

# This will be populated with all registered commands
commands: Dict[str, Any] = {}

def register_command(name: str, func: callable, **kwargs):
    """Register a command with the given name and handler function."""
    if name in commands:
        logging.warning(f"Command '{name}' is already registered and will be overwritten")
    
    commands[name] = {
        'handler': func,
        'name': name,
        'aliases': kwargs.get('aliases', []),
        'description': kwargs.get('description', 'No description provided'),
        'usage': kwargs.get('usage', ''),
        'admin_only': kwargs.get('admin_only', False)
    }
    
    # Register aliases
    for alias in commands[name]['aliases']:
        if alias in commands and commands[alias] != commands[name]:
            logging.warning(f"Alias '{alias}' for command '{name}' is already registered")
        commands[alias] = commands[name]

def get_command(name: str) -> dict:
    """Get a command by name."""
    return commands.get(name)

def get_all_commands() -> dict:
    """Get all registered commands."""
    # Return only the primary commands (no aliases)
    return {name: cmd for name, cmd in commands.items() if name == cmd['name']}

async def setup_commands(bot):
    """
    Set up all command modules with the bot instance.
    This function is called during bot startup to register all commands.
    """
    import logging
    
    try:
        from . import test_cmds, memory_cmds, tts_cmds, config_commands, video_commands

        # A set to keep track of loaded cogs and avoid duplicates
        loaded_cogs = set(bot.cogs.keys())

        # Define cogs to load
        cogs_to_load = {
            'TestCommands': test_cmds,
            'MemoryCommands': memory_cmds,
            'TTSCommands': tts_cmds,
            'ConfigCommands': config_commands,
            'VideoCommands': video_commands
        }

        for cog_name, module in cogs_to_load.items():
            if cog_name not in loaded_cogs:
                await module.setup(bot)
                logging.info(f"✅ {cog_name} registered")
            else:
                logging.debug(f"Skipping already loaded cog: {cog_name}")
        

        logging.info("🎉 All command modules registered successfully")
        
    except Exception as e:
        logging.error(f"Error setting up commands: {e}", exc_info=True)
        raise

# Import all command modules to register them
# This must be done after the command registry is set up
# and after all command decorators are defined
