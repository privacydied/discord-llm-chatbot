"""
Command handlers for the Discord bot.
"""
from typing import Dict, Type, Any
import logging

# Import all command modules here to register them
from . import memory_cmds
from . import test_cmds  # Add test commands for debugging

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
        # Import and set up test commands first
        from . import test_cmds
        await test_cmds.setup(bot)
        logging.info("✅ Test commands registered")
        
        # Import and set up memory commands
        from . import memory_cmds
        await memory_cmds.setup(bot)
        logging.info("✅ Memory commands registered")
        
        # Import and set up TTS commands  
        from . import tts_cmds
        await tts_cmds.setup(bot)
        logging.info("✅ TTS commands registered")
        
        # Import and set up event handlers
        from ..events import setup as setup_events
        await setup_events(bot)
        logging.info("✅ Event handlers registered")
        
        logging.info("🎉 All command modules registered successfully")
        
    except Exception as e:
        logging.error(f"Error setting up commands: {e}", exc_info=True)
        raise

# Import all command modules to register them
# This must be done after the command registry is set up
# and after all command decorators are defined
