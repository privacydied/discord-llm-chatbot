"""
Command handlers for the Discord bot.
"""
from typing import Dict, Any
import logging

# Command modules are loaded via the setup_commands function below.

# This will be populated with all registered commands
commands: Dict[str, Any] = {}

def register_command(name: str, func: callable, **kwargs):
    """
    Register a command with the given name and handler function.
    
    [REH] Robust error handling with input validation
    [SFT] Security-first thinking with parameter validation
    """
    # Input validation [SFT]
    if not name or not isinstance(name, str):
        raise ValueError("Command name must be a non-empty string")
    if not callable(func):
        raise ValueError("Command handler must be callable")
    
    # Sanitize command name
    name = name.strip().lower()
    if not name.replace('_', '').replace('-', '').isalnum():
        raise ValueError(f"Command name '{name}' contains invalid characters")
    
    # Check for duplicates with stronger handling [REH]
    if name in commands:
        existing_cmd = commands[name]
        if existing_cmd['handler'] != func:
            logging.error(f"❌ CRITICAL: Command '{name}' already exists with different handler!")
            logging.error(f"   Existing: {existing_cmd['handler'].__module__}.{existing_cmd['handler'].__name__}")
            logging.error(f"   New: {func.__module__}.{func.__name__}")
            raise ValueError(f"Command '{name}' is already registered with a different handler")
        else:
            logging.debug(f"🔄 Command '{name}' re-registered with same handler")
            return  # Skip duplicate registration
    
    # Validate aliases [SFT]
    aliases = kwargs.get('aliases', [])
    if not isinstance(aliases, list):
        raise ValueError("Aliases must be a list")
    
    validated_aliases = []
    for alias in aliases:
        if not isinstance(alias, str) or not alias.strip():
            logging.warning(f"⚠️ Invalid alias '{alias}' for command '{name}', skipping")
            continue
        alias = alias.strip().lower()
        if not alias.replace('_', '').replace('-', '').isalnum():
            logging.warning(f"⚠️ Alias '{alias}' contains invalid characters, skipping")
            continue
        validated_aliases.append(alias)
    
    commands[name] = {
        'handler': func,
        'name': name,
        'aliases': validated_aliases,
        'description': kwargs.get('description', 'No description provided')[:500],  # Limit description length
        'usage': kwargs.get('usage', '')[:200],  # Limit usage length
        'admin_only': bool(kwargs.get('admin_only', False))
    }
    
    # Register aliases with conflict detection [REH]
    for alias in validated_aliases:
        if alias in commands:
            existing_cmd = commands[alias]
            if existing_cmd != commands[name]:
                logging.error(f"❌ CONFLICT: Alias '{alias}' for command '{name}' conflicts with existing command/alias")
                continue  # Skip conflicting alias instead of overwriting
        commands[alias] = commands[name]
    
    logging.debug(f"✅ Registered command '{name}' with {len(validated_aliases)} aliases")

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
    
    logging.info("[Commands Setup] Starting command module initialization...")
    
    try:
        # Import command modules with individual error handling
        modules = {}
        module_imports = {
            'test_cmds': 'TestCommands',
            'memory_cmds': 'MemoryCommands', 
            'tts_cmds': 'TTSCommands',
            'config_commands': 'ConfigCommands',
            'video_commands': 'VideoCommands',
            'rag_commands': 'RAGCommands',
            'img_commands': 'ImgCommands'
        }
        
        for module_name, cog_name in module_imports.items():
            try:
                logging.info(f"[Commands Setup] Importing {module_name}...")
                module = __import__(f'bot.commands.{module_name}', fromlist=[module_name])
                modules[cog_name] = module
                logging.info(f"[Commands Setup] ✅ Successfully imported {module_name}")
            except Exception as import_error:
                logging.error(f"[Commands Setup] ❌ Failed to import {module_name}: {import_error}", exc_info=True)
                continue

        # Get currently loaded cogs to avoid duplicates
        loaded_cogs = set(bot.cogs.keys())
        logging.info(f"[Commands Setup] Currently loaded cogs: {list(loaded_cogs)}")

        # Load each cog with individual error handling
        successful_loads = 0
        failed_loads = 0
        
        for cog_name, module in modules.items():
            try:
                if cog_name not in loaded_cogs:
                    logging.info(f"[Commands Setup] Loading {cog_name} cog...")
                    
                    # Check if module has setup function
                    if not hasattr(module, 'setup'):
                        logging.error(f"[Commands Setup] ❌ {cog_name} module missing setup function")
                        failed_loads += 1
                        continue
                    
                    # Call the setup function
                    await module.setup(bot)
                    
                    # Verify the cog was loaded
                    if bot.get_cog(cog_name):
                        logging.info(f"[Commands Setup] ✅ {cog_name} loaded successfully")
                        successful_loads += 1
                    else:
                        logging.error(f"[Commands Setup] ❌ {cog_name} setup completed but cog not found")
                        failed_loads += 1
                else:
                    logging.debug(f"[Commands Setup] Skipping already loaded cog: {cog_name}")
                    
            except Exception as cog_error:
                logging.error(f"[Commands Setup] ❌ Failed to load {cog_name}: {cog_error}", exc_info=True)
                failed_loads += 1
                continue
        
        # Final status report
        final_cogs = list(bot.cogs.keys())
        logging.info(f"[Commands Setup] 🎉 Command setup complete: {successful_loads} loaded, {failed_loads} failed")
        logging.info(f"[Commands Setup] Final loaded cogs: {final_cogs}")
        
        # List all registered commands for debugging
        all_commands = []
        for cog in bot.cogs.values():
            cog_commands = [cmd.name for cmd in cog.get_commands()]
            all_commands.extend(cog_commands)
        
        logging.info(f"[Commands Setup] Total registered commands: {len(all_commands)}")
        logging.debug(f"[Commands Setup] Command list: {all_commands}")
        
        if failed_loads > 0:
            logging.warning(f"[Commands Setup] ⚠️ {failed_loads} cogs failed to load - some commands may not be available")
        
    except Exception as e:
        logging.error(f"[Commands Setup] ❌ Critical error during command setup: {e}", exc_info=True)
        raise

# Import all command modules to register them
# This must be done after the command registry is set up
# and after all command decorators are defined
