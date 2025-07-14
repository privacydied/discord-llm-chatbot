"""
Discord bot main entry point - BOOTSTRAP ONLY
This module should contain NO business logic, only orchestration.
"""
import argparse
import asyncio
import aiohttp
import os
import sys
from typing import NoReturn, Optional

import discord
from discord.ext import commands

# Import all required helpers
from bot.config import load_config, validate_required_env, ConfigurationError, audit_env_file, validate_prompt_files, check_venv_activation
from bot.logs import configure_logging
from bot.router import setup_router, get_router
from bot.tasks import spawn_background_tasks
from bot.shutdown import setup_signal_handlers
from bot.memory import load_all_profiles, load_all_server_profiles, user_profiles, server_profiles
from bot.tts_manager import tts_manager
from bot.tts_state import tts_state
from bot.events import cache_maintenance_task

# Add the bot directory to the Python path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.absolute()))


def create_bot_intents() -> discord.Intents:
    """Create Discord intents with all required permissions."""
    intents = discord.Intents.default()  # Start with default intents
    intents.message_content = True  # Enable message content intent
    
    # Log the intents being used
    print("\n🔍 Bot Intents:")
    print(f"  - message_content: {intents.message_content}")
    print(f"  - members: {intents.members}")
    print(f"  - guilds: {intents.guilds}")
    print(f"  - messages: {intents.messages}")
    print(f"  - guild_messages: {intents.guild_messages}")
    print(f"  - dm_messages: {intents.dm_messages}")
    
    return intents


class LLMBot(commands.Bot):
    """Minimal bot class with bootstrap functionality only."""
    
    def __init__(self, command_prefix, intents, *args, **kwargs):
        super().__init__(command_prefix, intents=intents, *args, **kwargs)
        self.startup_time = None
        self.config = load_config()
    
    async def setup_hook(self) -> None:
        """Bootstrap setup - load cogs and start services."""
        print("🔧 Starting bot setup...")
        self.startup_time = discord.utils.utcnow()
        
        try:
            # Initialize router first
            print("🔄 Initializing router...")
            self.router = setup_router(self)
            print("✅ Router initialized")
            
            # Load all user and server profiles
            print("🔄 Loading profiles...")
            await self._load_profiles()
            print("✅ Profiles loaded")
            
            # Load all command cogs
            print("\n🔄 Loading command cogs...")
            for ext in ['bot.commands.memory_cmds', 'bot.commands.tts_cmds']:
                try:
                    print(f"  ⏳ Loading {ext}...")
                    await self.load_extension(ext)
                    print(f"  ✅ Loaded extension: {ext}")
                except Exception as e:
                    print(f"  ❌ Failed to load extension {ext}: {str(e)}")
                    import traceback
                    traceback.print_exc()
            
            # Load events last
            print("\n🔄 Loading events extension...")
            try:
                await self.load_extension('bot.events')
                print("✅ Events extension loaded")
            except Exception as e:
                print(f"❌ Failed to load events extension: {str(e)}")
                import traceback
                traceback.print_exc()
            
            # Set up background tasks
            print("\n🔄 Setting up background tasks...")
            await spawn_background_tasks(self)
            asyncio.create_task(cache_maintenance_task())
            
            print("\n🎉 Bot setup complete!")
            
        except Exception as e:
            print(f"❌ Critical error during setup: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    async def _load_profiles(self) -> None:
        """Load all user and server profiles."""
        load_all_profiles()
        load_all_server_profiles()
        print(f"✅ Loaded {len(user_profiles)} user profiles and {len(server_profiles)} server profiles")
    
    async def on_ready(self) -> None:
        """Bootstrap completion notification."""
        import logging
        
        if not hasattr(self, 'startup_time'):
            self.startup_time = discord.utils.utcnow()
        
        # Set presence now that bot is connected
        activity = discord.Activity(
            type=discord.ActivityType.listening,
            name=f"{self.config.get('COMMAND_PREFIX', '!')}help"
        )
        await self.change_presence(activity=activity)
        
        print(f"✅ {self.user} is ready! Connected to {len(self.guilds)} guilds")
        
        # Start TTS initialization in background
        async def load_tts_model():
            try:
                # Direct async call with timeout
                await asyncio.wait_for(tts_manager.load_model(), timeout=30.0)
                logging.info("🔊 Kokoro-ONNX TTS initialized successfully")
                tts_manager.set_available(True)
                
                # Log cache stats after successful init
                cache_stats = tts_manager.get_cache_stats()
                logging.info(f"🔊 TTS cache: {cache_stats['files']} files ({cache_stats['size_mb']:.1f}MB)")
            except asyncio.TimeoutError:
                logging.error("❌ TTS initialization timed out after 30 seconds")
                tts_manager.set_available(False)
            except Exception as e:
                logging.error(f"❌ TTS initialization failed: {str(e)}")
                tts_manager.set_available(False)
        
        # Start TTS initialization task
        self.tts_init_task = asyncio.create_task(load_tts_model())
        
        # Startup watchdog to ensure bot remains responsive
        async def startup_watchdog():
            await asyncio.sleep(60)  # Wait 60 seconds
            if not tts_manager.is_available():
                logging.error("❌ TTS initialization watchdog: TTS still unavailable after 60 seconds")
                tts_manager.set_available(False)  # Force availability to false
        
        asyncio.create_task(startup_watchdog())


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Discord LLM Chatbot - Production-ready AI Discord bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m bot.main                    # Start the bot normally
  python -m bot.main --debug            # Enable debug logging
  python -m bot.main --config-check     # Validate configuration only
  python -m bot.main --version          # Show version information
"""
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    parser.add_argument(
        "--config-check",
        action="store_true",
        help="Validate configuration and exit"
    )
    
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version information and exit"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set log level (overrides LOG_LEVEL env var)"
    )
    
    return parser.parse_args()


def show_version_info() -> None:
    """Display version and system information."""
    from . import __version__, __title__, __author__, __description__
    
    print(f"{__title__} v{__version__}")
    print(f"Description: {__description__}")
    print(f"Author: {__author__}")
    print(f"Python: {sys.version}")
    print(f"Discord.py: {discord.__version__}")
    print(f"Platform: {sys.platform}")


def validate_configuration_only() -> None:
    """Validate configuration and exit."""
    try:
        config = load_config()
        validate_required_env()
        print("✅ Configuration is valid")
        print(f"  • Discord token: {'✅ Present' if config.get('DISCORD_TOKEN') else '❌ Missing'}")
        print(f"  • Command prefix: {config.get('COMMAND_PREFIX', '!')}")
        print(f"  • Text backend: {config.get('TEXT_BACKEND', 'ollama')}")
        
        if config.get('TEXT_BACKEND') == 'openai':
            print(f"  • OpenAI API key: {'✅ Present' if config.get('OPENAI_API_KEY') else '❌ Missing'}")
            print(f"  • OpenAI model: {config.get('OPENAI_TEXT_MODEL', 'gpt-4')}")
            print(f"  • OpenAI base URL: {config.get('OPENAI_API_BASE', 'https://api.openai.com/v1')}")
        else:
            print(f"  • Ollama base URL: {config.get('OLLAMA_BASE_URL', 'http://localhost:11434')}")
            print(f"  • Ollama model: {config.get('OLLAMA_MODEL', 'llama3')}")
        
        print(f"  • Log level: {config.get('LOG_LEVEL', 'INFO')}")
        
    except ConfigurationError as e:
        print(f"❌ Configuration error: {e}")
        sys.exit(1)


async def main(args: Optional[argparse.Namespace] = None) -> None:
    """Main bot execution function with enhanced error handling and CLI support."""
    if args is None:
        args = parse_arguments()
    
    # Handle version request
    if args.version:
        show_version_info()
        return
    
    # Handle configuration check
    if args.config_check:
        validate_configuration_only()
        return
    
    # Configure logging first
    configure_logging()
    
    # Override log level if specified
    if args.log_level:
        import logging
        logging.getLogger().setLevel(getattr(logging, args.log_level))
        print(f"Log level set to {args.log_level}")
    
    if args.debug:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
        print("Debug logging enabled")
    
    # CHANGE: Enforce .venv usage and perform comprehensive startup validation
    try:
        # Step 1: Check .venv activation
        check_venv_activation()
        
        # Step 2: Audit .env file (logs all variables)
        audit_env_file()
        
        # Step 3: Load and validate configuration
        config = load_config()
        validate_required_env()
        
        # Step 4: Validate prompt files exist and are readable
        validate_prompt_files()
        
        print("✅ All startup validations passed")
        
    except ConfigurationError as e:
        print(f"❌ Configuration error: {e}")
        print("💡 Use --config-check to validate your configuration")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Startup validation error: {e}")
        print("💡 Check your .env file and prompt files")
        sys.exit(1)
    
    # Create bot instance
    intents = discord.Intents.default()
    intents.message_content = True
    
    import logging
    logger = logging.getLogger(__name__)
    
    def get_prefix(bot: commands.Bot, message: discord.Message) -> list[str]:
        """Determine command prefix based on context."""
        # Load prefix from config
        config = load_config()
        prefix = config.get('COMMAND_PREFIX', '!')
        
        # In DMs, use just the prefix
        if message.guild is None:
            return [prefix]
            
        # In guilds, require both mention and the prefix
        # This returns patterns like ['<@bot_id> !', '<@!bot_id> !'] 
        mentions = commands.when_mentioned(bot, message)
        return [f"{m}{prefix}" for m in mentions]
    
    bot = LLMBot(
        command_prefix=get_prefix,
        intents=intents,
        help_command=None,
    )
    
    # Setup signal handlers
    try:
        setup_signal_handlers(bot)
    except Exception as e:
        print(f"⚠️  Warning: Failed to setup signal handlers: {e}")
    
    # Start the bot with retry logic
    max_retries = 3
    base_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            print(f"🚀 Connection attempt {attempt + 1}/{max_retries}")
            await bot.start(config["DISCORD_TOKEN"])
            break
        except (discord.HTTPException, aiohttp.ClientConnectorError) as e:
            last_exception = e
            if attempt == max_retries - 1:
                print(f"❌ Failed after {max_retries} attempts. Last error: {type(e).__name__}: {e}")
                if isinstance(e, aiohttp.ClientConnectorError):
                    print("💡 Check your internet connection and firewall settings")
                sys.exit(1)
            delay = base_delay * (attempt + 1)
            print(f"⚠️  Retrying in {delay}s... (Error: {e})")
            await asyncio.sleep(delay)
    
    # Start the bot
    try:
        print(f"🚀 Starting {bot.__class__.__name__}...")
    except discord.LoginFailure:
        print("❌ Invalid Discord token. Please check your DISCORD_TOKEN environment variable.")
        sys.exit(1)
    except discord.HTTPException as e:
        print(f"❌ Discord HTTP error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_bot() -> None:
    """Entry point for running the bot with proper error handling."""
    try:
        args = parse_arguments()
        asyncio.run(main(args))
    except KeyboardInterrupt:
        print("\n👋 Bot shutdown requested by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


asyncio.run(main())
