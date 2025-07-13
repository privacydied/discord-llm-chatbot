"""
Discord bot main entry point - BOOTSTRAP ONLY
This module should contain NO business logic, only orchestration.
"""
import argparse
import asyncio
import os
import sys
from typing import NoReturn, Optional

import discord
from discord.ext import commands

# Import all required helpers
from .config import load_config, validate_required_env, ConfigurationError, audit_env_file, validate_prompt_files, check_venv_activation
from .logs import configure_logging
from .commands import setup_commands
from .tasks import spawn_background_tasks
from .shutdown import setup_signal_handlers
from .memory import load_all_profiles, load_all_server_profiles, user_profiles, server_profiles
from .tts import setup_tts


def create_bot_intents() -> discord.Intents:
    """Create Discord intents based on environment configuration."""
    intents = discord.Intents.default()
    intents.message_content = os.getenv("ENABLE_MSG_INTENT", "true").lower() == "true"
    intents.members = True
    intents.guilds = True
    return intents


class LLMBot(commands.Bot):
    """Minimal bot class with bootstrap functionality only."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.startup_time = None
        self.config = load_config()
    
    async def setup_hook(self) -> None:
        """Bootstrap setup - load cogs and start services."""
        self.startup_time = discord.utils.utcnow()
        await self._load_profiles()
        await setup_tts()
        await setup_commands(self)
        await spawn_background_tasks(self)
    
    async def _load_profiles(self) -> None:
        """Load all user and server profiles."""
        load_all_profiles()
        load_all_server_profiles()
        print(f"✅ Loaded {len(user_profiles)} user profiles and {len(server_profiles)} server profiles")
    
    async def on_ready(self) -> None:
        """Bootstrap completion notification."""
        if not hasattr(self, 'startup_time'):
            self.startup_time = discord.utils.utcnow()
        
        # Set presence now that bot is connected
        activity = discord.Activity(
            type=discord.ActivityType.listening,
            name=f"{self.config.get('COMMAND_PREFIX', '!')}help"
        )
        await self.change_presence(activity=activity)
        
        print(f"✅ {self.user} is ready! Connected to {len(self.guilds)} guilds")


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
    try:
        bot = LLMBot(
            command_prefix=commands.when_mentioned_or(config.get("COMMAND_PREFIX", "!")),
            intents=create_bot_intents(),
            case_insensitive=True
        )
    except Exception as e:
        print(f"❌ Failed to create bot instance: {e}")
        sys.exit(1)
    
    # Setup signal handlers
    try:
        setup_signal_handlers(bot)
    except Exception as e:
        print(f"⚠️  Warning: Failed to setup signal handlers: {e}")
    
    # Start the bot
    try:
        print(f"🚀 Starting {bot.__class__.__name__}...")
        await bot.start(config["DISCORD_TOKEN"])
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


if __name__ == "__main__":
    run_bot()
