"""
Discord bot main entry point - BOOTSTRAP ONLY
This module should contain NO business logic, only orchestration.
"""
import asyncio
import os
import sys
from typing import NoReturn

import discord
from discord.ext import commands

# Import all required helpers
from .config import load_config, validate_required_env, ConfigurationError
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
        
        # Set presence
        activity = discord.Activity(
            type=discord.ActivityType.listening,
            name=f"{self.config.get('COMMAND_PREFIX', '!')}help"
        )
        await self.change_presence(activity=activity)
    
    async def _load_profiles(self) -> None:
        """Load all user and server profiles."""
        load_all_profiles()
        load_all_server_profiles()
        print(f"✅ Loaded {len(user_profiles)} user profiles and {len(server_profiles)} server profiles")
    
    async def on_ready(self) -> None:
        """Bootstrap completion notification."""
        if not hasattr(self, 'startup_time'):
            self.startup_time = discord.utils.utcnow()
        print(f"✅ {self.user} is ready! Connected to {len(self.guilds)} guilds")


async def _main() -> NoReturn:
    """Main bot execution function."""
    # Configure logging first
    configure_logging()
    
    # Load and validate configuration
    try:
        config = load_config()
        validate_required_env()
    except ConfigurationError as e:
        print(f"❌ Configuration error: {e}")
        sys.exit(1)
    
    # Create bot instance
    bot = LLMBot(
        command_prefix=commands.when_mentioned_or(config.get("COMMAND_PREFIX", "!")),
        intents=create_bot_intents(),
        case_insensitive=True
    )
    
    # Setup signal handlers
    setup_signal_handlers(bot)
    
    # Start the bot
    try:
        await bot.start(config["DISCORD_TOKEN"])
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        print("\n👋 Bot shutdown requested by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
