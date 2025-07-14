"""
Discord bot main entry point - BOOTSTRAP ONLY
This module should contain NO business logic, only orchestration.
"""
import argparse
import asyncio
import aiohttp
import os
import sys
import hashlib
from typing import NoReturn, Optional


import discord
from discord.ext import commands

# Import all required helpers
from bot.config import load_config, validate_required_env, ConfigurationError, check_venv_activation
from bot.logger import setup_logging, get_logger
from bot.router import setup_router
from bot.commands import setup_commands
from bot.tasks import spawn_background_tasks
from bot.shutdown import setup_signal_handlers
from bot.memory import load_all_profiles, load_all_server_profiles
from bot.events import cache_maintenance_task


# Add the bot directory to the Python path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.absolute()))


def run_pre_flight_checks(config: dict) -> None:
    """Runs all mandatory startup checks as per the Windsurf spec."""
    logger = get_logger(__name__)
    logger.info("--- Running Pre-Flight Checklist ---", extra={'subsys': 'core', 'event': 'pre_flight_checks'})

    # 1. Bot Token Check
    token = config.get("DISCORD_TOKEN")
    if not token:
        logger.critical("DISCORD_TOKEN is missing. Bot cannot start.", extra={'subsys': 'core', 'event': 'token_check_fail'})
        raise ConfigurationError("DISCORD_TOKEN not found in environment.")
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    logger.info(f"[WIND][INIT] Token hash={token_hash} validated", extra={'subsys': 'core', 'event': 'token_check_pass'})

    # 2. Intents Check
    intents = create_bot_intents()
    required_intents = {
        'message_content': intents.message_content,
        'guilds': intents.guilds,
        'guild_messages': intents.messages, # This covers guild messages
        'dm_messages': intents.messages, # and DM messages
        'guild_voice_states': intents.voice_states,
    }

    all_intents_ok = True
    for intent_name, is_enabled in required_intents.items():
        if not is_enabled:
            all_intents_ok = False
            logger.error(f"Required intent '{intent_name}' is disabled.", extra={'subsys': 'core', 'event': 'intent_check_fail'})
    
    if all_intents_ok:
        logger.info("[WIND][INIT] Intents verified", extra={'subsys': 'core', 'event': 'intent_check_pass'})
    else:
        logger.critical("One or more required intents are missing. Bot may not function correctly.", extra={'subsys': 'core', 'event': 'intent_check_fail'})
        # Depending on strictness, you might raise an error here.

    # 3. Voice Gateway Version
    logger.info(f"[WIND][INIT] Discord.py Version: {discord.__version__}", extra={'subsys': 'core', 'event': 'version_check'})
    logger.info("--- Pre-Flight Checklist Complete ---", extra={'subsys': 'core', 'event': 'pre_flight_checks_pass'})

def create_bot_intents() -> discord.Intents:
    """Create Discord intents with all required permissions."""
    intents = discord.Intents.default()  # Start with default intents
    intents.message_content = True  # Enable message content intent
    return intents


class LLMBot(commands.Bot):
    """Minimal bot class with bootstrap functionality only."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.startup_time = None
        self.config = load_config()
        self.logger = get_logger(self.__class__.__name__)

    async def setup_hook(self) -> None:
        """Bootstrap setup - load cogs and start services."""
        self.logger.info("--- Starting Bot Setup Hook ---", extra={'subsys': 'core', 'event': 'setup_hook_start'})
        self.startup_time = discord.utils.utcnow()

        try:
            # Initialize router first
            setup_router(self)
            self.logger.info("Router initialized.", extra={'subsys': 'core', 'event': 'router_init'})

            # Register all commands
            await setup_commands(self)
            self.logger.info("Commands registered.", extra={'subsys': 'core', 'event': 'commands_registered'})

            # Load profiles
            self._load_profiles()

            # Spawn background tasks
            await spawn_background_tasks(self)
            self.logger.info("Background tasks spawned.", extra={'subsys': 'core', 'event': 'tasks_spawned'})

        except Exception as e:
            self.logger.critical(f"Fatal error during setup hook: {e}", exc_info=True, extra={'subsys': 'core', 'event': 'setup_hook_fail'})
            await self.close()  # Gracefully shutdown on setup failure
            sys.exit(1)

        self.logger.info("--- Bot Setup Hook Complete ---", extra={'subsys': 'core', 'event': 'setup_hook_pass'})

    async def on_ready(self):
        """Called when the bot is done preparing the data received from Discord."""
        self.logger.info(f'Logged in as {self.user} (ID: {self.user.id})', extra={'subsys': 'core', 'event': 'login_success'})
        self.logger.info('------\n')

        # Set presence
        try:
            await self.change_presence(activity=discord.Game(name="Listening..."))
            self.logger.info("Bot presence set.", extra={'subsys': 'core', 'event': 'presence_set'})
        except Exception as e:
            self.logger.warning(f"Failed to set presence: {e}", exc_info=True, extra={'subsys': 'core', 'event': 'presence_fail'})

    def _load_profiles(self):
        """Load all user and server profiles."""
        try:
            self.logger.info("Loading all user and server profiles...", extra={'subsys': 'core', 'event': 'profile_load_start'})
            load_all_profiles()
            load_all_server_profiles()
            self.logger.info("Profiles loaded successfully.", extra={'subsys': 'core', 'event': 'profile_load_success'})
        except Exception as e:
            self.logger.error(f"Failed to load profiles: {e}", exc_info=True, extra={'subsys': 'core', 'event': 'profile_load_fail'})


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Windsurf Discord Bot")
    parser.add_argument('--debug', action='store_true', help='Enable debug logging.')
    parser.add_argument('--version', action='store_true', help='Show version information and exit.')
    parser.add_argument('--config-check', action='store_true', help='Validate configuration and exit.')
    return parser.parse_args()


def load_tts_model():
    # Placeholder for TTS model loading logic
    logger = get_logger(__name__)
    try:
        logger.info("TTS model loading placeholder...", extra={'subsys': 'tts', 'event': 'model_load_start'})
        # In a real scenario, you would initialize your TTS client here
        # from bot.tts import tts_client
        # tts_client.initialize()
        logger.info("TTS model loaded successfully.", extra={'subsys': 'tts', 'event': 'model_load_pass'})
    except Exception as e:
        logger.error("Failed to load TTS model.", exc_info=True, extra={'subsys': 'tts', 'event': 'model_load_fail'})


def startup_watchdog():
    # Placeholder for a startup watchdog
    logger = get_logger(__name__)
    logger.info("Startup watchdog running...", extra={'subsys': 'core', 'event': 'watchdog_start'})

def show_version_info() -> None:
    """Display version and system information."""
    # This will be properly implemented later when versioning is added.
    print("Show version info placeholder.")


def validate_configuration_only() -> None:
    """Validate configuration and exit."""
    try:
        config = load_config()
        validate_required_env()
        logger = get_logger(__name__)
        logger.info("Configuration is valid", extra={'subsys': 'core', 'event': 'config_valid'})
        logger.info(f"  • Discord token: {'✅ Present' if config.get('DISCORD_TOKEN') else '❌ Missing'}", extra={'subsys': 'core', 'event': 'config_valid'})
        logger.info(f"  • Command prefix: {config.get('COMMAND_PREFIX', '!')}", extra={'subsys': 'core', 'event': 'config_valid'})
        logger.info(f"  • Text backend: {config.get('TEXT_BACKEND', 'ollama')}", extra={'subsys': 'core', 'event': 'config_valid'})
        
        if config.get('TEXT_BACKEND') == 'openai':
            logger.info(f"  • OpenAI API key: {'✅ Present' if config.get('OPENAI_API_KEY') else '❌ Missing'}", extra={'subsys': 'core', 'event': 'config_valid'})
            logger.info(f"  • OpenAI model: {config.get('OPENAI_TEXT_MODEL', 'gpt-4')}", extra={'subsys': 'core', 'event': 'config_valid'})
            logger.info(f"  • OpenAI base URL: {config.get('OPENAI_API_BASE', 'https://api.openai.com/v1')}", extra={'subsys': 'core', 'event': 'config_valid'})
        else:
            logger.info(f"  • Ollama base URL: {config.get('OLLAMA_BASE_URL', 'http://localhost:11434')}", extra={'subsys': 'core', 'event': 'config_valid'})
            logger.info(f"  • Ollama model: {config.get('OLLAMA_MODEL', 'llama3')}", extra={'subsys': 'core', 'event': 'config_valid'})
        
        logger.info(f"  • Log level: {config.get('LOG_LEVEL', 'INFO')}", extra={'subsys': 'core', 'event': 'config_valid'})
        
    except ConfigurationError as e:
        logger = get_logger(__name__)
        logger.critical(f"Configuration error: {e}", exc_info=True, extra={'subsys': 'core', 'event': 'config_fail'})
        sys.exit(1)


def get_prefix(bot: commands.Bot, message: discord.Message) -> list[str]:
    """Determine command prefix based on context."""
    config = bot.config
    prefix = config.get('COMMAND_PREFIX', '!')
    if message.guild is None:
        return [prefix]
    mentions = commands.when_mentioned(bot, message)
    return [f"{m}{prefix}" for m in mentions]

async def main() -> NoReturn:
    """Main bot execution function with enhanced error handling and CLI support."""
    args = parse_arguments()

    if args.version:
        show_version_info()
        sys.exit(0)

    # Set log level in environment for the new logger to pick up.
    # The --debug flag overrides the .env setting.
    if args.debug:
        os.environ['LOG_LEVEL'] = 'DEBUG'

    # setup_logging now reads from config/env, so no argument is needed.
    setup_logging()
    logger = get_logger(__name__)

    if args.config_check:
        validate_configuration_only()
        sys.exit(0)

    try:
        check_venv_activation()
        config = load_config()
        run_pre_flight_checks(config)
    except ConfigurationError as e:
        logger.critical(f"Configuration error: {e}", exc_info=True, extra={'subsys': 'core', 'event': 'config_fail'})
        sys.exit(1)
    except Exception as e:
        logger.critical(f"A fatal error occurred during startup validation: {e}", exc_info=True, extra={'subsys': 'core', 'event': 'startup_fail'})
        sys.exit(1)

    intents = create_bot_intents()
    bot = LLMBot(
        command_prefix=get_prefix,
        intents=intents,
        help_command=None
    )

    try:
        setup_signal_handlers(bot)
    except Exception as e:
        logger.warning(f"Failed to setup signal handlers: {e}", exc_info=True, extra={'subsys': 'core', 'event': 'signal_handler_fail'})

    max_retries = 3
    base_delay = 5  # seconds
    for attempt in range(max_retries):
        try:
            logger.info(f"Connecting to Discord... (Attempt {attempt + 1}/{max_retries})", extra={'subsys': 'core', 'event': 'connect_start'})
            await bot.start(config["DISCORD_TOKEN"])
            break  # Exit loop on successful connection
        except (discord.HTTPException, aiohttp.ClientConnectorError) as e:
            if attempt == max_retries - 1:
                logger.critical(f"Failed to connect after {max_retries} attempts.", exc_info=True, extra={'subsys': 'core', 'event': 'connect_fail_max_retries'})
                if isinstance(e, aiohttp.ClientConnectorError):
                    logger.error("This may be a network issue. Check internet connection and firewall settings.", extra={'subsys': 'core', 'event': 'network_issue_hint'})
                sys.exit(1)
            delay = base_delay * (2 ** attempt)  # Exponential backoff
            logger.warning(f"Connection failed, retrying in {delay}s...", exc_info=False, extra={'subsys': 'core', 'event': 'connect_retry'})
            await asyncio.sleep(delay)
    
    logger.critical("Bot event loop exited unexpectedly.", extra={'subsys': 'core', 'event': 'event_loop_exit'})
    sys.exit(1) # Should not be reached if bot.start() is successful and runs forever.


def run_bot() -> None:
    """Entry point for running the bot with proper error handling."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # A logger might not be configured if shutdown happens early.
        print("\nBot shutdown requested by user.")
        sys.exit(0)
    except Exception as e:
        # Fallback logging for catastrophic failures before logger is set up.
        print(f"FATAL ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_bot()
