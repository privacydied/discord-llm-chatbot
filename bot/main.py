"""
Discord bot main entry point - BOOTSTRAP ONLY
This module should contain NO business logic, only orchestration.
"""
import asyncio
import os
import sys
from typing import NoReturn

import aiohttp
import discord

from bot.config import load_config, check_venv_activation, ConfigurationError, load_system_prompts
from bot.core.bot import LLMBot
from bot.core.cli import parse_arguments, show_version_info, validate_configuration_only
from bot.core.startup import run_pre_flight_checks, create_bot_intents, get_prefix
from bot.util.logging import init_logging, get_logger, shutdown_logging_and_exit
from bot.shutdown import setup_signal_handlers


async def main() -> NoReturn:
    """Main bot execution function with enhanced error handling and CLI support."""
    init_logging()
    logger = get_logger(__name__)
    args = parse_arguments()

    if args.version:
        show_version_info()
        shutdown_logging_and_exit(0)

    if args.debug:
        os.environ['LOG_LEVEL'] = 'DEBUG'

    logger = get_logger(__name__)

    if args.config_check:
        validate_configuration_only()
        shutdown_logging_and_exit(0)

    try:
        check_venv_activation()
        config = load_config()
        config.update(load_system_prompts())
        run_pre_flight_checks(config)
    except ConfigurationError as e:
        logger.critical(f"Unhandled exception during bot startup: {e}", exc_info=True)
        shutdown_logging_and_exit(1)
    except Exception as e:
        logger.critical(f"A fatal error occurred during startup validation: {e}", exc_info=True)
        shutdown_logging_and_exit(1)

    intents = create_bot_intents()
    bot = LLMBot(
        config=config,
        command_prefix=get_prefix,
        intents=intents,
        help_command=None
    )

    try:
        setup_signal_handlers(bot)
    except Exception as e:
        logger.warning(f"Failed to setup signal handlers: {e}", exc_info=True)

    max_retries = 3
    base_delay = 5  # seconds
    for attempt in range(max_retries):
        try:
            logger.info(f"Connecting to Discord... (Attempt {attempt + 1}/{max_retries})")
            await bot.start(config["DISCORD_TOKEN"])
            break
        except (discord.HTTPException, aiohttp.ClientConnectorError):
            if attempt == max_retries - 1:
                logger.error("Failed to log in. Please check your Discord token.")
                shutdown_logging_and_exit(1)
            delay = base_delay * (2 ** attempt)
            logger.warning(f"Connection failed, retrying in {delay}s...")
            await asyncio.sleep(delay)
    
    logger.critical("Bot event loop exited unexpectedly.")
    shutdown_logging_and_exit(1)


def run_bot() -> None:
    """Entry point for running the bot with proper error handling."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot shutdown requested by user.")
        shutdown_logging_and_exit(0)
    except Exception as e:
        print(f"FATAL ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        shutdown_logging_and_exit(1)


if __name__ == "__main__":
    run_bot()
