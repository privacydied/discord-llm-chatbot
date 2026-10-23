"""Discord bot main entry point - BOOTSTRAP ONLY
This module should contain NO business logic, only orchestration.
"""

import asyncio
import contextlib
import os
import sys
from typing import Any, NoReturn

import aiohttp
import discord

from bot.config import check_venv_activation, load_config, load_system_prompts
from bot.config_reload import setup_config_reload, start_file_watcher
from bot.core.bot import LLMBot
from bot.core.cli import parse_arguments, show_version_info, validate_configuration_only
from bot.core.startup import create_bot_intents, get_prefix, run_pre_flight_checks
from bot.exceptions import ConfigurationError
from bot.shutdown import setup_signal_handlers
from bot.tasks import spawn_background_tasks
from bot.utils.logging import get_logger, init_logging, shutdown_logging_and_exit

# ---------------------------------------------------------------------------
# Phase 3: Thread caps — MUST run before heavy ML modules are initialized.
# Use setdefault so explicit user env vars take priority.
# These are set after the standard imports above; heavy ML libs (torch, numpy)
# are not imported transitively by the modules imported above, so this is safe.
# ---------------------------------------------------------------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _tune_glibc_malloc_arenas() -> None:
    """Cap glibc malloc arenas to curb per-thread heap fragmentation. [PA][CMV]

    STT/TTS/RAG work runs across executor threads; glibc's default arena cap
    (8 * ncpus) lets each thread claim its own arena, and arenas don't shrink
    back to the OS once grown even after their allocations are freed. That
    fragmentation was a measured large contributor to this process's resident
    memory while otherwise idle. `mallopt()` is a live runtime call -- unlike
    the MALLOC_ARENA_MAX env var, it doesn't depend on being set before the
    process starts, so it's applied here unconditionally. Linux/glibc only;
    a no-op everywhere else (musl, macOS, Windows).
    """
    if not sys.platform.startswith("linux"):
        return
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        m_arena_max = -8  # glibc malloc.h: M_ARENA_MAX
        max_arenas = int(os.getenv("MALLOC_ARENA_MAX", "2"))
        libc.mallopt(m_arena_max, max_arenas)
    except (OSError, AttributeError, ValueError):
        pass  # best-effort tuning; never block startup on this


_tune_glibc_malloc_arenas()


async def main(bot_ref: dict[str, LLMBot] | None = None) -> NoReturn:
    """Main bot execution function with enhanced error handling and CLI support."""
    init_logging()
    logger = get_logger(__name__)
    args = parse_arguments()

    if args.version:
        show_version_info()
        shutdown_logging_and_exit(0)

    if args.debug:
        os.environ["LOG_LEVEL"] = "DEBUG"

    logger = get_logger(__name__)

    if args.config_check:
        validate_configuration_only()
        shutdown_logging_and_exit(0)

    try:
        check_venv_activation()

        # Set up dynamic configuration reload system
        setup_config_reload()

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

    # Discord message cache size — respect LOW_RESOURCE_MODE default
    _max_msgs = int(config.get("DISCORD_MESSAGE_CACHE_MAX", 256))
    bot = LLMBot(
        config=config,
        command_prefix=get_prefix,
        intents=intents,
        help_command=None,
        max_messages=_max_msgs,
    )
    if bot_ref is not None:
        bot_ref["bot"] = bot

    try:
        setup_signal_handlers(bot)
    except (OSError, RuntimeError, ValueError) as e:
        logger.warning(f"Failed to setup signal handlers: {e}", exc_info=True)

    # Set up background tasks and file watcher before starting bot
    await spawn_background_tasks(bot)
    await start_file_watcher()

    exit_code = await connect_with_retries(bot, config["DISCORD_TOKEN"], logger)
    shutdown_logging_and_exit(exit_code)


CONNECT_MAX_ATTEMPTS = 6
CONNECT_BASE_DELAY_S = 5.0
CONNECT_MAX_DELAY_S = 60.0


def _is_transient_connect_error(exc: BaseException) -> bool:
    """True when ``exc`` (or what it was raised while handling) is a network/5xx blip.

    A gateway 503 on the initial websocket handshake surfaces as discord.py's own
    ``AttributeError: 'NoneType' object has no attribute 'sequence'`` (it tries to
    resume a websocket that never opened) raised while handling the
    ``WSServerHandshakeError``, so the whole chain is checked. [REH]
    """
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, discord.LoginFailure):
            return False
        if isinstance(cur, discord.HTTPException):
            return int(getattr(cur, "status", 0) or 0) >= 500 or getattr(cur, "status", None) == 429
        if isinstance(cur, (aiohttp.ClientError, discord.GatewayNotFound, discord.ConnectionClosed, OSError, TimeoutError)):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


async def connect_with_retries(bot: Any, token: str, logger: Any, sleep: Any = asyncio.sleep) -> int:
    """Log in once, then connect; retry transient failures with capped backoff.

    Only ``connect`` is retried: ``login`` runs ``setup_hook`` (cogs, tasks), so
    calling ``bot.start`` again would set everything up twice. Returns the exit code.
    """
    logged_in = False
    for attempt in range(1, CONNECT_MAX_ATTEMPTS + 1):
        try:
            logger.info(f"Connecting to Discord... (Attempt {attempt}/{CONNECT_MAX_ATTEMPTS})")
            if not logged_in:
                await bot.login(token)
                logged_in = True
            logger.info("🚀 Bot is ready and operational!")
            await bot.connect(reconnect=True)  # blocks until the bot disconnects
            logger.info("Bot disconnected gracefully")
            return 0
        except discord.LoginFailure:
            logger.critical("Discord rejected the token (LoginFailure). Check DISCORD_TOKEN.")
            return 1
        except Exception as e:
            if not _is_transient_connect_error(e) or attempt == CONNECT_MAX_ATTEMPTS:
                logger.critical(f"Unexpected error during bot execution: {e}", exc_info=True)
                return 1
            delay = min(CONNECT_BASE_DELAY_S * (2 ** (attempt - 1)), CONNECT_MAX_DELAY_S)
            logger.warning(f"Discord connection failed ({type(e).__name__}: {e}); retrying in {delay:.0f}s")
            await sleep(delay)
    return 1


def run_bot() -> None:
    """Entry point for running the bot with proper error handling."""
    try:
        # Store bot instance for proper cleanup
        asyncio.run(main_with_cleanup())
    except KeyboardInterrupt:
        shutdown_logging_and_exit(0)
    except BaseException:  # noqa: BLE001 - top-level crash handler; KeyboardInterrupt handled above
        import traceback

        traceback.print_exc()
        shutdown_logging_and_exit(1)


async def main_with_cleanup() -> NoReturn:
    """Main function with proper resource cleanup."""
    bot_state: dict[str, LLMBot] = {}
    try:
        # Run the main bot logic
        await main(bot_state)
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Error in main: {e}", exc_info=True)
        raise
    finally:
        # Ensure proper cleanup even if main() fails
        bot_instance = bot_state.get("bot")
        if bot_instance:
            with contextlib.suppress(Exception):
                await bot_instance.close()


if __name__ == "__main__":
    run_bot()
