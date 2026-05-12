"""
Contains bot startup and pre-flight check logic.
"""

import discord
from discord.ext import commands
import hashlib
import os
import socket
import subprocess
import urllib.parse
from pathlib import Path
from typing import Optional

from bot.config import ConfigurationError
from bot.utils.logging import get_logger


def _get_playwright_chromium_path() -> Optional[Path]:
    """Checks for the Playwright Chromium executable and returns its path if found."""
    try:
        cache_dir = Path.home() / ".cache/ms-playwright"
        if not cache_dir.exists():
            return None

        for item in cache_dir.iterdir():
            if item.is_dir() and item.name.startswith("chromium-"):
                executable_path = item / "chrome-linux" / "chrome"
                if executable_path.exists():
                    return executable_path
        return None
    except Exception:
        return None


def _validate_remote_playwright_url(raw_url: str, logger) -> None:
    """Validate that PW_SERVER_URL is reachable and version is aligned.

    Raises ConfigurationError when the remote server cannot be reached
    or the client/server Playwright versions are mismatched.
    """
    import importlib.metadata as _meta

    parsed = urllib.parse.urlparse(raw_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 3006

    # Verify that the client Playwright version matches the Docker server
    # (must be updated in lockstep with requirements.txt).
    try:
        client_ver = _meta.version("playwright")
    except Exception:
        client_ver = "(unknown)"
    expected_server_ver = "1.59"  # must track requirements.txt
    if not client_ver.startswith(expected_server_ver.split(".")[0] + "."):
        logger.warning(
            f"Playwright version mismatch: client={client_ver}, "
            f"expected~=server {expected_server_ver}. "
            f"Run: pip install playwright=={expected_server_ver}"
        )

    try:
        sock = socket.create_connection((host, port), timeout=5)
        sock.close()
        logger.info(
            f"Playwright remote server reachable at {host}:{port} "
            f"(client v{client_ver}, expected server v{expected_server_ver})"
        )
    except (ConnectionRefusedError, socket.timeout, OSError) as exc:
        logger.warning(
            f"Playwright remote server at {host}:{port} is unreachable: {exc}. "
            f"Features requiring Playwright (web scraping, screenshots) will be unavailable. "
            f"Start the container to restore: sudo docker restart playwright"
        )


def check_playwright_browsers(logger) -> None:
    """Checks for Playwright browsers and validates the configured path.

    When PW_SERVER_URL is set (remote Playwright server), we validate
    that the remote server is reachable.  When not set, we check for
    local browser binaries.
    """
    pw_server = os.getenv("PW_SERVER_URL", "").strip()
    if pw_server:
        logger.info(
            f"Playwright remote server configured ({pw_server}); validating reachability"
        )
        _validate_remote_playwright_url(pw_server, logger)
        return

    logger.info("Checking for Playwright browser binaries...")

    if _get_playwright_chromium_path():
        logger.info("✅ Playwright Browser (Chromium) is installed.")
        return

    logger.warning(
        "❌ Playwright Browser (Chromium) not found. Attempting auto-installation..."
    )

    try:
        result = subprocess.run(
            ["uv", "run", "playwright", "install", "chromium"],
            capture_output=True,
            text=True,
            check=True,
            timeout=300,
        )
        logger.info(f"Playwright auto-install successful: {result.stdout}")
        if _get_playwright_chromium_path():
            logger.info("✅ Successfully installed Playwright Browser (Chromium).")
        else:
            logger.error(
                "📛 Failed to install Playwright Browser automatically. Please run `uv run playwright install chromium` manually."
            )

    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.error(f"📛 Failed to auto-install Playwright browser: {e.stderr or e}")
    except FileNotFoundError:
        logger.error(
            "📛 `uv` or `playwright` command not found. Cannot auto-install browsers."
        )


def run_pre_flight_checks(config: dict) -> None:
    """Runs all mandatory startup checks as per the Windsurf spec."""
    logger = get_logger(__name__)
    logger.info("--- Running Pre-Flight Checklist ---")

    # 1. Bot Token Check
    token = config.get("DISCORD_TOKEN")
    if not token:
        logger.critical("DISCORD_TOKEN is missing. Bot cannot start.")
        raise ConfigurationError("DISCORD_TOKEN not found in environment.")
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    logger.info(f"[WIND][INIT] Token hash={token_hash} validated")

    # 2. Intents Check
    intents = create_bot_intents()
    required_intents = {
        "message_content": intents.message_content,
        "guilds": intents.guilds,
        "guild_messages": intents.messages,  # This covers guild messages
        "dm_messages": intents.messages,  # and DM messages
        "guild_voice_states": intents.voice_states,
    }

    all_intents_ok = True
    for intent_name, is_enabled in required_intents.items():
        if not is_enabled:
            all_intents_ok = False
            logger.error(f"Required intent '{intent_name}' is disabled.")

    if all_intents_ok:
        logger.info("[WIND][INIT] Intents verified")
    else:
        logger.critical(
            "One or more required intents are missing. Bot may not function correctly."
        )
        # Depending on strictness, you might raise an error here.

    # 3. Voice Gateway Version
    logger.info(f"[WIND][INIT] Discord.py Version: {discord.__version__}")

    # 4. Playwright Browser Check
    check_playwright_browsers(logger)

    logger.info("--- Pre-Flight Checklist Complete ---")


def create_bot_intents() -> discord.Intents:
    """Create Discord intents with all required permissions."""
    intents = discord.Intents.default()
    intents.message_content = True
    intents.guilds = True
    intents.members = True  # Required for many user-related operations
    intents.voice_states = True  # Required for TTS
    return intents


def get_prefix(bot, message: discord.Message) -> list[str]:
    """Dynamically get the command prefix for the bot."""
    # Base prefixes from env (comma-separated), stripped and non-empty
    base_prefixes = [
        p.strip() for p in os.getenv("BOT_PREFIX", "!").split(",") if p.strip()
    ]

    # Standard behavior: allow mention OR base prefixes
    dynamic = commands.when_mentioned_or(*base_prefixes)
    prefixes = list(dynamic(bot, message))

    # Enhancement: also allow "mention + symbol" usage, e.g. "@Bot !say ..."
    # We build combinations of mention forms with each base prefix, with and without a space.
    mention_forms = commands.when_mentioned(bot, message)
    combined = []
    for m in mention_forms:
        for p in base_prefixes:
            if m.endswith(" "):
                # Typical discord.py mention prefixes include a trailing space
                combined.append(f"{m}{p}")  # "@Bot !"
            else:
                combined.append(f"{m} {p}")  # ensure a space between mention and symbol

    # Prepend combined forms so they match first when present
    return list(dict.fromkeys([*combined, *prefixes]))
