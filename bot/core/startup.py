"""
Contains bot startup and pre-flight check logic.
"""
import discord
from discord.ext import commands
import hashlib
import os
import subprocess
from pathlib import Path
from typing import Optional

from bot.config import ConfigurationError
from bot.util.logging import get_logger

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

def check_playwright_browsers(logger) -> None:
    """Checks for Playwright browsers and attempts to install them if missing."""
    logger.info("Checking for Playwright browser binaries...")
    
    if _get_playwright_chromium_path():
        logger.info("✅ Playwright Browser (Chromium) is installed.")
        return

    logger.warning("❌ Playwright Browser (Chromium) not found. Attempting auto-installation...")
    
    try:
        result = subprocess.run(
            ["uv", "run", "playwright", "install", "chromium"],
            capture_output=True,
            text=True,
            check=True,
            timeout=300
        )
        logger.info(f"Playwright auto-install successful: {result.stdout}")
        if _get_playwright_chromium_path():
            logger.info("✅ Successfully installed Playwright Browser (Chromium).")
        else:
            logger.error("📛 Failed to install Playwright Browser automatically. Please run `uv run playwright install chromium` manually.")

    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.error(f"📛 Failed to auto-install Playwright browser: {e.stderr or e}")
    except FileNotFoundError:
        logger.error("📛 `uv` or `playwright` command not found. Cannot auto-install browsers.")

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
            logger.error(f"Required intent '{intent_name}' is disabled.")
    
    if all_intents_ok:
        logger.info("[WIND][INIT] Intents verified")
    else:
        logger.critical("One or more required intents are missing. Bot may not function correctly.")
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
    intents.voice_states = True # Required for TTS
    return intents

def get_prefix(bot, message: discord.Message) -> list[str]:
    """Dynamically get the command prefix for the bot."""
    base_prefixes = os.getenv('BOT_PREFIX', '!').split(',')
    return commands.when_mentioned_or(*base_prefixes)(bot, message)
