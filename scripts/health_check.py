# scripts/health_check.py
"""
Health Check Script

This script runs a series of ordered imports to verify that the application's
major components can be initialized without circular dependency errors.
A successful run of this script indicates that the core application is stable.
"""

import os
import sys

# Ensure the bot's root directory is in the Python path
# This allows the script to be run from the project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def run_health_check():
    """Imports and instantiates core components to check for errors."""
    print("--- Running Health Check ---")
    
    try:
        print("1. Initializing logging...")
        from bot.utils.logging import init_logging, get_logger
        init_logging()
        logger = get_logger(__name__)
        logger.info("Logging initialized successfully.")

        print("2. Importing core BotAction...")
        from bot.router import BotAction
        action = BotAction(content="test")
        assert action.content == "test"
        logger.info("BotAction imported and instantiated successfully.")

        print("3. Importing and instantiating LLMBot...")
        # This is the main test. If this works, all major components
        # (Router, TTSManager, Metrics, etc.) can be initialized.
        from bot.core.bot import LLMBot
        # We don't need to run the bot, just instantiate it.
        # We pass dummy values because we are not connecting to Discord.
        LLMBot(command_prefix="!", intents=None)
        logger.info("LLMBot instantiated successfully.")

        print("\n\033[92m✔ HEALTH CHECK PASSED\033[0m")
        print("All major components imported and initialized without errors.")

    except Exception as e:
        print("\n\033[91m✖ HEALTH CHECK FAILED\033[0m")
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_health_check()
