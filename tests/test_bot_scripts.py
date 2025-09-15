"""
Minimal test bot to verify basic Discord bot functionality.
"""

import os
import sys
import logging
import discord
from discord.ext import commands
from dotenv import load_dotenv

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("test_bot")

# Load environment variables
load_dotenv()


def _running_under_pytest() -> bool:
    """Return True when the module is imported by pytest collection.

    Pytest sets the ``PYTEST_CURRENT_TEST`` environment variable while it is
    importing test modules.  The original script called ``sys.exit`` at import
    time when the ``DISCORD_TOKEN`` environment variable was missing which made
    the entire test run abort before collection completed.  By detecting the
    pytest runtime we can keep the helpful error message for manual execution
    while allowing the module to be imported safely during automated tests.
    """

    return bool(os.getenv("PYTEST_CURRENT_TEST")) or "pytest" in sys.modules


TOKEN = os.getenv("DISCORD_TOKEN")

if not TOKEN:
    logger.error("DISCORD_TOKEN not found in .env file")
    if not _running_under_pytest():
        sys.exit(1)

# Set up intents
intents = discord.Intents.default()
intents.message_content = True
intents.messages = True

# Create bot instance with command prefix
bot = commands.Bot(command_prefix="!", intents=intents)


@bot.event
async def on_ready():
    """Event triggered when the bot is ready."""
    logger.info(f"Logged in as {bot.user.name} (ID: {bot.user.id})")
    logger.info(f"Bot is in {len(bot.guilds)} guilds:")
    for guild in bot.guilds:
        logger.info(f" - {guild.name} (ID: {guild.id})")

    # Set presence
    try:
        await bot.change_presence(
            activity=discord.Activity(type=discord.ActivityType.listening, name="!ping")
        )
        logger.info('Bot presence set to "Listening to !ping"')
    except Exception as e:
        logger.error(f"Error setting presence: {e}")


@bot.command()
async def ping(ctx):
    """Simple ping command to test if the bot is responding."""
    logger.info(
        f"Ping command received from {ctx.author} in {ctx.guild.name if ctx.guild else 'DM'}"
    )
    try:
        await ctx.send("Pong! 🏓")
        logger.info("Successfully sent pong response")
    except Exception as e:
        logger.error(f"Error sending pong: {e}")


@bot.event
async def on_message(message):
    """Event triggered when a message is received."""
    # Don't respond to ourselves
    if message.author == bot.user:
        return

    logger.debug(
        f"Message from {message.author} in {message.guild.name if message.guild else 'DM'}: {message.content}"
    )

    # Process commands
    try:
        await bot.process_commands(message)
    except Exception as e:
        logger.error(f"Error processing command: {e}")


if __name__ == "__main__":
    if not TOKEN:
        print("Error: DISCORD_TOKEN not found in .env file")
    else:
        print("Starting test bot...")
        bot.run(TOKEN)
