"""
Test commands for debugging and basic functionality verification.
"""
from discord.ext import commands
import logging

logger = logging.getLogger(__name__)

class TestCommands(commands.Cog):
    """Simple test commands for debugging."""
    
    def __init__(self, bot):
        self.bot = bot
        self.config = {}
        self.router = None
        print("✅ TestCommands cog initialized")
    
    @commands.command(name="ping")
    async def ping(self, ctx):
        """Respond with 'Pong!' to verify the bot is working."""
        print(f"🏓 Ping command received from {ctx.author}")
        await ctx.send("Pong! 🏓")

async def setup(bot):
    """Add the test commands to the bot."""
    logger.info("Setting up test commands...")
    if not bot.get_cog('TestCommands'):
        cog = TestCommands(bot)
        await bot.add_cog(cog)
    else:
        logger.warning("'TestCommands' cog already loaded, skipping setup.")
    logger.info("Test commands set up")
