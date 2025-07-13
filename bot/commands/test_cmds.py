"""
Test commands for debugging and basic functionality verification.
"""
import discord
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

def setup(bot):
    """Add the test commands to the bot."""
    print("🔄 Setting up test commands...")
    cog = TestCommands(bot)
    bot.add_cog(cog)
    print("✅ Test commands set up")
    return cog
