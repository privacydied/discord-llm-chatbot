from discord.ext import commands


class PingCog(commands.Cog):
    """A simple cog for testing command registration."""

    def __init__(self, bot):
        self.bot = bot
        print("✅ TestCommands cog initialized")

    @commands.command()
    async def ping(self, ctx):
        await ctx.send("pong")


async def setup(bot):
    """The setup function for the PingCog cog."""
    await bot.add_cog(PingCog(bot))
