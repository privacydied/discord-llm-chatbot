from discord.ext import commands


class PingCog(commands.Cog):
    """A simple cog for testing command registration."""

    def __init__(self, bot) -> None:
        self.bot = bot

    @commands.command()
    async def ping(self, ctx) -> None:
        await ctx.send("pong")


async def setup(bot) -> None:
    """The setup function for the PingCog cog."""
    await bot.add_cog(PingCog(bot))
