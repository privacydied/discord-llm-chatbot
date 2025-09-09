from discord.ext import commands


class Bot(commands.Bot):
    """A subclass of discord.ext.commands.Bot with a placeholder for the app router."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.router = None  # Placeholder for the router instance
