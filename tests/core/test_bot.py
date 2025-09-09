import pytest
from unittest.mock import MagicMock

import discord
from discord.ext import commands

# Import the cog class directly for testing
from tests.core.cogs.test_cog import PingCog


@pytest.mark.asyncio
async def test_command_registration_direct_add():
    """
    Verify that a cog's commands are correctly registered when added directly
    via bot.add_cog(). This bypasses the extension loading mechanism to isolate
    the issue.
    """
    # Initialize a standard bot with minimal setup for testing
    intents = discord.Intents.default()
    intents.message_content = True
    bot = commands.Bot(command_prefix='!', intents=intents)

    # Mock essential bot attributes that would be set up by bot.run()
    user_mock = MagicMock(spec=discord.ClientUser)
    user_mock.id = 12345
    bot._connection._user = user_mock
    bot._connection.http = MagicMock()

    # Add the cog directly to the bot
    await bot.add_cog(PingCog(bot))

    # After the hook, check the bot's own command registry
    registered_commands = bot.commands
    command_names = {cmd.name for cmd in registered_commands}

    # Assert that our test command and the default help command were registered
    assert 'ping' in command_names
    assert 'help' in command_names
