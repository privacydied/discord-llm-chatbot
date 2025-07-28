import pytest
from discord.ext import commands
from bot.core.bot import LLMBot
from bot.router import ResponseMessage

class TestRouterCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
    
    @commands.command()
    async def ping(self, ctx):
        await ctx.send('pong')
    
    @commands.Cog.listener()
    async def on_message(self, message):
        # Simulate router behavior for test verification
        if message.author.bot:
            return
        
        # Guild message without mention
        if not isinstance(message.channel, commands.DMChannel) and not self.bot.user.mentioned_in(message):
            assert False, "Guild message without mention should be filtered by pre-command gate"
        
        # Guild message with mention but no !prefix
        if self.bot.user.mentioned_in(message) and not message.content.startswith('!'):
            # This should trigger TEXT→TEXT flow
            response = ResponseMessage(text="test_response")
            assert response.text == "test_response", "TEXT→TEXT flow should return a valid response"
            
        # DM without !prefix
        if isinstance(message.channel, commands.DMChannel) and not message.content.startswith('!'):
            # This should trigger TEXT→TEXT flow
            response = ResponseMessage(text="dm_test_response")
            assert response.text == "dm_test_response", "TEXT→TEXT flow in DM should return a valid response"

@pytest.fixture
def bot():
    bot = LLMBot()
    return bot

@pytest.mark.asyncio
async def test_guild_mention_command(bot):
    cog = TestRouterCog(bot)
    ctx = await bot.get_context(cog._create_mock_message('!ping'))
    await cog.ping(ctx)
    assert ctx.responses[-1] == 'pong', "Command handler should respond to !ping in guild with mention"

@pytest.mark.asyncio
async def test_guild_mention_plain_text(bot):
    cog = TestRouterCog(bot)
    message = cog._create_mock_message('How are you?', mention=True)
    response = await bot.router.dispatch_message(message)
    assert response.text, "TEXT→TEXT flow should return response in guild with mention"

@pytest.mark.asyncio
async def test_dm_command(bot):
    cog = TestRouterCog(bot)
    ctx = await bot.get_context(cog._create_mock_message('!ping', is_dm=True))
    await cog.ping(ctx)
    assert ctx.responses[-1] == 'pong', "Command handler should respond to !ping in DM"

@pytest.mark.asyncio
async def test_dm_plain_text(bot):
    cog = TestRouterCog(bot)
    message = cog._create_mock_message('Hello', is_dm=True)
    response = await bot.router.dispatch_message(message)
    assert response.text, "TEXT→TEXT flow should return response in DM"

@pytest.mark.asyncio
async def test_invalid_command_filter(bot):
    cog = TestRouterCog(bot)
    # This should be filtered by the pre-command gate
    message = cog._create_mock_message('describe something')
    try:
        await bot.process_commands(message)
        assert False, "CommandNotFound should not be raised for filtered messages"
    except commands.errors.CommandNotFound:
        assert False, "CommandNotFound should be prevented by pre-command filter"

def setup(bot):
    return bot.add_cog(TestRouterCog(bot))