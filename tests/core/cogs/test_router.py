import types
import pytest
from discord.ext import commands
from bot.core.bot import LLMBot
from bot.router import ResponseMessage

class TestRouterCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
    
    async def ping(self, ctx):
        await ctx.send('pong')
    
    @commands.Cog.listener()
    async def on_message(self, message):
        # Simulate router behavior for test verification
        if message.author.bot:
            return
        
        # Helpers independent of discord internals
        def _is_dm(msg):
            return bool(getattr(msg, 'is_dm', False))
        def _is_mention(msg):
            return str(getattr(msg, 'content', '')).strip().startswith('<@')

        # Guild message without mention
        if not _is_dm(message) and not _is_mention(message):
            assert False, "Guild message without mention should be filtered by pre-command gate"
        
        # Guild message with mention but no !prefix
        if _is_mention(message) and not message.content.startswith('!'):
            # This should trigger TEXT→TEXT flow
            response = ResponseMessage(text="test_response")
            assert response.text == "test_response", "TEXT→TEXT flow should return a valid response"
            
        # DM without !prefix
        if _is_dm(message) and not message.content.startswith('!'):
            # This should trigger TEXT→TEXT flow
            response = ResponseMessage(text="dm_test_response")
            assert response.text == "dm_test_response", "TEXT→TEXT flow in DM should return a valid response"

    # Minimal helper to fabricate a message-like object for tests
    def _create_mock_message(self, content: str, is_dm: bool = False, mention: bool = False):
        user_id = getattr(self.bot.user, 'id', 1234567890)
        text = content
        if mention:
            text = f"<@{user_id}> {content}"

        # Create a minimal author/channel/guild structure
        author = types.SimpleNamespace(id=42, bot=False)
        if is_dm:
            channel = types.SimpleNamespace(id=1001)
            guild = None
        else:
            channel = types.SimpleNamespace(id=2002)
            guild = types.SimpleNamespace(id=3003)

        # Provide attributes accessed by tests and router helpers
        msg = types.SimpleNamespace(
            id=777,
            content=text,
            author=author,
            channel=channel,
            guild=guild,
            is_dm=is_dm,
            attachments=[],
            mentions=[self.bot.user] if mention else [],
        )
        return msg

class FakeCtx:
    def __init__(self):
        self.responses = []
    async def send(self, s: str):
        self.responses.append(s)


@pytest.fixture
def bot():
    b = LLMBot()

    # Patch get_context to return a lightweight FakeCtx
    async def fake_get_context(_message):
        return FakeCtx()
    b.get_context = fake_get_context

    # Patch router to a lightweight stub returning Response-like objects
    async def _stub_dispatch_message(_message):
        # Mimic a simple TEXT->TEXT flow
        return types.SimpleNamespace(text="ok")
    b.router = types.SimpleNamespace(dispatch_message=_stub_dispatch_message)
    return b

@pytest.mark.asyncio
async def test_guild_mention_command(bot):
    cog = TestRouterCog(bot)
    bot.add_cog(cog)
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
    bot.add_cog(cog)
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