import pytest
from unittest.mock import AsyncMock, MagicMock

from bot.commands.screenshot_commands import ScreenshotCommands


@pytest.fixture
def mock_bot():
    bot = MagicMock()
    bot.router = MagicMock()
    bot.router._handle_screenshot_url = AsyncMock(return_value="Test analysis result")
    return bot


@pytest.fixture
def mock_ctx(mock_bot):
    ctx = MagicMock()
    ctx.bot = mock_bot
    ctx.message = MagicMock()
    ctx.message.content = ""

    # reply() returns a message we can edit
    processing_msg = MagicMock()
    processing_msg.edit = AsyncMock()
    ctx.reply = AsyncMock(return_value=processing_msg)

    # typing() async context manager
    class _Typing:
        async def __aenter__(self):
            return None
        async def __aexit__(self, exc_type, exc, tb):
            return False
    ctx.typing = MagicMock(return_value=_Typing())
    return ctx


@pytest.mark.asyncio
async def test_ss_missing_url_prompts_usage(mock_bot, mock_ctx):
    cog = ScreenshotCommands(mock_bot)
    mock_ctx.message.content = "no url here"

    await cog.screenshot_cmd.callback(cog, mock_ctx, url=None)

    # Should prompt for usage when URL missing
    assert mock_ctx.reply.await_count == 1
    assert "Please provide a valid URL" in mock_ctx.reply.await_args.kwargs["content"]


@pytest.mark.asyncio
async def test_ss_valid_url_delegates_to_router_and_edits_embed(mock_bot, mock_ctx):
    cog = ScreenshotCommands(mock_bot)
    url = "https://example.com"

    await cog.screenshot_cmd.callback(cog, mock_ctx, url=url)

    # Should have called router._handle_screenshot_url with InputItem
    assert mock_bot.router._handle_screenshot_url.await_count == 1
    arg = mock_bot.router._handle_screenshot_url.await_args.args[0]
    assert getattr(arg, "source_type", None) == "url"
    assert getattr(arg, "payload", None) == url

    # Should have edited the processing message with an embed
    processing_msg = await mock_ctx.reply.await_args.kwargs.get("return_value")
    # We used MagicMock above; access the original return
    assert mock_ctx.reply.await_count >= 1
    mock_msg = mock_ctx.reply.return_value
    mock_msg.edit.assert_awaited()
    kwargs = mock_msg.edit.await_args.kwargs
    assert "embed" in kwargs and kwargs["embed"] is not None
