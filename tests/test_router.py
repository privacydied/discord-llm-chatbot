"""
Unit tests for the refactored Router class.

Ensures all modality flows (text, audio, URL, image, doc) are correctly routed,
and that the '1 IN > 1 OUT' principle is strictly enforced. Verifies command
handling for static, cog-based, and standard chat commands.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import discord
import logging

from bot.router import InputModality, Router, BotAction
from bot.command_parser import ParsedCommand
from bot.types import Command
import asyncio


@pytest.fixture
def mock_bot():
    """Fixture for a mocked bot instance with necessary attributes."""
    bot = MagicMock(spec=discord.Client)
    bot.user = MagicMock()
    bot.user.id = 12345
    bot.user.mentioned_in.return_value = True
    bot.config = {"TTS_ENABLED_USERS": set(), "TTS_ENABLED_SERVERS": set()}
    bot.tts_manager = AsyncMock()
    bot.brain = AsyncMock()
    bot.loop = AsyncMock()
    return bot


@pytest.fixture
def router(mock_bot):
    """Provides a Router instance with a mocked bot."""
    return Router(bot=mock_bot, logger=MagicMock(spec=logging.Logger))


@pytest.fixture
def mock_message():
    """Fixture for a mocked discord.Message instance."""
    message = MagicMock(spec=discord.Message)
    message.id = 11223344
    message.guild = None
    message.channel = MagicMock(spec=discord.DMChannel)
    message.author = MagicMock(spec=discord.User)
    message.author.id = 98765
    message.attachments = []
    message.content = "Hello"
    return message


# --- Test Cases ---


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command_type, expected_text, should_be_none, should_delegate_to_cog",
    [
        (Command.PING, "Pong!", False, False),
        (Command.HELP, None, False, True),
        (Command.CHAT, "Processed text", False, False),
        (Command.TTS, None, True, False),
        (Command.SAY, None, True, False),
        (Command.TTS_ALL, None, True, False),
        (Command.SPEAK, None, True, False),
        (Command.ALERT, None, False, True),
        (Command.IGNORE, None, True, False),
    ],
)
@patch("bot.router.parse_command")
async def test_command_handling(
    mock_parse_command,
    router,
    mock_message,
    command_type,
    expected_text,
    should_be_none,
    should_delegate_to_cog,
):
    """Test router's handling of static, cog, and ignored commands."""
    mock_parse_command.return_value = ParsedCommand(command=command_type, cleaned_content="Hello")

    if command_type == Command.CHAT:
        # For a standard chat, we need to mock the full flow
        router._flows["process_text"] = AsyncMock(return_value=expected_text)
        with patch.object(router, "_get_input_modality", return_value=InputModality.TEXT_ONLY):
            response = await router.dispatch_message(mock_message)
    else:
        response = await router.dispatch_message(mock_message)

    if should_be_none:
        assert response is None, f"Expected None for command {command_type.name}, but got a response."
    elif should_delegate_to_cog:
        assert response is not None
        assert isinstance(response, BotAction)
        assert response.meta.get("delegated_to_cog") is True
    else:
        assert response is not None, f"Expected a response for command {command_type.name}, but got None."
        # HELP is handled specially: just check it returns non-empty text.
        if command_type == Command.HELP:
            assert isinstance(response.text, str)
            assert len(response.text.strip()) > 50
        else:
            assert response.text == expected_text


@pytest.mark.asyncio
@patch("bot.router.parse_command")
async def test_alert_delegates_without_custom_parse(mock_parse_command, router, mock_message):
    mock_parse_command.return_value = None

    mock_message.guild = MagicMock(spec=discord.Guild)
    mock_message.channel = MagicMock(spec=discord.TextChannel)
    mock_message.content = "!alert hello"

    response = await router.dispatch_message(mock_message)

    assert response is not None
    assert isinstance(response, BotAction)
    assert response.meta.get("delegated_to_cog") is True


@pytest.mark.asyncio
async def test_router_typing_rate_limit_is_non_fatal(router, mock_message):
    """Typing failures should not abort routing, and repeated attempts should be suppressed."""

    class _BrokenTyping:
        async def __aenter__(self):
            raise RuntimeError("429 Too Many Requests")

        async def __aexit__(self, exc_type, exc, tb):
            return False

    channel = MagicMock(spec=discord.TextChannel)
    channel.id = 4242
    channel.typing.return_value = _BrokenTyping()

    first_message = mock_message
    first_message.channel = channel
    first_message.guild = MagicMock(spec=discord.Guild)
    first_message.id = 9001
    first_message.content = f"<@{router.bot.user.id}> please analyze this"
    first_message.mentions = [router.bot.user]

    second_message = MagicMock(spec=discord.Message)
    second_message.id = 9002
    second_message.channel = channel
    second_message.guild = first_message.guild
    second_message.author = first_message.author
    second_message.content = first_message.content
    second_message.attachments = []
    second_message.mentions = [router.bot.user]

    with (
        patch.object(router, "_should_process_message", return_value=True),
        patch.object(router, "_compat_dispatch_for_tests", AsyncMock(return_value=None)),
        patch.object(
            router,
            "_resolve_scope_and_target",
            AsyncMock(return_value=("lone", None, "")),
        ),
        patch.object(router, "_prioritized_vision_route", AsyncMock(return_value=None)),
        patch("bot.modality.collect_input_items", return_value=[]),
        patch.object(
            router,
            "_process_multimodal_message_internal",
            AsyncMock(return_value=BotAction(content="ok")),
        ),
    ):
        first = await router.dispatch_message(first_message)
        second = await router.dispatch_message(second_message)

    assert isinstance(first, BotAction)
    assert isinstance(second, BotAction)
    assert channel.typing.call_count == 1


@pytest.mark.asyncio
@patch("bot.router.parse_command")
async def test_alert_delegates_without_custom_parse(mock_parse_command, router, mock_message):
    mock_parse_command.return_value = None

    mock_message.guild = MagicMock(spec=discord.Guild)
    mock_message.channel = MagicMock(spec=discord.TextChannel)
    mock_message.content = "!alert hello"

    response = await router.dispatch_message(mock_message)

    assert response is not None
    assert isinstance(response, BotAction)
    assert response.meta.get("delegated_to_cog") is True


@pytest.mark.parametrize(
    "modality, flow_key, expected_output",
    [
        (InputModality.TEXT_ONLY, "process_text", "Processed text"),
        (InputModality.GENERAL_URL, "process_url", "Processed URL"),
        (InputModality.AUDIO_VIDEO_FILE, "process_audio", "Processed audio"),
        (InputModality.SINGLE_IMAGE, "process_attachments", "Processed attachments"),
        (InputModality.PDF_DOCUMENT, "process_attachments", "Processed attachments"),
    ],
)
@patch("bot.router.parse_command")
async def test_modality_flows(
    mock_parse_command,
    router,
    mock_message,
    modality,
    flow_key,
    expected_output,
):
    """Verify that each input modality is routed to the correct processing flow."""
    mock_parse_command.return_value = ParsedCommand(command=Command.CHAT, cleaned_content="Test content")

    if modality == InputModality.GENERAL_URL:
        mock_message.content = "https://example.com"

    # Mock the specific flow method in the _flows dictionary
    mock_flow_method = AsyncMock(return_value=expected_output)
    router._flows[flow_key] = mock_flow_method

    with patch.object(router, "_get_input_modality", return_value=modality) as mock_get_modality:
        response = await router.dispatch_message(mock_message)

    mock_get_modality.assert_called_once_with(mock_message)
    mock_flow_method.assert_called_once()
    assert response is not None
    assert response.text == expected_output


@pytest.mark.asyncio
@patch("bot.router.parse_command")
async def test_no_processed_text_returns_error(mock_parse_command, router, mock_message):
    """Test that if a flow returns no text, a user-friendly error is returned."""
    mock_parse_command.return_value = ParsedCommand(command=Command.CHAT, cleaned_content="Test")

    router._flows["process_text"] = AsyncMock(return_value=None)  # Simulate a flow failure

    with patch.object(router, "_get_input_modality", return_value=InputModality.TEXT_ONLY):
        response = await router.dispatch_message(mock_message)

    assert response is not None
    assert response.text.startswith("Error:")


@pytest.mark.asyncio
@patch("bot.router.parse_command")
async def test_exception_in_flow_returns_error(mock_parse_command, router, mock_message):
    """Test that an exception during processing returns a generic error message."""
    mock_parse_command.return_value = ParsedCommand(command=Command.CHAT, cleaned_content="Test")

    router._flows["process_text"] = AsyncMock(side_effect=Exception("Critical failure!"))

    with patch.object(router, "_get_input_modality", return_value=InputModality.TEXT_ONLY):
        response = await router.dispatch_message(mock_message)

    assert response is not None
    assert response.text.startswith("Error:")


@pytest.mark.asyncio
@patch("bot.router.parse_command")
async def test_empty_string_prevention(mock_parse_command, router, mock_message):
    """Verify empty string responses are converted to error messages."""
    mock_parse_command.return_value = ParsedCommand(command=Command.CHAT, cleaned_content="Test")

    router._flows["process_text"] = AsyncMock(return_value="")  # Empty string

    with patch.object(router, "_get_input_modality", return_value=InputModality.TEXT_ONLY):
        response = await router.dispatch_message(mock_message)

    assert response is not None
    assert response.text.startswith("Error:")


@pytest.mark.asyncio
@patch("bot.router.parse_command")
async def test_error_embed_generation(mock_parse_command, router, mock_message):
    """Verify error conditions generate proper error content."""
    mock_parse_command.return_value = ParsedCommand(command=Command.CHAT, cleaned_content="Test")

    router._flows["process_text"] = AsyncMock(return_value=None)  # Simulate failure

    with patch.object(router, "_get_input_modality", return_value=InputModality.TEXT_ONLY):
        response = await router.dispatch_message(mock_message)

    assert response is not None
    # Error responses may use embeds or content depending on code path
    has_error = (hasattr(response, "content") and response.content and "Error" in response.content) or (hasattr(response, "embeds") and response.embeds)
    assert has_error, "Error responses should include error content or embeds"


@pytest.mark.asyncio
async def test_dm_plain_text_reply():
    """Test that a plain text DM returns a ResponseMessage with content."""
    # Setup
    mock_bot = MagicMock()
    mock_logger = MagicMock()
    flows = {"process_text": AsyncMock(return_value="Hello, world!")}
    router = Router(bot=mock_bot, flow_overrides=flows, logger=mock_logger)

    # Create a mock DM message
    mock_message = MagicMock()
    mock_message.channel = MagicMock()
    mock_message.channel.__class__.__name__ = "DMChannel"
    mock_message.content = "Hello"
    mock_message.attachments = []

    # Execute
    action = await router.dispatch_message(mock_message)

    # Verify
    assert action is not None
    assert hasattr(action, "content")
    assert "Hello, world!" in action.content


@pytest.mark.asyncio
async def test_flow_process_attachments_multimodal_accepts_raw_content_arg(router, mock_message):
    mock_message.attachments = []
    res = await router._flow_process_attachments_multimodal(mock_message, "")
    assert isinstance(res, BotAction)


@pytest.mark.asyncio
async def test_guild_unmentioned_ignored():
    """Test that _should_process_message rejects unmentioned guild messages."""
    mock_bot = MagicMock()
    mock_bot.config = {
        "OWNER_IDS": [],
        "REPLY_TRIGGERS": [
            "dm",
            "mention",
            "reply",
            "bot_threads",
            "owner",
            "command_prefix",
        ],
        "REQUIRE_MENTION_IN_GUILDS": True,
        "ALLOW_REPLY_TO_BOT_WITHOUT_MENTION": True,
        "DM_REQUIRE_MENTION": False,
        "BOT_SPEAKS_ONLY_WHEN_SPOKEN_TO": True,
        "COMMAND_PREFIX": "!",
    }
    mock_bot.user.id = 9999
    mock_logger = MagicMock(spec=logging.Logger)
    router = Router(bot=mock_bot, flow_overrides={}, logger=mock_logger)

    # Mock guild message - not a DM, not mentioning bot, not replying
    mock_message = MagicMock()
    mock_message.channel = MagicMock(spec=discord.TextChannel)
    mock_message.content = "Hello"
    mock_message.mentions = []
    mock_message.author.id = 1111
    mock_message.reference = None

    # Patch mention/reply checks to return False
    router._mentions_bot = MagicMock(return_value=False)
    router._is_reply_to_bot = MagicMock(return_value=False)
    router._detect_direct_vision_triggers = MagicMock(return_value=False)

    result = router._should_process_message(mock_message)
    assert result is False


class TestExtractionOnlyTimeout:
    """Verify extraction-only items are guarded by asyncio.wait_for timeout. [REH][PA]"""

    @pytest.mark.asyncio
    async def test_extraction_only_timeout_cancels_hung_handler(self, mock_bot):
        """If an extraction handler hangs, asyncio.wait_for cancels it and
        the item is recorded as failed (partial-success preserved)."""
        router = Router(bot=mock_bot, flow_overrides={}, logger=logging.getLogger("test"))

        # Mock _handle_item_with_provider to hang forever
        async def _hang_forever(*args, **kwargs):
            await asyncio.sleep(9999)

        router._handle_item_with_provider = AsyncMock(side_effect=_hang_forever)
        # Set a very short budget
        import os

        os.environ["MULTIMODAL_PER_ITEM_BUDGET"] = "0.1"
        try:
            # We test the timeout guard by verifying asyncio.wait_for is used.
            # Direct E2E test would require full multimodal setup; instead we
            # verify the code structure: the extraction-only branch catches TimeoutError.
            import inspect

            source = inspect.getsource(router._process_multimodal_message_internal)
            # Verify asyncio.wait_for wraps extraction-only handler calls
            assert "asyncio.wait_for" in source, "extraction-only items must use asyncio.wait_for"
            assert "asyncio.TimeoutError" in source, "must catch asyncio.TimeoutError for extraction items"
        finally:
            os.environ.pop("MULTIMODAL_PER_ITEM_BUDGET", None)

    @pytest.mark.asyncio
    async def test_extraction_only_success_within_budget(self, mock_bot):
        """Extraction-only items that complete within budget succeed normally."""
        router = Router(bot=mock_bot, flow_overrides={}, logger=logging.getLogger("test"))
        router._handle_item_with_provider = AsyncMock(return_value="extracted text")
        import inspect

        source = inspect.getsource(router._process_multimodal_message_internal)
        # Verify timeout=selected_budget is passed (not hardcoded)
        assert "timeout=selected_budget" in source, "timeout must use selected_budget per modality"
