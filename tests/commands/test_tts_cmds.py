import pytest
from unittest.mock import AsyncMock, MagicMock
from types import SimpleNamespace

from bot.commands.tts_cmds import TTSCommands
from bot.tts.state import tts_state


@pytest.fixture
def mock_bot():
    """Fixture for a mocked bot object."""
    bot = MagicMock()
    bot.router = AsyncMock()
    bot.tts_manager = MagicMock()
    bot.add_cog = AsyncMock()
    return bot


@pytest.fixture
def tts_cog(mock_bot):
    """Fixture for the TTSCommands cog."""
    return TTSCommands(mock_bot)


@pytest.fixture
def mock_ctx():
    """Fixture for a mocked command context."""
    ctx = MagicMock()
    ctx.author = MagicMock()
    ctx.author.id = 12345
    ctx.send = AsyncMock()
    ctx.message = MagicMock()
    ctx.message.attachments = []
    return ctx


@pytest.fixture(autouse=True)
def reset_tts_state():
    """Reset the tts_state before each test to ensure isolation."""
    tts_state.user_preferences.clear()
    tts_state.global_enabled = False
    tts_state.one_time_tts.clear()


@pytest.mark.asyncio
async def test_tts_on(tts_cog, mock_ctx):
    """Verify that the 'tts on' command enables TTS for a user."""
    await tts_cog.tts_on.callback(tts_cog, mock_ctx)

    assert tts_state.user_preferences.get(mock_ctx.author.id) is True
    mock_ctx.send.assert_called_once_with("✅ TTS responses enabled for you.")


@pytest.mark.asyncio
async def test_tts_off(tts_cog, mock_ctx):
    """Verify that the 'tts off' command disables TTS for a user."""
    # First, enable it to ensure the test correctly changes the state
    tts_state.set_user_tts(mock_ctx.author.id, True)
    assert tts_state.user_preferences.get(mock_ctx.author.id) is True

    await tts_cog.tts_off.callback(tts_cog, mock_ctx)

    assert tts_state.user_preferences.get(mock_ctx.author.id) is False
    mock_ctx.send.assert_called_once_with("✅ TTS responses disabled for you.")


@pytest.mark.asyncio
async def test_tts_all_on_by_admin(tts_cog, mock_ctx):
    """Verify that an admin can enable global TTS."""
    # Simulate admin permissions
    mock_ctx.author.guild_permissions.administrator = True
    await tts_cog.tts_all.callback(tts_cog, mock_ctx, setting='on')

    assert tts_state.global_enabled is True
    mock_ctx.send.assert_called_once_with("✅ TTS responses enabled globally.")


@pytest.mark.asyncio
async def test_tts_all_off_by_admin(tts_cog, mock_ctx):
    """Verify that an admin can disable global TTS."""
    tts_state.set_global_tts(True)
    mock_ctx.author.guild_permissions.administrator = True
    await tts_cog.tts_all.callback(tts_cog, mock_ctx, setting='off')

    assert tts_state.global_enabled is False
    mock_ctx.send.assert_called_once_with("✅ TTS responses disabled globally.")


@pytest.mark.asyncio
async def test_tts_all_invalid_setting(tts_cog, mock_ctx):
    """Verify that an invalid setting for 'tts all' is handled correctly."""
    mock_ctx.author.guild_permissions.administrator = True
    await tts_cog.tts_all.callback(tts_cog, mock_ctx, setting='invalid')

    assert tts_state.global_enabled is False
    mock_ctx.send.assert_called_once_with("❌ Invalid setting. Use 'on' or 'off'.")


@pytest.mark.asyncio
async def test_speak_command_no_text(tts_cog, mock_ctx):
    """Verify that '!speak' without text sets a one-time TTS flag."""
    await tts_cog.speak.callback(tts_cog, mock_ctx)

    assert tts_state.get_and_clear_one_time_tts(mock_ctx.author.id) is True
    mock_ctx.send.assert_called_once_with("🗯️ The next response will be spoken.")


@pytest.mark.asyncio
async def test_speak_command_with_text(tts_cog, mock_ctx):
    """Verify that '!speak <text>' delegates to the 'say' command."""
    tts_cog.say = AsyncMock()
    text_to_speak = "Hello world"

    await tts_cog.speak.callback(tts_cog, mock_ctx, text=text_to_speak)

    tts_cog.say.assert_called_once_with(mock_ctx, text=text_to_speak)


@pytest.mark.asyncio
async def test_say_command_with_text(tts_cog, mock_ctx, mock_bot, monkeypatch):
    """Verify that '!say <text>' generates a TTS response directly via TTSManager.process."""
    # Mock discord.File to prevent FileNotFoundError in a test environment
    mock_discord_file = MagicMock()
    monkeypatch.setattr('discord.File', mock_discord_file)

    mock_bot.tts_manager.is_available.return_value = True
    # TTSManager.process should return an object with audio_path
    mock_bot.tts_manager.process = AsyncMock(return_value=SimpleNamespace(audio_path="/fake/path/audio.wav"))
    text_to_say = "This is a direct command."

    await tts_cog.say.callback(tts_cog, mock_ctx, text=text_to_say)

    mock_bot.tts_manager.process.assert_called_once()
    mock_ctx.send.assert_called_once()
    # Check that discord.File was called with the correct path
    mock_discord_file.assert_called_once_with("/fake/path/audio.wav")
    # Check that the file object was sent
    assert 'file' in mock_ctx.send.call_args.kwargs

@pytest.mark.asyncio
async def test_say_command_timeout_meta_forwarding(tts_cog, mock_ctx, mock_bot, monkeypatch):
    """Verify that '!say' forwards timeout meta to TTSManager.process."""
    mock_discord_file = MagicMock()
    monkeypatch.setattr('discord.File', mock_discord_file)

    mock_bot.tts_manager.is_available.return_value = True
    mock_bot.tts_manager.process = AsyncMock(return_value=SimpleNamespace(audio_path="/fake/path/audio.wav"))

    await tts_cog.say.callback(
        tts_cog,
        mock_ctx,
        text="hello",
        timeout_s=3.3,
        cold=True,
        timeout_cold_s=9.9,
        timeout_warm_s=1.1,
    )

    # Inspect BotAction passed to process
    assert mock_bot.tts_manager.process.call_count == 1
    call_args, call_kwargs = mock_bot.tts_manager.process.call_args
    assert len(call_args) == 1
    action = call_args[0]
    meta = getattr(action, 'meta', {})
    assert pytest.approx(meta.get('tts_timeout_s'), rel=1e-6) == 3.3
    assert meta.get('tts_cold') is True
    assert pytest.approx(meta.get('tts_timeout_cold_s'), rel=1e-6) == 9.9
    assert pytest.approx(meta.get('tts_timeout_warm_s'), rel=1e-6) == 1.1


@pytest.mark.asyncio
async def test_say_command_with_attachment(tts_cog, mock_ctx, mock_bot):
    """Verify that '!say' with an attachment dispatches to the router."""
    mock_ctx.message.attachments = [MagicMock()]

    await tts_cog.say.callback(tts_cog, mock_ctx, text="some text")

    # Verify one-time TTS is set and router is called
    assert tts_state.get_and_clear_one_time_tts(mock_ctx.author.id) is True
    mock_bot.router.dispatch_message.assert_called_once_with(mock_ctx.message)


@pytest.mark.asyncio
async def test_tts_group_with_text(tts_cog, mock_ctx):
    """Verify that '!tts <text>' delegates to the 'speak' command."""
    tts_cog.speak = AsyncMock()
    text_to_speak = "Speak this"

    await tts_cog.tts_group.callback(tts_cog, mock_ctx, text=text_to_speak)

    tts_cog.speak.assert_called_once_with(mock_ctx, text_to_speak)


@pytest.mark.asyncio
async def test_tts_group_no_text(tts_cog, mock_ctx):
    """Verify that '!tts' without text sends a help message."""
    await tts_cog.tts_group.callback(tts_cog, mock_ctx, text=None)

    mock_ctx.send.assert_called_once_with("Please specify 'on', 'off', or text to speak.")
