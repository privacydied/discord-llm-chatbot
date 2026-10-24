"""send_voice_note tool: the model can answer with a voice note on request. [CA][REH]"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from bot.action import BotAction
from bot.tools import ToolContext, get_registry
from bot.tools.builtins.voice_note import send_voice_note
from bot.tts.state import TTSState, apply_tts_preference, tts_state


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    monkeypatch.delenv("TTS_GLOBAL_DEFAULT", raising=False)
    tts_state._voice_reply_messages.clear()
    yield
    tts_state._voice_reply_messages.clear()


def _ctx(message_id=111, tts_ok=True, guild=None):
    tts = SimpleNamespace(is_available=lambda: tts_ok)
    message = SimpleNamespace(id=message_id, author=SimpleNamespace(id=5), guild=SimpleNamespace(id=guild) if guild else None)
    return ToolContext(message=message, bot=SimpleNamespace(tts_manager=tts)), message


def test_registered_and_allowlisted():
    assert "send_voice_note" in {s["function"]["name"] for s in get_registry().schemas()}


async def test_tool_makes_only_that_reply_spoken():
    ctx, message = _ctx()
    result = await send_voice_note(ctx, {})
    assert result.ok and "spoken" in result.content
    action = BotAction(content="yo what's good, this is ronin")
    assert apply_tts_preference(action, message) is True
    assert action.meta["requires_tts"] is True
    # consumed: the next reply to the same message, or any other message, stays text
    assert apply_tts_preference(BotAction(content="next"), message) is False
    other = SimpleNamespace(id=222, author=SimpleNamespace(id=5), guild=None)
    assert apply_tts_preference(BotAction(content="other"), other) is False


async def test_refuses_when_tts_down_or_server_switched_off(monkeypatch):
    ctx, _ = _ctx(tts_ok=False)
    result = await send_voice_note(ctx, {})
    assert not result.ok and "down" in (result.error or "")
    monkeypatch.setattr("bot.server_features.is_server_feature_enabled", lambda gid, name: False)
    ctx, _ = _ctx(guild=42)
    result = await send_voice_note(ctx, {})
    assert not result.ok and "switched off" in (result.error or "")
    assert not tts_state._voice_reply_messages


def test_pending_requests_are_bounded(monkeypatch):
    import bot.tts.state as mod

    monkeypatch.setattr(mod, "MAX_PENDING_VOICE_REPLIES", 3)
    st = TTSState()
    for i in range(5):
        st.request_voice_reply(i)
    assert list(st._voice_reply_messages) == [2, 3, 4]
