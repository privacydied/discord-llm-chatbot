"""TTSState: env default for 'every reply is voice', overrides, per-user opt-out."""

from bot.tts.state import TTSState


def test_env_default_is_read_per_check(monkeypatch):
    state = TTSState()
    monkeypatch.delenv("TTS_GLOBAL_DEFAULT", raising=False)
    assert state.is_user_tts_enabled(1) is False
    monkeypatch.setenv("TTS_GLOBAL_DEFAULT", "true")  # .env hot-reload / late load
    assert state.is_user_tts_enabled(1) is True


def test_admin_toggle_overrides_env(monkeypatch):
    monkeypatch.setenv("TTS_GLOBAL_DEFAULT", "true")
    state = TTSState()
    state.set_global_tts(False)
    assert state.is_user_tts_enabled(1) is False


def test_user_opt_out_beats_global_and_opt_in_beats_off(monkeypatch):
    monkeypatch.setenv("TTS_GLOBAL_DEFAULT", "true")
    state = TTSState()
    state.set_user_tts(1, False)
    assert state.is_user_tts_enabled(1) is False
    assert state.is_user_tts_enabled(2) is True
    monkeypatch.setenv("TTS_GLOBAL_DEFAULT", "false")
    state.set_user_tts(3, True)
    assert state.is_user_tts_enabled(3) is True


# -- apply_tts_preference: one rule for every reply path ----------------------------

from types import SimpleNamespace  # noqa: E402

from bot.action import BotAction  # noqa: E402
from bot.tts.state import apply_tts_preference  # noqa: E402


def _msg(user=1, guild=None):
    return SimpleNamespace(author=SimpleNamespace(id=user), guild=SimpleNamespace(id=guild) if guild else None)


def test_global_default_voices_tool_loop_text_reply(monkeypatch):
    """The 2026-10-23 bug: text_default/tool-loop replies never checked TTS prefs."""
    monkeypatch.setenv("TTS_GLOBAL_DEFAULT", "true")
    action = BotAction(content="paris, obviously")
    assert apply_tts_preference(action, _msg(), state=TTSState()) is True
    assert action.meta["requires_tts"] is True


def test_no_voice_when_off_opted_out_or_not_plain_text(monkeypatch):
    monkeypatch.setenv("TTS_GLOBAL_DEFAULT", "true")
    state = TTSState()
    state.set_user_tts(2, False)
    assert apply_tts_preference(BotAction(content="hi"), _msg(user=2), state=state) is False
    assert apply_tts_preference(BotAction(content="look", files=[object()]), _msg(), state=state) is False
    assert apply_tts_preference(BotAction(content="   "), _msg(), state=state) is False
    monkeypatch.setenv("TTS_GLOBAL_DEFAULT", "false")
    assert apply_tts_preference(BotAction(content="hi"), _msg(), state=TTSState()) is False


def test_one_time_flag_is_consumed_and_server_switch_respected(monkeypatch):
    monkeypatch.delenv("TTS_GLOBAL_DEFAULT", raising=False)
    state = TTSState()
    state.set_one_time_tts(1)
    assert apply_tts_preference(BotAction(content="hi"), _msg(), state=state) is True
    assert apply_tts_preference(BotAction(content="hi again"), _msg(), state=state) is False
    monkeypatch.setenv("TTS_GLOBAL_DEFAULT", "true")
    monkeypatch.setattr("bot.server_features.is_server_feature_enabled", lambda gid, name: False)
    assert apply_tts_preference(BotAction(content="hi"), _msg(guild=42), state=state) is False
