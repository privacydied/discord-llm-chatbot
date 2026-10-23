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
