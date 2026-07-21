"""Tests for TTSManager idle-TTL unload of the Kokoro ONNX session. [PA]"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

from bot.tts.manager import TTSManager


def _manager_with_loaded_kokoro() -> TTSManager:
    manager = TTSManager(config={})
    manager.kokoro = MagicMock(name="kokoro_session")
    manager.available = True
    return manager


class TestUnloadIfIdle:
    def test_unloads_after_idle_ttl(self) -> None:
        manager = _manager_with_loaded_kokoro()
        manager._last_used = time.monotonic() - 1000

        assert manager.unload_if_idle(idle_seconds=900) is True
        assert manager.kokoro is None

    def test_noop_when_recently_used(self) -> None:
        manager = _manager_with_loaded_kokoro()
        manager._last_used = time.monotonic()

        assert manager.unload_if_idle(idle_seconds=900) is False
        assert manager.kokoro is not None

    def test_noop_when_disabled_or_not_loaded(self) -> None:
        manager = _manager_with_loaded_kokoro()
        manager._last_used = time.monotonic() - 1000
        assert manager.unload_if_idle(idle_seconds=0) is False  # disabled
        assert manager.kokoro is not None

        manager.kokoro = None
        assert manager.unload_if_idle(idle_seconds=900) is False  # nothing loaded

    def test_generate_speech_reloads_and_touches_idle_clock(self, monkeypatch, tmp_path) -> None:
        """After an idle unload, the next synthesis lazily reloads the model
        and resets the idle clock so it isn't immediately unloaded again."""
        manager = _manager_with_loaded_kokoro()
        manager._last_used = time.monotonic() - 1000
        assert manager.unload_if_idle(idle_seconds=900) is True

        session = MagicMock(name="reloaded_session")
        session.create.return_value = tmp_path / "out.wav"

        def fake_load_model() -> None:
            manager.kokoro = session
            manager.available = True

        monkeypatch.setattr(manager, "load_model", fake_load_model)

        result = manager.generate_speech("hello", voice="af_heart")

        assert result == tmp_path / "out.wav"
        session.create.assert_called_once()
        assert manager.unload_if_idle(idle_seconds=900) is False  # clock refreshed
