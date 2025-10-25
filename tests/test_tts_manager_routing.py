from pathlib import Path
import sys
import asyncio
import logging
import types
from types import SimpleNamespace

import pytest


class _FakeAsyncClient:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        raise RuntimeError("httpx is not available in tests")

    async def __aexit__(self, exc_type, exc, tb):
        return False


sys.modules.setdefault(
    "httpx",
    SimpleNamespace(
        AsyncClient=_FakeAsyncClient,
        HTTPStatusError=Exception,
        RequestError=Exception,
    ),
)


class _FakeClientTimeout:
    def __init__(self, *args, **kwargs):
        pass


class _FakeClientSession:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        raise RuntimeError("aiohttp is not available in tests")

    async def __aexit__(self, exc_type, exc, tb):
        return False


sys.modules.setdefault(
    "aiohttp",
    SimpleNamespace(
        ClientSession=_FakeClientSession,
        ClientTimeout=_FakeClientTimeout,
    ),
)


class _FakeRichHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - stub
        pass


sys.modules.setdefault(
    "rich.logging",
    SimpleNamespace(RichHandler=_FakeRichHandler),
)


kokoro_v8_stub = types.ModuleType("bot.tts.engines.kokoro_v8")


class _FakeKokoroV8Engine:
    async def synthesize(self, text: str) -> bytes:  # pragma: no cover - stub
        return b"v8"


kokoro_v8_stub.KokoroV8Engine = _FakeKokoroV8Engine
sys.modules.setdefault("bot.tts.engines.kokoro_v8", kokoro_v8_stub)


bot_action_stub = types.ModuleType("bot.action")


class _FakeBotAction:
    def __init__(self, content: str | None = None, meta: dict | None = None):
        self.content = content or ""
        self.meta = meta or {}
        self.audio_path: str | None = None


bot_action_stub.BotAction = _FakeBotAction
sys.modules.setdefault("bot.action", bot_action_stub)

from bot.tts.interface import TTSManager
from bot.tts.errors import SynthesisError


class _DummyKokoro:
    def __init__(self, model_path=None, voices_path=None):
        self.model_path = model_path
        self.voices_path = voices_path

    async def synthesize(self, text: str) -> bytes:
        return b"ok"


def test_default_engine_prefers_kokoro_when_assets_available(monkeypatch, tmp_path):
    tts_dir = tmp_path / "tts"
    tts_dir.mkdir()
    model_file = tts_dir / "kokoro-v1.0.onnx"
    voices_file = tts_dir / "voices-v1.0.bin"
    model_file.write_bytes(b"m")
    voices_file.write_bytes(b"v")

    monkeypatch.delenv("TTS_ENGINE", raising=False)
    monkeypatch.setenv("TTS_MODEL_PATH", str(model_file))
    monkeypatch.setenv("TTS_VOICES_PATH", str(voices_file))

    import bot.tts.interface as tts_interface

    monkeypatch.setattr(tts_interface, "KokoroONNXEngine", _DummyKokoro)
    monkeypatch.setitem(tts_interface.ENGINES, "kokoro-onnx", _DummyKokoro)

    manager = TTSManager()
    try:
        status = manager.get_status()
        assert status["engine"] == "kokoro-onnx"
        assert manager.is_available()

        audio = asyncio.run(manager.synthesize("hello there"))
        assert audio == b"ok"
    finally:
        asyncio.run(manager.close())


def test_missing_assets_raise_synthesis_error(monkeypatch, tmp_path):
    async def failing_ensure(_out_dir: Path):
        raise RuntimeError("download failed")

    monkeypatch.setenv("TTS_ENGINE", "kokoro-onnx")
    monkeypatch.delenv("TTS_MODEL_PATH", raising=False)
    monkeypatch.delenv("TTS_VOICES_PATH", raising=False)
    import bot.tts.interface as tts_interface

    monkeypatch.setattr(tts_interface, "ensure_kokoro_assets", failing_ensure)

    manager = TTSManager()
    try:
        status = manager.get_status()
        assert status["degraded"]
        with pytest.raises(SynthesisError):
            asyncio.run(manager.synthesize("please work"))
    finally:
        asyncio.run(manager.close())


def test_runtime_engine_error_propagates(monkeypatch, tmp_path):
    tts_dir = tmp_path / "tts"
    tts_dir.mkdir()
    model_file = tts_dir / "kokoro-v1.0.onnx"
    voices_file = tts_dir / "voices-v1.0.bin"
    model_file.write_bytes(b"m")
    voices_file.write_bytes(b"v")

    class ExplodingEngine:
        def __init__(self, model_path=None, voices_path=None):
            self.model_path = model_path
            self.voices_path = voices_path

        def synthesize(self, text: str) -> bytes:
            raise RuntimeError("boom")

    async def ensure_paths(_out_dir: Path):
        return model_file, voices_file

    monkeypatch.delenv("TTS_ENGINE", raising=False)
    monkeypatch.setenv("TTS_MODEL_PATH", str(model_file))
    monkeypatch.setenv("TTS_VOICES_PATH", str(voices_file))
    import bot.tts.interface as tts_interface

    monkeypatch.setattr(tts_interface, "KokoroONNXEngine", ExplodingEngine)
    monkeypatch.setitem(tts_interface.ENGINES, "kokoro-onnx", ExplodingEngine)
    monkeypatch.setattr(tts_interface, "ensure_kokoro_assets", ensure_paths)

    manager = TTSManager()
    try:
        assert manager.is_available()
        with pytest.raises(SynthesisError):
            asyncio.run(manager.synthesize("general kenobi"))
    finally:
        asyncio.run(manager.close())
