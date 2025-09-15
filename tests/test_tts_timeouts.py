import os
from pathlib import Path
import tempfile
import pytest

from bot.tts.interface import TTSManager
from bot.action import BotAction


@pytest.mark.asyncio
async def test_generate_tts_dynamic_timeout(monkeypatch, tmp_path):
    # Set env timeouts: base 25, cold 5.5, warm 1.1
    os.environ["TTS_TIMEOUT_S"] = "25.0"
    os.environ["TTS_TIMEOUT_COLD_S"] = "5.5"
    os.environ["TTS_TIMEOUT_WARM_S"] = "1.1"

    mgr = TTSManager(bot=None)

    captured = {"timeout": None}

    async def fake_synthesize(self, text: str, timeout: float = 0.0) -> bytes:
        captured["timeout"] = timeout
        # return minimal WAV-like bytes
        return b"RIFF....WAVEfmt "

    # Patch synthesize to capture timeout chosen by generate_tts
    monkeypatch.setattr(TTSManager, "synthesize", fake_synthesize, raising=False)

    # Cold path (warmed_up False) -> cold timeout
    mgr._warmed_up = False
    out1 = tmp_path / "a.wav"
    p1 = await mgr.generate_tts("hello", out_path=out1)
    assert p1 == out1
    assert pytest.approx(captured["timeout"], rel=1e-6) == 5.5

    # Warm path (warmed_up True) -> warm timeout
    mgr._warmed_up = True
    out2 = tmp_path / "b.wav"
    p2 = await mgr.generate_tts("hello again", out_path=out2)
    assert p2 == out2
    assert pytest.approx(captured["timeout"], rel=1e-6) == 1.1


@pytest.mark.asyncio
async def test_process_timeout_selection_with_meta(monkeypatch, tmp_path):
    # Env defaults
    os.environ["TTS_TIMEOUT_S"] = "25.0"
    os.environ["TTS_TIMEOUT_COLD_S"] = "10.0"
    os.environ["TTS_TIMEOUT_WARM_S"] = "2.0"

    mgr = TTSManager(bot=None)

    captured = {"timeout": None}

    async def fake_generate(
        self, text: str, out_path=None, timeout: float | None = None
    ):
        captured["timeout"] = timeout
        # Create a temporary path to return
        fd, name = tempfile.mkstemp(prefix="test_tts_", suffix=".wav", dir=tmp_path)
        os.close(fd)
        return Path(name)

    monkeypatch.setattr(TTSManager, "generate_tts", fake_generate, raising=False)

    # Force cold via meta + per-call overrides
    action = BotAction(
        content="hello world",
        meta={
            "tts_cold": True,
            "tts_timeout_cold_s": 7.77,
            "tts_timeout_warm_s": 0.88,
        },
    )
    res = await mgr.process(action)
    assert isinstance(res.audio_path, str)
    assert pytest.approx(captured["timeout"], rel=1e-6) == 7.77

    # Force warm via meta
    # Use different text to avoid cache hit
    action2 = BotAction(
        content="hello world again",
        meta={
            "tts_cold": False,
            "tts_timeout_cold_s": 7.77,
            "tts_timeout_warm_s": 0.88,
        },
    )
    res2 = await mgr.process(action2)
    assert isinstance(res2.audio_path, str)
    assert pytest.approx(captured["timeout"], rel=1e-6) == 0.88

    # Hard override via tts_timeout_s should take precedence
    # Different text again to avoid cache
    action3 = BotAction(
        content="another sample",
        meta={
            "tts_timeout_s": 3.3,
            "tts_cold": True,  # should be ignored due to explicit timeout
            "tts_timeout_cold_s": 9.99,
            "tts_timeout_warm_s": 0.11,
        },
    )
    res3 = await mgr.process(action3)
    assert isinstance(res3.audio_path, str)
    assert pytest.approx(captured["timeout"], rel=1e-6) == 3.3


@pytest.mark.asyncio
async def test_process_heuristic_env_when_no_meta(monkeypatch, tmp_path):
    # Set env only
    os.environ["TTS_TIMEOUT_S"] = "25.0"
    os.environ["TTS_TIMEOUT_COLD_S"] = "4.4"
    os.environ["TTS_TIMEOUT_WARM_S"] = "1.1"

    mgr = TTSManager(bot=None)

    captured = {"timeout": None}

    async def fake_generate(
        self, text: str, out_path=None, timeout: float | None = None
    ):
        captured["timeout"] = timeout
        fd, name = tempfile.mkstemp(prefix="test_tts_", suffix=".wav", dir=tmp_path)
        os.close(fd)
        return Path(name)

    monkeypatch.setattr(TTSManager, "generate_tts", fake_generate, raising=False)

    # Cold heuristic (not warmed yet) -> cold env
    mgr._warmed_up = False
    a1 = BotAction(content="X")
    await mgr.process(a1)
    assert pytest.approx(captured["timeout"], rel=1e-6) == 4.4

    # Warm heuristic -> warm env
    mgr._warmed_up = True
    a2 = BotAction(content="Y")
    await mgr.process(a2)
    assert pytest.approx(captured["timeout"], rel=1e-6) == 1.1
