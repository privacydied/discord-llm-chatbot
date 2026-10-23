"""Pocket TTS engine + TTSManager wiring. [CA][REH]

A tiny aiohttp app stands in for the Pocket server (``/health``, ``POST /tts``
streaming a WAV with placeholder sizes) so tests never need torch or weights.
"""

from __future__ import annotations

import asyncio
import io
import struct
import sys
import wave
from pathlib import Path

import pytest
from aiohttp import web

from bot.tts.engines.pocket import PocketTTSEngine, fix_wav_header
from bot.tts.errors import SynthesisError
from bot.tts.interface import TTSManager


def _streamed_wav(samples: int = 2400, rate: int = 24000) -> bytes:
    """A WAV as Pocket streams it: RIFF/data sizes unknown (0xFFFFFFFF)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x01\x00" * samples)
    data = bytearray(buf.getvalue())
    struct.pack_into("<I", data, 4, 0xFFFFFFFF)
    struct.pack_into("<I", data, data.index(b"data") + 4, 0xFFFFFFFF)
    return bytes(data)


@pytest.fixture(autouse=True)
def _independent_of_live_env(monkeypatch):
    """The NAS .env turns these on; tests state what they need explicitly."""
    for key in ("POCKET_TTS_QUANTIZE", "POCKET_TTS_KEEP_WARM"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
async def pocket_server(unused_tcp_port):
    calls: list[dict] = []

    async def health(_req):
        return web.json_response({"status": "healthy"})

    async def tts(req):
        form = await req.post()
        calls.append(dict(form))
        if form.get("text") == "boom":
            return web.Response(status=500, text="model exploded")
        return web.Response(body=_streamed_wav(), content_type="audio/wav")

    app = web.Application()
    app.router.add_get("/health", health)
    app.router.add_post("/tts", tts)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", unused_tcp_port).start()
    yield f"http://127.0.0.1:{unused_tcp_port}", calls
    await runner.cleanup()


def test_fix_wav_header_repairs_streamed_sizes():
    fixed = fix_wav_header(_streamed_wav(samples=100))
    with wave.open(io.BytesIO(fixed)) as w:
        assert w.getnframes() == 100 and w.getframerate() == 24000
    with pytest.raises(SynthesisError):
        fix_wav_header(b"OggS....")


async def test_synthesize_posts_text_and_voice(pocket_server):
    url, calls = pocket_server
    engine = PocketTTSEngine({"POCKET_TTS_URL": url, "POCKET_TTS_AUTOSTART": "false"}, voice="marius")
    audio = await engine.synthesize("hello there")
    assert calls == [{"text": "hello there", "voice_url": "marius"}]
    with wave.open(io.BytesIO(audio)) as w:
        assert w.getnframes() == 2400


async def test_server_error_and_unreachable_raise_synthesis_error(pocket_server, unused_tcp_port_factory):
    url, _ = pocket_server
    engine = PocketTTSEngine({"POCKET_TTS_URL": url, "POCKET_TTS_AUTOSTART": "false"})
    with pytest.raises(SynthesisError, match="HTTP 500"):
        await engine.synthesize("boom")
    down = PocketTTSEngine({"POCKET_TTS_URL": f"http://127.0.0.1:{unused_tcp_port_factory()}", "POCKET_TTS_AUTOSTART": "false"})
    with pytest.raises(SynthesisError, match="not reachable"):
        await down.synthesize("hi")


async def test_autostart_spawns_server_waits_for_health_and_idle_stops(tmp_path, unused_tcp_port, monkeypatch):
    """Autostart launches POCKET_TTS_PYTHON; a fake ``pocket_tts`` module serves the API."""
    pkg = tmp_path / "pocket_tts"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    wav = _streamed_wav()
    (pkg / "__main__.py").write_text(
        "import sys, time\n"
        "from aiohttp import web\n"
        "args = sys.argv[1:]\n"
        "port = int(args[args.index('--port') + 1])\n"
        "assert args[args.index('--default-voice') + 1] == 'alba', args\n"
        "time.sleep(1.5)  # model load\n"
        "async def health(_r): return web.json_response({'status': 'healthy'})\n"
        f"async def tts(_r): return web.Response(body={wav!r}, content_type='audio/wav')\n"
        "app = web.Application(); app.router.add_get('/health', health); app.router.add_post('/tts', tts)\n"
        "web.run_app(app, host='127.0.0.1', port=port, print=None)\n",
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))  # the child inherits os.environ
    monkeypatch.delenv("TTS_VOICE", raising=False)  # default voice, independent of the live .env
    engine = PocketTTSEngine(
        {
            "POCKET_TTS_URL": f"http://127.0.0.1:{unused_tcp_port}",
            "POCKET_TTS_PYTHON": sys.executable,
            "POCKET_TTS_START_TIMEOUT_S": "30",
            "POCKET_TTS_LOG": str(tmp_path / "pocket.log"),
        },
    )
    assert engine.timeout_budget("x" * 100) > 30  # cold start counted in the budget
    try:
        audio = await engine.synthesize("first call starts the server")
        assert audio[:4] == b"RIFF"
        proc = engine._proc
        assert proc is not None and proc.returncode is None
        assert engine.timeout_budget("x" * 100) < 30  # warm now
        assert not engine.idle_stop_due(3600)
        assert engine.idle_stop_due(0)
        await engine.close()
        assert proc.returncode is not None and engine._proc is None
    finally:
        await engine.close()


async def test_autostart_without_interpreter_fails_clearly(unused_tcp_port):
    engine = PocketTTSEngine({"POCKET_TTS_URL": f"http://127.0.0.1:{unused_tcp_port}", "POCKET_TTS_PYTHON": "/nope/python"})
    with pytest.raises(SynthesisError, match="POCKET_TTS_PYTHON"):
        await engine.synthesize("hi")


@pytest.mark.parametrize("legacy", ["kokoro-onnx", "kokoro", "kokoro-v8", ""])
def test_manager_maps_legacy_and_default_backends_to_pocket(tmp_path, legacy, monkeypatch):
    monkeypatch.delenv("TTS_ENGINE", raising=False)
    monkeypatch.delenv("TTS_VOICE", raising=False)
    manager = TTSManager(config={"TTS_ENGINE": legacy, "TTS_CACHE_DIR": str(tmp_path)})
    assert manager.backend == "pocket"
    assert isinstance(manager._engine, PocketTTSEngine)
    assert manager.voice == "alba"


async def test_manager_generates_ogg_via_pocket_and_widens_short_timeouts(pocket_server, tmp_path, monkeypatch):
    url, calls = pocket_server
    monkeypatch.delenv("TTS_ENGINE", raising=False)
    manager = TTSManager(
        config={"TTS_ENGINE": "pocket", "POCKET_TTS_URL": url, "POCKET_TTS_AUTOSTART": "false", "TTS_CACHE_DIR": str(tmp_path), "TTS_VOICE": "javert"},
    )
    path = await manager.generate_tts("**bold** take https://x.y", timeout=0.001, output_format="wav")
    assert Path(path).read_bytes()[:4] == b"RIFF"
    assert calls[-1] == {"text": "bold take", "voice_url": "javert"}  # markdown/URL cleaned, tiny timeout widened
    assert manager.unload_if_idle(0) is False  # server not bot-owned: never stopped
    await manager.close()
    await asyncio.sleep(0)


def test_quantize_flag_and_keep_warm_disable_idle_stop():
    engine = PocketTTSEngine({"POCKET_TTS_URL": "http://127.0.0.1:8767", "POCKET_TTS_QUANTIZE": "true", "POCKET_TTS_KEEP_WARM": "true", "POCKET_TTS_PYTHON": sys.executable}, voice="jean")
    assert engine._serve_command()[-1] == "--quantize"
    assert "--quantize" not in PocketTTSEngine({"POCKET_TTS_URL": "http://127.0.0.1:8767"})._serve_command()

    class _Alive:
        returncode = None

    engine._proc = _Alive()
    engine._last_used = 0.0
    assert engine.idle_stop_due(1.0) is False  # keep-warm never idles out
    engine.keep_warm = False
    assert engine.idle_stop_due(1.0) is True


async def test_warm_confirms_running_server(pocket_server):
    url, calls = pocket_server
    engine = PocketTTSEngine({"POCKET_TTS_URL": url, "POCKET_TTS_AUTOSTART": "false", "POCKET_TTS_KEEP_WARM": "true"})
    await engine.warm()  # healthy server: no spawn, no synthesis
    assert calls == []
