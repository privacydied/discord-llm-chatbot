"""STT speed path: bounded probe, stall detector, direct demux, model prefetch, budget. [PA][REH]"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

import pytest

import bot.hear as hear
import bot.video_ingest as vi
from bot.exceptions import InferenceError
from bot.hear import ProbeTimeoutError, SpanRecorder
from bot.stt_pipeline import direct_demux
from bot.stt_pipeline.prefetch import ModelPrefetcher
from bot.stt_pipeline.spec_select import predict_model_spec

# ---------------------------------------------------------------- probe (1)


class _HangingProc:
    def __init__(self) -> None:
        self.killed = False
        self.returncode = None

    async def communicate(self) -> tuple[bytes, bytes]:
        await asyncio.sleep(3600)
        return b"", b""

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int:
        return -9


async def test_ffprobe_hang_is_bounded_and_killed(monkeypatch, tmp_path: Path) -> None:
    proc = _HangingProc()

    async def fake_exec(*_a, **_k):
        return proc

    monkeypatch.setattr(hear, "FFPROBE_TIMEOUT_S", 0.05)
    monkeypatch.setattr(hear, "_resolve_ffprobe_bin", lambda: "/usr/bin/ffprobe7")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    with pytest.raises(ProbeTimeoutError):
        await hear._ffprobe(tmp_path / "x.mp4")
    assert proc.killed


async def test_probe_timeout_falls_back_to_download_duration(monkeypatch, tmp_path: Path) -> None:
    async def boom(_path):
        raise ProbeTimeoutError("ffprobe timed out after 20s")

    monkeypatch.setattr(hear, "_ffprobe", boom)
    spans = SpanRecorder()
    download = SimpleNamespace(metadata=SimpleNamespace(duration_seconds=223.4))
    duration, sr, channels = await hear._probe_source(tmp_path / "x.mp4", spans, download)
    assert (duration, sr, channels) == (223.4, hear.SAMPLE_RATE, 1)
    assert "probe" in spans.spans


async def test_probe_timeout_without_metadata_raises(monkeypatch, tmp_path: Path) -> None:
    async def boom(_path):
        raise ProbeTimeoutError("ffprobe timed out")

    monkeypatch.setattr(hear, "_ffprobe", boom)
    with pytest.raises(ProbeTimeoutError):
        await hear._probe_source(tmp_path / "x.mp4", SpanRecorder(), SimpleNamespace(metadata=SimpleNamespace(duration_seconds=0)))


async def test_ffmpeg_fallback_probe_never_decodes(monkeypatch, tmp_path: Path) -> None:
    """Regression: the fallback ran `-f null -` = full decode (78 s on a 95 MB mp4)."""
    seen: list[tuple] = []

    class _Proc:
        returncode = 1

        async def communicate(self):
            return b"", b"Duration: 00:03:43.40, start: 0\n  Stream #0:1: Audio: aac, 44100 Hz, stereo"

    async def fake_exec(*cmd, **_k):
        seen.append(cmd)
        return _Proc()

    monkeypatch.setattr(hear, "_resolve_ffprobe_bin", lambda: None)
    monkeypatch.setattr(hear, "_resolve_ffmpeg_bin", lambda: "ffmpeg7")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    duration, sr, channels = await hear._ffprobe(tmp_path / "x.mp4")
    assert (duration, sr, channels) == (223.4, 44100, 2)
    assert "null" not in seen[0] and "-f" not in seen[0]


# ------------------------------------------------------- stall detector (5)


async def _open_stream(code: str, stall_s: float, tmp_path: Path) -> hear.FFMpegPCMStream:
    proc = await asyncio.create_subprocess_exec(sys.executable, "-c", code, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    stream = hear.FFMpegPCMStream(
        process=proc,
        sample_rate=16000,
        frame_samples=160,
        temp_path=tmp_path / "pcm.tmp",
        cache_key="k",
        spans=SpanRecorder(),
        stall_timeout_s=stall_s,
    )
    await stream.start()
    return stream


async def test_slow_consumer_is_not_a_stall(monkeypatch, tmp_path: Path) -> None:
    """Backpressure: ffmpeg blocks writing while the consumer is slow. Must not time out."""
    monkeypatch.setattr(hear, "FFMPEG_MONITOR_POLL_S", 0.05)
    stream = await _open_stream("import sys; sys.stdout.buffer.write(b'\\0' * 320 * 400)", 1.0, tmp_path)
    frames = 0
    async for _ in stream.iter_frames():
        frames += 1
        if frames < 4:
            await asyncio.sleep(0.6)  # 1.8 s of consumer time > 1 s stall limit
    await stream.wait_finished()
    assert frames == 400
    assert stream._error is None
    await stream.finalize(success=False)


async def test_ffmpeg_that_produces_nothing_is_killed(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(hear, "FFMPEG_MONITOR_POLL_S", 0.05)
    stream = await _open_stream("import time; time.sleep(60)", 0.3, tmp_path)

    async def drain():
        async for _ in stream.iter_frames():
            pass

    with pytest.raises(InferenceError, match="stalled"):
        await asyncio.wait_for(drain(), timeout=10)
    assert stream._proc.returncode is not None  # ffmpeg was killed, not left running
    with pytest.raises(InferenceError):
        await stream.finalize(success=False)


# ------------------------------------------------------- direct demux (2)


def test_demux_cmd_copies_audio_only_with_network_whitelist() -> None:
    cmd = direct_demux.build_demux_cmd("ffmpeg7", "https://video.twimg.com/a.mp4", Path("/c/o.m4a"), "mp4")
    joined = " ".join(cmd)
    assert "-map 0:a:0" in joined and "-c:a copy" in joined and "-vn" in cmd
    assert cmd[cmd.index("-protocol_whitelist") + 1] == "https,http,tcp,tls"
    assert direct_demux.demux_target(".MP4") == ("m4a", "mp4")
    assert direct_demux.demux_target("m4a") is None


def test_stderr_watch_reports_duration_once_and_enforces_limit() -> None:
    seen: list[float] = []
    watch = direct_demux._StderrWatch(seen.append, max_duration_s=600)
    watch.feed("  Duration: 00:03:43.40, start: 0.000000, bitrate: 3411 kb/s")
    watch.feed("  Duration: 00:09:00.00, start: 0")
    assert seen == [223.4]
    with pytest.raises(direct_demux.MediaTooLongError):
        direct_demux._StderrWatch(None, max_duration_s=60).feed("Duration: 00:02:00.00,")


async def test_demux_maps_no_audio_stream(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(direct_demux, "find_ffmpeg_bin", lambda: sys.executable)
    code = "import sys; sys.stderr.write('Stream map 0:a:0 matches no streams.\\n'); sys.exit(1)"
    monkeypatch.setattr(direct_demux, "build_demux_cmd", lambda *_a: [sys.executable, "-c", code])
    with pytest.raises(direct_demux.NoAudioTrackError):
        await direct_demux.demux_audio("https://x/a.mp4", tmp_path / "o.m4a", ext="mp4", timeout_s=10)


async def test_demux_success_reports_duration_and_size(monkeypatch, tmp_path: Path) -> None:
    out = tmp_path / "o.m4a"
    monkeypatch.setattr(direct_demux, "find_ffmpeg_bin", lambda: sys.executable)
    code = f"import sys; sys.stderr.write('  Duration: 00:00:12.50, start: 0\\n'); open({str(out)!r}, 'wb').write(b'a' * 100)"
    monkeypatch.setattr(direct_demux, "build_demux_cmd", lambda *_a: [sys.executable, "-c", code])
    hints: list[float] = []
    result = await direct_demux.demux_audio("https://x/a.mp4", out, ext="mp4", timeout_s=10, on_duration=hints.append)
    assert (result.duration_s, result.size, hints) == (12.5, 100, [12.5])


async def test_direct_video_url_uses_demux_and_sets_still_source(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(vi, "CACHE_DIR", tmp_path / "cache")
    mgr = vi.VideoIngestionManager()
    url = "https://video.twimg.com/amplify_video/1/vid/avc1/2560x1408/HgF.mp4"

    async def fake_demux(media_url, out_path, *, ext, timeout_s, on_duration=None, max_duration_s=None):
        out_path.write_bytes(b"a" * 64)
        on_duration(223.4)
        return direct_demux.DemuxResult(path=out_path, duration_s=223.4, size=64)

    hints: list[float] = []
    monkeypatch.setattr(vi, "demux_audio", fake_demux)
    monkeypatch.setattr(vi, "_is_private_hostname", lambda _h: False)
    token = vi.DURATION_HINT.set(hints.append)
    try:
        got = await mgr._direct_media_fallback(original_url=url, media_url=url, parsed=urlparse(url), force_refresh=False, timeout_s=30)
        again = await mgr._direct_media_fallback(original_url=url, media_url=url, parsed=urlparse(url), force_refresh=False, timeout_s=30)
    finally:
        vi.DURATION_HINT.reset(token)
    assert got.raw_path.suffix == ".m4a" and got.still_source == url
    assert got.metadata.duration_seconds == 223.4 and not got.cache_hit
    assert again.cache_hit and again.still_source == url
    assert hints == [223.4, 223.4]


async def test_direct_video_without_audio_falls_back_to_full_download(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(vi, "CACHE_DIR", tmp_path / "cache")
    mgr = vi.VideoIngestionManager()
    url = "https://video.twimg.com/a/silent.mp4"

    async def no_audio(*_a, **_k):
        raise direct_demux.NoAudioTrackError("matches no streams")

    async def full(_url, ext, _timeout):
        path = tmp_path / f"full.{ext}"
        path.write_bytes(b"v" * 10)
        return path, 10

    monkeypatch.setattr(vi, "demux_audio", no_audio)
    monkeypatch.setattr(vi, "_is_private_hostname", lambda _h: False)
    monkeypatch.setattr(mgr, "_download_direct_media", full)
    got = await mgr._direct_media_fallback(original_url=url, media_url=url, parsed=urlparse(url), force_refresh=False, timeout_s=30)
    assert got.raw_path.suffix == ".mp4" and got.still_source is None


# ------------------------------------------------------- model prefetch (3)


class _Mgr:
    def __init__(self) -> None:
        self.default_spec = SimpleNamespace(size="base")
        self.tiny = SimpleNamespace(size="tiny")
        self.loaded: list[str] = []

    def downgrade_spec(self, spec):
        return self.tiny if spec is self.default_spec else None

    async def ensure_model(self, spec):
        self.loaded.append(spec.size)


async def test_prefetch_loads_predicted_spec_once() -> None:
    mgr = _Mgr()
    pre = ModelPrefetcher(manager=mgr, downgrade_threshold_s=120, logger=hear.logger)
    pre.on_duration(223.4)
    pre.on_duration(223.4)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert mgr.loaded == ["tiny"]
    assert predict_model_spec(manager=mgr, duration_in_s=30, downgrade_threshold_s=120) is mgr.default_spec


async def test_prefetch_ignores_unknown_duration() -> None:
    mgr = _Mgr()
    pre = ModelPrefetcher(manager=mgr, downgrade_threshold_s=120, logger=hear.logger)
    pre.on_duration(0)
    await asyncio.sleep(0)
    assert mgr.loaded == [] and pre.spec is None


# ------------------------------------------------------- budget guardrail


def test_transcribe_budget_uses_download_duration() -> None:
    download = SimpleNamespace(metadata=SimpleNamespace(duration_seconds=223.4))
    assert hear._transcribe_budget_s(lambda d: d * 2, download) == pytest.approx(446.8)
    assert hear._transcribe_budget_s(None, download) is None


def test_x_stt_timeout_scales_with_real_duration() -> None:
    from bot.router import X_STT_MIN_TIMEOUT_S, Router

    router = Router.__new__(Router)
    router.config = {}
    assert router._x_stt_timeout_for(0) == X_STT_MIN_TIMEOUT_S
    assert router._x_stt_timeout_for(None) == X_STT_MIN_TIMEOUT_S
    # 223 s clip: 223/1.5*1.6+45 = 283 s, not the 120 s floor the dur=0 path gave.
    assert router._x_stt_timeout_for(223.4) > 250
    router.config = {"X_STT_TIMEOUT_S": "75"}
    assert router._x_stt_timeout_for(223.4) == 75
