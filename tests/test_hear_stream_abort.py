import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

import bot.hear as hear
from bot.hear import BasePCMStream


class _BlockedProducerStream(BasePCMStream):
    """Producer that blocks on queue put when no consumer drains frames."""

    async def _produce(self) -> None:
        await self._queue.put(b"a")
        await self._queue.put(b"b")
        # With queue_depth=2 and no consumer, this put blocks indefinitely
        # unless abort() cancels producer and signals stream end.
        await self._queue.put(b"c")


@pytest.mark.asyncio
async def test_abort_finalize_does_not_hang_with_full_queue() -> None:
    stream = _BlockedProducerStream(sample_rate=16000, frame_samples=160, queue_depth=2)
    await stream.start()
    await asyncio.sleep(0.05)

    await stream.abort()
    await asyncio.wait_for(stream.finalize(success=False), timeout=1.0)


# ---------------------------------------------------------------------------
# _preprocess_audio must drain ffmpeg's stdout eagerly, not defer it to
# whenever the caller first calls iter_frames(). Regression: ffmpeg starts
# writing PCM to its stdout pipe the moment it's spawned; the OS pipe buffer
# (~64KB on Linux) fills in well under a second for a real audio stream. If
# nothing reads from it before then, ffmpeg blocks on write() -- a real
# deadlock -- until the _monitor() pre_budget timeout kills it. Previously,
# reading only started inside iter_frames() (called deep inside
# _transcribe_with_model, itself gated behind ensure_model()'s executor
# round-trip), leaving a window where ffmpeg could silently deadlock for
# however long that took. [REH][PA]
# ---------------------------------------------------------------------------


class _FakeStdout:
    """Mimics asyncio.StreamReader closely enough for FFMpegPCMStream._produce."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)
        self.read_calls = 0

    async def read(self, _n: int) -> bytes:
        self.read_calls += 1
        if self._chunks:
            return self._chunks.pop(0)
        return b""


class _FakeStderr:
    async def read(self) -> bytes:
        return b""


class _FakeProcess:
    def __init__(self, stdout_chunks: list[bytes]) -> None:
        self.stdout = _FakeStdout(stdout_chunks)
        self.stderr = _FakeStderr()
        self.returncode: int | None = None
        self._wait_event = asyncio.Event()

    async def wait(self) -> int:
        await self._wait_event.wait()
        return 0

    def kill(self) -> None:
        self.returncode = -9
        self._wait_event.set()


@pytest.mark.asyncio
async def test_preprocess_audio_starts_draining_ffmpeg_stdout_before_returning(monkeypatch, tmp_path) -> None:
    fake_proc = _FakeProcess(stdout_chunks=[b"\x00\x00" * 100, b"\x00\x00" * 100])

    async def fake_create_subprocess_exec(*_args, **_kwargs):
        return fake_proc

    monkeypatch.setattr(hear.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr(hear, "PCM_CACHE_DIR", tmp_path)
    monkeypatch.setattr(hear, "_ffprobe", lambda _path: _async_return((1.0, 16000, 1)))
    monkeypatch.setattr(hear, "_resolve_ffmpeg_bin", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(hear, "_FFMPEG_BIN_HAS_AAC", True)
    monkeypatch.setattr(hear, "_source_hash", lambda *_a, **_k: "deadbeef" * 2)

    ram_guard = SimpleNamespace(check=lambda *_a, **_k: None)
    spans = hear.SpanRecorder()

    pre = await hear._preprocess_audio(
        source_path=Path("fake.ogg"),
        spans=spans,
        download=None,
        voice_note=True,
        ram_guard=ram_guard,
    )

    # The producer task must already exist -- start() must have been called
    # by _preprocess_audio itself, not deferred to the caller's first
    # iter_frames() call.
    assert pre.stream._producer_task is not None

    # Let the scheduled producer task actually run at least one iteration and
    # confirm it is genuinely reading from ffmpeg's stdout, not just
    # constructed-but-idle.
    for _ in range(5):
        await asyncio.sleep(0)
    assert fake_proc.stdout.read_calls >= 1

    await pre.stream.abort()
    await asyncio.wait_for(pre.stream.finalize(success=False), timeout=1.0)


async def _async_return(value):
    return value
