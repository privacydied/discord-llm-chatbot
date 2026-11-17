"""Copy only the audio track out of a direct video URL. [PA][REH][SFT]

A direct video URL (for example ``video.twimg.com/.../2560x1408/x.mp4``) has
no audio-only rendition, so the old path downloaded the whole file: 95 MB of
2560x1408 video when STT needs only the AAC track. ffmpeg reads the container
index over HTTP range requests and copies just the audio packets. Measured on
the NAS for that clip: 5.5 s and 3.6 MB, against 27 s and 95 MB before.

ffmpeg prints the container duration as soon as it has read the index, so
``on_duration`` fires within a second or two of the start. The caller uses it
to load the right whisper model while the audio is still downloading.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shutil
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .ffmpeg_runtime import ffmpeg_candidates_from_env

if TYPE_CHECKING:
    from collections.abc import Callable

# source extension -> (output extension, ffmpeg muxer). AAC copies cleanly into
# mp4; webm carries opus/vorbis, which matroska takes without re-encoding.
_VIDEO_CONTAINERS: dict[str, tuple[str, str]] = {
    "mp4": ("m4a", "mp4"),
    "m4v": ("m4a", "mp4"),
    "mov": ("m4a", "mp4"),
    "webm": ("mka", "matroska"),
}
# Only network protocols: a crafted container cannot make ffmpeg read local
# files (file:, concat:, subfile:) through a nested reference. [SFT]
ALLOWED_PROTOCOLS = "https,http,tcp,tls"
# Abort when the server sends nothing for this long (ffmpeg takes microseconds).
RW_TIMEOUT_US = 15_000_000
_REAP_TIMEOUT_S = 5.0
_STDERR_TAIL_LINES = 20
_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_NO_AUDIO_MARKERS = ("matches no streams", "does not contain any stream")
_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"


class DemuxError(RuntimeError):
    """ffmpeg could not copy the audio track (network, codec, timeout)."""


class NoAudioTrackError(DemuxError):
    """The media has no audio stream; the caller needs the full video for a still."""


class MediaTooLongError(DemuxError):
    """The container duration is above the caller's limit."""

    def __init__(self, duration_s: float, limit_s: float) -> None:
        super().__init__(f"media too long: {duration_s:.1f}s (max {limit_s:.0f}s)")
        self.duration_s = duration_s
        self.limit_s = limit_s


@dataclass(frozen=True)
class DemuxResult:
    path: Path
    duration_s: float
    size: int


def demux_target(ext: str) -> tuple[str, str] | None:
    """(output extension, muxer) for a video container extension, else None."""
    return _VIDEO_CONTAINERS.get((ext or "").lower().lstrip("."))


def find_ffmpeg_bin() -> str | None:
    """First installed ffmpeg. A stream copy decodes nothing, so no AAC probe."""
    for candidate in ffmpeg_candidates_from_env():
        if os.path.sep in candidate:
            if Path(candidate).exists():
                return candidate
            continue
        found = shutil.which(candidate)
        if found:
            return found
    return None


def build_demux_cmd(ffmpeg_bin: str, url: str, out_path: Path, muxer: str) -> list[str]:
    return [
        ffmpeg_bin,
        "-hide_banner",
        "-nostdin",
        "-nostats",
        "-loglevel",
        "info",
        "-protocol_whitelist",
        ALLOWED_PROTOCOLS,
        "-rw_timeout",
        str(RW_TIMEOUT_US),
        "-user_agent",
        _USER_AGENT,
        "-i",
        url,
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-c:a",
        "copy",
        "-f",
        muxer,
        "-y",
        str(out_path),
    ]


def parse_duration(line: str) -> float | None:
    match = _DURATION_RE.search(line)
    if not match:
        return None
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


class _StderrWatch:
    """Reads ffmpeg stderr by line: keeps a tail, reports the duration once."""

    def __init__(self, on_duration: Callable[[float], None] | None, max_duration_s: float | None) -> None:
        self.duration_s = 0.0
        self.tail: deque[str] = deque(maxlen=_STDERR_TAIL_LINES)
        self._on_duration = on_duration
        self._max = max_duration_s

    def feed(self, line: str) -> None:
        self.tail.append(line)
        if self.duration_s:
            return
        duration = parse_duration(line)
        if not duration:
            return
        self.duration_s = duration
        if self._max and duration > self._max:
            raise MediaTooLongError(duration, self._max)
        if self._on_duration is not None:
            with contextlib.suppress(Exception):
                self._on_duration(duration)

    async def run(self, stream: asyncio.StreamReader) -> None:
        async for raw in stream:
            self.feed(raw.decode(errors="ignore").strip())


async def _reap(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is None:
        with contextlib.suppress(ProcessLookupError, OSError):
            proc.kill()
    with contextlib.suppress(Exception):
        await asyncio.wait_for(proc.wait(), timeout=_REAP_TIMEOUT_S)


def _failure(returncode: int | None, watch: _StderrWatch) -> DemuxError:
    text = " | ".join(watch.tail)
    if any(marker in text for marker in _NO_AUDIO_MARKERS):
        return NoAudioTrackError(text[-300:])
    return DemuxError(f"ffmpeg exit {returncode}: {text[-300:]}")


async def demux_audio(
    url: str,
    out_path: Path,
    *,
    ext: str,
    timeout_s: float,
    on_duration: Callable[[float], None] | None = None,
    max_duration_s: float | None = None,
) -> DemuxResult:
    """Copy the first audio stream of ``url`` into ``out_path``. Raises DemuxError."""
    target = demux_target(ext)
    ffmpeg_bin = find_ffmpeg_bin()
    if target is None or ffmpeg_bin is None:
        msg = f"demux not possible ext={ext} ffmpeg={bool(ffmpeg_bin)}"
        raise DemuxError(msg)
    cmd = build_demux_cmd(ffmpeg_bin, url, out_path, target[1])
    proc = await asyncio.create_subprocess_exec(*cmd, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
    watch = _StderrWatch(on_duration, max_duration_s)
    try:
        await asyncio.wait_for(asyncio.gather(watch.run(proc.stderr), proc.wait()), timeout=timeout_s)
    except TimeoutError as exc:
        msg = f"demux timed out after {timeout_s:.0f}s"
        raise DemuxError(msg) from exc
    finally:
        await _reap(proc)
    if proc.returncode != 0:
        raise _failure(proc.returncode, watch)
    size = out_path.stat().st_size if out_path.exists() else 0
    if size <= 0:
        msg = "demux produced an empty file"
        raise DemuxError(msg)
    return DemuxResult(path=out_path, duration_s=watch.duration_s, size=size)
