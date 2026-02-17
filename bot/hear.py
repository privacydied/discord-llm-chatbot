"""
Centralised speech-to-text pipeline optimised for CPU-only deployments.

Stages:
    yt-dlp download (URL inputs only)
    FFmpeg preprocess (mono/16k PCM with optional silence removal + atempo)
    faster-whisper transcription (chunked, adaptive model selection)
    Stitch (packaging transcript + metadata)

Instrumentation emits stt.span and stt.summary breadcrumbs to keep visibility tight.
"""

from __future__ import annotations

import asyncio
import ctypes
import gc
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import psutil

from .exceptions import InferenceError
from .utils.logging import get_logger
from .video_ingest import (
    DownloadedAudio,
    VideoIngestError,
    fetch_and_prepare_url_audio,
)
from .config import load_config
from .stt import ModelSpec, stt_manager
from .stt_module.failure_classifier import STTFailureClassifier
from .stt_module.multimodal_fallback import multimodal_fallback_provider
from .stt_pipeline import (
    build_url_transcript_result,
    build_youtube_transcript_result,
    ensure_stt_manager_ready,
    ffmpeg_bin_has_aac,
    ffmpeg_candidates_from_env,
    ffmpeg_supports_aac_decoder,
    load_stt_runtime_compat,
    parse_stt_max_ram_mb,
)
from .youtube_transcript import resolve_youtube_transcript

if TYPE_CHECKING:
    import discord
    from faster_whisper import WhisperModel

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants & thresholds tuned for dual-core CPU-only environments
# ---------------------------------------------------------------------------

PCM_CACHE_DIR = Path("cache/stt_pcm")
TRANSCRIPT_CACHE_DIR = Path("cache/stt_transcripts")
PCM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

PRE_PARAMS_VERSION = "mono16k_v2"
SAMPLE_RATE = 16000
SHORT_CLIP_S = 12.0
ATEMPO_THRESHOLD_S = 30.0
ATEMPO_FACTOR = 1.25
LONG_AUDIO_THRESHOLD_S = 360.0  # 6 minutes
CHUNK_WINDOW_S = 40.0  # aim for 30–45s windows
CHUNK_OVERLAP_S = 0.2
MEMORY_THRESHOLD_BYTES = 64 * 1024 * 1024
SLOW_DECODE_THRESHOLD_S = 8.0
NO_SPEECH_PROB_THRESHOLD = 0.65
STREAM_FRAME_S = 0.25
_JOB_SEMAPHORE = asyncio.Semaphore(1)
MAX_CHUNK_MULTIPLIER = 1.25
MAX_CHUNK_ABS_LIMIT = 512
MEMORY_ABORT_THRESHOLD_MB = 900

STT_MAX_RAM_MB = parse_stt_max_ram_mb()

# Backward-compatible cache symbols retained for tests and monkeypatching.
_FFMPEG_BIN_CACHE: Optional[str] = None
_FFMPEG_BIN_HAS_AAC: Optional[bool] = None


def _ffmpeg_supports_aac_decoder(ffmpeg_bin: str) -> bool:
    return ffmpeg_supports_aac_decoder(ffmpeg_bin)


def _resolve_ffmpeg_bin() -> str:
    global _FFMPEG_BIN_CACHE, _FFMPEG_BIN_HAS_AAC
    if _FFMPEG_BIN_CACHE:
        return _FFMPEG_BIN_CACHE

    for candidate in ffmpeg_candidates_from_env():
        ffmpeg_bin: Optional[str] = None
        if os.path.sep in candidate:
            candidate_path = Path(candidate)
            if candidate_path.exists():
                ffmpeg_bin = str(candidate_path)
        else:
            ffmpeg_bin = shutil.which(candidate)
        if not ffmpeg_bin:
            continue

        has_aac = bool(_ffmpeg_supports_aac_decoder(ffmpeg_bin))
        _FFMPEG_BIN_CACHE = ffmpeg_bin
        _FFMPEG_BIN_HAS_AAC = has_aac
        logger.info(
            "stt.ffmpeg.selected path=%s aac_decoder=%s",
            ffmpeg_bin,
            str(has_aac).lower(),
        )
        return ffmpeg_bin

    raise InferenceError(
        "ffmpeg executable not found; set STT_FFMPEG_BIN to an installed ffmpeg binary"
    )


# ---------------------------------------------------------------------------
# Utility dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PreprocessResult:
    stream: "BasePCMStream"
    sample_rate: int
    duration_in: float
    atempo_applied: bool
    silence_applied: bool
    silence_removed_ms: int
    cache_key: str
    cache_hit: bool
    source_hash: str
    source_path: Path
    duration_out: float = 0.0

    def update_from_stream(self) -> None:
        samples = self.stream.total_samples
        if samples > 0:
            self.duration_out = samples / float(self.sample_rate or SAMPLE_RATE or 1)
            expected = (
                self.duration_in / ATEMPO_FACTOR
                if self.atempo_applied
                else self.duration_in
            )
            self.silence_removed_ms = max(0, int((expected - self.duration_out) * 1000))
        else:
            self.duration_out = 0.0
            self.silence_removed_ms = 0


@dataclass
class TranscriptResult:
    text: str
    segments: List[Dict[str, Any]]
    chunks: List[Dict[str, Any]]
    duration_out: float
    model_spec: ModelSpec
    cache_hit: bool
    first_chunk_runtime: float
    aborted: bool = False
    abort_reason: Optional[str] = None
    # Fallback metadata
    is_fallback: bool = False
    fallback_provider: Optional[str] = None
    failure_context: Optional[str] = None


class STTJob:
    """Track STT job lifecycle to guarantee exactly-once finalization."""

    __slots__ = (
        "kind",
        "spans",
        "ram_guard",
        "state",
        "_finalized",
        "_cleanup_done",
        "_lock",
        "pre",
        "download",
        "temp_handle",
        "temp_path",
        "transcript",
        "status",
        "abort_reason",
        "chunks_done",
        "dur_done_s",
        "result_payload",
        "_error",
        "_rss_snapshot",
        "_rss_cleanup_before",
        "_rss_cleanup_after",
        "_ps_process",
    )

    def __init__(self, kind: str, spans: SpanRecorder, ram_guard: STTRAMGuard) -> None:
        self.kind = kind
        self.spans = spans
        self.ram_guard = ram_guard
        self.state = "running"
        self._finalized = False
        self._cleanup_done = False
        self._lock = asyncio.Lock()
        self.pre: Optional[PreprocessResult] = None
        self.download: Optional[DownloadedAudio] = None
        self.temp_handle: Optional[tempfile.NamedTemporaryFile] = None
        self.temp_path: Optional[Path] = None
        self.transcript: Optional[TranscriptResult] = None
        self.status = "ok"
        self.abort_reason: Optional[str] = None
        self.chunks_done = 0
        self.dur_done_s = 0.0
        self.result_payload: Any = None
        self._error: Optional[BaseException] = None
        self._ps_process = psutil.Process()
        try:
            self._rss_snapshot = self._ps_process.memory_info().rss
        except Exception:
            self._rss_snapshot = 0
        self._rss_cleanup_before = self._rss_snapshot
        self._rss_cleanup_after = self._rss_snapshot

    def register_pre(self, pre: PreprocessResult) -> None:
        self.pre = pre

    def register_download(self, download: Optional[DownloadedAudio]) -> None:
        self.download = download

    def register_temp(
        self,
        temp_handle: Optional[tempfile.NamedTemporaryFile],
        path: Optional[Path],
    ) -> None:
        self.temp_handle = temp_handle
        self.temp_path = path

    def enter_aborting(
        self, reason: str, chunks_done: int = 0, dur_done: float = 0.0
    ) -> None:
        if self.state == "running":
            self.state = "aborting"
        self.abort_reason = reason or self.abort_reason
        if chunks_done > self.chunks_done:
            self.chunks_done = chunks_done
        if dur_done > self.dur_done_s:
            self.dur_done_s = dur_done

    def register_transcript(self, transcript: TranscriptResult) -> None:
        self.transcript = transcript
        if transcript.aborted:
            self.enter_aborting(
                transcript.abort_reason or "abort",
                len(transcript.chunks),
                transcript.chunks[-1]["end"] if transcript.chunks else 0.0,
            )
        else:
            self.chunks_done = len(transcript.chunks)
            if transcript.chunks:
                self.dur_done_s = transcript.chunks[-1]["end"]

    async def finish_success(self, payload: Any) -> Any:
        async with self._lock:
            self.result_payload = payload
            if self.transcript and self.transcript.aborted:
                self.status = "partial"
            else:
                self.status = "ok"
            await self._finalize_locked()
        return payload

    async def finish_failure(self, exc: BaseException) -> None:
        async with self._lock:
            self.status = "fail"
            self._error = exc
            await self._finalize_locked()

    async def ensure_finalized(self) -> None:
        async with self._lock:
            if not self._finalized:
                if self.status not in ("ok", "partial", "fail"):
                    self.status = "fail"
                await self._finalize_locked()

    async def _finalize_locked(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        self.state = "finalized"
        if self.status == "partial":
            transcript_text = (
                (self.transcript.text if self.transcript else "")
                if self.transcript
                else ""
            )
            try:
                logger.info(
                    "stt.partial_ok chars=%s reason=%s",
                    len(transcript_text),
                    (self.abort_reason or ""),
                )
            except Exception:
                pass
        elif self.status == "ok":
            try:
                logger.info("stt.ok")
            except Exception:
                pass
        else:
            reason = self.abort_reason or (
                str(self._error)[:80] if self._error else "unknown"
            )
            try:
                logger.info("stt.fail reason=%s", reason)
            except Exception:
                pass
        await self._cleanup_resources()

    async def _cleanup_resources(self) -> None:
        if self._cleanup_done:
            return
        self._cleanup_done = True
        try:
            self._rss_cleanup_before = self._ps_process.memory_info().rss
        except Exception:
            self._rss_cleanup_before = 0

        stream = self.pre.stream if self.pre else None
        success = (
            bool(self.transcript)
            and not self.transcript.aborted
            and self.status == "ok"
        )
        if stream is not None:
            try:
                await stream.finalize(success=success)
            except Exception:
                logger.debug("⚠️ Failed to finalize stream after job", exc_info=True)

        removed_temp: Optional[Path] = None
        if self.temp_handle is not None:
            try:
                temp_name = self.temp_handle.name
                os.unlink(temp_name)
                removed_temp = Path(temp_name)
            except FileNotFoundError:
                pass
            except Exception:
                logger.debug("⚠️ Failed to remove temp attachment", exc_info=True)
            self.temp_handle = None
        if self.temp_path is not None and (
            removed_temp is None or self.temp_path != removed_temp
        ):
            try:
                os.unlink(self.temp_path)
            except FileNotFoundError:
                pass
            except Exception:
                logger.debug(
                    "⚠️ Failed to remove temp path %s", self.temp_path, exc_info=True
                )
            self.temp_path = None
        else:
            self.temp_path = None

        # Drop large references
        self.pre = None
        self.download = None
        self.transcript = None

        gc.collect()
        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except Exception:
            pass

        try:
            self._rss_cleanup_after = self._ps_process.memory_info().rss
        except Exception:
            self._rss_cleanup_after = self._rss_cleanup_before
        freed_bytes = max(0, self._rss_cleanup_before - self._rss_cleanup_after)
        freed_mb = freed_bytes / (1024 * 1024)
        try:
            logger.info("stt.cleanup freed_mb=%.1f", freed_mb)
        except Exception:
            pass

    async def close(self) -> None:
        await self.ensure_finalized()


class RAMGuardExceeded(RuntimeError):
    """Raised when STT processing exceeds configured RAM budget."""


class STTRAMGuard:
    """Process RSS watchdog for STT stages."""

    def __init__(self, limit_mb: Optional[int]) -> None:
        self.limit_mb = limit_mb
        self._process = psutil.Process()

    def check(self, stage: str) -> None:
        if self.limit_mb is None:
            return
        rss_bytes = self._process.memory_info().rss
        rss_mb = rss_bytes / (1024 * 1024)
        action = "continue"
        if rss_mb > self.limit_mb:
            action = "abort"
        logger.info(
            "stt.ram_guard stage=%s limit_mb=%s rss_mb=%.1f action=%s",
            stage,
            self.limit_mb,
            rss_mb,
            action,
        )
        if action == "abort":
            raise RAMGuardExceeded(
                f"STT memory limit exceeded at stage '{stage}': "
                f"{rss_mb:.1f}MB > {self.limit_mb}MB"
            )


class BasePCMStream:
    """Base streaming container that feeds PCM frames through a bounded queue."""

    bytes_per_sample = 2  # s16le

    def __init__(
        self,
        sample_rate: int,
        frame_samples: int,
        queue_depth: int = 2,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_samples = max(1, frame_samples)
        self.frame_bytes = self.frame_samples * self.bytes_per_sample
        self._queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue(maxsize=queue_depth)
        self._producer_task: Optional[asyncio.Task] = None
        self._monitor_tasks: List[asyncio.Task] = []
        self._error: Optional[BaseException] = None
        self._aborted = False
        self._total_samples = 0
        self._finalized = False

    @property
    def total_samples(self) -> int:
        return self._total_samples

    @property
    def duration_out(self) -> float:
        if self.sample_rate <= 0 or self._total_samples <= 0:
            return 0.0
        return self._total_samples / float(self.sample_rate)

    async def start(self) -> None:
        if self._producer_task is None:
            self._producer_task = asyncio.create_task(self._produce_wrapper())

    async def _produce_wrapper(self) -> None:
        try:
            await self._produce()
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            self._error = exc
        finally:
            self._signal_end()

    def _signal_end(self) -> None:
        """Best-effort enqueue of stream sentinel without blocking on full queue."""
        try:
            self._queue.put_nowait(None)
            return
        except asyncio.QueueFull:
            # Drop buffered frames to guarantee end-of-stream marker is visible.
            try:
                while True:
                    self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self._queue.put_nowait(None)
            except Exception:
                pass
        except Exception:
            pass

    async def iter_frames(self) -> AsyncIterator[bytes]:
        await self.start()
        while True:
            frame = await self._queue.get()
            if frame is None:
                break
            yield frame
        await self.wait_finished()

    async def wait_finished(self) -> None:
        if self._producer_task:
            try:
                await self._producer_task
            except asyncio.CancelledError:
                if not self._aborted:
                    raise
        for task in self._monitor_tasks:
            try:
                await task
            except asyncio.CancelledError:
                if not self._aborted:
                    raise
        if self._error and not isinstance(self._error, asyncio.CancelledError):
            raise self._error

    async def abort(self) -> None:
        self._aborted = True
        if self._producer_task and not self._producer_task.done():
            self._producer_task.cancel()
        self._signal_end()

    async def finalize(self, success: bool) -> None:
        if self._finalized:
            return
        self._finalized = True
        await self.wait_finished()

    async def _produce(self) -> None:
        raise NotImplementedError


class CachedPCMStream(BasePCMStream):
    """Stream PCM frames from cached on-disk artifact."""

    def __init__(
        self,
        path: Path,
        sample_rate: int,
        frame_samples: int,
        total_samples: int,
    ) -> None:
        super().__init__(sample_rate=sample_rate, frame_samples=frame_samples)
        self._path = path
        self._expected_samples = max(0, int(total_samples))

    async def _produce(self) -> None:
        if not self._path.exists():
            raise FileNotFoundError(f"Cached PCM missing: {self._path}")
        # Memory-map for efficient slicing without loading entire file.
        mem = np.memmap(
            str(self._path),
            dtype="<i2",
            mode="r",
            shape=(self._expected_samples,) if self._expected_samples else None,
        )
        try:
            idx = 0
            total_samples = mem.shape[0]
            while idx < total_samples:
                end = min(total_samples, idx + self.frame_samples)
                frame = mem[idx:end]
                bytes_chunk = frame.tobytes()
                self._total_samples += frame.shape[0]
                await self._queue.put(bytes_chunk)
                idx = end
        finally:
            del mem


class FFMpegPCMStream(BasePCMStream):
    """Stream PCM from a running ffmpeg process while persisting to cache."""

    def __init__(
        self,
        process: asyncio.subprocess.Process,
        temp_path: Path,
        cache_key: str,
        sample_rate: int,
        frame_samples: int,
        spans: SpanRecorder,
        pre_timeout: float,
    ) -> None:
        super().__init__(sample_rate=sample_rate, frame_samples=frame_samples)
        self._proc = process
        self._temp_path = temp_path
        self._cache_key = cache_key
        self._spans = spans
        self._pre_timeout = max(pre_timeout, 1.0)
        self._stderr_task: Optional[asyncio.Task] = None
        self._monitor_tasks.append(asyncio.create_task(self._monitor()))

    async def start(self) -> None:
        if self._stderr_task is None and self._proc.stderr is not None:
            self._stderr_task = asyncio.create_task(self._proc.stderr.read())
        await super().start()

    async def abort(self) -> None:
        await super().abort()
        if self._proc.returncode is None:
            try:
                self._proc.kill()
            except ProcessLookupError:
                pass
            except Exception:
                logger.debug("⚠️ Failed to kill ffmpeg during abort", exc_info=True)

    async def _monitor(self) -> None:
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=self._pre_timeout)
        except asyncio.TimeoutError:
            self._error = InferenceError("Audio preprocessing timed out")
            await self.abort()
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            self._error = exc
        finally:
            if self._stderr_task is not None:
                try:
                    stderr = await self._stderr_task
                except Exception:
                    stderr = b""
            else:
                stderr = b""
            if (
                self._error is None
                and self._proc.returncode not in (0, None)
                and not self._aborted
            ):
                err_txt = stderr.decode(errors="ignore").strip()
                self._error = InferenceError(
                    f"Audio preprocessing failed ({self._proc.returncode}): {err_txt}"
                )

    async def _produce(self) -> None:
        stdout = self._proc.stdout
        if stdout is None:
            raise RuntimeError("ffmpeg stdout pipe missing")
        self._temp_path.parent.mkdir(parents=True, exist_ok=True)
        with self._temp_path.open("wb") as fh:
            while True:
                chunk = await stdout.read(self.frame_bytes)
                if not chunk:
                    break
                self._total_samples += len(chunk) // self.bytes_per_sample
                fh.write(chunk)
                await self._queue.put(bytes(chunk))

    async def finalize(self, success: bool) -> None:
        if self._finalized:
            return
        await super().finalize(success)
        if success and not self._aborted and self._error is None:
            _store_pcm_cache_from_temp(
                self._cache_key,
                self._temp_path,
                self._total_samples,
                self.sample_rate,
            )
            self._spans.end("pre", ok=True, reason="ok")
        else:
            self._temp_path.unlink(missing_ok=True)
            reason = "abort" if self._aborted else "error"
            ok = success and self._error is None and not self._aborted
            self._spans.end("pre", ok=ok, reason=reason)


class NoSpeechDetected(RuntimeError):
    """Raised when VAD confirms no speech and the first chunk yields no tokens."""


class SpanRecorder:
    """Lightweight span recorder that logs stt.span breadcrumbs."""

    def __init__(self) -> None:
        self._start: Dict[str, float] = {}
        self.spans: Dict[str, int] = {}

    def start(self, stage: str) -> None:
        self._start[stage] = time.perf_counter()

    def end(self, stage: str, ok: bool = True, reason: str = "ok") -> None:
        start = self._start.pop(stage, None)
        if start is None:
            return
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        self.spans[stage] = elapsed_ms
        logger.info(
            "stt.span stage=%s ms=%s ok=%s reason=%s",
            stage,
            elapsed_ms,
            str(ok).lower(),
            reason,
        )


# ---------------------------------------------------------------------------
# Helpers: hashing, caching, probing
# ---------------------------------------------------------------------------


def _duration_bucket(duration_s: float) -> str:
    bucket = int(max(0, math.floor(duration_s / 5) * 5))
    return f"{bucket:03d}"


def _extract_hash_from_download(download: Optional[DownloadedAudio]) -> Optional[str]:
    if not download:
        return None
    key = download.download_key
    if "-" in key:
        return key.split("-", 1)[0]
    return key[:16]


def _source_hash(path: Path, download: Optional[DownloadedAudio]) -> str:
    download_hash = _extract_hash_from_download(download)
    if download_hash:
        return download_hash
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _preprocess_cache_key(
    source_hash: str, atempo: bool, silence: bool, duration_s: float
) -> str:
    flags = "".join(
        [
            PRE_PARAMS_VERSION,
            "a1" if atempo else "a0",
            "s1" if silence else "s0",
            _duration_bucket(duration_s),
        ]
    )
    return f"{source_hash}-{flags}"


def _pcm_cache_paths(cache_key: str) -> Tuple[Path, Path]:
    base = PCM_CACHE_DIR / cache_key
    return base.with_suffix(".pcm"), base.with_suffix(".json")


def _load_pcm_cache(cache_key: str) -> Optional[Tuple[Path, Dict[str, Any]]]:
    pcm_path, meta_path = _pcm_cache_paths(cache_key)
    if not pcm_path.exists() or not meta_path.exists():
        return None
    try:
        with meta_path.open("r", encoding="utf-8") as fh:
            meta = json.load(fh)
        return pcm_path, meta
    except Exception as exc:
        logger.warning(
            "⚠️ Failed to load preprocessed cache metadata %s: %s",
            meta_path,
            exc,
        )
        return None


def _store_pcm_cache_from_temp(
    cache_key: str,
    temp_path: Path,
    total_samples: int,
    sample_rate: int,
) -> None:
    pcm_path, meta_path = _pcm_cache_paths(cache_key)
    pcm_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(temp_path), pcm_path)
    except Exception as exc:
        logger.warning(
            "⚠️ Failed to move PCM cache temp=%s dest=%s err=%s",
            temp_path,
            pcm_path,
            exc,
        )
        try:
            shutil.copy2(str(temp_path), pcm_path)
        except Exception as copy_exc:
            logger.warning(
                "⚠️ Failed to copy PCM cache temp=%s dest=%s err=%s",
                temp_path,
                pcm_path,
                copy_exc,
            )
            temp_path.unlink(missing_ok=True)
            return
        finally:
            temp_path.unlink(missing_ok=True)

    meta = {
        "total_samples": int(total_samples),
        "sample_rate": int(sample_rate),
        "duration_out": (
            float(total_samples) / float(sample_rate) if sample_rate else 0.0
        ),
        "stored_at": time.time(),
    }
    try:
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(meta, fh)
        logger.info(
            "cache.store stage=audio key=%s samples=%s",
            cache_key[:12],
            total_samples,
        )
    except Exception as exc:
        logger.warning("⚠️ Failed to write PCM cache metadata %s: %s", meta_path, exc)


def _reset_stream_from_cache(pre: PreprocessResult) -> None:
    cached = _load_pcm_cache(pre.cache_key)
    if not cached:
        raise InferenceError("PCM cache unavailable for STT retry")
    pcm_path, meta = cached
    total_samples = int(meta.get("total_samples") or 0)
    duration_out = float(meta.get("duration_out") or 0.0)
    frame_samples = int(STREAM_FRAME_S * pre.sample_rate)
    pre.stream = CachedPCMStream(
        path=pcm_path,
        sample_rate=pre.sample_rate,
        frame_samples=frame_samples,
        total_samples=total_samples,
    )
    pre.cache_hit = True
    pre.duration_out = duration_out
    expected_after_atempo = (
        pre.duration_in / ATEMPO_FACTOR if pre.atempo_applied else pre.duration_in
    )
    pre.silence_removed_ms = max(0, int((expected_after_atempo - duration_out) * 1000))


def _transcript_cache_key(
    audio_cache_key: str, spec: ModelSpec, vad_enabled: bool = True
) -> str:
    base = f"{audio_cache_key}|{spec.size}|{spec.compute_type}|beam1|temp0|vad{int(vad_enabled)}"
    return hashlib.sha256(base.encode()).hexdigest()[:24]


def _transcript_cache_path(cache_key: str) -> Path:
    return TRANSCRIPT_CACHE_DIR / f"{cache_key}.json"


def _load_transcript_cache(cache_key: str) -> Optional[TranscriptResult]:
    path = _transcript_cache_path(cache_key)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:
        logger.warning("⚠️ Failed to load transcript cache %s: %s", path, exc)
        return None

    try:
        spec = ModelSpec(
            data["model"]["size"],
            data["model"]["compute_type"],
        )
        segments = data.get("segments", [])
        chunks = data.get("chunks", [])
        return TranscriptResult(
            text=data.get("text", ""),
            segments=segments,
            chunks=chunks,
            duration_out=float(data.get("duration_out", 0.0)),
            model_spec=spec,
            cache_hit=True,
            first_chunk_runtime=float(data.get("first_chunk_runtime", 0.0)),
        )
    except Exception as exc:
        logger.warning("⚠️ Invalid transcript cache %s: %s", path, exc)
        return None


def _store_transcript_cache(cache_key: str, result: TranscriptResult) -> None:
    path = _transcript_cache_path(cache_key)
    payload = {
        "text": result.text,
        "segments": result.segments,
        "chunks": result.chunks,
        "duration_out": result.duration_out,
        "first_chunk_runtime": result.first_chunk_runtime,
        "model": {
            "size": result.model_spec.size,
            "compute_type": result.model_spec.compute_type,
        },
        "cached_at": time.time(),
    }
    try:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh)
    except Exception as exc:
        logger.warning("⚠️ Failed to store transcript cache %s: %s", path, exc)


async def _ffprobe(path: Path) -> Tuple[float, int, int]:
    """Probe audio file for duration, sample rate, channels. Falls back to ffmpeg if ffprobe unavailable. [REH]"""
    ffprobe_bin = shutil.which("ffprobe")
    if ffprobe_bin:
        cmd = [
            ffprobe_bin,
            "-v",
            "error",
            "-print_format",
            "json",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels",
            "-show_entries",
            "format=duration",
            str(path),
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise InferenceError(
                f"ffprobe failed ({proc.returncode}): {stderr.decode(errors='ignore')}"
            )

        try:
            info = json.loads(stdout.decode())
            duration = float(info.get("format", {}).get("duration") or 0.0)
            stream = (info.get("streams") or [{}])[0]
            sample_rate = int(stream.get("sample_rate") or SAMPLE_RATE)
            channels = int(stream.get("channels") or 1)
        except Exception as exc:
            raise InferenceError(f"Failed to parse ffprobe output: {exc}") from exc
        return duration, sample_rate, channels

    # Fallback: use ffmpeg -i to probe (parses stderr output) [REH]
    ffmpeg_bin = _resolve_ffmpeg_bin()

    cmd = [ffmpeg_bin, "-i", str(path), "-hide_banner", "-f", "null", "-"]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await proc.communicate()
    stderr_text = stderr.decode(errors="ignore")

    # Parse duration from "Duration: HH:MM:SS.ms" or "Duration: N/A"
    duration = 0.0
    dur_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", stderr_text)
    if dur_match:
        h, m, s = dur_match.groups()
        duration = int(h) * 3600 + int(m) * 60 + float(s)

    # Parse sample rate from "44100 Hz" or similar
    sample_rate = SAMPLE_RATE
    sr_match = re.search(r"(\d+)\s*Hz", stderr_text)
    if sr_match:
        sample_rate = int(sr_match.group(1))

    # Parse channels from "stereo", "mono", or "N channels"
    channels = 1
    if "stereo" in stderr_text.lower():
        channels = 2
    elif ch_match := re.search(r"(\d+)\s*channels?", stderr_text, re.IGNORECASE):
        channels = int(ch_match.group(1))

    return duration, sample_rate, channels


# ---------------------------------------------------------------------------
# Preprocess stage
# ---------------------------------------------------------------------------


def _is_codec_error(error: Exception) -> bool:
    """Check if error is related to audio codec decoding issues. [REH]"""
    error_str = str(error).lower()
    codec_patterns = [
        "decoder",
        "codec",
        "aac",
        "h264",
        "hevc",
        "avc",
        "unsupported",
        "invalid data",
        "truncated",
        "corrupted",
    ]
    return any(pattern in error_str for pattern in codec_patterns)


async def _extract_audio_to_wav_forced(
    source_path: Path,
    target_path: Path,
    timeout: float = 120.0,
) -> bool:
    """
    Force-extract audio to WAV using ffmpeg with explicit codec handling.

    This is a Tier 2 extraction retry that uses a two-step approach:
    1. Extract audio stream to WAV with codec auto-detection
    2. Use -acodec pcm_s16le to force PCM output

    This retries extraction with the configured STT ffmpeg binary. [REH]
    """
    try:
        ffmpeg_bin = _resolve_ffmpeg_bin()
    except Exception:
        logger.warning("ffmpeg not found for forced extraction")
        return False

    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source_path),
        "-vn",  # No video
        "-acodec",
        "pcm_s16le",  # Force PCM output
        "-ar",
        str(SAMPLE_RATE),
        "-ac",
        "1",
        "-y",  # Overwrite
        str(target_path),
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)

        if (
            proc.returncode == 0
            and target_path.exists()
            and target_path.stat().st_size > 0
        ):
            logger.info(
                "pre.extract_forced_success src=%s wav=%s bytes=%d",
                source_path.name,
                target_path.name,
                target_path.stat().st_size,
            )
            return True

        error_msg = stderr.decode(errors="ignore") if stderr else "unknown"
        logger.warning(
            "pre.extract_forced_failed src=%s returncode=%d error=%s",
            source_path.name,
            proc.returncode,
            error_msg[:200],
        )
        return False

    except asyncio.TimeoutError:
        logger.warning("pre.extract_forced_timeout src=%s", source_path.name)
        try:
            proc.kill()
        except Exception:
            pass
        return False
    except Exception as exc:
        logger.warning(
            "pre.extract_forced_error src=%s error=%s",
            source_path.name,
            exc,
        )
        return False


async def _preprocess_audio_with_retry(
    source_path: Path,
    spans: SpanRecorder,
    download: Optional[DownloadedAudio],
    voice_note: bool,
    ram_guard: STTRAMGuard,
) -> PreprocessResult:
    """
    Preprocess audio with Tier 2 extraction retry for codec errors. [REH]

    Tier 1: Direct preprocessing (existing logic)
    Tier 2: If codec error, force-extract to WAV first, then preprocess

    This ensures that AAC decoder issues don't cause complete STT failure.
    """
    try:
        # Tier 1: Try normal preprocessing
        return await _preprocess_audio(
            source_path=source_path,
            spans=spans,
            download=download,
            voice_note=voice_note,
            ram_guard=ram_guard,
        )
    except Exception as primary_error:
        # Check if this is a codec-related error that warrants extraction retry
        if not _is_codec_error(primary_error):
            # Not a codec error - re-raise immediately
            raise

        logger.info(
            "pre.codec_error_detected error=%s attempting_tier2_retry",
            str(primary_error)[:150],
        )

        # Tier 2: Force-extract to WAV and retry
        wav_path = None
        try:
            # Create temp WAV file for forced extraction
            wav_handle = tempfile.NamedTemporaryFile(
                delete=False, suffix=".wav", dir=str(PCM_CACHE_DIR)
            )
            wav_path = Path(wav_handle.name)
            wav_handle.close()

            # Force-extract to WAV
            spans.start("pre_extract_forced")
            success = await _extract_audio_to_wav_forced(
                source_path=source_path,
                target_path=wav_path,
                timeout=120.0,
            )
            spans.end("pre_extract_forced", ok=success)

            if not success:
                # Extraction failed - this is unrecoverable
                raise InferenceError(
                    f"Failed to extract audio from {source_path.name}: "
                    f"forced WAV conversion failed"
                ) from primary_error

            # Retry preprocessing with the WAV file
            logger.info("pre.tier2_retry wav=%s", wav_path.name)
            result = await _preprocess_audio(
                source_path=wav_path,
                spans=spans,
                download=download,  # Still track original download info
                voice_note=voice_note,
                ram_guard=ram_guard,
            )

            logger.info("pre.tier2_success wav=%s", wav_path.name)
            return result

        finally:
            # Clean up the temporary WAV file
            if wav_path and wav_path.exists():
                try:
                    wav_path.unlink()
                except Exception:
                    pass


async def _preprocess_audio(
    source_path: Path,
    spans: SpanRecorder,
    download: Optional[DownloadedAudio],
    voice_note: bool,
    ram_guard: STTRAMGuard,
) -> PreprocessResult:
    spans.start("pre")
    duration_in, sr, channels = await _ffprobe(source_path)
    logger.info("pre.probe dur_in=%.2f sr=%s ch=%s", duration_in, sr, channels)
    ram_guard.check("ffprobe")

    silence_applied = not voice_note and duration_in > SHORT_CLIP_S
    atempo_applied = not voice_note and duration_in >= ATEMPO_THRESHOLD_S

    source_hash = _source_hash(source_path, download)
    cache_key = _preprocess_cache_key(
        source_hash, atempo_applied, silence_applied, duration_in
    )

    cached_pcm = _load_pcm_cache(cache_key)
    if cached_pcm is not None:
        pcm_path, meta = cached_pcm
        total_samples = int(meta.get("total_samples") or 0)
        duration_out = float(meta.get("duration_out") or 0.0)
        frame_samples = int(STREAM_FRAME_S * SAMPLE_RATE)
        stream = CachedPCMStream(
            path=pcm_path,
            sample_rate=SAMPLE_RATE,
            frame_samples=frame_samples,
            total_samples=total_samples,
        )
        expected_after_atempo = (
            duration_in / ATEMPO_FACTOR if atempo_applied else duration_in
        )
        silence_removed_ms = max(0, int((expected_after_atempo - duration_out) * 1000))
        pre_result = PreprocessResult(
            stream=stream,
            sample_rate=SAMPLE_RATE,
            duration_in=duration_in,
            atempo_applied=atempo_applied,
            silence_applied=silence_applied,
            silence_removed_ms=silence_removed_ms,
            cache_key=cache_key,
            cache_hit=True,
            source_hash=source_hash,
            source_path=source_path,
            duration_out=duration_out,
        )
        spans.end("pre", ok=True, reason="cache")
        ram_guard.check("pre-cache")
        logger.info(
            "cache.hit stage=audio key=%s samples=%s",
            cache_key[:12],
            total_samples,
        )
        return pre_result

    filter_chain: List[str] = []
    if silence_applied:
        filter_chain.append(
            "silenceremove=start_periods=1:start_duration=0.3:"
            "start_threshold=-40dB:stop_periods=-1:stop_duration=0.3:stop_threshold=-40dB"
        )
    if atempo_applied:
        filter_chain.append(f"atempo={ATEMPO_FACTOR}")
    filter_chain.append("aresample=async=1:first_pts=0")
    filters = ",".join(filter_chain)

    ffmpeg_bin = _resolve_ffmpeg_bin()
    aac_decoder_available = _FFMPEG_BIN_HAS_AAC
    if aac_decoder_available is None:
        aac_decoder_available = ffmpeg_bin_has_aac()
    if aac_decoder_available is False and source_path.suffix.lower() in {
        ".mp4",
        ".m4a",
        ".aac",
    }:
        logger.warning(
            "stt.ffmpeg.aac_missing path=%s source=%s",
            ffmpeg_bin,
            source_path.name,
        )

    ffmpeg_cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source_path),
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-acodec",
        "pcm_s16le",
        "-threads",
        "1",
        "-af",
        filters,
        "-f",
        "s16le",
        "pipe:1",
    ]

    proc = await asyncio.create_subprocess_exec(
        *ffmpeg_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    pre_budget = min(45.0, max(6.0, duration_in * 0.6 + 3.0))
    temp_handle = tempfile.NamedTemporaryFile(
        delete=False, suffix=".pcm", dir=str(PCM_CACHE_DIR)
    )
    temp_path = Path(temp_handle.name)
    temp_handle.close()
    frame_samples = int(STREAM_FRAME_S * SAMPLE_RATE)
    stream = FFMpegPCMStream(
        process=proc,
        temp_path=temp_path,
        cache_key=cache_key,
        sample_rate=SAMPLE_RATE,
        frame_samples=frame_samples,
        spans=spans,
        pre_timeout=pre_budget,
    )
    try:
        ram_guard.check("pre-ffmpeg_start")
    except RAMGuardExceeded:
        await stream.abort()
        raise
    return PreprocessResult(
        stream=stream,
        sample_rate=SAMPLE_RATE,
        duration_in=duration_in,
        atempo_applied=atempo_applied,
        silence_applied=silence_applied,
        silence_removed_ms=0,
        cache_key=cache_key,
        cache_hit=False,
        source_hash=source_hash,
        source_path=source_path,
    )


# ---------------------------------------------------------------------------
# Whisper transcription stage
# ---------------------------------------------------------------------------


def _pop_samples(frames: deque[np.ndarray], sample_count: int) -> np.ndarray:
    """Remove up to sample_count samples from the frame deque."""
    if sample_count <= 0:
        return np.empty((0,), dtype=np.float32)
    parts: List[np.ndarray] = []
    remaining = sample_count
    while frames and remaining > 0:
        head = frames[0]
        if head.shape[0] <= remaining:
            parts.append(head)
            frames.popleft()
            remaining -= head.shape[0]
        else:
            parts.append(head[:remaining])
            frames[0] = head[remaining:]
            remaining = 0
    if not parts:
        return np.empty((0,), dtype=np.float32)
    if len(parts) == 1:
        return parts[0]
    return np.concatenate(parts)


def _drain_frames(frames: deque[np.ndarray]) -> np.ndarray:
    """Drain and concatenate all remaining frames."""
    if not frames:
        return np.empty((0,), dtype=np.float32)
    if len(frames) == 1:
        return frames.popleft()
    parts = list(frames)
    frames.clear()
    return np.concatenate(parts)


def _segments_to_dict(segments: List[Any], offset: float) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for seg in segments:
        results.append(
            {
                "start": float(seg.start + offset),
                "end": float(seg.end + offset),
                "text": seg.text.strip(),
            }
        )
    return results


def _join_segments(segments: List[Dict[str, Any]]) -> str:
    return " ".join(seg["text"] for seg in segments).strip()


async def _run_whisper(
    pre: PreprocessResult,
    spans: SpanRecorder,
    initial_spec: ModelSpec,
    ram_guard: STTRAMGuard,
    job: Optional[STTJob] = None,
) -> TranscriptResult:
    spec = initial_spec
    attempted_slow_downgrade = False

    while True:
        model = await stt_manager.ensure_model(spec)
        try:
            transcript = await _transcribe_with_model(
                model=model,
                spec=spec,
                pre=pre,
                spans=spans,
                ram_guard=ram_guard,
                job=job,
            )
        except Exception:
            try:
                await pre.stream.finalize(success=False)
            except Exception:
                logger.debug("⚠️ Failed to finalize stream after error", exc_info=True)
            raise

        try:
            await pre.stream.finalize(success=not transcript.aborted)
        except Exception as exc:
            if isinstance(
                exc, InferenceError
            ) and "Audio preprocessing timed out" in str(exc):
                logger.debug("⚠️ Stream finalization failed: %s", exc)
            else:
                logger.debug("⚠️ Stream finalization failed", exc_info=True)

        pre.update_from_stream()

        if (
            not attempted_slow_downgrade
            and transcript.first_chunk_runtime > SLOW_DECODE_THRESHOLD_S
        ):
            next_spec = stt_manager.downgrade_spec(spec)
            if next_spec and next_spec != spec:
                logger.info(
                    "whisper.model_downgrade from=%s to=%s reason=slow_decode",
                    spec.size,
                    next_spec.size,
                )
                spec = next_spec
                attempted_slow_downgrade = True
                try:
                    _reset_stream_from_cache(pre)
                except InferenceError as exc:
                    # PCM cache is unavailable for retry; keep the existing transcript
                    # and treat this as a soft failure rather than aborting STT. [REH]
                    try:
                        logger.info(
                            "stt.fail reason=PCM cache unavailable for STT retry; using existing transcript (%s)",
                            str(exc),
                        )
                    except Exception:
                        pass
                    if job:
                        job.register_transcript(transcript)
                    return transcript
                continue
        if job:
            job.register_transcript(transcript)
        return transcript


async def _transcribe_with_model(
    model: "WhisperModel",
    spec: ModelSpec,
    pre: PreprocessResult,
    spans: SpanRecorder,
    ram_guard: STTRAMGuard,
    job: Optional[STTJob] = None,
) -> TranscriptResult:
    cache_key = _transcript_cache_key(pre.cache_key, spec, vad_enabled=True)
    cached = _load_transcript_cache(cache_key)
    if cached:
        spans.spans["whisper"] = 0
        logger.info("stt.span stage=whisper ms=0 ok=true reason=cache")
        logger.info("cache.hit stage=transcript key=%s", cache_key[:12])
        return cached

    spans.start("whisper")
    chunk_records: List[Dict[str, Any]] = []
    segments_accum: List[Dict[str, Any]] = []
    first_chunk_runtime = 0.0
    aborted_reason: Optional[str] = None
    sample_rate = pre.sample_rate
    chunk_samples = int(CHUNK_WINDOW_S * sample_rate)
    overlap_samples = int(CHUNK_OVERLAP_S * sample_rate)
    frames: deque[np.ndarray] = deque()
    tail: Optional[np.ndarray] = None
    samples_buffered = 0
    chunk_idx = 0
    start_sample = 0
    whisper_budget = min(180.0, max(20.0, pre.duration_in * 2.5 + 10.0))
    start_time = time.perf_counter()
    estimated_chunks = (
        pre.duration_out / max(CHUNK_WINDOW_S - CHUNK_OVERLAP_S, 1e-3)
        if pre.duration_out > 0
        else 1.0
    )
    dynamic_limit = int(math.ceil(estimated_chunks * MAX_CHUNK_MULTIPLIER) + 1)
    max_chunks = max(1, min(dynamic_limit, MAX_CHUNK_ABS_LIMIT))
    process = psutil.Process()
    confirm_memory_abort = pre.duration_in <= 90.0
    pending_memory_abort = False
    memory_probe_started = 0.0
    MEMORY_CONFIRM_DELAY = 0.3

    async def _should_abort_for_memory(rss_mb: float) -> bool:
        nonlocal pending_memory_abort, memory_probe_started
        if rss_mb < MEMORY_ABORT_THRESHOLD_MB:
            if pending_memory_abort:
                try:
                    logger.info(
                        "stt.guard.memory rss_mb=%.1f action=continue",
                        rss_mb,
                    )
                except Exception:
                    pass
            pending_memory_abort = False
            return False

        if not confirm_memory_abort:
            try:
                logger.info(
                    "stt.guard.memory rss_mb=%.1f action=abort_partial",
                    rss_mb,
                )
            except Exception:
                pass
            return True

        if not pending_memory_abort:
            pending_memory_abort = True
            memory_probe_started = time.perf_counter()
            try:
                logger.info(
                    "stt.guard.memory rss_mb=%.1f action=confirm_abort",
                    rss_mb,
                )
            except Exception:
                pass
            await asyncio.sleep(MEMORY_CONFIRM_DELAY)
            return False

        elapsed = time.perf_counter() - memory_probe_started
        if elapsed < MEMORY_CONFIRM_DELAY:
            await asyncio.sleep(max(0.0, MEMORY_CONFIRM_DELAY - elapsed))
        try:
            rss_second = process.memory_info().rss / (1024 * 1024)
        except Exception:
            rss_second = rss_mb

        if rss_second >= MEMORY_ABORT_THRESHOLD_MB:
            try:
                logger.info(
                    "stt.guard.memory rss_mb=%.1f action=abort_partial",
                    rss_second,
                )
            except Exception:
                pass
            return True

        try:
            logger.info(
                "stt.guard.memory rss_mb=%.1f action=continue",
                rss_second,
            )
        except Exception:
            pass
        pending_memory_abort = False
        return False

    async def _decode_chunk(chunk_audio: np.ndarray) -> Tuple[List[Any], Any, float]:
        chunk_begin = time.perf_counter()

        def _run() -> Tuple[List[Any], Any]:
            seg_iter, info = model.transcribe(
                chunk_audio,
                beam_size=1,
                best_of=1,
                temperature=0.0,
                vad_filter=True,
                word_timestamps=False,
                task="transcribe",
                language=None,
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=NO_SPEECH_PROB_THRESHOLD,
                condition_on_previous_text=False,
                initial_prompt=None,
            )
            return list(seg_iter), info

        segments, info = await asyncio.to_thread(_run)
        runtime = time.perf_counter() - chunk_begin
        return segments, info, runtime

    try:
        async for frame_bytes in pre.stream.iter_frames():
            if tail is not None:
                if tail.size:
                    frames.append(tail)
                    samples_buffered += tail.shape[0]
                tail = None

            frame = np.frombuffer(frame_bytes, dtype="<i2").astype(np.float32) / 32768.0
            if frame.size == 0:
                continue

            frames.append(frame)
            samples_buffered += frame.shape[0]

            while samples_buffered >= chunk_samples:
                remaining_ms = int(
                    max(
                        0.0,
                        (whisper_budget - (time.perf_counter() - start_time)) * 1000,
                    )
                )
                try:
                    logger.info(
                        "stt.budget remaining_ms=%s next_chunk_idx=%s",
                        remaining_ms,
                        chunk_idx,
                    )
                except Exception:
                    pass
                if remaining_ms <= 0:
                    aborted_reason = "time_budget"
                    break
                if chunk_idx >= max_chunks:
                    aborted_reason = "chunk_limit"
                    break
                chunk_audio = _pop_samples(frames, chunk_samples)
                if chunk_audio.size == 0:
                    break
                samples_buffered -= chunk_audio.shape[0]
                chunk_start_s = start_sample / sample_rate
                chunk_end_s = chunk_start_s + chunk_audio.shape[0] / sample_rate
                segments, info, runtime = await _decode_chunk(chunk_audio)
                if chunk_idx == 0:
                    first_chunk_runtime = runtime
                    if not segments and info.no_speech_prob >= NO_SPEECH_PROB_THRESHOLD:
                        spans.end("whisper", ok=False, reason="no_speech")
                        logger.info("stt.no_speech_fast_exit")
                        raise InferenceError(
                            f"No speech detected (prob={info.no_speech_prob:.3f})"
                        )

                logger.info(
                    "whisper.chunk idx=%s len_s=%.2f",
                    chunk_idx,
                    chunk_end_s - chunk_start_s,
                )
                seg_dicts = _segments_to_dict(segments, offset=chunk_start_s)
                segments_accum.extend(seg_dicts)
                chunk_records.append(
                    {
                        "idx": chunk_idx,
                        "start": chunk_start_s,
                        "end": chunk_end_s,
                        "segments": seg_dicts,
                    }
                )
                chunk_idx += 1
                start_sample += chunk_audio.shape[0] - overlap_samples
                tail = (
                    chunk_audio[-overlap_samples:].copy()
                    if overlap_samples > 0 and chunk_audio.shape[0] > overlap_samples
                    else None
                )
                ram_guard.check("whisper-chunk")
                try:
                    rss_mb = process.memory_info().rss / (1024 * 1024)
                except Exception:
                    rss_mb = 0.0
                if rss_mb > 0.0 and await _should_abort_for_memory(rss_mb):
                    frames.clear()
                    samples_buffered = 0
                    tail = None
                    aborted_reason = "memory_guard"
                    if job:
                        current_end = (
                            chunk_records[-1]["end"] if chunk_records else chunk_end_s
                        )
                        job.enter_aborting(
                            "memory_guard", len(chunk_records), current_end
                        )
                    gc.collect()
                    break
                if (time.perf_counter() - start_time) > whisper_budget:
                    aborted_reason = "time_budget"
                    break
                await asyncio.sleep(0)
            if aborted_reason:
                break

        if not aborted_reason and (chunk_idx == 0 or frames):
            remaining_ms = int(
                max(0.0, (whisper_budget - (time.perf_counter() - start_time)) * 1000)
            )
            try:
                logger.info(
                    "stt.budget remaining_ms=%s next_chunk_idx=%s",
                    remaining_ms,
                    chunk_idx,
                )
            except Exception:
                pass
            if remaining_ms <= 0:
                aborted_reason = "time_budget"
            elif chunk_idx >= max_chunks:
                aborted_reason = "chunk_limit"
            else:
                remainder = _drain_frames(frames)
                if chunk_idx == 0 and tail is not None and remainder.size == 0:
                    remainder = tail
                tail = None
                if remainder.size > 0:
                    chunk_start_s = start_sample / sample_rate
                    chunk_end_s = chunk_start_s + remainder.shape[0] / sample_rate
                    segments, info, runtime = await _decode_chunk(remainder)
                    if chunk_idx == 0:
                        first_chunk_runtime = runtime
                        if (
                            not segments
                            and info.no_speech_prob >= NO_SPEECH_PROB_THRESHOLD
                        ):
                            spans.end("whisper", ok=False, reason="no_speech")
                            logger.info("stt.no_speech_fast_exit")
                            raise InferenceError(
                                f"No speech detected (prob={info.no_speech_prob:.3f})"
                            )
                    logger.info(
                        "whisper.chunk idx=%s len_s=%.2f",
                        chunk_idx,
                        chunk_end_s - chunk_start_s,
                    )
                    seg_dicts = _segments_to_dict(segments, offset=chunk_start_s)
                    segments_accum.extend(seg_dicts)
                    chunk_records.append(
                        {
                            "idx": chunk_idx,
                            "start": chunk_start_s,
                            "end": chunk_end_s,
                            "segments": seg_dicts,
                        }
                    )
                    chunk_idx += 1
                    ram_guard.check("whisper-chunk")
                    try:
                        rss_mb = process.memory_info().rss / (1024 * 1024)
                    except Exception:
                        rss_mb = 0.0
                    if rss_mb > 0.0 and await _should_abort_for_memory(rss_mb):
                        frames.clear()
                        tail = None
                        aborted_reason = "memory_guard"
                        if job:
                            current_end = (
                                chunk_records[-1]["end"]
                                if chunk_records
                                else chunk_end_s
                            )
                            job.enter_aborting(
                                "memory_guard", len(chunk_records), current_end
                            )
                        gc.collect()
                    if (
                        aborted_reason is None
                        and (time.perf_counter() - start_time) > whisper_budget
                    ):
                        aborted_reason = "time_budget"
    except InferenceError:
        spans.end("whisper", ok=False, reason="error")
        raise
    except RAMGuardExceeded:
        await pre.stream.abort()
        spans.end("whisper", ok=False, reason="ram_guard")
        logger.info("stt.fail reason=ram_guard")
        raise
    except Exception as exc:
        logger.error(
            "whisper.transcribe_failed err=%s spec=%s dur=%.2f",
            exc,
            spec.size,
            pre.duration_in,
            exc_info=True,
        )
        spans.end("whisper", ok=False, reason="error")
        raise InferenceError(f"Transcription failed: {exc}") from exc

    if aborted_reason:
        try:
            await pre.stream.abort()
        except Exception:
            logger.debug("⚠️ Stream abort failed", exc_info=True)
    if aborted_reason:
        spans.end("whisper", ok=False, reason=aborted_reason)
    else:
        spans.end("whisper", ok=True, reason="ok")
    ram_guard.check("whisper-complete")
    text = _join_segments(segments_accum)
    if aborted_reason:
        chunks_done = len(chunk_records)
        dur_done_s = chunk_records[-1]["end"] if chunk_records else 0.0
        if job:
            job.enter_aborting(aborted_reason, chunks_done, dur_done_s)
        try:
            logger.info(
                "stt.abort reason=%s chunks_done=%s dur_done_s=%.2f",
                aborted_reason,
                chunks_done,
                dur_done_s,
            )
        except Exception:
            pass
    result = TranscriptResult(
        text=text,
        segments=segments_accum,
        chunks=chunk_records,
        duration_out=pre.stream.duration_out,
        model_spec=spec,
        cache_hit=False,
        first_chunk_runtime=first_chunk_runtime,
        aborted=aborted_reason is not None,
        abort_reason=aborted_reason,
    )
    if not aborted_reason:
        _store_transcript_cache(cache_key, result)
        logger.info(
            "cache.store stage=transcript key=%s segments=%s",
            cache_key[:12],
            len(segments_accum),
        )
    return result


# ---------------------------------------------------------------------------
# Stitch / Summary helpers
# ---------------------------------------------------------------------------


def _format_spans(spans: Dict[str, int]) -> str:
    ordered = ["yt-dlp", "pre", "whisper", "stitch"]
    parts = []
    for key in ordered:
        if key in spans:
            parts.append(f"{key}:{spans[key]}")
    for key, val in spans.items():
        if key not in ordered:
            parts.append(f"{key}:{val}")
    return "{%s}" % ",".join(parts)


def _log_summary(
    spans: SpanRecorder,
    pre: PreprocessResult,
    transcript: TranscriptResult,
    cache_hit: bool,
) -> None:
    logger.info(
        "stt.summary dur_in=%.2f dur_out=%.2f spans_ms=%s model=%s compute=%s cpu_threads=%s cache_hit=%s",
        pre.duration_in,
        pre.duration_out,
        _format_spans(spans.spans),
        transcript.model_spec.size,
        transcript.model_spec.compute_type,
        stt_manager.cpu_threads,
        str(cache_hit).lower(),
    )


# ---------------------------------------------------------------------------
# Attachment helpers
# ---------------------------------------------------------------------------


async def _ensure_local_audio(
    audio: Union[Path, "discord.Attachment"],
) -> Tuple[Path, Optional[tempfile.NamedTemporaryFile], bool]:
    if isinstance(audio, Path):
        return audio, None, False

    suffix = os.path.splitext(getattr(audio, "filename", "") or "")[1] or ".audio"
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_path = Path(temp.name)
    temp.close()
    await audio.save(temp_path)
    return temp_path, temp, True


def _is_voice_note(attachment: Optional["discord.Attachment"]) -> bool:
    if attachment is None:
        return False
    if getattr(attachment, "voice_message", False) or getattr(
        attachment, "is_voice_message", False
    ):
        return True
    content_type = (getattr(attachment, "content_type", "") or "").lower()
    if content_type in {
        "application/ogg",
        "audio/ogg",
        "audio/webm",
        "audio/webm; codecs=opus",
        "video/webm; codecs=opus",
    }:
        return True
    filename = (getattr(attachment, "filename", "") or "").lower()
    return filename.endswith((".ogg", ".opus"))


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


async def hear_infer(audio: Union[Path, "discord.Attachment"]) -> str:
    """
    Transcribe an attachment or local audio file.
    """
    spans = SpanRecorder()
    ram_guard = STTRAMGuard(STT_MAX_RAM_MB)
    attachment = audio if not isinstance(audio, Path) else None
    job = STTJob(kind="attachment", spans=spans, ram_guard=ram_guard)

    try:
        async with _JOB_SEMAPHORE:
            local_path, temp_handle, created_temp = await _ensure_local_audio(audio)
            job.register_temp(temp_handle, local_path if created_temp else None)
            voice_note = _is_voice_note(attachment) if attachment is not None else False
            pre = await _preprocess_audio_with_retry(
                source_path=local_path,
                spans=spans,
                download=None,
                voice_note=voice_note,
                ram_guard=ram_guard,
            )
            job.register_pre(pre)
            ram_guard.check("pre-stage")

            spec = stt_manager.default_spec
            if pre.duration_in > 120:
                downgraded = stt_manager.downgrade_spec(spec)
                if downgraded:
                    logger.info(
                        "whisper.model_downgrade from=%s to=%s reason=long_audio",
                        spec.size,
                        downgraded.size,
                    )
                    spec = downgraded

            transcript = await _run_whisper_with_fallback(
                pre, spans, spec, ram_guard, job=job
            )

            spans.start("stitch")
            result_text = transcript.text
            spans.end("stitch", ok=True)
            _log_summary(spans, pre, transcript, cache_hit=transcript.cache_hit)
            return await job.finish_success(result_text)
    except RAMGuardExceeded as exc:
        if job.pre and job.pre.stream:
            try:
                await job.pre.stream.abort()
            except Exception:
                logger.debug(
                    "⚠️ Stream abort failed after RAM guard trigger", exc_info=True
                )
        await job.finish_failure(exc)
        raise InferenceError(str(exc)) from exc
    except Exception as exc:
        if job.pre and job.pre.stream:
            try:
                await job.pre.stream.abort()
            except Exception:
                logger.debug("⚠️ Stream abort failed after error", exc_info=True)
        await job.finish_failure(exc)
        raise
    finally:
        await job.close()


async def hear_infer_from_url(url: str, force_refresh: bool = False) -> Dict[str, Any]:
    """
    Transcribe audio fetched via yt-dlp for the given URL.
    """
    # Log the exact URL being processed for STT job identity tracking [REH]
    logger.info(
        "stt.job.start kind=url url=%s force_refresh=%s",
        url[:120] if url else "none",
        force_refresh,
    )

    spans = SpanRecorder()
    ram_guard = STTRAMGuard(STT_MAX_RAM_MB)
    job = STTJob(kind="url", spans=spans, ram_guard=ram_guard)
    download: Optional[DownloadedAudio] = None
    try:
        async with _JOB_SEMAPHORE:
            runtime_compat = load_stt_runtime_compat()
            # YouTube transcript-first: try caption tracks before yt-dlp/ffmpeg/whisper.
            # On failure/unavailable captions we fail open to the existing STT pipeline.
            if runtime_compat.youtube_transcript_first:
                try:
                    yt = await resolve_youtube_transcript(
                        url, force_refresh=force_refresh
                    )
                except Exception as exc:
                    yt = None
                    logger.debug(
                        "stt.youtube_transcript.fail_open url=%s err=%s",
                        url[:120] if url else "none",
                        exc,
                    )

                if yt and yt.text:
                    result = build_youtube_transcript_result(
                        url=url,
                        transcript_text=yt.text,
                        title=yt.title,
                        uploader=yt.uploader,
                        duration_s=yt.duration_s,
                        cache_hit=bool(yt.cache_hit),
                        source=yt.source,
                        language=yt.language,
                    )
                    logger.info(
                        "stt.youtube_transcript.ok video_id=%s lang=%s source=%s chars=%d cache_hit=%s",
                        yt.video_id,
                        yt.language or "unknown",
                        yt.source,
                        len(yt.text),
                        str(bool(yt.cache_hit)).lower(),
                    )
                    transcript_preview = (
                        (yt.text[:60] + "...") if len(yt.text) > 60 else yt.text
                    )
                    logger.info(
                        "stt.job.complete url=%s chars=%d preview=%s",
                        url[:80] if url else "none",
                        len(yt.text),
                        repr(transcript_preview),
                    )
                    return await job.finish_success(result)

            ready = await ensure_stt_manager_ready(stt_manager)
            if not ready:
                exc = InferenceError("STT engine not available")
                await job.finish_failure(exc)
                raise exc

            spans.start("yt-dlp")
            try:
                download = await fetch_and_prepare_url_audio(
                    url, force_refresh=force_refresh
                )
                spans.end("yt-dlp", ok=True)
            except VideoIngestError as exc:
                spans.end("yt-dlp", ok=False, reason="error")
                await job.finish_failure(exc)
                raise InferenceError(str(exc)) from exc

            job.register_download(download)
            ram_guard.check("yt-dlp")

            pre = await _preprocess_audio_with_retry(
                source_path=download.raw_path,
                spans=spans,
                download=download,
                voice_note=False,
                ram_guard=ram_guard,
            )
            job.register_pre(pre)
            ram_guard.check("pre-stage")

            spec = stt_manager.default_spec
            if pre.duration_in > 120:
                downgraded = stt_manager.downgrade_spec(spec)
                if downgraded:
                    logger.info(
                        "whisper.model_downgrade from=%s to=%s reason=long_audio",
                        spec.size,
                        downgraded.size,
                    )
                    spec = downgraded

            transcript = await _run_whisper_with_fallback(
                pre, spans, spec, ram_guard, job=job
            )

            spans.start("stitch")
            result = build_url_transcript_result(
                transcript=transcript,
                download=download,
                pre=pre,
                atempo_factor=ATEMPO_FACTOR,
            )
            spans.end("stitch", ok=True)
            _log_summary(spans, pre, transcript, cache_hit=transcript.cache_hit)

            # Log transcript completion with URL identity for debugging [REH]
            transcript_preview = (
                (transcript.text[:60] + "...")
                if len(transcript.text) > 60
                else transcript.text
            )
            logger.info(
                "stt.job.complete url=%s chars=%d preview=%s",
                url[:80] if url else "none",
                len(transcript.text),
                repr(transcript_preview),
            )

            return await job.finish_success(result)
    except RAMGuardExceeded as exc:
        if job.pre and job.pre.stream:
            try:
                await job.pre.stream.abort()
            except Exception:
                logger.debug(
                    "⚠️ Stream abort failed after RAM guard trigger", exc_info=True
                )
        await job.finish_failure(exc)
        raise InferenceError(str(exc)) from exc
    except Exception as exc:
        if job.pre and job.pre.stream:
            try:
                await job.pre.stream.abort()
            except Exception:
                logger.debug(
                    "⚠️ Stream abort failed in hear_infer_from_url", exc_info=True
                )
        await job.finish_failure(exc)
        raise
    finally:
        await job.close()


async def _run_whisper_with_fallback(
    pre: PreprocessResult,
    spans: SpanRecorder,
    initial_spec: ModelSpec,
    ram_guard: STTRAMGuard,
    job: Optional[STTJob] = None,
) -> TranscriptResult:
    """
    Primary whisper transcription with multimodal fallback support.

    This function attempts transcription with faster-whisper first, and falls back
    to multimodal models if the primary transcription fails with recoverable errors.
    """

    try:
        # Try the primary faster-whisper transcription
        transcript = await _run_whisper(pre, spans, initial_spec, ram_guard, job=job)

        # If successful, return the primary result
        return transcript

    except Exception as primary_error:
        # Classify the failure to determine if fallback should be attempted
        failure = STTFailureClassifier.classify_failure(
            error=primary_error, pre_result=pre, audio_path=pre.source_path
        )

        logger.info(f"[STT] Primary transcription failed: {failure}")

        # Check if multimodal fallback should be attempted
        config = load_config()
        fallback_enabled = config.get("STT_MULTIMODAL_FALLBACK_ENABLED", False)

        if not fallback_enabled:
            logger.info(
                "[STT] Multimodal fallback is disabled, re-raising primary error"
            )
            raise primary_error

        if not STTFailureClassifier.should_attempt_fallback(
            classification=failure,
            has_audio_data=pre.duration_in > 0,
            pre_duration=pre.duration_in,
        ):
            logger.info(
                f"[STT] Fallback not appropriate for failure type: {failure.category}"
            )
            raise primary_error

        # Attempt multimodal fallback
        spans.start("multimodal_fallback")

        try:
            logger.info("[STT] Attempting multimodal fallback transcription")

            # Get multimodal configuration
            model_config = {
                "timeout": 30.0,  # Default timeout
                "min_confidence": 0.5,  # Default confidence
                "max_retries": 1,  # Default retries
            }

            # Call the multimodal fallback provider
            fallback_result = (
                await multimodal_fallback_provider.transcribe_with_fallback(
                    audio_path=pre.source_path,
                    pre_result=pre,
                    failure_reason=failure,
                    model_config=model_config,
                )
            )

            spans.end("multimodal_fallback", ok=True)

            # Convert fallback result to Transcript format
            transcript = TranscriptResult(
                text=fallback_result.text,
                segments=[],  # Fallback doesn't provide segments
                chunks=[],  # Fallback doesn't provide chunks
                duration_out=pre.duration_in,
                model_spec=initial_spec,
                cache_hit=False,
                first_chunk_runtime=fallback_result.processing_time_ms / 1000.0,
                aborted=False,
                abort_reason=None,
                is_fallback=True,
                fallback_provider=fallback_result.provider,
                failure_context=str(failure),
            )

            # Log the fallback usage
            logger.info(
                f"[STT] Multimodal fallback succeeded with {fallback_result.provider} "
                f"(confidence: {fallback_result.confidence:.2f}, "
                f"time: {fallback_result.processing_time_ms:.1f}ms)"
            )

            return transcript

        except Exception as fallback_error:
            spans.end("multimodal_fallback", ok=False)
            logger.error(f"[STT] Multimodal fallback failed: {fallback_error}")

            # Re-raise the original error - fallback failed too
            raise primary_error
