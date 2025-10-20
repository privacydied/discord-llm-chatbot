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
import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .exceptions import InferenceError
from .utils.logging import get_logger
from .video_ingest import (
    DownloadedAudio,
    VideoIngestError,
    fetch_and_prepare_url_audio,
)
from .stt import ModelSpec, stt_manager

if TYPE_CHECKING:
    import discord

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


# ---------------------------------------------------------------------------
# Utility dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PreprocessResult:
    pcm: np.ndarray
    sample_rate: int
    duration_in: float
    duration_out: float
    atempo_applied: bool
    silence_applied: bool
    silence_removed_ms: int
    cache_key: str
    cache_hit: bool
    source_hash: str


@dataclass
class TranscriptResult:
    text: str
    segments: List[Dict[str, Any]]
    chunks: List[Dict[str, Any]]
    duration_out: float
    model_spec: ModelSpec
    cache_hit: bool
    first_chunk_runtime: float


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


def _load_pcm_cache(cache_key: str) -> Optional[np.ndarray]:
    path = PCM_CACHE_DIR / f"{cache_key}.npz"
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            pcm = data["pcm"]
            return pcm
    except Exception as exc:
        logger.warning("⚠️ Failed to load preprocessed cache %s: %s", path, exc)
        return None


def _store_pcm_cache(cache_key: str, pcm: np.ndarray) -> None:
    path = PCM_CACHE_DIR / f"{cache_key}.npz"
    try:
        np.savez_compressed(path, pcm=pcm.astype(np.float32))
    except Exception as exc:
        logger.warning("⚠️ Failed to store preprocessed cache %s: %s", path, exc)


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
    cmd = [
        "ffprobe",
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


# ---------------------------------------------------------------------------
# Preprocess stage
# ---------------------------------------------------------------------------


async def _preprocess_audio(
    source_path: Path,
    spans: SpanRecorder,
    download: Optional[DownloadedAudio],
    voice_note: bool,
) -> PreprocessResult:
    spans.start("pre")
    duration_in, sr, channels = await _ffprobe(source_path)
    logger.info(
        "pre.probe dur_in=%.2f sr=%s ch=%s", duration_in, sr, channels
    )

    silence_applied = not voice_note and duration_in > SHORT_CLIP_S
    atempo_applied = not voice_note and duration_in >= ATEMPO_THRESHOLD_S

    source_hash = _source_hash(source_path, download)
    cache_key = _preprocess_cache_key(
        source_hash, atempo_applied, silence_applied, duration_in
    )

    cached_pcm = _load_pcm_cache(cache_key)
    if cached_pcm is not None:
        spans.end("pre", ok=True, reason="cache")
        logger.info(
            "cache.hit stage=audio key=%s samples=%s",
            cache_key[:12],
            cached_pcm.shape[0],
        )
        duration_out = cached_pcm.shape[0] / SAMPLE_RATE
        silence_removed_ms = max(
            0,
            int(
                (
                    (duration_in / ATEMPO_FACTOR if atempo_applied else duration_in)
                    - duration_out
                )
                * 1000
            ),
        )
        return PreprocessResult(
            pcm=cached_pcm,
            sample_rate=SAMPLE_RATE,
            duration_in=duration_in,
            duration_out=duration_out,
            atempo_applied=atempo_applied,
            silence_applied=silence_applied,
            silence_removed_ms=silence_removed_ms,
            cache_key=cache_key,
            cache_hit=True,
            source_hash=source_hash,
        )

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

    ffmpeg_cmd = [
        "ffmpeg",
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
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=pre_budget)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        spans.end("pre", ok=False, reason="timeout")
        logger.error("stt.fail reason=pre_timeout")
        raise InferenceError("Audio preprocessing timed out")

    if proc.returncode != 0:
        spans.end("pre", ok=False, reason="error")
        raise InferenceError(
            f"Audio preprocessing failed ({proc.returncode}): {stderr.decode(errors='ignore')}"
        )

    pcm_bytes = stdout
    pcm_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
    pcm = pcm_int16.astype(np.float32) / 32768.0

    duration_out = pcm.shape[0] / SAMPLE_RATE
    expected_after_atempo = (
        duration_in / ATEMPO_FACTOR if atempo_applied else duration_in
    )
    silence_removed_ms = max(
        0, int((expected_after_atempo - duration_out) * 1000)
    )

    if silence_applied:
        logger.info("pre.silence removed_ms=%s", silence_removed_ms)
    if atempo_applied:
        logger.info("pre.atempo factor=%.2f", ATEMPO_FACTOR)

    if pcm.nbytes <= MEMORY_THRESHOLD_BYTES:
        _store_pcm_cache(cache_key, pcm)
        logger.info(
            "cache.store stage=audio key=%s bytes=%s",
            cache_key[:12],
            pcm.nbytes,
        )
    else:
        logger.info(
            "pre.warn memory_threshold_exceeded bytes=%s threshold=%s",
            pcm.nbytes,
            MEMORY_THRESHOLD_BYTES,
        )

    spans.end("pre", ok=True, reason="ok")
    return PreprocessResult(
        pcm=pcm,
        sample_rate=SAMPLE_RATE,
        duration_in=duration_in,
        duration_out=duration_out,
        atempo_applied=atempo_applied,
        silence_applied=silence_applied,
        silence_removed_ms=silence_removed_ms,
        cache_key=cache_key,
        cache_hit=False,
        source_hash=source_hash,
    )


# ---------------------------------------------------------------------------
# Whisper transcription stage
# ---------------------------------------------------------------------------


def _iter_chunks(audio: np.ndarray, sample_rate: int) -> List[Tuple[int, np.ndarray, float, float]]:
    duration = audio.shape[0] / sample_rate
    if duration <= LONG_AUDIO_THRESHOLD_S:
        return [(0, audio, 0.0, duration)]

    chunk_samples = int(CHUNK_WINDOW_S * sample_rate)
    overlap_samples = int(CHUNK_OVERLAP_S * sample_rate)
    chunks: List[Tuple[int, np.ndarray, float, float]] = []
    start = 0
    idx = 0
    total_samples = audio.shape[0]

    while start < total_samples:
        end = min(total_samples, start + chunk_samples)
        chunk_audio = audio[start:end]
        chunk_start_s = start / sample_rate
        chunk_end_s = end / sample_rate
        chunks.append((idx, chunk_audio, chunk_start_s, chunk_end_s))
        start = end - overlap_samples
        if start <= 0:
            start = end
        idx += 1
    return chunks


def _segments_to_dict(
    segments: List[Any], offset: float
) -> List[Dict[str, Any]]:
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
) -> TranscriptResult:
    audio = pre.pcm
    sample_rate = pre.sample_rate
    spec = initial_spec
    attempted_slow_downgrade = False

    while True:
        model = await stt_manager.ensure_model(spec)
        transcript = await _transcribe_with_model(
            model=model,
            spec=spec,
            audio=audio,
            sample_rate=sample_rate,
            pre=pre,
            spans=spans,
        )

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
                continue
        return transcript


async def _transcribe_with_model(
    model: "WhisperModel",
    spec: ModelSpec,
    audio: np.ndarray,
    sample_rate: int,
    pre: PreprocessResult,
    spans: SpanRecorder,
) -> TranscriptResult:
    cache_key = _transcript_cache_key(pre.cache_key, spec, vad_enabled=True)
    cached = _load_transcript_cache(cache_key)
    if cached:
        spans.spans["whisper"] = 0
        logger.info("stt.span stage=whisper ms=0 ok=true reason=cache")
        logger.info(
            "cache.hit stage=transcript key=%s", cache_key[:12]
        )
        return cached

    spans.start("whisper")
    chunks = _iter_chunks(audio, sample_rate)
    chunk_records: List[Dict[str, Any]] = []
    segments_accum: List[Dict[str, Any]] = []
    first_chunk_runtime = 0.0

    def _run() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float]:
        nonlocal first_chunk_runtime
        for idx, chunk_audio, chunk_start_s, chunk_end_s in chunks:
            chunk_begin = time.perf_counter()
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
            segment_list = list(seg_iter)
            runtime = time.perf_counter() - chunk_begin
            if idx == 0:
                first_chunk_runtime = runtime
                if not segment_list and info.no_speech_prob >= NO_SPEECH_PROB_THRESHOLD:
                    raise NoSpeechDetected(
                        f"No speech detected (prob={info.no_speech_prob:.3f})"
                    )
            logger.info(
                "whisper.chunk idx=%s len_s=%.2f",
                idx,
                chunk_end_s - chunk_start_s,
            )
            seg_dicts = _segments_to_dict(segment_list, offset=chunk_start_s)
            segments_accum.extend(seg_dicts)
            chunk_records.append(
                {
                    "idx": idx,
                    "start": chunk_start_s,
                    "end": chunk_end_s,
                    "segments": seg_dicts,
                }
            )
        return segments_accum, chunk_records, first_chunk_runtime

    whisper_budget = min(180.0, max(20.0, pre.duration_out * 2.5 + 10.0))
    loop = asyncio.get_running_loop()
    try:
        await asyncio.wait_for(loop.run_in_executor(None, _run), timeout=whisper_budget)
    except NoSpeechDetected as exc:
        spans.end("whisper", ok=False, reason="no_speech")
        logger.info("stt.no_speech_fast_exit")
        raise InferenceError(str(exc))
    except asyncio.TimeoutError:
        spans.end("whisper", ok=False, reason="timeout")
        logger.error("stt.fail reason=whisper_timeout")
        raise InferenceError("Transcription timed out")
    except Exception as exc:
        logger.error(
            "whisper.transcribe_failed err=%s spec=%s dur=%.2f",
            exc,
            spec.size,
            pre.duration_out,
            exc_info=True,
        )
        spans.end("whisper", ok=False, reason="error")
        raise InferenceError(f"Transcription failed: {exc}") from exc

    spans.end("whisper", ok=True, reason="ok")
    text = _join_segments(segments_accum)
    result = TranscriptResult(
        text=text,
        segments=segments_accum,
        chunks=chunk_records,
        duration_out=pre.duration_out,
        model_spec=spec,
        cache_hit=False,
        first_chunk_runtime=first_chunk_runtime,
    )
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
    audio: Union[Path, "discord.Attachment"]
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
    temp_handle = None
    attachment = audio if not isinstance(audio, Path) else None

    try:
        local_path, temp_handle, _ = await _ensure_local_audio(audio)
        voice_note = _is_voice_note(attachment) if attachment is not None else False
        pre = await _preprocess_audio(
            source_path=local_path,
            spans=spans,
            download=None,
            voice_note=voice_note,
        )

        # Model selection heuristic: drop one size for very long clips (>120s)
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

        transcript = await _run_whisper(pre, spans, spec)

        spans.start("stitch")
        result_text = transcript.text
        spans.end("stitch", ok=True)
        _log_summary(spans, pre, transcript, cache_hit=transcript.cache_hit)
        return result_text

    finally:
        if temp_handle is not None:
            try:
                os.unlink(temp_handle.name)
            except Exception:
                pass


async def hear_infer_from_url(
    url: str, force_refresh: bool = False
) -> Dict[str, Any]:
    """
    Transcribe audio fetched via yt-dlp for the given URL.
    """
    spans = SpanRecorder()
    try:
        spans.start("yt-dlp")
        download = await fetch_and_prepare_url_audio(url, force_refresh=force_refresh)
        spans.end("yt-dlp", ok=True)
    except VideoIngestError as exc:
        spans.end("yt-dlp", ok=False, reason="error")
        raise InferenceError(str(exc)) from exc

    pre = await _preprocess_audio(
        source_path=download.raw_path,
        spans=spans,
        download=download,
        voice_note=False,
    )

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

    transcript = await _run_whisper(pre, spans, spec)

    spans.start("stitch")
    metadata = download.metadata
    result = {
        "transcription": transcript.text,
        "metadata": {
            "source": metadata.source_type,
            "url": metadata.url,
            "title": metadata.title,
            "uploader": metadata.uploader,
            "upload_date": metadata.upload_date,
            "original_duration_s": metadata.duration_seconds,
            "processed_duration_s": pre.duration_out,
            "speedup_factor": ATEMPO_FACTOR if pre.atempo_applied else 1.0,
            "cache_hit": download.cache_hit or transcript.cache_hit,
            "timestamp": download.timestamp.isoformat(),
        },
    }
    spans.end("stitch", ok=True)
    _log_summary(spans, pre, transcript, cache_hit=transcript.cache_hit)
    return result
