"""STT pipeline compatibility helpers."""

from .runtime import (
    STTRuntimeCompat,
    ensure_stt_manager_ready,
    load_stt_runtime_compat,
    parse_stt_max_ram_mb,
)
from .ffmpeg_runtime import (
    ffmpeg_bin_has_aac,
    ffmpeg_candidates_from_env,
    ffmpeg_supports_aac_decoder,
    resolve_ffmpeg_bin,
    reset_ffmpeg_runtime_cache,
)
from .youtube_path import build_youtube_transcript_result, try_youtube_transcript_first
from .result_payload import build_url_transcript_result
from .logging import log_stt_job_complete, transcript_preview
from .spec_select import select_initial_model_spec
from .lifecycle import abort_job_stream_if_present
from .url_ingest import fetch_url_audio_with_span
from .transcribe_flow import preprocess_and_transcribe

__all__ = [
    "STTRuntimeCompat",
    "build_youtube_transcript_result",
    "try_youtube_transcript_first",
    "build_url_transcript_result",
    "log_stt_job_complete",
    "abort_job_stream_if_present",
    "fetch_url_audio_with_span",
    "preprocess_and_transcribe",
    "select_initial_model_spec",
    "transcript_preview",
    "ensure_stt_manager_ready",
    "ffmpeg_bin_has_aac",
    "ffmpeg_candidates_from_env",
    "ffmpeg_supports_aac_decoder",
    "load_stt_runtime_compat",
    "parse_stt_max_ram_mb",
    "resolve_ffmpeg_bin",
    "reset_ffmpeg_runtime_cache",
]
