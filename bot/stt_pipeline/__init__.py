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
from .youtube_path import build_youtube_transcript_result
from .result_payload import build_url_transcript_result

__all__ = [
    "STTRuntimeCompat",
    "build_youtube_transcript_result",
    "build_url_transcript_result",
    "ensure_stt_manager_ready",
    "ffmpeg_bin_has_aac",
    "ffmpeg_candidates_from_env",
    "ffmpeg_supports_aac_decoder",
    "load_stt_runtime_compat",
    "parse_stt_max_ram_mb",
    "resolve_ffmpeg_bin",
    "reset_ffmpeg_runtime_cache",
]
