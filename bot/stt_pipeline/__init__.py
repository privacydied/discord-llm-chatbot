"""STT pipeline compatibility helpers."""

from .ffmpeg_runtime import (
    ffmpeg_bin_has_aac,
    ffmpeg_candidates_from_env,
    ffmpeg_supports_aac_decoder,
    reset_ffmpeg_runtime_cache,
    resolve_ffmpeg_bin,
)
from .lifecycle import (
    abort_and_finish_failure,
    abort_job_stream_if_present,
    create_stt_job,
)
from .logging import log_stt_job_complete, transcript_preview
from .result_payload import build_url_transcript_result
from .runtime import (
    STTRuntimeCompat,
    ensure_stt_manager_ready,
    load_stt_runtime_compat,
    parse_stt_max_ram_mb,
)
from .spec_select import select_initial_model_spec
from .stitch import run_stitch_stage
from .transcribe_flow import preprocess_and_transcribe
from .url_ingest import (
    ensure_manager_ready_or_raise,
    fetch_url_audio_or_raise,
    fetch_url_audio_with_span,
    prepare_url_download_for_stt,
)
from .youtube_path import build_youtube_transcript_result, try_youtube_transcript_first

__all__ = [
    "STTRuntimeCompat",
    "abort_and_finish_failure",
    "abort_job_stream_if_present",
    "build_url_transcript_result",
    "build_youtube_transcript_result",
    "create_stt_job",
    "ensure_manager_ready_or_raise",
    "ensure_stt_manager_ready",
    "fetch_url_audio_or_raise",
    "fetch_url_audio_with_span",
    "ffmpeg_bin_has_aac",
    "ffmpeg_candidates_from_env",
    "ffmpeg_supports_aac_decoder",
    "load_stt_runtime_compat",
    "log_stt_job_complete",
    "parse_stt_max_ram_mb",
    "prepare_url_download_for_stt",
    "preprocess_and_transcribe",
    "reset_ffmpeg_runtime_cache",
    "resolve_ffmpeg_bin",
    "run_stitch_stage",
    "select_initial_model_spec",
    "transcript_preview",
    "try_youtube_transcript_first",
]
