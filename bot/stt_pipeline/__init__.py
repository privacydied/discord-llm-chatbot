"""STT pipeline compatibility helpers."""

from .runtime import STTRuntimeCompat, ensure_stt_manager_ready, load_stt_runtime_compat

__all__ = [
    "STTRuntimeCompat",
    "ensure_stt_manager_ready",
    "load_stt_runtime_compat",
]
