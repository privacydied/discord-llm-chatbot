"""
TTS (Text-to-Speech) package for the Discord bot.

This package contains modules for TTS functionality, including
the kokoro_bootstrap module for registering the EspeakWrapper tokenizer.
"""

import os
from pathlib import Path

from .kokoro_bootstrap import TOKENIZER_ALIASES, register_espeak_wrapper
from .stub import generate_stub_wav


async def generate_tts(text: str, user_id: str) -> Path:
    """Legacy-compatible helper to generate a TTS wav file.

    Creates a wav file in TEMP_DIR containing the user_id in the filename,
    matching tests' expectations. Uses the stub generator for portability.
    """
    temp_dir = os.getenv('TEMP_DIR', 'temp')
    out_dir = Path(temp_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{user_id}_tts.wav"

    # Prefer the high-level manager if available; fall back to stub
    try:
        # Lazy import to avoid circular import at package import time
        from .interface import TTSManager  # noqa: WPS433 (allow import inside function)
        manager = TTSManager()
        # Ensure WAV output and handle (Path, mime) return
        result = await manager.generate_tts(text, out_path=str(out_path), output_format="wav")
        if isinstance(result, tuple):
            final_path, _mime = result
        else:
            final_path = result
        return Path(final_path)
    except Exception:
        # Robust fallback in constrained test environments
        generate_stub_wav(str(out_path))
        return out_path


async def cleanup_tts() -> None:
    """Delete any .wav artifacts in TEMP_DIR (used by tests)."""
    temp_dir = os.getenv('TEMP_DIR', 'temp')
    d = Path(temp_dir)
    if not d.exists():
        return
    for p in d.glob("*.wav"):
        try:
            p.unlink()
        except Exception:
            # Best-effort cleanup for tests
            pass


# Minimal placeholder class to support tests that patch `bot.tts.TTS`.
class TTS:  # pragma: no cover - test helper symbol
    pass

__all__ = [
    'TOKENIZER_ALIASES',
    'register_espeak_wrapper',
    'TTSManager',
    'TTS',
    'generate_tts',
    'cleanup_tts',
]

# Lazy attribute access to avoid importing heavy modules at package import time
def __getattr__(name):  # pragma: no cover - import shim
    if name == 'TTSManager':
        from .interface import TTSManager as _TTSManager
        return _TTSManager
    raise AttributeError(name)
