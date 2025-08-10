#!/usr/bin/env python3
"""
Minimal TTS smoke test.
- Uses TTSManager to synthesize a short phrase.
- Ensures Kokoro ONNX assets (downloads if missing) and writes a WAV file.
Run:
  uv run python utils/tts_smoke.py
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from bot.tts.interface import TTSManager


async def main() -> None:
    # Prefer kokoro-onnx for this test
    os.environ.setdefault("TTS_ENGINE", "kokoro-onnx")
    out_dir = Path("tts")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "smoke.wav"

    mgr = TTSManager()
    text = "Hello! This is a TTS smoke test."
    wav_path = await mgr.generate_tts(text, out_path)

    # Basic file validation
    data = wav_path.read_bytes()
    size_kb = len(data) / 1024.0
    print(f"OK: wrote {wav_path} ({size_kb:.1f} KiB)")


if __name__ == "__main__":
    asyncio.run(main())
