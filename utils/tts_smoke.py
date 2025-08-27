#!/usr/bin/env python3
"""
Minimal TTS smoke test.
- Uses TTSManager to synthesize a short phrase.
- Ensures Kokoro ONNX assets (downloads if missing) and writes a WAV file.

Run:
  uv run python utils/tts_smoke.py

With custom input/output/engine/voice:
  uv run python utils/tts_smoke.py \
    --text "Cavalli Furs going out of business sale!" \
    --out tts/oov.wav \
    --engine kokoro-onnx \
    --voice en_US-hfc_male-medium
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
import argparse
import sys

# Ensure project root is importable (so 'bot' and 'utils' resolve correctly)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from bot.tts.interface import TTSManager


async def main() -> None:
    parser = argparse.ArgumentParser(description="TTS smoke test using TTSManager")
    parser.add_argument(
        "--text",
        type=str,
        default="Hello! This is a TTS smoke test.",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="tts/smoke.wav",
        help="Output WAV path",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="kokoro-onnx",
        help="TTS engine to use (e.g., kokoro-onnx)",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=None,
        help="Optional voice name (sets TTS_VOICE environment variable)",
    )
    args = parser.parse_args()

    # Engine default (can be overridden by env outside)
    os.environ.setdefault("TTS_ENGINE", args.engine)
    if args.voice:
        os.environ["TTS_VOICE"] = args.voice

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mgr = TTSManager()
    wav_path = await mgr.generate_tts(args.text, out_path)

    # Basic file validation
    data = wav_path.read_bytes()
    size_kb = len(data) / 1024.0
    # Minimal WAV header validation
    if not data.startswith(b"RIFF"):
        print(f"WARN: Output does not start with RIFF header: {wav_path} ({size_kb:.1f} KiB)")
    else:
        print(f"OK: wrote {wav_path} ({size_kb:.1f} KiB)")


if __name__ == "__main__":
    asyncio.run(main())
