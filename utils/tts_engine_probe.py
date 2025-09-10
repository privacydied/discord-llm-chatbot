#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Probe KokoroONNXEngine.synthesize() with a given phrase and save WAV output.
Usage:
  uv run python utils/tts_engine_probe.py [text]

It uses environment variables for model and voices with sane defaults:
  TTS_MODEL_FILE (default: tts/onnx/kokoro-v1.0.onnx)
  TTS_VOICE_FILE (default: tts/voices/voices-v1.0.bin)
  TTS_VOICE      (default: af_heart)

Note: This touches no Discord APIs.
"""

import os
import sys
import asyncio
from pathlib import Path

# Ensure project root is on sys.path so absolute imports like `utils.opus` work
# when running this script from the utils/ directory.
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _get_env_path(name: str, default: str) -> str:
    p = os.environ.get(name, default)
    return p


async def _run(text: str) -> int:
    # Lazy import to avoid side effects if module fails
    from bot.tts.engines.kokoro import KokoroONNXEngine

    model_path = _get_env_path("TTS_MODEL_FILE", "tts/onnx/kokoro-v1.0.onnx")
    voices_path = _get_env_path("TTS_VOICE_FILE", "tts/voices/voices-v1.0.bin")
    voice = os.environ.get("TTS_VOICE", "af_heart")

    print(f"Model: {model_path}")
    print(f"Voices: {voices_path}")
    print(f"Voice: {voice}")
    print(f"Text: {text}")

    eng = KokoroONNXEngine(
        model_path=model_path, voices_path=voices_path, tokenizer="espeak", voice=voice
    )

    try:
        audio_bytes = await eng.synthesize(text)
    except Exception as e:
        print(f"ERROR: synthesize failed: {e}")
        return 2

    out = Path("utils/tts_engine_probe_output.wav")
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("wb") as f:
            f.write(audio_bytes)
        print(f"Wrote {out} ({out.stat().st_size} bytes)")
    except Exception as e:
        print(f"ERROR: failed to write output: {e}")
        return 3

    return 0


def main() -> int:
    text = (
        "pyrex stirs turned to cavalli furs"
        if len(sys.argv) < 2
        else " ".join(sys.argv[1:])
    )
    return asyncio.run(_run(text))


if __name__ == "__main__":
    raise SystemExit(main())
