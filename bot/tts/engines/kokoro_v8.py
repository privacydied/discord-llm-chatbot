"""
Engine adapter for kokoro>=0.8 (KPipeline), which does not require espeak.

[RAT][REH][PA][CMV]
"""
from __future__ import annotations

import io
import logging
import os
import wave
from typing import Optional

import numpy as np
from bot.util.logging import get_logger
from .base import BaseEngine
from bot.tts.errors import TTSError

logger = get_logger(__name__)


class KokoroV8Engine(BaseEngine):
    def __init__(self, voice: Optional[str] = None, lang_code: Optional[str] = None):
        # Defaults align with common Kokoro-82M examples
        self.voice = voice or os.getenv("TTS_VOICE", "af_heart")
        self.lang_code = lang_code or os.getenv("TTS_LANG_CODE", "a")  # 'a' American English
        self._pipeline = None

    def load(self):
        try:
            from kokoro import KPipeline  # type: ignore
            self._pipeline = KPipeline(lang_code=self.lang_code)
            logger.info("KokoroV8Engine loaded (KPipeline)")
        except Exception as e:
            logger.error(f"Failed to load KokoroV8Engine: {e}", exc_info=True)
            raise TTSError(f"Failed to load KokoroV8Engine: {e}") from e

    def _to_wav_bytes(self, audio: np.ndarray, sr: int) -> bytes:
        # Normalize to int16 PCM
        if audio.dtype != np.int16:
            # clip then scale
            audio = np.clip(audio, -1.0, 1.0)
            audio = (audio * 32767.0).astype(np.int16)
        # Ensure mono
        if audio.ndim > 1:
            # Mix down
            audio = audio.mean(axis=1).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(sr))
            wf.writeframes(audio.tobytes())
        return buf.getvalue()

    async def synthesize(self, text: str) -> bytes:
        if self._pipeline is None:
            self.load()
        try:
            # KPipeline returns (audio: np.ndarray, sample_rate: int)
            audio, sr = self._pipeline(text, voice=self.voice)
            return self._to_wav_bytes(audio, sr)
        except Exception as e:
            logger.error(f"KokoroV8Engine synthesis failed: {e}", exc_info=True)
            raise TTSError(f"Kokoro v8 synthesis failed: {e}") from e
