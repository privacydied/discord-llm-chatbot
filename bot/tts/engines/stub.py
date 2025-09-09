# /bot/tts/engines/stub.py
import asyncio
import io
import wave
import math
import struct
from .base import BaseEngine
from ...utils.logging import get_logger

logger = get_logger(__name__)


def _generate_stub_wav_bytes(duration: float = 0.25, freq: int = 440) -> bytes:
    """Generates a short sine wave and returns it as WAV bytes."""
    rate = 16000  # 16kHz sample rate
    frames = int(rate * duration)

    # Generate PCM data (16-bit signed)
    pcm_data = bytearray()
    for i in range(frames):
        amplitude = 32767 * 0.2  # 20% volume to avoid clipping
        angle = 2 * math.pi * freq * i / rate
        value = int(amplitude * math.sin(angle))
        pcm_data.extend(struct.pack("<h", value))

    # Create a wave file in memory
    with io.BytesIO() as wav_buffer:
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 2 bytes = 16 bits
            wav_file.setframerate(rate)
            wav_file.writeframes(pcm_data)
        return wav_buffer.getvalue()


class StubEngine(BaseEngine):
    """A TTS engine that generates a stub audio signal as a fallback."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        logger.info("TTS StubEngine initialized.")

    async def synthesize(self, text: str) -> bytes:
        """
        Generates a stub WAV audio file in-memory, ignoring the input text.
        """
        logger.warning(f"Synthesizing STUB audio for text: '{text[:40]}...'")
        try:
            # This is a CPU-bound operation, but it's very fast.
            # We run it in an executor to avoid blocking the event loop just in case.
            loop = asyncio.get_running_loop()
            audio_data = await loop.run_in_executor(None, _generate_stub_wav_bytes)
            logger.info("Successfully generated stub audio bytes.")
            return audio_data
        except Exception as e:
            logger.error(f"Failed to generate stub WAV bytes: {e}", exc_info=True)
            return b""  # Return empty bytes on failure
