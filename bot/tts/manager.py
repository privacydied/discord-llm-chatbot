from .stub import generate_stub_wav
import os
import logging

logger = logging.getLogger(__name__)

class TTSManager:
    async def generate_tts(self, text: str, out_path: str) -> str:
        # Try real TTS here if implemented
        # For now, use stub
        try:
            # Placeholder for real TTS
            raise NotImplementedError("Real TTS not implemented")
        except Exception as e:
            logger.warning(f"TTS fallback: stub, error: {e}")
            generate_stub_wav(out_path)
            return out_path
