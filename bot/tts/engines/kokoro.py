import logging
from kokoro_onnx import Kokoro
from .base import BaseEngine
from bot.tts.errors import TTSError

logger = logging.getLogger(__name__)

class KokoroONNXEngine(BaseEngine):
    def __init__(self, model_path: str, voices_path: str, tokenizer: str = "espeak"):
        self.model_path = model_path
        self.voices_path = voices_path
        self.tokenizer = tokenizer
        self.engine = None
        
    def load(self):
        try:
            self.engine = Kokoro(
                model_path=self.model_path,
                voices_path=self.voices_path
            )
            self.engine.tokenizer = self.tokenizer
            logger.info("KokoroEngine loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load KokoroEngine: {e}", exc_info=True)
            raise TTSError(f"Failed to load KokoroEngine: {e}") from e
            
    async def synthesize(self, text: str) -> bytes:
        if not self.engine:
            self.load()
            
        try:
            return await self.engine.generate_audio(text)
        except Exception as e:
            logger.error(f"Kokoro synthesis failed: {e}", exc_info=True)
            raise TTSError(f"Synthesis failed: {e}") from e