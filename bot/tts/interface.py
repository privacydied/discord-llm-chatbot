import asyncio
import logging
import os
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

from .engines.base import BaseEngine
from .engines.stub import StubEngine
from .engines.kokoro import KokoroONNXEngine
from .errors import SynthesisError
from ..util.logging import get_logger

logger = get_logger(__name__)

ENGINES = {
    "stub": StubEngine,
    "kokoro-onnx": KokoroONNXEngine,
}

class TTSManager:
    """Manages loading and interacting with the configured TTS engine."""

    def __init__(self, bot):
        self.bot = bot
        self.engine: BaseEngine = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='tts-worker')
        self.load()

    def load(self):
        """Loads the primary TTS engine, falling back to the StubEngine on any error."""
        engine_name = os.getenv('TTS_ENGINE', 'stub')
        logger.info(f"Attempting to load TTS engine: {engine_name}")

        try:
            engine_class = ENGINES.get(engine_name)
            if not engine_class:
                raise ValueError(f"Unsupported TTS engine: {engine_name}")

            if engine_name == "kokoro-onnx":
                model_path = os.getenv('TTS_MODEL_PATH')
                voices_path = os.getenv('TTS_VOICES_PATH')
                if not model_path or not voices_path:
                    raise ValueError("Missing TTS config: TTS_MODEL_PATH and TTS_VOICES_PATH must be set for kokoro-onnx.")
                self.engine = engine_class(model_path=model_path, voices_path=voices_path)
            else:
                self.engine = engine_class()

            logger.info(f"Successfully loaded TTS engine: {engine_name}")

        except Exception as e:
            logger.error(f"Failed to load primary TTS engine '{engine_name}': {e}. Falling back to stub.", exc_info=True)
            self.engine = StubEngine()

    def is_available(self) -> bool:
        """Checks if the primary TTS engine is loaded (and not the stub)."""
        return self.engine is not None and not isinstance(self.engine, StubEngine)

    async def synthesize(self, text: str, timeout: float = 10.0) -> bytes:
        """Generates audio from text using the loaded TTS engine."""
        if self.engine is None:
            # This should not happen due to the fallback in load(), but as a safeguard:
            logger.error("TTS engine not loaded, cannot synthesize.")
            return b''

        try:
            loop = asyncio.get_running_loop()
            future = self._executor.submit(self.engine.synthesize, text)
            audio_bytes = await asyncio.wait_for(loop.run_in_executor(None, future.result), timeout=timeout)
            
            logger.info(
                f"TTS synthesis successful (engine: {self.engine.__class__.__name__})",
                extra={"subsys": "tts", "event": "synthesis_complete", "text_length": len(text)}
            )
            return audio_bytes

        except concurrent.futures.TimeoutError:
            logger.error(f"TTS synthesis timed out after {timeout}s.", extra={"subsys": "tts"})
            raise SynthesisError(f"TTS synthesis timed out after {timeout} seconds")
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}", extra={"subsys": "tts"}, exc_info=True)
            raise SynthesisError(f"Synthesis failed: {e}") from e

    def close(self):
        """Cleans up TTS resources."""
        if self._executor:
            self._executor.shutdown(wait=True)