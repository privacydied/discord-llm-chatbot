import asyncio
import logging
import os
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

from .engines.stub import StubEngine
from .engines.base import BaseEngine
from .errors import SynthesisError
from ..util.logging import get_logger

logger = get_logger(__name__)

class TTSManager:
    """Manages loading and interacting with the configured TTS engine."""

    def __init__(self, bot):
        self.bot = bot
        self.engine: BaseEngine = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='tts-worker')
        self.load()

    def load(self):
        """Loads the primary TTS engine, falling back to the StubEngine on any error."""
        try:
            engine_name = os.getenv('TTS_BACKEND')
            if not engine_name:
                raise ValueError("TTS_BACKEND environment variable not set.")

            logger.info(f"Attempting to load TTS engine: {engine_name}")
            if engine_name == 'kokoro':
                from .engines import kokoro as kokoro_engine
                
                # Read configuration from environment
                model_path = os.getenv('TTS_MODEL_PATH')
                voices_path = os.getenv('TTS_VOICES_PATH')
                tokenizer = os.getenv('TTS_TOKENISER', 'espeak')
                
                if not model_path or not voices_path:
                    raise ValueError("Missing required TTS environment variables (TTS_MODEL_PATH, TTS_VOICES_PATH).")

                self.engine = kokoro_engine.KokoroEngine(
                    model_path=model_path,
                    voices_path=voices_path,
                    tokenizer=tokenizer
                )
                logger.info("Successfully loaded Kokoro TTS engine.")
            # Add other engines here with 'elif engine_name == ...'
            else:
                raise ValueError(f"Unsupported TTS engine: {engine_name}")

        except Exception as e:
            logger.warning(f"Failed to load primary TTS engine: {e}. Falling back to stub.", exc_info=True)
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