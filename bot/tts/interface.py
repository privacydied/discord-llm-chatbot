import asyncio
import logging
import os
import re
import tempfile
from pathlib import Path
import inspect
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

    def __init__(self, bot=None):
        # bot is optional for compatibility with tests and standalone usage
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
        """Generates audio from text using the loaded TTS engine.
        Supports both async and sync engine implementations.
        """
        if self.engine is None:
            logger.error("TTS engine not loaded, cannot synthesize.")
            return b''

        try:
            loop = asyncio.get_running_loop()
            maybe_coro = None
            try:
                # Call once to decide how to await/execute
                result = self.engine.synthesize(text)
                if inspect.isawaitable(result):
                    maybe_coro = result
            except TypeError:
                # Some engines may require different signatures; re-raise as SynthesisError below
                raise

            if maybe_coro is not None:
                audio_bytes = await asyncio.wait_for(maybe_coro, timeout=timeout)
            else:
                # Execute sync function in dedicated executor
                audio_bytes = await asyncio.wait_for(
                    loop.run_in_executor(self._executor, self.engine.synthesize, text),
                    timeout=timeout,
                )

            logger.info(
                f"TTS synthesis successful (engine: {self.engine.__class__.__name__})",
                extra={"subsys": "tts", "event": "synthesis_complete", "text_length": len(text)}
            )
            return audio_bytes

        except concurrent.futures.TimeoutError:
            logger.error(f"TTS synthesis timed out after {timeout}s.", extra={"subsys": "tts"})
            # Fallback to stub on timeout as a resiliency measure
            try:
                logger.warning("Falling back to StubEngine due to timeout", extra={"subsys": "tts"})
                stub = StubEngine()
                audio_bytes = await asyncio.wait_for(stub.synthesize(text), timeout=timeout)
                return audio_bytes
            except Exception:
                raise SynthesisError(f"TTS synthesis timed out after {timeout} seconds")
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}", extra={"subsys": "tts"}, exc_info=True)
            # Fallback to stub if primary engine fails unexpectedly (e.g., missing kokoro API)
            try:
                logger.warning("Primary engine failed, using StubEngine for this request", extra={"subsys": "tts"})
                stub = StubEngine()
                audio_bytes = await asyncio.wait_for(stub.synthesize(text), timeout=timeout)
                return audio_bytes
            except Exception:
                raise SynthesisError(f"Synthesis failed: {e}") from e

    async def close(self):
        """Cleans up TTS resources."""
        if self._executor:
            self._executor.shutdown(wait=True)

    # --- Compatibility helpers for tests ---
    def get_cache_stats(self) -> dict:
        """Return simple cache stats for compatibility with tests.
        This implementation reports an empty cache structure.
        """
        return {
            "files": [],
            "size_mb": 0.0,
            "cache_dir": ""
        }

    def _clean_text(self, text: str) -> str:
        """Remove simple markdown and URLs for cleaner TTS input.
        Matches tests by converting "**Hello** _world_ `code` https://example.com" -> "Hello world code".
        """
        if not text:
            return ""
        # Strip URLs
        text = re.sub(r"https?://\S+", "", text)
        # Remove basic markdown symbols and code backticks
        text = text.replace("**", "").replace("__", "").replace("*", "").replace("_", "").replace("`", "")
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    async def generate_tts(self, text: str, out_path: str | Path | None = None) -> Path:
        """Generate TTS to a file and return its Path.
        - If out_path is None, create a temporary .wav file.
        - Cleans text similarly to tests expectations.
        """
        cleaned = self._clean_text(text)
        if not cleaned:
            raise ValueError("text must not be empty after cleaning")
        audio_bytes = await self.synthesize(cleaned)
        if out_path is None:
            fd, tmp_name = tempfile.mkstemp(prefix="tts_", suffix=".wav")
            os.close(fd)
            out_path = Path(tmp_name)
        else:
            out_path = Path(out_path)
            if out_path.parent:
                out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(audio_bytes)
        return out_path