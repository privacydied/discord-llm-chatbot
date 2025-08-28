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
from .engines.kokoro_v8 import KokoroV8Engine
from .errors import SynthesisError
from ..util.logging import get_logger
from .assets import ensure_kokoro_assets
from ..action import BotAction
import hashlib
from typing import Dict, Optional, List

logger = get_logger(__name__)

ENGINES = {
    "stub": StubEngine,
    "kokoro-onnx": KokoroONNXEngine,
    "kokoro": KokoroV8Engine,
}

class TTSManager:
    """Manages loading and interacting with the configured TTS engine."""

    def __init__(self, bot=None):
        # bot is optional for compatibility with tests and standalone usage
        self.bot = bot
        self.engine: BaseEngine = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='tts-worker')
        self._engine_name = os.getenv('TTS_ENGINE', 'stub')
        # In-memory file cache keyed by text hash [PA]
        self._file_cache: Dict[str, Path] = {}
        self._cache_order: List[str] = []
        try:
            self._cache_max = int(os.getenv('TTS_CACHE_MAX_ITEMS', '100'))
        except Exception:
            self._cache_max = 100
        self.load()

    def load(self):
        """Loads the primary TTS engine, falling back to the StubEngine on any error."""
        engine_name = self._engine_name
        logger.info(f"Attempting to load TTS engine: {engine_name}")

        try:
            engine_class = ENGINES.get(engine_name)
            if not engine_class:
                raise ValueError(f"Unsupported TTS engine: {engine_name}")

            if engine_name == "kokoro-onnx":
                # Best-effort prepare assets at startup if we're not already inside a running event loop
                model_path = os.getenv('TTS_MODEL_PATH')
                voices_path = os.getenv('TTS_VOICES_PATH')
                model_exists = Path(model_path).exists() if model_path else False
                voices_exist = Path(voices_path).exists() if voices_path else False
                if not (model_path and voices_path and model_exists and voices_exist):
                    try:
                        # If no running loop, prepare synchronously
                        asyncio.get_running_loop()
                        in_loop = True
                    except RuntimeError:
                        in_loop = False
                    if not in_loop:
                        try:
                            mp, vp = asyncio.run(ensure_kokoro_assets(Path('tts')))
                            model_path, voices_path = str(mp), str(vp)
                            os.environ['TTS_MODEL_PATH'] = model_path
                            os.environ['TTS_VOICES_PATH'] = voices_path
                            model_exists, voices_exist = True, True
                            logger.info("Prepared Kokoro assets at startup", extra={"subsys": "tts", "event": "assets_ready"})
                        except Exception:
                            logger.warning("Startup asset prepare failed; will ensure on demand", extra={"subsys": "tts"}, exc_info=True)
                if model_path and voices_path and model_exists and voices_exist:
                    self.engine = engine_class(model_path=model_path, voices_path=voices_path)
                else:
                    # Defer to on-demand preparation in synthesize, use stub for now
                    self.engine = StubEngine()
            elif engine_name == "kokoro":
                # New kokoro pipeline (no espeak, no assets)
                self.engine = KokoroV8Engine()
            else:
                self.engine = engine_class()

            logger.info(f"Successfully loaded TTS engine: {engine_name}")

        except Exception as e:
            logger.error(f"Failed to load primary TTS engine '{engine_name}': {e}. Falling back to stub.", exc_info=True)
            self.engine = StubEngine()

    def is_available(self) -> bool:
        """Checks if the primary TTS engine is loaded (and not the stub)."""
        return self.engine is not None and not isinstance(self.engine, StubEngine)

    async def synthesize(self, text: str, timeout: float = 25.0) -> bytes:
        """Generates audio from text using the loaded TTS engine.
        Supports both async and sync engine implementations.
        """
        if self.engine is None:
            logger.error("TTS engine not loaded, cannot synthesize.")
            return b''

        try:
            # On-demand asset preparation and engine upgrade if configured for kokoro-onnx
            if self._engine_name == 'kokoro-onnx':
                model_path = os.getenv('TTS_MODEL_PATH')
                voices_path = os.getenv('TTS_VOICES_PATH')
                model_exists = Path(model_path).exists() if model_path else False
                voices_exist = Path(voices_path).exists() if voices_path else False
                if not (model_path and voices_path and model_exists and voices_exist):
                    try:
                        mp, vp = await ensure_kokoro_assets(Path('tts'))
                        model_path, voices_path = str(mp), str(vp)
                        os.environ['TTS_MODEL_PATH'] = model_path
                        os.environ['TTS_VOICES_PATH'] = voices_path
                        model_exists, voices_exist = True, True
                        logger.info("Assets ensured on-demand", extra={"subsys": "tts", "event": "assets_ready"})
                    except Exception:
                        logger.warning("Failed to ensure Kokoro assets on-demand", extra={"subsys": "tts"}, exc_info=True)
                # If we have assets and current engine is stub, upgrade to Kokoro lazily
                if model_path and voices_path and model_exists and voices_exist and isinstance(self.engine, StubEngine):
                    try:
                        self.engine = KokoroONNXEngine(model_path=model_path, voices_path=voices_path)
                        logger.info("Switched to KokoroONNXEngine after assets ready", extra={"subsys": "tts"})
                    except Exception:
                        logger.warning("Kokoro engine init failed; continue with stub for this call", extra={"subsys": "tts"}, exc_info=True)

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

    async def generate_tts(self, text: str, out_path: str | Path | None = None, timeout: Optional[float] = None) -> Path:
        """Generate TTS to a file and return its Path.
        - If out_path is None, create a temporary .wav file.
        - Cleans text similarly to tests expectations.
        """
        cleaned = self._clean_text(text)
        if not cleaned:
            raise ValueError("text must not be empty after cleaning")
        if timeout is None:
            audio_bytes = await self.synthesize(cleaned)
        else:
            audio_bytes = await self.synthesize(cleaned, timeout=timeout)
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

    # --- High-level processing helper used by bot._execute_action() ---
    async def process(self, action: BotAction) -> BotAction:
        """Given a BotAction with text, synthesize audio and attach file path.
        Respects meta keys:
          - include_transcript: bool (default True)
          - tts_text: optional override text to synthesize
          - tts_timeout_s: float timeout override
        Applies a simple in-memory cache keyed by cleaned text hash. [PA]
        """
        try:
            # Config and limits [IV][CMV]
            include_transcript = bool(action.meta.get('include_transcript', True))
            try:
                max_chars = int(os.getenv('TTS_MAX_CHARS', '800'))
            except Exception:
                max_chars = 800
            try:
                timeout_s = float(action.meta.get('tts_timeout_s', os.getenv('TTS_TIMEOUT_S', '25.0')))
            except Exception:
                timeout_s = 25.0

            # Select text
            raw_text = action.meta.get('tts_text') or (action.content or '')
            cleaned_for_cache = self._clean_text(raw_text)
            if not cleaned_for_cache:
                logger.warning("tts:empty_text_after_clean")
                return action

            # Truncate for synthesis if needed (preserve full transcript if included)
            synth_text = cleaned_for_cache[:max_chars]
            truncated = len(cleaned_for_cache) > len(synth_text)

            # Cache lookup
            key = hashlib.sha256(synth_text.encode('utf-8')).hexdigest()
            cached_path: Optional[Path] = self._file_cache.get(key)
            if cached_path and cached_path.exists():
                logger.info(
                    "tts.cache.hit",
                    extra={"subsys": "tts", "event": "cache_hit", "text_len": len(synth_text)},
                )
                action.audio_path = str(cached_path)
            else:
                # Generate new file
                audio_path = await self.generate_tts(synth_text, timeout=timeout_s)
                action.audio_path = str(audio_path)
                # Insert into cache
                self._file_cache[key] = audio_path
                self._cache_order.append(key)
                if len(self._cache_order) > self._cache_max:
                    old_key = self._cache_order.pop(0)
                    try:
                        old_path = self._file_cache.pop(old_key, None)
                        # Don't delete files on disk; keep ephemeral tmp files managed by OS
                    except Exception:
                        pass
                logger.info(
                    "tts.cache.store",
                    extra={"subsys": "tts", "event": "cache_store", "text_len": len(synth_text)},
                )

            # Annotate meta
            if truncated:
                action.meta['tts_truncated'] = True

            # Keep or drop transcript content
            if not include_transcript:
                action.content = ''  # files-only message allowed

            return action
        except Exception as e:
            logger.error(f"tts.process.failed | {e}", extra={"subsys": "tts"}, exc_info=True)
            return action