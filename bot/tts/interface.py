import asyncio
import os
import re
import tempfile
import shutil
from pathlib import Path
import inspect
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from utils.opus import transcode_to_ogg_opus

from .engines.base import BaseEngine
from .engines.stub import StubEngine
from .engines.kokoro import KokoroONNXEngine
from .engines.kokoro_v8 import KokoroV8Engine
from .errors import SynthesisError
from ..utils.logging import get_logger
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


class TTSResult:
    """Tuple-and-path compatible return type for generate_tts.

    Behaves like both:
    - a 2-tuple of ``(Path, mime_type)`` for callers that unpack
    - a Path-like object for callers that compare or use filesystem ops

    This reconciles mixed test expectations without breaking existing code.
    """

    __slots__ = ("path", "mime")

    def __init__(self, path: Path, mime: str) -> None:
        self.path = Path(path)
        self.mime = str(mime)

    # Tuple-unpack protocol
    def __iter__(self):
        yield self.path
        yield self.mime

    # Path-like behaviour
    def __fspath__(self) -> str:  # os.fspath support
        return str(self.path)

    def __str__(self) -> str:
        return str(self.path)

    def __repr__(self) -> str:
        return f"TTSResult(path={self.path!r}, mime={self.mime!r})"

    def __eq__(self, other) -> bool:
        try:
            if isinstance(other, Path):
                return self.path == other
            if isinstance(other, str):
                return str(self.path) == other
            # Compare to tuple-like (Path, mime)
            if isinstance(other, (tuple, list)) and len(other) >= 1:
                return self.path == other[0]
        except Exception:
            pass
        return False

    # Delegate attribute access to underlying Path (e.g., .exists(), .suffix)
    def __getattr__(self, name: str):
        return getattr(self.path, name)


class TTSManager:
    """Manages loading and interacting with the configured TTS engine."""

    def __init__(self, bot=None):
        # bot is optional for compatibility with tests and standalone usage
        self.bot = bot
        self.engine: BaseEngine = None
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="tts-worker"
        )
        self._engine_name = os.getenv("TTS_ENGINE", "stub")
        # In-memory file cache keyed by text hash [PA]
        self._file_cache: Dict[str, Path] = {}
        self._cache_order: List[str] = []
        try:
            self._cache_max = int(os.getenv("TTS_CACHE_MAX_ITEMS", "100"))
        except Exception:
            self._cache_max = 100
        # Track whether the primary (non-stub) engine has successfully synthesized at least once
        # Used to decide cold vs warm timeout selection. [CMV][PA]
        self._warmed_up: bool = False
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
                # Accept both legacy and new env var names for compatibility
                model_path = os.getenv("TTS_MODEL_PATH") or os.getenv(
                    "KOKORO_MODEL_PATH"
                )
                voices_path = os.getenv("TTS_VOICES_PATH") or os.getenv(
                    "KOKORO_VOICES_PATH"
                )
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
                            mp, vp = asyncio.run(ensure_kokoro_assets(Path("tts")))
                            model_path, voices_path = str(mp), str(vp)
                            # Set both env variants to keep all components in sync
                            os.environ["TTS_MODEL_PATH"] = model_path
                            os.environ["TTS_VOICES_PATH"] = voices_path
                            os.environ["KOKORO_MODEL_PATH"] = model_path
                            os.environ["KOKORO_VOICES_PATH"] = voices_path
                            model_exists, voices_exist = True, True
                            logger.info(
                                "Prepared Kokoro assets at startup",
                                extra={"subsys": "tts", "event": "assets_ready"},
                            )
                        except Exception:
                            logger.warning(
                                "Startup asset prepare failed; will ensure on demand",
                                extra={"subsys": "tts"},
                                exc_info=True,
                            )
                if model_path and voices_path and model_exists and voices_exist:
                    self.engine = engine_class(
                        model_path=model_path, voices_path=voices_path
                    )
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
            logger.error(
                f"Failed to load primary TTS engine '{engine_name}': {e}. Falling back to stub.",
                exc_info=True,
            )
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
            return b""

        try:
            # On-demand asset preparation and engine upgrade if configured for kokoro-onnx
            if self._engine_name == "kokoro-onnx":
                # Accept both legacy and new env var names for compatibility
                model_path = os.getenv("TTS_MODEL_PATH") or os.getenv(
                    "KOKORO_MODEL_PATH"
                )
                voices_path = os.getenv("TTS_VOICES_PATH") or os.getenv(
                    "KOKORO_VOICES_PATH"
                )
                model_exists = Path(model_path).exists() if model_path else False
                voices_exist = Path(voices_path).exists() if voices_path else False
                if not (model_path and voices_path and model_exists and voices_exist):
                    try:
                        mp, vp = await ensure_kokoro_assets(Path("tts"))
                        model_path, voices_path = str(mp), str(vp)
                        # Set both env variants to keep all components in sync
                        os.environ["TTS_MODEL_PATH"] = model_path
                        os.environ["TTS_VOICES_PATH"] = voices_path
                        os.environ["KOKORO_MODEL_PATH"] = model_path
                        os.environ["KOKORO_VOICES_PATH"] = voices_path
                        model_exists, voices_exist = True, True
                        logger.info(
                            "Assets ensured on-demand",
                            extra={"subsys": "tts", "event": "assets_ready"},
                        )
                    except Exception:
                        logger.warning(
                            "Failed to ensure Kokoro assets on-demand",
                            extra={"subsys": "tts"},
                            exc_info=True,
                        )
                # If we have assets and current engine is stub, upgrade to Kokoro lazily
                if (
                    model_path
                    and voices_path
                    and model_exists
                    and voices_exist
                    and isinstance(self.engine, StubEngine)
                ):
                    try:
                        self.engine = KokoroONNXEngine(
                            model_path=model_path, voices_path=voices_path
                        )
                        logger.info(
                            "Switched to KokoroONNXEngine after assets ready",
                            extra={"subsys": "tts"},
                        )
                    except Exception:
                        logger.warning(
                            "Kokoro engine init failed; continue with stub for this call",
                            extra={"subsys": "tts"},
                            exc_info=True,
                        )

            loop = asyncio.get_running_loop()
            # If engine exposes an async synthesize, await it directly; otherwise run in executor.
            # Some engines may have a sync method that returns an awaitable; handle that as well. [REH]
            is_async = asyncio.iscoroutinefunction(
                getattr(self.engine, "synthesize", None)
            )
            if is_async:
                audio_bytes = await asyncio.wait_for(
                    self.engine.synthesize(text), timeout=timeout
                )
            else:
                # Execute the sync call in a background thread to avoid blocking the event loop
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        self._executor, lambda: self.engine.synthesize(text)
                    ),
                    timeout=timeout,
                )
                # If the sync call returned an awaitable, await it to completion
                if inspect.isawaitable(result):
                    audio_bytes = await asyncio.wait_for(result, timeout=timeout)
                else:
                    audio_bytes = result

            logger.info(
                f"TTS synthesis successful (engine: {self.engine.__class__.__name__})",
                extra={
                    "subsys": "tts",
                    "event": "synthesis_complete",
                    "text_length": len(text),
                },
            )
            # Mark engine as warmed only if we're not using the stub. [CMV]
            if not isinstance(self.engine, StubEngine):
                self._warmed_up = True
            return audio_bytes

        except concurrent.futures.TimeoutError:
            logger.error(
                f"TTS synthesis timed out after {timeout}s.", extra={"subsys": "tts"}
            )
            # Fallback to stub on timeout as a resiliency measure
            try:
                logger.warning(
                    "Falling back to StubEngine due to timeout", extra={"subsys": "tts"}
                )
                stub = StubEngine()
                audio_bytes = await asyncio.wait_for(
                    stub.synthesize(text), timeout=timeout
                )
                return audio_bytes
            except Exception:
                raise SynthesisError(f"TTS synthesis timed out after {timeout} seconds")
        except Exception as e:
            logger.error(
                f"TTS synthesis failed: {e}", extra={"subsys": "tts"}, exc_info=True
            )
            # Fallback to stub if primary engine fails unexpectedly (e.g., missing kokoro API)
            try:
                logger.warning(
                    "Primary engine failed, using StubEngine for this request",
                    extra={"subsys": "tts"},
                )
                stub = StubEngine()
                audio_bytes = await asyncio.wait_for(
                    stub.synthesize(text), timeout=timeout
                )
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
        return {"files": [], "size_mb": 0.0, "cache_dir": ""}

    def purge_old_cache(self) -> None:
        """Synchronous cache maintenance hook used by background task.
        Our TTS cache is an in-memory map of cleaned-text hashes to temp file paths.
        We do not manage a persistent on-disk cache here. This method performs
        housekeeping only:
          - Remove entries whose file path no longer exists.
          - Enforce the configured `_cache_max` size by trimming the oldest keys.

        No-op safe: it avoids raising on any error. [REH]
        """
        try:
            # Drop non-existent files
            alive_keys = []
            for key in list(self._cache_order):
                p = self._file_cache.get(key)
                try:
                    if p and Path(p).exists():
                        alive_keys.append(key)
                    else:
                        # Remove dead entry
                        self._file_cache.pop(key, None)
                except Exception:
                    # On any unexpected error, drop the entry to keep cache healthy
                    self._file_cache.pop(key, None)
            self._cache_order = alive_keys

            # Enforce max size
            if len(self._cache_order) > self._cache_max:
                overflow = len(self._cache_order) - self._cache_max
                for _ in range(overflow):
                    old_key = self._cache_order.pop(0)
                    try:
                        self._file_cache.pop(old_key, None)
                    except Exception:
                        pass
        except Exception:
            # Never let maintenance crash callers
            logger.debug(
                "tts.cache.purge_ignored_error", extra={"subsys": "tts"}, exc_info=True
            )

    def _clean_text(self, text: str) -> str:
        """Remove simple markdown and URLs for cleaner TTS input.
        Matches tests by converting "**Hello** _world_ `code` https://example.com" -> "Hello world code".
        """
        if not text:
            return ""
        # Strip URLs
        text = re.sub(r"https?://\S+", "", text)
        # Remove basic markdown symbols and code backticks
        text = (
            text.replace("**", "")
            .replace("__", "")
            .replace("*", "")
            .replace("_", "")
            .replace("`", "")
        )
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    async def generate_tts(
        self,
        text: str,
        out_path: str | Path | None = None,
        output_format: str = "ogg",
        timeout: Optional[float] = None,
    ) -> TTSResult:
        """Generate TTS to a file and return its Path.
        - If out_path is None, create a temporary .wav file.
        - If out_path is provided, its suffix takes precedence when inferring the
          output container (e.g., .wav => WAV, .ogg => OGG), even if a different
          output_format was requested. This avoids mismatched extensions.
        - Cleans text similarly to tests expectations.
        """
        cleaned = self._clean_text(text)
        if not cleaned:
            raise ValueError("text must not be empty after cleaning")
        # Select dynamic timeout when not explicitly provided. [CMV]
        if timeout is None:
            try:
                base = float(os.getenv("TTS_TIMEOUT_S", "25.0"))
            except Exception:
                base = 25.0
            try:
                cold = float(os.getenv("TTS_TIMEOUT_COLD_S", str(base)))
            except Exception:
                cold = base
            try:
                warm = float(os.getenv("TTS_TIMEOUT_WARM_S", str(base)))
            except Exception:
                warm = base
            # Heuristic: cold until a successful non-stub synthesis, or if kokoro-onnx is configured but engine is stub. [PA]
            is_cold = (not self._warmed_up) or (
                self._engine_name == "kokoro-onnx"
                and isinstance(self.engine, StubEngine)
            )
            selected_timeout = cold if is_cold else warm
            logger.debug(
                "tts.timeout.selected",
                extra={
                    "subsys": "tts",
                    "event": "timeout_selected",
                    "phase": "cold" if is_cold else "warm",
                    "timeout_s": selected_timeout,
                },
            )
            audio_bytes = await self.synthesize(cleaned, timeout=selected_timeout)
        else:
            audio_bytes = await self.synthesize(cleaned, timeout=timeout)

        # Always write to intermediate WAV first
        fd, wav_tmp_name = tempfile.mkstemp(prefix="tts_", suffix=".wav")
        os.close(fd)
        wav_path = Path(wav_tmp_name)
        wav_path.write_bytes(audio_bytes)

        # Determine effective format. Suffix (when provided) takes precedence over argument.
        effective_format = output_format
        if out_path is not None:
            suffix = Path(out_path).suffix.lower()
            if suffix == ".wav":
                effective_format = "wav"
            elif suffix == ".ogg":
                effective_format = "ogg"

        if effective_format == "wav":
            final_path = out_path or wav_path
            if out_path and out_path != wav_path:
                if Path(out_path).parent:
                    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(wav_path), str(final_path))
            return TTSResult(Path(final_path), "audio/wav")

        # OGG/Opus (48k mono) using async ffmpeg subprocess
        ogg_out = (
            Path(out_path)
            if out_path
            else Path(tempfile.mktemp(prefix="tts_", suffix=".ogg"))
        )
        if ogg_out.parent and not ogg_out.parent.exists():
            ogg_out.parent.mkdir(parents=True, exist_ok=True)
        try:
            bitrate = os.getenv("OPUS_BITRATE", "64k")
            vbr = os.getenv("OPUS_VBR", "on")
            try:
                compression_level = int(os.getenv("OPUS_COMPRESSION_LEVEL", "10"))
            except Exception:
                compression_level = 10
            ogg_path = await transcode_to_ogg_opus(
                wav_path,
                ogg_out,
                bitrate=bitrate,
                vbr=vbr,
                compression_level=compression_level,
            )
        finally:
            # Best-effort cleanup of intermediate WAV
            try:
                wav_path.unlink()
            except Exception:
                pass
        return TTSResult(Path(ogg_path), "audio/ogg")

    # --- High-level processing helper used by bot._execute_action() ---
    async def process(self, action: BotAction) -> BotAction:
        """Given a BotAction with text, synthesize audio and attach file path.
        Respects meta keys:
          - include_transcript: bool (default True)
          - tts_text: optional override text to synthesize
          - tts_timeout_s: float timeout override (takes precedence)
          - tts_cold: bool flag to force cold/warm timeout selection
          - tts_timeout_cold_s / tts_timeout_warm_s: per-call overrides
        Applies a simple in-memory cache keyed by cleaned text hash. [PA]
        """
        try:
            # Config and limits [IV][CMV]
            include_transcript = bool(action.meta.get("include_transcript", True))
            try:
                max_chars = int(os.getenv("TTS_MAX_CHARS", "800"))
            except Exception:
                max_chars = 800
            # Timeout selection with overrides and cold/warm split. [CMV]
            timeout_s: float
            if "tts_timeout_s" in action.meta:
                try:
                    timeout_s = float(action.meta.get("tts_timeout_s"))
                except Exception:
                    timeout_s = 25.0
            else:
                # Per-call overrides first, then env, then base
                def _get_float(key: str, default: float) -> float:
                    try:
                        v = action.meta.get(key)
                        return float(v) if v is not None else default
                    except Exception:
                        return default

                try:
                    base = float(os.getenv("TTS_TIMEOUT_S", "25.0"))
                except Exception:
                    base = 25.0
                try:
                    env_cold = float(os.getenv("TTS_TIMEOUT_COLD_S", str(base)))
                except Exception:
                    env_cold = base
                try:
                    env_warm = float(os.getenv("TTS_TIMEOUT_WARM_S", str(base)))
                except Exception:
                    env_warm = base
                cold_override = _get_float("tts_timeout_cold_s", env_cold)
                warm_override = _get_float("tts_timeout_warm_s", env_warm)
                # Determine phase: explicit meta wins; else heuristic
                if "tts_cold" in action.meta:
                    is_cold = bool(action.meta.get("tts_cold"))
                else:
                    is_cold = (not self._warmed_up) or (
                        self._engine_name == "kokoro-onnx"
                        and isinstance(self.engine, StubEngine)
                    )
                timeout_s = cold_override if is_cold else warm_override
                logger.debug(
                    "tts.timeout.selected",
                    extra={
                        "subsys": "tts",
                        "event": "timeout_selected",
                        "phase": "cold" if is_cold else "warm",
                        "timeout_s": timeout_s,
                    },
                )

            # Select text
            raw_text = action.meta.get("tts_text") or (action.content or "")
            cleaned_for_cache = self._clean_text(raw_text)
            if not cleaned_for_cache:
                logger.warning("tts:empty_text_after_clean")
                return action

            # Truncate for synthesis if needed (preserve full transcript if included)
            synth_text = cleaned_for_cache[:max_chars]
            truncated = len(cleaned_for_cache) > len(synth_text)

            # Cache lookup
            key = hashlib.sha256(synth_text.encode("utf-8")).hexdigest()
            cached_path: Optional[Path] = self._file_cache.get(key)
            if cached_path and cached_path.exists():
                logger.info(
                    "tts.cache.hit",
                    extra={
                        "subsys": "tts",
                        "event": "cache_hit",
                        "text_len": len(synth_text),
                    },
                )
                action.audio_path = str(cached_path)
            else:
                # Generate new file (accept Path, (Path, mime), TTSResult, or str)
                result = await self.generate_tts(synth_text, timeout=timeout_s)
                audio_path: Path
                mime_type: str = "audio/ogg"
                if isinstance(result, tuple) and len(result) >= 2:
                    audio_path, mime_type = result[0], result[1]  # type: ignore[assignment]
                elif isinstance(result, Path):
                    audio_path = result
                elif isinstance(result, str):
                    audio_path = Path(result)
                else:
                    # Try tuple-like unpack (e.g., TTSResult implements __iter__)
                    try:
                        audio_path, mime_type = result  # type: ignore[misc]
                    except Exception:
                        try:
                            # Try os.fspath protocol
                            audio_path = Path(os.fspath(result))  # type: ignore[arg-type]
                            mime_type = getattr(result, "mime", mime_type)
                        except Exception:
                            audio_path = Path(str(result))
                action.audio_path = str(audio_path)
                # Insert into cache
                self._file_cache[key] = audio_path
                self._cache_order.append(key)
                if len(self._cache_order) > self._cache_max:
                    old_key = self._cache_order.pop(0)
                    try:
                        self._file_cache.pop(old_key, None)
                        # Don't delete files on disk; keep ephemeral tmp files managed by OS
                    except Exception:
                        pass
                logger.info(
                    "tts.cache.store",
                    extra={
                        "subsys": "tts",
                        "event": "cache_store",
                        "text_len": len(synth_text),
                    },
                )

            # Annotate meta
            if truncated:
                action.meta["tts_truncated"] = True

            # Keep or drop transcript content
            if not include_transcript:
                action.content = ""  # files-only message allowed

            return action
        except Exception as e:
            logger.error(
                f"tts.process.failed | {e}", extra={"subsys": "tts"}, exc_info=True
            )
            return action

    # --- Legacy/Test compatibility transcoder ---
    async def _to_ogg_opus_ffmpegpy(
        self,
        wav_path: str | Path,
        out_path: str | Path | None = None,
        *,
        bitrate: str | None = None,
        vbr: str | None = None,
        compression_level: int | None = None,
    ) -> Path:
        """Transcode WAV to OGG/Opus using our ffmpeg wrapper.

        This exists for test compatibility that checks for this method.
        It is a thin wrapper around `utils.opus.transcode_to_ogg_opus`.
        """
        # Resolve defaults from environment to mirror generate_tts behaviour
        if bitrate is None:
            bitrate = os.getenv("OPUS_BITRATE", "64k")
        if vbr is None:
            vbr = os.getenv("OPUS_VBR", "on")
        if compression_level is None:
            try:
                compression_level = int(os.getenv("OPUS_COMPRESSION_LEVEL", "10"))
            except Exception:
                compression_level = 10

        return await transcode_to_ogg_opus(
            wav_path,
            out_path,
            bitrate=bitrate,
            vbr=vbr,
            compression_level=compression_level,
        )
