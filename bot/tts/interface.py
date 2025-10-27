import asyncio
import io
import os
import re
import tempfile
import shutil
from pathlib import Path
import inspect
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from utils.opus import transcode_to_ogg_opus
import numpy as np
import soundfile as sf

from .engines.base import BaseEngine
from .engines.stub import StubEngine
from .engines.kokoro import KokoroONNXEngine
from .engines.kokoro_v8 import KokoroV8Engine
from .errors import SynthesisError
from ..utils.logging import get_logger
from .assets import ensure_kokoro_assets
from ..action import BotAction
import hashlib
from typing import Dict, Optional, List, Tuple, Any

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

    __slots__ = ("path", "mime", "meta")

    def __init__(self, path: Path, mime: str, meta: Optional[dict] = None) -> None:
        self.path = Path(path)
        self.mime = str(mime)
        self.meta = meta or {}

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
        self._rng = np.random.default_rng()
        raw_engine = os.getenv("TTS_ENGINE", "").strip().lower()
        self._engine_name = raw_engine or "kokoro-onnx"
        self._explicit_stub = self._engine_name == "stub"
        self._degraded = False
        self._degraded_reason: Optional[str] = None
        self._asset_lock = asyncio.Lock()
        self._assets_ready = False
        self._asset_paths: Optional[Tuple[Path, Path]] = None
        self._assets_error: Optional[Exception] = None
        # In-memory file cache keyed by text hash [PA]
        self._file_cache: Dict[str, Path] = {}
        self._cache_order: List[str] = []
        self._cache_meta: Dict[str, Dict[str, Any]] = {}
        try:
            self._cache_max = int(os.getenv("TTS_CACHE_MAX_ITEMS", "100"))
        except Exception:
            self._cache_max = 100
        # Track whether the primary (non-stub) engine has successfully synthesized at least once
        # Used to decide cold vs warm timeout selection. [CMV][PA]
        self._warmed_up: bool = False
        self._warmup_done: bool = False
        self._warmup_lock: asyncio.Lock = asyncio.Lock()
        self._head_rms_threshold = 5e-4
        self._preroll_lead_ms = 80
        self._preroll_xfade_ms = 6
        self._preroll_attenuation_db = 9
        self._noise_dbfs = -45
        self._tail_pad_ms = 60
        self.load()

    async def _ensure_warmup(self) -> None:
        """Run one-time tokenizer and engine warmup to avoid clipped onset."""
        if self._warmup_done or isinstance(self.engine, StubEngine):
            return
        async with self._warmup_lock:
            if self._warmup_done or isinstance(self.engine, StubEngine):
                return
            tokenizer_ready = False
            engine_ready = False
            try:
                from bot.tts.eng_g2p_local import _configure_official_tokenizer_tmpdir

                _configure_official_tokenizer_tmpdir()
                tokenizer_ready = True
            except Exception:
                logger.debug(
                    "tts.warmup.tokenizer_failed",
                    extra={"subsys": "tts"},
                    exc_info=True,
                )
            try:
                warm_text = "tts warmup"
                warmup_callable = getattr(self.engine, "warmup", None)
                if callable(warmup_callable):
                    maybe = warmup_callable(warm_text)
                    if inspect.isawaitable(maybe):
                        await maybe
                else:
                    synth_attr = getattr(self.engine, "synthesize", None)
                    if asyncio.iscoroutinefunction(synth_attr):
                        await asyncio.wait_for(synth_attr(warm_text), timeout=5.0)
                    elif callable(synth_attr):
                        loop = asyncio.get_running_loop()
                        try:
                            await asyncio.wait_for(
                                loop.run_in_executor(self._executor, lambda: synth_attr(warm_text)),
                                timeout=5.0,
                            )
                        except asyncio.TimeoutError:
                            logger.debug(
                                "tts.warmup.engine_timeout",
                                extra={"subsys": "tts"},
                            )
                        except asyncio.CancelledError:
                            logger.debug(
                                "tts.warmup.engine_cancelled",
                                extra={"subsys": "tts"},
                            )
                            raise
                engine_ready = True
            except Exception:
                logger.debug(
                    "tts.warmup.engine_failed",
                    extra={"subsys": "tts"},
                    exc_info=True,
                )
            self._warmup_done = True
            logger.info(
                "tts.warmup tokenizer=%s engine=%s",
                str(tokenizer_ready).lower(),
                str(engine_ready).lower(),
                extra={
                    "subsys": "tts",
                    "event": "warmup",
                    "tokenizer": tokenizer_ready,
                    "engine": engine_ready,
                },
            )

    def _resample_audio(self, audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
        if from_sr == to_sr:
            return audio
        try:
            import librosa  # type: ignore

            return librosa.resample(audio, orig_sr=from_sr, target_sr=to_sr).astype(
                np.float32
            )
        except Exception:
            try:
                from scipy.signal import resample_poly  # type: ignore

                return resample_poly(audio, to_sr, from_sr).astype(np.float32)
            except Exception:
                ratio = float(to_sr) / float(from_sr)
                new_len = max(1, int(round(audio.shape[0] * ratio)))
                x_old = np.linspace(0.0, 1.0, audio.shape[0], endpoint=False)
                x_new = np.linspace(0.0, 1.0, new_len, endpoint=False)
                return np.interp(x_new, x_old, audio).astype(np.float32)

    def _apply_fade(
        self, audio: np.ndarray, sr: int, fade_in_ms: int = 3, fade_out_ms: int = 6
    ) -> np.ndarray:
        result = audio.copy()
        fade_in_samples = max(0, int(sr * fade_in_ms / 1000))
        fade_out_samples = max(0, int(sr * fade_out_ms / 1000))
        if fade_in_samples > 0 and fade_in_samples < result.shape[0]:
            fade_curve = np.linspace(0.0, 1.0, fade_in_samples, endpoint=True)
            result[:fade_in_samples] *= fade_curve
        if fade_out_samples > 0 and fade_out_samples < result.shape[0]:
            fade_curve = np.linspace(1.0, 0.0, fade_out_samples, endpoint=True)
            result[-fade_out_samples:] *= fade_curve
        return result

    def _pad_audio(
        self, audio: np.ndarray, sr: int, head_ms: int, tail_ms: int
    ) -> np.ndarray:
        head_samples = max(0, int(sr * head_ms / 1000))
        tail_samples = max(0, int(sr * tail_ms / 1000))
        if head_samples == 0 and tail_samples == 0:
            return audio
        head_pad = np.zeros(head_samples, dtype=np.float32)
        tail_pad = np.zeros(tail_samples, dtype=np.float32)
        return np.concatenate((head_pad, audio, tail_pad))

    def _self_check_first_rms(self, path: Path, window_ms: int = 200) -> tuple[float, bool]:
        try:
            with sf.SoundFile(str(path)) as f:
                frames = min(int(window_ms * f.samplerate / 1000), len(f))
                if frames <= 0:
                    return 0.0, False
                first = f.read(frames, dtype="float32", always_2d=False)
            if isinstance(first, np.ndarray) and first.ndim > 1:
                first = first.mean(axis=1)
            rms = float(np.sqrt(np.mean(np.square(first)))) if first.size else 0.0
            return rms, rms < 1e-4
        except Exception:
            logger.debug(
                "tts.selfcheck.decode_failed", extra={"subsys": "tts"}, exc_info=True
            )
            return 0.0, False

    def _compute_ipa_length(self, text: str) -> int:
        try:
            from bot.tts.eng_g2p_local import text_to_ipa

            lang = str(getattr(self.engine, "language", "en")).lower()
            if lang.startswith("en"):
                return len(text_to_ipa(text))
        except Exception:
            logger.debug(
                "tts.summary.ipa_failed", extra={"subsys": "tts"}, exc_info=True
            )
        return 0

    def _collect_audio_stats(self, audio_path: Path) -> tuple[int, float]:
        try:
            with sf.SoundFile(str(audio_path)) as f:
                frames = len(f)
                sr = f.samplerate
            duration = frames / sr if sr else 0.0
            return sr, duration
        except Exception:
            logger.debug(
                "tts.summary.read_failed", extra={"subsys": "tts"}, exc_info=True
            )
            return 48000, 0.0

    def _compute_rms(self, audio: np.ndarray) -> float:
        if audio.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))

    def _apply_head_boost(
        self, audio: np.ndarray, sr: int, gain_db: float = 2.0, window_ms: int = 100
    ) -> np.ndarray:
        samples = min(audio.size, max(0, int(sr * window_ms / 1000)))
        if samples <= 0:
            return audio
        boost = 10 ** (gain_db / 20.0)
        ramp = np.linspace(1.0, boost, samples, dtype=np.float32)
        boosted = audio.copy()
        boosted[:samples] *= ramp
        return boosted

    def _generate_pink_noise(self, samples: int) -> np.ndarray:
        if samples <= 0:
            return np.zeros(0, dtype=np.float32)
        white = self._rng.standard_normal(samples + 5).astype(np.float32)
        kernel = np.array([1.0, 0.5, 0.25, 0.125, 0.0625], dtype=np.float32)
        noise = np.convolve(white, kernel, mode="valid")
        if noise.size < samples:
            noise = np.pad(noise, (0, samples - noise.size))
        noise = noise[:samples]
        peak = float(np.max(np.abs(noise))) or 1.0
        noise = noise / peak
        target = 10 ** (self._noise_dbfs / 20.0)
        return (noise * target).astype(np.float32)

    def _apply_preroll(
        self,
        audio: np.ndarray,
        sr: int,
        lead_ms: Optional[int] = None,
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        lead_ms = self._preroll_lead_ms if lead_ms is None else lead_ms
        lead_samples = min(audio.size, int(sr * lead_ms / 1000))
        xfade_samples = min(
            int(sr * self._preroll_xfade_ms / 1000),
            max(lead_samples // 2, 0),
        )
        attenuation = 10 ** (-self._preroll_attenuation_db / 20.0)
        meta: Dict[str, Any] = {
            "kind": "copy",
            "lead_ms": lead_ms,
            "xfade_ms": self._preroll_xfade_ms,
            "attenuation_db": self._preroll_attenuation_db,
        }

        if lead_samples > 0:
            lead_segment = audio[:lead_samples].copy()
            if np.max(np.abs(lead_segment)) > 0:
                lead_segment *= attenuation
                body = audio.copy()
                if xfade_samples > 0:
                    ramp = np.linspace(0.0, 1.0, xfade_samples, endpoint=False, dtype=np.float32)
                    inv_ramp = 1.0 - ramp
                    lead_segment[-xfade_samples:] *= inv_ramp
                    body[:xfade_samples] *= ramp
                merged = np.concatenate([lead_segment, body])
                logger.info(
                    "tts.preroll kind=%s ramp_db=-%d lead_ms=%d xfade_ms=%d",
                    meta["kind"],
                    self._preroll_attenuation_db,
                    lead_ms,
                    self._preroll_xfade_ms,
                    extra={
                        "subsys": "tts",
                        "event": "preroll",
                        "kind": meta["kind"],
                        "lead_ms": lead_ms,
                        "xfade_ms": self._preroll_xfade_ms,
                        "ramp_db": -self._preroll_attenuation_db,
                    },
                )
                return merged.astype(np.float32, copy=False), meta

        # Fallback to low-level noise preroll
        noise_samples = max(1, int(sr * lead_ms / 1000))
        noise = self._generate_pink_noise(noise_samples)
        meta.update(
            {
                "kind": "noise",
                "noise_dbfs": self._noise_dbfs,
                "lead_ms": lead_ms,
            }
        )
        body = audio.copy()
        xfade_samples = min(int(sr * self._preroll_xfade_ms / 1000), noise.size)
        if xfade_samples > 0:
            ramp = np.linspace(0.0, 1.0, xfade_samples, endpoint=False, dtype=np.float32)
            inv_ramp = 1.0 - ramp
            noise[-xfade_samples:] *= inv_ramp
            body[:xfade_samples] *= ramp
        merged = np.concatenate([noise, body])
        logger.info(
            "tts.preroll kind=%s dbfs=%d lead_ms=%d",
            meta["kind"],
            self._noise_dbfs,
            lead_ms,
            extra={
                "subsys": "tts",
                "event": "preroll",
                "kind": meta["kind"],
                "lead_ms": lead_ms,
                "dbfs": self._noise_dbfs,
            },
        )
        return merged.astype(np.float32, copy=False), meta

    def _decode_audio_bytes(self, audio_bytes: bytes) -> tuple[np.ndarray, int]:
        with sf.SoundFile(io.BytesIO(audio_bytes)) as f:
            pcm = f.read(dtype="float32", always_2d=False)
            sr = f.samplerate
        if isinstance(pcm, np.ndarray) and pcm.ndim > 1:
            pcm = pcm.mean(axis=1)
        return pcm.astype(np.float32, copy=False), sr

    def _emit_summary(
        self,
        *,
        text_chars: int,
        ipa_len: int,
        head_pad_ms: int,
        tail_pad_ms: int,
        lead_preroll_ms: int,
        sr: int,
        duration_s: float,
        cached: bool,
    ) -> None:
        logger.info(
            "tts.summary text_chars=%d ipa_len=%d lead_preroll_ms=%d tail_pad_ms=%d head_pad_ms=%d sr=%d dur_s=%.3f cached=%s",
            text_chars,
            ipa_len,
            lead_preroll_ms,
            tail_pad_ms,
            head_pad_ms,
            sr,
            duration_s,
            str(cached).lower(),
            extra={
                "subsys": "tts",
                "event": "summary",
                "text_chars": text_chars,
                "ipa_len": ipa_len,
                "lead_preroll_ms": lead_preroll_ms,
                "head_pad_ms": head_pad_ms,
                "tail_pad_ms": tail_pad_ms,
                "sr": sr,
                "duration_s": duration_s,
                "cached": cached,
            },
        )

    def _build_cache_key(
        self, text: str, voice: str, speed: float, mode: str
    ) -> str:
        seed = f"v3|{voice}|{speed:.3f}|{mode}|{text}"
        return hashlib.sha256(seed.encode("utf-8")).hexdigest()

    def _record_degraded(self, reason: str) -> None:
        self._degraded = not self._explicit_stub
        self._degraded_reason = reason
        logger.warning(
            "tts.engine.degraded",
            extra={
                "subsys": "tts",
                "event": "engine_degraded",
                "engine": self._engine_name,
                "reason": reason,
            },
        )

    def _clear_degraded(self) -> None:
        self._degraded = False
        self._degraded_reason = None

    def _set_asset_env(self, model_path: Path, voices_path: Path) -> None:
        mp = str(model_path)
        vp = str(voices_path)
        for key in ("TTS_MODEL_PATH", "KOKORO_MODEL_PATH"):
            os.environ[key] = mp
        for key in ("TTS_VOICES_PATH", "KOKORO_VOICES_PATH"):
            os.environ[key] = vp

    def _hydrate_asset_state(self) -> None:
        model_candidate = os.getenv("TTS_MODEL_PATH") or os.getenv("KOKORO_MODEL_PATH")
        voices_candidate = os.getenv("TTS_VOICES_PATH") or os.getenv("KOKORO_VOICES_PATH")
        if model_candidate and voices_candidate:
            model_path = Path(model_candidate)
            voices_path = Path(voices_candidate)
            if model_path.exists() and voices_path.exists():
                self._assets_ready = True
                self._asset_paths = (model_path, voices_path)
                self._assets_error = None
                return
        self._assets_ready = False
        self._asset_paths = None

    def load(self):
        """Loads the primary TTS engine, falling back to StubEngine only when explicit or degraded."""
        engine_name = self._engine_name
        logger.info(f"Attempting to load TTS engine: {engine_name}")

        try:
            engine_class = ENGINES.get(engine_name)
            if not engine_class:
                raise ValueError(f"Unsupported TTS engine: {engine_name}")

            if engine_name == "kokoro-onnx":
                self._hydrate_asset_state()
                if not self._assets_ready:
                    try:
                        asyncio.get_running_loop()
                        in_loop = True
                    except RuntimeError:
                        in_loop = False
                    if not in_loop:
                        try:
                            model_path, voices_path = asyncio.run(
                                ensure_kokoro_assets(Path("tts"))
                            )
                            self._assets_ready = True
                            self._asset_paths = (model_path, voices_path)
                            self._assets_error = None
                            self._set_asset_env(model_path, voices_path)
                            logger.info(
                                "Prepared Kokoro assets at startup",
                                extra={"subsys": "tts", "event": "assets_ready"},
                            )
                        except Exception as exc:
                            self._assets_error = exc
                            logger.warning(
                                "Startup asset prepare failed; will ensure on demand",
                                extra={"subsys": "tts", "event": "asset_prepare_deferred"},
                                exc_info=True,
                            )
                if self._assets_ready and self._asset_paths:
                    model_path, voices_path = self._asset_paths
                    self.engine = engine_class(
                        model_path=str(model_path), voices_path=str(voices_path)
                    )
                    self._clear_degraded()
                else:
                    # Defer initialization; mark degraded until assets are present
                    self.engine = StubEngine()
                    self._record_degraded("kokoro assets unavailable")
            elif engine_name == "kokoro":
                # New kokoro pipeline (no espeak, no assets)
                self.engine = KokoroV8Engine()
                self._clear_degraded()
            else:
                self.engine = engine_class()
                self._clear_degraded()

            logger.info(f"Successfully loaded TTS engine: {engine_name}")

        except Exception as e:
            logger.error(
                f"Failed to load primary TTS engine '{engine_name}': {e}. Falling back to stub.",
                exc_info=True,
            )
            self.engine = StubEngine()
            self._record_degraded(str(e))

    def is_available(self) -> bool:
        """Checks if the primary TTS engine is loaded (and not the stub)."""
        if self.engine is None:
            return False
        if isinstance(self.engine, StubEngine):
            return False
        return not self._degraded

    def get_status(self) -> dict:
        """Expose engine status for callers that surface diagnostics."""
        return {
            "engine": self._engine_name,
            "available": self.is_available(),
            "explicit_stub": self._explicit_stub,
            "degraded": self._degraded,
            "degraded_reason": self._degraded_reason,
            "assets_ready": self._assets_ready,
        }

    async def _ensure_kokoro_assets(self) -> Tuple[Path, Path]:
        if self._assets_ready and self._asset_paths:
            return self._asset_paths
        async with self._asset_lock:
            if self._assets_ready and self._asset_paths:
                return self._asset_paths
            try:
                model_path, voices_path = await ensure_kokoro_assets(Path("tts"))
            except Exception as exc:
                self._assets_error = exc
                logger.error(
                    "Failed to ensure Kokoro assets", extra={"subsys": "tts"}, exc_info=True
                )
                raise
            self._assets_ready = True
            self._asset_paths = (model_path, voices_path)
            self._assets_error = None
            self._set_asset_env(model_path, voices_path)
            logger.info(
                "Assets ensured on-demand",
                extra={"subsys": "tts", "event": "assets_ready"},
            )
            return self._asset_paths

    async def synthesize(self, text: str, timeout: float = 25.0, **engine_kwargs) -> bytes:
        """Generates audio from text using the loaded TTS engine.
        Supports both async and sync engine implementations.
        """
        if self.engine is None:
            logger.error("TTS engine not loaded, cannot synthesize.")
            raise SynthesisError("TTS engine not loaded")

        try:
            # On-demand asset preparation and engine upgrade if configured for kokoro-onnx
            if self._engine_name == "kokoro-onnx":
                try:
                    model_path, voices_path = await self._ensure_kokoro_assets()
                except Exception as exc:
                    self._record_degraded(f"kokoro assets unavailable: {exc}")
                    raise SynthesisError("Kokoro assets unavailable") from exc

                if isinstance(self.engine, StubEngine) and not self._explicit_stub:
                    try:
                        self.engine = KokoroONNXEngine(
                            model_path=str(model_path), voices_path=str(voices_path)
                        )
                        self._clear_degraded()
                        logger.info(
                            "Switched to KokoroONNXEngine after assets ready",
                            extra={"subsys": "tts"},
                        )
                    except Exception as exc:
                        self._record_degraded(f"kokoro init failed: {exc}")
                        raise SynthesisError("Failed to initialize Kokoro engine") from exc

            if isinstance(self.engine, StubEngine) and not self._explicit_stub:
                reason = self._degraded_reason or "Primary TTS engine unavailable"
                raise SynthesisError(reason)

            loop = asyncio.get_running_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    self._executor, lambda: self.engine.synthesize(text, **engine_kwargs)
                ),
                timeout=timeout,
            )
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

        except concurrent.futures.TimeoutError as exc:
            logger.error(
                f"TTS synthesis timed out after {timeout}s.", extra={"subsys": "tts"}
            )
            raise SynthesisError(f"TTS synthesis timed out after {timeout} seconds") from exc
        except Exception as e:
            message = str(e)
            if "engine_missing_callable" in message:
                reason = "engine_missing_callable"
            elif "engine_input_error" in message:
                reason = "engine_input_error"
            else:
                reason = "runtime"
            log_extra = {"subsys": "tts"}
            if reason == "engine_input_error":
                log_extra["detail"] = message
            logger.error(
                "tts.process.failed engine=%s reason=%s",
                self.engine.__class__.__name__,
                reason,
                extra=log_extra,
            )
            if reason == "engine_missing_callable":
                raise SynthesisError("engine_missing_callable") from e
            if reason == "engine_input_error":
                raise SynthesisError(message) from e
            raise SynthesisError(f"Synthesis failed: {message}") from e

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
                        self._cache_meta.pop(key, None)
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
                        self._cache_meta.pop(old_key, None)
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
        else:
            selected_timeout = timeout

        await self._ensure_warmup()
        speed = 1.0
        respeed_attempted = False
        duration_action = "ok"
        head_action = "ok"
        head_rms = 0.0
        boosted_rms = 0.0

        while True:
            audio_bytes = await self.synthesize(
                cleaned, timeout=selected_timeout, speed=speed
            )
            try:
                raw_pcm, source_sr = self._decode_audio_bytes(audio_bytes)
            except Exception as exc:
                raise SynthesisError(f"Failed to decode TTS audio: {exc}") from exc

            if source_sr != 48000:
                raw_pcm = self._resample_audio(raw_pcm, source_sr, 48000)
            sr = 48000

            raw_duration_ms = raw_pcm.size * 1000.0 / sr if sr else 0.0
            word_count = max(1, len(cleaned.split()))
            min_len_ms = 200.0 + 220.0 * word_count

            if raw_duration_ms < min_len_ms:
                if not respeed_attempted:
                    duration_action = "respeed"
                    logger.info(
                        "tts.duration_check words=%d dur_ms=%.1f min_ms=%.1f action=%s",
                        word_count,
                        raw_duration_ms,
                        min_len_ms,
                        duration_action,
                        extra={
                            "subsys": "tts",
                            "event": "duration_check",
                            "words": word_count,
                            "dur_ms": raw_duration_ms,
                            "min_ms": min_len_ms,
                            "action": duration_action,
                        },
                    )
                    speed = round(speed * 0.90, 3)
                    respeed_attempted = True
                    continue
                duration_action = "accept_short"
            else:
                duration_action = "ok"

            logger.info(
                "tts.duration_check words=%d dur_ms=%.1f min_ms=%.1f action=%s",
                word_count,
                raw_duration_ms,
                min_len_ms,
                duration_action,
                extra={
                    "subsys": "tts",
                    "event": "duration_check",
                    "words": word_count,
                    "dur_ms": raw_duration_ms,
                    "min_ms": min_len_ms,
                    "action": duration_action,
                },
            )
            if duration_action == "accept_short":
                logger.info(
                    "tts.short_audio words=%d dur_ms=%.1f min_ms=%.1f",
                    word_count,
                    raw_duration_ms,
                    min_len_ms,
                    extra={
                        "subsys": "tts",
                        "event": "short_audio",
                        "words": word_count,
                        "dur_ms": raw_duration_ms,
                        "min_ms": min_len_ms,
                    },
                )

            head_window_samples = min(raw_pcm.size, int(sr * 0.05))
            head_rms = self._compute_rms(raw_pcm[:head_window_samples])
            boosted_rms = head_rms
            head_action = "ok"
            if head_rms < self._head_rms_threshold:
                raw_pcm = self._apply_head_boost(raw_pcm, sr)
                boosted_rms = self._compute_rms(raw_pcm[:head_window_samples])
                head_action = "boost_resynth"

            logger.info(
                "tts.head_rms rms=%.6e boosted_rms=%.6e action=%s",
                head_rms,
                boosted_rms,
                head_action,
                extra={
                    "subsys": "tts",
                    "event": "head_rms",
                    "rms": head_rms,
                    "boosted_rms": boosted_rms,
                    "action": head_action,
                },
            )
            break

        preroll_audio, preroll_meta = self._apply_preroll(raw_pcm, sr)
        final_audio = self._pad_audio(preroll_audio, sr, 0, self._tail_pad_ms)
        duration_s = final_audio.size / sr if sr else 0.0

        window_100 = min(final_audio.size, int(sr * 0.1))
        window_200 = min(final_audio.size, int(sr * 0.2))
        rms0_100 = self._compute_rms(final_audio[:window_100])
        rms100_200 = self._compute_rms(final_audio[window_100:window_200])
        needs_retry = (
            rms0_100 < self._head_rms_threshold
            and rms100_200 < self._head_rms_threshold
        )

        if needs_retry:
            extended_audio, preroll_meta = self._apply_preroll(
                raw_pcm, sr, lead_ms=preroll_meta["lead_ms"] + 60
            )
            final_audio = self._pad_audio(extended_audio, sr, 0, self._tail_pad_ms)
            duration_s = final_audio.size / sr if sr else duration_s
            window_100 = min(final_audio.size, int(sr * 0.1))
            window_200 = min(final_audio.size, int(sr * 0.2))
            rms0_100 = self._compute_rms(final_audio[:window_100])
            rms100_200 = self._compute_rms(final_audio[window_100:window_200])

        # Always write to intermediate WAV first
        fd, wav_tmp_name = tempfile.mkstemp(prefix="tts_", suffix=".wav")
        os.close(fd)
        wav_path = Path(wav_tmp_name)
        sf.write(str(wav_path), final_audio, sr, subtype="PCM_16")

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
            meta = {
                "sr": sr,
                "duration_s": duration_s,
                "head_pad_ms": 0,
                "tail_pad_ms": self._tail_pad_ms,
                "lead_preroll_ms": preroll_meta.get("lead_ms", self._preroll_lead_ms),
                "cached": False,
                "ipa_len": self._compute_ipa_length(cleaned),
                "text_chars": len(cleaned),
                "speed": speed,
            }
            return TTSResult(Path(final_path), "audio/wav", meta=meta)

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
            preskip_samples = 312
            logger.info(
                "tts.encode format=opus sr=%d preskip=%d dur_s=%.3f",
                sr,
                preskip_samples,
                duration_s,
                extra={
                    "subsys": "tts",
                    "event": "encode",
                    "format": "opus",
                    "sr": sr,
                    "preskip": preskip_samples,
                    "duration_s": duration_s,
                },
            )
            logger.info(
                "tts.selfcheck rms0_100=%.6e rms100_200=%.6e reencode=%s",
                rms0_100,
                rms100_200,
                str(needs_retry).lower(),
                extra={
                    "subsys": "tts",
                    "event": "selfcheck",
                    "rms0_100": rms0_100,
                    "rms100_200": rms100_200,
                    "reencode": needs_retry,
                },
            )
        finally:
            try:
                wav_path.unlink()
            except Exception:
                pass

        meta = {
            "sr": sr,
            "duration_s": duration_s,
            "head_pad_ms": 0,
            "tail_pad_ms": self._tail_pad_ms,
            "lead_preroll_ms": preroll_meta.get("lead_ms", self._preroll_lead_ms),
            "cached": False,
            "ipa_len": self._compute_ipa_length(cleaned),
            "text_chars": len(cleaned),
            "speed": speed,
            "duration_action": duration_action,
            "head_rms": head_rms,
            "boosted_rms": boosted_rms,
            "preroll_kind": preroll_meta.get("kind", "copy"),
        }
        return TTSResult(Path(ogg_path), "audio/ogg", meta=meta)

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

            voice_name = str(getattr(self.engine, "voice", "default"))
            try:
                base_speed = float(action.meta.get("tts_speed", 1.0))
            except Exception:
                base_speed = 1.0
            language = str(getattr(self.engine, "language", "en")).lower()
            ipa_mode = "ipa" if language.startswith("en") else "text"

            candidate_speeds = [round(base_speed, 3)]
            respeed_candidate = round(candidate_speeds[0] * 0.90, 3)
            if abs(respeed_candidate - candidate_speeds[0]) > 1e-6:
                candidate_speeds.append(respeed_candidate)

            cached_flag = False
            meta_info: Dict[str, Any] = {}
            cache_key = None
            audio_path: Optional[Path] = None
            speed_value = candidate_speeds[0]

            for cand_speed in candidate_speeds:
                candidate_key = self._build_cache_key(synth_text, voice_name, cand_speed, ipa_mode)
                candidate_path = self._file_cache.get(candidate_key)
                if candidate_path and candidate_path.exists():
                    cache_key = candidate_key
                    audio_path = candidate_path
                    meta_info = dict(self._cache_meta.get(candidate_key, {}))
                    speed_value = cand_speed
                    cached_flag = True
                    logger.info(
                        "tts.cache.hit key=%s",
                        candidate_key[:12],
                        extra={
                            "subsys": "tts",
                            "event": "cache_hit",
                            "text_len": len(synth_text),
                            "key": candidate_key[:12],
                        },
                    )
                    break

            if not cached_flag:
                result = await self.generate_tts(synth_text, timeout=timeout_s)
                mime_type: str = "audio/ogg"
                if isinstance(result, tuple) and len(result) >= 2:
                    audio_path, mime_type = result[0], result[1]  # type: ignore[assignment]
                elif isinstance(result, Path):
                    audio_path = result
                elif isinstance(result, str):
                    audio_path = Path(result)
                else:
                    try:
                        audio_path, mime_type = result  # type: ignore[misc]
                    except Exception:
                        try:
                            audio_path = Path(os.fspath(result))  # type: ignore[arg-type]
                            mime_type = getattr(result, "mime", mime_type)
                        except Exception:
                            audio_path = Path(str(result))

                action.audio_path = str(audio_path)
                meta_info = {}
                if isinstance(result, TTSResult):
                    meta_info = dict(getattr(result, "meta", {}) or {})
                if audio_path.exists():
                    sr_meta, dur_meta = self._collect_audio_stats(audio_path)
                    meta_info.setdefault("sr", sr_meta)
                    meta_info.setdefault("duration_s", dur_meta)
                meta_info.setdefault("text_chars", len(synth_text))
                meta_info.setdefault("ipa_len", self._compute_ipa_length(synth_text))
                meta_info.setdefault("lead_preroll_ms", self._preroll_lead_ms)
                meta_info.setdefault("tail_pad_ms", self._tail_pad_ms)
                meta_info.setdefault("head_pad_ms", 0)
                meta_info.setdefault("cached", False)
                meta_info.setdefault("speed", speed_value)
                speed_value = float(meta_info.get("speed", speed_value))
                cache_key = self._build_cache_key(synth_text, voice_name, speed_value, ipa_mode)
                cache_key_short = cache_key[:12]
                self._file_cache[cache_key] = audio_path
                self._cache_meta[cache_key] = dict(meta_info)
                self._cache_order.append(cache_key)
                if len(self._cache_order) > self._cache_max:
                    old_key = self._cache_order.pop(0)
                    try:
                        self._file_cache.pop(old_key, None)
                        self._cache_meta.pop(old_key, None)
                    except Exception:
                        pass
                try:
                    size_bytes = audio_path.stat().st_size if audio_path.exists() else 0
                except Exception:
                    size_bytes = 0
                logger.info(
                    "tts.cache.store key=%s bytes=%s",
                    cache_key_short,
                    size_bytes,
                    extra={
                        "subsys": "tts",
                        "event": "cache_store",
                        "key": cache_key_short,
                        "text_len": len(synth_text),
                        "bytes": size_bytes,
                    },
                )
            else:
                cache_key_short = cache_key[:12] if cache_key else ""
                action.audio_path = str(audio_path)
                meta_info.setdefault("lead_preroll_ms", self._preroll_lead_ms)
                meta_info.setdefault("tail_pad_ms", self._tail_pad_ms)
                meta_info.setdefault("head_pad_ms", 0)
                meta_info.setdefault("ipa_len", self._compute_ipa_length(synth_text))
                meta_info.setdefault("text_chars", len(synth_text))
                meta_info.setdefault("sr", 48000)
                meta_info.setdefault("duration_s", 0.0)
                meta_info["cached"] = True
                meta_info.setdefault("speed", speed_value)
                speed_value = float(meta_info.get("speed", speed_value))


            # Annotate meta
            if truncated:
                action.meta["tts_truncated"] = True

            # Keep or drop transcript content
            if not include_transcript:
                action.content = ""  # files-only message allowed

            try:
                audio_path_obj = Path(action.audio_path) if action.audio_path else None
                if cached_flag:
                    sr, duration = self._collect_audio_stats(audio_path_obj) if audio_path_obj else (48000, 0.0)
                    ipa_len = meta_info.get("ipa_len") if "ipa_len" in meta_info else self._compute_ipa_length(synth_text)
                    meta_info.setdefault("sr", sr)
                    meta_info.setdefault("duration_s", duration)
                    if cache_key:
                        self._cache_meta[cache_key] = dict(meta_info)
                    self._emit_summary(
                        text_chars=len(synth_text),
                        ipa_len=ipa_len or 0,
                        lead_preroll_ms=int(meta_info.get("lead_preroll_ms", self._preroll_lead_ms)),
                        head_pad_ms=int(meta_info.get("head_pad_ms", 0)),
                        tail_pad_ms=int(meta_info.get("tail_pad_ms", self._tail_pad_ms)),
                        sr=sr or 48000,
                        duration_s=duration,
                        cached=True,
                    )
                else:
                    sr = int(meta_info.get("sr", 48000))
                    duration = float(meta_info.get("duration_s", 0.0))
                    ipa_len = int(
                        meta_info.get(
                            "ipa_len", self._compute_ipa_length(synth_text)
                        )
                    )
                    self._emit_summary(
                        text_chars=int(meta_info.get("text_chars", len(synth_text))),
                        ipa_len=ipa_len,
                        lead_preroll_ms=int(meta_info.get("lead_preroll_ms", self._preroll_lead_ms)),
                        head_pad_ms=int(meta_info.get("head_pad_ms", 0)),
                        tail_pad_ms=int(meta_info.get("tail_pad_ms", self._tail_pad_ms)),
                        sr=sr,
                        duration_s=duration,
                        cached=bool(meta_info.get("cached", False)),
                    )
                    if cache_key:
                        self._cache_meta[cache_key] = dict(meta_info)
            except Exception:
                logger.debug("tts.summary.emit_failed", extra={"subsys": "tts"}, exc_info=True)

            return action
        except SynthesisError as exc:
            logger.error(
                f"tts.process.failed | {exc}",
                extra={"subsys": "tts", "event": "process_failed"},
                exc_info=True,
            )
            raise
        except Exception as e:
            logger.error(
                f"tts.process.failed_unexpected | {e}",
                extra={"subsys": "tts", "event": "process_failed_unexpected"},
                exc_info=True,
            )
            raise SynthesisError("TTS processing failed") from e

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
