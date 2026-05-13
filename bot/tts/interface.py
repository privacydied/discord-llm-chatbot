from __future__ import annotations

import asyncio
import inspect
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from bot.config import load_config
from bot.tts.errors import SynthesisError
from bot.utils.logging import get_logger
from utils.opus import transcode_to_ogg_opus
from .engines.kokoro import KokoroONNXEngine
from .engines.kokoro_v8 import KokoroV8Engine
from .engines.stub import StubEngine

ENGINES = {
    "stub": StubEngine,
    "kokoro-onnx": KokoroONNXEngine,
    "kokoro": KokoroV8Engine,
}


logger = get_logger(__name__)


def _default_cache_dir() -> Path:
    return Path(os.getenv("TEMP_DIR", "temp")) / "tts"


class TTSManager:
    """Async-first TTS manager used by the bot and tests.

    This is intentionally small and conservative: it loads one backend, can
    synthesize to WAV or OGG, and exposes a few compatibility helpers used by
    existing call sites.
    """

    def __init__(self, bot: Any | None = None, config: Optional[dict[str, Any]] = None):
        if config is None and isinstance(bot, dict):
            config = bot
            bot = None

        self.bot = bot
        self.config = dict(config or load_config())
        if bot is not None and hasattr(bot, "config"):
            try:
                self.config = dict(getattr(bot, "config"))
            except Exception:
                pass

        self.enabled = bool(self.config.get("TTS_ENABLE", True))
        env_engine = (
            (self.config.get("TTS_ENGINE") or os.getenv("TTS_ENGINE") or "")
            .strip()
            .lower()
        )
        model_hint = self.config.get("TTS_MODEL_PATH") or os.getenv("TTS_MODEL_PATH")
        voices_hint = self.config.get("TTS_VOICES_PATH") or os.getenv("TTS_VOICES_PATH")
        has_local_assets = bool(model_hint and voices_hint)
        if not env_engine:
            default_model = Path("tts/kokoro-v1.0.onnx")
            default_voices = Path("tts/voices-v1.0.bin")
            has_local_assets = has_local_assets or (
                default_model.exists() and default_voices.exists()
            )
        self.backend = env_engine or ("kokoro-onnx" if has_local_assets else "stub")
        self.voice = str(
            self.config.get("TTS_VOICE") or os.getenv("TTS_VOICE") or "af_heart"
        )
        self.voices: list[str] = []
        self.available = False
        self.degraded = False
        self.degraded_reason: Optional[str] = None
        self._warmed_up = False
        self._engine = None
        self._current_jobs: dict[str, asyncio.Task[Any]] = {}
        self._job_locks: dict[str, asyncio.Lock] = {}
        self.cache_dir = Path(self.config.get("TTS_CACHE_DIR") or _default_cache_dir())
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.timeout_s = float(
            os.getenv("TTS_TIMEOUT_S", self.config.get("TTS_TIMEOUT_S", 25.0))
        )
        self.timeout_cold_s = float(
            os.getenv("TTS_TIMEOUT_COLD_S", self.config.get("TTS_TIMEOUT_COLD_S", 10.0))
        )
        self.timeout_warm_s = float(
            os.getenv("TTS_TIMEOUT_WARM_S", self.config.get("TTS_TIMEOUT_WARM_S", 2.0))
        )

        self._warmup_status = "not_started"  # not_started | running | complete | failed
        self._warmup_task: asyncio.Task[Any] | None = None

        if self.enabled:
            try:
                self._load_backend()
            except Exception as exc:
                self.degraded = True
                self.degraded_reason = str(exc)
                self.available = False
                logger.error("Failed to initialize TTS backend: %s", exc, exc_info=True)

        # Start warm-up in background after backend is loaded (only if event loop is running)
        if self.is_available():
            try:
                loop = asyncio.get_running_loop()
                self._warmup_task = loop.create_task(self._warm_up(), name="tts-warmup")
            except RuntimeError:
                # No running event loop (e.g. during tests or synchronous init).
                # Warm-up will happen lazily on first synthesize() call.
                pass

    async def _warm_up(self) -> None:
        """Synthesise a short phrase after model load to avoid cold-start latency.

        Runs as a background task; failure is non-fatal and logged structurally.
        Sets _warmup_status to 'complete' or 'failed'.
        Uses TTS_TIMEOUT_COLD_S as the budget (default 10 s).
        """
        self._warmup_status = "running"
        logger.info("tts:warmup start backend=%s timeout=%.0fs", self.backend, self.timeout_cold_s)
        try:
            timeout = max(self.timeout_cold_s, 60.0)
            await asyncio.wait_for(
                self.synthesize("hello"),
                timeout=timeout,
            )
            self._warmed_up = True
            self._warmup_status = "complete"
            logger.info("tts:warmup ok")
        except asyncio.TimeoutError:
            self._warmup_status = "failed"
            logger.warning("tts:warmup timeout after %.0fs", timeout)
        except Exception as exc:
            self._warmup_status = "failed"
            logger.warning("tts:warmup error=%s", type(exc).__name__)
        # In all cases: non-fatal, normal synthesis proceeds with cold/warm timeout.

    def is_available(self) -> bool:
        return bool(
            self.enabled
            and self.available
            and self._engine is not None
            and not self.degraded
        )

    def get_status(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "available": bool(self.is_available()),
            "degraded": bool(self.degraded),
            "degraded_reason": self.degraded_reason,
            "explicit_stub": self.backend == "stub",
            "backend": self.backend,
            "engine": self.backend,
            "voice": self.voice,
            "cache_dir": str(self.cache_dir),
            "warmed_up": bool(self._warmed_up),
            "warmup_status": self._warmup_status,
        }

    def get_cache_stats(self) -> dict[str, Any]:
        files = list(self.cache_dir.glob("*")) if self.cache_dir.exists() else []
        total_bytes = 0
        for path in files:
            try:
                if path.is_file():
                    total_bytes += path.stat().st_size
            except Exception:
                continue
        return {
            "files": len([p for p in files if p.is_file()]),
            "size_mb": round(total_bytes / (1024 * 1024), 3),
            "cache_dir": str(self.cache_dir),
        }

    def purge_old_cache(self) -> None:
        if not self.cache_dir.exists():
            return
        cutoff = time.time() - 7 * 24 * 60 * 60
        for path in self.cache_dir.glob("*"):
            try:
                if path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink(missing_ok=True)
            except Exception:
                logger.debug("Failed to purge cache file %s", path, exc_info=True)

    def _load_backend(self) -> None:
        engine_cls = ENGINES.get(self.backend)
        if engine_cls is None:
            raise RuntimeError(f"unsupported tts backend: {self.backend}")

        if self.backend == "stub":
            self._engine = engine_cls()
            self.available = True
            return

        if self.backend in {"kokoro", "kokoro-v8"}:
            self._engine = engine_cls(voice=self.voice)
            self.available = True
            return

        # Default to kokoro-onnx.
        model_path = self.config.get("TTS_MODEL_PATH") or os.getenv("TTS_MODEL_PATH")
        voices_path = self.config.get("TTS_VOICES_PATH") or os.getenv("TTS_VOICES_PATH")
        if not model_path or not voices_path:
            raise RuntimeError("kokoro assets missing")
        try:
            self._engine = engine_cls(
                model_path=str(model_path),
                voices_path=str(voices_path),
                voice=self.voice,
            )
        except TypeError:
            self._engine = engine_cls(
                model_path=str(model_path),
                voices_path=str(voices_path),
            )
        self.available = True

    async def _ensure_engine(self) -> None:
        if self._engine is not None and self.available:
            return
        if not self.enabled:
            raise SynthesisError("TTS is disabled")
        try:
            self._load_backend()
        except Exception as exc:
            self.degraded = True
            self.degraded_reason = str(exc)
            raise SynthesisError(str(exc)) from exc
        if not self._engine or not self.available:
            raise SynthesisError(self.degraded_reason or "TTS backend unavailable")

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"https?://\S+", "", text or "")
        text = re.sub(r"[`*_>#~]|\[(.*?)\]\((.*?)\)", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    async def synthesize(self, text: str, timeout: float | None = None) -> bytes:
        await self._ensure_engine()
        cleaned = self._clean_text(text)
        if not cleaned:
            raise ValueError("Text cannot be empty")

        engine = self._engine
        if engine is None:
            raise SynthesisError("TTS engine is unavailable")

        try:
            result = engine.synthesize(cleaned)
            if inspect.isawaitable(result):
                if timeout is not None:
                    result = await asyncio.wait_for(result, timeout=timeout)
                else:
                    result = await result
            elif timeout is not None:
                result = await asyncio.wait_for(
                    asyncio.to_thread(lambda: result), timeout=timeout
                )

            audio_bytes = self._coerce_audio_bytes(result)
            self._warmed_up = True
            return audio_bytes
        except SynthesisError:
            raise
        except Exception as exc:
            self.degraded = True
            self.degraded_reason = str(exc)
            raise SynthesisError(str(exc)) from exc

    def _coerce_audio_bytes(self, result: Any) -> bytes:
        if result is None:
            raise SynthesisError("TTS backend returned no audio")
        if isinstance(result, (bytes, bytearray)):
            return bytes(result)
        if isinstance(result, (str, Path)):
            path = Path(result)
            if not path.exists():
                raise SynthesisError(f"Audio path does not exist: {path}")
            return path.read_bytes()
        if isinstance(result, tuple) and result:
            return self._coerce_audio_bytes(result[0])
        raise SynthesisError(f"Unsupported audio result type: {type(result)!r}")

    async def generate_tts(
        self,
        text: str,
        voice: Optional[str] = None,
        *,
        out_path: str | Path | None = None,
        timeout: float | None = None,
        output_format: str = "wav",
    ) -> Path:
        await self._ensure_engine()
        cleaned = self._clean_text(text)
        if not cleaned:
            raise ValueError("Text cannot be empty")

        if timeout is None:
            timeout = (
                self.timeout_cold_s if not self._warmed_up else self.timeout_warm_s
            )

        chosen_voice = voice or self.voice
        old_voice = getattr(self._engine, "voice", None)
        if hasattr(self._engine, "voice"):
            try:
                setattr(self._engine, "voice", chosen_voice)
            except Exception:
                pass

        try:
            audio_bytes = await self.synthesize(cleaned, timeout=timeout)
        finally:
            if hasattr(self._engine, "voice") and old_voice is not None:
                try:
                    setattr(self._engine, "voice", old_voice)
                except Exception:
                    pass

        output_format = (output_format or "wav").strip().lower()
        if out_path is None:
            suffix = ".ogg" if output_format == "ogg" else ".wav"
            fd, temp_name = tempfile.mkstemp(
                prefix="tts_", suffix=suffix, dir=str(self.cache_dir)
            )
            os.close(fd)
            out_path = Path(temp_name)
        else:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)

        if output_format == "ogg":
            # If the engine returned a valid WAV payload, transcode it; otherwise
            # preserve the bytes for tests and degraded paths.
            if (
                len(audio_bytes) >= 44
                and audio_bytes[:4] == b"RIFF"
                and audio_bytes[8:12] == b"WAVE"
            ):
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False, dir=str(self.cache_dir)
                ) as tmp:
                    tmp.write(audio_bytes)
                    wav_path = Path(tmp.name)
                try:
                    final_path = await transcode_to_ogg_opus(wav_path, out_path)
                    self._warmed_up = True
                    return final_path
                finally:
                    try:
                        wav_path.unlink(missing_ok=True)
                    except Exception:
                        pass
            out_path.write_bytes(audio_bytes)
            self._warmed_up = True
            return out_path

        out_path.write_bytes(audio_bytes)
        self._warmed_up = True
        return out_path

    async def process(self, action: Any) -> Any:
        meta = getattr(action, "meta", None) or {}
        text = getattr(action, "content", "") or ""
        explicit_timeout = meta.get("tts_timeout_s")
        if explicit_timeout is not None:
            timeout = float(explicit_timeout)
        else:
            cold = meta.get("tts_cold")
            if cold is None:
                cold = not self._warmed_up
            cold = bool(cold)
            timeout = float(
                meta.get(
                    "tts_timeout_cold_s" if cold else "tts_timeout_warm_s",
                    self.timeout_cold_s if cold else self.timeout_warm_s,
                )
            )

        audio_path = await self.generate_tts(
            text,
            getattr(self, "voice", None),
            timeout=timeout,
        )
        try:
            setattr(action, "audio_path", str(audio_path))
        except Exception:
            pass
        return action

    async def close(self) -> None:
        for task in list(self._current_jobs.values()):
            task.cancel()
        self._current_jobs.clear()
        engine = self._engine
        self._engine = None
        self.available = False
        if engine is None:
            return
        close = getattr(engine, "close", None)
        if close is None:
            return
        try:
            result = close()
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.debug("TTS engine close failed", exc_info=True)

    async def __aenter__(self) -> "TTSManager":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()
