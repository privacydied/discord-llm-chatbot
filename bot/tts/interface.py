from __future__ import annotations

import asyncio
import contextlib
import hashlib
import inspect
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Self

from bot.config import load_config
from bot.tts.errors import SynthesisError
from bot.utils.logging import get_logger
from utils.opus import transcode_to_ogg_opus

from .engines.pocket import DEFAULT_VOICE, PocketTTSEngine
from .engines.stub import StubEngine

ENGINES = {
    "stub": StubEngine,
    "pocket": PocketTTSEngine,
}
DEFAULT_BACKEND = "pocket"
# Kokoro was removed; old .env values select its replacement instead of failing.
LEGACY_BACKENDS = {"kokoro", "kokoro-onnx", "kokoro-v8"}


logger = get_logger(__name__)

# Pre-compiled text-cleaning patterns, hoisted to module scope so they are
# compiled once at import rather than on every _clean_text() call. [PA]
_URL_RE = re.compile(r"https?://\S+")
_MARKDOWN_RE = re.compile(r"[`*_>#~]|\[(.*?)\]\((.*?)\)")
_WHITESPACE_RE = re.compile(r"\s+")
# Stretched words ("Nnnnnnniiiii", "hahahahaha") have no split point: Pocket's
# ~50-token chunker turned one into a 391-token chunk and the model churned
# until the timeout. Collapse runs and cap word length before synthesis. [REH]
_CHAR_RUN_RE = re.compile(r"([^\W\d_])\1{2,}", re.IGNORECASE)  # letters only: "2000" stays
_PATTERN_RUN_RE = re.compile(r"((?:[^\W\d_]){2,4}?)\1{2,}", re.IGNORECASE)
_LONG_WORD_RE = re.compile(r"\S{41,}")
MAX_TTS_WORD_CHARS = 40
# Pocket's English model cannot say CJK/Cyrillic/Arabic/etc., and a run with no
# sentence punctuation becomes one huge chunk (2026-10-24: the Heart Sutra in
# Chinese = 815 tokens, pinyin = 867, max 50) that churns until the timeout, so
# the reply fell back to text. Drop unspeakable scripts; break long runs. [REH]
_UNSPEAKABLE_RE = re.compile(r"[^\x00-\u024f\u1e00-\u1eff\u2000-\u206f\u20a0-\u20cf\s]+")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?;:])\s+")
MAX_WORDS_PER_SENTENCE = 25


def _default_cache_dir() -> Path:
    """Default TTS scratch/cache directory.

    Defaults to the system temp dir (tmpfs/RAM-backed on most hosts, confirmed on
    this deployment) rather than a repo-relative "temp/" path -- these files are
    synthesized, uploaded to Discord, and are disposable, so there's no reason for
    them to ever touch the spinning-disk-backed project directory. [PA]
    Set TTS_CACHE_DIR explicitly to opt back into a persistent, disk-backed cache.
    """
    return Path(os.getenv("TEMP_DIR") or tempfile.gettempdir()) / "discord-llm-chatbot-tts"


def _break_long_runs(text: str) -> str:
    """Add a full stop every MAX_WORDS_PER_SENTENCE words inside unpunctuated runs."""
    out = []
    for sentence in _SENTENCE_END_RE.split(text):
        words = sentence.split(" ")
        parts = [" ".join(words[i : i + MAX_WORDS_PER_SENTENCE]) for i in range(0, len(words), MAX_WORDS_PER_SENTENCE)]
        out.append(". ".join(p.rstrip(",") for p in parts))
    return " ".join(out)


class TTSManager:
    """Async-first TTS manager used by the bot and tests.

    This is intentionally small and conservative: it loads one backend, can
    synthesize to WAV or OGG, and exposes a few compatibility helpers used by
    existing call sites.
    """

    def __init__(self, bot: Any | None = None, config: dict[str, Any] | None = None) -> None:
        if config is None and isinstance(bot, dict):
            config = bot
            bot = None

        self.bot = bot
        self.config = dict(config or load_config())
        if bot is not None and hasattr(bot, "config"):
            with contextlib.suppress(Exception):
                self.config = dict(bot.config)

        self.enabled = bool(self.config.get("TTS_ENABLE", True))
        env_engine = (self.config.get("TTS_ENGINE") or os.getenv("TTS_ENGINE") or self.config.get("TTS_BACKEND") or "").strip().lower()
        if env_engine in LEGACY_BACKENDS:
            logger.warning("tts:legacy_backend %s removed; using %s", env_engine, DEFAULT_BACKEND)
            env_engine = DEFAULT_BACKEND
        self.backend = env_engine or DEFAULT_BACKEND
        self.voice = str(self.config.get("TTS_VOICE") or os.getenv("TTS_VOICE") or DEFAULT_VOICE)
        self.voices: list[str] = []
        self.available = False
        self.degraded = False
        self.degraded_reason: str | None = None
        self._warmed_up = False
        self._engine = None
        self._current_jobs: dict[str, asyncio.Task[Any]] = {}
        self._job_locks: dict[str, asyncio.Lock] = {}
        self.cache_dir = Path(self.config.get("TTS_CACHE_DIR") or _default_cache_dir())
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.timeout_s = float(os.getenv("TTS_TIMEOUT_S", self.config.get("TTS_TIMEOUT_S", 25.0)))
        self.timeout_cold_s = float(os.getenv("TTS_TIMEOUT_COLD_S", self.config.get("TTS_TIMEOUT_COLD_S", 10.0)))
        self.timeout_warm_s = float(os.getenv("TTS_TIMEOUT_WARM_S", self.config.get("TTS_TIMEOUT_WARM_S", 2.0)))

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

        # Start warm-up in background after backend is loaded (only if event loop is running).
        # Pocket starts its server on first use unless POCKET_TTS_KEEP_WARM is set
        # (every-reply-is-voice setups): then it loads at boot and never idles out.
        warm_pocket = self.backend == "pocket" and getattr(self._engine, "keep_warm", False)
        if self.is_available() and (self.backend != "pocket" or warm_pocket):
            try:
                loop = asyncio.get_running_loop()
                warmer = self.warm() if warm_pocket else self._warm_up()
                self._warmup_task = loop.create_task(warmer, name="tts-warmup")
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
        except TimeoutError:
            self._warmup_status = "failed"
            logger.warning("tts:warmup timeout after %.0fs", timeout)
        except Exception as exc:  # noqa: BLE001
            self._warmup_status = "failed"
            logger.warning("tts:warmup error=%s", type(exc).__name__)
        # In all cases: non-fatal, normal synthesis proceeds with cold/warm timeout.

    def is_available(self) -> bool:
        return bool(self.enabled and self.available and self._engine is not None and not self.degraded)

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
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"cache stat failed: {exc}")
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
            except Exception:  # noqa: BLE001
                logger.debug("Failed to purge cache file %s", path, exc_info=True)

    @staticmethod
    def _cache_key(cleaned_text: str, voice: str, output_format: str) -> str:
        """Deterministic cache key for (text, voice, format) so repeated phrases reuse
        an existing rendered file instead of re-synthesizing and re-writing. [PA]
        """
        h = hashlib.sha256(usedforsecurity=False)
        h.update(cleaned_text.encode("utf-8"))
        h.update(b"\x00")
        h.update(voice.encode("utf-8"))
        h.update(b"\x00")
        h.update(output_format.encode("utf-8"))
        return h.hexdigest()[:24]

    def _load_backend(self) -> None:
        engine_cls = ENGINES.get(self.backend)
        if engine_cls is None:
            msg = f"unsupported tts backend: {self.backend}"
            raise RuntimeError(msg)

        if self.backend == "stub":
            self._engine = engine_cls()
            self.available = True
            return

        self._engine = engine_cls(self.config, voice=self.voice)
        self.available = True

    def unload_if_idle(self, ttl_s: float) -> bool:
        """Idle-TTL hook (bot/tasks.py): stop a bot-owned Pocket server. [PA]"""
        engine = self._engine
        due = getattr(engine, "idle_stop_due", None)
        if engine is None or due is None or not due(ttl_s):
            return False
        try:
            asyncio.get_running_loop().create_task(engine.close(), name="tts-idle-stop")
        except RuntimeError:
            return False
        return True

    async def _ensure_engine(self) -> None:
        if self._engine is not None and self.available:
            return
        if not self.enabled:
            msg = "TTS is disabled"
            raise SynthesisError(msg)
        try:
            self._load_backend()
        except Exception as exc:
            self.degraded = True
            self.degraded_reason = str(exc)
            raise SynthesisError(str(exc)) from exc
        if not self._engine or not self.available:
            raise SynthesisError(self.degraded_reason or "TTS backend unavailable")

    def _clean_text(self, text: str) -> str:
        text = _URL_RE.sub("", text or "")
        text = _MARKDOWN_RE.sub(" ", text)
        text = _CHAR_RUN_RE.sub(lambda m: m.group(0)[:2], text)
        text = _PATTERN_RUN_RE.sub(lambda m: m.group(1) * 2, text)
        text = _LONG_WORD_RE.sub(lambda m: m.group(0)[:MAX_TTS_WORD_CHARS], text)
        text = _UNSPEAKABLE_RE.sub(" ", text)
        text = _WHITESPACE_RE.sub(" ", text).strip()
        if not any(ch.isalnum() for ch in text):
            return ""
        return _break_long_runs(text)

    async def synthesize(self, text: str, timeout: float | None = None) -> bytes:
        await self._ensure_engine()
        cleaned = self._clean_text(text)
        if not cleaned:
            msg = "Text cannot be empty"
            raise ValueError(msg)

        engine = self._engine
        if engine is None:
            msg = "TTS engine is unavailable"
            raise SynthesisError(msg)

        try:
            result = await self._run_engine_synthesize(engine, cleaned, timeout)
            audio_bytes = self._coerce_audio_bytes(result)
            self._warmed_up = True
            return audio_bytes
        except SynthesisError:
            raise
        except Exception as exc:
            self.degraded = True
            self.degraded_reason = str(exc)
            raise SynthesisError(str(exc)) from exc

    @staticmethod
    async def _run_engine_synthesize(engine: Any, cleaned: str, timeout: float | None) -> Any:
        """Invoke engine.synthesize without blocking the event loop.

        Async engines are awaited directly; synchronous engines are offloaded to a
        worker thread via ``asyncio.to_thread`` (the previous ``to_thread(lambda: result)``
        ran the synthesis inline first, so the offload was a no-op). ``timeout``, when
        set, bounds the whole operation via ``asyncio.wait_for``. [PA][REH]
        """
        # Engines that know their own cost (Pocket: cold start + text length) widen
        # short warm/cold defaults so a long reply is not cut off mid-synthesis.
        budget = getattr(engine, "timeout_budget", None)
        if timeout is not None and budget is not None:
            timeout = max(timeout, float(budget(cleaned)))

        if inspect.iscoroutinefunction(engine.synthesize):
            awaitable = engine.synthesize(cleaned)
        else:
            awaitable = asyncio.to_thread(engine.synthesize, cleaned)

        result = await (asyncio.wait_for(awaitable, timeout=timeout) if timeout is not None else awaitable)

        # A sync engine may still hand back a coroutine/future; resolve it too.
        if inspect.isawaitable(result):
            result = await (asyncio.wait_for(result, timeout=timeout) if timeout is not None else result)
        return result

    def _coerce_audio_bytes(self, result: Any) -> bytes:
        if result is None:
            msg = "TTS backend returned no audio"
            raise SynthesisError(msg)
        if isinstance(result, (bytes, bytearray)):
            return bytes(result)
        if isinstance(result, (str, Path)):
            path = Path(result)
            if not path.exists():
                msg = f"Audio path does not exist: {path}"
                raise SynthesisError(msg)
            return path.read_bytes()
        if isinstance(result, tuple) and result:
            return self._coerce_audio_bytes(result[0])
        msg = f"Unsupported audio result type: {type(result)!r}"
        raise SynthesisError(msg)

    async def generate_tts(
        self,
        text: str,
        voice: str | None = None,
        *,
        out_path: str | Path | None = None,
        timeout: float | None = None,
        output_format: str = "wav",
    ) -> Path:
        await self._ensure_engine()
        cleaned = self._clean_text(text)
        if not cleaned:
            msg = "Text cannot be empty"
            raise ValueError(msg)

        if timeout is None:
            timeout = self.timeout_cold_s if not self._warmed_up else self.timeout_warm_s

        chosen_voice = voice or self.voice
        output_format = (output_format or "wav").strip().lower()

        # Cache-hit fast path: identical (text, voice, format) already rendered and
        # still sitting in cache_dir -- skip synthesis and the disk write entirely.
        # Only applies when the caller wants a cache-managed path (out_path is None);
        # an explicit out_path means the caller owns that exact location and always
        # wants a fresh write there. [PA]
        cache_key: str | None = None
        cached_path: Path | None = None
        lock: asyncio.Lock | None = None
        if out_path is None:
            suffix = ".ogg" if output_format == "ogg" else ".wav"
            cache_key = self._cache_key(cleaned, chosen_voice, output_format)
            cached_path = self.cache_dir / f"tts_{cache_key}{suffix}"
            lock = self._job_locks.setdefault(cache_key, asyncio.Lock())

        if cached_path is not None and lock is not None:
            async with lock:
                if cached_path.exists():
                    with contextlib.suppress(Exception):
                        os.utime(cached_path, None)  # bump mtime so purge_old_cache sees it as fresh
                    logger.debug("tts:cache_hit path=%s", cached_path)
                    return cached_path

                out_path = await self._synthesize_to_path(cleaned, chosen_voice, timeout, output_format, cached_path)
                return out_path

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return await self._synthesize_to_path(cleaned, chosen_voice, timeout, output_format, out_path)

    async def _synthesize_to_path(
        self,
        cleaned: str,
        chosen_voice: str,
        timeout: float | None,
        output_format: str,
        out_path: Path,
    ) -> Path:
        old_voice = getattr(self._engine, "voice", None)
        if hasattr(self._engine, "voice"):
            with contextlib.suppress(Exception):
                self._engine.voice = chosen_voice

        try:
            audio_bytes = await self.synthesize(cleaned, timeout=timeout)
        finally:
            if hasattr(self._engine, "voice") and old_voice is not None:
                with contextlib.suppress(Exception):
                    self._engine.voice = old_voice

        if output_format == "ogg":
            # If the engine returned a valid WAV payload, transcode it; otherwise
            # preserve the bytes for tests and degraded paths.
            if len(audio_bytes) >= 44 and audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE":
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=str(self.cache_dir)) as tmp:
                    tmp.write(audio_bytes)
                    wav_path = Path(tmp.name)
                try:
                    final_path = await transcode_to_ogg_opus(wav_path, out_path)
                    self._warmed_up = True
                    return final_path
                finally:
                    with contextlib.suppress(Exception):
                        wav_path.unlink(missing_ok=True)
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
                ),
            )

        audio_path = await self.generate_tts(
            text,
            getattr(self, "voice", None),
            timeout=timeout,
        )
        with contextlib.suppress(Exception):
            action.audio_path = str(audio_path)
        return action

    async def warm(self) -> bool:
        """Preload the engine when it is configured to stay warm. Best effort. [PA]"""
        engine = self._engine
        if engine is None or not getattr(engine, "keep_warm", False) or not callable(getattr(engine, "warm", None)):
            return False
        try:
            await engine.warm()
            return True
        except Exception as exc:  # noqa: BLE001 - a failed warm-up retries on first use
            logger.warning("tts:warm_failed %s", exc)
            return False

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
        except Exception:  # noqa: BLE001
            logger.debug("TTS engine close failed", exc_info=True)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()
