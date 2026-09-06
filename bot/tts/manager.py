"""Canonical TTS manager for the Discord bot.

Exposes ``TTSManager`` with environment variable resolution, tokenizer
registry bootstrap, lazy ``KokoroDirect`` loading, and a synchronous
``generate_speech`` helper that returns a :class:`pathlib.Path`.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bot.config import _low_resource_bool, _low_resource_int

if TYPE_CHECKING:
    from .kokoro_direct import KokoroDirect

# ---------- Lazy accessor to defer heavy numpy / onnxruntime import ----------


def _kokoro_direct():
    """Lazy accessor — deferred KokoroDirect load (carries numpy)."""
    if "_kokoro_cls" not in globals():
        from .kokoro_direct import KokoroDirect as _kd

        globals()["_kokoro_cls"] = _kd
    return globals()["_kokoro_cls"]


logger = logging.getLogger(__name__)

# Defaults aligned with asset manager [CMV]
DEFAULT_MODEL_PATH = "tts/kokoro-v1.0.onnx"
DEFAULT_VOICES_PATH = "tts/voices-v1.0.bin"

# Resource caps [Phase 12-16]
_TTS_MAX_CHARS = _low_resource_int("TTS_MAX_CHARS", 4000, 2000)
_TTS_SKIP_LONG_RESPONSES = _low_resource_bool("TTS_SKIP_LONG_RESPONSES", False, True)

# Warmup configuration
_TTS_SKIP_WARMUP = _low_resource_bool("TTS_SKIP_WARMUP", False, True)
_WARMUP_TIMEOUT = 60  # seconds budget for warmup synthesis


class TTSManager:
    """Minimal TTS manager compatible with legacy tests and code.

    Responsibilities:
    - Resolve model/voices paths from env or config with new→old precedence
    - Initialize tokenizer registry (no-op friendly)
    - Lazily load KokoroDirect on first use
    - Provide synchronous generate_speech() that returns a Path
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = config or {}

        # Public attributes expected by tests/scripts
        self.backend: str = str(self.config.get("TTS_BACKEND") or os.getenv("TTS_BACKEND") or "kokoro-onnx")
        self.voice: str = str(self.config.get("TTS_VOICE") or os.getenv("TTS_VOICE") or "default")
        self.available: bool = False

        # Internal state
        self.kokoro: KokoroDirect | None = None
        self._warmup_status: str = "not_started"
        self._last_used: float = time.monotonic()

        # Best‑effort tokenizer registry init (safe if patched in tests)
        try:
            self._init_tokenizer_registry()
        except Exception as e:  # noqa: BLE001 - best-effort init, safe under test patching (logged below) [REH]
            logger.debug(
                f"Tokenizer registry init skipped: {e}",
                extra={"subsys": "tts", "event": "manager.registry_init.skip"},
            )

    # ----- Initialization helpers -----
    def _init_tokenizer_registry(self) -> None:
        """Initialize tokenizer registry discovery. Safe to call multiple times."""
        try:
            from .tokenizer_registry import TokenizerRegistry

            registry = TokenizerRegistry.get_instance()
            registry.discover_tokenizers()
            logger.debug(
                "Tokenizer registry initialized",
                extra={"subsys": "tts", "event": "manager.registry_init"},
            )
        except Exception as e:  # noqa: BLE001 - third-party registry discovery; logged, TTS works without it [REH]
            logger.info(
                f"Tokenizer registry unavailable: {e}",
                extra={"subsys": "tts", "event": "manager.registry_init.unavailable"},
            )

    def _resolve_paths(self) -> tuple[str, str]:
        """Resolve model and voices paths with precedence:
        1) New env vars: TTS_MODEL_PATH, TTS_VOICES_PATH
        2) Old env vars: TTS_MODEL_FILE, TTS_VOICE_FILE
        3) Config nested: config['tts']['model_path'|'voices_path']
        4) Config flat: config['TTS_MODEL_PATH'|'TTS_VOICES_PATH'|'TTS_MODEL_FILE'|'TTS_VOICE_FILE']
        5) Reasonable defaults.
        """
        # 1) New env
        model_path = os.getenv("TTS_MODEL_PATH")
        voices_path = os.getenv("TTS_VOICES_PATH")

        # 2) Old env (fallback)
        # If new env vars are not set OR equal to our known defaults, prefer old env if provided.
        old_model_env = os.getenv("TTS_MODEL_FILE")
        old_voices_env = os.getenv("TTS_VOICE_FILE")
        if (not model_path or model_path == DEFAULT_MODEL_PATH) and old_model_env:
            model_path = old_model_env
        if (not voices_path or voices_path == DEFAULT_VOICES_PATH) and old_voices_env:
            voices_path = old_voices_env

        # 3) Config nested
        tts_cfg = self.config.get("tts") or {}
        if not model_path:
            model_path = tts_cfg.get("model_path")
        if not voices_path:
            voices_path = tts_cfg.get("voices_path")

        # 4) Config flat fallbacks
        if not model_path:
            model_path = self.config.get("TTS_MODEL_PATH") or self.config.get("TTS_MODEL_FILE")
        if not voices_path:
            voices_path = self.config.get("TTS_VOICES_PATH") or self.config.get("TTS_VOICE_FILE")

        # 5) Defaults
        model_path = str(model_path or DEFAULT_MODEL_PATH)
        voices_path = str(voices_path or DEFAULT_VOICES_PATH)

        logger.debug(
            f"Resolved model_path={model_path}, voices_path={voices_path}",
            extra={"subsys": "tts", "event": "manager.paths"},
        )
        return model_path, voices_path

    def _load_kokoro(self, model_path: str, voices_path: str) -> KokoroDirect:
        """Create KokoroDirect instance. Broken out for test patching.

        Resolves KokoroDirect through the module so that
        ``@patch('bot.tts.manager.KokoroDirect')`` works correctly.
        """
        import bot.tts.manager as _mgr  # type: ignore[import-not-checked]

        KokoroDirect = getattr(_mgr, "KokoroDirect", _kokoro_direct())
        logger.info(
            "Loading KokoroDirect",
            extra={
                "subsys": "tts",
                "event": "manager.kokoro.load",
                "model_path": model_path,
                "voices_path": voices_path,
            },
        )
        return KokoroDirect(model_path=model_path, voices_path=voices_path)

    # ----- Warmup -----
    @property
    def warmup_status(self) -> str:
        """Return current warmup state: 'not_started', 'running', 'complete', or 'failed'."""
        return self._warmup_status

    def get_status(self) -> dict[str, Any]:
        """Return a dict of TTS subsystem status for !status display and diagnostics."""
        return {
            "available": self.available,
            "engine": self.backend,
            "loaded": self.kokoro is not None,
            "warmup_status": self._warmup_status,
        }

    def start_warmup(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Schedule TTS warmup as a non-blocking background task.

        If TTS_SKIP_WARMUP is True (default in LOW_RESOURCE_MODE), warmup is skipped
        entirely.  Otherwise a trivial ``"ready"`` string is synthesised through the
        normal KokoroDirect pipeline to warm the ONNX model cache before real user
        requests arrive.  The operation runs in the default ThreadPoolExecutor, is
        non-fatal, and uses a 60-second budget.
        """
        if _TTS_SKIP_WARMUP:
            logger.info(
                "TTS warmup skipped (TTS_SKIP_WARMUP=True)",
                extra={"subsys": "tts", "event": "manager.warmup.skip"},
            )
            return

        event_loop = loop or asyncio.get_running_loop()

        def _run_warmup() -> None:
            import concurrent.futures

            logger.info(
                "TTS warmup started",
                extra={"subsys": "tts", "event": "manager.warmup.start"},
            )
            self._warmup_status = "running"
            try:
                # Reuse the existing lazy-load path
                self.load_model()
                if not self.kokoro:
                    msg = "TTS engine not available for warmup"
                    raise RuntimeError(msg)

                warmup_path = Path(tempfile.mkdtemp()) / "warmup.wav"

                # Run synthesis in a dedicated sub-executor so we can enforce the
                # 60 s budget without leaking background threads on timeout.
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(
                        lambda p=warmup_path: self.kokoro.create(  # type: ignore[union-attr]
                            "ready",
                            self.voice,
                            out_path=p,
                        ),
                    )
                    fut.result(timeout=_WARMUP_TIMEOUT)

                # Discard the generated audio
                with contextlib.suppress(Exception):
                    warmup_path.unlink(missing_ok=True)

                self._warmup_status = "complete"
                logger.info(
                    "TTS warmup completed successfully",
                    extra={"subsys": "tts", "event": "manager.warmup.done"},
                )

            except concurrent.futures.TimeoutError:
                self._warmup_status = "failed"
                logger.warning(
                    "TTS warmup timed out after %ds",
                    _WARMUP_TIMEOUT,
                    extra={"subsys": "tts", "event": "manager.warmup.timeout", "timeout_s": _WARMUP_TIMEOUT},
                )
            except Exception as exc:  # noqa: BLE001 - warmup worker failure; logged, status=failed fallback
                self._warmup_status = "failed"
                logger.warning(
                    "TTS warmup failed: %s",
                    exc,
                    extra={"subsys": "tts", "event": "manager.warmup.fail", "error": str(exc)},
                )
                # Non-fatal: do not re-raise

        event_loop.run_in_executor(None, _run_warmup)

    # ----- Public API -----
    def load_model(self) -> None:
        """Load KokoroDirect if not already loaded."""
        if self.kokoro is not None:
            return
        model_path, voices_path = self._resolve_paths()
        self.kokoro = self._load_kokoro(model_path, voices_path)
        self.available = self.kokoro is not None
        logger.debug(
            f"TTS available={self.available}",
            extra={"subsys": "tts", "event": "manager.available"},
        )

    def unload_if_idle(self, idle_seconds: float) -> bool:
        """Release the Kokoro ONNX session after prolonged idle. [PA]

        The ~310 MB model + session buffers otherwise stay resident forever
        after the first TTS request. `generate_speech` lazily reloads on next
        use (a few seconds). In-flight synthesis holds its own reference to
        the session, so this never breaks active work. Returns True if unloaded.
        """
        if idle_seconds <= 0 or self.kokoro is None:
            return False
        if time.monotonic() - self._last_used < idle_seconds:
            return False
        self.kokoro = None
        logger.info(
            "tts.model.unload_idle | idle_s=%.0f",
            idle_seconds,
            extra={"subsys": "tts", "event": "tts.model.unload_idle"},
        )
        return True

    def generate_speech(self, text: str, voice: str | None = None, *, out_path: Path | None = None) -> Path:
        """Generate speech synchronously using KokoroDirect.create.

        Args:
            text: Text to synthesize
            voice: Optional voice id/name. Defaults to manager.voice
            out_path: Optional explicit output path for WAV file

        Returns:
            Path to generated WAV file

        """
        self._last_used = time.monotonic()
        if self.kokoro is None:
            self.load_model()
        # Local strong reference: idle-TTL unload may null self.kokoro from the
        # event loop while this runs in a worker thread; the captured ref keeps
        # the session alive for the duration of this synthesis. [REH][PA]
        kokoro = self.kokoro
        if not kokoro:
            msg = "TTS engine not available"
            raise RuntimeError(msg)  # [REH]

        # Enforce TTS text length cap [Phase 12-16]
        if len(text) > _TTS_MAX_CHARS:
            if _TTS_SKIP_LONG_RESPONSES:
                logger.info(
                    "tts:skip_long_response len=%d max=%d",
                    len(text),
                    _TTS_MAX_CHARS,
                )
                msg = f"TTS text exceeds {_TTS_MAX_CHARS} chars; skipped (TTS_SKIP_LONG_RESPONSES=True)"
                raise RuntimeError(msg)
            logger.warning(
                "tts:trim_text len=%d max=%d",
                len(text),
                _TTS_MAX_CHARS,
            )
            text = text[:_TTS_MAX_CHARS]

        chosen_voice = voice or self.voice
        logger.debug(
            f"Generating speech (voice={chosen_voice})",
            extra={"subsys": "tts", "event": "manager.generate"},
        )
        return kokoro.create(text, chosen_voice, out_path=out_path)


# Lazy re-export for KokoroDirect so external modules can still do
# ``from bot.tts.manager import KokoroDirect`` without triggering the
# heavy numpy/onnxruntime import chain at module load time.
def __getattr__(name):
    if name == "KokoroDirect":
        return _kokoro_direct()
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = ["KokoroDirect", "TTSManager"]
