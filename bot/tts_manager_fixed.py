"""
Compatibility shim for legacy imports expecting `bot.tts_manager_fixed`.

Provides a TTSManager with the minimal interface expected by tests and older
code: environment variable handling, tokenizer registry bootstrap, KokoroDirect
loading, and a synchronous generate_speech() wrapper that returns a Path.
"""
from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Import KokoroDirect here so tests can patch 'bot.tts_manager_fixed.KokoroDirect'
from .kokoro_direct_fixed import KokoroDirect

logger = logging.getLogger(__name__)

# Defaults aligned with asset manager [CMV]
DEFAULT_MODEL_PATH = "tts/kokoro-v1.0.onnx"
DEFAULT_VOICES_PATH = "tts/voices-v1.0.bin"


class TTSManager:
    """
    Minimal TTS manager compatible with legacy tests and code.

    Responsibilities:
    - Resolve model/voices paths from env or config with new→old precedence
    - Initialize tokenizer registry (no-op friendly)
    - Lazily load KokoroDirect on first use
    - Provide synchronous generate_speech() that returns a Path
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = config or {}

        # Public attributes expected by tests/scripts
        self.backend: str = (
            str(self.config.get("TTS_BACKEND") or os.getenv("TTS_BACKEND") or "kokoro-onnx")
        )
        self.voice: str = str(self.config.get("TTS_VOICE") or os.getenv("TTS_VOICE") or "default")
        self.available: bool = False

        # Internal state
        self.kokoro: Optional[KokoroDirect] = None

        # Best‑effort tokenizer registry init (safe if patched in tests)
        try:
            self._init_tokenizer_registry()
        except Exception as e:  # [REH]
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
        except Exception as e:  # [REH]
            logger.info(
                f"Tokenizer registry unavailable: {e}",
                extra={"subsys": "tts", "event": "manager.registry_init.unavailable"},
            )

    def _resolve_paths(self) -> Tuple[str, str]:
        """
        Resolve model and voices paths with precedence:
        1) New env vars: TTS_MODEL_PATH, TTS_VOICES_PATH
        2) Old env vars: TTS_MODEL_FILE, TTS_VOICE_FILE
        3) Config nested: config['tts']['model_path'|'voices_path']
        4) Config flat: config['TTS_MODEL_PATH'|'TTS_VOICES_PATH'|'TTS_MODEL_FILE'|'TTS_VOICE_FILE']
        5) Reasonable defaults
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
            model_path = (
                self.config.get("TTS_MODEL_PATH")
                or self.config.get("TTS_MODEL_FILE")
            )
        if not voices_path:
            voices_path = (
                self.config.get("TTS_VOICES_PATH")
                or self.config.get("TTS_VOICE_FILE")
            )

        # 5) Defaults
        model_path = str(model_path or DEFAULT_MODEL_PATH)
        voices_path = str(voices_path or DEFAULT_VOICES_PATH)

        logger.debug(
            f"Resolved model_path={model_path}, voices_path={voices_path}",
            extra={"subsys": "tts", "event": "manager.paths"},
        )
        return model_path, voices_path

    def _load_kokoro(self, model_path: str, voices_path: str) -> KokoroDirect:
        """Create KokoroDirect instance. Broken out for test patching."""
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

    def generate_speech(
        self, text: str, voice: Optional[str] = None, *, out_path: Optional[Path] = None
    ) -> Path:
        """
        Generate speech synchronously using KokoroDirect.create.

        Args:
            text: Text to synthesize
            voice: Optional voice id/name. Defaults to manager.voice
            out_path: Optional explicit output path for WAV file

        Returns:
            Path to generated WAV file
        """
        if self.kokoro is None:
            self.load_model()
        if not self.kokoro:
            raise RuntimeError("TTS engine not available")  # [REH]

        chosen_voice = voice or self.voice
        logger.debug(
            f"Generating speech (voice={chosen_voice})",
            extra={"subsys": "tts", "event": "manager.generate"},
        )
        return self.kokoro.create(text, chosen_voice, out_path=out_path)


__all__ = ["TTSManager", "KokoroDirect"]
