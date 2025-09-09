# -*- coding: utf-8 -*-
"""
Thin adapter for kokoro-onnx to provide a stable import surface within our codebase.

Goals:
- Isolate direct imports of kokoro_onnx to this module only.
- Export a symbol named `Kokoro` so tests patching `bot.tts.engines.kokoro.Kokoro`
  remain fully compatible.
- Provide helper constructors for future evolution without changing call sites.
- Provide a safe submodule importer so other modules can access kokoro_onnx.*
  without importing kokoro_onnx directly.

This adapter intentionally keeps a minimal API surface. It returns the underlying
kokoro_onnx.Kokoro object when available, or raises a clear ImportError if the
package is not installed. Tests typically patch the `Kokoro` symbol so they do
not depend on the actual package.

Environment variables commonly used by the surrounding engine:
- KOKORO_MODEL_PATH, KOKORO_VOICES_PATH
- TTS_LANGUAGE, TTS_TOKENISER
- KOKORO_TTS_TIMEOUT_COLD, KOKORO_TTS_TIMEOUT_WARM
"""

from __future__ import annotations

from typing import Any, Optional
import logging
import importlib

__all__ = [
    "Kokoro",
    "get_kokoro_engine",
    "get_direct_wrapper",
    "import_kokoro_submodule",
]

logger = logging.getLogger(__name__)

try:
    # Import once here; rest of the codebase imports via this adapter
    from kokoro_onnx import Kokoro as _RealKokoro  # type: ignore

    _IMPORT_ERR: Exception | None = None
except Exception as e:  # pragma: no cover - exercised only when kokoro_onnx missing
    _RealKokoro = None  # type: ignore
    _IMPORT_ERR = e


def Kokoro(*args: Any, **kwargs: Any):  # pragma: no cover - thin wrapper
    """Factory returning the real kokoro_onnx.Kokoro when available.

    Keeping this as a function (not a class) ensures that tests using
    `patch('bot.tts.engines.kokoro.Kokoro')` still work, since they patch the
    symbol imported into that module. If kokoro_onnx is missing, raise a clear
    ImportError with the original exception chained.
    """
    if _RealKokoro is None:
        raise ImportError(
            "kokoro_onnx is not available. Install it or patch the Kokoro symbol in tests."
        ) from _IMPORT_ERR
    return _RealKokoro(*args, **kwargs)


def get_kokoro_engine(
    model_path: Optional[str] = None, voices_path: Optional[str] = None
):
    """Construct and return a Kokoro engine instance.

    This wrapper exists to centralize construction logic in case we need to
    inject provider options, patches, or environment toggles in one place.
    """
    return Kokoro(model_path=model_path, voices_path=voices_path)


def get_direct_wrapper():  # pragma: no cover - import shim
    """Return our local KokoroDirect wrapper class.

    Keeping this import localized avoids circular imports and keeps a single
    source of truth for KokoroDirect access.
    """
    from .kokoro_direct_fixed import KokoroDirect

    return KokoroDirect


def import_kokoro_submodule(name: str):  # pragma: no cover - import shim
    """Import and return a kokoro_onnx submodule by name via the adapter.

    Example: import_kokoro_submodule('tokenizer_registry') returns
    the module object kokoro_onnx.tokenizer_registry if kokoro_onnx is
    available, otherwise raises ImportError with the original cause chained.
    """
    # Ensure the base package is importable first
    if _RealKokoro is None:
        # Attempt to import the package to surface a better error message
        try:
            import kokoro_onnx  # noqa: F401
        except Exception as e:
            raise ImportError("kokoro_onnx is not available") from e
    try:
        return importlib.import_module(f"kokoro_onnx.{name}")
    except Exception as e:
        raise ImportError(f"Failed to import kokoro_onnx.{name}") from e
