"""
Central exception hierarchy for the Discord bot.

All bot-specific exceptions should be defined here and imported
wherever needed.  Legacy local definitions in other modules should
be replaced with imports from this module.

Hierarchy (single-chain inheritance, no diamond MRO):

    BotBaseException                     (root for everything)
    |-- ConfigurationError               (config-only, not a runtime error)
    |-- BotError
    |   |-- BackendError
    |   |   |-- BackendTimeoutError
    |   |   |-- BackendRateLimitError
    |   |   |-- BackendUnavailableError
    |   |   |-- AllProvidersExhaustedError
    |   |   +-- APIError                (API / HTTP transport errors)
    |   |-- InferenceError
    |   |   +-- VisionError*            (* defined in vision.types, inherits BotError here)
    |   |-- MemoryError
    |   |-- PersistenceError            (canonical definition)
    |   |   +-- CorruptionError
    |   |   +-- AtomicWriteError
    |   +-- RAGIndexError
    |   +-- PermissionDeniedError
    |   +-- UrlSafetyError
    |   +-- CommandError
    +-- DispatchEmptyError
    +-- DispatchTypeError
    +-- TTSAudioError
    +-- FileProcessingError

PersistenceError is defined canonically here.
VisionError is defined in bot/vision/types.py and re-exported here.
"""

from __future__ import annotations

from typing import Any


class BotBaseException(Exception):
    """Base exception for all custom exceptions in this bot."""


# ------------------------------------------------------------------ #
#  Config-level                                                       #
# ------------------------------------------------------------------ #

class ConfigurationError(BotBaseException):
    """Raised for errors in bot configuration."""


# ------------------------------------------------------------------ #
#  Dispatch-level (not routed through BotError chain)                 #
# ------------------------------------------------------------------ #

class DispatchEmptyError(BotBaseException):
    """Raised when a dispatch returns no result."""


class DispatchTypeError(BotBaseException):
    """Raised when a dispatch returns an invalid type."""


# ------------------------------------------------------------------ #
#  TTS / file processing                                              #
# ------------------------------------------------------------------ #

class TTSAudioError(BotBaseException):
    """Raised for Text-to-Speech audio errors."""


class FileProcessingError(BotBaseException):
    """Raised for errors processing user-uploaded files."""


# ------------------------------------------------------------------ #
#  Domain errors under BotError                                       #
# ------------------------------------------------------------------ #

class BotError(BotBaseException):
    """General bot domain error; parent of most structured exceptions."""


class BackendError(BotError):
    """Base for backend/provider operation failures."""


class BackendTimeoutError(BackendError):
    """Raised when a backend request exceeds its deadline."""


class BackendRateLimitError(BackendError):
    """Raised when a backend returns a rate-limit response."""


class BackendUnavailableError(BackendError):
    """Raised when a backend is offline or unreachable."""


class AllProvidersExhaustedError(BackendError):
    """Raised when every configured backend has failed."""


class APIError(BackendError):
    """Raised for errors related to external API interactions."""


class InferenceError(BotError):
    """Raised for errors during model inference (text, vision, etc.)."""


class MemoryError(BotError):
    """Raised for memory subsystem failures."""


class PersistenceError(BotError):
    """Raised for persistence-layer failures (JSON, SQLite, etc.)."""


class AtomicWriteError(PersistenceError):
    """Raised when an atomic file write operation fails."""


class CorruptionError(PersistenceError):
    """Raised when data corruption is detected."""


class RAGIndexError(BotError):
    """Raised for RAG / vector index failures."""


class PermissionDeniedError(BotError):
    """Raised when a user lacks required permissions."""


class UrlSafetyError(BotError):
    """Raised for URL safety / SSRF violations."""


class CommandError(BotError):
    """Raised for command parsing / execution failures."""


# ------------------------------------------------------------------ #
#  Re-export: VisionError from bot.vision.types                       #
#  --------------------------------------------------------------- #
#  VisionError is a dataclass defined in vision/types.py.  That module #
#  imports BotError from here (fully defined above).  We re-import     #
#  the class to expose it as bot.exceptions.VisionError so that       #
#  tests doing `from bot.exceptions import VisionError` work.         #
#                                                                      #
#  We cannot import at module level because that would create a       #
#  cycle: exceptions.py → vision/types.py → exceptions.py.  Instead   #
#  we register a module-level __getattr__ to lazily resolve the name. #
# ------------------------------------------------------------------ #

# Placeholder sentinel for lazy resolution
_VisionError_UNRESOLVED = True


def __getattr__(name: str) -> Any:
    """Lazy attribute resolution for VisionError re-export."""
    if name.startswith("Vision") and _VisionError_UNRESOLVED:
        _resolve_vision_exceptions()
    if name in ("VisionError", "VisionModelUnavailableError",
                "VisionInputTooLargeError", "VisionUnsupportedMediaError",
                "VisionErrorType"):
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _resolve_vision_exceptions() -> None:
    """Import vision types and populate module-level names."""
    global _VisionError_UNRESOLVED
    _VisionError_UNRESOLVED = False

    from bot.vision import types as _vt

    # The canonical dataclass
    globals()["VisionError"] = _vt.VisionError
    globals()["VisionErrorType"] = _vt.VisionErrorType

    # Shortcut subclasses (if they exist in vision.types, use them;
    # otherwise we just leave them undefined since they are rare)
    for _name in ("VisionModelUnavailableError", "VisionInputTooLargeError",
                  "VisionUnsupportedMediaError"):
        if hasattr(_vt, _name):
            globals()[_name] = getattr(_vt, _name)
