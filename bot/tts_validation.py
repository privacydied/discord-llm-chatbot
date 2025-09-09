"""
Shim module providing legacy TTS validation API expected by tests.

Implements tokenizer discovery/selection utilities and two globals:
- AVAILABLE_TOKENIZERS: set of discovered tokenizers
- TOKENIZER_WARNING_SHOWN: flag toggled when a warning message is surfaced

This intentionally mirrors a simplified subset of the old interface
so tests importing `bot.tts_validation` continue to work.
"""

from __future__ import annotations

import logging
import os
import subprocess
import shutil
import importlib.util
from enum import Enum
from typing import Dict, Set, Optional, Any

from .tts_errors import MissingTokeniserError
from .tts.validation import (
    detect_gibberish_audio,
    detect_gibberish_audio_with_metrics,
    validate_voice_vector,
    check_sample_rate_consistency,
)

logger = logging.getLogger(__name__)

# Global tokenizer availability cache
AVAILABLE_TOKENIZERS: Set[str] = set()
# Global flag to indicate whether we've already shown a tokenizer warning
TOKENIZER_WARNING_SHOWN: bool = False

# Language-to-tokenizer mapping
TOKENISER_MAP = {
    "en": ["espeak", "espeak-ng", "phonemizer", "g2p_en"],
    "ja": ["misaki"],
    "zh": ["misaki"],
    # Default for other languages
    "*": ["phonemizer", "espeak", "espeak-ng"],
}

# Default to grapheme for all languages
DEFAULT_TOKENIZER = "grapheme"


class TokenizerType(Enum):
    ESPEAK = "espeak"
    ESPEAK_NG = "espeak-ng"
    PHONEMIZER = "phonemizer"
    G2P_EN = "g2p_en"
    MISAKI = "misaki"
    GRAPHEME = "grapheme"
    UNKNOWN = "unknown"


def dump_environment_diagnostics() -> Dict[str, Any]:
    """Collect environment diagnostics to aid tokenizer discovery.

    Returns a dict with binary/module availability hints. Tests may patch
    this function directly, so keep its name and shape stable.
    """
    diagnostics: Dict[str, Any] = {
        "espeak_binary": shutil.which("espeak"),
        "espeak_ng_binary": shutil.which("espeak-ng"),
        "phonemizer_module": importlib.util.find_spec("phonemizer") is not None,
        "g2p_en_module": importlib.util.find_spec("g2p_en") is not None,
        "misaki_module": importlib.util.find_spec("misaki") is not None,
        # Minimal extras for logging/debug
        "PATH": os.environ.get("PATH", "").split(os.pathsep),
        "site_packages": [],
    }
    return diagnostics


def detect_available_tokenizers() -> Dict[str, bool]:
    """Detect which tokenizers are available in the current environment.

    Populates the global AVAILABLE_TOKENIZERS set.
    """
    global AVAILABLE_TOKENIZERS

    diagnostics = dump_environment_diagnostics()

    available = {
        TokenizerType.ESPEAK.value: False,
        TokenizerType.ESPEAK_NG.value: False,
        TokenizerType.PHONEMIZER.value: False,
        TokenizerType.G2P_EN.value: False,
        TokenizerType.MISAKI.value: False,
        TokenizerType.GRAPHEME.value: True,  # Grapheme is always available
    }

    # espeak
    if diagnostics.get("espeak_binary"):
        try:
            res = subprocess.run(
                ["espeak", "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            available[TokenizerType.ESPEAK.value] = res.returncode == 0
        except (FileNotFoundError, subprocess.SubprocessError):
            pass

    # espeak-ng
    if diagnostics.get("espeak_ng_binary"):
        try:
            res = subprocess.run(
                ["espeak-ng", "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            available[TokenizerType.ESPEAK_NG.value] = res.returncode == 0
        except (FileNotFoundError, subprocess.SubprocessError):
            pass

    # phonemizer
    if diagnostics.get("phonemizer_module"):
        available[TokenizerType.PHONEMIZER.value] = True

    # g2p_en
    if diagnostics.get("g2p_en_module"):
        available[TokenizerType.G2P_EN.value] = True

    # misaki
    if diagnostics.get("misaki_module"):
        available[TokenizerType.MISAKI.value] = True

    # Update global set
    # IMPORTANT: mutate in place to preserve object identity for external imports
    AVAILABLE_TOKENIZERS.clear()
    AVAILABLE_TOKENIZERS.update(name for name, ok in available.items() if ok)

    if AVAILABLE_TOKENIZERS - {TokenizerType.GRAPHEME.value}:
        logger.info("Tokenisers found: %s", sorted(AVAILABLE_TOKENIZERS))
    else:
        logger.warning("No known tokenization methods found")

    # Warn about invalid env override
    env_tok = os.environ.get("TTS_TOKENISER")
    if env_tok and env_tok not in AVAILABLE_TOKENIZERS:
        logger.error(
            "TTS_TOKENISER '%s' is not available. Available: %s",
            env_tok,
            sorted(AVAILABLE_TOKENIZERS),
        )

    return available


def select_tokenizer_for_language(
    language: str, available_tokenizers: Optional[Dict[str, bool]] = None
) -> str:
    """Select the best tokenizer for the given language.

    If available_tokenizers is provided, use it; otherwise, use the
    global AVAILABLE_TOKENIZERS set.
    """
    language = (language or "").lower().strip()

    if available_tokenizers is None:
        available_set = AVAILABLE_TOKENIZERS
    else:
        available_set = {k for k, v in available_tokenizers.items() if v}

    # Environment override
    env_tok = os.environ.get("TTS_TOKENISER")
    if env_tok:
        if env_tok in available_set:
            logger.info("Using tokenizer from TTS_TOKENISER: %s", env_tok)
            return env_tok
        else:
            logger.error("TTS_TOKENISER '%s' is not available", env_tok)
            # continue with auto-selection

    # Preference by language prefix
    lang_key = language[:2] if language else "*"
    preferred = TOKENISER_MAP.get(lang_key, TOKENISER_MAP["*"])

    for tok in preferred:
        if tok in available_set:
            logger.info("Selected tokenizer '%s' for language '%s'", tok, language)
            return tok

    # For English, require a phonetic tokenizer
    if language.startswith("en") and not any(
        t in available_set for t in ["espeak", "espeak-ng", "phonemizer", "g2p_en"]
    ):
        required = ["espeak-ng", "phonemizer", "g2p_en"]
        raise MissingTokeniserError(language, list(available_set), required)

    # Fallback to grapheme
    logger.warning(
        "No preferred tokenizer available for '%s', using grapheme fallback", language
    )
    return DEFAULT_TOKENIZER


def is_tokenizer_warning_needed() -> bool:
    """Whether a user-facing tokenizer warning should be shown now.

    Uses TTS_LANGUAGE env (default 'en'). Only show once until cleared
    by `get_tokenizer_warning_message()`.
    """
    lang = os.environ.get("TTS_LANGUAGE", "en").lower().strip()

    # Note: We intentionally do not gate this by TOKENIZER_WARNING_SHOWN to avoid
    # cross-call/process leakage; this function should reflect current availability.

    if lang.startswith("en"):
        return not any(
            t in AVAILABLE_TOKENIZERS
            for t in {"espeak", "espeak-ng", "phonemizer", "g2p_en"}
        )
    if lang.startswith("ja") or lang.startswith("zh"):
        return "misaki" not in AVAILABLE_TOKENIZERS
    # Other languages: warn if nothing except grapheme
    return AVAILABLE_TOKENIZERS == {"grapheme"} or not AVAILABLE_TOKENIZERS


def get_tokenizer_warning_message(language: str = "en") -> str:
    """Return a user-friendly warning message and mark it as shown."""
    global TOKENIZER_WARNING_SHOWN
    TOKENIZER_WARNING_SHOWN = True

    lang = (language or "en").split("-")[0].lower()

    if lang == "en":
        return (
            "⚠️ You are missing a phonetic tokeniser for English. "
            "Install espeak, phonemizer, or g2p_en for better quality."
        )
    elif lang in {"ja", "zh"}:
        return (
            "⚠️ No Asian language tokenizer found. Speech quality may be reduced. "
            "Install misaki for better results."
        )
    else:
        return (
            "⚠️ You are missing a phonetic tokeniser for this language. "
            "Install phonemizer or espeak/espeak-ng for better quality."
        )


# Public API for legacy imports (explicit)
__all__ = [
    # Tokenizer discovery/selection
    "TokenizerType",
    "detect_available_tokenizers",
    "select_tokenizer_for_language",
    "is_tokenizer_warning_needed",
    "get_tokenizer_warning_message",
    "AVAILABLE_TOKENIZERS",
    "TOKENIZER_WARNING_SHOWN",
    # Audio/voice validation utilities (re-exported)
    "detect_gibberish_audio",
    "detect_gibberish_audio_with_metrics",
    "validate_voice_vector",
    "check_sample_rate_consistency",
]
