# -*- coding: utf-8 -*-
"""Load the official Kokoro IPA vocabulary at import time.

The upstream ``kokoro-onnx`` package exposes the authoritative phoneme mapping
via ``kokoro_onnx.config.DEFAULT_VOCAB``.  We import it here instead of keeping
our own hand-written table so that token IDs stay perfectly aligned with the
model.  When the dependency is unavailable (tests or minimal environments) we
expose an empty mapping and let callers detect the missing vocabulary.
"""

from __future__ import annotations

from typing import Dict
import logging

logger = logging.getLogger(__name__)

IS_PLACEHOLDER: bool = False
_IMPORT_ERROR: Exception | None = None

try:  # pragma: no cover - exercised in integration
    from kokoro_onnx import config as _kokoro_config  # type: ignore

    PHONEME_TO_ID: Dict[str, int] = dict(_kokoro_config.DEFAULT_VOCAB)
    EXPECTED_VOCAB_SIZE: int | None = len(PHONEME_TO_ID)
except Exception as exc:  # pragma: no cover - dependency missing in some tests
    IS_PLACEHOLDER = True
    _IMPORT_ERROR = exc
    PHONEME_TO_ID = {}
    EXPECTED_VOCAB_SIZE = None
    logger.debug("kokoro_onnx unavailable; IPA vocabulary disabled", exc_info=True)
else:
    logger.debug("Loaded official Kokoro IPA vocabulary with %d entries", len(PHONEME_TO_ID))


if PHONEME_TO_ID:
    _id_counts: Dict[int, str] = {}
    for phoneme, id_val in PHONEME_TO_ID.items():
        if id_val in _id_counts:
            raise ValueError(f"Duplicate ID {id_val} for phonemes '{_id_counts[id_val]}' and '{phoneme}'")
        _id_counts[id_val] = phoneme

    ID_TO_PHONEME: Dict[int, str] = {v: k for k, v in PHONEME_TO_ID.items()}
    MAX_ID = max(PHONEME_TO_ID.values())
else:
    ID_TO_PHONEME = {}
    MAX_ID = -1

__all__ = [
    "EXPECTED_VOCAB_SIZE",
    "ID_TO_PHONEME",
    "IS_PLACEHOLDER",
    "MAX_ID",
    "PHONEME_TO_ID",
]
