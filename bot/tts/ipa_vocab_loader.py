# -*- coding: utf-8 -*-
"""
IPA Vocabulary Loader with ONNX validation and greedy encoder.
Uses the hardcoded official Kokoro mapping from ipa_vocab_kokoro_v1.py.
"""

from __future__ import annotations
import re
import unicodedata
import numpy as np
from typing import NamedTuple
import logging

try:
    from onnxruntime import InferenceSession
except ImportError:
    # For testing without onnxruntime
    InferenceSession = object

from bot.tts.ipa_vocab_kokoro_v1 import PHONEME_TO_ID, EXPECTED_VOCAB_SIZE, ID_TO_PHONEME, MAX_ID

logger = logging.getLogger(__name__)


class UnsupportedIPASymbolError(ValueError):
    """Raised when IPA contains symbols not in the official vocabulary."""
    def __init__(self, unsupported_symbols: list[str]):
        self.unsupported_symbols = unsupported_symbols
        super().__init__(f"Unsupported IPA symbols: {unsupported_symbols}")


class Vocab(NamedTuple):
    """Official Kokoro IPA vocabulary with validation."""
    phoneme_to_id: dict[str, int]
    id_to_phoneme: list[str]
    rows: int


def _get_embedding_rows(session: InferenceSession) -> int:
    """Get embedding rows from ONNX session metadata or inputs."""
    # Try to find vocab size in model metadata first
    if hasattr(session, 'get_modelmeta'):
        try:
            meta = session.get_modelmeta()
            if meta and hasattr(meta, 'custom_metadata_map'):
                for key, value in meta.custom_metadata_map.items():
                    if key.lower() in {"vocab_size", "token_vocab_size"}:
                        try:
                            return int(value)
                        except ValueError:
                            pass
        except Exception:
            pass
    
    # Fallback: assume size from embedded mapping
    return len(PHONEME_TO_ID)


def load_vocab(session: InferenceSession) -> Vocab:
    """
    Load hardcoded official Kokoro IPA vocabulary with ONNX validation.
    
    Args:
        session: ONNX inference session for validation
        
    Returns:
        Vocab: Official vocabulary with validation
        
    Raises:
        RuntimeError: If vocabulary doesn't match model
    """
    # Use hardcoded mapping - no external file hunting
    p2i = dict(PHONEME_TO_ID)
    rows = _get_embedding_rows(session)
    
    # Validate expected size if specified
    if EXPECTED_VOCAB_SIZE is not None and len(p2i) != EXPECTED_VOCAB_SIZE:
        raise RuntimeError(f"Embedded IPA vocab size ({len(p2i)}) != EXPECTED_VOCAB_SIZE ({EXPECTED_VOCAB_SIZE}).")
    
    # Hard fail if vocab doesn't match model - prevents gibberish
    if len(p2i) != rows:
        raise RuntimeError(f"Kokoro IPA vocab size ({len(p2i)}) != model embedding rows ({rows}).")
    
    # Build reverse mapping
    id_to_phoneme = [""] * (max(p2i.values()) + 1)
    for phoneme, id_val in p2i.items():
        if id_val < 0:
            raise RuntimeError(f"Negative id for symbol {phoneme}: {id_val}")
        if id_val >= len(id_to_phoneme):
            # Extend array if needed
            id_to_phoneme.extend([""] * (id_val - len(id_to_phoneme) + 1))
        id_to_phoneme[id_val] = phoneme
    
    logger.debug(f"Hardcoded IPA vocab loaded: {len(p2i)} symbols, max_id={max(p2i.values())}")
    
    return Vocab(phoneme_to_id=p2i, id_to_phoneme=id_to_phoneme, rows=rows)


# Guarded rewrites: only perform if replacement exists in table AND source isn't already valid
GUARDED_REWRITES = (
    ("ɡ", "g"),   # Alternative g symbol  
    ("r", "ɹ"),   # English rhotic
    ("a", "ɑ"),   # Fix 'a' OOV: only if 'ɑ' exists and 'a' not in table
)


def encode_ipa(ipa: str, session: InferenceSession) -> list[int]:
    """
    Encode IPA string to token IDs using greedy longest-match.
    
    Args:
        ipa: IPA phoneme string
        session: ONNX session for vocab loading
        
    Returns:
        list[int]: Token IDs
        
    Raises:
        UnsupportedIPASymbolError: If any symbol cannot be encoded
    """
    vocab = load_vocab(session)
    s = unicodedata.normalize("NFC", ipa)
    
    # Apply guarded rewrites
    for src, dst in GUARDED_REWRITES:
        if src in s and src not in vocab.phoneme_to_id and dst in vocab.phoneme_to_id:
            s = s.replace(src, dst)
    
    # Precompute longest-first symbol list for greedy matching
    symbols = sorted(vocab.phoneme_to_id.keys(), key=len, reverse=True)
    out: list[int] = []
    i = 0
    N = len(s)
    
    while i < N:
        if s[i].isspace():
            i += 1
            continue
            
        matched = False
        for sym in symbols:
            if s.startswith(sym, i):
                out.append(vocab.phoneme_to_id[sym])
                i += len(sym)
                matched = True
                break
                
        if not matched:
            raise UnsupportedIPASymbolError([s[i]])
    
    return out
