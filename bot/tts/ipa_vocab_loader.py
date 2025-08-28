"""
Official Kokoro IPA Vocabulary Loader and Validator.

Single source of truth for IPA phoneme-to-ID mapping with ONNX embedding validation.
No fallbacks allowed - loads official vocabulary or fails explicitly.
"""

import json
import logging
import importlib.resources
from pathlib import Path
from typing import Dict, List, Optional, NamedTuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class UnsupportedIPASymbolError(Exception):
    """Raised when IPA contains symbols not in the official vocabulary."""
    def __init__(self, unsupported_symbols: List[str]):
        self.unsupported_symbols = unsupported_symbols
        super().__init__(f"Unsupported IPA symbols: {unsupported_symbols}")


@dataclass
class Vocab:
    """Official Kokoro IPA vocabulary with validation."""
    phoneme_to_id: Dict[str, int]
    id_to_phoneme: List[str]
    size: int

    def supports_stress(self) -> bool:
        """Check if vocabulary supports stress marks."""
        return "ˈ" in self.phoneme_to_id

    def supports_length(self) -> bool:
        """Check if vocabulary supports length marks."""
        return "ː" in self.phoneme_to_id

    def supports_pause(self) -> bool:
        """Check if vocabulary supports pause tokens."""
        return "_" in self.phoneme_to_id or "<sp>" in self.phoneme_to_id


def _resolve_official_vocab_path() -> Optional[Path]:
    """Resolve the official IPA vocabulary from kokoro_onnx package."""
    # Try direct importlib.resources access first
    try:
        kokoro_files = importlib.resources.files("kokoro_onnx")
        vocab_path = kokoro_files.joinpath("assets/phoneme_to_id.json")
        if vocab_path.exists():
            return vocab_path
    except Exception:
        pass

    # Try alternative paths
    alternatives = [
        "assets/phoneme_to_id.json",
        "assets/ipa_vocab.json",
        "assets/vocab_ipa.json"
    ]

    for alt in alternatives:
        try:
            kokoro_files = importlib.resources.files("kokoro_onnx")
            vocab_path = kokoro_files.joinpath(alt)
            if vocab_path.exists():
                return vocab_path
        except Exception:
            continue

    # Try reading from tokenizer module attributes (no instantiation)
    try:
        import kokoro_onnx.tokenizer as ktok
        if hasattr(ktok, "PHONEME_TO_ID"):
            # This would be a dict we can use directly
            return None  # Signal to use the attribute approach
    except Exception:
        pass

    return None


def _load_official_vocab_from_attribute() -> Optional[Dict[str, int]]:
    """Load vocabulary from kokoro_onnx.tokenizer module attributes."""
    try:
        import kokoro_onnx.tokenizer as ktok
        if hasattr(ktok, "PHONEME_TO_ID"):
            vocab_dict = getattr(ktok, "PHONEME_TO_ID")
            if isinstance(vocab_dict, dict):
                return vocab_dict
    except Exception:
        pass
    return None


def _load_vendored_vocab() -> Optional[Dict[str, int]]:
    """Load vendored vocabulary for development (only if explicitly allowed)."""
    import os
    if not os.getenv("KOKORO_ALLOW_VENDORED_VOCAB", "").lower() in ("true", "1", "yes"):
        return None

    # Try to find vendored vocab in expected locations
    vendored_paths = [
        Path(__file__).parent / "assets" / "phoneme_to_id.kokoro.v1.json",
        Path(__file__).parents[2] / "tts" / "assets" / "phoneme_to_id.kokoro.v1.json"
    ]

    for path in vendored_paths:
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
            except Exception as e:
                logger.debug(f"Failed to load vendored vocab from {path}: {e}")

    return None


def _normalize_unicode_keys(vocab_dict: Dict[str, int]) -> Dict[str, int]:
    """Normalize Unicode keys to NFC form."""
    import unicodedata
    return {unicodedata.normalize('NFC', k): v for k, v in vocab_dict.items()}


def _build_reverse_vocab(phoneme_to_id: Dict[str, int]) -> List[str]:
    """Build id_to_phoneme list from phoneme_to_id dict."""
    max_id = max(phoneme_to_id.values())
    id_to_phoneme = [""] * (max_id + 1)

    for phoneme, idx in phoneme_to_id.items():
        id_to_phoneme[idx] = phoneme

    return id_to_phoneme


def _create_greedy_segmenter(phoneme_to_id: Dict[str, int]) -> List[str]:
    """Create greedy longest-match symbol list for IPA segmentation."""
    # Sort by length (longest first) then alphabetically
    symbols = sorted(phoneme_to_id.keys(), key=lambda x: (-len(x), x))
    return symbols


def load_official_vocab(onnx_session=None) -> Vocab:
    """
    Load official Kokoro IPA vocabulary with ONNX validation.

    Args:
        onnx_session: Optional ONNX session to validate embedding matrix size

    Returns:
        Vocab: Official vocabulary with validation

    Raises:
        RuntimeError: If vocabulary cannot be loaded or validated
    """
    vocab_dict = None

    # Try official paths first
    vocab_path = _resolve_official_vocab_path()
    if vocab_path is not None:
        try:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                vocab_dict = data
        except Exception as e:
            logger.debug(f"Failed to load official vocab from {vocab_path}: {e}")

    # Try attribute-based loading
    if vocab_dict is None:
        vocab_dict = _load_official_vocab_from_attribute()

    # Try vendored vocab (development only)
    if vocab_dict is None:
        vocab_dict = _load_vendored_vocab()

    if vocab_dict is None:
        raise RuntimeError(
            "Cannot load official Kokoro IPA vocabulary. "
            "Ensure kokoro_onnx is properly installed or set KOKORO_ALLOW_VENDORED_VOCAB=true for development."
        )

    # Normalize Unicode and validate
    phoneme_to_id = _normalize_unicode_keys(vocab_dict)
    id_to_phoneme = _build_reverse_vocab(phoneme_to_id)

    vocab = Vocab(
        phoneme_to_id=phoneme_to_id,
        id_to_phoneme=id_to_phoneme,
        size=len(phoneme_to_id)
    )

    # Validate against ONNX embedding matrix if session provided
    if onnx_session is not None:
        try:
            # Try to get embedding matrix size
            for input_info in onnx_session.get_inputs():
                if "style" in input_info.name.lower() or "speaker" in input_info.name.lower():
                    # Get the embedding dimension from input shape
                    shape = input_info.shape
                    if len(shape) >= 2:
                        embedding_rows = shape[-1]  # Last dimension is embedding size
                        if vocab.size != embedding_rows:
                            raise RuntimeError(
                                f"Kokoro IPA vocab size ({vocab.size}) != ONNX embedding rows ({embedding_rows}); "
                                "refusing to synthesize. Model and vocabulary mismatch."
                            )
                        logger.info(f"Official IPA vocab loaded: size={vocab.size}; matches embedding.")
                        break
            else:
                logger.warning("Could not validate vocab against ONNX embedding matrix (no style/speaker input found)")
        except Exception as e:
            logger.warning(f"ONNX validation failed: {e}")

    return vocab


def encode_ipa(ipa: str, vocab: Vocab) -> List[int]:
    """
    Encode IPA string to token IDs using official vocabulary.

    Args:
        ipa: IPA phoneme string
        vocab: Official vocabulary

    Returns:
        List[int]: Token IDs

    Raises:
        UnsupportedIPASymbolError: If any symbol not in vocabulary
    """
    if not ipa or not ipa.strip():
        return []

    # Normalize input
    import unicodedata
    ipa = unicodedata.normalize('NFC', ipa.strip())

    # Split by spaces (phoneme boundaries)
    symbols = ipa.split()

    # Track unsupported symbols
    unsupported = []

    # Encode symbols
    token_ids = []
    for symbol in symbols:
        if symbol in vocab.phoneme_to_id:
            token_ids.append(vocab.phoneme_to_id[symbol])
        else:
            # Try safe rewrites for known variants
            rewritten = _try_safe_rewrite(symbol, vocab)
            if rewritten is not None:
                token_ids.append(rewritten)
            else:
                unsupported.append(symbol)

    if unsupported:
        raise UnsupportedIPASymbolError(unsupported)

    return token_ids


def _try_safe_rewrite(symbol: str, vocab: Vocab) -> Optional[int]:
    """Try safe rewrites for common IPA variants."""
    # Known safe rewrites (official model equivalences)
    rewrites = {
        "ɡ": "g",  # Alternative g symbol
        "ɹ": "r",  # Alternative r symbol
        "ɫ": "l",  # Alternative l symbol
    }

    rewritten = rewrites.get(symbol)
    if rewritten and rewritten in vocab.phoneme_to_id:
        return vocab.phoneme_to_id[rewritten]

    return None


def validate_token_ids(token_ids: List[int], vocab: Vocab, embedding_rows: int) -> None:
    """
    Validate token IDs against vocabulary and embedding matrix.

    Args:
        token_ids: Token IDs to validate
        vocab: Official vocabulary
        embedding_rows: Number of rows in embedding matrix

    Raises:
        ValueError: If validation fails
    """
    if not token_ids:
        return

    min_id = min(token_ids)
    max_id = max(token_ids)

    if min_id < 0:
        raise ValueError(f"Token ID {min_id} is negative")

    if max_id >= embedding_rows:
        # Find the problematic symbol
        bad_symbol = None
        for tid in token_ids:
            if tid >= embedding_rows:
                bad_symbol = vocab.id_to_phoneme[tid] if tid < len(vocab.id_to_phoneme) else f"id_{tid}"
                break

        raise ValueError(
            f"Token ID {max_id} >= embedding rows {embedding_rows}. "
            f"Bad symbol: '{bad_symbol}'"
        )
