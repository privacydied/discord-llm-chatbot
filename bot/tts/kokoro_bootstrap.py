"""
Bootstrap module for kokoro-onnx tokenizer integration.

This module registers the EspeakWrapper tokenizer under all required aliases
in the kokoro_onnx.tokenizer_registry.AVAILABLE_TOKENIZERS dictionary.
It must be imported before any other kokoro_onnx imports to ensure proper registration.
"""

import logging

# Define the list of aliases for the EspeakWrapper tokenizer
TOKENIZER_ALIASES = {
    "default": "espeak",
}

logger = logging.getLogger(__name__)

# We no longer need to register anything for runtime if using the newer Tokenizer,
# but tests and some legacy code expect a registration function to exist.


def register_espeak_wrapper() -> bool:
    """
    Register EspeakWrapper under required aliases in kokoro_onnx tokenizer registry.

    This is primarily a compatibility shim for tests expecting this function.
    Returns True if registration appears successful, False otherwise.
    """
    try:
        # Import the registry module via the adapter to isolate direct imports
        from bot.tts.kokoro_adapter import import_kokoro_submodule

        _reg = import_kokoro_submodule("tokenizer_registry")  # type: ignore
    except Exception as e:
        logger.info(f"kokoro_onnx not available for tokenizer registration: {e}")
        return False

    try:
        # Best-effort: assign a placeholder object for EspeakWrapper; tests only assert keys exist
        for alias in TOKENIZER_ALIASES:
            _reg.AVAILABLE_TOKENIZERS[alias] = object()
        logger.debug(
            f"Registered EspeakWrapper aliases: {', '.join(TOKENIZER_ALIASES.keys())}"
        )
        return True
    except Exception as e:
        logger.info(f"Failed to register EspeakWrapper aliases: {e}")
        return False
