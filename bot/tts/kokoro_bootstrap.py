"""
Bootstrap module for kokoro-onnx tokenizer integration.

This module registers the EspeakWrapper tokenizer under all required aliases
in the kokoro_onnx.tokenizer_registry.AVAILABLE_TOKENIZERS dictionary.
It must be imported before any other kokoro_onnx imports to ensure proper registration.
"""

import logging
import importlib
from typing import Dict, Type, List, Optional

# Define the list of aliases for the EspeakWrapper tokenizer
TOKENIZER_ALIASES = {
    "default": "espeak",
}

logger = logging.getLogger(__name__)

# We no longer need to register anything. The Tokenizer class now handles espeak internally.
