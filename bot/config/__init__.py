"""Configuration loading and environment setup.

Refactored from a 1000+ line monolith (bot/config.py) into a domain-specific
package. All public symbols from the original bot/config.py are re-exported
here for backward compatibility.

Public API (from bot.config import ...):
    load_config
    load_config_candidate
    validate_config_candidate
    get_last_good_config
    get_vl_model_ladder
    invalidate_config_cache
    get_config_cache
    get_cache_timestamp
    load_system_prompts
    audit_env_file
    validate_required_env
    validate_prompt_files
    check_venv_activation
    ConfigurationError  (re-exported from bot.exceptions)
    KOKORO_FORCE_IPA_EN
"""

from __future__ import annotations

# Core config loading + helpers (extracted from flat bot/config.py)
from ._base import (
    load_config,
    load_config_candidate,
    validate_config_candidate,
    get_last_good_config,
    get_vl_model_ladder,
    invalidate_config_cache,
    get_config_cache,
    get_cache_timestamp,
    KOKORO_FORCE_IPA_EN,
    # Context trimming limits
    CONTEXT_MAX_MESSAGES,
    CONTEXT_MAX_CHARS_PER_MESSAGE,
    CONTEXT_MAX_TOTAL_CHARS,
    CONTEXT_IGNORE_BOT_CONTINUATION_CHUNKS,
    # LOW_RESOURCE helpers [Phase 17-23]
    _low_resource_int,  # noqa: F401
    _low_resource_float,  # noqa: F401
    _low_resource_bool,  # noqa: F401
    # Startup / validation helpers (still in _base; extracted incrementally)
    audit_env_file,
    validate_required_env,
    validate_prompt_files,
    check_venv_activation,
)

# System prompt loading (extracted to dedicated submodule)
from .prompts import load_system_prompts

# ConfigurationError is canonical in bot.exceptions; re-export for convenience
from bot.exceptions import ConfigurationError

__all__ = [
    "load_config",
    "load_config_candidate",
    "validate_config_candidate",
    "get_last_good_config",
    "get_vl_model_ladder",
    "invalidate_config_cache",
    "get_config_cache",
    "get_cache_timestamp",
    "load_system_prompts",
    "audit_env_file",
    "validate_required_env",
    "validate_prompt_files",
    "check_venv_activation",
    "ConfigurationError",
    "KOKORO_FORCE_IPA_EN",
    # Context trimming limits
    "CONTEXT_MAX_MESSAGES",
    "CONTEXT_MAX_CHARS_PER_MESSAGE",
    "CONTEXT_MAX_TOTAL_CHARS",
    "CONTEXT_IGNORE_BOT_CONTINUATION_CHUNKS",
]
