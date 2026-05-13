"""Configuration loading and environment setup.

Refactored from a 1000+ line monolith (bot/config.py) into a domain-specific
package. All public symbols from the original bot/config.py are re-exported
here for backward compatibility.

Public API (from bot.config import ...):
    load_config
    get_vl_model_ladder
    invalidate_config_cache
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
    get_vl_model_ladder,
    invalidate_config_cache,
    KOKORO_FORCE_IPA_EN,
    # Internal cache state — re-exported for invalidation / test introspection
    _config_cache,
    _cache_timestamp,
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
    "get_vl_model_ladder",
    "invalidate_config_cache",
    "load_system_prompts",
    "audit_env_file",
    "validate_required_env",
    "validate_prompt_files",
    "check_venv_activation",
    "ConfigurationError",
    "KOKORO_FORCE_IPA_EN",
]
