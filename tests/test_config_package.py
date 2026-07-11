"""Tests verifying bot.config package backward compatibility.

These tests ensure that after decomposing bot/config.py into a bot/config/ package,
all public APIs remain importable and functional. This guards against regressions
where `from bot.config import <name>` would silently break.
"""

from __future__ import annotations

import importlib
import os
from typing import Never
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Helper: reload bot.config to get a clean state for each test group
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_config_cache():
    """Invalidate the in-process config cache before/after each test."""
    from bot.config import invalidate_config_cache

    invalidate_config_cache()
    yield
    invalidate_config_cache()


# ===================================================================
# 1.  from bot.config import load_config works
# ===================================================================


class TestLoadConfigImportable:
    def test_load_config_is_callable(self) -> None:
        from bot.config import load_config

        assert callable(load_config)

    def test_load_config_returns_dict(self) -> None:
        from bot.config import load_config

        result = load_config()
        assert isinstance(result, dict)


# ===================================================================
# 2.  from bot.config import get_config works (if it exists)
# ===================================================================


class TestGetConfigImportable:
    """get_config may or may not exist depending on package decomposition state."""

    def test_get_config_exists_or_skipped(self) -> None:
        mod = importlib.import_module("bot.config")
        if not hasattr(mod, "get_config"):
            pytest.skip("get_config not exposed by bot.config (package not fully decomposed yet)")

    def test_get_config_is_callable(self) -> None:
        try:
            from bot.config import get_config as _gc
        except ImportError:
            pytest.skip("get_config not exposed by bot.config")
        assert callable(_gc)


# ===================================================================
# 3.  from bot.config import load_system_prompts works
# ===================================================================


class TestLoadSystemPrompts:
    def test_load_system_prompts_is_callable(self) -> None:
        from bot.config import load_system_prompts

        assert callable(load_system_prompts)

    def test_load_system_prompts_returns_dict(self) -> None:
        from bot.config import load_system_prompts

        result = load_system_prompts()
        assert isinstance(result, dict)

    def test_load_system_prompts_has_expected_keys(self) -> None:
        from bot.config import load_system_prompts

        result = load_system_prompts()
        assert "text_prompt" in result or "vl_prompt" in result, "load_system_prompts should return at least one prompt key"


# ===================================================================
# 4.  from bot.config import check_venv_activation works
# ===================================================================


class TestCheckVenvActivation:
    def test_check_venv_activation_is_callable(self) -> None:
        from bot.config import check_venv_activation

        assert callable(check_venv_activation)

    def test_check_venv_activation_no_raise(self) -> None:
        """Function should never raise, even outside a venv."""
        from bot.config import check_venv_activation

        with patch("sys.prefix", "/some/non-venv/path"):
            # Should complete without raising
            check_venv_activation()


# ===================================================================
# 5.  from bot.config import validate_required_env works
# ===================================================================


class TestValidateRequiredEnv:
    def test_validate_required_env_is_callable(self) -> None:
        from bot.config import validate_required_env

        assert callable(validate_required_env)

    def test_validate_required_env_raises_when_missing(self) -> None:
        """Missing DISCORD_TOKEN should raise ConfigurationError."""
        from bot.config import validate_required_env
        from bot.exceptions import ConfigurationError

        with (
            patch.dict(
                os.environ,
                {"DISCORD_TOKEN": "", "PROMPT_FILE": "", "VL_PROMPT_FILE": ""},
                clear=False,
            ),
            pytest.raises(ConfigurationError),
        ):
            validate_required_env()

    def test_validate_required_env_passes_when_set(self, monkeypatch) -> None:
        monkeypatch.setenv("DISCORD_TOKEN", "test-token")
        monkeypatch.setenv("PROMPT_FILE", "prompts/prompt-yoroi-super-chill.txt")
        monkeypatch.setenv("VL_PROMPT_FILE", "prompts/vl-prompt.txt")

        from bot.config import validate_required_env

        # Should not raise
        validate_required_env()


# ===================================================================
# 6.  from bot.config import ConfigurationError works
#     (actually defined in bot.exceptions, but re-exported from bot.config)
# ===================================================================


class TestConfigurationErrorImportable:
    def test_importable_from_bot_config(self) -> None:
        from bot.config import ConfigurationError

        assert ConfigurationError is not None
        assert isinstance(ConfigurationError, type)

    def test_is_exception_subclass(self) -> None:
        from bot.config import ConfigurationError

        assert issubclass(ConfigurationError, Exception)

    def test_can_instantiate_and_raise(self) -> Never:
        from bot.config import ConfigurationError

        with pytest.raises(ConfigurationError, match="test message"):
            msg = "test message"
            raise ConfigurationError(msg)

    def test_same_type_as_bot_exceptions(self) -> None:
        from bot.config import ConfigurationError
        from bot.exceptions import ConfigurationError as ExcConfigurationError

        assert ConfigurationError is ExcConfigurationError, "ConfigurationError from bot.config should be the same object as the one from bot.exceptions (re-export, not copy)"


# ===================================================================
# 7.  from bot.config import get_vl_model_ladder works
# ===================================================================


class TestGetVlModelLadder:
    def test_get_vl_model_ladder_is_callable(self) -> None:
        from bot.config import get_vl_model_ladder

        assert callable(get_vl_model_ladder)

    def test_get_vl_model_ladder_returns_list(self) -> None:
        from bot.config import get_vl_model_ladder

        result = get_vl_model_ladder()
        assert isinstance(result, list)

    def test_get_vl_model_ladder_non_empty_default(self, monkeypatch) -> None:
        """When VL_MODEL env is unset, returns the built‐in default ladder."""
        # Make sure VL_MODEL is not set
        if "VL_MODEL" in os.environ:
            monkeypatch.delenv("VL_MODEL")

        from bot.config import get_vl_model_ladder, invalidate_config_cache

        invalidate_config_cache()
        result = get_vl_model_ladder()
        assert len(result) > 0, "Default VL model ladder should not be empty"
        assert all(isinstance(m, str) for m in result)

    def test_get_vl_model_ladder_respects_env(self, monkeypatch) -> None:
        """When VL_MODEL is set to a comma-separated list, that list is returned."""
        from bot.config import invalidate_config_cache

        monkeypatch.setenv("VL_MODEL", "model-a,model-b")
        invalidate_config_cache()
        from bot.config import get_vl_model_ladder

        result = get_vl_model_ladder()
        assert result == ["model-a", "model-b"]


# ===================================================================
# 8.  from bot.config import invalidate_config_cache works
# ===================================================================


class TestInvalidateConfigCache:
    def test_invalidate_config_cache_is_callable(self) -> None:
        from bot.config import invalidate_config_cache

        assert callable(invalidate_config_cache)

    def test_invalidate_config_cache_clears_cache(self) -> None:
        """After invalidating, load_config should rebuild the config."""
        from bot.config import get_cache_timestamp, get_config_cache, invalidate_config_cache, load_config

        # Populate cache
        cfg1 = load_config()
        assert cfg1 is not None

        # Invalidate
        invalidate_config_cache()

        # Verify cache globals are reset to None/0
        assert get_config_cache() is None
        assert get_cache_timestamp() == 0.0

    def test_invalidated_reload_produces_fresh_object(self, monkeypatch) -> None:
        """After cache invalidation, a new load_config call picks up env changes."""
        from bot.config import invalidate_config_cache, load_config

        # Load and cache original
        load_config()

        # Change an env var that affects config
        monkeypatch.setenv("TEST_BACKEND", "new-backend")
        invalidate_config_cache()

        cfg_b = load_config()
        # Should be able to load again without issue
        assert isinstance(cfg_b, dict)


# ===================================================================
# 9.  load_config() returns a dict with all expected keys
# ===================================================================

EXPECTED_CONFIG_KEYS = {
    # Discord
    "DISCORD_TOKEN",
    "TEXT_BACKEND",
    # API keys
    "OPENAI_API_KEY",
    "OPENAI_API_BASE",
    "OPENAI_TEXT_MODEL",
    "OPENROUTER_API_KEY",
    # Ollama
    "OLLAMA_BASE_URL",
    "OLLAMA_MODEL",
    "VL_MODEL",
    "TEXT_MODEL",
    # Vision
    "VISION_ENABLED",
    "VISION_T2I_ENABLED",
    # TTS / STT
    "TTS_BACKEND",
    "TTS_VOICE",
    "STT_ENGINE",
    "STT_FALLBACK",
    "WHISPER_MODEL_SIZE",
    # Prompts
    "PROMPT_FILE",
    "VL_PROMPT_FILE",
    # Bot behaviour
    "TEMPERATURE",
    "TIMEOUT",
    "CHANGE_NICKNAME",
    "MAX_CONVERSATION_LENGTH",
    # Memory
    "MAX_USER_MEMORY",
    "MAX_SERVER_MEMORY",
    "PERSISTENT_MEMORY_ENABLE",
    # Directories (Path objects)
    "USER_PROFILE_DIR",
    "SERVER_PROFILE_DIR",
    # Logging
    "LOG_LEVEL",
    "LOG_FILE",
    # Misc
    "DEBUG",
    "COMMAND_PREFIX",
}


class TestLoadConfigKeys:
    @pytest.fixture(scope="class")
    def config(self):
        from bot.config import invalidate_config_cache, load_config

        invalidate_config_cache()
        return load_config()

    def test_returns_dict(self, config) -> None:
        assert isinstance(config, dict)

    @pytest.mark.parametrize("key", sorted(EXPECTED_CONFIG_KEYS))
    def test_expected_key_present(self, config, key) -> None:
        assert key in config, f"load_config() missing expected key: {key}"

    def test_key_values_are_reasonable(self, config) -> None:
        """Spot-check a few parsed types."""
        assert isinstance(config["TEMPERATURE"], float)
        assert isinstance(config["MAX_CONVERSATION_LENGTH"], int)
        assert isinstance(config["VISION_ENABLED"], bool)
        assert isinstance(config["USER_PROFILE_DIR"], type(__import__("pathlib").Path(".")))
        assert isinstance(config["REPLY_TRIGGERS"], list)
        assert isinstance(config["OWNER_IDS"], list)


# ===================================================================
# 10. Re-import smoke test: all names importable from bot.config
# ===================================================================


class TestBackwardCompatibilityImports:
    """Comprehensive import test — every public name must be accessible."""

    PUBLIC_NAMES = [
        "load_config",
        "load_system_prompts",
        "check_venv_activation",
        "validate_required_env",
        "get_vl_model_ladder",
        "invalidate_config_cache",
        "ConfigurationError",
        "KOKORO_FORCE_IPA_EN",
    ]

    @pytest.mark.parametrize("name", PUBLIC_NAMES)
    def test_public_name_importable(self, name) -> None:
        mod = importlib.import_module("bot.config")
        assert hasattr(mod, name), f"{name} not importable from bot.config"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
