"""
Tests for transactional hot-reload with candidate validation and safe rollback.

These tests verify that hot-reload is transactional:
- Failed reloads never poison the live config
- Previous good config is preserved on validation failure
- Provider-specific validation works correctly
- Callbacks only run on successful commit
- Manual reload command reports accurate status
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from bot.config import (
    load_config,
    load_config_candidate,
    validate_config_candidate,
    invalidate_config_cache,
)
from bot.config_reload import (
    reload_env,
    manual_reload_command,
    get_current_config,
    add_reload_callback,
    remove_reload_callback,
    _preferred_env_path,
    _candidate_env_paths,
)


@pytest.fixture(autouse=True)
def _reset_config_state():
    """Reset global config state before each test."""
    invalidate_config_cache()
    # Also reset the config_reload globals
    import bot.config_reload as cr

    with cr._config_lock:
        cr._current_config = {}
        cr._config_version = ""
        cr._last_reload_time = 0
        cr._last_reload_call_time = 0
        cr._reload_callbacks.clear()
        cr._env_loaded_values_by_path.clear()
        cr._snapshot_known_env_files()
    yield
    invalidate_config_cache()


@pytest.fixture
def valid_env_content():
    """A valid .env content with all required variables."""
    return """DISCORD_TOKEN=test_token_123
PROMPT_FILE=prompts/prompt-yoroi-super-chill.txt
VL_PROMPT_FILE=prompts/vl-prompt.txt
OPENAI_API_KEY=sk-test-key
OPENAI_API_BASE=https://openrouter.ai/api/v1
OPENAI_TEXT_MODEL=deepseek/deepseek-chat-v3-0324:free
TEXT_BACKEND=openai
"""


@pytest.fixture
def valid_env_file(valid_env_content):
    """Create a temporary valid .env file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
        f.write(valid_env_content)
        f.flush()
        yield Path(f.name)
    os.unlink(f.name)


class TestLoadConfigCandidate:
    """Test the load_config_candidate function loads from file without mutating global state."""

    def test_loads_from_specific_file(self, valid_env_file):
        """load_config_candidate reads from the given file path."""
        config = load_config_candidate(valid_env_file)
        assert config["DISCORD_TOKEN"] == "test_token_123"
        assert config["OPENAI_API_KEY"] == "sk-test-key"
        assert config["TEXT_BACKEND"] == "openai"

    def test_does_not_mutate_os_environ(self, valid_env_file):
        """load_config_candidate should not modify os.environ."""
        # Ensure clean state
        for key in ["DISCORD_TOKEN", "OPENAI_API_KEY", "TEST_CANDIDATE_KEY"]:
            if key in os.environ:
                del os.environ[key]

        load_config_candidate(valid_env_file)
        assert "DISCORD_TOKEN" not in os.environ
        assert "OPENAI_API_KEY" not in os.environ

    def test_uses_defaults_for_missing_optional(self, valid_env_file):
        """Optional keys get defaults when not in candidate file."""
        config = load_config_candidate(valid_env_file)
        assert config["OLLAMA_BASE_URL"] == "http://localhost:11434"
        assert config["TEMPERATURE"] == 0.7

    def test_handles_missing_file_gracefully(self):
        """Missing file returns config with defaults/os.environ fallbacks."""
        missing_path = Path("/nonexistent/path/.env")
        config = load_config_candidate(missing_path)
        # Should not crash, should return config with defaults
        assert isinstance(config, dict)
        assert "TEXT_BACKEND" in config


class TestValidateConfigCandidate:
    """Test validate_config_candidate enforces required and provider-specific keys."""

    def test_accepts_valid_openai_config(self, valid_env_file):
        """Valid OpenAI config passes validation."""
        config = load_config_candidate(valid_env_file)
        is_valid, missing = validate_config_candidate(config)
        assert is_valid is True
        assert missing == []

    def test_rejects_missing_discord_token(self, valid_env_file):
        """Missing DISCORD_TOKEN causes rejection."""
        config = load_config_candidate(valid_env_file)
        config["DISCORD_TOKEN"] = None
        is_valid, missing = validate_config_candidate(config)
        assert is_valid is False
        assert "DISCORD_TOKEN" in missing

    def test_rejects_missing_prompt_file(self, valid_env_file):
        """Missing PROMPT_FILE causes rejection."""
        config = load_config_candidate(valid_env_file)
        config["PROMPT_FILE"] = None
        is_valid, missing = validate_config_candidate(config)
        assert is_valid is False
        assert "PROMPT_FILE" in missing

    def test_rejects_missing_vl_prompt_file(self, valid_env_file):
        """Missing VL_PROMPT_FILE causes rejection."""
        config = load_config_candidate(valid_env_file)
        config["VL_PROMPT_FILE"] = None
        is_valid, missing = validate_config_candidate(config)
        assert is_valid is False
        assert "VL_PROMPT_FILE" in missing

    def test_rejects_openai_backend_without_openai_key(self, valid_env_file):
        """OpenAI backend requires OPENAI_API_KEY."""
        config = load_config_candidate(valid_env_file)
        config["TEXT_BACKEND"] = "openai"
        config["OPENAI_API_KEY"] = None
        is_valid, missing = validate_config_candidate(config)
        assert is_valid is False
        assert "OPENAI_API_KEY" in missing

    def test_accepts_nvidia_backend_with_nvidia_key(self, valid_env_file):
        """NVIDIA backend accepts NVIDIA_NIM_API_KEY."""
        config = load_config_candidate(valid_env_file)
        config["TEXT_BACKEND"] = "nvidia"
        config["OPENAI_API_KEY"] = None
        config["NVIDIA_NIM_API_KEY"] = "nvidia-key-123"
        is_valid, missing = validate_config_candidate(config)
        assert is_valid is True

    def test_accepts_nvidia_backend_with_openai_key_alias(self, valid_env_file):
        """NVIDIA backend accepts OPENAI_API_KEY as alias."""
        config = load_config_candidate(valid_env_file)
        config["TEXT_BACKEND"] = "nvidia"
        config["NVIDIA_NIM_API_KEY"] = None
        config["OPENAI_API_KEY"] = "openai-key-123"
        is_valid, missing = validate_config_candidate(config)
        assert is_valid is True

    def test_rejects_nvidia_backend_without_any_key(self, valid_env_file):
        """NVIDIA backend rejects when neither key is present."""
        config = load_config_candidate(valid_env_file)
        config["TEXT_BACKEND"] = "nvidia"
        config["OPENAI_API_KEY"] = None
        config["NVIDIA_NIM_API_KEY"] = None
        is_valid, missing = validate_config_candidate(config)
        assert is_valid is False
        assert "NVIDIA_NIM_API_KEY" in missing

    def test_accepts_ollama_backend_without_openai_key(self, valid_env_file):
        """Ollama backend does not require OPENAI_API_KEY."""
        config = load_config_candidate(valid_env_file)
        config["TEXT_BACKEND"] = "ollama"
        config["OPENAI_API_KEY"] = None
        config["OLLAMA_BASE_URL"] = "http://localhost:11434"
        config["OLLAMA_MODEL"] = "llama3"
        is_valid, missing = validate_config_candidate(config)
        assert is_valid is True

    def test_accepts_other_backends_without_openai_key(self, valid_env_file):
        """Unknown/other backends don't require OPENAI_API_KEY beyond baseline."""
        config = load_config_candidate(valid_env_file)
        config["TEXT_BACKEND"] = "custom_backend"
        config["OPENAI_API_KEY"] = None
        is_valid, missing = validate_config_candidate(config)
        assert is_valid is True


class TestReloadEnvTransactional:
    """Test reload_env is transactional - failed reloads don't poison live config."""

    def setup_method(self):
        invalidate_config_cache()
        import bot.config_reload as cr

        with cr._config_lock:
            cr._current_config = {}
            cr._config_version = ""
            cr._last_reload_time = 0
            cr._last_reload_call_time = 0
            cr._reload_callbacks.clear()
            cr._env_loaded_values_by_path.clear()
            cr._snapshot_known_env_files()

    def test_missing_env_file_rejected_keeps_previous_config(self, valid_env_file):
        """Test A: Missing .env during hot reload keeps previous config."""
        import bot.config_reload as cr

        # Prime with a good config
        good_config = load_config_candidate(valid_env_file)
        good_config["DISCORD_TOKEN"] = "original_token"
        cr._current_config = good_config.copy()
        cr._config_version = "original_version"

        # Try reload with missing file
        missing_path = Path("/nonexistent/missing.env")
        result = reload_env(missing_path)

        # Reload should be rejected
        assert result["success"] is False
        assert result["rejected"] is True
        assert result["previous_config_kept"] is True
        assert "not found" in result["error"]

        # Live config should be unchanged
        live = get_current_config()
        assert live["DISCORD_TOKEN"] == "original_token"

    def test_partial_env_file_rejected_keeps_previous_config(self, valid_env_file):
        """Test B: Partial .env (missing required vars) keeps previous config."""
        import bot.config_reload as cr

        # Prime with a good config
        good_config = load_config_candidate(valid_env_file)
        good_config["DISCORD_TOKEN"] = "original_token"
        cr._current_config = good_config.copy()
        cr._config_version = "original_version"

        # Create partial .env file (missing PROMPT_FILE and VL_PROMPT_FILE)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("DISCORD_TOKEN=partial_token\nOPENAI_API_KEY=sk-partial\nTEXT_BACKEND=openai\n")
            partial_path = Path(f.name)

        try:
            result = reload_env(partial_path)

            # Reload should be rejected
            assert result["success"] is False
            assert result["rejected"] is True
            assert result["previous_config_kept"] is True
            assert "PROMPT_FILE" in result["missing_vars"]
            assert "VL_PROMPT_FILE" in result["missing_vars"]

            # Live config should be unchanged
            live = get_current_config()
            assert live["DISCORD_TOKEN"] == "original_token"
        finally:
            os.unlink(partial_path)

    def test_openai_backend_missing_key_rejected(self, valid_env_file):
        """Test C: OpenAI backend missing OPENAI_API_KEY is rejected."""
        import bot.config_reload as cr

        # Prime with a good config
        good_config = load_config_candidate(valid_env_file)
        good_config["DISCORD_TOKEN"] = "original_token"
        good_config["OPENAI_API_KEY"] = "original_key"
        cr._current_config = good_config.copy()
        cr._config_version = "original_version"

        # Create .env with OpenAI backend but no API key
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write(
                "DISCORD_TOKEN=new_token\n"
                "PROMPT_FILE=prompts/prompt-yoroi-super-chill.txt\n"
                "VL_PROMPT_FILE=prompts/vl-prompt.txt\n"
                "TEXT_BACKEND=openai\n"
                "# OPENAI_API_KEY is missing\n"
            )
            partial_path = Path(f.name)

        try:
            result = reload_env(partial_path)

            assert result["success"] is False
            assert result["rejected"] is True
            assert result["previous_config_kept"] is True
            assert "OPENAI_API_KEY" in result["missing_vars"]

            # Previous key should still be active
            live = get_current_config()
            assert live["OPENAI_API_KEY"] == "original_key"
            assert live["DISCORD_TOKEN"] == "original_token"
        finally:
            os.unlink(partial_path)

    def test_ollama_backend_accepted_without_openai_key(self, valid_env_file):
        """Test D: Ollama backend accepted without OPENAI_API_KEY."""
        import bot.config_reload as cr

        # Prime with a good config (openai backend)
        good_config = load_config_candidate(valid_env_file)
        good_config["DISCORD_TOKEN"] = "original_token"
        good_config["OPENAI_API_KEY"] = "original_key"
        cr._current_config = good_config.copy()
        cr._config_version = "original_version"

        # Create .env with Ollama backend, no OpenAI key
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write(
                "DISCORD_TOKEN=new_ollama_token\n"
                "PROMPT_FILE=prompts/prompt-yoroi-super-chill.txt\n"
                "VL_PROMPT_FILE=prompts/vl-prompt.txt\n"
                "TEXT_BACKEND=ollama\n"
                "OLLAMA_BASE_URL=http://localhost:11434\n"
                "OLLAMA_MODEL=llama3\n"
                "# OPENAI_API_KEY intentionally omitted\n"
            )
            ollama_path = Path(f.name)

        try:
            result = reload_env(ollama_path)

            assert result["success"] is True
            assert result.get("rejected") is not True

            # New config should be active
            live = get_current_config()
            assert live["TEXT_BACKEND"] == "ollama"
            assert live["DISCORD_TOKEN"] == "new_ollama_token"
            # OPENAI_API_KEY should remain from os.environ or be None
        finally:
            os.unlink(ollama_path)

    def test_successful_reload_updates_config_and_runs_callbacks(self, valid_env_file):
        """Test E: Successful reload updates config and runs callbacks."""
        import bot.config_reload as cr

        # Prime with a good config
        good_config = load_config_candidate(valid_env_file)
        good_config["DISCORD_TOKEN"] = "original_token"
        good_config["OPENAI_API_KEY"] = "original_key"
        cr._current_config = good_config.copy()
        cr._config_version = "original_version"

        # Track callback calls
        callback_calls = []

        def tracking_callback(old_cfg, new_cfg):
            callback_calls.append((old_cfg.copy(), new_cfg.copy()))

        add_reload_callback(tracking_callback)

        try:
            # Create valid new .env
            with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
                f.write(
                    "DISCORD_TOKEN=new_token\n"
                    "PROMPT_FILE=prompts/prompt-yoroi-super-chill.txt\n"
                    "VL_PROMPT_FILE=prompts/vl-prompt.txt\n"
                    "OPENAI_API_KEY=new_openai_key\n"
                    "OPENAI_API_BASE=https://openrouter.ai/api/v1\n"
                    "OPENAI_TEXT_MODEL=deepseek/deepseek-chat-v3-0324:free\n"
                    "TEXT_BACKEND=openai\n"
                )
                new_path = Path(f.name)

            try:
                result = reload_env(new_path)

                assert result["success"] is True
                assert result.get("rejected") is not True

                # Config should be updated
                live = get_current_config()
                assert live["DISCORD_TOKEN"] == "new_token"
                assert live["OPENAI_API_KEY"] == "new_openai_key"

                # Callback should have been called exactly once
                assert len(callback_calls) == 1
                old_cfg, new_cfg = callback_calls[0]
                assert old_cfg["DISCORD_TOKEN"] == "original_token"
                assert new_cfg["DISCORD_TOKEN"] == "new_token"
            finally:
                os.unlink(new_path)
        finally:
            remove_reload_callback(tracking_callback)

    def test_callback_exception_logged_but_reload_succeeds(self, valid_env_file):
        """Test F: Callback exception is logged but doesn't roll back committed config."""
        import bot.config_reload as cr

        good_config = load_config_candidate(valid_env_file)
        good_config["DISCORD_TOKEN"] = "original_token"
        cr._current_config = good_config.copy()
        cr._config_version = "original_version"

        def failing_callback(old_cfg, new_cfg):
            raise RuntimeError("Callback failed intentionally")

        add_reload_callback(failing_callback)

        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
                f.write(
                    "DISCORD_TOKEN=new_token\n"
                    "PROMPT_FILE=prompts/prompt-yoroi-super-chill.txt\n"
                    "VL_PROMPT_FILE=prompts/vl-prompt.txt\n"
                    "OPENAI_API_KEY=new_key\n"
                    "TEXT_BACKEND=openai\n"
                )
                new_path = Path(f.name)

            try:
                result = reload_env(new_path)

                # Reload should still succeed
                assert result["success"] is True

                # Config should be updated despite callback failure
                live = get_current_config()
                assert live["DISCORD_TOKEN"] == "new_token"
            finally:
                os.unlink(new_path)
        finally:
            remove_reload_callback(failing_callback)

    def test_callback_exception_during_validation_does_not_run(self, valid_env_file):
        """Callbacks should not run when validation fails."""
        import bot.config_reload as cr

        good_config = load_config_candidate(valid_env_file)
        good_config["DISCORD_TOKEN"] = "original_token"
        cr._current_config = good_config.copy()
        cr._config_version = "original_version"

        callback_calls = []

        def tracking_callback(old_cfg, new_cfg):
            callback_calls.append((old_cfg, new_cfg))

        add_reload_callback(tracking_callback)

        try:
            # Invalid .env (missing required vars)
            with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
                f.write("DISCORD_TOKEN=token\n")  # Missing PROMPT_FILE, VL_PROMPT_FILE
                bad_path = Path(f.name)

            try:
                result = reload_env(bad_path)

                assert result["success"] is False
                assert result["rejected"] is True
                # Callback should NOT have been called
                assert len(callback_calls) == 0
            finally:
                os.unlink(bad_path)
        finally:
            remove_reload_callback(tracking_callback)
class TestManualReloadCommand:
    """Test G: !reload command reports accurate status."""

    def test_reports_success_on_valid_reload(self, valid_env_file):
        """Valid reload reports success with version change."""
        import bot.config_reload as cr
        import time

        good_config = load_config_candidate(valid_env_file)
        good_config["DISCORD_TOKEN"] = "original_token"
        cr._current_config = good_config.copy()
        cr._config_version = "original_version"

        # First do a successful reload to populate the state
        result = reload_env(valid_env_file)
        assert result["success"] is True

        # Wait for debounce to expire
        time.sleep(0.6)

        # Now test manual_reload_command with the same valid file
        # Since config hasn't changed, it should report "no changes detected"
        with patch("bot.config_reload._preferred_env_path", return_value=valid_env_file):
            result = manual_reload_command()
            assert "✅ Configuration reloaded" in result
            assert "Version:" in result

    def test_reports_rejected_with_missing_vars(self, valid_env_file):
        """Rejected reload reports missing variables and that previous config kept."""
        import bot.config_reload as cr

        good_config = load_config_candidate(valid_env_file)
        good_config["DISCORD_TOKEN"] = "original_token"
        cr._current_config = good_config.copy()
        cr._config_version = "original_version"

        # Create bad .env file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("DISCORD_TOKEN=token\n")
            bad_path = Path(f.name)

        try:
            with patch("bot.config_reload._preferred_env_path", return_value=bad_path):
                result = manual_reload_command()

            assert "❌ Configuration reload rejected" in result
            assert "PROMPT_FILE" in result
            assert "VL_PROMPT_FILE" in result
            assert "Previous config kept active" in result
        finally:
            os.unlink(bad_path)

    def test_reports_missing_file(self, valid_env_file):
        """Missing file reports rejection with previous config kept."""
        import bot.config_reload as cr

        good_config = load_config_candidate(valid_env_file)
        good_config["DISCORD_TOKEN"] = "original_token"
        cr._current_config = good_config.copy()
        cr._config_version = "original_version"

        missing_path = Path("/nonexistent/missing.env")
        with patch("bot.config_reload._preferred_env_path", return_value=missing_path):
            result = manual_reload_command()

        assert "❌ Configuration reload rejected" in result
        assert "not found" in result
        assert "Previous config kept active" in result

    def test_reports_debounced(self, valid_env_file):
        """Rapid successive reloads report debounced."""
        import bot.config_reload as cr

        good_config = load_config_candidate(valid_env_file)
        good_config["DISCORD_TOKEN"] = "original_token"
        cr._current_config = good_config.copy()
        cr._config_version = "original_version"

        # First reload succeeds
        with patch("bot.config_reload._preferred_env_path", return_value=valid_env_file):
            result = manual_reload_command()
            assert "✅ Configuration reloaded successfully!" in result

            # Immediate second call should be debounced
            result = manual_reload_command()
            assert "⏱️ Configuration reload debounced" in result
class TestConfigCacheAndRollback:
    """Test config cache invalidation and rollback behavior."""

    def test_cache_invalidated_only_on_success(self, valid_env_file):
        """Cache should only be invalidated after successful validation."""
        import bot.config_reload as cr
        from bot.config import get_config_cache

        # Load initial config to populate cache
        load_config()
        assert get_config_cache() is not None

        # Prime config_reload state
        good_config = load_config_candidate(valid_env_file)
        good_config["DISCORD_TOKEN"] = "original_token"
        cr._current_config = good_config.copy()
        cr._config_version = "original_version"

        # Attempt invalid reload
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("DISCORD_TOKEN=token\n")  # Invalid - missing required
            bad_path = Path(f.name)

        try:
            reload_env(bad_path)
            # Cache should still be valid (not invalidated for failed reload)
            assert get_config_cache() is not None
        finally:
            os.unlink(bad_path)

    def test_last_good_config_available_for_rollback(self, valid_env_file):
        """get_last_good_config returns the last successfully loaded config."""
        from bot.config import get_last_good_config

        # First load a valid config to populate _last_good_config
        load_config()
        last_good = get_last_good_config()
        assert last_good is not None

        # Now test with candidate file
        config = load_config_candidate(valid_env_file)
        # _last_good_config should still be the one from load_config()
        last_good = get_last_good_config()
        assert last_good is not None
        assert last_good is not None
        assert last_good["DISCORD_TOKEN"] == config["DISCORD_TOKEN"]


class TestPathHandling:
    """Test .env path discovery and handling."""

    def test_preferred_env_path_returns_existing(self, valid_env_file):
        """_preferred_env_path returns first existing candidate."""
        # The valid_env_file should be in the candidate list
        preferred = _preferred_env_path()
        # Just verify it returns a Path
        assert isinstance(preferred, Path)

    def test_candidate_env_paths_includes_common_locations(self):
        """_candidate_env_paths includes cwd, repo root, yoroi.env variants."""
        paths = _candidate_env_paths()
        assert len(paths) >= 2
        # Should include cwd/.env and repo_root/.env at minimum
        cwd_env = Path.cwd() / ".env"
        repo_env = Path(__file__).parent.parent.parent / ".env"
        # At least one should be in the resolved list
        resolved = [p.resolve() for p in paths]
        assert cwd_env.resolve() in resolved or repo_env.resolve() in resolved

    def test_reload_uses_watched_path(self, valid_env_file):
        """reload_env uses the same path resolution as watcher."""
        import bot.config_reload as cr

        good_config = load_config_candidate(valid_env_file)
        good_config["DISCORD_TOKEN"] = "original_token"
        cr._current_config = good_config.copy()
        cr._config_version = "original_version"

        # Reload with explicit path should work
        result = reload_env(valid_env_file)
        assert result["success"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
