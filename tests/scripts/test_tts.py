"""Test suite for TTS functionality."""

from unittest.mock import Mock, patch  # noqa: F401

import pytest

from bot.tts import TTSManager
from bot.tts.state import TTSState


class TestTTSManager:
    @pytest.fixture
    def tts_manager(self):
        return TTSManager()

    @pytest.fixture
    def tts_state(self, monkeypatch):
        monkeypatch.delenv("TTS_GLOBAL_DEFAULT", raising=False)  # independent of the live .env
        return TTSState()

    def test_text_cleaning(self, tts_manager) -> None:
        """Test text cleaning functionality."""
        dirty_text = "**Hello** _world_ `code` https://example.com"
        clean_text = tts_manager._clean_text(dirty_text)
        assert clean_text == "Hello world code"

    def test_user_tts_toggle(self, tts_state) -> None:
        """Test user TTS toggle functionality."""
        user_id = 12345

        # Initially disabled
        assert not tts_state.is_user_tts_enabled(user_id)

        # Enable for user
        tts_state.set_user_tts(user_id, True)
        assert tts_state.is_user_tts_enabled(user_id)

        # Disable for user
        tts_state.set_user_tts(user_id, False)
        assert not tts_state.is_user_tts_enabled(user_id)

    def test_global_tts_toggle(self, tts_state) -> None:
        """Test global TTS toggle functionality."""
        user_id = 12345

        # Initially disabled
        assert not tts_state.is_user_tts_enabled(user_id)

        # Enable globally
        tts_state.set_global_tts(True)
        assert tts_state.is_user_tts_enabled(user_id)

        # Disable globally
        tts_state.set_global_tts(False)
        assert not tts_state.is_user_tts_enabled(user_id)

    def test_admin_management(self, tts_state) -> None:
        """Test admin user management."""
        user_id = 12345

        assert not tts_state.is_admin(user_id)

        tts_state.add_admin(user_id)
        assert tts_state.is_admin(user_id)

        tts_state.remove_admin(user_id)
        assert not tts_state.is_admin(user_id)

    @pytest.mark.asyncio
    async def test_cache_stats(self, tts_manager) -> None:
        """Test cache statistics."""
        stats = tts_manager.get_cache_stats()
        assert "files" in stats
        assert "size_mb" in stats
        assert "cache_dir" in stats


# Manager -> engine path, hermetic (stub engine; Pocket itself: tests/tts/test_pocket_engine.py)
@pytest.mark.asyncio
async def test_tts_integration() -> None:
    """Integration test for TTS synthesis."""
    tts_manager = TTSManager(config={"TTS_ENGINE": "stub"})
    result = await tts_manager.synthesize("Hello world")
    assert result[:4] == b"RIFF"
