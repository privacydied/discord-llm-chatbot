"""Tests for the tokenizer registry implementation."""

import pytest

# Skip — requires system-level tokenizer binaries
pytestmark = pytest.mark.skip(reason="Requires system-level tokenizer binaries")

import os
import unittest
from unittest.mock import MagicMock, patch

# Import the module under test
from bot.tokenizer_registry import TokenizerRegistry


class TestTokenizerRegistry(unittest.TestCase):
    """Test suite for tokenizer registry functionality."""

    def setUp(self) -> None:
        """Set up test environment."""
        # Save original environment
        self.original_env = os.environ.copy()

        # Create a fresh registry for each test
        self.registry = TokenizerRegistry()
        TokenizerRegistry._instance = self.registry

    def tearDown(self) -> None:
        """Clean up after tests."""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)

        # Reset the singleton instance
        TokenizerRegistry._instance = None

    @patch("bot.tokenizer_registry.TokenizerRegistry._dump_environment_diagnostics")
    @patch("subprocess.run")
    def test_discovery_post_boot(self, mock_run, mock_dump) -> None:
        """Test tokenizer discovery after boot process."""
        # Mock environment diagnostics to return espeak available
        mock_dump.return_value = {
            "espeak_binary": "/usr/bin/espeak",
            "espeak_ng_binary": None,
            "phonemizer_module": True,
            "g2p_en_module": False,
            "misaki_module": False,
        }

        # Mock subprocess.run to simulate espeak being found
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_run.return_value = mock_process

        # First import should initialize with empty set
        registry1 = TokenizerRegistry.get_instance()
        assert len(registry1._available_tokenizers) == 0
        assert not registry1._initialized

        # Discover tokenizers
        registry1.discover_tokenizers()

        # Verify discovery results
        assert registry1._initialized
        assert "espeak" in registry1._available_tokenizers
        assert "grapheme" in registry1._available_tokenizers

        # Second import should get the same instance with populated set
        registry2 = TokenizerRegistry.get_instance()
        assert registry1 is registry2
        assert "espeak" in registry2._available_tokenizers
        assert "grapheme" in registry2._available_tokenizers

    @patch("bot.tokenizer_registry.TokenizerRegistry._dump_environment_diagnostics")
    @patch("subprocess.run")
    def test_env_override_blank(self, mock_run, mock_dump) -> None:
        """Test that blank TTS_TOKENISER environment variable is ignored."""
        # Mock diagnostics: g2p_en available (in English preferences)
        mock_dump.return_value = {
            "espeak_binary": None,
            "espeak_ng_binary": None,
            "phonemizer_module": False,
            "g2p_en_module": True,
            "misaki_module": False,
        }

        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_run.return_value = mock_process

        registry = TokenizerRegistry.get_instance()
        registry.discover_tokenizers()

        # Set blank TTS_TOKENISER
        with patch.dict(os.environ, {"TTS_TOKENISER": ""}):
            tokenizer = registry.select_tokenizer_for_language("en")
            assert tokenizer

        # Set whitespace TTS_TOKENISER
        with patch.dict(os.environ, {"TTS_TOKENISER": "  "}):
            tokenizer = registry.select_tokenizer_for_language("en")
            assert tokenizer

    @patch("bot.tokenizer_registry.TokenizerRegistry._dump_environment_diagnostics")
    @patch("subprocess.run")
    def test_registry_persistence(self, mock_run, mock_dump) -> None:
        """Test that the registry persists across imports in different modules."""
        # Mock diagnostics with g2p_en (in English preferences)
        mock_dump.return_value = {
            "espeak_binary": None,
            "espeak_ng_binary": None,
            "phonemizer_module": False,
            "g2p_en_module": True,
            "misaki_module": False,
        }

        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_run.return_value = mock_process

        # First module imports and initializes registry
        registry1 = TokenizerRegistry.get_instance()
        registry1.discover_tokenizers()

        # Find available phoneme tokenizer for English
        available = set(registry1._available_tokenizers)
        known = {"eng_g2p_local", "g2p_en", "misaki", "grapheme"}
        available & known

        # Simulate registry corruption (another module resets it)
        registry1._available_tokenizers.clear()

        # Second module imports registry
        registry2 = TokenizerRegistry.get_instance()

        # Verify it's the same object (singleton pattern)
        assert registry1 is registry2

        # Verify registry is empty after corruption
        assert len(registry2._available_tokenizers) == 0

        # But size_at_init should still be set from first initialization
        assert registry2._size_at_init == len(known)

        # Selecting a tokenizer should trigger rediscovery due to corruption detection
        with patch.object(registry2, "discover_tokenizers", wraps=registry2.discover_tokenizers) as mock_discover:
            tokenizer = registry2.select_tokenizer_for_language("en")
            mock_discover.assert_called_once_with(force=True)
            assert tokenizer

    @patch("bot.tokenizer_registry.TokenizerRegistry._dump_environment_diagnostics")
    def test_language_canonicalization(self, mock_dump) -> None:
        """Test language code canonicalization."""
        # Mock environment diagnostics
        mock_dump.return_value = {
            "espeak_binary": None,
            "espeak_ng_binary": None,
            "phonemizer_module": False,
            "g2p_en_module": False,
            "misaki_module": False,
        }

        registry = TokenizerRegistry.get_instance()

        # Test various language code formats
        assert registry._canonicalize_language("en") == "en"
        assert registry._canonicalize_language("EN") == "en"
        assert registry._canonicalize_language("en-US") == "en"
        assert registry._canonicalize_language("en_US") == "en_us"  # Underscores preserved
        assert registry._canonicalize_language("eng") == "en"
        assert registry._canonicalize_language("  en  ") == "en"
        assert registry._canonicalize_language("ja-JP") == "ja"
        assert registry._canonicalize_language("jpn") == "ja"
        assert registry._canonicalize_language("zh-CN") == "zh"
        assert registry._canonicalize_language("zho") == "zh"
        assert registry._canonicalize_language("") == "en"  # Default
        assert registry._canonicalize_language(None) == "en"  # Default

    @patch("bot.tokenizer_registry.TokenizerRegistry._dump_environment_diagnostics")
    def test_warning_message_format(self, mock_dump) -> None:
        """Test warning message formatting for different languages."""
        # Mock environment diagnostics
        mock_dump.return_value = {
            "espeak_binary": None,
            "espeak_ng_binary": None,
            "phonemizer_module": False,
            "g2p_en_module": False,
            "misaki_module": False,
        }

        registry = TokenizerRegistry.get_instance()

        # Test English warning message
        en_message = registry.get_tokenizer_warning_message("en")
        assert "English phonetic tokeniser missing" in en_message
        assert "espeak" in en_message
        assert "phonemizer" in en_message
        assert "g2p_en" in en_message

        # Test Japanese warning message
        ja_message = registry.get_tokenizer_warning_message("ja")
        assert "Asian language tokenizer missing" in ja_message
        assert "misaki" in ja_message
        assert "ja speech" in ja_message

        # Test Chinese warning message
        zh_message = registry.get_tokenizer_warning_message("zh")
        assert "Asian language tokenizer missing" in zh_message
        assert "misaki" in zh_message
        assert "zh speech" in zh_message

        # Test other language warning message
        fr_message = registry.get_tokenizer_warning_message("fr")
        assert "Phonetic tokeniser missing for fr" in fr_message
        assert "phonemizer" in fr_message
        assert "espeak" in fr_message


if __name__ == "__main__":
    unittest.main()
