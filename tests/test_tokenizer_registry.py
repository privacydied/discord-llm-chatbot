"""
Tests for the tokenizer registry implementation.
"""

import unittest
from unittest.mock import patch, MagicMock
import os

# Import the module under test
from bot.tokenizer_registry import TokenizerRegistry


class TestTokenizerRegistry(unittest.TestCase):
    """Test suite for tokenizer registry functionality."""

    def setUp(self):
        """Set up test environment."""
        # Save original environment
        self.original_env = os.environ.copy()

        # Create a fresh registry for each test
        self.registry = TokenizerRegistry()
        TokenizerRegistry._instance = self.registry

    def tearDown(self):
        """Clean up after tests."""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)

        # Reset the singleton instance
        TokenizerRegistry._instance = None

    @patch("bot.tokenizer_registry.TokenizerRegistry._dump_environment_diagnostics")
    @patch("subprocess.run")
    def test_discovery_post_boot(self, mock_run, mock_dump):
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
        self.assertEqual(len(registry1._available_tokenizers), 0)
        self.assertFalse(registry1._initialized)

        # Discover tokenizers
        registry1.discover_tokenizers()

        # Verify discovery results
        self.assertTrue(registry1._initialized)
        self.assertTrue("espeak" in registry1._available_tokenizers)
        self.assertTrue("grapheme" in registry1._available_tokenizers)

        # Second import should get the same instance with populated set
        registry2 = TokenizerRegistry.get_instance()
        self.assertIs(registry1, registry2)
        self.assertTrue("espeak" in registry2._available_tokenizers)
        self.assertTrue("grapheme" in registry2._available_tokenizers)

    @patch("bot.tokenizer_registry.TokenizerRegistry._dump_environment_diagnostics")
    @patch("subprocess.run")
    def test_env_override_blank(self, mock_run, mock_dump):
        """Test that blank TTS_TOKENISER environment variable is ignored."""
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

        # Discover tokenizers
        registry = TokenizerRegistry.get_instance()
        registry.discover_tokenizers()

        # Set blank TTS_TOKENISER
        with patch.dict(os.environ, {"TTS_TOKENISER": ""}):
            # Should select espeak for English despite blank override
            tokenizer = registry.select_tokenizer_for_language("en")
            self.assertEqual(tokenizer, "espeak")

        # Set whitespace TTS_TOKENISER
        with patch.dict(os.environ, {"TTS_TOKENISER": "  "}):
            # Should select espeak for English despite whitespace override
            tokenizer = registry.select_tokenizer_for_language("en")
            self.assertEqual(tokenizer, "espeak")

    @patch("bot.tokenizer_registry.TokenizerRegistry._dump_environment_diagnostics")
    @patch("subprocess.run")
    def test_registry_persistence(self, mock_run, mock_dump):
        """Test that the registry persists across imports in different modules."""
        # Mock environment diagnostics to return espeak available
        mock_dump.return_value = {
            "espeak_binary": "/usr/bin/espeak",
            "espeak_ng_binary": None,
            "phonemizer_module": False,
            "g2p_en_module": False,
            "misaki_module": False,
        }

        # Mock subprocess.run to simulate espeak being found
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_run.return_value = mock_process

        # First module imports and initializes registry
        registry1 = TokenizerRegistry.get_instance()
        registry1.discover_tokenizers()

        # Verify first module's view of registry
        self.assertTrue("espeak" in registry1._available_tokenizers)
        self.assertTrue("grapheme" in registry1._available_tokenizers)

        # Simulate registry corruption (another module resets it)
        registry1._available_tokenizers.clear()

        # Second module imports registry
        registry2 = TokenizerRegistry.get_instance()

        # Verify it's the same object (singleton pattern)
        self.assertIs(registry1, registry2)

        # Verify registry is empty after corruption
        self.assertEqual(len(registry2._available_tokenizers), 0)

        # But size_at_init should still be set from first initialization
        self.assertEqual(registry2._size_at_init, 2)  # espeak + grapheme

        # Selecting a tokenizer should trigger rediscovery due to corruption detection
        with patch.object(
            registry2, "discover_tokenizers", wraps=registry2.discover_tokenizers
        ) as mock_discover:
            tokenizer = registry2.select_tokenizer_for_language("en")
            mock_discover.assert_called_once_with(force=True)
            self.assertEqual(tokenizer, "espeak")

    @patch("bot.tokenizer_registry.TokenizerRegistry._dump_environment_diagnostics")
    def test_language_canonicalization(self, mock_dump):
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
        self.assertEqual(registry._canonicalize_language("en"), "en")
        self.assertEqual(registry._canonicalize_language("EN"), "en")
        self.assertEqual(registry._canonicalize_language("en-US"), "en")
        self.assertEqual(
            registry._canonicalize_language("en_US"), "en_us"
        )  # Underscores preserved
        self.assertEqual(registry._canonicalize_language("eng"), "en")
        self.assertEqual(registry._canonicalize_language("  en  "), "en")
        self.assertEqual(registry._canonicalize_language("ja-JP"), "ja")
        self.assertEqual(registry._canonicalize_language("jpn"), "ja")
        self.assertEqual(registry._canonicalize_language("zh-CN"), "zh")
        self.assertEqual(registry._canonicalize_language("zho"), "zh")
        self.assertEqual(registry._canonicalize_language(""), "en")  # Default
        self.assertEqual(registry._canonicalize_language(None), "en")  # Default

    @patch("bot.tokenizer_registry.TokenizerRegistry._dump_environment_diagnostics")
    def test_warning_message_format(self, mock_dump):
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
        self.assertIn("English phonetic tokeniser missing", en_message)
        self.assertIn("espeak", en_message)
        self.assertIn("phonemizer", en_message)
        self.assertIn("g2p_en", en_message)

        # Test Japanese warning message
        ja_message = registry.get_tokenizer_warning_message("ja")
        self.assertIn("Asian language tokenizer missing", ja_message)
        self.assertIn("misaki", ja_message)
        self.assertIn("ja speech", ja_message)

        # Test Chinese warning message
        zh_message = registry.get_tokenizer_warning_message("zh")
        self.assertIn("Asian language tokenizer missing", zh_message)
        self.assertIn("misaki", zh_message)
        self.assertIn("zh speech", zh_message)

        # Test other language warning message
        fr_message = registry.get_tokenizer_warning_message("fr")
        self.assertIn("Phonetic tokeniser missing for fr", fr_message)
        self.assertIn("phonemizer", fr_message)
        self.assertIn("espeak", fr_message)


if __name__ == "__main__":
    unittest.main()
