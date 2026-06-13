"""Tests for tokenizer selection logic."""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Skip all tests in this module — they require system-level tokenizer binaries
pytestmark = pytest.mark.skip(reason="Requires system-level tokenizer binaries (espeak-ng, phonemizer)")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.tts.validation import (
    AVAILABLE_TOKENIZERS,
    detect_available_tokenizers,
    get_tokenizer_warning_message,
    is_tokenizer_warning_needed,
    select_tokenizer_for_language,
)


class TestTokenizerSelection(unittest.TestCase):
    """Test tokenizer selection logic."""

    def setUp(self) -> None:
        """Set up test environment."""
        # Reset global state before each test
        global AVAILABLE_TOKENIZERS, TOKENIZER_WARNING_SHOWN
        AVAILABLE_TOKENIZERS.clear()
        TOKENIZER_WARNING_SHOWN = False

    @patch("shutil.which")
    @patch("subprocess.run")
    @patch("importlib.import_module")
    def test_detect_available_tokenizers(self, mock_import, mock_run, mock_which) -> None:
        """Test detection of available tokenizers."""
        # Mock shutil.which to return paths for espeak
        mock_which.side_effect = lambda cmd: "/usr/bin/espeak" if cmd == "espeak" else None

        # Mock subprocess.run to return success for espeak
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_run.return_value = mock_process

        # Mock imports to succeed for phonemizer but fail for others
        def mock_import_side_effect(name):
            if name == "phonemizer":
                return MagicMock()
            msg = f"No module named '{name}'"
            raise ImportError(msg)

        mock_import.side_effect = mock_import_side_effect

        # Patch __import__ to control module imports
        with patch("builtins.__import__") as mock_builtin_import:

            def import_mock(name, *args):
                if name == "phonemizer":
                    return MagicMock()
                msg = f"No module named '{name}'"
                raise ImportError(msg)

            mock_builtin_import.side_effect = import_mock

            # Call the function
            result = detect_available_tokenizers()

            # Check results
            assert result["espeak"]
            assert result["grapheme"]  # Always available
            assert not result["misaki"]
            assert not result["g2p_en"]

            # Check global state
            assert "espeak" in AVAILABLE_TOKENIZERS
            assert "grapheme" in AVAILABLE_TOKENIZERS

    def test_tokeniser_auto_pick_en(self) -> None:
        """Test auto-selection of tokenizer for English."""
        # Mock available tokenizers
        global AVAILABLE_TOKENIZERS
        AVAILABLE_TOKENIZERS = {"espeak", "grapheme"}

        # Select tokenizer for English
        selected = select_tokenizer_for_language("en")

        # Should select espeak for English
        assert selected == "espeak"

        # Try with en-US
        selected = select_tokenizer_for_language("en-US")
        assert selected == "espeak"

        # Try with empty available tokenizers
        AVAILABLE_TOKENIZERS.clear()
        AVAILABLE_TOKENIZERS.add("grapheme")

        # Should fall back to grapheme
        selected = select_tokenizer_for_language("en")
        assert selected == "grapheme"

        # Check warning flag is set
        assert is_tokenizer_warning_needed()

    def test_tokeniser_warning(self) -> None:
        """Test tokenizer warning flag and message."""
        # Mock available tokenizers with only grapheme
        global AVAILABLE_TOKENIZERS
        AVAILABLE_TOKENIZERS = {"grapheme"}

        # Select tokenizer for English (should set warning flag)
        selected = select_tokenizer_for_language("en")
        assert selected == "grapheme"

        # Check warning flag is set
        assert is_tokenizer_warning_needed()

        # Get warning message
        message = get_tokenizer_warning_message("en")
        assert "missing a phonetic tokeniser for English" in message

        # Warning flag should be cleared after getting message
        assert not is_tokenizer_warning_needed()

        # Reset flag for next test
        global TOKENIZER_WARNING_SHOWN
        TOKENIZER_WARNING_SHOWN = False

        # Test with Spanish
        selected = select_tokenizer_for_language("es")
        assert selected == "grapheme"

        # Check warning flag is set
        assert is_tokenizer_warning_needed()

        # Get warning message for Spanish
        message = get_tokenizer_warning_message("es")
        assert "missing a phonetic tokeniser for this language" in message

    def test_tokenizer_preference_order(self) -> None:
        """Test that tokenizers are selected in the correct preference order."""
        # Mock available tokenizers with multiple options
        global AVAILABLE_TOKENIZERS
        AVAILABLE_TOKENIZERS = {"espeak", "phonemizer", "g2p_en", "grapheme"}

        # Should select espeak (first in preference list)
        selected = select_tokenizer_for_language("en")
        assert selected == "espeak"

        # Remove espeak
        AVAILABLE_TOKENIZERS.remove("espeak")

        # Should select phonemizer (second in preference list)
        selected = select_tokenizer_for_language("en")
        assert selected == "phonemizer"

        # Remove phonemizer
        AVAILABLE_TOKENIZERS.remove("phonemizer")

        # Should select g2p_en (third in preference list)
        selected = select_tokenizer_for_language("en")
        assert selected == "g2p_en"

        # Remove g2p_en
        AVAILABLE_TOKENIZERS.remove("g2p_en")

        # Should fall back to grapheme
        selected = select_tokenizer_for_language("en")
        assert selected == "grapheme"


if __name__ == "__main__":
    unittest.main()
