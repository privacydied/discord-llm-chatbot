"""Tests for tokenizer selection logic."""

import unittest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.tts_validation import (
    detect_available_tokenizers,
    select_tokenizer_for_language,
    is_tokenizer_warning_needed,
    get_tokenizer_warning_message,
    AVAILABLE_TOKENIZERS
)


class TestTokenizerSelection(unittest.TestCase):
    """Test tokenizer selection logic."""
    
    def setUp(self):
        """Set up test environment."""
        # Reset global state before each test
        global AVAILABLE_TOKENIZERS, TOKENIZER_WARNING_SHOWN
        AVAILABLE_TOKENIZERS.clear()
        TOKENIZER_WARNING_SHOWN = False
    
    @patch('shutil.which')
    @patch('subprocess.run')
    @patch('importlib.import_module')
    def test_detect_available_tokenizers(self, mock_import, mock_run, mock_which):
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
            raise ImportError(f"No module named '{name}'")
        
        mock_import.side_effect = mock_import_side_effect
        
        # Patch __import__ to control module imports
        with patch('builtins.__import__') as mock_builtin_import:
            def import_mock(name, *args):
                if name == "phonemizer":
                    return MagicMock()
                raise ImportError(f"No module named '{name}'")
            
            mock_builtin_import.side_effect = import_mock
            
            # Call the function
            result = detect_available_tokenizers()
            
            # Check results
            self.assertTrue(result["espeak"])
            self.assertTrue(result["grapheme"])  # Always available
            self.assertFalse(result["misaki"])
            self.assertFalse(result["g2p_en"])
            
            # Check global state
            self.assertIn("espeak", AVAILABLE_TOKENIZERS)
            self.assertIn("grapheme", AVAILABLE_TOKENIZERS)
    
    def test_tokeniser_auto_pick_en(self):
        """Test auto-selection of tokenizer for English."""
        # Mock available tokenizers
        global AVAILABLE_TOKENIZERS
        AVAILABLE_TOKENIZERS = {"espeak", "grapheme"}
        
        # Select tokenizer for English
        selected = select_tokenizer_for_language("en")
        
        # Should select espeak for English
        self.assertEqual(selected, "espeak")
        
        # Try with en-US
        selected = select_tokenizer_for_language("en-US")
        self.assertEqual(selected, "espeak")
        
        # Try with empty available tokenizers
        AVAILABLE_TOKENIZERS.clear()
        AVAILABLE_TOKENIZERS.add("grapheme")
        
        # Should fall back to grapheme
        selected = select_tokenizer_for_language("en")
        self.assertEqual(selected, "grapheme")
        
        # Check warning flag is set
        self.assertTrue(is_tokenizer_warning_needed())
    
    def test_tokeniser_warning(self):
        """Test tokenizer warning flag and message."""
        # Mock available tokenizers with only grapheme
        global AVAILABLE_TOKENIZERS
        AVAILABLE_TOKENIZERS = {"grapheme"}
        
        # Select tokenizer for English (should set warning flag)
        selected = select_tokenizer_for_language("en")
        self.assertEqual(selected, "grapheme")
        
        # Check warning flag is set
        self.assertTrue(is_tokenizer_warning_needed())
        
        # Get warning message
        message = get_tokenizer_warning_message("en")
        self.assertIn("missing a phonetic tokeniser for English", message)
        
        # Warning flag should be cleared after getting message
        self.assertFalse(is_tokenizer_warning_needed())
        
        # Reset flag for next test
        global TOKENIZER_WARNING_SHOWN
        TOKENIZER_WARNING_SHOWN = False
        
        # Test with Spanish
        selected = select_tokenizer_for_language("es")
        self.assertEqual(selected, "grapheme")
        
        # Check warning flag is set
        self.assertTrue(is_tokenizer_warning_needed())
        
        # Get warning message for Spanish
        message = get_tokenizer_warning_message("es")
        self.assertIn("missing a phonetic tokeniser for this language", message)
    
    def test_tokenizer_preference_order(self):
        """Test that tokenizers are selected in the correct preference order."""
        # Mock available tokenizers with multiple options
        global AVAILABLE_TOKENIZERS
        AVAILABLE_TOKENIZERS = {"espeak", "phonemizer", "g2p_en", "grapheme"}
        
        # Should select espeak (first in preference list)
        selected = select_tokenizer_for_language("en")
        self.assertEqual(selected, "espeak")
        
        # Remove espeak
        AVAILABLE_TOKENIZERS.remove("espeak")
        
        # Should select phonemizer (second in preference list)
        selected = select_tokenizer_for_language("en")
        self.assertEqual(selected, "phonemizer")
        
        # Remove phonemizer
        AVAILABLE_TOKENIZERS.remove("phonemizer")
        
        # Should select g2p_en (third in preference list)
        selected = select_tokenizer_for_language("en")
        self.assertEqual(selected, "g2p_en")
        
        # Remove g2p_en
        AVAILABLE_TOKENIZERS.remove("g2p_en")
        
        # Should fall back to grapheme
        selected = select_tokenizer_for_language("en")
        self.assertEqual(selected, "grapheme")


if __name__ == "__main__":
    unittest.main()
