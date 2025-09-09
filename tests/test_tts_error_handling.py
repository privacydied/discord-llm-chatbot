"""
Test TTS error handling for zero audio and OCR soft-dependency.
"""

import unittest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the modules we want to test
from bot.tts.errors import TTSSynthesisError
from bot.tts.kokoro_direct_fixed import KokoroDirect
from bot.pdf_utils import PDFProcessor


class TestTTSErrorHandling(unittest.TestCase):
    """Test TTS error handling."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
    def tearDown(self):
        """Clean up after tests."""
        # Clean up temporary directory
        self.temp_dir.cleanup()
    
    @patch('bot.tts.kokoro_direct_fixed.np.max')
    @patch('bot.tts.kokoro_direct_fixed.np.sqrt')
    @patch('bot.tts.kokoro_direct_fixed.np.mean')
    @patch('bot.tts.kokoro_direct_fixed.np.square')
    def test_zero_audio_detection(self, mock_square, mock_mean, mock_sqrt, mock_max):
        """Test that all-zero audio is detected and raises an error."""
        # Mock audio stats to simulate all-zero audio
        mock_square.return_value = np.zeros(100)
        mock_mean.return_value = 0.0
        mock_sqrt.return_value = 0.0
        mock_max.return_value = 0.0
        
        # Create KokoroDirect with mocked methods
        with patch.object(KokoroDirect, '_load_model'), \
             patch.object(KokoroDirect, '_detect_tokenization_methods'), \
             patch.object(KokoroDirect, '_tokenize_text', return_value=np.array([1, 2, 3])), \
             patch.object(KokoroDirect, '_get_voice_embedding', return_value=np.zeros((1, 256))), \
             patch('bot.tts.kokoro_direct_fixed.soundfile.write') as mock_write, \
             patch.object(KokoroDirect, 'sess') as mock_sess:
            
            # Mock ONNX session to return all-zero audio
            mock_sess.run.return_value = [np.zeros((1, 24000))]
            
            kokoro = KokoroDirect(
                model_path='dummy_model.onnx',
                voices_path='dummy_voices.bin'
            )
            
            # Test that creating audio with all zeros raises TTSSynthesisError
            output_path = self.temp_path / 'output.wav'
            with self.assertRaises(TTSSynthesisError):
                kokoro.create(
                    text="Test text",
                    output_path=output_path
                )
            
            # Verify that soundfile.write was not called (no file written)
            mock_write.assert_not_called()
    
    @patch('bot.pdf_utils.shutil.which')
    def test_tesseract_detection(self, mock_which):
        """Test that Tesseract availability is correctly detected."""
        # Test when Tesseract is available
        mock_which.return_value = '/usr/bin/tesseract'
        
        # Import module again to trigger detection
        with patch('bot.pdf_utils.logging'):
            import importlib
            importlib.reload(__import__('bot.pdf_utils'))
            from bot.pdf_utils import TESSERACT_AVAILABLE
            
            self.assertTrue(TESSERACT_AVAILABLE)
        
        # Test when Tesseract is not available
        mock_which.return_value = None
        
        # Import module again to trigger detection
        with patch('bot.pdf_utils.logging'):
            import importlib
            importlib.reload(__import__('bot.pdf_utils'))
            from bot.pdf_utils import TESSERACT_AVAILABLE
            
            self.assertFalse(TESSERACT_AVAILABLE)
    
    @patch('bot.pdf_utils.TESSERACT_AVAILABLE', False)
    @patch('bot.pdf_utils.fitz.open')
    def test_pdf_ocr_warning(self, mock_open):
        """Test that a warning is logged when OCR is needed but not available."""
        # Mock PDF with image-only content
        mock_pdf = MagicMock()
        mock_pdf.get_text.return_value = ""  # Empty text indicates image-only PDF
        mock_open.return_value = mock_pdf
        
        # Mock page with image
        mock_page = MagicMock()
        mock_page.get_text.return_value = ""
        mock_pdf.__getitem__.return_value = mock_page
        
        # Create PDF processor
        processor = PDFProcessor()
        
        # Process PDF with mocked logger
        with patch('bot.pdf_utils.logging.warning') as mock_warning:
            result = processor.extract_text_from_pdf(b'dummy pdf content')
            
            # Check that warning was logged
            mock_warning.assert_called_with(
                "Tesseract OCR is not available. Cannot extract text from image-based PDF.",
                extra={'subsys': 'pdf', 'event': 'ocr.unavailable'}
            )
            
            # Check that result is empty
            self.assertEqual(result, "")


if __name__ == '__main__':
    unittest.main()
