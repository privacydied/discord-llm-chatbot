import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from bot.tts.engines.kokoro import KokoroEngine
from bot.tts.errors import TTSError


class TestKokoroEngine(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.model_path = "test_model_path"
        self.voices_path = "test_voices_path"
        self.tokenizer = "test_tokenizer"
        self.engine = KokoroEngine(
            model_path=self.model_path,
            voices_path=self.voices_path,
            tokenizer=self.tokenizer,
        )

    @patch("bot.tts.engines.kokoro.Kokoro")
    def test_load_success(self, mock_kokoro):
        # Setup
        mock_instance = MagicMock()
        mock_kokoro.return_value = mock_instance

        # Execute
        self.engine.load()

        # Verify
        mock_kokoro.assert_called_once_with(
            model_path=self.model_path, voices_path=self.voices_path
        )
        self.assertEqual(mock_instance.tokenizer, self.tokenizer)
        self.assertEqual(self.engine.engine, mock_instance)

    @patch("bot.tts.engines.kokoro.Kokoro")
    def test_load_failure(self, mock_kokoro):
        # Setup
        mock_kokoro.side_effect = Exception("Test error")

        # Execute & Verify
        with self.assertRaises(TTSError):
            self.engine.load()

    @patch("bot.tts.engines.kokoro.Kokoro")
    async def test_synthesize_success(self, mock_kokoro):
        # Setup
        mock_instance = MagicMock()
        mock_instance.generate_audio = AsyncMock(return_value=b"audio_data")
        mock_kokoro.return_value = mock_instance
        self.engine.engine = None  # Force load

        # Execute (use non-English to route through registry path)
        result = await self.engine.synthesize("test text", language="es")

        # Verify
        self.assertEqual(result, b"audio_data")
        mock_instance.generate_audio.assert_awaited_once_with("test text")

    @patch("bot.tts.engines.kokoro.Kokoro")
    async def test_synthesize_failure(self, mock_kokoro):
        # Setup
        mock_instance = MagicMock()
        mock_instance.generate_audio = AsyncMock(side_effect=Exception("Test error"))
        mock_kokoro.return_value = mock_instance
        self.engine.engine = mock_instance

        # Execute & Verify
        with self.assertRaises(TTSError):
            await self.engine.synthesize("test text")


if __name__ == "__main__":
    unittest.main()
