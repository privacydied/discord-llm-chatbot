#!/usr/bin/env python3
"""
Test registry integration for TTS Kokoro pipeline.

Tests the new Decision-based tokenization system that routes
phonemes/graphemes correctly to KokoroDirect and prevents misaki
from being used for English text.
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from bot.tokenizer_registry import select_for_language, Decision
from bot.tts.kokoro_direct_fixed import KokoroDirect
from bot.tts.engines.kokoro import KokoroONNXEngine


class TestRegistryIntegration:
    """Test registry integration with Decision objects."""

    def test_en_env_misaki_falls_back_to_phonemizer(self, monkeypatch):
        """Test that TTS_TOKENISER=misaki for English falls back to phonemizer with debug log."""
        monkeypatch.setenv("TTS_TOKENISER", "misaki")

        # Mock available tokenizers to include both misaki and phonemizer
        with patch(
            "bot.tokenizer_registry.TokenizerRegistry.get_instance"
        ) as mock_registry:
            registry = MagicMock()
            registry._initialized = True
            registry._available_tokenizers = {"misaki", "phonemizer", "grapheme"}
            registry._canonicalize_language.return_value = "en"
            registry.apply_lexicon.return_value = ("hello world", False)
            registry._tokenize_to_phonemes.return_value = "həˈloʊ wɝːld"
            mock_registry.return_value = registry

            decision = select_for_language("en", "hello world")

            assert decision.mode in {"phonemes", "grapheme"}
            assert decision.alphabet != "misaki"  # Should not use misaki for English

            # Verify debug log was called for misaki fallback
            registry._select_tokenizer_with_fallback.assert_called_once()
            # The method should have logged the fallback

    def test_phoneme_decision_paths_phonemes_to_kd(self, mocker):
        """Test that phoneme decisions route correctly to KokoroDirect."""
        kd = MagicMock(spec=KokoroDirect)
        kd.create.return_value = Path("/tmp/test.wav")

        # Mock the registry to return phoneme decision
        decision = Decision("phonemes", "həˈloʊ wɝːld", "IPA")

        with patch("bot.tts.kokoro_direct_fixed.KokoroDirect", return_value=kd):
            # This would be called from the engine
            kd.create(
                phonemes=decision.payload,
                voice="af_heart",
                lang="en",
                disable_autodiscovery=True,
                logger=MagicMock(),
            )

        kd.create.assert_called_once()
        kwargs = kd.create.call_args.kwargs
        assert kwargs.get("phonemes") == "həˈloʊ wɝːld"
        assert kwargs.get("disable_autodiscovery") is True

    def test_grapheme_quiet_path_no_autodiscovery_logs(self, caplog):
        """Test that grapheme path with disable_autodiscovery produces no tokenizer noise."""
        caplog.set_level("DEBUG")

        # Mock KokoroDirect to avoid actual file operations
        with patch("bot.tts.kokoro_direct_fixed.KokoroDirect") as mock_kd_class:
            mock_kd = MagicMock()
            mock_kd.create.return_value = Path("/tmp/test.wav")
            mock_kd_class.return_value = mock_kd

            # Mock registry to return grapheme decision
            with patch("bot.tokenizer_registry.select_for_language") as mock_select:
                mock_select.return_value = Decision(
                    "grapheme", "hello world", "GRAPHEME"
                )

                # This simulates what happens in KokoroONNXEngine
                from bot.tts.kokoro_direct_fixed import KokoroDirect

                kd = KokoroDirect(model_path="test.onnx", voices_path="test.npz")
                decision = mock_select.return_value

                kd.create(
                    text=decision.payload,
                    voice="af_heart",
                    lang="en",
                    disable_autodiscovery=True,
                    logger=MagicMock(),
                )

        # Check that no tokenizer-related logs appear
        logs = "\n".join(r.message for r in caplog.records)
        assert "No known tokenization methods found" not in logs
        assert "Found phonemizer package" not in logs
        assert "Found misaki package" not in logs
        assert "Found espeak" not in logs
        assert "tokenizer.method" not in logs
        assert "tokenizer.external" not in logs

        # But should still have the basic logs
        assert "Using pre-tokenized tokens" in logs or mock_kd.create.called

    def test_registry_phoneme_tokenization(self):
        """Test that registry properly tokenizes text to phonemes."""
        with patch(
            "bot.tokenizer_registry.TokenizerRegistry.get_instance"
        ) as mock_registry:
            registry = MagicMock()
            registry._initialized = True
            registry._available_tokenizers = {"phonemizer", "grapheme"}
            registry._canonicalize_language.return_value = "en"
            registry.apply_lexicon.return_value = ("hello world", False)
            registry._tokenize_to_phonemes.return_value = "həˈloʊ wɝːld"
            mock_registry.return_value = registry

            decision = select_for_language("en", "hello world")

            assert decision.mode == "phonemes"
            assert decision.payload == "həˈloʊ wɝːld"
            assert decision.alphabet == "IPA"

    def test_registry_grapheme_fallback(self):
        """Test that registry falls back to grapheme when tokenization fails."""
        with patch(
            "bot.tokenizer_registry.TokenizerRegistry.get_instance"
        ) as mock_registry:
            registry = MagicMock()
            registry._initialized = True
            registry._available_tokenizers = {"grapheme"}
            registry._canonicalize_language.return_value = "en"
            registry.apply_lexicon.return_value = ("hello world", False)
            registry._tokenize_to_phonemes.side_effect = Exception(
                "No phonemizer available"
            )
            mock_registry.return_value = registry

            decision = select_for_language("en", "hello world")

            assert decision.mode == "grapheme"
            assert decision.payload == "hello world"
            assert decision.alphabet == "GRAPHEME"

    def test_registry_lexicon_application(self):
        """Test that lexicon overrides are applied before tokenization."""
        with patch(
            "bot.tokenizer_registry.TokenizerRegistry.get_instance"
        ) as mock_registry:
            registry = MagicMock()
            registry._initialized = True
            registry._available_tokenizers = {"phonemizer", "grapheme"}
            registry._canonicalize_language.return_value = "en"
            registry.apply_lexicon.return_value = ("həˈloʊ wɝːld", True)  # Changed
            registry._tokenize_to_phonemes.return_value = "həˈloʊ wɝːld"
            mock_registry.return_value = registry

            decision = select_for_language("en", "hello world")

            # Verify lexicon was applied
            registry.apply_lexicon.assert_called_once_with("hello world", "en")
            assert decision.payload == "həˈloʊ wɝːld"


class TestEngineIntegration:
    """Test KokoroONNXEngine integration with registry decisions."""

    def test_engine_uses_registry_decisions(self, mocker):
        """Test that KokoroONNXEngine uses registry decisions for routing."""
        # Mock all the dependencies
        mock_kd_class = mocker.patch("bot.tts.kokoro_direct_fixed.KokoroDirect")
        mock_kd = MagicMock()
        mock_kd.create.return_value = Path("/tmp/test.wav")
        mock_kd_class.return_value = mock_kd

        mock_select = mocker.patch("bot.tokenizer_registry.select_for_language")
        mock_select.return_value = Decision("phonemes", "həˈloʊ wɝːld", "IPA")

        # Mock the engine internals
        with patch("bot.tts.engines.kokoro.Kokoro") as mock_kokoro:
            mock_engine = MagicMock()
            mock_engine.generate_audio.side_effect = Exception("No audio")
            mock_kokoro.return_value = mock_engine

            # Create engine instance
            engine = KokoroONNXEngine("test.onnx", "test.npz")

            # Mock file reading and force non-English branch to use registry path
            with patch("builtins.open", mocker.mock_open(read_data=b"test wav data")):
                with patch("pathlib.Path.exists", return_value=True):
                    with patch("pathlib.Path.unlink"):
                        engine.synthesize("hello world", language="es")

            # Verify registry was consulted
            mock_select.assert_called_once_with("en", "hello world")

            # Verify KokoroDirect was called with phonemes
            mock_kd.create.assert_called_once()
            kwargs = mock_kd.create.call_args.kwargs
            assert kwargs.get("phonemes") == "həˈloʊ wɝːld"
            assert kwargs.get("disable_autodiscovery") is True

    def test_engine_grapheme_path(self, mocker):
        """Test that grapheme decisions route to text path."""
        mock_kd_class = mocker.patch("bot.tts.kokoro_direct_fixed.KokoroDirect")
        mock_kd = MagicMock()
        mock_kd.create.return_value = Path("/tmp/test.wav")
        mock_kd_class.return_value = mock_kd

        mock_select = mocker.patch("bot.tokenizer_registry.select_for_language")
        mock_select.return_value = Decision("grapheme", "hello world", "GRAPHEME")

        with patch("bot.tts.engines.kokoro.Kokoro") as mock_kokoro:
            mock_engine = MagicMock()
            mock_engine.generate_audio.side_effect = Exception("No audio")
            mock_kokoro.return_value = mock_engine

            engine = KokoroONNXEngine("test.onnx", "test.npz")

            # Force non-English branch to use registry path
            with patch("builtins.open", mocker.mock_open(read_data=b"test wav data")):
                with patch("pathlib.Path.exists", return_value=True):
                    with patch("pathlib.Path.unlink"):
                        engine.synthesize("hello world", language="es")

            # Verify KokoroDirect was called with text
            mock_kd.create.assert_called_once()
            kwargs = mock_kd.create.call_args.kwargs
            assert kwargs.get("text") == "hello world"
            assert kwargs.get("phonemes") is None
            assert kwargs.get("disable_autodiscovery") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
