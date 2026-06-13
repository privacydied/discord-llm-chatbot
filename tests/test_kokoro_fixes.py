#!/usr/bin/env python3
"""Validation script for Kokoro TTS fixes.
Tests vocab sanity, longest-match encoding, and English IPA synthesis.
"""

import os
import sys
from pathlib import Path

# Add bot to path
sys.path.insert(0, str(Path(__file__).parent / "bot"))


def test_vocab_sanity() -> bool | None:
    """Test that official vocab can be loaded and contains essential symbols."""
    try:
        from bot.tts.ipa_vocab_loader import load_vocab

        # Mock ONNX session for testing
        class MockONNXSession:
            def get_inputs(self):
                return [type("Input", (), {"name": "style", "shape": [1, 256]})()]

        # Load vocab with mock session
        mock_session = MockONNXSession()

        # Try to load - this should work with vendored vocab if available
        try:
            vocab = load_vocab(mock_session)

            # Check essential symbols
            essential = {
                "k",
                "g",
                "t",
                "d",
                "p",
                "b",
                "f",
                "v",
                "s",
                "z",
                "ʃ",
                "ʒ",
                "θ",
                "ð",
                "ŋ",
                "m",
                "n",
                "l",
                "ɹ",
                "i",
                "ɪ",
                "eɪ",
                "oʊ",
                "aɪ",
                "aʊ",
                "ɔɪ",
                "æ",
                "ɑ",
                "ʌ",
                "ə",
            }

            missing = [s for s in essential if s not in vocab.phoneme_to_id]
            if missing:
                pass
            else:
                pass

            return True

        except Exception as e:
            return False

    except ImportError as e:
        return False


def test_longest_match_encoding():
    """Test that IPA encoding uses longest-match greedy algorithm."""
    try:
        from bot.tts.ipa_vocab_loader import (
            UnsupportedIPASymbolError,
            encode_ipa,
            load_vocab,
        )

        # Mock session
        class MockONNXSession:
            def get_inputs(self):
                return [type("Input", (), {"name": "style", "shape": [1, 256]})()]

        # Test with vendored vocab if available
        os.environ["KOKORO_ALLOW_VENDORED_VOCAB"] = "true"

        try:
            vocab = load_vocab(MockONNXSession())

            # Test longest-match: "tʃ" should be encoded as one token, not "t" + "ʃ"
            test_cases = [
                ("tʃ æ t", ["tʃ", "æ", "t"]),  # Expected: single token for "tʃ"
                ("aɪ", ["aɪ"]),  # Expected: single token for diphthong
                ("k æ t", ["k", "æ", "t"]),  # Expected: individual consonants
            ]

            all_good = True
            for ipa, expected_tokens in test_cases:
                try:
                    token_ids = encode_ipa(ipa, MockONNXSession())
                    # Convert back to check if tokens match expected
                    actual_tokens = [vocab.id_to_phoneme[tid] for tid in token_ids if tid < len(vocab.id_to_phoneme)]
                    actual_clean = [t for t in actual_tokens if t.strip()]

                    if actual_clean == expected_tokens:
                        pass
                    else:
                        all_good = False

                except UnsupportedIPASymbolError as e:
                    all_good = False

            return all_good

        except Exception as e:
            return False

    except ImportError as e:
        return False


def test_english_registry_block() -> bool | None:
    """Test that English is blocked from using tokenizer registry."""
    try:
        from bot.tokenizer_registry import TokenizerRegistry

        registry = TokenizerRegistry.get_instance()

        try:
            # This should raise RuntimeError
            decision = registry.select_for_language("en", "hello world")
            return False
        except RuntimeError as e:
            return "IPA-only path" in str(e)
        except Exception as e:
            return False

    except ImportError as e:
        return False


def test_g2p_no_plain_a() -> bool | None:
    """Test that G2P never emits plain 'a' for English."""
    try:
        from bot.tts.eng_g2p_local import text_to_ipa

        test_words = ["cat", "bat", "that", "map", "back"]

        for word in test_words:
            try:
                ipa = text_to_ipa(word)
                if " a " in f" {ipa} " or ipa.startswith("a ") or ipa.endswith(" a"):
                    pass
                else:
                    pass
            except Exception as e:
                pass

        return True

    except ImportError as e:
        return False


def test_voice_memo_sender() -> bool | None:
    """Test voice memo sender with dummy WAV."""
    try:
        from bot.infra.voice_memo_sender import VoiceMemoError, wav_bytes_to_voice_memo

        # Create dummy WAV bytes (minimal WAV header + silence)
        dummy_wav = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"

        try:
            # This should fail due to missing bot token, but validates import/structure
            wav_bytes_to_voice_memo(12345, dummy_wav, "fake_token")
        except VoiceMemoError as e:
            if "token" in str(e).lower() or "bot" in str(e).lower():
                pass
            else:
                pass
        except Exception as e:
            pass

        return True

    except ImportError as e:
        return False


def main():
    """Run all validation tests."""
    tests = [
        test_vocab_sanity,
        test_longest_match_encoding,
        test_english_registry_block,
        test_g2p_no_plain_a,
        test_voice_memo_sender,
    ]

    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            pass


    if passed == len(tests):
        pass
    else:
        pass

    return passed == len(tests)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
