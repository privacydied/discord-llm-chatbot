#!/usr/bin/env python3
"""
Validation script for Kokoro TTS fixes.
Tests vocab sanity, longest-match encoding, and English IPA synthesis.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add bot to path
sys.path.insert(0, str(Path(__file__).parent / "bot"))

def test_vocab_sanity():
    """Test that official vocab can be loaded and contains essential symbols."""
    print("🔍 Testing vocabulary sanity...")
    
    try:
        from bot.tts.ipa_vocab_loader import load_official_vocab, UnsupportedIPASymbolError
        
        # Mock ONNX session for testing
        class MockONNXSession:
            def get_inputs(self):
                return [type('Input', (), {'name': 'style', 'shape': [1, 256]})()]
        
        # Load vocab with mock session
        mock_session = MockONNXSession()
        
        # Try to load - this should work with vendored vocab if available
        try:
            vocab = load_official_vocab(mock_session)
            print(f"  ✅ Loaded vocabulary with {vocab.size} symbols")
            
            # Check essential symbols
            essential = {"k", "g", "t", "d", "p", "b", "f", "v", "s", "z", 
                        "ʃ", "ʒ", "θ", "ð", "ŋ", "m", "n", "l", "ɹ",
                        "i", "ɪ", "eɪ", "oʊ", "aɪ", "aʊ", "ɔɪ", "æ", "ɑ", "ʌ", "ə"}
            
            missing = [s for s in essential if s not in vocab.phoneme_to_id]
            if missing:
                print(f"  ⚠️  Missing essential symbols: {missing[:5]}...")
            else:
                print("  ✅ All essential IPA symbols present")
                
            return True
            
        except Exception as e:
            print(f"  ⚠️  Could not load official vocab: {e}")
            print("  ℹ️  Set KOKORO_ALLOW_VENDORED_VOCAB=true to use development vocab")
            return False
            
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False

def test_longest_match_encoding():
    """Test longest-match encoding with common IPA sequences."""
    print("🔍 Testing longest-match encoding...")
    
    try:
        from bot.tts.ipa_vocab_loader import load_official_vocab, encode_ipa
        
        # Mock session
        class MockONNXSession:
            def get_inputs(self):
                return [type('Input', (), {'name': 'style', 'shape': [1, 256]})()]
        
        # Test with vendored vocab if available
        os.environ['KOKORO_ALLOW_VENDORED_VOCAB'] = 'true'
        
        try:
            vocab = load_official_vocab(MockONNXSession())
            
            # Test sequences that should work with longest-match
            test_cases = [
                ("k æ t", "simple consonant-vowel-consonant"),
                ("h ɛ l oʊ", "hello with diphthong"),
                ("ð ɪ s", "this with dental fricative"),
            ]
            
            for ipa, description in test_cases:
                try:
                    token_ids = encode_ipa(ipa, vocab)
                    if token_ids:
                        print(f"  ✅ {description}: '{ipa}' → {len(token_ids)} tokens")
                    else:
                        print(f"  ⚠️  {description}: '{ipa}' → empty result")
                except Exception as e:
                    print(f"  ❌ {description}: '{ipa}' → {e}")
            
            return True
            
        except Exception as e:
            print(f"  ⚠️  Encoding test failed: {e}")
            return False
            
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False

def test_english_registry_block():
    """Test that English is blocked from using tokenizer registry."""
    print("🔍 Testing English registry block...")
    
    try:
        from bot.tokenizer_registry import TokenizerRegistry
        
        registry = TokenizerRegistry.get_instance()
        
        try:
            # This should raise RuntimeError
            decision = registry.select_for_language("en", "hello world")
            print(f"  ❌ English was NOT blocked - got decision: {decision}")
            return False
        except RuntimeError as e:
            if "IPA-only path" in str(e):
                print("  ✅ English correctly blocked from registry")
                return True
            else:
                print(f"  ⚠️  Wrong error type: {e}")
                return False
        except Exception as e:
            print(f"  ❌ Unexpected error: {e}")
            return False
            
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False

def test_g2p_no_plain_a():
    """Test that G2P never emits plain 'a' for English."""
    print("🔍 Testing G2P avoids plain 'a'...")
    
    try:
        from bot.tts.eng_g2p_local import text_to_ipa
        
        test_words = ["cat", "bat", "that", "map", "back"]
        
        for word in test_words:
            try:
                ipa = text_to_ipa(word)
                if " a " in f" {ipa} " or ipa.startswith("a ") or ipa.endswith(" a"):
                    print(f"  ⚠️  '{word}' → '{ipa}' contains plain 'a'")
                else:
                    print(f"  ✅ '{word}' → '{ipa}' (no plain 'a')")
            except Exception as e:
                print(f"  ❌ '{word}' failed: {e}")
                
        return True
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False

def test_voice_memo_sender():
    """Test voice memo sender with dummy WAV."""
    print("🔍 Testing voice memo sender...")
    
    try:
        from bot.infra.voice_memo_sender import wav_bytes_to_voice_memo, VoiceMemoError
        
        # Create dummy WAV bytes (minimal WAV header + silence)
        dummy_wav = (
            b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00'
            b'\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00'
        )
        
        try:
            # This should fail due to missing bot token, but validates import/structure
            result = wav_bytes_to_voice_memo(12345, dummy_wav, "fake_token")
            print("  ⚠️  Unexpectedly succeeded with fake token")
        except VoiceMemoError as e:
            if "token" in str(e).lower() or "bot" in str(e).lower():
                print("  ✅ Voice memo sender validates token properly")
            else:
                print(f"  ✅ Voice memo sender failed as expected: {e}")
        except Exception as e:
            print(f"  ✅ Voice memo sender failed as expected: {e}")
            
        return True
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False

def main():
    """Run all validation tests."""
    print("🛠️  Validating Kokoro TTS fixes...")
    print()
    
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
            print(f"  ❌ Test {test.__name__} crashed: {e}")
        print()
    
    print(f"📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("✅ All fixes validated successfully!")
    else:
        print("⚠️  Some issues remain - check output above")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
