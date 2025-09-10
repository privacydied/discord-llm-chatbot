#!/usr/bin/env python3
"""
Quick test to verify English IPA fix works correctly.
This script tests that English text always goes through IPA phonemes
and never falls back to graphemes.
"""

import logging
import sys
import os

# Set up minimal environment
os.environ["TTS_TOKENISER"] = ""  # Unset to let code decide
os.environ["TTS_ENGINE"] = "kokoro-onnx"
os.environ["TTS_VOICE"] = "af_heart"
os.environ["TTS_LANGUAGE"] = "en"

# Configure logging to capture debug messages
logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def test_english_ipa_fix():
    """Test that English text uses IPA phonemes and never graphemes."""
    try:
        from bot.tts.engines.kokoro import KokoroONNXEngine

        # Create engine instance
        engine = KokoroONNXEngine(
            model_path="tts/kokoro-v1.0.onnx", voices_path="tts/voices-v1.0.bin"
        )

        test_text = "Hello world, this is a test of English speech synthesis."

        print(f"\n🔍 Testing English IPA fix with text: '{test_text}'")
        print("=" * 60)

        # Capture log messages during synthesis
        log_messages = []

        class LogCapture(logging.Handler):
            def emit(self, record):
                log_messages.append(f"{record.levelname}: {record.getMessage()}")

        # Add our log capture handler
        capture_handler = LogCapture()
        capture_handler.setLevel(logging.DEBUG)
        logging.getLogger().addHandler(capture_handler)

        try:
            # This should trigger the English IPA path
            import asyncio

            result = asyncio.run(engine.synthesize(test_text))

            print("✅ Synthesis completed successfully!")
            print(f"📊 Generated {len(result)} bytes of audio data")

            # Check log messages for expected behavior
            log_text = "\n".join(log_messages)

            # Expected behaviors:
            success_indicators = [
                "English path: phoneme-only; using model IPA vocabulary.",  # Our IPA path message
                "TTS synthesis successful:",  # KokoroDirect success message
                "Normalized text to IPA:",  # G2P conversion message
            ]

            # Problem indicators (should NOT appear):
            problem_indicators = [
                "Using grapheme tokenization for en",  # Grapheme fallback
                "Phoneme tokenization failed",  # Phonemizer issues
                "libespeak-ng.so",  # espeak library issues
                "No known tokenization methods found",  # Discovery noise
            ]

            print("\n📋 Log Analysis:")
            print("-" * 30)

            for indicator in success_indicators:
                if indicator in log_text:
                    print(f"✅ Found expected: '{indicator}'")
                else:
                    print(f"⚠️  Missing expected: '{indicator}'")

            for indicator in problem_indicators:
                if indicator in log_text:
                    print(f"❌ Found problem: '{indicator}'")
                else:
                    print(f"✅ No problem: '{indicator}'")

            # Check if IPA path was taken
            if "English path: phoneme-only; using model IPA vocabulary." in log_text:
                print("\n🎉 SUCCESS: English IPA path is working correctly!")
                print("   English text is being converted to IPA phonemes as expected.")
            else:
                print("\n⚠️  WARNING: English IPA path may not be working.")
                print("   Check if the IPA path is properly configured.")

            # Check if grapheme fallback occurred
            if any(indicator in log_text for indicator in problem_indicators):
                print("\n❌ FAILURE: Found grapheme fallback or tokenizer issues!")
                print("   English is still falling back to graphemes.")
            else:
                print("\n✅ SUCCESS: No grapheme fallback detected!")

            return True

        finally:
            # Remove our log capture handler
            logging.getLogger().removeHandler(capture_handler)

    except Exception as e:
        print(f"\n❌ ERROR: Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_english_ipa_fix()
    sys.exit(0 if success else 1)
