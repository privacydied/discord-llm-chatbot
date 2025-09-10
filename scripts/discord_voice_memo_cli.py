#!/usr/bin/env python3
"""
Discord Voice Memo CLI Tool.

Converts text to speech using Kokoro TTS and sends as Discord voice memo.
Optional script for testing voice memo functionality.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add bot module to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.tts.engines.kokoro import KokoroONNXEngine
from bot.infra.voice_memo_sender import send_tts_voice_memo, VoiceMemoError

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Setup basic logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Send TTS audio as Discord voice memo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s 123456789 "Hello world"
  %(prog)s --voice af_sky --speed 1.2 123456789 "Testing voice memo"
  %(prog)s --token $BOT_TOKEN 123456789 "Custom token example"

Environment Variables:
  DISCORD_TOKEN    - Discord bot token (required if --token not provided)
  KOKORO_MODEL     - Path to Kokoro ONNX model
  KOKORO_VOICES    - Path to Kokoro voice embeddings
        """,
    )

    parser.add_argument(
        "channel_id", type=int, help="Discord channel ID to send voice memo to"
    )

    parser.add_argument("text", help="Text to convert to speech")

    parser.add_argument(
        "--token", help="Discord bot token (overrides DISCORD_TOKEN env var)"
    )

    parser.add_argument(
        "--voice", default="af_sky", help="Voice to use for synthesis (default: af_sky)"
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speech speed multiplier (default: 1.0)",
    )

    parser.add_argument(
        "--model-path", help="Path to Kokoro ONNX model (overrides KOKORO_MODEL env)"
    )

    parser.add_argument(
        "--voices-path",
        help="Path to Kokoro voice embeddings (overrides KOKORO_VOICES env)",
    )

    parser.add_argument("--language", default="en", help="Language code (default: en)")

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    # Validate Discord token
    bot_token = args.token or os.getenv("DISCORD_TOKEN")
    if not bot_token:
        logger.error(
            "Discord bot token required. Use --token or set DISCORD_TOKEN environment variable."
        )
        return 1

    # Get model paths
    model_path = args.model_path or os.getenv("KOKORO_MODEL")
    voices_path = args.voices_path or os.getenv("KOKORO_VOICES")

    if not model_path:
        logger.error(
            "Kokoro model path required. Use --model-path or set KOKORO_MODEL environment variable."
        )
        return 1

    if not voices_path:
        logger.error(
            "Kokoro voices path required. Use --voices-path or set KOKORO_VOICES environment variable."
        )
        return 1

    # Validate paths exist
    if not Path(model_path).exists():
        logger.error(f"Model file not found: {model_path}")
        return 1

    if not Path(voices_path).exists():
        logger.error(f"Voices file not found: {voices_path}")
        return 1

    try:
        # Initialize TTS engine
        logger.info("Initializing Kokoro TTS engine...")
        engine = KokoroONNXEngine(model_path=model_path, voices_path=voices_path)

        # Generate speech
        logger.info(
            f"Generating speech for: '{args.text[:50]}{'...' if len(args.text) > 50 else ''}'"
        )
        wav_bytes = engine.synthesize(
            text=args.text, language=args.language, voice=args.voice, speed=args.speed
        )

        if not wav_bytes:
            logger.error("TTS synthesis failed - no audio data generated")
            return 1

        logger.info(f"Generated {len(wav_bytes)} bytes of audio")

        # Send as voice memo
        logger.info(f"Sending voice memo to channel {args.channel_id}...")
        response = send_tts_voice_memo(
            channel_id=args.channel_id, wav_bytes=wav_bytes, bot_token=bot_token
        )

        message_id = response.get("id")
        logger.info(f"Voice memo sent successfully! Message ID: {message_id}")

        return 0

    except VoiceMemoError as e:
        logger.error(f"Voice memo error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
