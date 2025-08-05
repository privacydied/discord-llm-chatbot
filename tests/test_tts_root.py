#!/usr/bin/env python3
"""
Test script for TTSManager with KokoroDirect integration.
"""

import asyncio
import logging
import os
from pathlib import Path

import discord
from discord.ext import commands
from bot.tts import TTSManager

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("tts_test")

async def test_tts():
    """Test the TTSManager with KokoroDirect integration."""
    logger.info("Creating bot instance for TTSManager test")
    # Create intents for the bot
    intents = discord.Intents.default()
    intents.message_content = True
    bot = commands.Bot(command_prefix='!', intents=intents)
    
    logger.info("Initializing TTSManager")
    tts = TTSManager(bot)
    
    # Check if TTS is available
    is_available = tts.is_available()
    logger.info(f"TTS available: {is_available}")
    
    # Get default voice
    default_voice = tts.voice
    logger.info(f"Default voice: {default_voice}")
    
    # List available voices
    if tts.voices:
        voice_sample = tts.voices[:5]
        logger.info(f"Available voices: {voice_sample}... (total: {len(tts.voices)})")
    else:
        logger.warning("No voices available")
    
    if is_available:
        # Generate TTS
        test_text = "This is a test of the TTS system with our new KokoroDirect implementation."
        logger.info(f"Generating TTS for text: '{test_text}'")
        
        try:
            audio_path = await tts.generate_tts(test_text)
            if audio_path:
                file_size = Path(audio_path).stat().st_size if os.path.exists(audio_path) else 0
                logger.info(f"Generated audio path: {audio_path} (size: {file_size} bytes)")
                
                # Check if file exists and has content
                if os.path.exists(audio_path) and file_size > 0:
                    logger.info("✅ TTS test PASSED: Audio file generated successfully")
                else:
                    logger.error("❌ TTS test FAILED: Audio file empty or missing")
            else:
                logger.error("❌ TTS test FAILED: No audio path returned")
        except Exception as e:
            logger.error(f"❌ TTS test FAILED: {e}", exc_info=True)
    else:
        logger.error("❌ TTS test FAILED: TTS not available")
    
    # Clean up
    await tts.close()
    logger.info("TTSManager test completed")

if __name__ == "__main__":
    asyncio.run(test_tts())
