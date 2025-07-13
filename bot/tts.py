"""
Text-to-speech functionality using DIA TTS.
"""
import os
import tempfile
import logging
import asyncio
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import discord

# Import config
from .config import load_config
from .utils import is_audio_file, download_file

# Load configuration
config = load_config()

# Try to import DIA TTS
try:
    import dia_tts
    DIA_AVAILABLE = True
except ImportError:
    DIA_AVAILABLE = False
    logging.warning("DIA TTS not available. TTS functionality will be disabled.")

# Initialize TTS engine
if DIA_AVAILABLE:
    try:
        tts_engine = dia_tts.DIATTS()
        logging.info("DIA TTS engine initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize DIA TTS engine: {e}")
        DIA_AVAILABLE = False

# Track TTS settings
tts_settings = {
    'global_enabled': True,
    'user_settings': {},  # user_id -> {'enabled': bool}
    'voice': 'en-US-Wavenet-D',  # Default voice
    'speaking_rate': 1.0,  # Default speaking rate
}

# Lock for thread safety
tts_lock = asyncio.Lock()

async def is_tts_enabled(user_id: str) -> bool:
    """Check if TTS is enabled for a user."""
    if not tts_settings['global_enabled']:
        return False
    
    # Check user-specific settings
    user_setting = tts_settings['user_settings'].get(str(user_id), {})
    return user_setting.get('enabled', True)

def set_tts_enabled(user_id: str, enabled: bool) -> bool:
    """Enable or disable TTS for a user."""
    user_id = str(user_id)
    if user_id not in tts_settings['user_settings']:
        tts_settings['user_settings'][user_id] = {}
    
    tts_settings['user_settings'][user_id]['enabled'] = enabled
    return True

def set_global_tts(enabled: bool) -> bool:
    """Enable or disable TTS globally."""
    tts_settings['global_enabled'] = enabled
    return True

async def generate_tts_audio(text: str, voice: Optional[str] = None, speaking_rate: Optional[float] = None) -> Optional[Path]:
    """
    Generate TTS audio from text using DIA TTS.
    
    Args:
        text: The text to convert to speech
        voice: Optional voice to use (default: configured voice)
        speaking_rate: Optional speaking rate (default: configured rate)
        
    Returns:
        Path to the generated audio file, or None if generation failed
    """
    if not DIA_AVAILABLE or not text.strip():
        return None
    
    # Use configured defaults if not specified
    voice = voice or tts_settings['voice']
    speaking_rate = speaking_rate or tts_settings['speaking_rate']
    
    try:
        # Create temp directory if it doesn't exist
        temp_dir = Path(config["TEMP_DIR"])
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a temporary file for the output
        with tempfile.NamedTemporaryFile(
            suffix='.wav', 
            dir=temp_dir,
            delete=False
        ) as temp_file:
            output_path = Path(temp_file.name)
        
        # Generate speech
        tts_engine.text_to_speech(
            text=text,
            output_file=str(output_path),
            voice_name=voice,
            speaking_rate=speaking_rate
        )
        
        # Verify the file was created and has content
        if output_path.exists() and output_path.stat().st_size > 0:
            return output_path
        
        logging.error(f"TTS generated empty file: {output_path}")
        return None
        
    except Exception as e:
        logging.error(f"Error generating TTS audio: {e}")
        # Clean up temp file if it was created
        if 'output_path' in locals() and output_path.exists():
            try:
                output_path.unlink()
            except OSError:
                pass
        return None

async def send_tts_reply(
    message: discord.Message, 
    text: str, 
    voice: Optional[str] = None,
    speaking_rate: Optional[float] = None,
    delete_after: bool = True
) -> bool:
    """
    Send a TTS reply to a message.
    
    Args:
        message: The Discord message to reply to
        text: The text to convert to speech
        voice: Optional voice to use
        speaking_rate: Optional speaking rate
        delete_after: Whether to delete the audio file after sending
        
    Returns:
        bool: True if the TTS was sent successfully, False otherwise
    """
    if not DIA_AVAILABLE or not await is_tts_enabled(message.author.id):
        return False
    
    # Generate TTS audio
    audio_path = await generate_tts_audio(text, voice, speaking_rate)
    if not audio_path:
        return False
    
    try:
        # Send the audio file
        with open(audio_path, 'rb') as audio_file:
            await message.reply(
                file=discord.File(audio_file, filename="tts_reply.wav"),
                mention_author=False
            )
        
        return True
        
    except Exception as e:
        logging.error(f"Error sending TTS reply: {e}")
        return False
        
    finally:
        # Clean up the temp file
        if delete_after and audio_path.exists():
            try:
                audio_path.unlink()
            except OSError as e:
                logging.error(f"Error deleting TTS temp file {audio_path}: {e}")

def get_available_voices() -> Dict[str, Any]:
    """Get a list of available TTS voices."""
    if not DIA_AVAILABLE:
        return {}
    
    try:
        return tts_engine.list_voices()
    except Exception as e:
        logging.error(f"Error getting available voices: {e}")
        return {}

def set_voice(voice_name: str) -> bool:
    """Set the default TTS voice."""
    if not DIA_AVAILABLE:
        return False
    
    try:
        voices = tts_engine.list_voices()
        if voice_name in voices:
            tts_settings['voice'] = voice_name
            return True
        return False
    except Exception as e:
        logging.error(f"Error setting voice: {e}")
        return False

def set_speaking_rate(rate: float) -> bool:
    """Set the default speaking rate."""
    try:
        rate = float(rate)
        # Limit rate to reasonable values (0.25x to 4.0x)
        rate = max(0.25, min(4.0, rate))
        tts_settings['speaking_rate'] = rate
        return True
    except (ValueError, TypeError):
        return False

async def setup_tts() -> None:
    """Initialize TTS system."""
    try:
        # Ensure temp directory exists
        temp_dir = Path(config["TEMP_DIR"])
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        if DIA_AVAILABLE:
            logging.info("TTS system initialized successfully")
        else:
            logging.warning("TTS system initialized without DIA TTS support")
            
    except Exception as e:
        logging.error(f"Error during TTS setup: {e}")

def cleanup_tts() -> None:
    """Clean up TTS resources and temporary files."""
    try:
        # Clean up temporary TTS files
        temp_dir = Path(config["TEMP_DIR"])
        if temp_dir.exists():
            for file_path in temp_dir.glob("*.wav"):
                try:
                    file_path.unlink()
                except OSError as e:
                    logging.error(f"Error deleting TTS temp file {file_path}: {e}")
        
        # Reset TTS settings
        tts_settings['user_settings'].clear()
        
        logging.info("TTS cleanup completed")
        
    except Exception as e:
        logging.error(f"Error during TTS cleanup: {e}")
