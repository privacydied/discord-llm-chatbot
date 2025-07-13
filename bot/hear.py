"""
Centralized speech-to-text inference module (hear)
"""
import asyncio
import logging
import tempfile
import shutil
from pathlib import Path
from .stt import stt_manager, normalise_to_wav
from .exceptions import InferenceError

logger = logging.getLogger(__name__)

async def hear_infer(audio_path: Path) -> str:
    """Transcribe audio to text with automatic format normalization"""
    temp_wav = None
    try:
        logger.info(f"👂 STT inference started for {audio_path}")
        if not stt_manager.is_available():
            raise InferenceError("STT engine not available")
        
        # Create a temporary file for the normalized audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_wav_path = Path(temp_file.name)
        
        try:
            # Convert to WAV format with ffmpeg
            cmd = [
                "ffmpeg", "-y", "-i", str(audio_path),
                "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le",
                str(temp_wav_path)
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            
            if proc.returncode != 0:
                error_msg = f"ffmpeg conversion failed with code {proc.returncode}"
                if stderr:
                    error_msg += f": {stderr.decode()}"
                raise RuntimeError(error_msg)
            
            # Transcribe the normalized audio
            logger.debug(f"Transcribing normalized audio: {temp_wav_path}")
            return await stt_manager.transcribe_async(temp_wav_path)
            
        finally:
            # Clean up the temporary file
            if temp_wav_path.exists():
                try:
                    temp_wav_path.unlink()
                except Exception as e:
                    logger.warning(f"Error deleting temporary file {temp_wav_path}: {e}")
                    
    except Exception as e:
        logger.error(f"👂 STT inference failed: {str(e)}", exc_info=True)
        raise InferenceError(f"Speech recognition failed: {str(e)}")
    
    finally:
        # Ensure we clean up the temporary file even if an exception occurs
        if temp_wav and temp_wav.exists():
            try:
                temp_wav.unlink()
            except Exception as e:
                logger.warning(f"Error cleaning up temporary file: {e}")