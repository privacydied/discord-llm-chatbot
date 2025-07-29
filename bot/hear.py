"""
Centralized speech-to-text inference module (hear)
"""
import asyncio
import tempfile
import os
from pathlib import Path
from .stt import stt_manager
from .exceptions import InferenceError
from .util.logging import get_logger

logger = get_logger(__name__)

def _get_file_extension(filename: str) -> str:
    """Get the lowercase file extension with leading dot."""
    return os.path.splitext(filename)[1].lower()

async def hear_infer(audio_path: Path) -> str:
    """Transcribe audio to text with automatic format normalization and 1.5x speedup"""
    temp_wav = None
    temp_sped_up = None
    
    try:
        logger.info(f"👂 STT inference started for {audio_path}")
        if not stt_manager.is_available():
            raise InferenceError("STT engine not available")
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file1, \
             tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file2:
            
            temp_normalized = Path(temp_file1.name)
            temp_sped_up = Path(temp_file2.name)
        
        try:
            # First normalize the audio to 16kHz mono WAV
            normalize_cmd = [
                "ffmpeg", "-y", "-i", str(audio_path),
                "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le",
                "-af", "aresample=async=1:first_pts=0",  # Handle timestamp gaps
                str(temp_normalized)
            ]
            
            proc = await asyncio.create_subprocess_exec(
                *normalize_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            
            if proc.returncode != 0:
                error_msg = f"ffmpeg normalization failed with code {proc.returncode}"
                if stderr:
                    error_msg += f": {stderr.decode()}"
                raise RuntimeError(error_msg)
            
            # Apply 1.5x speedup
            speedup_cmd = [
                "ffmpeg", "-y", "-i", str(temp_normalized),
                "-filter:a", "atempo=1.5",  # 1.5x speedup
                "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le",
                str(temp_sped_up)
            ]
            
            proc = await asyncio.create_subprocess_exec(
                *speedup_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            
            if proc.returncode != 0:
                error_msg = f"ffmpeg speedup failed with code {proc.returncode}"
                if stderr:
                    error_msg += f": {stderr.decode()}"
                raise RuntimeError(error_msg)
            
            # Transcribe the processed audio
            logger.debug(f"Transcribing processed audio: {temp_sped_up}")
            return await stt_manager.transcribe_async(temp_sped_up)
            
        finally:
            # Clean up temporary files
            for temp_file in [temp_normalized, temp_sped_up]:
                if temp_file and temp_file.exists():
                    try:
                        temp_file.unlink()
                    except Exception as e:
                        logger.warning(f"Error deleting temporary file {temp_file}: {e}")
                    
    except Exception as e:
        logger.error(f"👂 STT inference failed: {str(e)}", exc_info=True)
        error_msg = str(e).lower()
        
        # Provide more user-friendly error messages
        if "no such file or directory" in error_msg:
            raise InferenceError("Could not access the audio file")
        elif "invalid data" in error_msg or "invalid argument" in error_msg:
            raise InferenceError("The audio file is corrupted or in an unsupported format")
        elif "operation not permitted" in error_msg:
            raise InferenceError("Permission denied when processing the audio file")
        else:
            raise InferenceError(f"Speech recognition failed: {str(e)}")
    
    finally:
        # Final cleanup in case anything was missed
        for temp_file in [temp_wav, temp_sped_up]:
            if temp_file and temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception as e:
                    logger.warning(f"Error in final cleanup of {temp_file}: {e}")