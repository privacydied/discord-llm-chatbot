"""
Centralized router for handling multimodal message processing.
"""
import logging
import os
import re
import tempfile
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union

import discord

# Set up logger at module level
logger = logging.getLogger(__name__)

from .brain import brain_infer
from .exceptions import InferenceError
from .hear import hear_infer
from .see import see_infer
from .speak import speak_infer

# Import PDF processor if available
try:
    from .pdf_utils import PDFProcessor
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    logger.warning("PDF processing not available (missing dependencies)")

# Import docx if available
try:
    import docx
    DOCX_SUPPORT = True
except ImportError as e:
    DOCX_SUPPORT = False
    logger.warning("DOCX processing not available (python-docx not installed): %s", str(e))

class ProcessingMode(Enum):
    """Supported processing modes for the router."""
    TEXT = auto()
    TTS = auto()
    STT = auto()
    VISION = auto()
    BOTH = auto()

class Router:
    """Handles routing of messages to appropriate processing pipelines."""
    
    def __init__(self, bot):
        self.bot = bot
        self.mode_pattern = re.compile(r'--mode=([a-z,]+)')
        self.valid_modes = {
            'text': ProcessingMode.TEXT,
            'tts': ProcessingMode.TTS,
            'stt': ProcessingMode.STT,
            'vl': ProcessingMode.VISION,
            'both': ProcessingMode.BOTH
        }
    
    def _extract_mode(self, content: str) -> Tuple[Optional[ProcessingMode], str]:
        """
        Extract processing mode from message content.
        
        Args:
            content: The raw message content
            
        Returns:
            Tuple of (mode, cleaned_content)
        """
        # Look for --mode=... pattern
        mode_match = self.mode_pattern.search(content)
        if not mode_match:
            return ProcessingMode.TEXT, content.strip()
            
        mode_str = mode_match.group(1).lower()
        if mode_str not in self.valid_modes:
            raise ValueError(f"Invalid mode '{mode_str}'. Valid modes: {', '.join(self.valid_modes.keys())}")
            
        # Remove the mode flag from content
        cleaned_content = self.mode_pattern.sub('', content).strip()
        return self.valid_modes[mode_str], cleaned_content
    
    async def handle(self, message: discord.Message, raw_input: str) -> None:
        """
        Process a message through the appropriate pipeline based on content and mode.
        
        Args:
            message: The Discord message object
            raw_input: The raw input string (with prefix/mention already stripped)
        """
        try:
            # Extract mode and clean input
            mode, cleaned_input = self._extract_mode(raw_input)
            
            # Handle different processing modes
            if mode in (ProcessingMode.TEXT, ProcessingMode.BOTH) and (cleaned_input or message.attachments):
                # If there are attachments and we're in TEXT mode, process them
                if mode == ProcessingMode.TEXT and message.attachments:
                    await self._process_attachments(message, mode, cleaned_input)
                # Otherwise, process as regular text
                elif cleaned_input:
                    await self._process_text(message, cleaned_input, include_tts=(mode == ProcessingMode.BOTH))
            elif mode == ProcessingMode.TTS and cleaned_input:
                await self._process_tts(message, cleaned_input)
            elif mode in (ProcessingMode.STT, ProcessingMode.VISION):
                await self._process_attachments(message, mode, cleaned_input)
            else:
                await message.channel.send("❌ No valid input or mode specified")
                
        except ValueError as e:
            await message.channel.send(f"❌ {str(e)}")
        except Exception as e:
            logger.error(f"Error in router.handle: {str(e)}", exc_info=True)
            await message.channel.send("⚠️ An error occurred while processing your request")
    
    async def _process_text(self, message: discord.Message, text: str, include_tts: bool = False) -> None:
        """Process text input through the LLM pipeline."""
        response = await brain_infer(text)
        
        # Send text response
        sent_message = await message.channel.send(response)
        
        # Generate TTS if requested and available
        if include_tts:
            await self._generate_tts(message, response, sent_message)
    
    async def _process_tts(self, message: discord.Message, text: str) -> None:
        """Process text as TTS only."""
        await self._generate_tts(message, text)
    
    async def _process_document(self, file_path: str, file_ext: str) -> str:
        """Process a document file and return its text content."""
        try:
            # Handle PDF files
            if file_ext == '.pdf':
                if not PDF_SUPPORT:
                    raise InferenceError("PDF processing is not available (missing dependencies)")
                
                processor = PDFProcessor()
                if not processor.supported:
                    raise InferenceError("PDF processing is not available (missing dependencies)")
                
                # Read the file content into memory
                with open(file_path, 'rb') as f:
                    file_content = f.read()
                
                # Create a temporary file for OCR processing
                import tempfile
                import os
                
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                    temp_file.write(file_content)
                    temp_path = temp_file.name
                
                try:
                    # Check if it's a scanned PDF using the in-memory content
                    from io import BytesIO
                    file_obj = BytesIO(file_content)
                    
                    if processor.is_scanned_pdf(file_obj):
                        logger.info("Detected scanned PDF, attempting OCR...")
                        # Use the temporary file path for OCR
                        result = processor.extract_all(temp_path)
                    else:
                        # For regular PDFs, use the in-memory content
                        result = processor.extract_all(file_obj)
                    
                    if result.get('error'):
                        raise InferenceError(f"Failed to extract text from PDF: {result['error']}")
                    
                    content = result.get('text', '').strip()
                    if not content:
                        raise InferenceError("The PDF appears to be empty or could not be read")
                    
                    return content
                    
                finally:
                    # Clean up the temporary file
                    try:
                        os.unlink(temp_path)
                    except Exception as e:
                        logger.warning(f"Error deleting temporary file {temp_path}: {e}")
            
            # Handle text files
            elif file_ext in ['.txt', '.md']:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read().strip()
                if not content:
                    raise InferenceError("The text file is empty")
                return content
            
            # Handle DOCX files
            elif file_ext == '.docx':
                if not DOCX_SUPPORT:
                    raise InferenceError("DOCX processing requires python-docx package")
                
                doc = docx.Document(file_path)
                content = '\n'.join([para.text for para in doc.paragraphs if para.text.strip()])
                if not content:
                    raise InferenceError("The document appears to be empty")
                return content
            
            else:
                raise InferenceError(f"Unsupported file type: {file_ext}")
                
        except UnicodeDecodeError:
            raise InferenceError(f"Could not decode the file. It may be corrupted or in an unsupported encoding.")
        except Exception as e:
            logger.error(f"Error in _process_document: {str(e)}", exc_info=True)
            raise InferenceError(f"Error processing document: {str(e)}")
    
    async def _process_attachments(self, message: discord.Message, mode: ProcessingMode, prompt: str = '') -> None:
        """Process message attachments based on mode."""
        if not message.attachments:
            await message.channel.send("❌ No attachment found. Please attach a file.")
            return
            
        attachment = message.attachments[0]
        file_ext = Path(attachment.filename).suffix.lower()
        
        try:
            # Document processing
            if mode == ProcessingMode.TEXT and file_ext in ['.pdf', '.txt', '.md', '.docx']:
                file_path = await self._download_attachment(attachment)
                logger.debug(f"📄 Processing document: {file_path}")
                
                try:
                    # Process the document
                    content = await self._process_document(file_path, file_ext)
                    logger.debug(f"📝 Extracted {len(content)} characters from document")
                    
                    # Process with LLM
                    enhanced_prompt = f"Document content:\n{content}\n\nUser prompt: {prompt if prompt else 'Please analyze this document'}"
                    response = await brain_infer(enhanced_prompt)
                    await message.channel.send(response)
                    
                except InferenceError as e:
                    await message.channel.send(f"❌ {str(e)}")
                except Exception as e:
                    logger.error(f"Error processing document: {str(e)}", exc_info=True)
                    await message.channel.send("❌ An error occurred while processing the document")
                finally:
                    # Clean up the downloaded file
                    try:
                        Path(file_path).unlink()
                    except Exception as e:
                        logger.warning(f"Error deleting temporary file {file_path}: {str(e)}")
            
            # Speech-to-text processing
            elif mode == ProcessingMode.STT:
                supported_audio_formats = ['.wav', '.mp3', '.ogg', '.opus']
                if not any(attachment.filename.lower().endswith(ext) for ext in supported_audio_formats):
                    await message.channel.send(f"❌ Please provide an audio file ({', '.join(supported_audio_formats)})")
                    return
                    
                audio_path_str = await self._download_attachment(attachment)
                audio_path = Path(audio_path_str)  # Ensure we have a Path object
                try:
                    logger.debug(f"🎙️ Processing audio file: {audio_path}")
                    transcribed = await hear_infer(audio_path)  # Pass Path object
                    logger.debug(f"🎙️ Transcribed text: {transcribed[:200]}...")
                    await self._process_text(message, transcribed)
                except Exception as e:
                    logger.error(f"Error in STT processing: {str(e)}", exc_info=True)
                    await message.channel.send(f"❌ Error processing audio: {str(e)}")
                finally:
                    try:
                        Path(audio_path).unlink()
                        logger.debug(f"🗑️ Deleted temporary audio file: {audio_path}")
                    except Exception as e:
                        logger.warning(f"Error deleting temporary file {audio_path}: {str(e)}")
                
            # Vision-language processing
            elif mode == ProcessingMode.VISION:
                if not any(attachment.filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif']):
                    await message.channel.send("❌ Please provide an image file (.jpg, .png, .gif)")
                    return
                    
                image_path = await self._download_attachment(attachment)
                try:
                    logger.debug(f"👁️ Processing image with VL model: {image_path}")
                    
                    # Get VL model description
                    vl_result = await see_infer(image_path, prompt)
                    logger.debug(f"👁️ VL model output: {vl_result[:200]}...")
                    
                    # Create enhanced prompt for text model
                    enhanced_prompt = f"Based on this image description: {vl_result}\n\nUser prompt: {prompt if prompt else 'Tell me about this image'}"
                    logger.debug(f"📝 Sending to text model: {enhanced_prompt[:200]}...")
                    
                    # Get text model response
                    text_response = await brain_infer(enhanced_prompt)
                    logger.debug("✅ Text model response generated")
                    
                    # Send the final response
                    await message.channel.send(text_response)
                finally:
                    try:
                        Path(image_path).unlink()
                    except:
                        pass
            else:
                await message.channel.send(f"❌ Unsupported file type or mode for processing: {file_ext}")
                
        except InferenceError as e:
            await message.channel.send(f"❌ {str(e)}")
        except Exception as e:
            logger.error(f"Error processing attachment: {str(e)}", exc_info=True)
            await message.channel.send("⚠️ An error occurred while processing the attachment")
    
    async def _generate_tts(self, message: discord.Message, text: str, reply_to: discord.Message = None) -> None:
        """Generate and send TTS audio."""
        try:
            audio_path = await speak_infer(text)
            
            # Send the audio file
            with open(audio_path, 'rb') as f:
                audio_file = discord.File(f, filename='tts_output.mp3')
                await (reply_to or message.channel).send(file=audio_file)
                
        except ImportError:
            logger.warning("TTS module not available")
            if not reply_to:  # Only send this if we're not in a combined text+tts flow
                await message.channel.send("❌ TTS is not available. The Kokoro-ONNX TTS package is not installed or failed to initialize.")
        except Exception as e:
            logger.error(f"Error in TTS generation: {str(e)}", exc_info=True)
            if not reply_to:  # Only send this if we're not in a combined text+tts flow
                await message.channel.send("⚠️ An error occurred while generating speech")
    
    async def _download_attachment(self, attachment: discord.Attachment) -> str:
        """Download an attachment to a temporary file and return the path."""
        import tempfile
        import os
        
        _, ext = os.path.splitext(attachment.filename)
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp_file:
            await attachment.save(temp_file.name)
            return temp_file.name

# Global router instance
router = None

def setup_router(bot) -> Router:
    """Initialize and return the global router instance."""
    global router
    router = Router(bot)
    return router

def get_router() -> Router:
    """Get the global router instance."""
    if router is None:
        raise RuntimeError("Router not initialized. Call setup_router() first.")
    return router
