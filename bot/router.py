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
from .tts_state import tts_state
from .tts_manager import tts_manager

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
        # Use the imported tts_state and tts_manager instances
        self.tts_state = tts_state
        self.tts_manager = tts_manager
    
    def _extract_mode(self, content: str) -> tuple[ProcessingMode, str]:
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
    
    def _auto_detect_mode(self, message: discord.Message, current_mode: ProcessingMode) -> ProcessingMode:
        """
        Auto-detect processing mode based on attachment types.
        
        Args:
            message: Discord message object with attachments
            current_mode: The current processing mode
            
        Returns:
            The auto-detected processing mode
        """
        # Check if any attachment is an audio file
        audio_extensions = {'.wav', '.mp3', '.ogg', '.opus', '.m4a', '.flac'}
        if any(attachment.filename.lower().endswith(tuple(audio_extensions)) for attachment in message.attachments):
            return ProcessingMode.STT
            
        # Check if any attachment is an image
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.webp'}
        if any(attachment.filename.lower().endswith(tuple(image_extensions)) for attachment in message.attachments):
            return ProcessingMode.VISION
            
        # Check if any attachment is a document
        document_extensions = {'.pdf', '.txt', '.md', '.docx'}
        if any(attachment.filename.lower().endswith(tuple(document_extensions)) for attachment in message.attachments):
            return ProcessingMode.TEXT
            
        return current_mode

    async def handle(self, message: discord.Message, raw_input: str, voice_only: bool = False) -> None:
        """
        Process a message through the appropriate pipeline based on content and mode.
        
        Args:
            message: The Discord message object
            raw_input: The raw input string (with prefix/mention already stripped)
            voice_only: If True, only send TTS response (no text response)
        """
        try:
            logger.debug(f"🔍 Raw input: '{raw_input}'")
            # Extract mode and clean input
            mode, cleaned_input = self._extract_mode(raw_input)
            logger.debug(f"🔍 Parsed mode: {mode}, Cleaned input: '{cleaned_input}'")
            
            # Check if this is a TTS command (!speak, !say) - preserve TTS intent
            is_tts_command = any(raw_input.strip().startswith(cmd) for cmd in ['!speak', '!say'])
            
            # Auto-detect mode based on attachments ONLY if no explicit mode AND not a TTS command
            if message.attachments and not self.mode_pattern.search(raw_input) and not is_tts_command:
                original_mode = mode
                mode = self._auto_detect_mode(message, mode)
                logger.debug(f"🔍 Auto-detected mode: {mode.name} (was {original_mode.name}) for message with attachments")
            
            logger.debug(f"🔍 Final processing mode: {mode.name}, TTS command: {is_tts_command}")
            
            # Handle TTS commands with attachments specially - single response flow
            if is_tts_command and message.attachments:
                logger.debug(f"🔊📎 Processing TTS command with attachments - single response flow")
                await self._process_tts_with_attachments(message, cleaned_input)
                return
            
            # Handle different processing modes
            if mode in (ProcessingMode.TEXT, ProcessingMode.BOTH) and (cleaned_input or message.attachments):
                # If there are attachments and we're in TEXT mode, process them
                if mode == ProcessingMode.TEXT and message.attachments:
                    logger.debug(f"📄 Processing TEXT mode with attachments")
                    await self._process_attachments(message, mode, cleaned_input)
                # Otherwise, process as regular text
                elif cleaned_input:
                    logger.debug(f"📝 Processing TEXT mode with input: '{cleaned_input[:50]}...'")
                    await self._process_text(message, cleaned_input, include_tts=(mode == ProcessingMode.BOTH), voice_only=voice_only)
            elif mode == ProcessingMode.TTS and cleaned_input:
                logger.debug(f"🔊 Processing TTS mode with input: '{cleaned_input[:50]}...'")
                await self._process_tts(message, cleaned_input)
            elif mode in (ProcessingMode.STT, ProcessingMode.VISION):
                logger.debug(f"🎙️ Processing {'STT' if mode == ProcessingMode.STT else 'VISION'} mode with attachments")
                await self._process_attachments(message, mode, cleaned_input)
            else:
                logger.warning(f"⚠️ No valid input or mode specified for message")
                await message.channel.send("❌ No valid input or mode specified")
                
        except ValueError as e:
            logger.error(f"❌ ValueError in router.handle: {str(e)}")
            await message.channel.send(f"❌ {str(e)}")
        except Exception as e:
            logger.error(f"Error in router.handle: {str(e)}", exc_info=True)
            await message.channel.send("⚠️ An error occurred while processing your request")
    
    async def _process_text(self, message: discord.Message, text: str, include_tts: bool = False, voice_only: bool = False) -> None:
        """Process text input through the LLM pipeline."""
        response = await brain_infer(text)
        
        # Check if we should use TTS
        use_tts = voice_only  # Always use TTS if voice_only is True
        
        if not use_tts:  # Only check other conditions if voice_only is False
            if self.tts_state.get_and_clear_one_time_tts(message.author.id):
                use_tts = True
                logging.debug(f"🔊 Using one-time TTS for user {message.author.id}")
            elif self.tts_state.is_user_tts_enabled(message.author.id):
                use_tts = True
                logging.debug(f"🔊 Using TTS for user {message.author.id} (user preference or global setting)")
        
        if use_tts and self.tts_manager.is_available():
            # Synthesize and send as voice ONLY
            try:
                logger.debug(f"🔊 Generating TTS for response (length: {len(response)}): '{response[:50]}...'")
                audio_path = await self.tts_manager.generate_tts(response, self.tts_manager.voice)
                await message.channel.send(file=discord.File(audio_path))
                logging.debug(f"🔊 TTS response sent successfully")
            except Exception as e:
                logger.error(f"Error in TTS synthesis: {e}", exc_info=True)
                # Only fall back to text if not voice_only
                if not voice_only:
                    await message.channel.send(response)  # Fallback to text
                    logging.warning(f"⚠️ TTS failed, falling back to text response")
                else:
                    await message.channel.send(f"⚠️ Failed to generate speech: {str(e)}")
        elif not voice_only:  # Only send text if not voice_only
            # Text-only response (TTS not enabled or not available)
            await message.channel.send(response)
    
    async def _process_tts(self, message: discord.Message, text: str) -> None:
        """Process text as TTS only."""
        try:
            if not self.tts_manager.is_available():
                await message.channel.send("❌ TTS is not available. Please check your configuration.")
                return
                
            audio_path = await self.tts_manager.generate_tts(text, self.tts_manager.voice)
            await message.channel.send(file=discord.File(audio_path))
            logging.debug(f"🔊 Direct TTS response sent successfully")
        except Exception as e:
            logger.error(f"Error in TTS synthesis: {e}", exc_info=True)
            await message.channel.send(f"⚠️ Failed to generate speech: {str(e)}")
    
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
    
    async def _process_attachments(self, message: discord.Message, mode: ProcessingMode, cleaned_input: str) -> None:
        """
        Process message attachments based on the specified mode.
        
        Args:
            message: The Discord message object
            mode: The processing mode
            cleaned_input: Additional text input
        """
        try:
            # Create temporary directory for downloads
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_dir_path = Path(tmp_dir)
                files = []
                
                # Download all attachments
                for attachment in message.attachments:
                    file_path = tmp_dir_path / attachment.filename
                    await attachment.save(file_path)
                    files.append(file_path)
                    logger.debug(f"⬇️ Downloaded attachment: {attachment.filename}")
                
                # Process based on mode
                if mode == ProcessingMode.STT:
                    # Validate at least one audio file exists
                    audio_files = [f for f in files if f.suffix.lower() in {'.wav', '.mp3', '.ogg', '.opus', '.m4a', '.flac'}]
                    if not audio_files:
                        raise ValueError("No valid audio files found for STT processing")
                    
                    # Process each audio file
                    for audio_file in audio_files:
                        transcript = await hear_infer(audio_file)
                        if transcript:
                            await self._process_text(message, f"{cleaned_input}\n{transcript}".strip(), include_tts=False)
                
                elif mode == ProcessingMode.VISION:
                    # Check if this is coming from a speak or say command (voice_only)
                    voice_only = self.tts_state.get_and_clear_one_time_tts(message.author.id) or self.tts_state.is_user_tts_enabled(message.author.id)
                    
                    # Get image files
                    image_files = [f for f in files if f.suffix.lower() in {'.png', '.jpg', '.jpeg', '.gif', '.webp'}]
                    if not image_files:
                        raise ValueError("No valid image files found for vision processing")
                    
                    # 1. Process image through VL pipeline
                    prompt = cleaned_input or "What's in this image?"
                    logger.info(f"👁️ Processing image with prompt: '{prompt}'")
                    vision_results = await see_infer(image_files, prompt)
                    
                    # 2. Pass VL results to text model for further processing if there's a prompt
                    if cleaned_input:
                        # For TTS commands, we want the full pipeline: image > VL > text > TTS
                        combined_prompt = f"Image description: {vision_results}\n\nUser request: {cleaned_input}"
                        logger.debug(f"👁️➡️💬 Sending VL results to text model: '{combined_prompt[:50]}...'")
                        await self._process_text(message, combined_prompt, include_tts=False, voice_only=voice_only)
                    else:
                        # Just the VL description
                        await self._process_text(message, vision_results, include_tts=False, voice_only=voice_only)
                
                elif mode == ProcessingMode.TEXT:
                    # Process document pipeline
                    document_text = await self._process_document(files)
                    await self._process_text(message, f"{cleaned_input}\n{document_text}".strip())
                
        except Exception as e:
            logger.error(f"Error in _process_attachments: {str(e)}", exc_info=True)
            await message.channel.send(f"⚠️ An error occurred while processing your request: {str(e)}")
    
    async def _process_tts_with_attachments(self, message: discord.Message, cleaned_input: str) -> None:
        """
        Handle TTS commands (!speak, !say) with attachments.
        Processes attachments through appropriate pipeline and returns ONLY a voice response.
        No text response is sent - voice only.
        
        Args:
            message: The Discord message object
            cleaned_input: The cleaned input text
        """
        try:
            # Create temporary directory for downloads
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_dir_path = Path(tmp_dir)
                files = []
                
                # Download all attachments
                for attachment in message.attachments:
                    file_path = tmp_dir_path / attachment.filename
                    await attachment.save(file_path)
                    files.append(file_path)
                    logger.debug(f"⬇️ Downloaded attachment for TTS: {attachment.filename}")
                
                # Auto-detect file types and process accordingly
                final_text = cleaned_input or ""
                
                # Process images through VL pipeline
                image_files = [f for f in files if f.suffix.lower() in {'.png', '.jpg', '.jpeg', '.gif', '.webp'}]
                if image_files:
                    prompt = cleaned_input or "What's in this image?"
                    logger.info(f"👁️ Processing image for TTS with prompt: '{prompt}'")
                    vision_results = await see_infer(image_files, prompt)
                    
                    if cleaned_input:
                        # Full pipeline: image > VL > text model > TTS
                        combined_prompt = f"Image description: {vision_results}\n\nUser request: {cleaned_input}"
                        final_text = await brain_infer(combined_prompt)
                    else:
                        # Just describe the image through text model
                        final_text = await brain_infer(f"Describe this image in a conversational way: {vision_results}")
                
                # Process documents
                document_files = [f for f in files if f.suffix.lower() in {'.pdf', '.txt', '.md', '.docx'}]
                if document_files:
                    document_text = await self._process_document(document_files)
                    if cleaned_input:
                        combined_prompt = f"Document content: {document_text}\n\nUser request: {cleaned_input}"
                        final_text = await brain_infer(combined_prompt)
                    else:
                        # Summarize the document
                        final_text = await brain_infer(f"Summarize this document: {document_text}")
                
                # Process audio files through STT
                audio_files = [f for f in files if f.suffix.lower() in {'.wav', '.mp3', '.ogg', '.opus', '.m4a', '.flac'}]
                if audio_files:
                    for audio_file in audio_files:
                        transcript = await hear_infer(audio_file)
                        if transcript:
                            if cleaned_input:
                                combined_prompt = f"Audio transcript: {transcript}\n\nUser request: {cleaned_input}"
                                final_text = await brain_infer(combined_prompt)
                            else:
                                # Respond to the transcribed audio
                                final_text = await brain_infer(f"Respond to this: {transcript}")
                            break  # Process only the first audio file
                
                # If no attachments processed but we have text, use text model
                if not final_text and cleaned_input:
                    final_text = await brain_infer(cleaned_input)
                
                # If still no text, provide default response
                if not final_text:
                    final_text = "I received your message but couldn't process the content. Could you try again?"
                
                # Generate and send TTS response (single response only)
                logger.debug(f"🔊 Generating TTS for: '{final_text[:50]}...'")
                audio_path = await self.tts_manager.generate_tts(final_text, self.tts_manager.voice)
                
                # Send only the TTS audio file (single response)
                with open(audio_path, 'rb') as f:
                    audio_file = discord.File(f, filename='tts_output.wav')
                    await message.channel.send(file=audio_file)
                
                logger.debug(f"✅ TTS with attachments completed successfully")
                
        except Exception as e:
            logger.error(f"Error in _process_tts_with_attachments: {str(e)}", exc_info=True)
            await message.channel.send(f"⚠️ An error occurred while processing TTS with attachments: {str(e)}")
    
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
