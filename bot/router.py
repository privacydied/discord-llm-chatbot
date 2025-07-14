"""
Centralized router for handling multimodal message processing.
"""
import logging
import os
import re
import tempfile
from enum import Enum, auto
from pathlib import Path
from typing import List, Optional

import discord
from discord.ext import commands

# Set up logger at module level
logger = logging.getLogger(__name__)

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


class ResponseMessage:
    """Represents a response message with optional text and audio."""
    def __init__(self, text: str = None, audio_path: str = None):
        self.text = text
        self.audio_path = audio_path


class Router:
    """Handles routing of messages to appropriate processing pipelines."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.mode_pattern = re.compile(r'--mode=([a-z,]+)')
        self.valid_modes = {
            'text': ProcessingMode.TEXT,
            'tts': ProcessingMode.TTS,
            'stt': ProcessingMode.STT,
            'vl': ProcessingMode.VISION,
            'both': ProcessingMode.BOTH
        }
        self.config = bot.config
        # Use the imported tts_state and tts_manager instances
        from .tts_state import tts_state
        from .tts_manager import tts_manager
        self.tts_state = tts_state
        self.tts_manager = tts_manager

        # Initialize PDF Processor
        self.pdf_processor = None
        if PDF_SUPPORT:
            self.pdf_processor = PDFProcessor()
            self.pdf_processor.loop = bot.loop
            logger.info("[WIND] PDFProcessor initialized and attached to bot event loop.", extra={'subsys': 'router', 'event': 'pdf_processor.init'})




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
        response_message: Optional[ResponseMessage] = None
        try:
            logger.debug(f"[WIND] Raw input: '{raw_input}'", extra={'subsys': 'router', 'event': 'handle.entry'})
            mode, cleaned_input = self._extract_mode(raw_input)
            logger.debug(f"[WIND] Parsed mode: {mode}, Cleaned input: '{cleaned_input}'", extra={'subsys': 'router', 'event': 'handle.mode_extract'})

            is_tts_command = any(raw_input.strip().startswith(cmd) for cmd in ['!speak', '!say'])

            if message.attachments and not self.mode_pattern.search(raw_input) and not is_tts_command:
                original_mode = mode
                mode = self._auto_detect_mode(message, mode)
                logger.debug(f"[WIND] Auto-detected mode: {mode.name} (was {original_mode.name}) for attachments", extra={'subsys': 'router', 'event': 'handle.mode_autodetect'})

            logger.debug(f"[WIND] Final processing mode: {mode.name}, TTS command: {is_tts_command}", extra={'subsys': 'router', 'event': 'handle.final_mode'})

            if is_tts_command and message.attachments:
                logger.debug("[WIND] Routing to TTS with attachments...", extra={'subsys': 'router', 'event': 'handle.route.tts_attachments'})
                response_message = await self._process_tts_with_attachments(message, cleaned_input)
            elif mode in (ProcessingMode.TEXT, ProcessingMode.BOTH) and (cleaned_input or message.attachments):
                if mode == ProcessingMode.TEXT and message.attachments:
                    logger.debug("[WIND] Routing to TEXT with attachments...", extra={'subsys': 'router', 'event': 'handle.route.text_attachments'})
                    response_message = await self._process_attachments(message, mode, cleaned_input)
                elif cleaned_input:
                    logger.debug(f"[WIND] Routing to TEXT...", extra={'subsys': 'router', 'event': 'handle.route.text'})
                    response_message = await self._process_text(message, cleaned_input, include_tts=(mode == ProcessingMode.BOTH), voice_only=voice_only)
            elif mode == ProcessingMode.TTS and cleaned_input:
                logger.debug(f"[WIND] Routing to TTS...", extra={'subsys': 'router', 'event': 'handle.route.tts'})
                response_message = await self._process_tts(message, cleaned_input)
            elif mode in (ProcessingMode.STT, ProcessingMode.VISION):
                logger.debug(f"[WIND] Routing to {mode.name} with attachments...", extra={'subsys': 'router', 'event': 'handle.route.stt_vision'})
                response_message = await self._process_attachments(message, mode, cleaned_input)
            else:
                if not is_tts_command and not cleaned_input and not message.attachments:
                    logger.warning("[WIND] No valid input, attachments, or commands found. No action taken.", extra={'subsys': 'router', 'event': 'handle.no_action'})

            # Unified response sending
            if response_message:
                logger.debug(f"[WIND] Dispatching response: {response_message}", extra={'subsys': 'router', 'event': 'handle.dispatch'})
                if response_message.audio_path:
                    await message.channel.send(file=discord.File(response_message.audio_path))
                elif response_message.text:
                    await message.channel.send(response_message.text)

        except (ValueError, Exception) as e:
            logger.error(f"[WIND] Validation/Processing Error in router: {str(e)}", extra={'subsys': 'router', 'event': 'handle.error.validation'}, exc_info=True)
            await message.channel.send(f"❌ {str(e)}")

    async def _process_text(self, message: discord.Message, text: str, include_tts: bool = False, voice_only: bool = False) -> ResponseMessage:
        """Process text input through the LLM pipeline."""
        logger.debug(f"[WIND] Processing text (TTS: {include_tts}, Voice-Only: {voice_only})", extra={'subsys': 'router', 'event': 'process_text.entry'})
        
        # 1. Get LLM inference
        logger.debug("[WIND] Requesting brain inference...", extra={'subsys': 'router', 'event': 'process_text.brain_infer.request'})
        from .brain import brain_infer
        response = await brain_infer(text)
        logger.debug(f"[WIND] Brain inference received (len: {len(response)}): '{response[:70]}...'", extra={'subsys': 'router', 'event': 'process_text.brain_infer.response'})

        # 2. Determine if TTS should be used
        use_tts = voice_only or include_tts
        if self.tts_state.get_and_clear_one_time_tts(message.author.id):
            use_tts = True
            logger.debug(f"[WIND] TTS mode enabled by one-time flag for user {message.author.id}", extra={'subsys': 'router', 'event': 'process_text.tts_onetime'})

        # 3. Handle response delivery (TTS or Text)
        if use_tts and self.tts_manager.is_available():
            try:
                logger.debug(f"[WIND] Generating TTS audio...", extra={'subsys': 'router', 'event': 'process_text.tts_generate.request'})
                audio_path = await self.tts_manager.generate_tts(response, self.tts_manager.voice)
                logger.debug(f"[WIND] TTS audio generated at {audio_path}", extra={'subsys': 'router', 'event': 'process_text.tts_generate.response'})
                logger.debug("[WIND] TTS audio sent successfully", extra={'subsys': 'router', 'event': 'process_text.tts_sent'})
                return ResponseMessage(audio_path=audio_path)
            except Exception as e:
                logger.error(f"[WIND] TTS synthesis failed: {e}", extra={'subsys': 'router', 'event': 'process_text.error.tts'}, exc_info=True)
                if not voice_only:
                    logger.warning("[WIND] TTS failed, falling back to text response.", extra={'subsys': 'router', 'event': 'process_text.tts_fallback'})
                    return ResponseMessage(text=response)
                else:
                    return ResponseMessage(text=f"⚠️ Failed to generate speech: {str(e)}")
        elif not voice_only:
            logger.debug("[WIND] Sending text-only response.", extra={'subsys': 'router', 'event': 'process_text.send_text'})
            logger.debug("[WIND] Returning text-only response.", extra={'subsys': 'router', 'event': 'process_text.return_text'})
            return ResponseMessage(text=response)

    async def _process_tts(self, message: discord.Message, text: str) -> ResponseMessage:
        """Process text as TTS only."""
        logger.debug(f"[WIND] Processing TTS-only request (len: {len(text)}): '{text[:70]}...'", extra={'subsys': 'router', 'event': 'process_tts.entry'})
        try:
            # 1. Check for TTS availability
            if not self.tts_manager.is_available():
                logger.warning("[WIND] TTS is not available, raising ConfigurationError.", extra={'subsys': 'router', 'event': 'process_tts.not_available'})
                raise Exception("TTS is not available. Please check your configuration.")
            logger.debug("[WIND] TTS is available.", extra={'subsys': 'router', 'event': 'process_tts.available'})

            # 2. Generate audio
            logger.debug("[WIND] Requesting TTS audio generation...", extra={'subsys': 'router', 'event': 'process_tts.generate.request'})
            audio_path = await self.tts_manager.generate_tts(text, self.tts_manager.voice)
            logger.debug(f"[WIND] TTS audio generated at {audio_path}", extra={'subsys': 'router', 'event': 'process_tts.generate.response'})

            # 3. Send audio file
            logger.debug(f"[WIND] Sending TTS audio file to channel {message.channel.id}", extra={'subsys': 'router', 'event': 'process_tts.send.request'})
            logger.debug("[WIND] TTS audio generated, returning response message.", extra={'subsys': 'router', 'event': 'process_tts.return_audio'})
            return ResponseMessage(audio_path=audio_path)
        except Exception as e:
            logger.error(f"[WIND] Error during TTS processing: {e}", extra={'subsys': 'router', 'event': 'process_tts.error.known'}, exc_info=True)
            return ResponseMessage(text=f"⚠️ {str(e)}")

    async def _process_document(self, file_path: str, file_ext: str) -> str:
        """Process a document file and return its text content."""
        file_ext = file_ext.lower()
        file_name = Path(file_path).name
        logger.debug(f"[WIND] Processing document '{file_name}' ({file_ext})", extra={'subsys': 'router', 'event': 'process_document.entry'})

        try:
            if file_ext == '.pdf':
                logger.debug(f"[WIND] Handling PDF file: {file_name}", extra={'subsys': 'router', 'event': 'process_document.pdf'})
                if not PDF_SUPPORT or not self.pdf_processor:
                    logger.error("[WIND] PDF processing is not available or initialized.", extra={'subsys': 'router', 'event': 'process_document.pdf.unavailable'})
                    raise Exception("PDF processing is not available.")
                
                result = await self.pdf_processor.process(file_path)
                
                if result.get('error'):
                    logger.error(f"[WIND] PDF processing failed for {file_name}: {result['error']}", extra={'subsys': 'router', 'event': 'process_document.pdf.fail'})
                    raise Exception(f"PDF processing failed: {result['error']}")
                
                if result.get('text') and result['text'].strip():
                    return result['text']
                else:
                    logger.warning(f"[WIND] No text could be extracted from PDF: {file_name}", extra={'subsys': 'router', 'event': 'process_document.pdf.no_text'})
                    return ""

            elif file_ext in ['.txt', '.md']:
                logger.debug(f"[WIND] Handling text file: {file_name}", extra={'subsys': 'router', 'event': 'process_document.text'})
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()

            elif file_ext == '.docx':
                logger.debug(f"[WIND] Handling DOCX file: {file_name}", extra={'subsys': 'router', 'event': 'process_document.docx'})
                if not DOCX_SUPPORT:
                    logger.error("[WIND] DOCX support not available.", extra={'subsys': 'router', 'event': 'process_document.docx.no_support'})
                    raise Exception("DOCX processing is not available (python-docx not installed).")
                
                doc = docx.Document(file_path)
                return '\n'.join([para.text for para in doc.paragraphs])

            else:
                logger.warning(f"[WIND] Unsupported file type '{file_ext}' for document processing.", extra={'subsys': 'router', 'event': 'process_document.unsupported'})
                raise Exception(f"Unsupported file type: {file_ext}")

        except Exception as e:
            logger.error(f"[WIND] Unexpected error processing document '{file_name}': {e}", extra={'subsys': 'router', 'event': 'process_document.error.unexpected'}, exc_info=True)
            raise Exception(f"An unexpected error occurred while processing '{file_name}'.") from e

    async def _process_attachments(self, message: discord.Message, mode: ProcessingMode, cleaned_input: str) -> Optional[ResponseMessage]:
        """
        Process message attachments based on the specified mode.
        """
        logger.debug(f"[WIND] Processing attachments for mode: {mode.name}", extra={'subsys': 'router', 'event': 'process_attachments.entry'})
        try:
            # Enforce attachment size limit
            max_size_bytes = self.config.get("MAX_ATTACHMENT_SIZE_MB", 25) * 1024 * 1024
            for attachment in message.attachments:
                if attachment.size > max_size_bytes:
                    error_msg = f"Attachment '{attachment.filename}' is too large ({attachment.size / 1024 / 1024:.2f}MB). The maximum size is {self.config.get('MAX_ATTACHMENT_SIZE_MB')}MB."
                    logger.warning(f"[WIND] {error_msg}", extra={'subsys': 'router', 'event': 'process_attachments.size_exceeded'})
                    raise Exception(error_msg)

            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_dir_path = Path(tmp_dir)
                final_text = ""
                logger.debug(f"[WIND] Created temporary directory: {tmp_dir_path}", extra={'subsys': 'router', 'event': 'process_attachments.tmpdir'})
                downloaded_files = []
                for attachment in message.attachments:
                    file_path = tmp_dir_path / attachment.filename
                    logger.debug(f"[WIND] Downloading attachment: '{attachment.filename}'", extra={'subsys': 'router', 'event': 'process_attachments.download.start'})
                    await attachment.save(file_path)
                    downloaded_files.append(file_path)
                    logger.info(f"[WIND] Downloaded attachment: {attachment.filename}", extra={'subsys': 'router', 'event': 'process_attachments.download.success'})

                # Vision processing for images
                if mode == ProcessingMode.VISION:
                    logger.debug(f"[WIND] Processing {len(downloaded_files)} file(s) with vision model.", extra={'subsys': 'router', 'event': 'process_attachments.vision'})
                    from .see import see_infer
                    prompt = cleaned_input or self.config.get("VL_PROMPT_FILE", "Describe this image.")
                    final_text = await see_infer(downloaded_files, prompt)

                # STT processing for audio
                elif mode == ProcessingMode.STT:
                    logger.debug(f"[WIND] Processing {len(downloaded_files)} file(s) with STT model.", extra={'subsys': 'router', 'event': 'process_attachments.stt'})
                    from .hear import hear_infer
                    transcriptions = await hear_infer(downloaded_files)
                    final_text = " ".join(transcriptions)

                # Document/text processing for other files
                elif mode == ProcessingMode.TEXT:
                    logger.debug(f"[WIND] Processing {len(downloaded_files)} file(s) as text documents.", extra={'subsys': 'router', 'event': 'process_attachments.docs'})
                    doc_texts = []
                    for doc_path in downloaded_files:
                        doc_texts.append(await self._process_document(str(doc_path), doc_path.suffix))
                    combined_docs = "\n\n".join(doc_texts)
                    from .brain import brain_infer
                    final_text = await brain_infer(f"{cleaned_input}\n\n{combined_docs}")

                if final_text:
                    logger.debug(f"[WIND] Final text from attachments is not empty, routing to _process_text.", extra={'subsys': 'router', 'event': 'process_attachments.route_to_text'})
                    return await self._process_text(message, final_text)
                else:
                    logger.debug(f"[WIND] Final text from attachments is empty, no further action.", extra={'subsys': 'router', 'event': 'process_attachments.no_text'})
                    return None

        except Exception as e:
            logger.error(f"[WIND] Unexpected error processing attachments: {e}", extra={'subsys': 'router', 'event': 'process_attachments.error.unexpected'}, exc_info=True)
            return ResponseMessage(text=f"⚠️ An error occurred while processing attachments.")

    async def _process_tts_with_attachments(self, message: discord.Message, cleaned_input: str) -> Optional[ResponseMessage]:
        """
        Handle TTS commands (!speak, !say) with attachments.
        Processes attachments through appropriate pipeline and returns ONLY a voice response.
        """
        logger.debug("[WIND] Processing TTS command with attachments.", extra={'subsys': 'router', 'event': 'process_tts_attachments.entry'})
        try:
            # Enforce attachment size limit
            max_size_bytes = self.config.get("MAX_ATTACHMENT_SIZE_MB", 25) * 1024 * 1024
            for attachment in message.attachments:
                if attachment.size > max_size_bytes:
                    error_msg = f"Attachment '{attachment.filename}' is too large ({attachment.size / 1024 / 1024:.2f}MB). The maximum size is {self.config.get('MAX_ATTACHMENT_SIZE_MB')}MB."
                    logger.warning(f"[WIND] {error_msg}", extra={'subsys': 'router', 'event': 'process_tts_attachments.size_exceeded'})
                    raise Exception(error_msg)

            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_dir_path = Path(tmp_dir)
                final_text = ""
                logger.debug(f"[WIND] Created temporary directory for TTS attachments: {tmp_dir_path}", extra={'subsys': 'router', 'event': 'process_tts_attachments.tmpdir'})

                downloaded_files = []
                for attachment in message.attachments:
                    file_path = tmp_dir_path / attachment.filename
                    logger.debug(f"[WIND] Downloading attachment for TTS: '{attachment.filename}'", extra={'subsys': 'router', 'event': 'process_tts_attachments.download.start'})
                    await attachment.save(file_path)
                    downloaded_files.append(file_path)
                    logger.info(f"[WIND] Downloaded attachment for TTS: {attachment.filename}", extra={'subsys': 'router', 'event': 'process_tts_attachments.download.success'})

                image_files = [f for f in downloaded_files if f.suffix.lower() in {'.png', '.jpg', '.jpeg', '.gif', '.webp'}]
                doc_files = [f for f in downloaded_files if f.suffix.lower() in {'.pdf', '.txt', '.md', '.docx'}]

                if image_files:
                    logger.debug(f"[WIND] Found {len(image_files)} image(s) for TTS context. Processing...", extra={'subsys': 'router', 'event': 'process_tts_attachments.vision'})
                    from .see import see_infer
                    prompt = cleaned_input or "Describe this image."
                    vision_results = await see_infer(image_files, prompt)
                    logger.debug("[WIND] Vision result obtained, generating final conversational response.", extra={'subsys': 'router', 'event': 'process_tts_attachments.vision.infer_brain'})
                    from .brain import brain_infer
                    final_text = await brain_infer(f"Based on the following description, provide a conversational response: {vision_results}")
                elif doc_files:
                    logger.debug(f"[WIND] Found {len(doc_files)} document(s) for TTS context. Processing...", extra={'subsys': 'router', 'event': 'process_tts_attachments.doc'})
                    doc_text = await self._process_document(str(doc_files[0]), doc_files[0].suffix)
                    prompt = cleaned_input or f"Summarize this document: {doc_text}"
                    logger.debug("[WIND] Document content obtained, generating final summary/response.", extra={'subsys': 'router', 'event': 'process_tts_attachments.doc.infer_brain'})
                    from .brain import brain_infer
                    final_text = await brain_infer(prompt)
                elif cleaned_input:
                    logger.debug("[WIND] No processable attachments found, using provided text for TTS.", extra={'subsys': 'router', 'event': 'process_tts_attachments.fallback_text'})
                    final_text = cleaned_input

                if final_text:
                    logger.info("[WIND] Final text generated, proceeding to TTS synthesis.", extra={'subsys': 'router', 'event': 'process_tts_attachments.synthesize'})
                    return await self._process_tts(message, final_text)
                else:
                    logger.warning("[WIND] Could not generate any text from attachments.", extra={'subsys': 'router', 'event': 'process_tts_attachments.no_text'})
                    return ResponseMessage(text="I couldn't find anything to say from the attachments provided.")

        except Exception as e:
            logger.error(f"[WIND] Unexpected error in _process_tts_with_attachments: {e}", extra={'subsys': 'router', 'event': 'process_tts_attachments.error.unexpected'}, exc_info=True)
            return ResponseMessage(text=f"⚠️ An error occurred while processing TTS with attachments.")


# Global router instance
router: Router = None

def setup_router(bot: commands.Bot) -> Router:
    """Initialize and return the global router instance."""
    global router
    if router is None:
        router = Router(bot)
        logger.info("[WIND] Router initialized.", extra={'subsys': 'router', 'event': 'init'})
    return router

def get_router() -> Router:
    """Get the global router instance."""
    if router is None:
        raise RuntimeError("Router not initialized. Call setup_router() first.")
    return router
