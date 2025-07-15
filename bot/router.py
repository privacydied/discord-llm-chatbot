"""
Centralized router enforcing the '1 IN > 1 OUT' principle for multimodal message processing.
"""
import logging
import tempfile
from enum import Enum, auto
from pathlib import Path
from typing import List, Optional, Tuple

import discord
from discord.ext import commands

# Local application imports
from .brain import brain_infer
from .command_parser import Command, parse_command
from .pdf_utils import PDFProcessor, PDF_SUPPORT
from .see import see_infer
from .tts_manager import tts_manager
from .tts_state import tts_state

logger = logging.getLogger(__name__)

# Try to import python-docx for .docx support
try:
    import docx
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False

class InputModality(Enum):
    """Defines the type of input received from the user."""
    TEXT_ONLY = auto()
    IMAGE = auto()
    DOCUMENT = auto()
    AUDIO = auto() # Not implemented in this refactor

class OutputModality(Enum):
    """Defines the type of output the bot should produce."""
    TEXT = auto()
    TTS = auto()

class ResponseMessage:
    """Represents a response with optional text and audio, ensuring a single response path."""
    def __init__(self, text: Optional[str] = None, audio_path: Optional[str] = None):
        self.text = text
        self.audio_path = audio_path

class Router:
    """Handles routing of messages to the correct processing flow."""

    def __init__(self, bot: commands.Bot, config: dict, tts_manager: "TTSManager"):
        self.bot = bot
        self.config = config
        self.tts_manager = tts_manager
        self.pdf_processor = PDFProcessor() if PDF_SUPPORT else None
        if self.pdf_processor:
            self.pdf_processor.loop = bot.loop
        logger.info("✔ Router initialized.")

    async def dispatch_message(self, message: discord.Message) -> Optional[ResponseMessage]:
        """
        Central dispatcher for all incoming messages. It parses, routes, processes, and 
        handles errors to guarantee the '1 IN > 1 OUT' principle.
        """
        try:
            # 1. Parse Command and Determine Context (DM vs. Guild)
            parsed_command = parse_command(message, self.bot)
            if not parsed_command:
                logger.debug(f"Ignoring message: not a command. (Guild: {message.guild is not None})")
                return None

            logger.debug(f"ℹ Command parsed: {parsed_command.command.name} with content='{parsed_command.cleaned_content[:50]}...' [Guild: {message.guild is not None}]", extra={'subsys': 'router'})

            # 2. Determine Input and Output Modalities
            input_modality = self._get_input_modality(message)
            output_modality = self._get_output_modality(parsed_command, message)
            logger.debug(f"ℹ Flow selected: {input_modality.name} -> {output_modality.name}", extra={'subsys': 'router'})

            # 3. Execute the appropriate processing flow
            if input_modality == InputModality.TEXT_ONLY:
                processed_text = await self._flow_process_text(parsed_command.cleaned_content, str(message.author.id))
            else: # IMAGE or DOCUMENT
                processed_text = await self._flow_process_attachments(message, parsed_command.cleaned_content)

            # 4. Generate the final response based on output modality
            if output_modality == OutputModality.TEXT:
                return ResponseMessage(text=processed_text)
            else: # TTS
                audio_path = await self._flow_generate_tts(processed_text)
                # For !say command, we don't send the original text back
                text_response = None if parsed_command.command == Command.SAY else processed_text
                return ResponseMessage(text=text_response, audio_path=audio_path)

            audio_path = await self.tts_manager.generate_tts(text, self.tts_manager.voice)
            logger.debug(f"[WIND] TTS audio generated at {audio_path}", extra={'subsys': 'router', 'event': 'process_tts.generate.response'})

            # 3. Send audio file
            logger.debug(f"[WIND] Sending TTS audio file to channel {message.channel.id}", extra={'subsys': 'router', 'event': 'process_tts.send.request'})
            logger.debug("[WIND] TTS audio generated, returning response message.", extra={'subsys': 'router', 'event': 'process_tts.return_audio'})
            return ResponseMessage(audio_path=audio_path)
        except Exception as e:
            logger.error(f"[WIND] Error during dispatch: {e}", extra={'subsys': 'router', 'event': 'dispatch.error'}, exc_info=True)
            return ResponseMessage(text=f"⚠️ An error occurred: {e}")

    def _get_input_modality(self, message: discord.Message) -> InputModality:
        """Determines the primary input modality of the message."""
        if message.attachments:
            attachment = message.attachments[0]
            content_type = attachment.content_type
            if content_type and content_type.startswith('image/'):
                return InputModality.IMAGE
            # Add more specific document checks if needed
            if any(attachment.filename.lower().endswith(ext) for ext in ['.pdf', '.txt', '.md', '.docx']):
                return InputModality.DOCUMENT
        return InputModality.TEXT_ONLY

    def _get_output_modality(self, parsed_command: Command, message: discord.Message) -> OutputModality:
        """Determines the desired output modality based on the command."""
        if parsed_command.command in [Command.SPEAK, Command.SAY, Command.TTS, Command.TTS_ALL]:
            return OutputModality.TTS
        return OutputModality.TEXT

    async def _flow_process_text(self, text: str, author_id: str) -> str:
        """Processes a text-only command through the AI brain."""
        return await self.brain_infer(text, author_id)

    async def _flow_process_attachments(self, message: discord.Message, text_content: str) -> str:
        """Downloads and processes message attachments."""
        attachment = message.attachments[0]
        file_ext = Path(attachment.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            await attachment.save(tmp_file.name)
            if self._get_input_modality(message) == InputModality.IMAGE:
                vision_text = await self.see_infer(tmp_file.name)
                prompt = f"User uploaded an image with the prompt: '{text_content}'. The image contains: {vision_text}"
                return await self.brain_infer(prompt, str(message.author.id))
            else: # DOCUMENT
                doc_text = await self._process_document(tmp_file.name, file_ext)
                prompt = f"User uploaded a document with the prompt: '{text_content}'. The document contains: {doc_text}"
                return await self.brain_infer(prompt, str(message.author.id))

    async def _flow_generate_tts(self, text: str) -> str:
        """Generates TTS audio for the given text."""
        return await self.tts_manager.generate_tts(text, self.tts_manager.voice)

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
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    logger.info(f"[WIND] Plain text file processed successfully.", extra={'subsys': 'router', 'event': 'process_document.text.success'})
                    return content
        except Exception as e:
            logger.error(f"[WIND] Error processing document {file_path}: {e}", extra={'subsys': 'router', 'event': 'process_document.error.unexpected'}, exc_info=True)
            return f"⚠️ An unexpected error occurred while processing the document."


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
