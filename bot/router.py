"""
Centralized router enforcing the '1 IN > 1 OUT' principle for multimodal message processing.
"""
import asyncio
import logging
import os
import re
import tempfile
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, Optional, TYPE_CHECKING
from discord import Message

if TYPE_CHECKING:
    from .core.bot import LLMBot

# Local application imports
from .brain import brain_infer
from .command_parser import Command, parse_command
from .hear import hear_infer
from .pdf_utils import PDFProcessor, PDF_SUPPORT
from .see import see_infer
from .web import process_url

logger = logging.getLogger(__name__)

# Try to import python-docx for .docx support
try:
    import docx  # noqa: F401
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False

class InputModality(Enum):
    """Defines the type of input received from the user."""
    TEXT_ONLY = auto()
    URL = auto()
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

    def __init__(self, bot: "LLMBot", flow_overrides: Optional[Dict[str, Callable]] = None, logger: Optional[logging.Logger] = None):
        self.bot = bot
        self.config = bot.config
        self.tts_manager = bot.tts_manager
        self.logger = logger or logging.getLogger(f"discord-bot.{self.__class__.__name__}")

        # Bind flow methods to the instance, allowing for test overrides
        self._bind_flow_methods(flow_overrides)

        self.pdf_processor = PDFProcessor() if PDF_SUPPORT else None
        if self.pdf_processor:
            self.pdf_processor.loop = bot.loop

        self.logger.info("✔ Router initialized.")

    def _bind_flow_methods(self, flow_overrides: Optional[Dict[str, Callable]] = None):
        """Binds flow methods to the instance, allowing for overrides for testing."""
        self._flows = {
            'process_text': self._flow_process_text,
            'process_url': self._flow_process_url,
            'process_audio': self._flow_process_audio,
            'process_attachments': self._flow_process_attachments,
            'generate_tts': self._flow_generate_tts,
        }

        if flow_overrides:
            self._flows.update(flow_overrides)

    async def dispatch_message(self, message: Message) -> Optional[ResponseMessage]:
        self.logger.debug(f"--- Dispatching message {message.id} ---", extra={'subsys': 'router'})
        try:
            # 1. Gatekeeping: Parse the command and ignore irrelevant messages.
            parsed_command = parse_command(message, self.bot)
            if parsed_command is None or parsed_command.command in [Command.IGNORE, Command.TTS, Command.SAY, Command.TTS_ALL, Command.SPEAK]:
                self.logger.debug("Ignoring message as it's not a valid or relevant command or is handled by a cog.", extra={'subsys': 'router'})
                return None

            # 2. Handle simple, static commands directly.
            if parsed_command.command == Command.PING:
                return ResponseMessage(text="Pong!")
            if parsed_command.command == Command.HELP:
                return ResponseMessage(text="See `/help` for a list of commands.")

            # 3. Determine modalities and process content
            input_modality = self._get_input_modality(message)
            output_modality = self._get_output_modality(parsed_command, message)
            self.logger.debug(f"MODALITIES: IN={input_modality.name} -> OUT={output_modality.name}", extra={'subsys': 'router'})

            processed_text = None
            content = parsed_command.cleaned_content

            if input_modality == InputModality.TEXT_ONLY:
                processed_text = await self._flows['process_text'](content)
            elif input_modality == InputModality.URL:
                url_match = re.search(r'https?://[\S]+', message.content)
                if url_match:
                    processed_text = await self._flows['process_url'](url_match.group(0))
            elif input_modality == InputModality.AUDIO:
                processed_text = await self._flows['process_audio'](message)
            elif input_modality in [InputModality.IMAGE, InputModality.DOCUMENT]:
                processed_text = await self._flows['process_attachments'](message, content)

            # 4. Handle cases where processing fails or yields no text.
            if not processed_text:
                self.logger.warning("Message processing resulted in no text.", extra={'subsys': 'router'})
                return ResponseMessage(text="I'm sorry, I wasn't able to process that.")

            # 5. Generate final response based on output modality.
            if output_modality == OutputModality.TTS:
                audio_path = await self._flows['generate_tts'](processed_text)
                return ResponseMessage(text=processed_text, audio_path=audio_path)
            
            return ResponseMessage(text=processed_text)

        except Exception as e:
            self.logger.error(f"Error dispatching message {message.id}: {e}", exc_info=True, extra={'subsys': 'router'})
            return ResponseMessage(text="An unexpected error occurred. Please try again later.")

    def _get_input_modality(self, message: Message) -> InputModality:
        """Determines the primary input modality of the message."""
        if message.attachments:
            attachment = message.attachments[0]
            content_type = getattr(attachment, 'content_type', '') or ''
            file_ext = Path(attachment.filename).suffix.lower()

            if 'audio' in content_type:
                return InputModality.AUDIO
            if 'image' in content_type:
                return InputModality.IMAGE
            if file_ext in ['.pdf', '.txt', '.md', '.docx']:
                return InputModality.DOCUMENT

        if re.search(r'https?://[\S]+', message.content):
            return InputModality.URL

        return InputModality.TEXT_ONLY

    def _get_output_modality(self, parsed_command: Command, message: Message) -> OutputModality:
        """Determines the desired output modality based on the command."""
        if parsed_command.command in [Command.SAY, Command.TTS]:
            return OutputModality.TTS
        return OutputModality.TEXT

    async def _flow_process_text(self, text: str) -> str:
        """Process text using the AI brain."""
        return await brain_infer(text)

    async def _flow_process_url(self, url: str) -> str:
        """Process a URL, extract content, and summarize it."""
        self.logger.debug(f"Processing URL: {url}", extra={'subsys': 'router'})
        try:
            web_data = await process_url(url, extract_content=True)
            if not web_data or 'error' in web_data:
                error_msg = web_data.get('error', 'Unknown error') if web_data else 'No data returned'
                self.logger.error(f"Failed to process URL {url}: {error_msg}", extra={'subsys': 'router'})
                return f"I'm sorry, I couldn't fetch the content from that URL."

            content = web_data.get('content', {})
            text_content = content.get('text', '')
            title = web_data.get('metadata', {}).get('title', 'the page')

            if not text_content.strip():
                self.logger.warning(f"No text content extracted from {url}", extra={'subsys': 'router'})
                return f"I was able to access the page '{title}', but couldn't extract any readable content."

            # Create a prompt for summarization
            prompt = (
                f"The user shared a link to the page titled '{title}'. "
                f"Please provide a concise summary of the following content:\n\n---\n{text_content[:4000]}...\n---"
            )

            # Get summary from the brain
            summary = await brain_infer(prompt)
            return summary

        except Exception as e:
            self.logger.error(f"An unexpected error occurred while processing URL {url}: {e}", exc_info=True, extra={'subsys': 'router'})
            return "An unexpected error occurred while I was analyzing the URL."

    async def _flow_process_audio(self, message: Message) -> str:
        """Process an audio attachment by transcribing it and then passing it to the brain."""
        attachment = message.attachments[0]
        self.logger.debug(f"Processing audio attachment: {attachment.filename}", extra={'subsys': 'router'})
        try:
            transcribed_text = await hear_infer(message.attachments[0])
            if not transcribed_text:
                return "I couldn't understand the audio. Please try again."

            # Now, treat the transcribed text as a new text-only query
            return await self._flow_process_text(transcribed_text)

        except Exception as e:
            self.logger.error(f"Error processing audio attachment: {e}", exc_info=True, extra={'subsys': 'router'})
            return "I'm sorry, I had trouble processing that audio file."
        finally:
            try:
                os.remove(tmp_audio_file.name)
            except (OSError, NameError):
                pass

    async def _flow_process_attachments(self, message: Message, text: str) -> str:
        """Process attachments using the AI vision module."""
        attachment = message.attachments[0]
        input_modality = self._get_input_modality(message)

        if attachment.content_type and attachment.content_type.startswith("image/"):
            self.logger.debug(f"Processing image attachment: {attachment.filename}")
            image_bytes = await attachment.read()
            prompt = f"User uploaded an image with the prompt: '{text}'"
            vision_response = await see_infer(image_data=image_bytes, prompt=prompt, mime_type=attachment.content_type)
            if not vision_response:
                return "I'm sorry, I couldn't understand the image."
            final_prompt = f"User uploaded an image with the prompt: '{text}'. The image contains: {vision_response}"
            return await brain_infer(final_prompt)

        elif input_modality == InputModality.DOCUMENT:
            logger.debug(f"Processing document attachment: {attachment.filename}", extra={'subsys': 'router'})
            file_ext = Path(attachment.filename).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                await attachment.save(Path(tmp_file.name))
                document_text = await self._process_document(tmp_file.name, file_ext)
            
            try:
                os.remove(tmp_file.name)
            except OSError as e:
                logger.error(f"Error removing temporary file {tmp_file.name}: {e}", extra={'subsys': 'router'})

            prompt = f"DOCUMENT CONTENT:\n---\n{document_text}\n---\n\nUSER'S PROMPT: {text}"
            return await brain_infer(prompt)

        # Fallback for unhandled attachment types
        logger.warning(f"Unhandled attachment type for file: {attachment.filename}", extra={'subsys': 'router'})
        return "I'm sorry, I can't process that type of attachment."

    async def _flow_generate_tts(self, text: str) -> str:
        """Generate TTS audio from text."""
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
                doc = docx.Document(file_path)
                full_text = [para.text for para in doc.paragraphs]
                return "\n".join(full_text)

            else:
                logger.warning(f"Unhandled document extension: {file_ext}", extra={'subsys': 'router', 'event': 'process_document.unhandled_ext'})
                return f"Unsupported document type: {file_ext}"
        except Exception as e:
            logger.error(f"[WIND] Error processing document {file_path}: {e}", extra={'subsys': 'router', 'event': 'process_document.error.unexpected'}, exc_info=True)
            return "⚠️ An unexpected error occurred while processing the document."


# Global router instance
router: Router = None

def setup_router(bot: "commands.Bot") -> "Router":
    global router
    router = Router(bot=bot)
    bot.router = router
    logger.info("[WIND] Router initialized.", extra={'subsys': 'router', 'event': 'init'})
    return router

def get_router() -> "Router":
    """Returns the global router instance. Note: This is only safe to call after the bot's setup_hook has completed."""
    if router is None:
        raise RuntimeError("Router not initialized. Call setup_router() first.")
    return router
