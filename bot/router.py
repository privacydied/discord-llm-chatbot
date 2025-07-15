"""
Centralized router enforcing the '1 IN > 1 OUT' principle for multimodal message processing.
"""
import logging
import tempfile
from enum import Enum, auto
from pathlib import Path
from typing import Optional, TYPE_CHECKING, Dict, Callable
from discord import Message

if TYPE_CHECKING:
    from .core.bot import LLMBot

# Local application imports
from .brain import brain_infer
from .command_parser import Command, parse_command
from .pdf_utils import PDFProcessor, PDF_SUPPORT
from .see import see_infer

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
            'process_attachments': self._flow_process_attachments,
            'generate_tts': self._flow_generate_tts,
        }

        if flow_overrides:
            self._flows.update(flow_overrides)

    async def dispatch_message(self, message: Message) -> Optional[ResponseMessage]:
        """
        Central dispatcher for all incoming messages. It parses, routes, processes, and
        handles errors to guarantee the '1 IN > 1 OUT' principle.
        """
        self.logger.debug(f"--- Dispatching message {message.id} ---", extra={'subsys': 'router'})

        try:
            # 1. Parse the command from the message
            self.logger.debug(f"Parsing command for message: {message.id}", extra={'msg_id': message.id})
            parsed_command = parse_command(message, self.bot)

            # 2. Handle ignored messages or no-ops early
            if parsed_command is None:
                self.logger.debug("Ignoring message (no command parsed).", extra={'subsys': 'router'})
                return None
            if parsed_command.command is Command.IGNORE:
                self.logger.debug("Ignoring message (IGNORE command).", extra={'subsys': 'router'})
                return None
        except Exception:
            # This case is for when parsing fails entirely, e.g., a message in a guild
            # without a mention. We should silently ignore these.
            self.logger.debug("Command parsing failed, ignoring message.", extra={'msg_id': message.id})
            return None

        # 3. Handle special simple commands
        if parsed_command.command == Command.PING:
            self.logger.info("Ping command received.", extra={'subsys': 'router'})
            return ResponseMessage(text="Pong!")

        try:
            # 4. Determine input and output modalities
            input_modality = self._get_input_modality(message)
            output_modality = self._get_output_modality(parsed_command, message)
            self.logger.debug(f"Determined modalities: Input={input_modality}, Output={output_modality}", extra={'subsys': 'router'})

            # 5. Process based on input modality
            processed_text = None
            if input_modality == InputModality.TEXT_ONLY:
                processed_text = await self._flows['process_text'](parsed_command.cleaned_content, str(message.author.id))
            elif input_modality in [InputModality.IMAGE, InputModality.DOCUMENT]:
                processed_text = await self._flows['process_attachments'](message, parsed_command.cleaned_content)
            else:
                # This case handles unhandled attachment types like AUDIO
                self.logger.warning(f"Unhandled input modality: {input_modality}", extra={'subsys': 'router'})
                return ResponseMessage(text="I'm sorry, I can't process that type of attachment.")

            # 6. Handle cases where processing yields no text
            if not processed_text:
                self.logger.warning("Processing resulted in no text.", extra={'subsys': 'router'})
                return ResponseMessage(text="I'm sorry, I couldn't process that. Please try again.")

            # 7. Generate response based on output modality
            if output_modality == OutputModality.TTS:
                audio_path = await self._flows['generate_tts'](processed_text)
                # For !say command, only return audio
                text_response = None if parsed_command.command == Command.SAY else processed_text
                return ResponseMessage(text=text_response, audio_path=audio_path)
            else:
                return ResponseMessage(text=processed_text)

        except Exception as e:
            self.logger.error(f"An unexpected error occurred in dispatch_message: {e}", exc_info=True)
            return ResponseMessage(text="I'm sorry, an unexpected error occurred.")

    def _get_input_modality(self, message: Message) -> InputModality:
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

    def _get_output_modality(self, parsed_command: Command, message: Message) -> OutputModality:
        """Determines the desired output modality based on the command."""
        if parsed_command.command in [Command.SAY, Command.TTS]:
            return OutputModality.TTS
        return OutputModality.TEXT

    async def _flow_process_text(self, text: str, author_id: str) -> str:
        """Process text using the AI brain."""
        return await brain_infer(text, author_id)

    async def _flow_process_attachments(self, message: Message, text: str) -> str:
        """Process attachments using the AI vision module."""
        attachment = message.attachments[0]
        file_ext = Path(attachment.filename).suffix
        input_modality = self._get_input_modality(message)

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            await attachment.save(tmp_file.name)
            
            if input_modality == InputModality.IMAGE:
                logger.debug(f"Processing image attachment: {attachment.filename}", extra={'subsys': 'router'})
                image_bytes = await attachment.read()
                prompt = f"User uploaded an image with the prompt: '{text}'"
                vision_response = await see_infer(image_bytes, prompt)
                # Combine the vision response with the original text prompt for the brain
                final_prompt = f"User uploaded an image with the prompt: '{text}'. The image contains: {vision_response}"
                return await brain_infer(final_prompt, str(message.author.id))

            elif input_modality == InputModality.DOCUMENT:
                logger.debug(f"Processing document attachment: {attachment.filename}", extra={'subsys': 'router'})
                doc_text = await self._process_document(tmp_file.name, file_ext)
                prompt = f"User uploaded a document with the prompt: '{text}'. The document contains: {doc_text}"
                return await brain_infer(prompt, str(message.author.id))

        # Fallback for unhandled attachment types
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
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    logger.info("[WIND] Plain text file processed successfully.", extra={'subsys': 'router', 'event': 'process_document.text.success'})
                    return content
        except Exception as e:
            logger.error(f"[WIND] Error processing document {file_path}: {e}", extra={'subsys': 'router', 'event': 'process_document.error.unexpected'}, exc_info=True)
            return "⚠️ An unexpected error occurred while processing the document."


# Global router instance
router: Router = None

def setup_router(bot: "LLMBot") -> None:
    """Set up the router and store it on the bot instance."""
    global router
    if not router:
        router = Router(bot=bot)
        bot.router = router
        logger.info("[WIND] Router initialized.", extra={'subsys': 'router', 'event': 'init'})

def get_router() -> Router:
    """Get the global router instance."""
    if router is None:
        raise RuntimeError("Router not initialized. Call setup_router() first.")
    return router
