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

    async def dispatch_message(self, message: Message) -> ResponseMessage:
        self.logger.debug(f"--- Dispatching message {message.id} ---", extra={'subsys': 'router'})

        try:
            parsed_command = parse_command(message, self.bot)
            if parsed_command is None or parsed_command.command is Command.IGNORE:
                self.logger.debug("Ignoring message.", extra={'subsys': 'router', 'msg_id': message.id})
                return ResponseMessage(text="")

            self.logger.debug(f"Parsed command: {parsed_command}", extra={'subsys': 'router', 'msg_id': message.id})

            # Handle simple, static commands first
            if parsed_command.command == Command.PING:
                return ResponseMessage(text="Pong!")
            if parsed_command.command == Command.HELP:
                return ResponseMessage(text="*yo what's good...*")

            # Determine modalities
            input_modality = self._get_input_modality(message)
            output_modality = self._get_output_modality(parsed_command, message)
            self.logger.debug(f"Modalities: IN={input_modality.name} -> OUT={output_modality.name}", extra={'subsys': 'router'})

            # Process input to get text
            response_text = None
            if parsed_command.command == Command.SAY:
                response_text = parsed_command.cleaned_content
            elif input_modality == InputModality.TEXT_ONLY:
                response_text = await self._flows['process_text'](parsed_command.cleaned_content, str(message.author.id))
            elif input_modality in [InputModality.IMAGE, InputModality.DOCUMENT]:
                response_text = await self._flows['process_attachments'](message, parsed_command.cleaned_content)
            
            if not response_text:
                self.logger.warning("No text generated from input processing.", extra={'subsys': 'router', 'msg_id': message.id})
                return ResponseMessage(text="I'm sorry, I wasn't able to process that. Please try again.")

            # Generate final response based on output modality
            if output_modality == OutputModality.TTS:
                audio_path = await self._flows['generate_tts'](response_text)
                # !say command is audio-only
                if parsed_command.command == Command.SAY:
                    return ResponseMessage(audio_path=audio_path)
                return ResponseMessage(text=response_text, audio_path=audio_path)
            else:
                return ResponseMessage(text=response_text)

        except Exception as e:
            self.logger.error(f"Unexpected error in dispatch_message for msg {message.id}: {e}", exc_info=True, extra={'subsys': 'router'})
            return ResponseMessage(text="I'm sorry, an unexpected error occurred. Please check the logs.")

    def _get_input_modality(self, message: Message) -> InputModality:
        """Determines the primary input modality of the message by checking attachments."""
        if not message.attachments:
            return InputModality.TEXT_ONLY

        # Define supported extensions
        image_exts = ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp']
        doc_exts = ['.pdf', '.txt', '.md', '.docx']
        audio_exts = ['.wav', '.mp3', '.ogg', '.opus', '.flac']

        # Check for each modality type in order of priority
        for attachment in message.attachments:
            file_ext = Path(attachment.filename).suffix.lower()
            if file_ext in image_exts:
                self.logger.debug(f"Detected IMAGE modality from attachment: {attachment.filename}")
                return InputModality.IMAGE

        for attachment in message.attachments:
            file_ext = Path(attachment.filename).suffix.lower()
            if file_ext in doc_exts:
                self.logger.debug(f"Detected DOCUMENT modality from attachment: {attachment.filename}")
                return InputModality.DOCUMENT

        for attachment in message.attachments:
            file_ext = Path(attachment.filename).suffix.lower()
            if file_ext in audio_exts:
                self.logger.debug(f"Detected AUDIO modality from attachment: {attachment.filename}")
                return InputModality.AUDIO

        # If no supported attachments are found, treat as text only
        self.logger.debug("Attachments present, but no supported modality detected.")
        return InputModality.TEXT_ONLY

    def _get_output_modality(self, parsed_command: Command, message: Message) -> OutputModality:
        """Determines the desired output modality based on the command."""
        if parsed_command.command in [Command.SAY, Command.TTS]:
            return OutputModality.TTS
        return OutputModality.TEXT

    async def _flow_process_text(self, text: str, author_id: str) -> str:
        """Process text using the AI brain."""
        # author_id is currently unused but kept for future context management.
        return await brain_infer(text)

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
                return await brain_infer(final_prompt)

            elif input_modality == InputModality.DOCUMENT:
                logger.debug(f"Processing document attachment: {attachment.filename}", extra={'subsys': 'router'})
                document_text = await self._process_document(tmp_file.name, file_ext)
                prompt = f"DOCUMENT CONTENT:\n---\n{document_text}\n---\n\nUSER'S PROMPT: {text}"
                return await brain_infer(prompt)

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
