"""
Centralized router enforcing the '1 IN > 1 OUT' principle for multimodal message processing.
"""
from __future__ import annotations

import asyncio
import io
from .util.logging import get_logger
import os
import re
import tempfile
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, Optional, TYPE_CHECKING, List

from .context import get_conversation_history
from . import web
from discord import Message, DMChannel, Embed, File

if TYPE_CHECKING:
    from bot.core.bot import LLMBot as DiscordBot
    from bot.metrics import Metrics
    from .command_parser import ParsedCommand

logger = get_logger(__name__)

# Local application imports
from .action import BotAction, ResponseMessage
from .brain import brain_infer
from .command_parser import Command, parse_command
from .exceptions import DispatchEmptyError, DispatchTypeError
from .hear import hear_infer
from .pdf_utils import PDFProcessor
from .see import see_infer
from .web import process_url

# Dependency availability flags
try:
    import docx  # noqa: F401
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False

try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

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


class Router:
    """Handles routing of messages to the correct processing flow."""

    def __init__(self, bot: "DiscordBot", flow_overrides: Optional[Dict[str, Callable]] = None, logger: Optional[logging.Logger] = None):
        self.bot = bot
        self.config = bot.config
        self.tts_manager = bot.tts_manager
        self.logger = logger or get_logger(f"discord-bot.{self.__class__.__name__}")

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

    async def dispatch_message(self, message: Message) -> Optional[BotAction]:
        """Process a message and ensure exactly one response is generated (1 IN > 1 OUT rule)."""
        self.logger.info(f"🔄 === ROUTER DISPATCH STARTED: MSG {message.id} ====")

        try:
            # 1. Parse for a command from the message content.
            parsed_command = parse_command(message, self.bot)

            # 2. If a command is found, delegate it to the command processor (cogs).
            if parsed_command:
                self.logger.info(f"Found command '{parsed_command.command.name}', delegating to cog. (msg_id: {message.id})")
                return BotAction(meta={'delegated_to_cog': True})

            # 3. Determine if the bot should process this message (DM, mention, or reply).
            is_dm = isinstance(message.channel, DMChannel)
            is_mentioned = self.bot.user in message.mentions
            is_reply = message.reference is not None
            
            if not (is_dm or is_mentioned or is_reply):
                self.logger.info(f"Ignoring message: not a command, DM, mention, or reply. (msg_id: {message.id})")
                return None

            # --- Start of processing for DMs, Mentions, and Replies ---
            self.logger.info(f"Processing message: DM={is_dm}, Mention={is_mentioned}, Reply={is_reply} (msg_id: {message.id})")

            # 4. Gather conversation history for context
            history = []
            try:
                history = get_conversation_history(message)
                self.logger.info(f"📚 Gathered {len(history)} messages for context. (msg_id: {message.id})")
            except Exception as e:
                self.logger.warning(f"Could not fetch message history: {e} (msg_id: {message.id})")

            # 5. Determine modalities and process
            input_modality = self._get_input_modality(message)
            self._metric_inc('router_input_modality', {'modality': input_modality.name.lower()})
            output_modality = self._get_output_modality(None, message)
            self._metric_inc('router_output_modality', {'modality': output_modality.name.lower()})

            action = None
            text_content = message.content
            if is_mentioned:
                text_content = re.sub(r'^(<@!?&?{}>)'.format(self.bot.user.id), '', text_content).strip()

            # Pass both history and current message to the appropriate flow
            if input_modality == InputModality.TEXT_ONLY:
                action = await self._invoke_text_flow(text_content, message, history)
            elif input_modality == InputModality.URL:
                url_match = re.search(r'https?://[\S]+', text_content)
                if url_match:
                    action = await self._flows['process_url'](url_match.group(0), message)
            elif input_modality == InputModality.IMAGE:
                action = await self._flows['process_attachments'](message, message.attachments[0])
            elif input_modality == InputModality.DOCUMENT:
                action = await self._flows['process_attachments'](message, message.attachments[0])
            
            # 6. If we have a response, determine output modality and finalize the action.
            if action:
                action.meta['is_reply'] = is_reply

                # Generate TTS if needed
                output_modality = self._get_output_modality(parsed_command, message)
                if output_modality == OutputModality.TTS and action.content and not action.files:
                    action.audio_path = await self._generate_tts_safe(action.content)
                
                return action

            self.logger.info(f"🏁 === ROUTER DISPATCH END: No Action Taken ====")
            return None

        except Exception as e:
            self.logger.error(f"❌ Error in router dispatch: {e} (msg_id: {message.id})", exc_info=True)
            return BotAction(content="I encountered an error while processing your message.", error=True)

    def _get_input_modality(self, message: Message) -> InputModality:
        """Determine the input modality of a message."""
        if message.attachments:
            attachment = message.attachments[0]
            content_type = attachment.content_type
            filename = attachment.filename.lower()
            if content_type and 'image' in content_type:
                return InputModality.IMAGE
            if filename.endswith(('.pdf', '.docx')):
                return InputModality.DOCUMENT
            if content_type and 'audio' in content_type:
                return InputModality.AUDIO

        if re.search(r'https?://[\S]+', message.content):
            return InputModality.URL
            
        return InputModality.TEXT_ONLY

    def _get_output_modality(self, parsed_command: Optional[ParsedCommand], message: Message) -> OutputModality:
        """Determine the output modality based on command or channel settings."""
        # Future: check for TTS commands or channel/user settings
        return OutputModality.TEXT

    async def _invoke_text_flow(self, content: str, message: Message, history: List[Dict[str, Any]] = None) -> Optional[BotAction]:
        """Invoke the text processing flow, formatting history into a context string."""
        self.logger.info(f"Routing to text flow. (msg_id: {message.id})")
        try:
            context_str = ""
            if history:
                formatted_history = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in history]
                context_str = "\n".join(formatted_history)
                self.logger.info(f"Providing {len(history)} messages as context. (msg_id: {message.id})")

            action = await self._flows['process_text'](content, context_str)
            if action and action.has_payload:
                return action
            else:
                self.logger.warning("Text flow returned no response. (msg_id: {message.id})")
                return None
        except Exception as e:
            self.logger.error(f"Text processing flow failed: {e} (msg_id: {message.id})", exc_info=True)
            return BotAction(content="I had trouble processing that text.", error=True)

    async def _flow_process_text(self, content: str, context: str = "") -> BotAction:
        """Process text input through the AI model, including conversation context."""
        self.logger.info("Processing text with AI model.")
        return await brain_infer(content, context=context)

    async def _flow_process_url(self, url: str, message: discord.Message) -> BotAction:
        """
        Processes a URL by fetching its content, summarizing it with an LLM,
        and returning a BotAction with the summary and an optional file attachment.
        """
        self.logger.info(f"Processing URL: {url} (msg_id: {message.id})")

        try:
            content_data = await web.process_url(url)
            if not content_data or content_data.get('error'):
                error_msg = content_data.get('error', 'Unknown error')
                self.logger.error(f"Failed to fetch or process URL: {error_msg} (msg_id: {message.id})")
                return BotAction(content=f"I'm sorry, I was unable to process that page. ({error_msg})", error=True)

            extracted_text = content_data.get('content', {}).get('text')
            if not extracted_text:
                self.logger.warning(f"No readable content found at URL. (msg_id: {message.id})")
                return BotAction(content="I couldn't find any readable content on that page.", error=True)

            self.logger.info(f"Extracted {len(extracted_text)} characters from URL. (msg_id: {message.id})")

            max_chars = 8000
            is_truncated = len(extracted_text) > max_chars
            truncated_text = extracted_text[:max_chars]

            system_prompt = (
                "A user linked the following webpage. Please provide a concise summary of its contents, "
                "explain any interesting or notable points, and if applicable, add a short critical commentary. "
                "Your goal is to be a helpful, intelligent assistant. "
                "IMPORTANT: Your entire response must be under 1800 characters."
            )

            self.logger.info(f"Summarizing content via LLM... (msg_id: {message.id})")

            summary_action = await brain_infer(prompt=truncated_text, system_prompt=system_prompt)

            if summary_action.error:
                self.logger.error(f"LLM summarization failed. (msg_id: {message.id})")
                return summary_action

            self.logger.info(f"Summarization complete. (msg_id: {message.id})")

            if len(extracted_text) > 4000 or is_truncated:
                file_content = io.BytesIO(extracted_text.encode('utf-8'))
                summary_action.files.append(File(file_content, filename="original_content.txt"))
                self.logger.info(f"Attaching full content ({len(extracted_text)} chars) as file. (msg_id: {message.id})")

            self.logger.info(f"Replying with summary (and optional attachment). (msg_id: {message.id})")

            return summary_action

        except Exception as e:
            self.logger.error(f"An unexpected error occurred during URL processing: {e} (msg_id: {message.id})", exc_info=True)
            return BotAction(content="A critical error occurred while I was handling that URL.", error=True)

    async def _flow_process_audio(self, message: Message) -> BotAction:
        """Process audio attachment through STT model."""
        self.logger.info(f"Processing audio attachment. (msg_id: {message.id})")
        return await hear_infer(message)

    async def _flow_process_attachments(self, message: Message, attachment) -> BotAction:
        """Process image/document attachments."""
        self.logger.info(f"Processing attachment: {attachment.filename} (msg_id: {message.id})")

        content_type = attachment.content_type
        filename = attachment.filename.lower()

        # Process image attachments
        if content_type and content_type.startswith("image/"):
            return await self._process_image_attachment(message, attachment)

        # Process document attachments
        elif filename.endswith('.pdf') and self.pdf_processor:
            return await self._process_pdf_attachment(message, attachment)

        else:
            self.logger.warning(f"Unsupported attachment type: {filename} (msg_id: {message.id})")
            return BotAction(content="I can't process that type of file attachment.")

    async def _process_image_attachment(self, message: Message, attachment) -> BotAction:
        self.logger.info(f"Processing image attachment: {attachment.filename} (msg_id: {message.id})")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(attachment.filename)[1] or '.jpg') as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            await attachment.save(tmp_path)
            self.logger.debug(f"Saved image to temp file: {tmp_path} (msg_id: {message.id})")

            prompt = message.content.strip() or "Describe this image."
            vision_response = await see_infer(image_path=tmp_path, prompt=prompt)

            if not vision_response or vision_response.error:
                self.logger.warning(f"Vision model returned no/error response (msg_id: {message.id})")
                return BotAction(content="I couldn't understand the image.", error=True)

            final_prompt = f"User uploaded an image with the prompt: '{prompt}'. The image contains: {vision_response.content}"
            return await brain_infer(final_prompt)

        except Exception as e:
            self.logger.error(f"❌ Image processing failed: {e} (msg_id: {message.id})", exc_info=True)
            return BotAction(content="⚠️ An error occurred while processing this image.", error=True)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    async def _process_pdf_attachment(self, message: Message, attachment) -> BotAction:
        self.logger.info(f"📄 Processing PDF attachment: {attachment.filename} (msg_id: {message.id})")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_path = tmp_file.name
        try:
            await attachment.save(tmp_path)
            text_content = await self.pdf_processor.process(tmp_path)
            if not text_content:
                return BotAction(content="I couldn't extract any text from that PDF.")
            
            final_prompt = f"User uploaded a PDF document. Here is the text content:\n\n{text_content}"
            return await brain_infer(final_prompt)
        except Exception as e:
            self.logger.error(f"❌ PDF processing failed: {e} (msg_id: {message.id})", exc_info=True)
            return BotAction(content="⚠️ An error occurred while processing this PDF.", error=True)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    async def _flow_generate_tts(self, text: str) -> Optional[str]:
        """Generate TTS audio from text."""
        self.logger.info(f"🔊 Generating TTS for text of length: {len(text)}")
        # This would integrate with a TTS service
        return None

    async def _generate_tts_safe(self, text: str) -> Optional[str]:
        """Safely generate TTS, handling any exceptions."""
        try:
            return await self._flows['generate_tts'](text)
        except Exception as e:
            self.logger.error(f"TTS generation failed: {e}", exc_info=True)
            return None

    def _metric_inc(self, metric_name: str, labels: Optional[Dict[str, str]] = None):
        """Increment a metric, if metrics are enabled."""
        if hasattr(self.bot, 'metrics') and self.bot.metrics:
            try:
                self.bot.metrics.increment(metric_name, labels or {})
            except Exception as e:
                self.logger.warning(f"Failed to increment metric {metric_name}: {e}")

# Backward compatibility
MessageRouter = Router

# Global router instance
_router_instance = None

def setup_router(bot: "DiscordBot") -> Router:
    """Factory to create and initialize the router."""
    global _router_instance
    if _router_instance is None:
        _router_instance = Router(bot)
    return _router_instance

def get_router() -> Router:
    """Get the singleton router instance."""
    if _router_instance is None:
        raise RuntimeError("Router has not been initialized. Call setup_router first.")
    return _router_instance