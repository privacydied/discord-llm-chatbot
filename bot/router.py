"""
Centralized router enforcing the '1 IN > 1 OUT' principle for multimodal message processing.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, Optional, TYPE_CHECKING, List

from discord import Message, DMChannel, Embed, File

if TYPE_CHECKING:
    from bot.core.bot import LLMBot as DiscordBot
    from bot.metrics import Metrics
    from .command_parser import ParsedCommand

logger = logging.getLogger(__name__)

# Local application imports
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

@dataclass
class BotAction:
    content: str = ""
    embeds: List[Embed] = field(default_factory=list)
    files: List[File] = field(default_factory=list)
    audio_path: Optional[str] = None
    error: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_payload(self):
        return bool(self.content or self.embeds or self.files or self.audio_path)

    # Backward compatibility:
    @property
    def text(self): return self.content
    @text.setter
    def text(self, v): self.content = v
    @property
    def embed(self): return self.embeds[0] if self.embeds else None
    @property
    def file(self): return self.files[0] if self.files else None

# Transitional alias for compatibility
ResponseMessage = BotAction

class Router:
    """Handles routing of messages to the correct processing flow."""

    def __init__(self, bot: "DiscordBot", flow_overrides: Optional[Dict[str, Callable]] = None, logger: Optional[logging.Logger] = None):
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

    async def dispatch_message(self, message: Message) -> Optional[BotAction]:
        """Process a message and ensure exactly one response is generated (1 IN > 1 OUT rule).

        This method enforces the strict 1 IN > 1 OUT principle, ensuring that each processed message
        will either be explicitly ignored (return None) or will generate exactly one response.

        Args:
            message: The Discord message to process

        Returns:
            BotAction: A single response with text and/or audio, or None if message is ignored
        """
        self.logger.debug(f"🛣️ [ROUTER-ENTER] MsgID={message.id}")
        msg_id = message.id
        user_id = str(message.author.id) if message.author else 'unknown'
        guild_id = str(message.guild.id) if message.guild else 'dm'

        self.logger.info(f"🔄 === ROUTER DISPATCH STARTED: MSG {msg_id} ====",
                       extra={'subsys': 'router', 'event': 'dispatch.start',
                              'msg_id': msg_id, 'user_id': user_id, 'guild_id': guild_id})

        # This flag will be used by the final block to decide if the message
        # should be passed to the bot's default processor (for cogs).
        should_process = False

        try:
            # 1. Gatekeeping: First check channel type (DM vs Guild)
            if isinstance(message.channel, DMChannel):
                self.logger.info(f"📩 === DM MESSAGE PROCESSING STARTED ====",
                               extra={'subsys': 'router', 'event': 'dm.start',
                                      'msg_id': msg_id, 'user_id': user_id})

                # 1a. Attachment Check: Prioritize attachments over commands or text.
                if message.attachments:
                    self.logger.info(f"📎 DM has attachments, starting attachment processing flow",
                                   extra={'subsys': 'router', 'event': 'dm.attachment', 'msg_id': msg_id})
                    self._metric_inc('messages_dm_attachment_handled')
                    # We assume the first attachment is the one to process
                    text_response = await self._flow_process_attachments(message, message.attachments[0])
                    return BotAction(content=text_response)

                # 1b. Command Check: If no attachments, check for commands.
                parsed_command = parse_command(message, self.bot)
                if parsed_command:
                    self.logger.debug(f"🔍 DM command detected: {parsed_command.command.name}",
                                    extra={'subsys': 'router', 'event': 'dm.command',
                                           'msg_id': msg_id, 'command': parsed_command.command.name})

                    if parsed_command.command in [Command.PING, Command.HELP]:
                        return await self._handle_simple_command(parsed_command, msg_id)
                    elif parsed_command.command == Command.IGNORE:
                        return None  # Explicitly ignore
                    elif parsed_command.command == Command.CHAT:
                        self.logger.info(f"📝 Processing DM as plain text message",
                                       extra={'subsys': 'router', 'event': 'dm.text', 'msg_id': msg_id})
                        self._metric_inc('messages_dm_text_handled')
                        return await self._invoke_text_flow(parsed_command.cleaned_content)
                    else:
                        # Other commands fall through to be handled by cogs
                        self.logger.info(f"⚙️ Passing DM command to cog: {parsed_command.command.name}",
                                       extra={'subsys': 'router', 'event': 'dm.command.pass', 'msg_id': msg_id})
                        should_process = True  # Let the default cog handler run
                else:
                    # 1c. Fallback: If no attachments and no command, it's an empty or invalid message.
                    self.logger.debug("No command or attachment detected in DM, ignoring.", extra={'subsys': 'router', 'event': 'dm.ignore.empty', 'msg_id': msg_id})
                    return None # Ignore empty/invalid messages

            else:  # It's a Guild Channel
                self.logger.debug(f"🏢 === GUILD MESSAGE PROCESSING STARTED ==== ",
                               extra={'subsys': 'router', 'event': 'guild.start',
                                      'msg_id': msg_id, 'guild_id': guild_id})
                parsed_command = parse_command(message, self.bot)

                if not parsed_command:
                    self.logger.debug(f"⏭️ Ignoring guild message {msg_id}: No mention or prefix",
                                    extra={'subsys': 'router', 'event': 'guild.ignore.unmentioned'})
                    return None # Not a command, and bot was not mentioned.

                # At this point, it's a command or a mention.
                # 2a. Attachment Check
                if message.attachments:
                    self.logger.info(f"📎 Guild message has attachments, starting attachment processing flow",
                                   extra={'subsys': 'router', 'event': 'guild.attachment', 'msg_id': msg_id})
                    self._metric_inc('messages_guild_attachment_handled')
                    text_response = await self._flow_process_attachments(message, message.attachments[0])
                    return BotAction(content=text_response)
                
                # 2b. Command/Text processing
                if parsed_command.command in [Command.PING, Command.HELP]:
                    return await self._handle_simple_command(parsed_command, msg_id)
                elif parsed_command.command == Command.IGNORE:
                    return None
                elif parsed_command.command == Command.CHAT:
                    self.logger.info(f"📝 Processing guild message as text",
                                   extra={'subsys': 'router', 'event': 'guild.text', 'msg_id': msg_id})
                    self._metric_inc('messages_guild_text_handled')
                    return await self._invoke_text_flow(parsed_command.cleaned_content)
                else:
                    self.logger.info(f"⚙️ Passing guild command to cog: {parsed_command.command.name}",
                                   extra={'subsys': 'router', 'event': 'guild.command.pass', 'msg_id': msg_id})
                    should_process = True # Let the default cog handler run

        except Exception as e:
            self.logger.error(f"❌ Error in router dispatch: {e}", exc_info=True,
                            extra={'subsys': 'router', 'event': 'dispatch.error', 'msg_id': msg_id})
            return BotAction(content="I encountered an error while processing your message.", error=True)

        # If we get here, it means the message should be handled by a cog
        # This is a safety net for commands not handled by the router directly
        if should_process:
            self.logger.info(f"GATE: decision=COMMAND, reason=handled_by_cog")
            self.logger.debug(f"⏭️ Passing message {msg_id} to default processor for cog handling",
                            extra={'subsys': 'router', 'event': 'command.cog_handled',
                                   'msg_id': msg_id})
        else:
            # This should not happen with the current logic, but as a final safety net
            self.logger.warning(f"⚠️ Message {msg_id} reached end of dispatch without action or pass-through",
                              extra={'subsys': 'router', 'event': 'dispatch.no_action', 'msg_id': msg_id})
            return None

        # This return is for the case where should_process is True
        # The message will be passed to the bot's default processor
        return None

    async def _handle_simple_command(self, parsed_command: ParsedCommand, msg_id: int) -> BotAction:
        """Handle simple, static commands like PING and HELP."""
        if parsed_command.command == Command.PING:
            self.logger.info(f"✅ Handling PING command",
                           extra={'subsys': 'router', 'event': 'command.ping', 'msg_id': msg_id})
            return BotAction(content="Pong!")
        elif parsed_command.command == Command.HELP:
            self.logger.info(f"✅ Handling HELP command",
                           extra={'subsys': 'router', 'event': 'command.help', 'msg_id': msg_id})
            return BotAction(content="See `/help` for a list of commands.")
        else:
            # This should not happen, but handle gracefully
            self.logger.warning(f"⚠️ Unexpected simple command: {parsed_command.command}",
                              extra={'subsys': 'router', 'event': 'command.simple.unexpected', 'msg_id': msg_id})
            return BotAction(content="I'm not sure how to handle that command.")

    def _get_input_modality(self, message: Message) -> InputModality:
        """Determine the input modality of a message."""
        # Check for attachments first
        if message.attachments:
            attachment = message.attachments[0]  # Process first attachment for now
            if attachment.content_type:
                if 'image' in attachment.content_type:
                    return InputModality.IMAGE
                elif attachment.filename.lower().endswith('.pdf'):
                    return InputModality.DOCUMENT
                # Add other document types as needed
            # If we can't determine from content_type, check extension
            if attachment.filename.lower().endswith(('.pdf', '.docx')):
                return InputModality.DOCUMENT
                
        # Check for URLs in content
        if re.search(r'https?://[\S]+', message.content):
            return InputModality.URL
            
        # Default to text
        return InputModality.TEXT_ONLY

    def _get_output_modality(self, parsed_command: Optional[ParsedCommand], message: Message) -> OutputModality:
        """Determine the output modality based on command or channel settings."""
        # For now, default to text output
        # In the future, this could check for TTS commands or channel settings
        return OutputModality.TEXT

    async def _invoke_text_flow(self, content: str) -> BotAction:
        """Invoke the text processing flow."""
        self.logger.debug(f"🔍 Invoking text flow with content length: {len(content)}",
                        extra={'subsys': 'router', 'event': 'text.invoke', 'content_length': len(content)})
        
        try:
            # Use the brain_infer function for text processing
            response_text = await brain_infer(content)
            
            if not response_text:
                self.logger.warning("⚠️ Text flow returned empty response",
                                  extra={'subsys': 'router', 'event': 'text.empty'})
                return BotAction(content="I couldn't generate a response to that.", error=True)
                
            self.logger.debug(f"✅ Text flow completed with response length: {len(response_text)}",
                            extra={'subsys': 'router', 'event': 'text.complete', 'response_length': len(response_text)})
            return BotAction(content=response_text)
            
        except Exception as e:
            self.logger.error(f"❌ Error in text flow: {e}", exc_info=True,
                            extra={'subsys': 'router', 'event': 'text.error'})
            return BotAction(content="I encountered an error while processing your text.", error=True)

    async def _flow_process_text(self, content: str) -> str:
        """Process text input through the AI model."""
        self.logger.info(f"💬 Processing text input through AI model",
                       extra={'subsys': 'router', 'event': 'flow.text.process'})
        return await brain_infer(content)

    async def _flow_process_url(self, url: str) -> str:
        """Process URL input by fetching and summarizing content."""
        self.logger.info(f"🌐 Processing URL: {url}",
                       extra={'subsys': 'router', 'event': 'flow.url.process'})
        return await process_url(url)

    async def _flow_process_audio(self, message: Message) -> str:
        """Process audio attachment through STT model."""
        self.logger.info(f"🔊 Processing audio attachment",
                       extra={'subsys': 'router', 'event': 'flow.audio.process'})
        # This would integrate with an STT service
        return await hear_infer(message)

    async def _flow_process_attachments(self, message: Message, attachment) -> str:
        """Process image/document attachments."""
        self.logger.info(f"📎 Processing attachment: {attachment.filename}",
                       extra={'subsys': 'router', 'event': 'flow.attachment.process'})
        
        # Process image attachments
        if attachment.content_type and attachment.content_type.startswith("image/"):
            self.logger.info(f"🖼 Processing image attachment: {attachment.filename}", 
                           extra={'subsys': 'router', 'event': 'image.process', 'msg_id': message.id})
            
            # Create a temporary file to save the image
            file_extension = os.path.splitext(attachment.filename)[1]
            if not file_extension:
                file_extension = ".jpg"  # Default extension
                self.logger.warning(f"⚠️ No extension found for image, defaulting to {file_extension}", 
                                  extra={'subsys': 'router', 'event': 'image.extension', 'msg_id': message.id})
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                tmp_path = tmp_file.name
                
            try:
                # Save the attachment to the temporary file
                self.logger.debug(f"💾 Saving image to temporary file: {tmp_path}", 
                                extra={'subsys': 'router', 'event': 'image.save', 'msg_id': message.id})
                await attachment.save(tmp_path)
                
                # Verify the file was created
                if not os.path.exists(tmp_path):
                    self.logger.error(f"❌ Image file not created: {tmp_path}", 
                                    extra={'subsys': 'router', 'event': 'image.save.fail', 'msg_id': message.id})
                    return "Error: Failed to save image"
                    
                file_size = os.path.getsize(tmp_path)
                file_permissions = oct(os.stat(tmp_path).st_mode)
                self.logger.debug(f"✅ Image saved successfully: path={tmp_path}, size={file_size} bytes, perms={file_permissions}", 
                                extra={'subsys': 'router', 'event': 'image.save.success', 
                                       'msg_id': message.id, 'file_path': tmp_path, 'file_size': file_size, 'file_permissions': file_permissions})
                
                # Call the vision model
                text = message.content.strip() or "Describe this image."
                self.logger.debug(f"👁 Calling vision model with prompt: '{text}'", 
                                extra={'subsys': 'router', 'event': 'image.vision.call', 'msg_id': message.id})
                
                try:
                    vision_response = await see_infer(image_path=tmp_path, prompt=text)
                except Exception as vision_error:
                    self.logger.error(f"❌ Vision model error: {vision_error}", 
                                    exc_info=True,
                                    extra={'subsys': 'router', 'event': 'image.vision.error', 
                                           'msg_id': message.id, 'error': str(vision_error)})
                    return f"Error: Vision model failed - {str(vision_error)}"
                
                if not vision_response:
                    self.logger.warning(f"⚠️ Vision model returned empty response", 
                                      extra={'subsys': 'router', 'event': 'image.vision.empty', 
                                             'msg_id': message.id})
                    return "I couldn't understand the image"
                    
                self.logger.debug(f"✅ Vision model response received: {len(vision_response)} chars", 
                                extra={'subsys': 'router', 'event': 'image.vision.success', 
                                       'msg_id': message.id, 'response_length': len(vision_response)})
                
                # Combine the vision response with the user's text prompt and call the text model
                final_prompt = f"User uploaded an image with the prompt: '{text}'. The image contains: {vision_response}"
                self.logger.debug(f"💬 Calling text model with combined prompt", 
                                extra={'subsys': 'router', 'event': 'image.text.call', 'msg_id': message.id})
                
                try:
                    text_response = await brain_infer(final_prompt)
                except Exception as text_error:
                    self.logger.error(f"❌ Text model error for image flow: {text_error}", 
                                    exc_info=True,
                                    extra={'subsys': 'router', 'event': 'image.text.error', 
                                           'msg_id': message.id, 'error': str(text_error)})
                    return f"Error: Text model failed - {str(text_error)}"
                
                self.logger.debug(f"✅ Text model response for image flow: {len(text_response)} chars", 
                                extra={'subsys': 'router', 'event': 'image.text.success', 
                                       'msg_id': message.id, 'response_length': len(text_response)})
                
                # Clean up the temporary file
                try:
                    os.unlink(tmp_path)
                    self.logger.debug(f"🚮 Temporary image file deleted: {tmp_path}", 
                                    extra={'subsys': 'router', 'event': 'image.cleanup', 'msg_id': message.id})
                except Exception as cleanup_err:
                    self.logger.warning(f"⚠️ Failed to delete temporary image: {cleanup_err}", 
                                      extra={'subsys': 'router', 'event': 'image.cleanup.fail', 
                                             'msg_id': message.id, 'error': str(cleanup_err)})
                
                self.logger.info(f"🏁 === IMAGE PROCESSING COMPLETED ==== ", 
                               extra={'subsys': 'router', 'event': 'image.complete', 'msg_id': message.id})
                return text_response
                
            except Exception as e:
                self.logger.error(f"❌ Image processing failed: {e}", 
                                exc_info=True,
                                extra={'subsys': 'router', 'event': 'image.fail', 
                                       'msg_id': message.id, 'error': str(e)})
                self.logger.info(f"🏁 === IMAGE PROCESSING FAILED ==== ", 
                               extra={'subsys': 'router', 'event': 'image.fail', 'msg_id': message.id})
                return "⚠️ An error occurred while processing this image."
                
        # Process document attachments (PDF, DOCX)
        elif attachment.filename.lower().endswith('.pdf') and self.pdf_processor:
            self.logger.info(f"📄 Processing PDF attachment: {attachment.filename}", 
                           extra={'subsys': 'router', 'event': 'pdf.process', 'msg_id': message.id})
            try:
                # Download the PDF to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_path = tmp_file.name
                    
                await attachment.save(tmp_path)
                
                # Process the PDF
                text_content = await self.pdf_processor.process(tmp_path)
                
                # Clean up
                os.unlink(tmp_path)
                
                if not text_content:
                    return "I couldn't extract any text from that PDF."
                
                # Pass the extracted text to the text model
                final_prompt = f"User uploaded a PDF document. Here is the text content:\n\n{text_content}"
                return await brain_infer(final_prompt)
                
            except Exception as e:
                self.logger.error(f"❌ PDF processing failed: {e}", exc_info=True)
                return "⚠️ An error occurred while processing this PDF."
                
        else:
            self.logger.warning(f"⚠️ Unsupported attachment type: {attachment.filename}", 
                              extra={'subsys': 'router', 'event': 'attachment.unsupported', 'msg_id': message.id})
            return "I can't process that type of file attachment."

    async def _flow_generate_tts(self, text: str) -> Optional[str]:
        """Generate TTS audio from text."""
        self.logger.info(f"🔊 Generating TTS for text of length: {len(text)}", 
                       extra={'subsys': 'router', 'event': 'tts.generate'})
        # This would integrate with a TTS service
        # For now, we'll just return None to indicate TTS is not implemented
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