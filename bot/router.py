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
from .pdf_utils import PDFProcessor
from .see import see_infer
from .web import process_url

logger = logging.getLogger(__name__)

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
        """Process a message and ensure exactly one response is generated (1 IN > 1 OUT rule).
        
        This method enforces the strict 1 IN > 1 OUT principle, ensuring that each processed message
        will either be explicitly ignored (return None) or will generate exactly one response.
        
        Args:
            message: The Discord message to process
            
        Returns:
            ResponseMessage: A single response with text and/or audio, or None if message is ignored
        """
        msg_id = message.id
        user_id = str(message.author.id) if message.author else 'unknown'
        guild_id = str(message.guild.id) if message.guild else 'dm'
        
        self.logger.info(f"🔄 === ROUTER DISPATCH STARTED: MSG {msg_id} ====", 
                       extra={'subsys': 'router', 'event': 'dispatch.start', 
                              'msg_id': msg_id, 'user_id': user_id, 'guild_id': guild_id})
        
        # Track if this message should be processed by the router
        should_process = True
        
        try:
            # 1. Gatekeeping: Parse the command and identify messages to be ignored
            self.logger.debug(f"🔍 Parsing command from message {msg_id}", 
                            extra={'subsys': 'router', 'event': 'command.parse', 'msg_id': msg_id})
            parsed_command = parse_command(message, self.bot)
            
            # Check if message should be ignored (handled by cogs or invalid)
            if parsed_command is None:
                self.logger.debug(f"⏭️ Ignoring message {msg_id}: No valid command parsed", 
                                extra={'subsys': 'router', 'event': 'command.ignore', 'msg_id': msg_id})
                should_process = False
                return None
                
            if parsed_command.command in [Command.IGNORE, Command.TTS, Command.SAY, Command.TTS_ALL, Command.SPEAK]:
                self.logger.debug(f"⏭️ Ignoring message {msg_id}: Command {parsed_command.command} handled by cog", 
                                extra={'subsys': 'router', 'event': 'command.cog_handled', 
                                       'msg_id': msg_id, 'command': str(parsed_command.command)})
                should_process = False
                return None

            # 2. Handle simple, static commands directly.
            self.logger.debug(f"🔍 Processing command: {parsed_command.command}", 
                            extra={'subsys': 'router', 'event': 'command.process', 
                                   'msg_id': msg_id, 'command': str(parsed_command.command)})
            
            # Simple commands with direct responses
            if parsed_command.command == Command.PING:
                self.logger.info(f"✅ Handling PING command", 
                               extra={'subsys': 'router', 'event': 'command.ping', 'msg_id': msg_id})
                self.logger.info(f"🏁 === ROUTER DISPATCH COMPLETED: PING COMMAND ====", 
                               extra={'subsys': 'router', 'event': 'dispatch.complete.ping', 'msg_id': msg_id})
                return ResponseMessage(text="Pong!")
                
            if parsed_command.command == Command.HELP:
                self.logger.info(f"✅ Handling HELP command", 
                               extra={'subsys': 'router', 'event': 'command.help', 'msg_id': msg_id})
                self.logger.info(f"🏁 === ROUTER DISPATCH COMPLETED: HELP COMMAND ====", 
                               extra={'subsys': 'router', 'event': 'dispatch.complete.help', 'msg_id': msg_id})
                return ResponseMessage(text="See `/help` for a list of commands.")

            # 3. Determine modalities and process content
            input_modality = self._get_input_modality(message)
            output_modality = self._get_output_modality(parsed_command, message)
            self.logger.info(f"🔀 MODALITIES: IN={input_modality.name} -> OUT={output_modality.name}", 
                           extra={'subsys': 'router', 'event': 'modality.select', 'msg_id': msg_id, 
                                  'input_modality': input_modality.name, 'output_modality': output_modality.name})
            
            # Enforce 1 IN > 1 OUT rule: Track that we're now committed to generating a response
            should_process = True

            # Initialize response tracking variables
            processed_text = None
            content = parsed_command.cleaned_content
            self.logger.debug(f"🔍 Processing content with length: {len(content)}", 
                            extra={'subsys': 'router', 'event': 'content.process', 'msg_id': msg_id})

            # Process based on input modality - each branch must produce text or raise an exception
            try:
                if input_modality == InputModality.TEXT_ONLY:
                    self.logger.info(f"💬 Processing TEXT_ONLY input", 
                                   extra={'subsys': 'router', 'event': 'flow.text', 'msg_id': msg_id})
                    processed_text = await self._flows['process_text'](content)
                    if not processed_text:
                        self.logger.warning(f"⚠️ TEXT flow returned empty response", 
                                         extra={'subsys': 'router', 'event': 'flow.text.empty', 'msg_id': msg_id})
                        processed_text = "I'm sorry, I wasn't able to process your text message."
                    
                elif input_modality == InputModality.URL:
                    url_match = re.search(r'https?://[\S]+', message.content)
                    if url_match:
                        url = url_match.group(0)
                        self.logger.info(f"🔗 Processing URL input: {url[:50]}{'...' if len(url) > 50 else ''}", 
                                       extra={'subsys': 'router', 'event': 'flow.url', 'msg_id': msg_id})
                        processed_text = await self._flows['process_url'](url)
                        if not processed_text:
                            self.logger.warning(f"⚠️ URL flow returned empty response", 
                                             extra={'subsys': 'router', 'event': 'flow.url.empty', 'msg_id': msg_id})
                            processed_text = "I'm sorry, I wasn't able to process that URL."
                    else:
                        self.logger.warning(f"⚠️ URL modality detected but no URL found in content", 
                                         extra={'subsys': 'router', 'event': 'flow.url.not_found', 'msg_id': msg_id})
                        processed_text = "I couldn't find a valid URL in your message."
                        
                elif input_modality == InputModality.AUDIO:
                    self.logger.info(f"🔊 Processing AUDIO input: {message.attachments[0].filename if message.attachments else 'unknown'}", 
                                   extra={'subsys': 'router', 'event': 'flow.audio', 'msg_id': msg_id})
                    processed_text = await self._flows['process_audio'](message)
                    if not processed_text:
                        self.logger.warning(f"⚠️ AUDIO flow returned empty response", 
                                         extra={'subsys': 'router', 'event': 'flow.audio.empty', 'msg_id': msg_id})
                        processed_text = "I'm sorry, I wasn't able to process that audio message."
                    
                elif input_modality in [InputModality.IMAGE, InputModality.DOCUMENT]:
                    attachment_name = message.attachments[0].filename if message.attachments else 'unknown'
                    attachment_type = 'IMAGE' if input_modality == InputModality.IMAGE else 'DOCUMENT'
                    self.logger.info(f"📎 Processing {attachment_type} attachment: {attachment_name}", 
                                   extra={'subsys': 'router', 'event': f'flow.{attachment_type.lower()}', 'msg_id': msg_id})
                    processed_text = await self._flows['process_attachments'](message, content)
                    if not processed_text:
                        self.logger.warning(f"⚠️ {attachment_type} flow returned empty response", 
                                         extra={'subsys': 'router', 'event': f'flow.{attachment_type.lower()}.empty', 'msg_id': msg_id})
                        processed_text = f"I'm sorry, I wasn't able to process that {attachment_type.lower()}."
                else:
                    # Fallback for any unhandled modality
                    self.logger.error(f"❌ Unhandled input modality: {input_modality}", 
                                    extra={'subsys': 'router', 'event': 'modality.unhandled', 'msg_id': msg_id})
                    processed_text = "I'm sorry, I don't know how to process that type of input."
            except Exception as flow_error:
                # Catch exceptions in flow processing to ensure 1 IN > 1 OUT
                self.logger.error(f"❌ Error in flow processing: {flow_error}", 
                                exc_info=True, 
                                extra={'subsys': 'router', 'event': 'flow.error', 
                                       'msg_id': msg_id, 'error': str(flow_error)})
                processed_text = "I encountered an error while processing your message."

            # 4. Verify we have a valid response (1 IN > 1 OUT rule enforcement)
            # This is a safety check - the flow processing should have already ensured a non-empty response
            if not processed_text:
                self.logger.error("❌ CRITICAL: Message processing resulted in no text despite safeguards", 
                                extra={'subsys': 'router', 'event': 'process.no_result', 'msg_id': msg_id})
                processed_text = "I'm sorry, I wasn't able to process that message. (Error: Empty response)"
                
            self.logger.info(f"✅ Content processed successfully: {len(processed_text)} chars", 
                           extra={'subsys': 'router', 'event': 'process.success', 'msg_id': msg_id})

            # 5. Generate final response based on output modality (1 IN > 1 OUT rule enforcement)
            response = None
            
            try:
                if output_modality == OutputModality.TTS:
                    self.logger.info(f"🔊 Generating TTS output for text of length: {len(processed_text)}", 
                                   extra={'subsys': 'router', 'event': 'output.tts', 'msg_id': msg_id})
                    audio_path = await self._flows['generate_tts'](processed_text)
                    
                    if audio_path:
                        self.logger.info(f"✅ TTS generation successful: {audio_path}", 
                                       extra={'subsys': 'router', 'event': 'output.tts.success', 'msg_id': msg_id})
                        response = ResponseMessage(text=processed_text, audio_path=audio_path)
                    else:
                        self.logger.warning("⚠️ TTS generation failed, falling back to text", 
                                         extra={'subsys': 'router', 'event': 'output.tts.fail', 'msg_id': msg_id})
                        response = ResponseMessage(text=processed_text)
                else:
                    # Default to text output
                    self.logger.info(f"✅ TEXT output generated: {len(processed_text)} chars", 
                                   extra={'subsys': 'router', 'event': 'output.text', 'msg_id': msg_id})
                    response = ResponseMessage(text=processed_text)
            except Exception as output_error:
                # Catch any exceptions in output generation to ensure 1 IN > 1 OUT
                self.logger.error(f"❌ Error in output generation: {output_error}", 
                                exc_info=True, 
                                extra={'subsys': 'router', 'event': 'output.error', 
                                       'msg_id': msg_id, 'error': str(output_error)})
                response = ResponseMessage(text="I encountered an error while generating the response.")
            
            # Final verification of 1 IN > 1 OUT rule
            if not response:
                self.logger.error("❌ CRITICAL: No response generated despite safeguards", 
                                extra={'subsys': 'router', 'event': 'output.missing', 'msg_id': msg_id})
                response = ResponseMessage(text="I'm sorry, an unexpected error occurred. (Error: No response generated)")
                
            self.logger.info(f"🏁 === ROUTER DISPATCH COMPLETED: MSG {msg_id} ====", 
                           extra={'subsys': 'router', 'event': 'dispatch.complete', 'msg_id': msg_id})
            return response

        except Exception as e:
            self.logger.error(f"❌ Error dispatching message {msg_id}: {e}", 
                            exc_info=True, 
                            extra={'subsys': 'router', 'event': 'dispatch.error', 'msg_id': msg_id, 'error': str(e)})
            
            # 1 IN > 1 OUT rule enforcement: If we should process this message, always return a response
            if should_process:
                self.logger.info(f"🏁 === ROUTER DISPATCH FAILED BUT RESPONDING: MSG {msg_id} ====", 
                               extra={'subsys': 'router', 'event': 'dispatch.fail_with_response', 'msg_id': msg_id})
                return ResponseMessage(text="An unexpected error occurred. Please try again later.")
            else:
                # This was a message we were ignoring anyway
                self.logger.info(f"🏁 === ROUTER DISPATCH FAILED: IGNORED MSG {msg_id} ====", 
                               extra={'subsys': 'router', 'event': 'dispatch.fail_ignored', 'msg_id': msg_id})
                return None

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
        
        # Get message metadata for logging
        msg_id = message.id
        user_id = str(message.author.id) if message.author else 'unknown'
        guild_id = str(message.guild.id) if message.guild else 'dm'
        
        # Log attachment details
        self.logger.info(f"📎 === ATTACHMENT PROCESSING STARTED ====", 
                       extra={'subsys': 'router', 'event': 'attachment.start', 
                              'msg_id': msg_id, 'user_id': user_id, 'guild_id': guild_id})
        self.logger.debug(f"📎 Attachment details: name={attachment.filename}, size={attachment.size}, type={attachment.content_type}", 
                        extra={'subsys': 'router', 'event': 'attachment.details', 'msg_id': msg_id})

        # Process image attachments
        if attachment.content_type and attachment.content_type.startswith("image/"):
            self.logger.info(f"🖼 Processing image attachment: {attachment.filename}", 
                           extra={'subsys': 'router', 'event': 'image.process', 'msg_id': msg_id})
            try:
                # Get the correct file extension from the attachment
                file_ext = Path(attachment.filename).suffix.lower()
                if not file_ext:
                    file_ext = ".png"  # Default to PNG if no extension
                    self.logger.debug(f"🖼 No file extension detected, using default: {file_ext}", 
                                    extra={'subsys': 'router', 'event': 'image.extension', 'msg_id': msg_id})
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_img:
                    tmp_path = tmp_img.name
                    self.logger.debug(f"💾 Saving image to temporary file: {tmp_path}", 
                                    extra={'subsys': 'router', 'event': 'image.save', 'msg_id': msg_id})
                    
                    # Save attachment to temp file
                    await attachment.save(tmp_path)
                    
                    # Verify file exists and has content
                    if not os.path.exists(tmp_path):
                        self.logger.error(f"❌ Image file not created: {tmp_path}", 
                                        extra={'subsys': 'router', 'event': 'image.save.fail', 'msg_id': msg_id})
                        return "Error: Failed to save image"
                        
                    file_size = os.path.getsize(tmp_path)
                    self.logger.debug(f"✅ Image saved successfully: {file_size} bytes", 
                                    extra={'subsys': 'router', 'event': 'image.save.success', 
                                           'msg_id': msg_id, 'file_size': file_size})
                    
                    # Process with vision model
                    self.logger.info(f"👁️ Calling vision model with prompt length: {len(text) if text else 0} chars", 
                                   extra={'subsys': 'router', 'event': 'image.vision.call', 'msg_id': msg_id})
                    
                    vision_start_time = asyncio.get_event_loop().time()
                    vision_response = await see_infer(image_path=tmp_path, prompt=text)
                    vision_duration = asyncio.get_event_loop().time() - vision_start_time
                    
                    # Check vision response
                    if not vision_response:
                        self.logger.warning("⚠️ Vision model returned empty response", 
                                         extra={'subsys': 'router', 'event': 'image.vision.empty', 
                                                'msg_id': msg_id, 'duration': vision_duration})
                        return "I couldn't understand the image"
                    
                    self.logger.info(f"✅ Vision model returned {len(vision_response)} chars in {vision_duration:.2f}s", 
                                   extra={'subsys': 'router', 'event': 'image.vision.success', 
                                          'msg_id': msg_id, 'duration': vision_duration})
                    
                    # Process with text model
                    final_prompt = f"User uploaded an image with the prompt: '{text}'. The image contains: {vision_response}"
                    self.logger.debug(f"💬 Sending to text model: prompt length={len(final_prompt)} chars", 
                                    extra={'subsys': 'router', 'event': 'image.text.call', 'msg_id': msg_id})
                    
                    text_start_time = asyncio.get_event_loop().time()
                    text_response = await brain_infer(final_prompt)
                    text_duration = asyncio.get_event_loop().time() - text_start_time
                    
                    self.logger.info(f"✅ Text model returned {len(text_response) if text_response else 0} chars in {text_duration:.2f}s", 
                                   extra={'subsys': 'router', 'event': 'image.text.success', 
                                          'msg_id': msg_id, 'duration': text_duration})
                    
                    # Clean up temp file
                    try:
                        os.unlink(tmp_path)
                        self.logger.debug(f"🚮 Temporary image file deleted: {tmp_path}", 
                                        extra={'subsys': 'router', 'event': 'image.cleanup', 'msg_id': msg_id})
                    except Exception as cleanup_err:
                        self.logger.warning(f"⚠️ Failed to delete temporary image: {cleanup_err}", 
                                         extra={'subsys': 'router', 'event': 'image.cleanup.fail', 'msg_id': msg_id})
                    
                    self.logger.info(f"🏁 === IMAGE PROCESSING COMPLETED ====", 
                                   extra={'subsys': 'router', 'event': 'image.complete', 'msg_id': msg_id})
                    return text_response
                    
            except Exception as e:
                self.logger.error(f"❌ Image processing failed: {e}", 
                                exc_info=True, 
                                extra={'subsys': 'router', 'event': 'image.fail', 'msg_id': msg_id, 'error': str(e)})
                self.logger.info(f"🏁 === IMAGE PROCESSING FAILED ====", 
                               extra={'subsys': 'router', 'event': 'image.fail', 'msg_id': msg_id})
                return "⚠️ An error occurred while processing this image."

        elif input_modality == InputModality.DOCUMENT:
            self.logger.info(f"📝 Processing document attachment: {attachment.filename}", 
                           extra={'subsys': 'router', 'event': 'document.process', 'msg_id': msg_id})
            
            file_ext = Path(attachment.filename).suffix
            self.logger.debug(f"📝 Document file extension: {file_ext}", 
                            extra={'subsys': 'router', 'event': 'document.extension', 'msg_id': msg_id})
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                tmp_path = tmp_file.name
                self.logger.debug(f"💾 Saving document to temporary file: {tmp_path}", 
                                extra={'subsys': 'router', 'event': 'document.save', 'msg_id': msg_id})
                
                # Save using string path, not Path object
                await attachment.save(tmp_path)
                
                # Verify file exists and has content
                if not os.path.exists(tmp_path):
                    self.logger.error(f"❌ Document file not created: {tmp_path}", 
                                    extra={'subsys': 'router', 'event': 'document.save.fail', 'msg_id': msg_id})
                    return "Error: Failed to save document"
                    
                file_size = os.path.getsize(tmp_path)
                self.logger.debug(f"✅ Document saved successfully: {file_size} bytes", 
                                extra={'subsys': 'router', 'event': 'document.save.success', 
                                       'msg_id': msg_id, 'file_size': file_size})
                
                # Process document content
                self.logger.info(f"📝 Processing document content: {attachment.filename}", 
                               extra={'subsys': 'router', 'event': 'document.extract', 'msg_id': msg_id})
                
                doc_start_time = asyncio.get_event_loop().time()
                try:
                    document_text = await self._process_document(tmp_path, file_ext)
                    doc_duration = asyncio.get_event_loop().time() - doc_start_time
                    
                    if document_text:
                        self.logger.info(f"✅ Document text extracted: {len(document_text)} chars in {doc_duration:.2f}s", 
                                       extra={'subsys': 'router', 'event': 'document.extract.success', 
                                              'msg_id': msg_id, 'duration': doc_duration})
                    else:
                        self.logger.warning("⚠️ Document processing returned empty text", 
                                         extra={'subsys': 'router', 'event': 'document.extract.empty', 
                                                'msg_id': msg_id, 'duration': doc_duration})
                        document_text = "No readable text found in document."
                        
                except Exception as e:
                    self.logger.error(f"❌ Error processing document: {e}", 
                                    exc_info=True, 
                                    extra={'subsys': 'router', 'event': 'document.extract.fail', 
                                           'msg_id': msg_id, 'error': str(e)})
                    document_text = "⚠️ An error occurred while processing this document."
            
            # Clean up temp file
            try:
                os.remove(tmp_path)
                self.logger.debug(f"🚮 Temporary document file deleted: {tmp_path}", 
                                extra={'subsys': 'router', 'event': 'document.cleanup', 'msg_id': msg_id})
            except OSError as e:
                self.logger.warning(f"⚠️ Error removing temporary file {tmp_path}: {e}", 
                                 extra={'subsys': 'router', 'event': 'document.cleanup.fail', 
                                        'msg_id': msg_id, 'error': str(e)})

            # Process with text model
            prompt = f"DOCUMENT CONTENT:\n---\n{document_text}\n---\n\nUSER'S PROMPT: {text}"
            self.logger.debug(f"💬 Sending to text model: prompt length={len(prompt)} chars", 
                            extra={'subsys': 'router', 'event': 'document.text.call', 'msg_id': msg_id})
            
            text_start_time = asyncio.get_event_loop().time()
            text_response = await brain_infer(prompt)
            text_duration = asyncio.get_event_loop().time() - text_start_time
            
            self.logger.info(f"✅ Text model returned {len(text_response) if text_response else 0} chars in {text_duration:.2f}s", 
                           extra={'subsys': 'router', 'event': 'document.text.success', 
                                  'msg_id': msg_id, 'duration': text_duration})
            
            self.logger.info(f"🏁 === DOCUMENT PROCESSING COMPLETED ====", 
                           extra={'subsys': 'router', 'event': 'document.complete', 'msg_id': msg_id})
            return text_response

        # Fallback for unhandled attachment types
        self.logger.warning(f"⚠️ Unhandled attachment type for file: {attachment.filename}", 
                         extra={'subsys': 'router', 'event': 'attachment.unsupported', 
                                'msg_id': msg_id, 'content_type': attachment.content_type})
        self.logger.info(f"🏁 === ATTACHMENT PROCESSING FAILED: UNSUPPORTED TYPE ====", 
                       extra={'subsys': 'router', 'event': 'attachment.fail', 'msg_id': msg_id})
        return "I'm sorry, I can't process that type of attachment."

    async def _flow_generate_tts(self, text: str) -> str:
        """Generate TTS audio from text."""
        self.logger.info(f"🔊 === TTS GENERATION STARTED ====", 
                       extra={'subsys': 'router', 'event': 'tts.start'})
        self.logger.debug(f"🔊 Input text length: {len(text)} chars", 
                        extra={'subsys': 'router', 'event': 'tts.input'})
        
        try:
            # Check TTS manager availability
            if not self.bot.tts_manager:
                self.logger.error("❌ TTS Manager not initialized", 
                                extra={'subsys': 'router', 'event': 'tts.manager.missing'})
                self.logger.info(f"🏁 === TTS GENERATION FAILED: NO MANAGER ====", 
                               extra={'subsys': 'router', 'event': 'tts.fail'})
                return None
            
            # Check if text is suitable for TTS
            if not text or len(text.strip()) == 0:
                self.logger.warning("⚠️ Empty text provided for TTS generation", 
                                 extra={'subsys': 'router', 'event': 'tts.empty_input'})
                self.logger.info(f"🏁 === TTS GENERATION FAILED: EMPTY INPUT ====", 
                               extra={'subsys': 'router', 'event': 'tts.fail'})
                return None
                
            # Generate TTS
            self.logger.info("🔊 Calling TTS manager to generate audio", 
                           extra={'subsys': 'router', 'event': 'tts.generate.call'})
            
            tts_start_time = asyncio.get_event_loop().time()
            audio_path = await self.bot.tts_manager.generate_tts(text)
            tts_duration = asyncio.get_event_loop().time() - tts_start_time
            
            # Verify TTS output
            if not audio_path:
                self.logger.error("❌ TTS generation returned no audio path", 
                                extra={'subsys': 'router', 'event': 'tts.generate.no_path', 
                                       'duration': tts_duration})
                self.logger.info(f"🏁 === TTS GENERATION FAILED: NO PATH ====", 
                               extra={'subsys': 'router', 'event': 'tts.fail'})
                return None
                
            if not os.path.exists(audio_path):
                self.logger.error(f"❌ TTS audio file not found: {audio_path}", 
                                extra={'subsys': 'router', 'event': 'tts.file.missing', 
                                       'path': audio_path, 'duration': tts_duration})
                self.logger.info(f"🏁 === TTS GENERATION FAILED: FILE NOT FOUND ====", 
                               extra={'subsys': 'router', 'event': 'tts.fail'})
                return None
            
            # Check file size and validity
            file_size = os.path.getsize(audio_path)
            if file_size == 0:
                self.logger.error(f"❌ TTS generated empty audio file: {audio_path}", 
                                extra={'subsys': 'router', 'event': 'tts.file.empty', 
                                       'path': audio_path, 'duration': tts_duration})
                self.logger.info(f"🏁 === TTS GENERATION FAILED: EMPTY FILE ====", 
                               extra={'subsys': 'router', 'event': 'tts.fail'})
                return None
                
            self.logger.info(f"✅ TTS generated successfully: {audio_path} ({file_size} bytes) in {tts_duration:.2f}s", 
                           extra={'subsys': 'router', 'event': 'tts.generate.success', 
                                  'path': audio_path, 'size': file_size, 'duration': tts_duration})
            self.logger.info(f"🏁 === TTS GENERATION COMPLETED ====", 
                           extra={'subsys': 'router', 'event': 'tts.complete'})
            return audio_path
            
        except Exception as e:
            self.logger.error(f"❌ Error generating TTS: {e}", 
                            exc_info=True, 
                            extra={'subsys': 'router', 'event': 'tts.error', 'error': str(e)})
            self.logger.info(f"🏁 === TTS GENERATION FAILED: EXCEPTION ====", 
                           extra={'subsys': 'router', 'event': 'tts.fail'})
            return None

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
