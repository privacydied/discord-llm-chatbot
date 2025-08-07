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


from . import web
from discord import Message, DMChannel, Embed, File
from bot.brain import brain_infer

if TYPE_CHECKING:
    from bot.core.bot import LLMBot as DiscordBot
    from bot.metrics import Metrics
    from .command_parser import ParsedCommand

logger = get_logger(__name__)

# Local application imports
from .action import BotAction, ResponseMessage
from .command_parser import Command, parse_command
from .exceptions import DispatchEmptyError, DispatchTypeError
from .hear import hear_infer, hear_infer_from_url
from .pdf_utils import PDFProcessor
from .see import see_infer
from .web import process_url
from .utils.mention_utils import ensure_single_mention

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
    """Defines the type of input the bot is processing."""
    TEXT_ONLY = auto()
    URL = auto()
    VIDEO_URL = auto()  # YouTube/TikTok URLs for audio transcription
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

    def _is_reply_to_bot(self, message: Message) -> bool:
        """Check if a message is a reply to the bot."""
        if message.reference and message.reference.message_id:
            # To check who the replied-to message is from, we might need to fetch the message
            # This is a simplification. For a robust solution, you might need to fetch the message
            # if it's not in the cache, which is an async operation.
            # Here we assume a simple check is enough, or the logic is handled elsewhere.
            ref_msg = message.reference.resolved
            if ref_msg and ref_msg.author.id == self.bot.user.id:
                return True
        return False

    def _should_process_message(self, message: Message) -> bool:
        """Determine if the message should be processed based on context (DM, mention)."""
        is_dm = isinstance(message.channel, DMChannel)
        is_mentioned = self.bot.user in message.mentions
        is_reply = self._is_reply_to_bot(message)

        if is_dm:
            self.logger.debug(f"Processing message {message.id}: It's a DM.")
            return True

        if is_mentioned:
            self.logger.debug(f"Processing message {message.id}: Bot is mentioned.")
            return True

        if is_reply:
            self.logger.debug(f"Processing message {message.id}: It's a reply to the bot.")
            return True

        self.logger.debug(f"Ignoring message {message.id}: Not a DM, mention, or reply to the bot.")
        return False

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
            if not self._should_process_message(message):
                self.logger.debug(f"Ignoring message {message.id} in guild {message.guild.id if message.guild else 'N/A'}: Not a DM or direct mention.")
                return None

            # --- Start of processing for DMs, Mentions, and Replies ---
            async with message.channel.typing():
                self.logger.info(f"Processing message: DM={isinstance(message.channel, DMChannel)}, Mention={self.bot.user in message.mentions} (msg_id: {message.id})")

                # 4. Gather conversation history for context
                context_str = await self.bot.context_manager.get_context_string(message)
                self.logger.info(f"📚 Gathered context. (msg_id: {message.id})")

                # 5. Determine modalities and process
                input_modality = self._get_input_modality(message)
                self._metric_inc('router_input_modality', {'modality': input_modality.name.lower()})
                output_modality = self._get_output_modality(None, message)
                action = None
                text_content = message.content
                if self.bot.user in message.mentions:
                    text_content = re.sub(r'^(\u003c@!?\u0026?{})'.format(self.bot.user.id), '', text_content).strip()

                if input_modality == InputModality.TEXT_ONLY:
                    action = await self._invoke_text_flow(text_content, message, context_str)
                elif input_modality == InputModality.VIDEO_URL:
                    # Extract video URL and process through STT pipeline
                    video_patterns = [
                        r'https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+',
                        r'https?://youtu\.be/[\w-]+',
                        r'https?://(?:www\.)?tiktok\.com/@[\w.-]+/video/\d+',
                        r'https?://(?:vm\.)?tiktok\.com/[\w-]+',
                    ]
                    video_url = None
                    for pattern in video_patterns:
                        match = re.search(pattern, text_content)
                        if match:
                            video_url = match.group(0)
                            break
                    
                    if video_url:
                        action = await self._flow_process_video_url(video_url, message)
                elif input_modality == InputModality.URL:
                    url_match = re.search(r'https?://[\S]+', text_content)
                    if url_match:
                        action = await self._flow_process_url(url_match.group(0), message)
                elif input_modality in [InputModality.IMAGE, InputModality.DOCUMENT, InputModality.AUDIO]:
                    if message.attachments:
                        action = await self._flow_process_attachments(message, message.attachments[0])

                # 7. Finalize and return the action
                if action and action.content:
                    # Note: User mentions are handled by Discord's mention_author=True in message.reply()
                    # No manual mention processing needed here
                    
                    self.logger.info(f"✅ Preparing to reply with content: {action.content[:100]}... (msg_id: {message.id})")

                    # Generate TTS if needed
                    if output_modality == OutputModality.TTS and not action.files:
                        action.audio_path = await self._generate_tts_safe(action.content)

                    return action
                else:
                    self.logger.info(f"🤔 No action taken for message {message.id}. Inference result was empty.")
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

        # Check for video URLs (YouTube/TikTok) first
        video_patterns = [
            r'https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+',
            r'https?://youtu\.be/[\w-]+',
            r'https?://(?:www\.)?tiktok\.com/@[\w.-]+/video/\d+',
            r'https?://(?:vm\.)?tiktok\.com/[\w-]+',
        ]
        
        for pattern in video_patterns:
            if re.search(pattern, message.content):
                return InputModality.VIDEO_URL
        
        # Check for other URLs
        if re.search(r'https?://[\S]+', message.content):
            return InputModality.URL
            
        return InputModality.TEXT_ONLY

    def _get_output_modality(self, parsed_command: Optional[ParsedCommand], message: Message) -> OutputModality:
        """Determine the output modality based on command or channel settings."""
        # Future: check for TTS commands or channel/user settings
        return OutputModality.TEXT

    async def _invoke_text_flow(self, content: str, message: Message, context_str: str) -> BotAction:
        """Invoke the text processing flow, formatting history into a context string."""
        self.logger.info(f"Routing to text flow. (msg_id: {message.id})")
        try:
            action = await self._flows['process_text'](content, context_str, message)
            if action and action.has_payload:
                return action
            else:
                self.logger.warning(f"Text flow returned no response. (msg_id: {message.id})")
                return None
        except Exception as e:
            self.logger.error(f"Text processing flow failed: {e} (msg_id: {message.id})", exc_info=True)
            return BotAction(content="I had trouble processing that text.", error=True)

    async def _flow_process_text(self, content: str, context: str = "", message: Optional[Message] = None) -> BotAction:
        """Process text input through the AI model with RAG integration and conversation context."""
        self.logger.info("Processing text with AI model and RAG integration.")
        
        enhanced_context = context
        
        # 1. RAG Integration - Search vector database for relevant knowledge
        if os.getenv("ENABLE_RAG", "true").lower() == "true":
            try:
                from bot.rag.hybrid_search import get_hybrid_search
                max_results = int(os.getenv("RAG_MAX_VECTOR_RESULTS", "5"))
                self.logger.info(f"🔍 RAG: Searching vector database for: '{content[:100]}{'...' if len(content) > 100 else ''}' [msg_id={message.id if message else 'N/A'}]")
                
                search_engine = await get_hybrid_search()
                if search_engine:
                    rag_results = await search_engine.search(
                        query=content,
                        max_results=max_results
                    )
                    self.logger.info(f"📊 RAG: Search completed, found {len(rag_results) if rag_results else 0} results (max_results={max_results})")
                    
                    if rag_results:
                        # Extract relevant content from search results (List[HybridSearchResult])
                        rag_context_parts = []
                        self.logger.info(f"📋 RAG: Processing {len(rag_results)} search results...")
                        
                        for i, result in enumerate(rag_results[:5]):  # Limit to top 5 results
                            # HybridSearchResult should have content attribute or similar
                            if hasattr(result, 'content'):
                                chunk_content = result.content.strip()
                            elif hasattr(result, 'text'):
                                chunk_content = result.text.strip()
                            elif isinstance(result, dict):
                                chunk_content = result.get('content', result.get('text', '')).strip()
                            else:
                                chunk_content = str(result).strip()
                            
                            if chunk_content:
                                rag_context_parts.append(chunk_content)
                                # Show preview of each chunk found
                                preview = chunk_content[:150] + "..." if len(chunk_content) > 150 else chunk_content
                                self.logger.info(f"📄 RAG: Chunk {i+1}: {preview}")
                            else:
                                self.logger.debug(f"⚠️ RAG: Chunk {i+1} was empty or invalid")
                        
                        if rag_context_parts:
                            rag_context = "\n\n".join(rag_context_parts)
                            enhanced_context = f"{context}\n\n=== Relevant Knowledge ===\n{rag_context}\n=== End Knowledge ===\n" if context else f"=== Relevant Knowledge ===\n{rag_context}\n=== End Knowledge ===\n"
                            self.logger.info(f"✅ RAG: Enhanced context with {len(rag_context_parts)} knowledge chunks (total chars: {len(rag_context)}) [msg_id={message.id if message else 'N/A'}]")
                        else:
                            self.logger.warning(f"⚠️ RAG: Search returned {len(rag_results)} results but all chunks were empty [msg_id={message.id if message else 'N/A'}]")
                    else:
                        self.logger.info(f"🚫 RAG: No relevant results found in vector database for query [msg_id={message.id if message else 'N/A'}]")
                else:
                    self.logger.warning(f"⚠️ RAG: Search engine not available - check RAG system initialization [msg_id={message.id if message else 'N/A'}]")
            except Exception as e:
                self.logger.error(f"❌ RAG: Search failed with error: {e} [msg_id={message.id if message else 'N/A'}]", exc_info=True)
        
        # 2. Use contextual brain inference if enhanced context manager is available and message is provided
        if (message and hasattr(self.bot, 'enhanced_context_manager') and 
            self.bot.enhanced_context_manager and 
            os.getenv("USE_ENHANCED_CONTEXT", "true").lower() == "true"):
            
            try:
                from bot.contextual_brain import contextual_brain_infer_simple
                self.logger.debug(f"🧠 Using contextual brain inference [msg_id={message.id}]")
                response_text = await contextual_brain_infer_simple(message, content, self.bot)
                return BotAction(content=response_text)
            except Exception as e:
                self.logger.warning(f"Contextual brain inference failed, falling back to basic: {e}")
        
        # 3. Fallback to basic brain inference with enhanced context (including RAG)
        return await brain_infer(content, context=enhanced_context)

    async def _flow_process_url(self, url: str, message: discord.Message) -> BotAction:
        """
        Processes a URL with smart media ingestion and graceful fallback to scraping.
        """
        self.logger.info(f"🌐 Processing URL: {url} (msg_id: {message.id})")
        
        try:
            # Use smart media ingestion system
            if not hasattr(self, '_media_ingestion_manager'):
                from .media_ingestion import create_media_ingestion_manager
                self._media_ingestion_manager = create_media_ingestion_manager(self.bot)
            
            return await self._media_ingestion_manager.process_url_smart(url, message)
            
        except Exception as e:
            self.logger.error(f"❌ Smart URL processing failed unexpectedly: {e} (msg_id: {message.id})", exc_info=True)
            return BotAction(content="⚠️ An unexpected error occurred while processing this URL.", error=True)

    async def _flow_process_video_url(self, url: str, message: Message) -> BotAction:
        """Process video URL through STT pipeline and integrate with conversation context."""
        self.logger.info(f"🎥 Processing video URL: {url} (msg_id: {message.id})")
        
        try:
            # Transcribe video URL audio
            result = await hear_infer_from_url(url)
            
            transcription = result['transcription']
            metadata = result['metadata']
            
            # Create enriched context for the LLM
            video_context = (
                f"User shared a {metadata['source']} video: '{metadata['title']}' "
                f"by {metadata['uploader']} (Duration: {metadata['original_duration_s']:.1f}s, "
                f"processed at {metadata['speedup_factor']}x speed). "
                f"The following is the audio transcription:\n\n{transcription}"
            )
            
            # Get existing conversation context
            context_str = await self.bot.context_manager.get_context_string(message)
            
            # Combine video context with conversation history
            if context_str:
                full_context = f"{context_str}\n\n--- VIDEO CONTENT ---\n{video_context}"
            else:
                full_context = video_context
            
            # Process through text flow with enriched context
            prompt = (
                f"Please summarize and discuss the key points from this video. "
                f"Provide insights, analysis, or answer any questions about the content."
            )
            
            # Use contextual brain inference if available
            if (hasattr(self.bot, 'enhanced_context_manager') and 
                self.bot.enhanced_context_manager and 
                os.getenv("USE_ENHANCED_CONTEXT", "true").lower() == "true"):
                
                try:
                    from bot.contextual_brain import contextual_brain_infer_simple
                    self.logger.debug(f"🧠🎥 Using contextual brain for video analysis [msg_id={message.id}]")
                    
                    # Add video metadata to enhanced context
                    video_metadata_context = {
                        'source': metadata['source'],
                        'url': metadata['url'],
                        'title': metadata['title'],
                        'uploader': metadata['uploader'],
                        'original_duration_s': metadata['original_duration_s'],
                        'processed_duration_s': metadata['processed_duration_s'],
                        'speedup_factor': metadata['speedup_factor'],
                        'timestamp': metadata['timestamp']
                    }
                    
                    response_text = await contextual_brain_infer_simple(
                        message, video_context, self.bot, additional_context=video_metadata_context
                    )
                    return BotAction(content=response_text)
                    
                except Exception as e:
                    self.logger.warning(f"Contextual brain inference failed for video, falling back: {e}")
            
            # Fallback to basic brain inference
            return await brain_infer(prompt, context=full_context)
            
        except Exception as e:
            self.logger.error(f"❌ Video URL processing failed: {e} (msg_id: {message.id})", exc_info=True)
            error_msg = str(e).lower()
            
            # Provide user-friendly error messages
            if "unsupported url" in error_msg:
                return BotAction(content="❌ This URL is not supported. Please use YouTube or TikTok links.", error=True)
            elif "video too long" in error_msg:
                return BotAction(content="❌ This video is too long to process. Please try a shorter video (max 10 minutes).", error=True)
            elif "download failed" in error_msg:
                return BotAction(content="❌ Could not download the video. It may be private, unavailable, or region-locked.", error=True)
            elif "audio processing failed" in error_msg:
                return BotAction(content="❌ Could not process the audio from this video. The audio format may be unsupported.", error=True)
            else:
                return BotAction(content="❌ An error occurred while processing the video. Please try again or use a different video.", error=True)

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

            prompt = message.content.strip() or (self.bot.system_prompts.get("VL_PROMPT_FILE") or "Describe this image.")
            vision_response = await see_infer(image_path=tmp_path, prompt=prompt)

            if not vision_response or vision_response.error:
                self.logger.warning(f"Vision model returned no/error response (msg_id: {message.id})")
                return BotAction(content="I couldn't understand the image.", error=True)

            vl_content = vision_response.content
            # Truncate if response is too long for Discord
            if len(vl_content) > 1999:
                self.logger.info(f"VL response is too long ({len(vl_content)} chars), truncating for text fallback.")
                vl_content = vl_content[:1999].rsplit('\n', 1)[0]

            final_prompt = f"User uploaded an image with the prompt: '{prompt}'. The image contains: {vl_content}"
            return await brain_infer(final_prompt)

        except Exception as e:
            self.logger.error(f"❌ Image processing failed: {e} (msg_id: {message.id})", exc_info=True)
            
            # Provide user-friendly error messages based on error type
            error_str = str(e).lower()
            if "502" in error_str or "provider returned error" in error_str:
                return BotAction(content="🔄 Vision processing failed. This could be due to a temporary service issue. Please try again in a moment.", error=True)
            elif "timeout" in error_str:
                return BotAction(content="⏱️ Vision processing timed out. Please try again with a smaller image.", error=True)
            elif "file format" in error_str or "unsupported" in error_str:
                return BotAction(content="📷 Unsupported image format. Please try uploading a JPEG, PNG, or WebP image.", error=True)
            elif "file size" in error_str or "too large" in error_str:
                return BotAction(content="📏 Image is too large. Please try uploading a smaller image.", error=True)
            else:
                return BotAction(content="⚠️ An error occurred while processing this image. Please try again.", error=True)
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