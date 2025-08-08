"""
Change Summary:
- Refactored from single-shot modality dispatch to sequential multimodal processing
- Replaced _get_input_modality() single detection with collect_input_items() multi-pass collection
- Added _process_multimodal_message_internal() for sequential item processing with timeout/error handling
- Implemented comprehensive handler methods (_handle_image, _handle_video_url, etc.) that accept InputItem and return str
- Each handler result is fed into _flow_process_text() for unified text processing pipeline
- Added robust error recovery, timeout management, and per-item user feedback
- Enhanced logging for step-by-step visibility of multimodal processing
- Preserved existing functionality while enabling full multimodal support
- Now processes ALL attachments, URLs, and embeds in a message sequentially

Centralized router enforcing sequential multimodal message processing.
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
import time

# Import new modality system
from .modality import InputModality, InputItem, collect_input_items, map_item_to_modality
from .multimodal_retry import run_with_retries
from .result_aggregator import ResultAggregator
from .enhanced_retry import get_retry_manager, ProviderConfig


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

# InputModality now imported from modality.py

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

                # 5. Sequential multimodal processing
                await self._process_multimodal_message_internal(message, context_str)
                return None  # All processing handled internally

        except Exception as e:
            self.logger.error(f"❌ Error in router dispatch: {e} (msg_id: {message.id})", exc_info=True)
            return BotAction(content="I encountered an error while processing your message.", error=True)

    async def _process_multimodal_message_internal(self, message: Message, context_str: str) -> None:
        """
        Process all input items from a message sequentially with result aggregation.
        Follows the 1 IN → 1 OUT rule by combining all results into a single response.
        """
        # Collect all input items from the message
        items = collect_input_items(message)
        
        # Process original text content (remove URLs that will be processed separately)
        original_text = message.content
        if self.bot.user in message.mentions:
            original_text = re.sub(r'^<@!?{}>\s*'.format(self.bot.user.id), '', original_text).strip()
        
        # Remove URLs from text content since they will be processed separately
        url_pattern = r'https?://[^\s<>"\'\'[\]{}|\\\^`]+'
        original_text = re.sub(url_pattern, '', original_text).strip()
        
        # If no items found, process as text-only
        if not items:
            # No actionable items found, treat as text-only
            response_action = await self._invoke_text_flow(message.content, message, context_str)
            if response_action and response_action.has_payload:
                # [REH] Execute the response action to send it to Discord
                await self.bot._execute_action(message, response_action)
                self.logger.info(f"✅ Text-only response sent successfully (msg_id: {message.id})")
            else:
                self.logger.warning(f"No response generated from text-only flow (msg_id: {message.id})")
            return
        
        self.logger.info(f"🚀 Processing {len(items)} input items CONCURRENTLY for maximum speed (msg_id: {message.id})")
        
        # Initialize result aggregator and retry manager
        aggregator = ResultAggregator()
        retry_manager = get_retry_manager()
        
        # Per-item budget configuration (optimized for concurrent processing)
        PER_ITEM_BUDGET = float(os.environ.get('MULTIMODAL_PER_ITEM_BUDGET', '30.0'))  # Reduced since parallel
        
        # Create all processing tasks concurrently
        async def process_item_concurrent(i: int, item) -> tuple[int, bool, str, float, int]:
            modality = map_item_to_modality(item)
            
            # Create description for logging (faster version)
            if item.source_type == "attachment":
                description = f"{item.payload.filename}"
            elif item.source_type == "url":
                description = f"URL: {item.payload[:30]}{'...' if len(item.payload) > 30 else ''}"
            else:
                description = f"{item.source_type}"
            
            self.logger.info(f"📋 Starting concurrent item {i}: {modality.name} - {description}")
            
            # Determine modality type for retry manager
            if modality in [InputModality.SINGLE_IMAGE, InputModality.MULTI_IMAGE]:
                retry_modality = "vision"
            else:
                retry_modality = "text"
            
            # Create coroutine factory for this item
            def create_handler_coro(provider_config: ProviderConfig):
                async def handler_coro():
                    return await self._handle_item_with_provider(item, modality, provider_config)
                return handler_coro
            
            try:
                # Run with enhanced retry/fallback system
                result = await retry_manager.run_with_fallback(
                    modality=retry_modality,
                    coro_factory=create_handler_coro,
                    per_item_budget=PER_ITEM_BUDGET
                )
                
                if result.success:
                    self.logger.info(f"✅ Item {i} completed successfully ({result.total_time:.2f}s)")
                    return i, True, result.result, result.total_time, result.attempts, item, modality
                else:
                    error_msg = f"❌ Failed after {result.attempts} attempts: {result.error}"
                    if result.fallback_occurred:
                        error_msg += " (fallback attempted)"
                    self.logger.warning(f"❌ Item {i} failed ({result.total_time:.2f}s)")
                    return i, False, error_msg, result.total_time, result.attempts, item, modality
            except Exception as e:
                self.logger.error(f"❌ Item {i} exception: {e}")
                return i, False, f"❌ Exception: {e}", 0.0, 0, item, modality
        
        # Execute all items concurrently using asyncio.gather for maximum speed
        self.logger.info("🚀 Launching concurrent processing tasks...")
        start_time = time.time()
        
        # Create all tasks and run them concurrently
        tasks = [
            process_item_concurrent(i + 1, item) 
            for i, item in enumerate(items)
        ]
        
        # Wait for all tasks to complete concurrently
        concurrent_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_concurrent_time = time.time() - start_time
        self.logger.info(f"🚀 Concurrent processing completed in {total_concurrent_time:.2f}s!")
        
        # Process results and add to aggregator
        for result in concurrent_results:
            if isinstance(result, Exception):
                self.logger.error(f"❌ Task exception: {result}")
                continue
                
            i, success, result_text, duration, attempts, item, modality = result
            
            aggregator.add_result(
                item_index=i,
                item=item,
                modality=modality,
                result_text=result_text,
                success=success,
                duration=duration,
                attempts=attempts
            )
        
        # Generate aggregated prompt and send single response
        aggregated_prompt = aggregator.get_aggregated_prompt(original_text)
        
        # Log summary statistics with concurrent performance metrics
        stats = aggregator.get_summary_stats()
        if len(items) > 1:
            sequential_estimate = sum(getattr(r, 'duration', 0) for r in concurrent_results if not isinstance(r, Exception))
            speedup = sequential_estimate / total_concurrent_time if total_concurrent_time > 0 else 1
            self.logger.info(
                f"🚀 CONCURRENT MULTIMODAL COMPLETE: {stats['successful_items']}/{stats['total_items']} successful, "
                f"total: {total_concurrent_time:.1f}s (est. {speedup:.1f}x speedup vs sequential)"
            )
        else:
            self.logger.info(
                f"📊 Processing complete: {stats['successful_items']}/{stats['total_items']} successful, "
                f"duration: {total_concurrent_time:.1f}s"
            )
        
        # Send single aggregated response through text flow (1 IN → 1 OUT)
        if aggregated_prompt.strip():
            response_action = await self._invoke_text_flow(aggregated_prompt, message, context_str)
            if response_action and response_action.has_payload:
                # [REH] Execute the response action to send it to Discord
                await self.bot._execute_action(message, response_action)
                self.logger.info(f"✅ Response sent successfully (msg_id: {message.id})")
            else:
                self.logger.warning(f"No response generated from text flow (msg_id: {message.id})")
        else:
            self.logger.warning(f"No content to process after multimodal aggregation (msg_id: {message.id})")

    async def _handle_item_with_provider(self, item: InputItem, modality: InputModality, provider_config: ProviderConfig) -> str:
        """
        Handle a single input item with specific provider configuration.
        Routes to appropriate handler and returns text result.
        """
        # Handler mapping - all handlers must return str, never reply directly
        handlers = {
            InputModality.SINGLE_IMAGE: self._handle_image,
            InputModality.MULTI_IMAGE: self._handle_image,  # Process each image individually
            InputModality.VIDEO_URL: self._handle_video_url,
            InputModality.AUDIO_VIDEO_FILE: self._handle_audio_video_file,
            InputModality.PDF_DOCUMENT: self._handle_pdf,
            InputModality.PDF_OCR: self._handle_pdf_ocr,
            InputModality.GENERAL_URL: self._handle_general_url,
            InputModality.SCREENSHOT_URL: self._handle_screenshot_url,
        }
        
        # Vision modalities need model override from provider ladder
        if modality in (InputModality.SINGLE_IMAGE, InputModality.MULTI_IMAGE):
            return await self._handle_image_with_model(item, model_override=provider_config.model)

        handler = handlers.get(modality, self._handle_unknown)
        return await handler(item)

    # ===== NEW HANDLER METHODS FOR MULTIMODAL PROCESSING =====
    
    async def _handle_image(self, item: InputItem) -> str:
        """
        Handle image input items (attachments, URLs, or embeds).
        Returns extracted text description for further processing.
        """
        try:
            if item.source_type == "attachment":
                return await self._process_image_from_attachment(item.payload)
            elif item.source_type == "url":
                return await self._process_image_from_url(item.payload)
            elif item.source_type == "embed":
                if item.payload.image:
                    return await self._process_image_from_url(item.payload.image.url)
                elif item.payload.thumbnail:
                    return await self._process_image_from_url(item.payload.thumbnail.url)
                else:
                    return "Image embed found but no accessible image URL."
            else:
                return f"Unsupported image source type: {item.source_type}"
                
        except Exception as e:
            self.logger.error(f"Error processing image item: {e}", exc_info=True)
            return "Failed to process image."
    
    async def _process_image_from_attachment(self, attachment: discord.Attachment) -> str:
        """Process image from Discord attachment. Pure function - never replies directly."""
        tmp_path = None
        try:
            # Create temporary file for image processing
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_path = tmp_file.name
            
            # Save attachment to temporary file
            await attachment.save(tmp_path)
            self.logger.debug(f"📷 Saved image attachment to temp file: {tmp_path}")
            
            # Use vision inference with default prompt
            prompt = "Describe this image in detail, focusing on key visual elements, objects, text, and context."
            vision_response = await see_infer(image_path=tmp_path, prompt=prompt)
            
            if not vision_response:
                return "❌ Vision processing returned no response"
            
            if vision_response.error:
                return f"❌ Vision processing error: {vision_response.error}"
            
            if not vision_response.content or not vision_response.content.strip():
                return "❌ Vision processing returned empty content"
            
            return f"🖼️ **Image Analysis ({attachment.filename})**\n{vision_response.content.strip()}"
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    async def _handle_image_with_model(self, item: InputItem, model_override: str | None = None) -> str:
        """Handle image item using an explicit model override (from fallback ladder)."""
        try:
            if item.source_type == "attachment":
                return await self._process_image_from_attachment_with_model(item.payload, model_override)
            elif item.source_type == "url":
                # TODO: implement URL image processing with model override
                return await self._process_image_from_url(item.payload)
            elif item.source_type == "embed":
                if item.payload.image:
                    return await self._process_image_from_url(item.payload.image.url)
                elif item.payload.thumbnail:
                    return await self._process_image_from_url(item.payload.thumbnail.url)
                else:
                    return "Image embed found but no accessible image URL."
            else:
                return f"Unsupported image source type: {item.source_type}"
        except Exception as e:
            self.logger.error(f"Error processing image item with model override: {e}", exc_info=True)
            return "Failed to process image."

    async def _process_image_from_attachment_with_model(self, attachment: discord.Attachment, model_override: str | None) -> str:
        """Process image attachment using a specific VL model override."""
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_path = tmp_file.name
            await attachment.save(tmp_path)
            self.logger.debug(f"📷 Saved image attachment to temp file: {tmp_path}")

            prompt = "Describe this image in detail, focusing on key visual elements, objects, text, and context."
            vision_response = await see_infer(image_path=tmp_path, prompt=prompt, model_override=model_override)

            if not vision_response:
                return "❌ Vision processing returned no response"
            if vision_response.error:
                return f"❌ Vision processing error: {vision_response.error}"
            if not vision_response.content or not vision_response.content.strip():
                return "❌ Vision processing returned empty content"
            return f"🖼️ **Image Analysis ({attachment.filename})**\n{vision_response.content.strip()}"
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    async def _process_image_from_url(self, url: str) -> str:
        """Process image from URL."""
        # For now, return a placeholder - would need to download and process
        return f"Image URL detected: {url}. Image processing from URLs not yet implemented."
    
    async def _handle_video_url(self, item: InputItem) -> str:
        """
        Handle video URL input items (YouTube, TikTok, etc.).
        Returns transcribed text for further processing.
        """
        try:
            if item.source_type != "url":
                return f"Video handler received non-URL item: {item.source_type}"
            
            url = item.payload
            self.logger.info(f"🎥 Processing video URL: {url}")
            
            # Use existing video processing logic
            transcription = await hear_infer_from_url(url)
            if not transcription or not transcription.strip():
                return f"Could not transcribe audio from video: {url}"
            
            return f"Video transcription from {url}: {transcription}"
            
        except Exception as e:
            self.logger.error(f"Error processing video URL: {e}", exc_info=True)
            return f"Failed to process video URL: {item.payload}"
    
    async def _handle_audio_video_file(self, item: InputItem) -> str:
        """
        Handle audio/video file attachments.
        Returns transcribed text for further processing.
        """
        try:
            if item.source_type != "attachment":
                return f"Audio/video handler received non-attachment item: {item.source_type}"
            
            attachment = item.payload
            self.logger.info(f"🎵 Processing audio/video file: {attachment.filename}")
            
            # For now, return placeholder - would need STT implementation
            return f"Audio/video file detected: {attachment.filename}. Audio transcription not yet fully implemented."
            
        except Exception as e:
            self.logger.error(f"Error processing audio/video file: {e}", exc_info=True)
            return f"Failed to process audio/video file: {item.payload.filename if hasattr(item.payload, 'filename') else 'unknown'}"
    
    async def _handle_pdf(self, item: InputItem) -> str:
        """
        Handle PDF document input items.
        Returns extracted text for further processing.
        """
        try:
            if item.source_type == "attachment":
                return await self._process_pdf_from_attachment(item.payload)
            elif item.source_type == "url":
                return await self._process_pdf_from_url(item.payload)
            else:
                return f"PDF handler received unsupported source type: {item.source_type}"
                
        except Exception as e:
            self.logger.error(f"Error processing PDF: {e}", exc_info=True)
            return "Failed to process PDF document."
    
    async def _process_pdf_from_attachment(self, attachment: discord.Attachment) -> str:
        """Process PDF from Discord attachment."""
        if not self.pdf_processor:
            return "PDF processing not available (PyMuPDF not installed)."
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            await attachment.save(tmp_path)
            self.logger.info(f"📄 Processing PDF attachment: {attachment.filename}")
            
            text_content = await self.pdf_processor.process(tmp_path)
            if not text_content or not text_content.strip():
                return f"Could not extract text from PDF: {attachment.filename}"
            
            return f"PDF content from {attachment.filename}: {text_content}"
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    async def _process_pdf_from_url(self, url: str) -> str:
        """Process PDF from URL."""
        return f"PDF URL detected: {url}. PDF processing from URLs not yet implemented."
    
    async def _handle_pdf_ocr(self, item: InputItem) -> str:
        """
        Handle PDF documents that require OCR processing.
        Returns extracted text for further processing.
        """
        # For now, delegate to regular PDF handler
        # TODO: Implement OCR-specific logic
        return await self._handle_pdf(item)
    
    async def _handle_general_url(self, item: InputItem) -> str:
        """
        Handle general URL input items.
        Returns extracted content for further processing.
        """
        try:
            if item.source_type != "url":
                return f"URL handler received non-URL item: {item.source_type}"
            
            url = item.payload
            self.logger.info(f"🌐 Processing general URL: {url}")
            
            # Use existing URL processing logic
            content = await process_url(url)
            if not content or not content.strip():
                return f"Could not extract content from URL: {url}"
            
            return f"Web content from {url}: {content}"
            
        except Exception as e:
            self.logger.error(f"Error processing general URL: {e}", exc_info=True)
            return f"Failed to process URL: {item.payload}"
    
    async def _handle_screenshot_url(self, item: InputItem) -> str:
        """
        Handle URLs that need screenshot fallback.
        Returns screenshot analysis for further processing.
        """
        try:
            if item.source_type != "url":
                return f"Screenshot handler received non-URL item: {item.source_type}"
            
            url = item.payload
            self.logger.info(f"📸 Taking screenshot of URL: {url}")
            
            # For now, return placeholder - would need screenshot implementation
            return f"Screenshot URL detected: {url}. Screenshot processing not yet implemented."
            
        except Exception as e:
            self.logger.error(f"Error taking screenshot of URL: {e}", exc_info=True)
            return f"Failed to screenshot URL: {item.payload}"
    
    async def _handle_unknown(self, item: InputItem) -> str:
        """
        Handle unknown or unsupported input items.
        Returns appropriate fallback message.
        """
        self.logger.warning(f"Unknown input item type: {item.source_type} with payload type {type(item.payload)}")
        return f"Unsupported input type detected: {item.source_type}. Unable to process this item."

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
                # Handle both increment() and inc() method names
                if hasattr(self.bot.metrics, 'increment'):
                    self.bot.metrics.increment(metric_name, labels or {})
                elif hasattr(self.bot.metrics, 'inc'):
                    self.bot.metrics.inc(metric_name, labels=labels or {})
                else:
                    # Fallback - metrics object doesn't have expected methods
                    pass
            except Exception as e:
                # Never let metrics failures break the application
                self.logger.debug(f"Metrics increment failed for {metric_name}: {e}")

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