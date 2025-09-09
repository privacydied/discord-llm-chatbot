"""
Unified media ingestion system with yt-dlp integration and graceful fallback.
Handles smart routing between media extraction and web scraping flows.
"""

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
from urllib.parse import urlparse

from .utils.logging import get_logger
from .media_capability import media_detector, ProbeResult
from .action import BotAction

logger = get_logger(__name__)

# Configuration
MAX_CONCURRENT_MEDIA_DOWNLOADS = int(os.getenv("MEDIA_MAX_CONCURRENT", "2"))
MEDIA_DOWNLOAD_TIMEOUT = int(os.getenv("MEDIA_DOWNLOAD_TIMEOUT", "60"))
MEDIA_RETRY_MAX_ATTEMPTS = int(os.getenv("MEDIA_RETRY_MAX_ATTEMPTS", "3"))
MEDIA_RETRY_BASE_DELAY = float(os.getenv("MEDIA_RETRY_BASE_DELAY", "2.0"))
MEDIA_SPEEDUP_FACTOR = float(os.getenv("MEDIA_SPEEDUP_FACTOR", "1.5"))

# Global semaphore for media download concurrency control
_media_download_semaphore = asyncio.Semaphore(MAX_CONCURRENT_MEDIA_DOWNLOADS)


@dataclass
class MediaIngestionResult:
    """Result of media ingestion attempt."""

    success: bool
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    fallback_triggered: bool = False
    source_type: str = "unknown"  # "media" or "scrape"
    processing_time_ms: Optional[float] = None


class MediaIngestionManager:
    """Manages smart media ingestion with fallback to web scraping."""

    def __init__(self, bot):
        self.bot = bot
        self.config = bot.config
        self.logger = logger
        self._retry_delays = {}  # URL -> next retry delay

        self.logger.info("✔ MediaIngestionManager initialized")

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize metadata to prevent prompt injection and ensure safe content.

        Args:
            metadata: Raw metadata from media extraction

        Returns:
            Sanitized metadata safe for LLM context
        """
        if not metadata:
            return {}

        sanitized = {}

        # Define safe fields and their max lengths
        safe_fields = {
            "title": 200,
            "uploader": 100,
            "source": 50,
            "duration_seconds": None,  # Numeric, no length limit
            "upload_date": 20,
            "url": 500,
        }

        for field, max_length in safe_fields.items():
            if field in metadata:
                value = metadata[field]

                if isinstance(value, str):
                    # Strip control characters and limit length
                    cleaned = "".join(
                        char for char in value if ord(char) >= 32 or char in "\n\t"
                    )
                    if max_length and len(cleaned) > max_length:
                        cleaned = cleaned[:max_length] + "..."
                    sanitized[field] = cleaned
                elif isinstance(value, (int, float)):
                    sanitized[field] = value
                else:
                    # Convert other types to string and sanitize
                    str_value = str(value)
                    cleaned = "".join(
                        char for char in str_value if ord(char) >= 32 or char in "\n\t"
                    )
                    if max_length and len(cleaned) > max_length:
                        cleaned = cleaned[:max_length] + "..."
                    sanitized[field] = cleaned

        return sanitized

    async def _extract_media_with_retry(
        self, url: str
    ) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Extract media with exponential backoff retry logic.

        Returns:
            Tuple of (success, result_data, error_message)
        """
        attempt = 0
        last_error = None

        while attempt < MEDIA_RETRY_MAX_ATTEMPTS:
            try:
                # Import here to avoid circular imports
                from .hear import hear_infer_from_url

                self.logger.debug(
                    f"🎵 Media extraction attempt {attempt + 1} for: {url}"
                )

                # Extract audio and transcribe
                result = await asyncio.wait_for(
                    hear_infer_from_url(url, speedup=MEDIA_SPEEDUP_FACTOR),
                    timeout=MEDIA_DOWNLOAD_TIMEOUT,
                )

                self.logger.info(f"✅ Media extraction successful for: {url}")
                return True, result, None

            except asyncio.TimeoutError:
                last_error = f"Media extraction timeout after {MEDIA_DOWNLOAD_TIMEOUT}s"
                self.logger.warning(f"⏰ {last_error} for {url}")

            except Exception as e:
                last_error = str(e)
                self.logger.warning(
                    f"❌ Media extraction attempt {attempt + 1} failed for {url}: {last_error}"
                )

                # Check if this is a "no media found" type error that should trigger immediate fallback
                error_lower = last_error.lower()
                if any(
                    phrase in error_lower
                    for phrase in [
                        "no video",
                        "unsupported url",
                        "no media",
                        "not available",
                        "private video",
                        "video unavailable",
                        "no audio",
                        "no formats",
                    ]
                ):
                    self.logger.info(
                        f"🔄 No media content found, triggering immediate fallback: {url}"
                    )
                    break  # Exit retry loop immediately for "no content" errors

            attempt += 1

            # Exponential backoff before retry
            if attempt < MEDIA_RETRY_MAX_ATTEMPTS:
                delay = MEDIA_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                self.logger.debug(f"⏳ Retrying media extraction in {delay}s...")
                await asyncio.sleep(delay)

        return False, None, last_error

    async def _process_media_path(self, url: str, message) -> MediaIngestionResult:
        """
        Process URL through media extraction path.

        Args:
            url: URL to process
            message: Discord message object

        Returns:
            MediaIngestionResult with processing outcome
        """
        start_time = time.time()

        try:
            async with _media_download_semaphore:
                self.logger.info(
                    f"🎵 Processing URL via media path: {url} (msg_id: {message.id})"
                )

                # Extract media with retry logic
                success, result_data, error_msg = await self._extract_media_with_retry(
                    url
                )

                if not success:
                    processing_time = (time.time() - start_time) * 1000
                    return MediaIngestionResult(
                        success=False,
                        error_message=error_msg,
                        source_type="media",
                        processing_time_ms=processing_time,
                    )

                # Extract transcription and metadata
                transcription = result_data.get("transcription", "")
                raw_metadata = result_data.get("metadata", {})

                # Sanitize metadata for safe LLM consumption
                sanitized_metadata = self._sanitize_metadata(raw_metadata)

                # Create enriched context for LLM
                media_context = self._build_media_context(
                    transcription, sanitized_metadata, url
                )

                processing_time = (time.time() - start_time) * 1000

                self.logger.info(
                    f"✅ Media processing completed in {processing_time:.1f}ms for: {url}"
                )

                return MediaIngestionResult(
                    success=True,
                    content=media_context,
                    metadata=sanitized_metadata,
                    source_type="media",
                    processing_time_ms=processing_time,
                )

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            error_msg = f"Media processing exception: {str(e)}"
            self.logger.error(
                f"❌ {error_msg} for {url} (msg_id: {message.id})", exc_info=True
            )

            return MediaIngestionResult(
                success=False,
                error_message=error_msg,
                source_type="media",
                processing_time_ms=processing_time,
            )

    def _build_media_context(
        self, transcription: str, metadata: Dict[str, Any], url: str
    ) -> str:
        """
        Build enriched context string from media transcription and metadata.

        Args:
            transcription: Audio transcription text
            metadata: Sanitized metadata
            url: Original URL

        Returns:
            Formatted context string for LLM processing
        """
        # Build context with available metadata
        context_parts = []

        if metadata.get("source"):
            source_info = f"User shared a {metadata['source']} video"
            if metadata.get("title"):
                source_info += f": '{metadata['title']}'"
            if metadata.get("uploader"):
                source_info += f" by {metadata['uploader']}"

            # Add duration info if available
            if metadata.get("duration_seconds"):
                duration = metadata["duration_seconds"]
                source_info += f" (Duration: {duration:.1f}s"
                if metadata.get("speedup_factor"):
                    source_info += f", processed at {metadata['speedup_factor']}x speed"
                source_info += ")"

            context_parts.append(source_info)
        else:
            context_parts.append(f"User shared a video from: {url}")

        # Add transcription
        if transcription.strip():
            context_parts.append("The following is the audio transcription:")
            context_parts.append(transcription)
        else:
            context_parts.append("No audio transcription was available.")

        return "\n\n".join(context_parts)

    async def _process_fallback_path(
        self, url: str, message, fallback_reason: str
    ) -> MediaIngestionResult:
        """
        Process URL through existing web scraping fallback path.

        Args:
            url: URL to process
            message: Discord message object
            fallback_reason: Reason for fallback

        Returns:
            MediaIngestionResult from fallback processing
        """
        start_time = time.time()

        try:
            self.logger.info(
                f"🌐 Processing URL via fallback path: {url} (reason: {fallback_reason}) (msg_id: {message.id})"
            )

            # Import here to avoid circular imports
            from . import web

            # Use existing web processing
            processed_data = await web.process_url(url)

            processing_time = (time.time() - start_time) * 1000

            # Check for errors in web processing
            if processed_data.get("error"):
                return MediaIngestionResult(
                    success=False,
                    error_message=processed_data["error"],
                    fallback_triggered=True,
                    source_type="scrape",
                    processing_time_ms=processing_time,
                )

            # Handle image processing with vision-language models (restore original flow)
            screenshot_path = processed_data.get("screenshot_path")
            text_content = processed_data.get("text")

            if screenshot_path:
                # Image processing: use vision-language model
                self.logger.info(f"🖼️ Processing image via vision-language model: {url}")

                try:
                    from .see import see_infer

                    # Use vision model to analyze the image
                    vision_result = await see_infer(
                        image_path=screenshot_path,
                        prompt=message.content or "Describe this image",
                    )

                    # Extract content from BotAction if needed
                    if hasattr(vision_result, "content"):
                        vision_content = vision_result.content
                    else:
                        vision_content = str(vision_result)

                    # Create enriched context combining vision analysis with any text
                    context_parts = [f"Image analysis from {url}:", vision_content]
                    if text_content:
                        context_parts.extend(
                            ["\nAdditional text content:", text_content]
                        )

                    enriched_content = "\n".join(context_parts)

                    self.logger.info(
                        f"✅ Vision processing completed in {processing_time:.1f}ms for: {url}"
                    )

                    return MediaIngestionResult(
                        success=True,
                        content=enriched_content,
                        metadata={
                            "fallback_reason": fallback_reason,
                            "has_vision": True,
                            "screenshot_path": screenshot_path,
                        },
                        fallback_triggered=True,
                        source_type="vision",
                        processing_time_ms=processing_time,
                    )

                except Exception as vision_error:
                    self.logger.warning(
                        f"⚠️ Vision processing failed: {vision_error}, falling back to text"
                    )
                    # Fall through to text processing

            # Text-only processing (original fallback)
            if text_content:
                content = text_content
            else:
                return MediaIngestionResult(
                    success=False,
                    error_message="No content could be extracted from URL",
                    fallback_triggered=True,
                    source_type="scrape",
                    processing_time_ms=processing_time,
                )

            self.logger.info(
                f"✅ Fallback processing completed in {processing_time:.1f}ms for: {url}"
            )

            return MediaIngestionResult(
                success=True,
                content=content,
                metadata={"fallback_reason": fallback_reason},
                fallback_triggered=True,
                source_type="scrape",
                processing_time_ms=processing_time,
            )

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            error_msg = f"Fallback processing exception: {str(e)}"
            self.logger.error(
                f"❌ {error_msg} for {url} (msg_id: {message.id})", exc_info=True
            )

            return MediaIngestionResult(
                success=False,
                error_message=error_msg,
                fallback_triggered=True,
                source_type="scrape",
                processing_time_ms=processing_time,
            )

    async def process_url_smart(self, url: str, message) -> BotAction:
        """
        Smart URL processing with media-first approach and graceful fallback.

        Args:
            url: URL to process
            message: Discord message object

        Returns:
            BotAction with processed content
        """
        try:
            self.logger.info(
                f"🧠 Smart URL processing started: {url} (msg_id: {message.id})"
            )

            # Step 1: Capability detection
            probe_result = await media_detector.is_media_capable(url)

            # Log probe result for observability
            cache_status = "cache hit" if probe_result.cached else "fresh probe"
            self.logger.info(
                f"🔍 Media capability probe: {url} -> {probe_result.is_media_capable} "
                f"({probe_result.reason}) [{cache_status}] "
                f"(msg_id: {message.id})"
            )

            # Step 2: Route based on capability
            if probe_result.is_media_capable:
                # Try media path first
                media_result = await self._process_media_path(url, message)

                if media_result.success:
                    # Media processing succeeded
                    self._log_success_metrics(url, message, media_result, probe_result)
                    return await self._create_bot_action_from_media(
                        media_result, message
                    )
                else:
                    # Media processing failed, fallback to scraping
                    fallback_reason = (
                        f"media extraction failed: {media_result.error_message}"
                    )
                    self.logger.warning(
                        f"🔄 Media processing failed, falling back to web scraping: {url} "
                        f"(reason: {fallback_reason}) (msg_id: {message.id})"
                    )

                    fallback_result = await self._process_fallback_path(
                        url, message, fallback_reason
                    )

                    if fallback_result.success:
                        self._log_fallback_metrics(
                            url, message, fallback_result, probe_result
                        )
                        return await self._create_bot_action_from_fallback(
                            fallback_result, message
                        )
                    else:
                        # Both paths failed
                        return BotAction(
                            content=f"❌ Could not process URL: {fallback_result.error_message}",
                            error=True,
                        )
            else:
                # Not media-capable, go straight to fallback
                fallback_result = await self._process_fallback_path(
                    url, message, probe_result.reason
                )

                if fallback_result.success:
                    self._log_fallback_metrics(
                        url, message, fallback_result, probe_result
                    )
                    return await self._create_bot_action_from_fallback(
                        fallback_result, message
                    )
                else:
                    return BotAction(
                        content=f"❌ Could not process URL: {fallback_result.error_message}",
                        error=True,
                    )

        except Exception as e:
            self.logger.error(
                f"❌ Smart URL processing failed unexpectedly: {e} (msg_id: {message.id})",
                exc_info=True,
            )
            return BotAction(
                content="⚠️ An unexpected error occurred while processing this URL.",
                error=True,
            )

    async def _create_bot_action_from_media(
        self, media_result: MediaIngestionResult, message
    ) -> BotAction:
        """Create BotAction from successful media processing."""
        try:
            # Import here to avoid circular imports
            from bot.brain import brain_infer

            # Get conversation context
            context_str = await self.bot.context_manager.get_context_string(message)

            # Combine media context with conversation history
            if context_str:
                full_context = (
                    f"{context_str}\n\n--- MEDIA CONTENT ---\n{media_result.content}"
                )
            else:
                full_context = media_result.content

            # Generate response using contextual brain inference if available
            if (
                hasattr(self.bot, "enhanced_context_manager")
                and self.bot.enhanced_context_manager
                and os.getenv("USE_ENHANCED_CONTEXT", "true").lower() == "true"
            ):
                try:
                    from bot.contextual_brain import contextual_brain_infer_simple

                    self.logger.debug(
                        f"🧠🎵 Using contextual brain for media analysis [msg_id={message.id}]"
                    )

                    # Enhance content with metadata for richer context
                    enhanced_content = media_result.content
                    if media_result.metadata:
                        metadata_str = "\n\n📊 Video Details:\n"
                        for key, value in media_result.metadata.items():
                            if key in [
                                "title",
                                "uploader",
                                "duration_seconds",
                                "source",
                            ]:
                                metadata_str += (
                                    f"• {key.replace('_', ' ').title()}: {value}\n"
                                )
                        enhanced_content += metadata_str

                    response_text = await contextual_brain_infer_simple(
                        message, enhanced_content, self.bot
                    )
                    return BotAction(content=response_text)

                except Exception as e:
                    self.logger.warning(
                        f"Contextual brain inference failed for media, falling back: {e}"
                    )

            # Fallback to basic brain inference
            prompt = (
                "Please summarize and discuss the key points from this media content. "
                "Provide insights, analysis, or answer any questions about the content."
            )
            return await brain_infer(prompt, context=full_context)

        except Exception as e:
            self.logger.error(
                f"Failed to create bot action from media result: {e}", exc_info=True
            )
            return BotAction(
                content="⚠️ Processed the media but failed to generate a response.",
                error=True,
            )

    async def _create_bot_action_from_fallback(
        self, fallback_result: MediaIngestionResult, message
    ) -> BotAction:
        """Create BotAction from successful fallback processing."""
        try:
            # Use existing router logic for fallback content
            # This ensures consistency with current web processing behavior

            # Check if content indicates screenshot path
            if (
                fallback_result.content
                and "Screenshot available at:" in fallback_result.content
            ):
                screenshot_path = fallback_result.content.replace(
                    "Screenshot available at: ", ""
                ).strip()

                # Route to vision flow
                from .see import see_infer

                prompt = (
                    self.bot.system_prompts.get("VL_PROMPT_FILE")
                    or "Describe this image based on the content of the URL."
                )
                vision_response = await see_infer(
                    image_path=screenshot_path, prompt=prompt
                )

                if not vision_response or vision_response.error:
                    return BotAction(
                        content="I couldn't understand the content of the URL.",
                        error=True,
                    )

                vl_content = vision_response.content
                if len(vl_content) > 1999:
                    vl_content = vl_content[:1999].rsplit("\n", 1)[0]

                # Import here to avoid circular imports
                from bot.brain import brain_infer

                final_prompt = (
                    f"User provided this URL. The content of the URL is: {vl_content}"
                )
                return await brain_infer(final_prompt)
            else:
                # Route to text flow
                context_str = await self.bot.context_manager.get_context_string(message)
                prompt = f"The user sent this URL. Here is the content:\n\n{fallback_result.content}"

                # Use router's text flow logic
                router = self.bot.router if hasattr(self.bot, "router") else None
                if router and hasattr(router, "_invoke_text_flow"):
                    return await router._invoke_text_flow(prompt, message, context_str)
                else:
                    # Fallback to direct brain inference
                    from bot.brain import brain_infer

                    full_context = (
                        f"{context_str}\n\n{prompt}" if context_str else prompt
                    )
                    return await brain_infer(full_context)

        except Exception as e:
            self.logger.error(
                f"Failed to create bot action from fallback result: {e}", exc_info=True
            )
            return BotAction(
                content="⚠️ Processed the URL but failed to generate a response.",
                error=True,
            )

    def _log_success_metrics(
        self,
        url: str,
        message,
        media_result: MediaIngestionResult,
        probe_result: ProbeResult,
    ):
        """Log success metrics for observability."""
        labels = {
            "source_type": media_result.source_type,
            "cache_hit": str(probe_result.cached).lower(),
            "domain": urlparse(url).netloc.lower(),
        }

        self._metric_inc("media_ingestion_success_total", labels)

        if media_result.processing_time_ms:
            self._metric_observe(
                "media_ingestion_duration_ms", media_result.processing_time_ms, labels
            )

    def _log_fallback_metrics(
        self,
        url: str,
        message,
        fallback_result: MediaIngestionResult,
        probe_result: ProbeResult,
    ):
        """Log fallback metrics for observability."""
        labels = {
            "source_type": fallback_result.source_type,
            "fallback_triggered": str(fallback_result.fallback_triggered).lower(),
            "probe_result": probe_result.reason,
            "domain": urlparse(url).netloc.lower(),
        }

        self._metric_inc("media_ingestion_fallback_total", labels)

        if fallback_result.processing_time_ms:
            self._metric_observe(
                "media_ingestion_duration_ms",
                fallback_result.processing_time_ms,
                labels,
            )

    def _metric_inc(self, metric_name: str, labels: Optional[Dict[str, str]] = None):
        """Increment a metric, if metrics are enabled."""
        if hasattr(self.bot, "metrics") and self.bot.metrics:
            try:
                self.bot.metrics.increment(metric_name, labels or {})
            except Exception as e:
                self.logger.warning(f"Failed to increment metric {metric_name}: {e}")

    def _metric_observe(
        self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None
    ):
        """Observe a metric value, if metrics are enabled."""
        if hasattr(self.bot, "metrics") and self.bot.metrics:
            try:
                self.bot.metrics.observe(metric_name, value, labels or {})
            except Exception as e:
                self.logger.warning(f"Failed to observe metric {metric_name}: {e}")


# Factory function for creating media ingestion manager
def create_media_ingestion_manager(bot) -> MediaIngestionManager:
    """Create and initialize media ingestion manager."""
    return MediaIngestionManager(bot)
