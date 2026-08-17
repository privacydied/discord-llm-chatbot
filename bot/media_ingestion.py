"""Unified media ingestion system with yt-dlp integration and graceful fallback.
Handles smart routing between media extraction and web scraping flows.
"""

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any, Never

from .action import BotAction
from .config import _low_resource_int
from .media_capability import ProbeResult, media_detector
from .media_ingestion_helpers import build_media_context, sanitize_metadata
from .utils.logging import get_logger

logger = get_logger(__name__)

# Resource caps for multimodal ingestion [Phase 12-16]
_MULTIMODAL_MAX_ITEMS = _low_resource_int("MULTIMODAL_MAX_ITEMS", 5, 2)
_MULTIMODAL_MAX_TOTAL_BYTES = _low_resource_int("MULTIMODAL_MAX_TOTAL_BYTES", 50 * 1024 * 1024, 10 * 1024 * 1024)

try:
    from .hear import hear_infer_from_url
except ImportError:  # pragma: no cover - compatibility fallback

    async def hear_infer_from_url(*_args, **_kwargs) -> Never:  # type: ignore[override]
        msg = "hear_infer_from_url unavailable"
        raise RuntimeError(msg)


try:
    from .brain import brain_infer
except ImportError:  # pragma: no cover - compatibility fallback

    async def brain_infer(*_args, **_kwargs) -> Never:  # type: ignore[override]
        msg = "brain_infer unavailable"
        raise RuntimeError(msg)


try:
    from .contextual_brain import contextual_brain_infer_simple
except ImportError:  # pragma: no cover - compatibility fallback

    async def contextual_brain_infer_simple(*_args, **_kwargs) -> Never:  # type: ignore[override]
        msg = "contextual_brain_infer_simple unavailable"
        raise RuntimeError(msg)


try:
    from .see import see_infer
except ImportError:  # pragma: no cover - compatibility fallback

    async def see_infer(*_args, **_kwargs) -> Never:  # type: ignore[override]
        msg = "see_infer unavailable"
        raise RuntimeError(msg)


# Configuration
MAX_CONCURRENT_MEDIA_DOWNLOADS = int(os.getenv("MEDIA_MAX_CONCURRENT", "2"))
MEDIA_DOWNLOAD_TIMEOUT = int(os.getenv("MEDIA_DOWNLOAD_TIMEOUT", "60"))
MEDIA_RETRY_MAX_ATTEMPTS = int(os.getenv("MEDIA_RETRY_MAX_ATTEMPTS", "3"))
MEDIA_RETRY_BASE_DELAY = float(os.getenv("MEDIA_RETRY_BASE_DELAY", "2.0"))
MEDIA_SPEEDUP_FACTOR = float(os.getenv("MEDIA_SPEEDUP_FACTOR", "1.5"))

# Global semaphore for media download concurrency control
_media_download_semaphore = asyncio.Semaphore(MAX_CONCURRENT_MEDIA_DOWNLOADS)


def _apply_multimodal_caps(items, *, max_items: int = _MULTIMODAL_MAX_ITEMS, max_total_bytes: int = _MULTIMODAL_MAX_TOTAL_BYTES):
    """Apply resource caps to a list of attachment-like items.

    Returns (capped_items, clipped_count) where clipped_count is how many
    items were removed due to the cap.  Size is estimated via a 'size' attr
    or len() fallback.
    """
    capped: list = []
    running_bytes = 0
    for item in items:
        if len(capped) >= max_items:
            break
        try:
            item_size = getattr(item, "size", None)
            if item_size is None:
                item_size = len(item) if hasattr(item, "__len__") else 0
        except (TypeError, AttributeError):
            item_size = 0
        if running_bytes + item_size > max_total_bytes:
            break
        capped.append(item)
        running_bytes += item_size
    clipped = len(items) - len(capped)
    return capped, clipped


@dataclass
class MediaIngestionResult:
    """Result of media ingestion attempt."""

    success: bool
    content: str | None = None
    metadata: dict[str, Any] | None = None
    error_message: str | None = None
    fallback_triggered: bool = False
    source_type: str = "unknown"  # "media" or "scrape"
    processing_time_ms: float | None = None


class MediaIngestionManager:
    """Manages smart media ingestion with fallback to web scraping."""

    def __init__(self, bot) -> None:
        self.bot = bot
        self.config = bot.config
        self.logger = logger
        self._retry_delays = {}  # URL -> next retry delay

        self.logger.info("✔ MediaIngestionManager initialized")

    def _sanitize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Sanitize metadata to prevent prompt injection and ensure safe content.

        Args:
            metadata: Raw metadata from media extraction

        Returns:
            Sanitized metadata safe for LLM context
        """
        return sanitize_metadata(metadata)

    async def _extract_media_with_retry(self, url: str) -> tuple[bool, dict | None, str | None]:
        """Extract media with exponential backoff retry logic.

        Returns:
            Tuple of (success, result_data, error_message)
        """
        attempt = 0
        last_error = None

        while attempt < MEDIA_RETRY_MAX_ATTEMPTS:
            try:
                self.logger.debug(f"🎵 Media extraction attempt {attempt + 1} for: {url}")

                # Extract audio and transcribe
                result = await asyncio.wait_for(
                    hear_infer_from_url(url),
                    timeout=MEDIA_DOWNLOAD_TIMEOUT,
                )

                self.logger.info(f"✅ Media extraction successful for: {url}")
                return True, result, None

            except TimeoutError:
                last_error = f"Media extraction timeout after {MEDIA_DOWNLOAD_TIMEOUT}s"
                self.logger.warning(f"⏰ {last_error} for {url}")

            except Exception as e:
                # Boundary contract: this helper NEVER raises — it returns
                # (False, None, error) so callers can fall back gracefully.
                # A lint-driven sweep (3abb61e) narrowed this to typed
                # exceptions, silently excluding the ones hear_infer_from_url
                # actually raises (InferenceError, VideoIngestError,
                # NoAudioStreamError), so real STT failures escaped the retry
                # loop and crashed the caller instead. [REH]
                last_error = str(e)
                self.logger.warning(f"❌ Media extraction attempt {attempt + 1} failed for {url}: {last_error}")

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
                    self.logger.info(f"🔄 No media content found, triggering immediate fallback: {url}")
                    break  # Exit retry loop immediately for "no content" errors

            attempt += 1

            # Exponential backoff before retry
            if attempt < MEDIA_RETRY_MAX_ATTEMPTS:
                delay = MEDIA_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                self.logger.debug(f"⏳ Retrying media extraction in {delay}s...")
                await asyncio.sleep(delay)

        return False, None, last_error

    async def _process_media_path(self, url: str, message) -> MediaIngestionResult:
        """Process URL through media extraction path.

        Args:
            url: URL to process
            message: Discord message object

        Returns:
            MediaIngestionResult with processing outcome
        """
        start_time = time.time()

        try:
            async with _media_download_semaphore:
                self.logger.info(f"🎵 Processing URL via media path: {url} (msg_id: {message.id})")

                # Extract media with retry logic
                success, result_data, error_msg = await self._extract_media_with_retry(url)

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
                media_context = self._build_media_context(transcription, sanitized_metadata, url)

                processing_time = (time.time() - start_time) * 1000

                self.logger.info(f"✅ Media processing completed in {processing_time:.1f}ms for: {url}")

                return MediaIngestionResult(
                    success=True,
                    content=media_context,
                    metadata=sanitized_metadata,
                    source_type="media",
                    processing_time_ms=processing_time,
                )

        except (AttributeError, TypeError, ValueError, RuntimeError, OSError) as e:
            processing_time = (time.time() - start_time) * 1000
            error_msg = f"Media processing exception: {e!s}"
            self.logger.error(f"❌ {error_msg} for {url} (msg_id: {message.id})", exc_info=True)

            return MediaIngestionResult(
                success=False,
                error_message=error_msg,
                source_type="media",
                processing_time_ms=processing_time,
            )

    def _build_media_context(self, transcription: str, metadata: dict[str, Any], url: str) -> str:
        """Build enriched context string from media transcription and metadata.

        Args:
            transcription: Audio transcription text
            metadata: Sanitized metadata
            url: Original URL

        Returns:
            Formatted context string for LLM processing
        """
        return build_media_context(transcription, metadata, url)

    async def _process_fallback_path(self, url: str, message, fallback_reason: str) -> MediaIngestionResult:
        """Process URL through existing web scraping fallback path.

        Args:
            url: URL to process
            message: Discord message object
            fallback_reason: Reason for fallback

        Returns:
            MediaIngestionResult from fallback processing
        """
        start_time = time.time()

        try:
            self.logger.info(f"🌐 Processing URL via fallback path: {url} (reason: {fallback_reason}) (msg_id: {message.id})")

            # Import here to avoid circular imports
            # Import tiered extractor lazily to avoid widening import surface
            from . import web, web_extraction_service

            # Use existing web processing
            processed_data = await web.process_url(url)

            processing_time = (time.time() - start_time) * 1000

            # Handle image processing with vision-language models (restore original flow)
            screenshot_path = processed_data.get("screenshot_path")
            text_content = processed_data.get("text")

            # If legacy scraping failed or produced no usable text, try tiered extractor once.
            try:
                needs_tiered = bool(processed_data.get("error")) or (not screenshot_path and not (text_content and str(text_content).strip()))
            except (KeyError, AttributeError, TypeError):
                needs_tiered = True

            if needs_tiered:
                try:
                    extract_res = await web_extraction_service.web_extractor.extract(url)
                except (AttributeError, TypeError, ValueError, RuntimeError) as e:
                    extract_res = None
                    self.logger.debug(f"Tiered extractor exception for {url}: {e}", exc_info=True)

                if extract_res is not None and getattr(extract_res, "success", False):
                    from bot.url_safety import wrap_untrusted_content

                    wrapped = wrap_untrusted_content(
                        extract_res.to_message(),
                        source=extract_res.canonical_url or url,
                    )
                    content = f"Web content from {extract_res.canonical_url or url}:\n{wrapped}"
                    return MediaIngestionResult(
                        success=True,
                        content=content,
                        metadata={
                            "fallback_reason": fallback_reason,
                            "tier_used": getattr(extract_res, "tier_used", None),
                        },
                        fallback_triggered=True,
                        source_type="scrape",
                        processing_time_ms=processing_time,
                    )

                # Bot-wall failure: surface the specific, actionable message
                # instead of the generic "No content could be extracted" text. [PAY]
                if extract_res is not None and getattr(extract_res, "bot_wall_marker", None) is not None:
                    return MediaIngestionResult(
                        success=False,
                        error_message=extract_res.to_message(),
                        fallback_triggered=True,
                        source_type="scrape",
                        processing_time_ms=processing_time,
                    )

                if processed_data.get("error"):
                    return MediaIngestionResult(
                        success=False,
                        error_message=processed_data["error"],
                        fallback_triggered=True,
                        source_type="scrape",
                        processing_time_ms=processing_time,
                    )

            if screenshot_path:
                return MediaIngestionResult(
                    success=True,
                    content=f"Screenshot available at: {screenshot_path}",
                    metadata={
                        "fallback_reason": fallback_reason,
                        "screenshot_path": screenshot_path,
                    },
                    fallback_triggered=True,
                    source_type="scrape",
                    processing_time_ms=processing_time,
                )

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

            self.logger.info(f"✅ Fallback processing completed in {processing_time:.1f}ms for: {url}")

            return MediaIngestionResult(
                success=True,
                content=content,
                metadata={"fallback_reason": fallback_reason},
                fallback_triggered=True,
                source_type="scrape",
                processing_time_ms=processing_time,
            )

        except (AttributeError, TypeError, ValueError, RuntimeError, OSError) as e:
            processing_time = (time.time() - start_time) * 1000
            error_msg = f"Fallback processing exception: {e!s}"
            self.logger.error(f"❌ {error_msg} for {url} (msg_id: {message.id})", exc_info=True)

            return MediaIngestionResult(
                success=False,
                error_message=error_msg,
                fallback_triggered=True,
                source_type="scrape",
                processing_time_ms=processing_time,
            )

    async def process_url_smart(self, url: str, message) -> BotAction:
        """Smart URL processing with media-first approach and graceful fallback.

        Args:
            url: URL to process
            message: Discord message object

        Returns:
            BotAction with processed content
        """
        try:
            self.logger.info(f"🧠 Smart URL processing started: {url} (msg_id: {message.id})")

            # Step 1: Capability detection
            probe_result = await media_detector.is_media_capable(url)

            # Log probe result for observability
            cache_status = "cache hit" if probe_result.cached else "fresh probe"
            self.logger.info(f"🔍 Media capability probe: {url} -> {probe_result.is_media_capable} ({probe_result.reason}) [{cache_status}] (msg_id: {message.id})")

            # Step 2: Route based on capability
            if probe_result.is_media_capable:
                # Try media path first
                media_result = await self._process_media_path(url, message)

                if media_result.success:
                    # Media processing succeeded
                    self._log_success_metrics(url, message, media_result, probe_result)
                    return await self._create_bot_action_from_media(media_result, message)
                # Media processing failed, fallback to scraping
                fallback_reason = f"media extraction failed: {media_result.error_message}"
                self.logger.warning(f"🔄 Media processing failed, falling back to web scraping: {url} (reason: {fallback_reason}) (msg_id: {message.id})")

                fallback_result = await self._process_fallback_path(url, message, fallback_reason)

                if fallback_result.success:
                    self._log_fallback_metrics(url, message, fallback_result, probe_result)
                    return await self._create_bot_action_from_fallback(fallback_result, message)
                # Both paths failed
                err = (fallback_result.error_message or "").strip()
                msg = f"Could not extract content from URL: {url} (Error: {err})" if err else f"Could not extract content from URL: {url}"
                return BotAction(
                    content=msg,
                    error=True,
                )
            # Not media-capable, go straight to fallback
            fallback_result = await self._process_fallback_path(url, message, probe_result.reason)

            if fallback_result.success:
                self._log_fallback_metrics(url, message, fallback_result, probe_result)
                return await self._create_bot_action_from_fallback(fallback_result, message)
            err = (fallback_result.error_message or "").strip()
            msg = f"Could not extract content from URL: {url} (Error: {err})" if err else f"Could not extract content from URL: {url}"
            return BotAction(
                content=msg,
                error=True,
            )

        except (AttributeError, TypeError, ValueError, RuntimeError, OSError) as e:
            self.logger.error(
                f"❌ Smart URL processing failed unexpectedly: {e} (msg_id: {message.id})",
                exc_info=True,
            )
            return BotAction(
                content="⚠️ An unexpected error occurred while processing this URL.",
                error=True,
            )

    async def _create_bot_action_from_media(self, media_result: MediaIngestionResult, message) -> BotAction:
        """Create BotAction from successful media processing."""
        try:
            # Get conversation context
            context_str = await self.bot.context_manager.get_context_string(message)

            # Combine media context with conversation history
            full_context = f"{context_str}\n\n--- MEDIA CONTENT ---\n{media_result.content}" if context_str else media_result.content

            # Generate response using contextual brain inference if available
            if hasattr(self.bot, "enhanced_context_manager") and self.bot.enhanced_context_manager and os.getenv("USE_ENHANCED_CONTEXT", "true").lower() == "true":
                try:
                    self.logger.debug(f"🧠🎵 Using contextual brain for media analysis [msg_id={message.id}]")

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
                                metadata_str += f"• {key.replace('_', ' ').title()}: {value}\n"
                        enhanced_content += metadata_str

                    response_text = await contextual_brain_infer_simple(message, enhanced_content, self.bot)
                    return BotAction(content=response_text)

                except (AttributeError, TypeError, ValueError, RuntimeError, OSError) as e:
                    self.logger.warning(f"Contextual brain inference failed for media, falling back: {e}")

            # Fallback to basic brain inference
            prompt = "Please summarize and discuss the key points from this media content. Provide insights, analysis, or answer any questions about the content."
            return await brain_infer(prompt, context=full_context)

        except (AttributeError, TypeError, ValueError, RuntimeError, OSError) as e:
            self.logger.error(f"Failed to create bot action from media result: {e}", exc_info=True)
            return BotAction(
                content="⚠️ Processed the media but failed to generate a response.",
                error=True,
            )

    async def _create_bot_action_from_fallback(self, fallback_result: MediaIngestionResult, message) -> BotAction:
        """Create BotAction from successful fallback processing."""
        try:
            # Use existing router logic for fallback content
            # This ensures consistency with current web processing behavior

            # Check if content indicates screenshot path
            if fallback_result.content and "Screenshot available at:" in fallback_result.content:
                screenshot_path = fallback_result.content.replace("Screenshot available at: ", "").strip()

                prompt = (getattr(self.bot, "system_prompts", {}) or {}).get("VL_PROMPT_FILE") or "Describe this image based on the content of the URL."
                vision_response = await see_infer(image_path=screenshot_path, prompt=prompt)

                if not vision_response or vision_response.error:
                    return BotAction(
                        content="I couldn't understand the content of the URL.",
                        error=True,
                    )

                vl_content = vision_response.content
                if len(vl_content) > 1999:
                    vl_content = vl_content[:1999].rsplit("\n", 1)[0]

                final_prompt = f"User provided this URL. The content of the URL is: {vl_content}"
                return await brain_infer(final_prompt)
            # Route to text flow
            context_str = await self.bot.context_manager.get_context_string(message)
            prompt = f"The user sent this URL. Here is the content:\n\n{fallback_result.content}"

            # Use router's text flow logic
            router = self.bot.router if hasattr(self.bot, "router") else None
            if router and hasattr(router, "_invoke_text_flow"):
                maybe = router._invoke_text_flow(prompt, message, context_str)
                if asyncio.iscoroutine(maybe):
                    return await maybe
                if isinstance(maybe, BotAction):
                    return maybe
            # Fallback to direct brain inference
            full_context = f"{context_str}\n\n{prompt}" if context_str else prompt
            return await brain_infer(full_context)

        except (AttributeError, TypeError, ValueError, RuntimeError, OSError) as e:
            self.logger.error(f"Failed to create bot action from fallback result: {e}", exc_info=True)
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
    ) -> None:
        """Log success metrics for observability."""
        pass  # Placeholder for metrics

    def _log_fallback_metrics(
        self,
        url: str,
        message,
        fallback_result: MediaIngestionResult,
        probe_result: ProbeResult,
    ) -> None:
        """Log fallback metrics for observability."""
        pass  # Placeholder for metrics

    def _metric_inc(self, metric_name: str, labels: dict[str, str] | None = None) -> None:
        """Increment a metric, if metrics are enabled."""
        if hasattr(self.bot, "metrics") and self.bot.metrics:
            try:
                self.bot.metrics.increment(metric_name, labels or {})
            except (AttributeError, TypeError, ValueError, RuntimeError) as e:
                self.logger.warning(f"Failed to increment metric {metric_name}: {e}")

    def _metric_observe(self, metric_name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Observe a metric value, if metrics are enabled."""
        if hasattr(self.bot, "metrics") and self.bot.metrics:
            try:
                self.bot.metrics.observe(metric_name, value, labels or {})
            except (AttributeError, TypeError, ValueError, RuntimeError) as e:
                self.logger.warning(f"Failed to observe metric {metric_name}: {e}")


# Factory function for creating media ingestion manager
def create_media_ingestion_manager(bot) -> MediaIngestionManager:
    """Create and initialize media ingestion manager."""
    return MediaIngestionManager(bot)
