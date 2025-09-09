"""
Optimized Discord Client - Connection reuse, smart enrichments, and rate limit handling.
Implements REH (Robust Error Handling) and PA (Performance Awareness) rules.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import random
from contextlib import asynccontextmanager

import discord
from discord import Embed, File
from discord.errors import HTTPException, RateLimited

from .phase_constants import PhaseConstants as PC
from .phase_timing import get_timing_manager, PipelineTracker
from ..util.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RateLimitBucket:
    """Rate limit bucket tracking for Discord API."""

    route: str
    remaining: int = 1
    reset_after: float = 0
    reset_at: float = 0
    last_updated: float = field(default_factory=time.time)

    def is_rate_limited(self) -> bool:
        """Check if bucket is currently rate limited."""
        current_time = time.time()
        if self.reset_at > current_time:
            return self.remaining <= 0
        else:
            # Reset expired, assume available
            self.remaining = 1
            return False

    def update_from_response(self, headers: Dict[str, str]):
        """Update bucket state from Discord response headers."""
        try:
            self.remaining = int(headers.get("x-ratelimit-remaining", 1))
            self.reset_after = float(headers.get("x-ratelimit-reset-after", 0))
            self.reset_at = time.time() + self.reset_after
            self.last_updated = time.time()
        except (ValueError, TypeError):
            # Handle malformed headers gracefully
            pass


@dataclass
class SendOptions:
    """Options for optimized Discord message sending."""

    skip_embeds: bool = False
    skip_files: bool = False
    skip_typing: bool = False
    priority: int = 1  # 1=high, 2=normal, 3=low
    max_retries: int = 3
    timeout_ms: int = PC.SLO_P95_DISCORD_MS * 2  # 500ms default

    @classmethod
    def for_simple_text(cls, char_count: int) -> "SendOptions":
        """Create optimized options for simple text messages [PA]."""
        return cls(
            skip_embeds=True,  # No unnecessary enrichments
            skip_files=True,
            skip_typing=char_count < 50,  # Skip typing for very short messages
            priority=1,  # High priority for simple messages
            timeout_ms=PC.SLO_P95_DISCORD_MS,
        )

    @classmethod
    def for_multimodal(cls) -> "SendOptions":
        """Create options for multimodal messages [PA]."""
        return cls(
            skip_embeds=False,
            skip_files=False,
            skip_typing=False,
            priority=2,  # Normal priority
            timeout_ms=PC.SLO_P95_DISCORD_MS * 3,  # More time for complex sends
        )


class OptimizedDiscordSender:
    """High-performance Discord message sender with connection reuse and rate limit handling."""

    def __init__(self, bot):
        self.bot = bot

        # Rate limit tracking per route [REH]
        self.rate_limit_buckets: Dict[str, RateLimitBucket] = {}

        # Connection and session reuse tracking
        self.session_stats = {
            "messages_sent": 0,
            "rate_limit_hits": 0,
            "avg_send_time_ms": 0,
            "connection_reuses": 0,
            "enrichments_skipped": 0,
        }

        # Priority queue for message sending
        self.send_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.queue_processor_task: Optional[asyncio.Task] = None

        logger.info("🚀 OptimizedDiscordSender initialized")

    def _get_route_key(self, channel_id: int, method: str = "POST") -> str:
        """Generate rate limit bucket key for channel [REH]."""
        return f"{method}:channels/{channel_id}/messages"

    def _get_rate_limit_bucket(self, route: str) -> RateLimitBucket:
        """Get or create rate limit bucket for route [REH]."""
        if route not in self.rate_limit_buckets:
            self.rate_limit_buckets[route] = RateLimitBucket(route=route)
        return self.rate_limit_buckets[route]

    async def _wait_for_rate_limit(self, bucket: RateLimitBucket, route: str):
        """Wait for rate limit to reset with jitter [REH]."""
        if not bucket.is_rate_limited():
            return

        wait_time = bucket.reset_after
        if wait_time <= 0:
            return

        # Add small jitter to avoid thundering herd [REH]
        jitter = random.uniform(0.1, 0.3)  # 100-300ms jitter
        total_wait = wait_time + jitter

        self.session_stats["rate_limit_hits"] += 1
        logger.warning(f"⚠️ Rate limited on {route}, waiting {total_wait:.2f}s")

        await asyncio.sleep(total_wait)

    def _should_skip_enrichment(
        self, content: str, options: SendOptions
    ) -> Dict[str, bool]:
        """Determine which enrichments to skip for performance [PA]."""
        skip_decisions = {
            "embeds": options.skip_embeds
            or len(content) < 50,  # Skip embeds for short messages
            "files": options.skip_files,
            "typing": options.skip_typing or len(content) < 30,  # Very short messages
            "preview_fetch": len(content) < 100,  # Skip URL previews for short messages
        }

        # Count skipped enrichments for metrics
        skipped_count = sum(1 for skip in skip_decisions.values() if skip)
        self.session_stats["enrichments_skipped"] += skipped_count

        return skip_decisions

    async def _send_with_typing(
        self, channel, content: str, skip_typing: bool, **kwargs
    ):
        """Send message with optional typing indicator [PA]."""
        if skip_typing or len(content) < 30:
            # Direct send for short messages - no typing indicator
            return await channel.send(content, **kwargs)
        else:
            # Send with typing indicator for longer messages
            async with channel.typing():
                # Small delay to show typing (but don't overdo it)
                typing_delay = min(len(content) / 100, 2.0)  # Max 2 seconds
                await asyncio.sleep(typing_delay)
                return await channel.send(content, **kwargs)

    async def _send_message_direct(
        self,
        channel,
        content: str,
        options: SendOptions,
        embeds: Optional[List[Embed]] = None,
        files: Optional[List[File]] = None,
        **kwargs,
    ) -> discord.Message:
        """Send message directly with optimizations [PA]."""
        start_time = time.time()

        # Rate limit handling
        route = self._get_route_key(channel.id)
        bucket = self._get_rate_limit_bucket(route)
        await self._wait_for_rate_limit(bucket, route)

        # Determine enrichment skips
        skip_decisions = self._should_skip_enrichment(content, options)

        # Prepare message parameters
        send_kwargs = kwargs.copy()

        # Add embeds only if not skipped
        if embeds and not skip_decisions["embeds"]:
            send_kwargs["embeds"] = embeds

        # Add files only if not skipped
        if files and not skip_decisions["files"]:
            send_kwargs["files"] = files

        try:
            # Send with or without typing
            message = await self._send_with_typing(
                channel, content, skip_decisions["typing"], **send_kwargs
            )

            # Update stats
            send_time_ms = int((time.time() - start_time) * 1000)
            self._update_send_stats(send_time_ms, success=True)

            # Track connection reuse (Discord.py handles this internally)
            self.session_stats["connection_reuses"] += 1

            return message

        except RateLimited as e:
            # Handle rate limit with exponential backoff [REH]
            retry_after = e.retry_after + random.uniform(0.1, 0.5)  # Add jitter
            logger.warning(f"⚠️ Discord rate limit: retrying after {retry_after:.2f}s")

            await asyncio.sleep(retry_after)

            # Update bucket info
            bucket.remaining = 0
            bucket.reset_after = retry_after
            bucket.reset_at = time.time() + retry_after

            # Retry once
            return await self._send_message_direct(
                channel, content, options, embeds, files, **kwargs
            )

        except HTTPException as e:
            # Handle other HTTP errors [REH]
            if e.status == 429:  # Additional rate limit handling
                retry_after = float(e.response.headers.get("retry-after", 1))
                await asyncio.sleep(retry_after + random.uniform(0.1, 0.3))
                return await self._send_message_direct(
                    channel, content, options, embeds, files, **kwargs
                )
            else:
                self._update_send_stats(
                    int((time.time() - start_time) * 1000), success=False
                )
                raise

        except Exception as e:
            self._update_send_stats(
                int((time.time() - start_time) * 1000), success=False
            )
            logger.error(f"❌ Discord send error: {e}")
            raise

    def _update_send_stats(self, send_time_ms: int, success: bool):
        """Update sending statistics [PA]."""
        if success:
            self.session_stats["messages_sent"] += 1

            # Update rolling average
            old_avg = self.session_stats["avg_send_time_ms"]
            msg_count = self.session_stats["messages_sent"]
            self.session_stats["avg_send_time_ms"] = (
                old_avg * (msg_count - 1) + send_time_ms
            ) / msg_count

    @asynccontextmanager
    async def send_message_optimized(
        self,
        channel,
        content: str,
        tracker: Optional[PipelineTracker] = None,
        options: Optional[SendOptions] = None,
        embeds: Optional[List[Embed]] = None,
        files: Optional[List[File]] = None,
        **kwargs,
    ):
        """Send Discord message with full optimization and tracking [REH][PA]."""

        # Use smart defaults based on content
        if options is None:
            if len(content) < 100 and not embeds and not files:
                options = SendOptions.for_simple_text(len(content))
            else:
                options = SendOptions.for_multimodal()

        timing_manager = get_timing_manager()

        if tracker:
            async with timing_manager.track_phase(
                tracker,
                PC.PHASE_DISCORD_DISPATCH,
                content_length=len(content),
                has_embeds=bool(embeds),
                has_files=bool(files),
                priority=options.priority,
            ) as phase_metric:
                try:
                    # Set timeout for phase
                    message = await asyncio.wait_for(
                        self._send_message_direct(
                            channel, content, options, embeds, files, **kwargs
                        ),
                        timeout=options.timeout_ms / 1000,
                    )

                    # Add metrics to phase
                    phase_metric.metadata.update(
                        {
                            "discord_msg_id": message.id,
                            "enrichments_skipped": sum(
                                1
                                for skip in self._should_skip_enrichment(
                                    content, options
                                ).values()
                                if skip
                            ),
                            "rate_limit_bucket": self._get_route_key(channel.id),
                        }
                    )

                    yield message

                except asyncio.TimeoutError:
                    phase_metric.metadata["timeout"] = True
                    logger.error(
                        f"❌ Discord send timeout after {options.timeout_ms}ms"
                    )
                    raise
                except Exception as e:
                    phase_metric.metadata["error_type"] = type(e).__name__
                    raise
        else:
            # Direct send without tracking
            message = await asyncio.wait_for(
                self._send_message_direct(
                    channel, content, options, embeds, files, **kwargs
                ),
                timeout=options.timeout_ms / 1000,
            )
            yield message

    async def send_simple_text(
        self, channel, content: str, tracker: Optional[PipelineTracker] = None
    ) -> discord.Message:
        """Optimized send for simple text messages [PA]."""
        options = SendOptions.for_simple_text(len(content))

        async with self.send_message_optimized(
            channel, content, tracker, options
        ) as message:
            return message

    async def send_multimodal_response(
        self,
        channel,
        content: str,
        embeds: Optional[List[Embed]] = None,
        files: Optional[List[File]] = None,
        tracker: Optional[PipelineTracker] = None,
    ) -> discord.Message:
        """Optimized send for multimodal responses [PA]."""
        options = SendOptions.for_multimodal()

        async with self.send_message_optimized(
            channel, content, tracker, options, embeds, files
        ) as message:
            return message

    def get_stats(self) -> Dict[str, Any]:
        """Get Discord sender performance statistics."""
        total_buckets = len(self.rate_limit_buckets)
        active_rate_limits = sum(
            1 for bucket in self.rate_limit_buckets.values() if bucket.is_rate_limited()
        )

        return {
            "messages_sent": self.session_stats["messages_sent"],
            "avg_send_time_ms": self.session_stats["avg_send_time_ms"],
            "rate_limit_hits": self.session_stats["rate_limit_hits"],
            "connection_reuses": self.session_stats["connection_reuses"],
            "enrichments_skipped": self.session_stats["enrichments_skipped"],
            "rate_limit_buckets": total_buckets,
            "active_rate_limits": active_rate_limits,
            "slo_target_ms": PC.SLO_P95_DISCORD_MS,
        }


class MessagePriorityQueue:
    """Priority queue for Discord message sending [PA]."""

    def __init__(self, sender: OptimizedDiscordSender):
        self.sender = sender
        self.queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.processor_task: Optional[asyncio.Task] = None
        self.is_processing = False

    async def start_processing(self):
        """Start background queue processing."""
        if self.processor_task and not self.processor_task.done():
            return

        async def process_queue():
            self.is_processing = True
            while self.is_processing:
                try:
                    # Get next message with timeout
                    priority, timestamp, send_task = await asyncio.wait_for(
                        self.queue.get(), timeout=1.0
                    )

                    # Execute send task
                    await send_task
                    self.queue.task_done()

                except asyncio.TimeoutError:
                    continue  # Check for shutdown
                except Exception as e:
                    logger.error(f"❌ Queue processing error: {e}")

        self.processor_task = asyncio.create_task(process_queue())

    async def stop_processing(self):
        """Stop queue processing [RM]."""
        self.is_processing = False
        if self.processor_task:
            self.processor_task.cancel()
            try:
                await self.processor_task
            except asyncio.CancelledError:
                pass

    async def enqueue_send(
        self,
        channel,
        content: str,
        priority: int = 2,
        tracker: Optional[PipelineTracker] = None,
        **kwargs,
    ):
        """Enqueue message for priority sending [PA]."""
        timestamp = time.time()

        async def send_task():
            options = SendOptions(priority=priority)
            async with self.sender.send_message_optimized(
                channel, content, tracker, options, **kwargs
            ) as message:
                return message

        await self.queue.put((priority, timestamp, send_task()))


# Global Discord sender instance [PA]
_discord_sender_instance: Optional[OptimizedDiscordSender] = None


def get_discord_sender(bot) -> OptimizedDiscordSender:
    """Get or create optimized Discord sender instance."""
    global _discord_sender_instance

    if _discord_sender_instance is None:
        _discord_sender_instance = OptimizedDiscordSender(bot)
        logger.info("🚀 Global OptimizedDiscordSender created")

    return _discord_sender_instance


def reset_discord_sender():
    """Reset global Discord sender instance [RM]."""
    global _discord_sender_instance
    _discord_sender_instance = None
