"""Core bot implementation for Discord LLM Chatbot."""

from __future__ import annotations

import asyncio
import io
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING, Any

import discord
from discord.ext import commands
from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree

from bot.config import load_system_prompts
from bot.config_reload import add_reload_callback
from bot.enhanced_retry import get_retry_manager
from bot.events import setup_command_error_handler
from bot.http_client import cleanup_http_client
from bot.memory import (
    enqueue_inferred_memory,
    load_all_profiles,
    start_memory_distiller,
    start_memory_service,
    stop_memory_distiller,
    stop_memory_service,
)
from bot.memory.context_manager import ContextManager
from bot.memory.enhanced_context_manager import EnhancedContextManager
from bot.memory.thread_tail import (
    _is_thread_channel,
    resolve_thread_reply_target,
)
from bot.metrics import NullMetrics
from bot.public_output import (
    sanitize_embed_for_public,
    sanitize_public_message_payload,
    sanitize_public_text,
)
from bot.server_archive import start_server_archive_service, stop_server_archive_service
from bot.tts.errors import SynthesisError
from bot.utils.logging import get_logger
from bot.voice import VoiceMessagePublisher

from .message_processor import MessageProcessor
from .text_chunking import (
    DISCORD_ERR_INVALID_FORM_BODY,
    DISCORD_ERR_UNKNOWN_MESSAGE,
    DISCORD_MAX_CONTENT_LEN,
    fence_wrap_markers,
    split_for_discord,
)

# Re-exported from the shared splitter; kept as module names because tests and
# call sites already import them from here. [CA]
# Discord hard limit is 2000; 1950 leaves headroom for mentions/overhead [REH][PA]
_DISCORD_MAX_CONTENT_LEN = DISCORD_MAX_CONTENT_LEN

# These two codes were previously conflated under one branch labelled "Unknown
# Message", which is 10008 -- 50035 is Invalid Form Body, what Discord returns for
# an oversize payload. The mislabel meant the deleted-trigger-message fallback
# never fired, and an oversize send was retried with the identical body. [CMV][REH]
_DISCORD_ERR_UNKNOWN_MESSAGE = DISCORD_ERR_UNKNOWN_MESSAGE
_DISCORD_ERR_INVALID_FORM_BODY = DISCORD_ERR_INVALID_FORM_BODY

if TYPE_CHECKING:
    from bot.router import BotAction, Router
    from bot.tts import TTSManager


def log_commands_setup(
    console: Console,
    command_modules: list[tuple[str, bool]],
    command_cogs: list[tuple[str, bool]],
    total_commands: int,
) -> None:
    """Log command setup progress using Rich's Tree and Panel for visual reporting.

    This function creates a structured visual report of the command setup process,
    showing the status of module imports and cog registrations in a tree format.

    Args:
        console: Rich Console instance for output
        command_modules: List of (module_name, success_status) tuples for imports
        command_cogs: List of (cog_name, success_status) tuples for cog registration
        total_commands: Total number of commands registered across all cogs

    The output includes:
    - A root node titled "🎬 Commands Setup"
    - A branch "📦 Import modules" with ✅/❌ status for each module
    - A branch "⚙️ Load cogs" with ✅/❌ status for each cog
    - A summary with success/failure counts and total command count

    """
    # Create the main tree structure
    tree = Tree("🎬 Commands Setup")

    # Add modules branch
    modules_branch = tree.add("📦 Import modules")
    modules_success = 0
    modules_failed = 0

    for module_name, module_ok in command_modules:
        status_icon = "✅" if module_ok else "❌"
        modules_branch.add(f"{status_icon} {module_name}")
        if module_ok:
            modules_success += 1
        else:
            modules_failed += 1

    # Add cogs branch
    cogs_branch = tree.add("⚙️ Load cogs")
    cogs_success = 0
    cogs_failed = 0

    for cog_name, cog_ok in command_cogs:
        status_icon = "✅" if cog_ok else "❌"
        cogs_branch.add(f"{status_icon} {cog_name}")
        if cog_ok:
            cogs_success += 1
        else:
            cogs_failed += 1

    # Add summary branch
    total_success = modules_success + cogs_success
    total_failed = modules_failed + cogs_failed

    summary_branch = tree.add("📊 Summary")
    summary_branch.add(f"🎉 Complete: {total_success} loaded, {total_failed} failed")
    summary_branch.add(f"📋 Total commands registered: {total_commands}")

    # Create panel and print
    panel = Panel(tree, title="Command Setup Report", border_style="blue", padding=(1, 2))

    console.print(panel)


class LLMBot(commands.Bot):
    """Main bot class that extends the base Bot class with LLM capabilities."""

    def __init__(self, *args, config: dict | None = None, **kwargs) -> None:
        # Provide sensible defaults for tests if not supplied
        if "command_prefix" not in kwargs:
            kwargs["command_prefix"] = os.getenv("COMMAND_PREFIX", "!")
        if "intents" not in kwargs:
            try:
                intents = discord.Intents.none()
            except (AttributeError, TypeError):
                intents = None
            kwargs["intents"] = intents

        super().__init__(*args, **kwargs)
        self.config = config or {}
        owner_ids = self.config.get("OWNER_IDS", [])
        try:
            self.owner_ids = {int(owner_id) for owner_id in owner_ids}
        except (ValueError, TypeError):
            self.owner_ids = set()
        self.logger = get_logger(__name__)
        self.metrics = NullMetrics()
        self.user_profiles = {}
        self.server_profiles = {}
        self.memory_save_task = None
        self.tts_manager: TTSManager | None = None
        self.archive_service = None
        self.router: Router | None = None
        self._background_tasks: set[asyncio.Task] = set()
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._is_ready = asyncio.Event()
        # Gateway watchdog state: last time we had a confirmed live connection
        # (initial connect, RESUME, or READY). See _gateway_watchdog(). [REH]
        self._last_gateway_ok: float = time.monotonic()
        self.system_prompts = {}
        self.message_processor = None  # set in setup_hook
        self._typing_suppressed_until: dict[int, float] = {}

        # Track active long-running tasks for cancellation
        self._active_long_running_tasks: dict[str, asyncio.Task] = {}  # task_id -> task
        self._task_metadata: dict[str, dict[str, Any]] = {}  # task_id -> metadata
        self._task_lock = asyncio.Lock()  # [BUGFIX] prevent callback race on task tracking dicts

        # Idempotency guard to prevent duplicate initialization [DRY][REH]
        self._boot_completed = False

        # Rich console for enhanced command setup logging
        self.console = Console()

        self.context_manager = ContextManager(
            self,
            filepath=self.config.get("CONTEXT_FILE_PATH", "context.json"),
            max_messages=self.config.get("MAX_CONTEXT_MESSAGES", 10),
        )
        # Enhanced context manager for multi-user conversation tracking
        self.enhanced_context_manager = EnhancedContextManager(
            self,
            filepath=self.config.get("ENHANCED_CONTEXT_FILE_PATH", "enhanced_context.json"),
            history_window=int(os.getenv("HISTORY_WINDOW", "10")),
            max_token_limit=self.config.get("MAX_CONTEXT_TOKENS", 4000),
        )
        self._public_output_safety_installed = False

    def _track_background_task(self, task: asyncio.Task) -> None:
        """Register a background task so it is tracked and cancelled on shutdown."""
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

        def _log_exception(t: asyncio.Task) -> None:
            try:
                exc = t.exception()
            except asyncio.CancelledError:
                return  # cancellation is normal
            except asyncio.InvalidStateError:
                return  # task did not raise
            if exc is not None:
                task_name = t.get_name() if hasattr(t, "get_name") else "unnamed"
                self.logger.warning(
                    f"Background task '{task_name}' raised: {exc}",
                    exc_info=exc,
                )

        task.add_done_callback(_log_exception)

    def _install_public_output_safety_hooks(self) -> None:
        """Patch Discord send/edit boundaries so public text is sanitized everywhere."""
        if self._public_output_safety_installed:
            return

        try:
            from discord.abc import Messageable

            def _sanitize_payload(args, kwargs):
                args_list = list(args)
                content = args_list[0] if args_list else kwargs.get("content", None)
                embed = kwargs.get("embed", None)
                embeds = kwargs.get("embeds", None)
                sanitized_content, sanitized_embed, sanitized_embeds = sanitize_public_message_payload(
                    content,
                    embed=embed,
                    embeds=embeds,
                )
                if args_list:
                    args_list[0] = sanitized_content
                elif "content" in kwargs:
                    kwargs["content"] = sanitized_content

                # Discord rejects payloads that specify both singular and plural
                # embed arguments. Normalize to exactly one representation here.
                has_embed = embed is not None or "embed" in kwargs
                has_embeds = embeds is not None or "embeds" in kwargs
                if has_embeds and sanitized_embeds:
                    kwargs.pop("embed", None)
                    kwargs["embeds"] = sanitized_embeds
                elif has_embed:
                    kwargs.pop("embeds", None)
                    kwargs["embed"] = sanitized_embed
                elif has_embeds:
                    kwargs["embeds"] = sanitized_embeds
                return tuple(args_list), kwargs

            def _wrap_method(owner, attr_name: str) -> None:
                original = getattr(owner, attr_name)
                sentinel = f"_public_output_{attr_name}_wrapped"
                if getattr(owner, sentinel, False):
                    return

                async def wrapper(self_obj, *args, **kwargs):
                    new_args, new_kwargs = _sanitize_payload(args, kwargs)
                    return await original(self_obj, *new_args, **new_kwargs)

                setattr(owner, attr_name, wrapper)
                setattr(owner, sentinel, True)

            _wrap_method(Messageable, "send")
            _wrap_method(discord.Message, "reply")
            _wrap_method(discord.Message, "edit")
            _wrap_method(discord.InteractionResponse, "send_message")
            _wrap_method(discord.InteractionResponse, "edit_message")
            _wrap_method(discord.Interaction, "edit_original_response")
            _wrap_method(discord.Webhook, "send")

            self._public_output_safety_installed = True
            self.logger.info("✅ Public output send/edit safety hooks installed")
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            self.logger.warning(
                f"⚠️ Failed to install public output safety hooks: {exc}",
                exc_info=True,
            )

    def _is_retryable_discord_http_error(self, error: Exception) -> bool:
        """Return True for transient Discord transport or upstream failures."""
        if not isinstance(error, discord.HTTPException):
            return False

        status = getattr(error, "status", None)
        if status == 429:
            return True
        if isinstance(status, int) and status >= 500:
            return True

        details = str(error).lower()
        transient_markers = (
            "service unavailable",
            "upstream connect error",
            "disconnect/reset before headers",
            "reset reason: overflow",
        )
        return any(marker in details for marker in transient_markers)

    def _discord_retry_delay(self, error: discord.HTTPException, attempt: int) -> float:
        """Compute a bounded retry delay using Retry-After when Discord provides one."""
        retry_after = getattr(error, "retry_after", None)
        if retry_after is None:
            response = getattr(error, "response", None)
            headers = getattr(response, "headers", None)
            if headers:
                raw_retry_after = headers.get("retry-after") or headers.get("Retry-After")
                if raw_retry_after is not None:
                    try:
                        retry_after = float(raw_retry_after)
                    except (TypeError, ValueError):
                        retry_after = None

        if retry_after is not None:
            return max(0.0, min(float(retry_after), 5.0))

        return min(0.5 * (2 ** (attempt - 1)), 2.0)

    async def _call_with_discord_retry(
        self,
        operation: str,
        func,
        *,
        base_extra: dict[str, Any] | None = None,
        attempts: int = 3,
    ):
        """Retry transient Discord HTTP failures with short bounded backoff."""
        extra = base_extra or {}

        for attempt in range(1, attempts + 1):
            try:
                return await func()
            except discord.HTTPException as exc:
                if not self._is_retryable_discord_http_error(exc) or attempt >= attempts:
                    raise

                delay = self._discord_retry_delay(exc, attempt)
                with suppress(Exception):
                    self.logger.warning(
                        f"discord.retry | op={operation} attempt={attempt}/{attempts} status={getattr(exc, 'status', 'n/a')} delay={delay:.2f}s details={exc!s}",
                        extra={**extra, "event": "discord.retry", "op": operation},
                    )
                await asyncio.sleep(delay)
        return None

    @asynccontextmanager
    async def _optional_typing(
        self,
        channel,
        *,
        base_extra: dict[str, Any] | None = None,
        enabled: bool = True,
    ):
        """Enter typing() when available, but don't fail the send path if it errors."""
        if not enabled:
            yield
            return

        typing_factory = getattr(channel, "typing", None)
        channel_id = getattr(channel, "id", None)
        channel_key = channel_id if channel_id is not None else id(channel)
        now = time.monotonic()

        suppressed_until = self._typing_suppressed_until.get(channel_key, 0.0)
        if now < suppressed_until:
            yield
            return

        if not callable(typing_factory):
            yield
            return

        ctx = None
        entered = False
        try:
            ctx = typing_factory()
            await ctx.__aenter__()
            entered = True
        except discord.HTTPException as exc:
            if getattr(exc, "status", None) == 429:
                self._typing_suppressed_until[channel_key] = now + 300.0
            else:
                self._typing_suppressed_until[channel_key] = now + 60.0
            with suppress(Exception):
                self.logger.warning(
                    "discord.typing.skip | enter_failed",
                    extra={
                        "event": "discord.typing.skip",
                        "channel_id": channel_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "status": getattr(exc, "status", None),
                    },
                )
            yield
            return
        except (discord.HTTPException, discord.NotFound, discord.Forbidden, asyncio.TimeoutError) as exc:
            self._typing_suppressed_until[channel_key] = now + 60.0
            with suppress(Exception):
                self.logger.warning(
                    "discord.typing.skip | enter_failed",
                    extra={
                        "event": "discord.typing.skip",
                        "channel_id": channel_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
            yield
            return

        try:
            yield
        finally:
            if entered and ctx is not None:
                with suppress(Exception):
                    await ctx.__aexit__(None, None, None)

    async def process_commands(self, message: discord.Message) -> Any | None:
        """Short-circuit non-command messages before invoking Discord's command pipeline.
        Prevents CommandNotFound surfacing for plain text that should be routed elsewhere.
        """
        content = (getattr(message, "content", "") or "").strip()
        try:
            prefixes = await self.get_prefix(message)
        except (discord.HTTPException, discord.NotFound, discord.Forbidden, AttributeError):
            prefixes = self.command_prefix
        if prefixes is None:
            prefix_list: list[str] = []
        elif isinstance(prefixes, (list, tuple)):
            prefix_list = [str(p) for p in prefixes if p is not None]
        else:
            prefix_list = [str(prefixes)]
        if not any(content.startswith(p) for p in prefix_list if p):
            return None

        ctx = await self.get_context(message)
        if not getattr(ctx, "command", None):
            return None

        try:
            return await super().process_commands(message)
        except commands.errors.CommandNotFound:
            return None

    async def setup_hook(self) -> None:
        """Asynchronous setup phase for the bot."""
        # Defer torch_compat shims here so torch is NOT loaded at import time.
        # Previously this ran in bot/__init__.py, pulling in ~600 modules / ~300 MB RSS
        # during every import (including tests, config load, etc.). [IV][REH]
        try:
            from bot.utils.torch_compat import ensure_reduce_op_alias

            ensure_reduce_op_alias()
            self.logger.debug("torch compat shim applied")
        except (ImportError, AttributeError, RuntimeError) as e:
            self.logger.debug(f"torch compat shim not available (fail-open): {e}")

        # Prevent duplicate initialization [DRY][REH]
        if self._boot_completed:
            self.logger.debug("🔄 Setup hook called but boot already completed, skipping")
            return

        self._boot_completed = True
        # Save the running event loop for thread-safe reload callbacks [PHASE2]
        self._event_loop = asyncio.get_running_loop()
        self.logger.info("🔧 Starting bot setup")
        self._install_public_output_safety_hooks()

        # Instantiate MessageProcessor for per-user queue + dedup + orchestration
        self.message_processor = MessageProcessor(self)

        try:
            # Initialize metrics
            try:
                import os

                # Read Prometheus configuration from environment
                prometheus_enabled = os.getenv("PROMETHEUS_ENABLED", "true").lower() == "true"
                prometheus_port = int(os.getenv("PROMETHEUS_PORT", "8000"))
                prometheus_http_server = os.getenv("PROMETHEUS_HTTP_SERVER", "true").lower() == "true"

                if prometheus_enabled:
                    from bot.metrics.prometheus_metrics import PrometheusMetrics

                    self.metrics = PrometheusMetrics(port=prometheus_port, enable_http_server=prometheus_http_server)
                    self.logger.info("✅ Prometheus metrics initialized")
                else:
                    from bot.metrics.null_metrics import NoopMetrics

                    self.metrics = NoopMetrics()
                    self.logger.info("📊 Prometheus disabled via config, using NoopMetrics")
            except (ImportError, AttributeError, ValueError, OSError):
                self.logger.warning("⚠️  Prometheus metrics not available, using NullMetrics")

            # Proactively register gate counters to avoid 'not defined' warnings [PA][REH][CMV]
            try:
                if hasattr(self.metrics, "define_counter"):
                    # Both counters use a 'reason' label
                    self.metrics.define_counter(
                        "gate.allowed",
                        "Messages allowed by SSOT gate",
                        labels=["reason"],
                    )
                    self.metrics.define_counter(
                        "gate.blocked",
                        "Messages blocked by SSOT gate",
                        labels=["reason"],
                    )
                    # X photo→VL routing counters [CMV][REH]
                    # Note on labels:
                    # - .attempt/.success/.failure use an index label (idx)
                    # - .skipped uses an 'enabled' label to indicate false
                    # - .enabled and .no_urls are simple counters (no labels)
                    self.metrics.define_counter(
                        "x.photo_to_vl.enabled",
                        "X photos routed to VL (feature enabled)",
                    )
                    self.metrics.define_counter(
                        "x.photo_to_vl.no_urls",
                        "X photo routing: no photo URLs available",
                    )
                    self.metrics.define_counter(
                        "x.photo_to_vl.skipped",
                        "X photo routing skipped",
                        labels=["enabled"],
                    )
                    self.metrics.define_counter(
                        "x.photo_to_vl.attempt",
                        "X photo routing attempts",
                        labels=["idx"],
                    )
                    self.metrics.define_counter(
                        "x.photo_to_vl.success",
                        "X photo routing success",
                        labels=["idx"],
                    )
                    self.metrics.define_counter(
                        "x.photo_to_vl.failure",
                        "X photo routing failure",
                        labels=["idx"],
                    )
                    # X syndication tier counters [CMV][REH][PA]
                    # Label schema mirrors Router._get_tweet_via_syndication() usage
                    self.metrics.define_counter(
                        "x.syndication.fetch",
                        "Syndication fetch attempts",
                        labels=["endpoint"],
                    )
                    self.metrics.define_counter(
                        "x.syndication.non_200",
                        "Syndication non-200 responses",
                        labels=["status", "endpoint"],
                    )
                    self.metrics.define_counter(
                        "x.syndication.invalid_json",
                        "Syndication: invalid JSON payload",
                        labels=["endpoint"],
                    )
                    self.metrics.define_counter("x.syndication.success", "Syndication: successful retrieval")
                    self.metrics.define_counter("x.syndication.error", "Syndication: unexpected exception")
                    self.metrics.define_counter(
                        "x.syndication.invalid",
                        "Syndication: structurally invalid response",
                    )
                    self.metrics.define_counter("x.syndication.neg_store", "Syndication: negative cache store")
                    self.metrics.define_counter("x.syndication.cache_hit", "Syndication: positive cache hit")
                    self.metrics.define_counter("x.syndication.neg_cache_hit", "Syndication: negative cache hit")
                    self.metrics.define_counter(
                        "x.syndication.cache_hit_locked",
                        "Syndication: positive cache hit (within lock)",
                    )
                    self.metrics.define_counter(
                        "x.syndication.neg_cache_hit_locked",
                        "Syndication: negative cache hit (within lock)",
                    )
                    self.metrics.define_counter(
                        "x.syndication.hit",
                        "Syndication: final tier hit (produced text)",
                    )
                    # Vision routing counters [CMV][REH]
                    self.metrics.define_counter(
                        "vision.route.vl_only_bypass_t2i",
                        "VL-only bypass of text-to-image",
                        labels=["route"],
                    )
                    self.metrics.define_counter(
                        "vision.route.direct",
                        "Direct vision route triggers",
                        labels=["stage"],
                    )
                    self.metrics.define_counter(
                        "vision.route.intent",
                        "Vision route intent processing",
                        labels=["route"],
                    )
                    self.metrics.define_counter(
                        "vision.route.blocked",
                        "Vision route blocked",
                        labels=["reason", "path"],
                    )
                    self.metrics.define_counter(
                        "vision.route.conversational_edit",
                        "Conversational (mention/reply) image-edit invocations",
                        labels=["outcome"],
                    )
                    # X/Vision image-only-tweet counters [CMV][REH]
                    self.metrics.define_counter(
                        "x.tweet_image_only.syndication",
                        "Image-only tweet detected via syndication",
                        labels=["photos", "source"],
                    )
                    self.metrics.define_counter(
                        "x.tweet_image_only.api",
                        "Image-only tweet detected via API",
                        labels=["photos"],
                    )
                    self.metrics.define_counter(
                        "vision.image_only_tweet.start",
                        "Image-only tweet: Vision/OCR processing started",
                        labels=["source", "images"],
                    )
                    self.metrics.define_counter(
                        "vision.image_only_tweet.success",
                        "Image-only tweet: per-image analysis success",
                        labels=["image_idx"],
                    )
                    self.metrics.define_counter(
                        "vision.image_only_tweet.failure",
                        "Image-only tweet: per-image analysis unavailable",
                        labels=["image_idx"],
                    )
                    self.metrics.define_counter(
                        "vision.image_only_tweet.error",
                        "Image-only tweet: per-image analysis raised an exception",
                        labels=["image_idx"],
                    )
                    self.metrics.define_counter(
                        "vision.image_only_tweet.complete",
                        "Image-only tweet: processing complete",
                        labels=["source", "images", "ocr_found", "safety_flags"],
                    )
                    self.metrics.define_counter(
                        "vision.route.skipped",
                        "Vision route skipped",
                        labels=["reason"],
                    )
                    self.metrics.define_counter(
                        "vision.route.dry_run",
                        "Vision route dry-run short-circuit",
                        labels=["path"],
                    )
                    self.metrics.define_counter(
                        "vision.intent.error",
                        "Vision intent routing raised an exception",
                    )
                    # Inline search provider counters [CMV][REH]
                    self.metrics.define_counter(
                        "inline_search.start",
                        "Inline search provider invocation started",
                        labels=["category", "provider"],
                    )
                    self.metrics.define_counter(
                        "inline_search.success",
                        "Inline search provider invocation succeeded",
                        labels=["category", "provider"],
                    )
                    self.metrics.define_counter(
                        "inline_search.error",
                        "Inline search provider invocation raised an exception",
                        labels=["category", "provider"],
                    )
                    # Input-item routing precedence counters [CMV][REH]
                    self.metrics.define_counter(
                        "routing.vision.precedence",
                        "Vision precedence check short-circuited routing",
                        labels=["stage"],
                    )
                    self.metrics.define_counter(
                        "routing.twitter.thumb_suppressed",
                        "Twitter/X thumbnail attachment suppressed from routing",
                    )
                    self.metrics.define_counter(
                        "routing.url.precedence.selected",
                        "URL items selected under routing precedence",
                        labels=["count"],
                    )
                    self.metrics.define_counter(
                        "routing.vl.default_bare_image.selected",
                        "Bare image attachments selected for default VL routing",
                        labels=["count"],
                    )
                    self.metrics.define_counter(
                        "vision.image_only_tweet.fatal_error",
                        "Image-only tweet: processing failed with an unhandled exception",
                        labels=["source"],
                    )
                    # Ambient reply gate counters [CMV][REH]
                    self.metrics.define_counter(
                        "ambient_reply_fired_total",
                        "Ambient (unprompted) replies fired",
                    )
                    self.metrics.define_counter(
                        "ambient_reply_suppressed_total",
                        "Ambient reply gate evaluated but suppressed",
                        labels=["reason"],
                    )
                    self.metrics.define_counter(
                        "ambient_context_build_total",
                        "Context mode used when building an ambient-reply prompt",
                        labels=["mode"],
                    )
                    self.logger.debug(
                        "📈 Registered gate counters",
                        extra={
                            "event": "metrics.define",
                            "counters": ["gate.allowed", "gate.blocked"],
                        },
                    )
            except (AttributeError, TypeError, ValueError):
                # Never allow metrics registration failure to impact startup
                self.logger.debug("Metrics counter registration failed")

            # Load system prompts
            self.system_prompts = load_system_prompts()
            self.logger.info("✅ Loaded system prompts")

            # Register config hot-reload callback to atomically swap live config and prompts [REH]
            try:

                def _on_config_reload(old_cfg: dict[str, Any], new_cfg: dict[str, Any]) -> None:
                    """Thread-safe config reload shim: schedules mutation onto the event loop."""
                    loop = self._event_loop
                    if loop is None or loop.is_closed():
                        self.logger.warning("Config reload skipped: event loop not running")
                        return
                    asyncio.run_coroutine_threadsafe(_apply_config_reload(old_cfg, new_cfg), loop)

                async def _apply_config_reload(old_cfg: dict[str, Any], new_cfg: dict[str, Any]) -> None:
                    try:
                        # Swap live config snapshot
                        self.config = dict(new_cfg)
                        # Refresh system prompts to follow updated PROMPT_FILE/VL_PROMPT_FILE
                        self.system_prompts = load_system_prompts()
                        # Ensure router sees the new snapshot
                        if getattr(self, "router", None) is not None:
                            with suppress(Exception):
                                self.router.config = self.config
                        # Scoped re-init based on changed keys
                        try:
                            changed_keys = set()
                            ok = old_cfg or {}
                            nk = new_cfg or {}
                            for k in set(ok.keys()) | set(nk.keys()):
                                if ok.get(k) != nk.get(k):
                                    changed_keys.add(str(k))
                            upper = {k.upper() for k in changed_keys}
                            # Hot-reload TTS
                            if any(k.startswith("TTS_") for k in upper):
                                try:
                                    if getattr(self, "tts_manager", None) and getattr(self.tts_manager, "_executor", None):
                                        self.tts_manager._executor.shutdown(wait=False)
                                except (AttributeError, TypeError, OSError) as e:
                                    self.logger.debug(f"TTS executor shutdown failed: {e}")
                                try:
                                    from bot.tts.interface import TTSManager

                                    self.tts_manager = TTSManager(self)
                                    self.logger.info("TTS manager hot-reloaded")
                                except (ImportError, AttributeError, TypeError, RuntimeError) as e:
                                    self.logger.exception(f"TTS hot-reload failed: {e}")
                            # Rebind Vision configs
                            if any(k.startswith(("VISION_", "VL_")) for k in upper):
                                vo = getattr(self, "vision_orchestrator", None)
                                if vo is not None:
                                    with suppress(Exception):
                                        vo.config = self.config
                                    try:
                                        if hasattr(vo, "gateway") and vo.gateway is not None:
                                            if hasattr(vo.gateway, "update_config"):
                                                vo.gateway.update_config(self.config)
                                            else:
                                                vo.gateway.config = self.config
                                                if hasattr(vo.gateway, "adapter") and vo.gateway.adapter is not None:
                                                    adapter = vo.gateway.adapter
                                                    if hasattr(adapter, "update_config"):
                                                        adapter.update_config(self.config)
                                                    else:
                                                        adapter.config = self.config
                                        self.logger.info("Vision orchestrator config rebound (hot)")
                                    except (AttributeError, TypeError, ValueError) as e:
                                        self.logger.debug(f"Vision hot-rebind failed: {e}")
                                try:
                                    retry_mgr = get_retry_manager()
                                    summary = retry_mgr.refresh_from_env()
                                    vision_order = summary.get("vision", [])
                                    head = vision_order[0] if vision_order else ""
                                    order_repr = "[" + ",".join(vision_order) + "]"
                                    self.logger.info(
                                        "vision.ladder.rebound head=%s order=%s source=env_hot",
                                        head,
                                        order_repr,
                                    )
                                except (AttributeError, TypeError, ImportError) as rebound_exc:
                                    self.logger.debug(f"Vision ladder rebound failed: {rebound_exc}")
                            # Restart shared HTTP client on HTTP/PROXY/TIMEOUT/RETRY changes
                            if any(k.startswith(("HTTP_", "PROXY_", "RETRY_")) or k.endswith("_TIMEOUT") for k in upper):
                                try:
                                    # Best-effort; next get_http_client() will start a new one with fresh config
                                    self.logger.info(
                                        "config.reload.http_client_restart",
                                        extra={
                                            "event": "config.reload.http_client_restart",
                                            "detail": {},
                                        },
                                    )

                                    # Schedule cleanup with retained ref and bounded timeout [PHASE2]
                                    async def _cleanup_with_timeout() -> None:
                                        try:
                                            async with asyncio.timeout(10.0):
                                                await cleanup_http_client()
                                        except TimeoutError:
                                            self.logger.warning(
                                                "config.reload.http_client_cleanup_timeout",
                                                extra={
                                                    "event": "config.reload.http_client_cleanup_timeout",
                                                    "detail": {"timeout": 10.0},
                                                },
                                            )
                                        except (OSError, RuntimeError) as e:
                                            self.logger.debug(f"HTTP client cleanup timeout: {e}")

                                    task = asyncio.create_task(_cleanup_with_timeout())
                                    self._track_background_task(task)
                                except (AttributeError, TypeError, RuntimeError) as e:
                                    self.logger.debug(f"HTTP client restart failed: {e}")
                            # Hot-reload dashboard config
                            if any(k.startswith("DASHBOARD_") for k in upper):
                                try:
                                    dashboard_server = getattr(self, "_dashboard_server", None)
                                    if dashboard_server is not None:
                                        from bot.dashboard.config import load_dashboard_config

                                        new_dash_cfg = load_dashboard_config()
                                        await dashboard_server.hot_reload_config(new_dash_cfg)
                                        # Audit-log the reload
                                        if dashboard_server._audit_store:
                                            await dashboard_server._audit_store.record(
                                                event_type="dashboard.config.reload",
                                                result="success",
                                                metadata={"trigger": "sighup"},
                                            )
                                    else:
                                        self.logger.debug("Dashboard hot-reload skipped: server not running")
                                except (AttributeError, TypeError, ImportError, RuntimeError) as e:
                                    self.logger.debug(f"Dashboard hot-reload failed: {e}")
                            # Breadcrumb
                            self.logger.info(
                                "config.reload.applied.bot",
                                extra={
                                    "event": "config.reload.applied.bot",
                                    "detail": {"keys": len(new_cfg or {})},
                                },
                            )
                        except (AttributeError, TypeError, ValueError) as e:
                            self.logger.exception(f"Reload callback failed: {e}")
                    except (AttributeError, TypeError, ValueError) as e:
                        self.logger.exception(f"Config reload apply failed: {e}")

                add_reload_callback(_on_config_reload)
            except (AttributeError, TypeError, ValueError):
                # Non-fatal — manual reload command still works
                self.logger.debug("Config reload callback registration failed")

            # Load user and server profiles
            await self.load_profiles()
            self.logger.info("✅ Loaded user profiles")

            # Set up background tasks
            self.setup_background_tasks()
            self.logger.info("✅ Background tasks configured")

            # Initialize TTS if configured
            await self.setup_tts()
            self.logger.info("✅ TTS system initialized")

            # Set up message router
            await self.setup_router()
            self.logger.info("✅ Message router configured")

            # Initialize RAG system (with eager loading if configured)
            await self.setup_rag()
            self.logger.info("✅ RAG system configured")

            # Load command extensions
            await self.load_extensions()
            self.logger.info("✅ Command extensions loaded")

            # Setup global command error handler
            self.command_error_handler = await setup_command_error_handler(self)
            self.logger.info("✅ Global command error handler configured")

            self._track_background_task(asyncio.create_task(self._gateway_watchdog(), name="gateway_watchdog"))

            self.logger.info("🚀 Bot setup complete")

        except Exception as e:
            self.logger.error(f"❌ Fatal error during bot setup: {e}", exc_info=True)
            self._boot_completed = False  # Reset flag on failure to allow retry
            raise

    async def on_ready(self) -> None:
        """Called when the bot is ready and connected to Discord."""
        self._last_gateway_ok = time.monotonic()
        # Simple ready state logging - all setup is handled in setup_hook() [DRY]
        if not self._is_ready.is_set():
            self.logger.info(f"🤖 Logged in as {self.user} (ID: {self.user.id})")
            self._is_ready.set()
            self.logger.info("🎉 Bot is ready to receive commands!")

    async def on_connect(self) -> None:
        """Gateway websocket established (pre-READY). Feeds the reconnect watchdog."""
        self._last_gateway_ok = time.monotonic()

    async def on_resumed(self) -> None:
        """Gateway session resumed after a drop. Feeds the reconnect watchdog."""
        self._last_gateway_ok = time.monotonic()
        self.logger.info("Gateway session resumed")

    async def _gateway_watchdog(self) -> None:
        """Self-restart if the gateway never recovers from a reconnect loop. [REH]

        discord.py's own Client.connect() loop can retry forever with growing
        backoff (see ReconnectNoiseFilter in bot/utils/logging.py) without ever
        raising -- nothing previously noticed when that loop never actually
        succeeds. Observed in production: the process stayed alive (background
        tasks kept ticking) while the gateway sat dead for ~13 hours, and even
        a manual SIGTERM couldn't cleanly close the wedged connection, forcing
        a SIGKILL. This watches time-since-last-successful-connect and, past a
        threshold, replaces the process image in place (os.execv, same PID --
        botctl's PID file keeps working) so it self-heals unattended.
        """
        interval_s = float(self.config.get("GATEWAY_WATCHDOG_INTERVAL_S", 30))
        stuck_threshold_s = float(self.config.get("GATEWAY_WATCHDOG_STUCK_S", 300))
        while not self.is_closed():
            await asyncio.sleep(interval_s)
            if self.is_closed() or self.is_ready():
                continue
            stuck_for = time.monotonic() - self._last_gateway_ok
            if stuck_for < stuck_threshold_s:
                continue
            self.logger.critical(f"Gateway stuck reconnecting for {stuck_for:.0f}s with no successful connect/resume -- self-restarting process.")
            with suppress(Exception):
                await asyncio.wait_for(self.close(), timeout=5.0)
            os.execv(sys.executable, [sys.executable, *sys.argv])  # nosec B606 - intentional exec self-restart, no shell

    def _get_user_queue(self, user_id: str) -> asyncio.Queue:
        """Compatibility shim — delegated to MessageProcessor."""
        return self.message_processor._get_queue(user_id)

    async def _ensure_user_processor(self, user_id: str) -> None:
        """Compatibility shim — delegated to MessageProcessor."""
        await self.message_processor._ensure_user_processor(user_id)

    async def _process_user_messages(self, user_id: str) -> None:
        """Compatibility shim — delegated to MessageProcessor."""
        await self.message_processor._process_user_messages(user_id)

    async def _process_single_message(self, message: discord.Message) -> None:
        """Process a single message through the full pipeline."""
        try:
            # Append message to context (both managers for backward compatibility)
            if self.context_manager:
                self.context_manager.append(message)

            # Enhanced context tracking for multi-user conversations
            if self.enhanced_context_manager:
                await self.enhanced_context_manager.append_message(message, role="user")

            # Best-effort curated memory ingestion must never block the hot path.
            try:
                is_command = await self._message_is_command(message)
            except (AttributeError, TypeError, discord.HTTPException, discord.NotFound, discord.Forbidden):
                is_command = False
            if not is_command and getattr(message, "content", "").strip():
                # No pre-filter here: MemoryCurator.curate_inferred_candidate owns the
                # raw-text policy (too_short / sensitive / internal_noise / denylist,
                # durability classification, then computed importance+confidence against
                # min_importance).  Gating here as well is what broke ingestion: the
                # scored gate was handed unscored text, so importance defaulted to 0.0
                # and every message was denied "below_threshold".  Let the curator decide.
                with suppress(Exception):
                    self._track_background_task(
                        asyncio.create_task(
                            enqueue_inferred_memory(
                                user_id=str(message.author.id),
                                text=message.content,
                                guild_id=str(message.guild.id) if getattr(message, "guild", None) else None,
                                channel_id=str(message.channel.id) if getattr(message, "channel", None) else None,
                                thread_id=str(message.channel.id) if isinstance(message.channel, discord.Thread) else None,
                                source_message_id=str(message.id),
                                metadata={"source": "bot_message_pipeline"},
                            ),
                        ),
                    )

            # Demoted from .info() -- pure per-message tracing breadcrumb, fires on every
            # single message the bot sees; dispatch:pre/attempt/ok already cover the
            # operationally-relevant signal downstream. Still emitted at DEBUG. [PA]
            guild_info = "DM" if isinstance(message.channel, discord.DMChannel) else f"guild:{message.guild.id}"
            self.logger.debug(
                " === DM MESSAGE PROCESSING STARTED ===="
                if guild_info == "DM"
                else f"Message queued: msg_id:{message.id} author:{message.author.id} in:{guild_info} len:{len(message.content)} queue_size:{self._get_user_queue(str(message.author.id)).qsize()}",
            )

            # The router decides if this is a command, a direct message, or something to ignore.
            if self.router:
                # Optional streaming status cards while the router works [CA][REH][PA]
                stream_ctx = None
                if self.config.get("STREAMING_ENABLE", False):
                    try:
                        eligible = {"eligible": False, "reason": "no_router"}
                        if hasattr(self.router, "compute_streaming_eligibility"):
                            eligible = self.router.compute_streaming_eligibility(message)  # cheap preflight
                        if eligible.get("eligible"):
                            stream_ctx = await self._start_streaming_status(message)
                        else:
                            self.logger.debug(f"stream:skipped | msg:{message.id} reason:{eligible.get('reason')} domains:{eligible.get('domains')} modality:{eligible.get('modality')}")
                    except (AttributeError, TypeError, ValueError, RuntimeError) as e:
                        self.logger.debug(f"stream:start_failed | {e}")

                action = await self.router.dispatch_message(message)
                dispatch_meta = self.router.get_dispatch_metadata(message.id) if self.router else {}
                try:
                    if action:
                        if action.meta.get("delegated_to_cog"):
                            self.logger.info(f"Message {message.id} delegated to command processor.")
                            # If streaming was started, clean it up as we hand off to cogs
                            if stream_ctx and stream_ctx.get("message"):
                                try:
                                    await stream_ctx["message"].delete()
                                except (discord.HTTPException, discord.NotFound, discord.Forbidden):
                                    # Fallback to editing the status message
                                    with suppress(Exception):
                                        await stream_ctx["message"].edit(
                                            content="",
                                            embeds=[
                                                self._build_stream_embed(
                                                    "ℹ️ Delegated to command…",
                                                    style=self.config.get(
                                                        "STREAMING_EMBED_STYLE",
                                                        "compact",
                                                    ),
                                                ),
                                            ],
                                        )
                            await self.process_commands(message)
                        elif action.has_payload:
                            # Stop streaming and mark as ready before sending the final response
                            target_msg = None
                            if stream_ctx and stream_ctx.get("task"):
                                await self._stop_streaming_status(stream_ctx)
                                target_msg = stream_ctx.get("message")
                            await self._execute_action(
                                message,
                                action,
                                target_message=target_msg,
                                dispatch_meta=dispatch_meta,
                            )
                        # If no payload and not delegated, the router decided to do nothing.
                    else:
                        if stream_ctx and stream_ctx.get("task"):
                            await self._stop_streaming_status(stream_ctx, final_label="🚫 No response generated")
                        gate_reason = self.router.pop_gate_denied_reason(message.id) if self.router else None
                        is_cmd = await self._message_is_command(message)
                        if gate_reason and not is_cmd:
                            with suppress(Exception):
                                self.logger.info(
                                    f"gate.drop | reason={gate_reason} msg_id:{message.id}",
                                    extra={
                                        "event": "gate.drop",
                                        "reason": gate_reason,
                                        "msg_id": message.id,
                                        "user_id": getattr(message.author, "id", None),
                                    },
                                )
                            return
                        if await self._message_is_command(message):
                            self.logger.info(f"Router returned no action for msg {message.id}; falling back to command processing.")
                            await self.process_commands(message)
                        else:
                            with suppress(Exception):
                                self.logger.info(
                                    f"router.noop | reason=no_route msg_id:{message.id}",
                                    extra={
                                        "event": "router.noop",
                                        "reason": "no_route",
                                        "msg_id": message.id,
                                    },
                                )
                finally:
                    if self.router:
                        self.router.clear_dispatch_metadata(message.id)
            else:
                self.logger.error("Router not initialized, falling back to command processing.")
                await self.process_commands(message)

        except Exception as e:
            from bot.exceptions import APIError as _APIError

            if isinstance(e, _APIError):
                # APIError messages are already descriptive — no traceback needed [REH]
                self.logger.warning(f"APIError in message {message.id}: {e}")
            else:
                self.logger.error(f"Error in _process_single_message for {message.id}: {e}", exc_info=True)

    def _infer_streaming_plan(self, message: discord.Message) -> list[str] | None:
        """Infer a labeled streaming plan (list of step labels) based on message content and attachments.
        Returns None if no specific plan can be inferred.
        """
        try:
            content = (message.content or "").lower().strip()
            atts = getattr(message, "attachments", []) or []

            # Helpers
            def has_ext(exts: set[str]) -> bool:
                for a in atts:
                    name = getattr(a, "filename", "").lower()
                    if any(name.endswith(ext) for ext in exts):
                        return True
                return False

            def count_ext(exts: set[str]) -> int:
                c = 0
                for a in atts:
                    name = getattr(a, "filename", "").lower()
                    if any(name.endswith(ext) for ext in exts):
                        c += 1
                return c

            IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
            VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
            AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".m4a", ".flac"}
            PDF_EXTS = {".pdf"}

            ONLINE_SEARCH = [
                "Parsing query",
                "Hitting provider",
                "Collecting results",
                "Ranking & dedupe",
                "Generating response",
            ]
            MULTI_IMAGE = [
                "Collecting images",
                "Pre-processing (hash/resize)",
                "Vision analysis",
                "Fusing context",
                "Generating response",
            ]
            VIDEO_URLS = [
                "Processing link",
                "Fetching metadata",
                "Extracting audio",
                "Transcribing audio",
                "Generating response",
            ]
            AV_FILES = [
                "Validating file",
                "Extracting audio",
                "Transcribing audio",
                "Generating response",
            ]
            SINGLE_IMAGE = [
                "Uploading/validating",
                "Vision analysis",
                "Generating response",
            ]
            PDF_DOCS = [
                "Parsing PDF",
                "Chunking pages",
                "Extracting text",
                "Summarizing",
                "Generating response",
            ]
            PDF_DOCS_OCR = [
                "Rasterizing pages",
                "OCR",
                "Text cleanup",
                "Summarizing",
                "Generating response",
            ]
            GENERAL_URLS = [
                "Fetching page",
                "Extracting content",
                "De-boilerplating",
                "Summarizing",
                "Generating response",
            ]
            RAG_BOOTSTRAP = [
                "Discovering sources",
                "Chunking",
                "Embedding",
                "Indexing",
                "Ready",
            ]
            RAG_SCAN = [
                "Scanning changes",
                "Chunking",
                "Embedding",
                "Indexing",
                "Updated",
            ]
            RAG_WIPE = [
                "Stopping jobs",
                "Dropping index",
                "Clearing cache",
                "Verifying",
                "Done",
            ]

            # Command-based plans
            if content.startswith(("!search", "[search]")):
                return ONLINE_SEARCH

            if content.startswith("!rag "):
                if " bootstrap" in content:
                    return RAG_BOOTSTRAP
                if " scan" in content:
                    return RAG_SCAN
                if " wipe" in content:
                    return RAG_WIPE

            # Attachment and URL heuristics
            img_count = count_ext(IMAGE_EXTS)
            has_pdf = has_ext(PDF_EXTS)

            # Combined plan when both images and a PDF are present
            if (img_count >= 1) and has_pdf:
                img_plan = MULTI_IMAGE if img_count >= 2 else SINGLE_IMAGE
                pdf_plan = PDF_DOCS_OCR if "ocr" in content else PDF_DOCS

                # Compose by concatenation then dedup while preserving order.
                combined = []
                seen = set()
                for step in img_plan + pdf_plan + ["Generating response"]:
                    if step == "Generating response" and (len(combined) > 0 and combined[-1] == "Generating response"):
                        continue
                    if step not in seen:
                        combined.append(step)
                        seen.add(step)
                # Ensure single final "Generating response"
                if combined and combined[-1] != "Generating response":
                    combined.append("Generating response")
                return combined

            # Image-only
            if img_count >= 2:
                return MULTI_IMAGE
            if img_count == 1:
                return SINGLE_IMAGE

            # PDF-only
            if has_pdf:
                if "ocr" in content:
                    return PDF_DOCS_OCR
                return PDF_DOCS

            if has_ext(VIDEO_EXTS) or has_ext(AUDIO_EXTS):
                return AV_FILES

            # URL-based detection in content
            if "http://" in content or "https://" in content:
                if "youtu" in content:
                    return VIDEO_URLS
                return GENERAL_URLS
        except (AttributeError, TypeError, ValueError) as e:
            self.logger.debug(f"stream:plan_infer_failed | {e}")
        return None

    async def _start_streaming_status(self, message: discord.Message) -> dict:
        """Start a streaming status card message and background updater.
        Returns a context dict with 'message' and 'task'.
        """
        # Build initial embed
        style = self.config.get("STREAMING_EMBED_STYLE", "compact")
        plan = self._infer_streaming_plan(message)
        max_steps = len(plan) if plan else int(self.config.get("STREAMING_MAX_STEPS", 8))
        first_label = plan[0] if plan else "Working…"
        initial = self._build_stream_embed(f"⏳ {first_label}", style=style, step=0, max_steps=max_steps)

        sent = await message.reply(content="", embeds=[initial], mention_author=True)
        # Track in enhanced context
        if self.enhanced_context_manager:
            await self.enhanced_context_manager.append_message(sent, role="bot")

        tick_ms = int(self.config.get("STREAMING_TICK_MS", 750))

        task = asyncio.create_task(self._streaming_updater(sent, style, tick_ms, max_steps, plan))
        return {"message": sent, "task": task, "plan": plan, "max_steps": max_steps}

    async def _stop_streaming_status(self, stream_ctx: dict, final_label: str = "✅ Generating reply…") -> None:
        """Stop the background updater and finalize the status card."""
        try:
            task = stream_ctx.get("task")
            if task and not task.done():
                task.cancel()
                with suppress(Exception):
                    await task
            msg: discord.Message = stream_ctx.get("message")
            if msg:
                style = self.config.get("STREAMING_EMBED_STYLE", "compact")
                await msg.edit(
                    content="",
                    embeds=[self._build_stream_embed(final_label, style=style, done=True)],
                )
        except (discord.HTTPException, discord.NotFound, discord.Forbidden) as e:
            self.logger.debug(f"stream:stop_failed | {e}")

    async def _streaming_updater(
        self,
        msg: discord.Message,
        style: str,
        tick_ms: int,
        max_steps: int,
        plan: list[str] | None,
    ) -> None:
        """Background loop to update the streaming status embed.
        Stops automatically after max_steps or if cancelled.
        """
        # Braille spinner frames
        frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        try:
            for i in range(max_steps):
                phase = plan[i] if plan and i < len(plan) else "Working…"
                label = f"{frames[i % len(frames)]} {phase}"
                embed = self._build_stream_embed(label, style=style, step=i + 1, max_steps=max_steps)
                await msg.edit(content="", embeds=[embed])
                await asyncio.sleep(max(0.05, tick_ms / 1000))
        except asyncio.CancelledError:
            # Normal cancellation path when finalizing
            return
        except (discord.HTTPException, discord.NotFound, discord.Forbidden, asyncio.TimeoutError) as e:
            # Swallow updater errors; streaming is best-effort
            self.logger.debug(f"stream:update_failed | {e}")

    def _build_stream_embed(
        self,
        label: str,
        *,
        style: str = "compact",
        step: int | None = None,
        max_steps: int | None = None,
        done: bool = False,
    ) -> discord.Embed:
        """Create an embed for streaming status according to style."""
        from bot.public_output import sanitize_public_text

        label = sanitize_public_text(label) or "Processing"
        color = 0x2ECC71 if done else 0x3498DB  # green when done, blue otherwise
        embed = discord.Embed(title=label if style == "compact" else "Processing", color=color)
        if style != "compact":
            desc_lines = ["I'm working on your request."]
            if step and max_steps:
                desc_lines.append(f"Step {step}/{max_steps}")
            if done:
                desc_lines.append("Ready to send the final answer.")
            desc_lines = [sanitize_public_text(ln) or ln for ln in desc_lines]
            embed.description = "\n".join(desc_lines)
        else:
            if step and max_steps and not done:
                embed.set_footer(text=sanitize_public_text(f"{step}/{max_steps}") or f"{step}/{max_steps}")
            if done:
                embed.set_footer(text=sanitize_public_text("done") or "done")
        return embed

    async def _execute_action(
        self,
        message: discord.Message,
        action: BotAction,
        target_message: discord.Message | None = None,
        dispatch_meta: dict[str, Any] | None = None,
    ) -> None:
        """Executes a BotAction by sending or editing a message.
        If target_message is provided and there are no files/audio, we edit it to keep 1 IN → 1 OUT.
        Otherwise, we delete the placeholder and send a new reply.
        """
        # Metadata for logging
        guild_id = getattr(message.guild, "id", None)
        ingress_channel_id = getattr(message.channel, "id", None)
        user_id = getattr(message.author, "id", None)
        is_dm = isinstance(message.channel, discord.DMChannel)
        debug_token = f"d{message.id}-{uuid.uuid4().hex[:8]}"

        dispatch_meta = dispatch_meta or {}
        force_reply_target = dispatch_meta.get("trigger_message") or message
        trigger_message_id = dispatch_meta.get("trigger_message_id") or getattr(force_reply_target, "id", None)
        trigger_channel = getattr(force_reply_target, "channel", None) or getattr(message, "channel", None)
        reply_in_thread = dispatch_meta.get("reply_in_thread")
        if reply_in_thread is None:
            reply_in_thread = _is_thread_channel(trigger_channel)

        target_channel_id = dispatch_meta.get("channel_id")
        if target_channel_id is None:
            target_channel_id = getattr(trigger_channel, "parent_id", None) or getattr(trigger_channel, "id", None) if reply_in_thread else getattr(trigger_channel, "id", None) or ingress_channel_id
        target_thread_id = dispatch_meta.get("thread_id")
        if target_thread_id is None and reply_in_thread:
            target_thread_id = getattr(trigger_channel, "id", None)

        mention_detected = dispatch_meta.get("mention_detected", False)
        reply_target_ok = dispatch_meta.get("reply_target_ok", force_reply_target is not None)
        context_label = dispatch_meta.get("context") or ("dm" if is_dm else "guild")

        base_extra = {
            "guild_id": guild_id,
            "user_id": user_id,
            "msg_id": message.id,
            "context": context_label,
            "trigger_message_id": trigger_message_id,
            "reply_target_ok": reply_target_ok,
            "mention_detected": mention_detected,
            "reply_in_thread": reply_in_thread,
            "channel_id": target_channel_id,
        }
        if target_thread_id is not None:
            base_extra["thread_id"] = target_thread_id

        self.logger.info(
            f"dispatch:pre | channel_id={ingress_channel_id} is_dm={is_dm} ready={self._is_ready.is_set()} embeds={len(action.embeds) if action.embeds else 0} meta={action.meta}",
            extra={**base_extra, "event": "dispatch.send.pre"},
        )

        files = None
        # If action requires TTS, process it.
        if action.meta.get("requires_tts"):
            self.logger.info(
                "TTS requested, processing…",
                extra={**base_extra, "event": "tts.process.start"},
            )
            if not self.tts_manager:
                self.logger.error(
                    "tts:missing",
                    extra={**base_extra, "event": "tts.manager.missing"},
                )
                action.content = "I tried to respond with voice, but the TTS service is not working."
            else:
                try:
                    action = await self.tts_manager.process(action)
                except SynthesisError as exc:
                    status = {}
                    try:
                        status = self.tts_manager.get_status() if self.tts_manager else {}
                    except (AttributeError, TypeError):
                        status = {}
                    reason = status.get("degraded_reason") or str(exc)
                    self.logger.error(
                        "tts.process.failed",
                        extra={
                            **base_extra,
                            "event": "tts.process.failed",
                            "reason": reason,
                            "engine": status.get("engine"),
                        },
                        exc_info=True,
                    )
                    action.audio_path = None
                    action.meta["tts_error"] = reason
                    action.meta["tts_failed"] = True
                    action.content = "I tried to respond with voice, but the TTS service is not working."

        # If action has an audio path after processing, prepare it for sending.
        if action.audio_path:
            if os.path.exists(action.audio_path):
                try:
                    from pathlib import Path as _Path

                    suffix = _Path(action.audio_path).suffix or ".wav"
                    safe_suffix = suffix if len(suffix) <= 6 else ".wav"
                    filename = f"voice_message{safe_suffix}"
                except (AttributeError, ValueError, TypeError):
                    filename = "voice_message.wav"
                files = [discord.File(action.audio_path, filename=filename)]
            else:
                self.logger.error(
                    f"tts:file_missing | path={action.audio_path}",
                    extra={**base_extra, "event": "tts.file.missing"},
                )
                action.content = "I tried to send a voice message, but the audio file was missing."
        elif action.meta.get("requires_tts"):  # TTS expected but failed
            self.logger.error(
                "tts:no_audio",
                extra={**base_extra, "event": "tts.audio.not_generated"},
            )
            action.content = "I tried to send a voice message, but the audio file was missing."

        # Attempt Discord native voice message flow if enabled and audio is present.
        # Discord voice messages cannot include embeds or content; enforce constraints. [CA][REH][IV]
        try:
            if action.audio_path and self.config.get("VOICE_ENABLE_NATIVE", False):
                self.logger.info(
                    "voice:native.attempt",
                    extra={**base_extra, "event": "voice.native.attempt"},
                )

                # Preserve originals in case we need to fallback
                _orig_content = action.content
                _orig_embeds = list(action.embeds) if action.embeds else []

                # Strip content/embeds to comply with Discord native voice restrictions
                if action.embeds:
                    self.logger.debug(
                        "voice:native.strip_embeds",
                        extra={**base_extra, "event": "voice.native.strip_embeds"},
                    )
                    action.embeds = []
                if action.content:
                    self.logger.debug(
                        "voice:native.strip_content",
                        extra={**base_extra, "event": "voice.native.strip_content"},
                    )
                    action.content = ""

                publisher = VoiceMessagePublisher(self.logger)
                res = await publisher.publish(message=message, wav_path=action.audio_path)
                if res and getattr(res, "ok", False):
                    # Remove placeholder if present and stop; publisher already posted the message.
                    if target_message:
                        with suppress(Exception):
                            await target_message.delete()
                    if self.enhanced_context_manager and res.message:
                        await self.enhanced_context_manager.append_message(res.message, role="bot")
                    self.logger.info(
                        "voice:native.ok",
                        extra={**base_extra, "event": "voice.native.ok"},
                    )
                    return
                self.logger.warning(
                    "voice:native.fallback",
                    extra={**base_extra, "event": "voice.native.fallback"},
                )
                # Restore original content/embeds for normal send path
                action.content = _orig_content
                action.embeds = _orig_embeds
        except Exception as e:
            self.logger.error(
                f"voice:native.exception | {e}",
                extra={**base_extra, "event": "voice.native.exception"},
                exc_info=True,
            )

        # Attach any files carried directly on the action (e.g. conversational
        # image-edit results) alongside any TTS audio file already queued.
        # `action.files` was previously a dead field - nothing read it. [RM][CA]
        if action.files:
            files = (files or []) + list(action.files)

        content = action.content or ""
        content = sanitize_public_text(content)
        if action.embeds:
            action.embeds = [sanitize_embed_for_public(e) for e in action.embeds]
        embed_count = len(action.embeds) if action.embeds else 0
        file_count = len(files) if files else 0

        # Pre-split content into Discord-safe chunks; most replies will be a single chunk.
        max_len = _DISCORD_MAX_CONTENT_LEN
        chunks = self._chunk_message_content(content) if content and len(content) > max_len else [content]
        needs_chunking = len(chunks) > 1
        if needs_chunking:
            try:
                joined = "".join(chunks)
                if joined != (content or ""):
                    self.logger.warning(
                        "dispatch.chunk.mismatch",
                        extra={
                            **base_extra,
                            "event": "dispatch.send.chunk_mismatch",
                            "content_len": len(content or ""),
                            "joined_len": len(joined or ""),
                            "parts": len(chunks),
                        },
                    )

                max_part_len = max(len(c) for c in chunks) if chunks else 0
                if max_part_len > max_len:
                    self.logger.warning(
                        "dispatch.chunk.oversize",
                        extra={
                            **base_extra,
                            "event": "dispatch.send.chunk_oversize",
                            "max_part_len": max_part_len,
                            "max_len": max_len,
                            "parts": len(chunks),
                        },
                    )

                self.logger.info(
                    "dispatch.chunked",
                    extra={
                        **base_extra,
                        "event": "dispatch.send.chunked",
                        "parts": len(chunks),
                        "content_len": len(content or ""),
                    },
                )
            except (AttributeError, TypeError, ValueError) as exc:
                self.logger.debug(f"dispatch.chunked logging failed: {exc}")

        # Oversize content is normally delivered in full as ordered multi-part
        # messages, so the full_response.txt attachment would just duplicate it on
        # every long reply. Attach it only when chunking did NOT cover the content --
        # i.e. the splitter returned a single oversize part. [REH][PA]
        if content and len(content) > _DISCORD_MAX_CONTENT_LEN and not needs_chunking:
            self.logger.warning(
                f"dispatch:overflow | length={len(content)} reason=unchunkable",
                extra={**base_extra, "event": "dispatch.content.overflow"},
            )
            file_content = io.BytesIO(content.encode("utf-8"))
            text_file = discord.File(fp=file_content, filename="full_response.txt")
            if files is None:
                files = []
            files.append(text_file)
            file_count = len(files)

        # Guardrail: if everything is empty, synthesize a fallback message
        if (not content.strip()) and embed_count == 0 and file_count == 0:
            content = f"ℹ️ I generated an empty response. I've logged this so it can be fixed. Please try again. [ref: {debug_token}]"
            self.logger.warning(
                f"dispatch:empty | ref={debug_token}",
                extra={**base_extra, "event": "dispatch.guard.empty"},
            )

        # Prepare preview for logs
        preview = content.replace("\n", " ")[:120] if content else ""
        self.logger.info(
            f'dispatch:attempt | content_len={len(content)} preview="{preview}" embeds={embed_count} files={file_count}',
            extra={**base_extra, "event": "dispatch.send.attempt"},
        )

        typing_enabled = bool(content.strip()) and (needs_chunking or embed_count > 0 or file_count > 0 or len(content.strip()) >= 30)

        async with self._optional_typing(message.channel, base_extra=base_extra, enabled=typing_enabled):
            try:
                if needs_chunking and (content or action.embeds or files):
                    sent_message = await self._send_chunked_reply(
                        message=message,
                        action=action,
                        base_extra=base_extra,
                        force_reply_target=force_reply_target,
                        target_message=target_message,
                        dispatch_meta=dispatch_meta,
                        content=content,
                        files=files,
                        chunks=chunks,
                    )
                elif content or action.embeds or files:
                    ch = getattr(force_reply_target, "channel", None) or getattr(message, "channel", None)
                    reply_target = force_reply_target
                    scope_case = "forced"
                    if reply_target is None:
                        scope_case = "fallback"
                        try:
                            if _is_thread_channel(ch):
                                reply_target, _ = await resolve_thread_reply_target(self, message, self.config)
                                scope_case = "thread"
                            elif getattr(message, "reference", None) is not None:
                                reply_target = message
                                scope_case = "reply"
                            else:
                                reply_target = message
                                scope_case = "plain"
                        except (AttributeError, TypeError, discord.HTTPException, discord.NotFound):
                            reply_target = message
                            scope_case = "reply"

                    # Scope + target breadcrumbs
                    try:
                        self.logger.info(
                            "scope_resolved",
                            extra={
                                **base_extra,
                                "subsys": "route",
                                "event": "scope_resolved",
                                "detail": {
                                    "case": scope_case,
                                    "scope": getattr(ch, "id", None),
                                },
                            },
                        )
                        if reply_target is not None:
                            self.logger.info(
                                "reply_target_ok",
                                extra={
                                    **base_extra,
                                    "subsys": self.config.get("MEM_LOG_SUBSYS", "mem.force"),
                                    "event": "reply_target_ok",
                                    "detail": {
                                        "id": getattr(reply_target, "id", None),
                                        "reason": "trigger_message",
                                    },
                                },
                            )
                    except (AttributeError, TypeError, ValueError) as exc:
                        self.logger.debug(f"scope/reply_target logging failed: {exc}")

                    with suppress(Exception):
                        self.logger.info(
                            "reply.target",
                            extra={
                                **base_extra,
                                "event": "dispatch.reply_target",
                                "detail": {
                                    "channel_id": base_extra.get("channel_id"),
                                    "thread_id": base_extra.get("thread_id"),
                                    "trigger_message_id": trigger_message_id,
                                },
                            },
                        )

                    # Decide whether it's safe to edit the placeholder
                    must_retarget = bool(target_message) and (reply_target is None or getattr(reply_target, "id", None) != getattr(message, "id", None))

                    # Resolve recipient to mention and ping strategy (avoid self-mention and double ping)
                    recipient = None
                    recipient_reason = "no_target"
                    try:
                        if reply_target is not None and hasattr(reply_target, "author"):
                            recipient = getattr(reply_target, "author", None)
                            recipient_reason = "target_author"
                        # Avoid self-mention (bot) and any bot authors
                        if recipient is not None and (getattr(recipient, "bot", False) or getattr(recipient, "id", None) == getattr(self.user, "id", None)):
                            # Fallback: latest human speaker in-scope; minimally choose triggering human author
                            recipient = message.author if not getattr(message.author, "bot", False) else None
                            recipient_reason = "fallback_human" if recipient else "no_human"
                    except (AttributeError, TypeError):
                        recipient = None
                        recipient_reason = "no_human"

                    # Decide ping mode
                    ping_mode = "none"
                    explicit_mention = False
                    try:
                        if recipient is not None:
                            # Preferred: rely on reply ping when the target author is the human recipient
                            if reply_target is not None and getattr(reply_target, "author", None) is recipient and not getattr(recipient, "bot", False):
                                ping_mode = "reply_ping"
                            else:
                                # Use explicit mention when target author is a bot or different; avoid double-ping
                                ping_mode = "explicit_mention"
                                explicit_mention = True
                        else:
                            ping_mode = "none"
                    except (AttributeError, TypeError):
                        ping_mode = "none"
                        explicit_mention = False

                    # Build AllowedMentions whitelist to enforce single notification path
                    try:
                        if ping_mode == "reply_ping":
                            allowed_mentions = discord.AllowedMentions(everyone=False, users=[], roles=False, replied_user=True)
                        elif ping_mode == "explicit_mention" and recipient is not None:
                            allowed_mentions = discord.AllowedMentions(
                                everyone=False,
                                users=[recipient],
                                roles=False,
                                replied_user=False,
                            )
                        else:
                            allowed_mentions = discord.AllowedMentions(
                                everyone=False,
                                users=[],
                                roles=False,
                                replied_user=False,
                            )
                    except (discord.HTTPException, discord.NotFound, discord.Forbidden, AttributeError):
                        allowed_mentions = None

                    # Optionally prefix explicit mention (single path); do not double-ping
                    try:
                        if explicit_mention and recipient is not None:
                            mention_prefix = getattr(recipient, "mention", None)
                            if mention_prefix and not (content or "").lstrip().startswith(str(mention_prefix)):
                                content = f"{mention_prefix} {content}" if content else f"{mention_prefix}"
                    except (AttributeError, TypeError) as exc:
                        self.logger.debug(f"explicit_mention prefix failed: {exc}")

                    # Minimal logging for recipient and ping strategy
                    try:
                        self.logger.info(
                            "recipient_resolved",
                            extra={
                                **base_extra,
                                "subsys": "mention",
                                "event": "recipient_resolved",
                                "detail": {
                                    "user": getattr(recipient, "id", None),
                                    "reason": recipient_reason,
                                },
                            },
                        )
                        self.logger.info(
                            "ping_strategy",
                            extra={
                                **base_extra,
                                "subsys": "mention",
                                "event": "ping_strategy",
                                "detail": {"mode": ping_mode},
                            },
                        )
                    except (AttributeError, TypeError, ValueError) as exc:
                        self.logger.debug(f"recipient/ping logging failed: {exc}")

                    if target_message and not files and not must_retarget:
                        # Edit the existing streaming message in-place
                        sent_message = target_message
                        await self._call_with_discord_retry(
                            "edit_message",
                            lambda: sent_message.edit(content=content, embeds=action.embeds),
                            base_extra=base_extra,
                        )
                        self.logger.info(
                            f"dispatch:edit.ok | discord_msg_id={getattr(sent_message, 'id', None)} embeds={embed_count} files={file_count}",
                            extra={**base_extra, "event": "dispatch.edit.ok"},
                        )
                        # Track bot response in enhanced context manager
                        if self.enhanced_context_manager and sent_message:
                            await self.enhanced_context_manager.append_message(sent_message, role="bot")
                    else:
                        # Remove placeholder if present, then send a proper reply with desired target
                        if target_message:
                            with suppress(Exception):
                                await self._call_with_discord_retry(
                                    "delete_message",
                                    target_message.delete,
                                    base_extra=base_extra,
                                )

                        # Log send mode
                        with suppress(Exception):
                            self.logger.info(
                                "send",
                                extra={
                                    **base_extra,
                                    "subsys": "route",
                                    "event": "send",
                                    "detail": {"mode": "delete_and_resend" if target_message else "direct"},
                                },
                            )

                        if reply_target is None and _is_thread_channel(ch):
                            # Send to thread without a reply reference (no reply-to-self loops available)
                            sent_message = await self._call_with_discord_retry(
                                "channel_send",
                                lambda: message.channel.send(
                                    content=content,
                                    embeds=action.embeds,
                                    files=files,
                                    allowed_mentions=allowed_mentions,
                                ),
                                base_extra=base_extra,
                            )
                        elif reply_target is not None:
                            sent_message = await self._call_with_discord_retry(
                                "reply_send",
                                lambda: reply_target.reply(
                                    content=content,
                                    embeds=action.embeds,
                                    files=files,
                                    mention_author=False,
                                    allowed_mentions=allowed_mentions,
                                ),
                                base_extra=base_extra,
                            )
                        else:
                            # Non-thread or fallback: reply to triggering message
                            sent_message = await self._call_with_discord_retry(
                                "reply_send",
                                lambda: message.reply(
                                    content=content,
                                    embeds=action.embeds,
                                    files=files,
                                    mention_author=False,
                                    allowed_mentions=allowed_mentions,
                                ),
                                base_extra=base_extra,
                            )

                        self.logger.info(
                            f"dispatch:ok | discord_msg_id={getattr(sent_message, 'id', None)} embeds={embed_count} files={file_count}",
                            extra={**base_extra, "event": "dispatch.send.ok"},
                        )

                        # Track bot response in enhanced context manager
                        if self.enhanced_context_manager and sent_message:
                            await self.enhanced_context_manager.append_message(sent_message, role="bot")
            except discord.errors.HTTPException as e:
                if e.code == _DISCORD_ERR_UNKNOWN_MESSAGE:
                    # Trigger message was deleted: degrade to a plain channel message.
                    self.logger.warning(
                        "dispatch:fallback | reason=unknown_message",
                        extra={**base_extra, "event": "dispatch.send.reply_fallback"},
                    )
                    sent_message = await self._call_with_discord_retry(
                        "channel_send",
                        lambda: message.channel.send(content=content, embeds=action.embeds, files=files),
                        base_extra=base_extra,
                    )

                    self.logger.info(
                        f"dispatch:ok | discord_msg_id={getattr(sent_message, 'id', None)} embeds={embed_count} files={file_count}",
                        extra={**base_extra, "event": "dispatch.send.ok"},
                    )
                    # Track bot response in enhanced context manager
                    if self.enhanced_context_manager and sent_message:
                        await self.enhanced_context_manager.append_message(sent_message, role="bot")
                else:
                    # Everything else, including _DISCORD_ERR_INVALID_FORM_BODY: log and
                    # propagate. Content-length 50035 cannot reach here (every chunk is
                    # capped at _DISCORD_MAX_CONTENT_LEN before sending), so a 50035 here
                    # means a malformed embed or file -- re-sending the identical payload,
                    # as this branch used to, could only fail the same way. [REH]
                    self.logger.error(
                        f"dispatch:error | code={e.code} status={getattr(e, 'status', 'n/a')} details={e!s}",
                        extra={**base_extra, "event": "dispatch.send.error"},
                        exc_info=True,
                    )
                    raise  # Re-raise other HTTP exceptions
            finally:
                sent = getattr(self, "_last_sent_message_for_finalize", None)
                self.logger.debug(
                    f"dispatch:finalize | sent={(sent is not None)}",
                    extra={**base_extra, "event": "dispatch.send.finalize"},
                )

    async def _send_chunked_reply(
        self,
        message: discord.Message,
        action: BotAction,
        *,
        base_extra: dict[str, Any],
        force_reply_target: discord.Message | None,
        target_message: discord.Message | None,
        dispatch_meta: dict[str, Any],
        content: str,
        files,
        chunks: list[str],
    ) -> discord.Message | None:
        """Send a long text reply as multiple Discord messages while preserving reply targeting."""
        guild_id = getattr(message.guild, "id", None)
        is_dm = isinstance(message.channel, discord.DMChannel)
        ingress_channel_id = getattr(message.channel, "id", None)
        _ = guild_id, is_dm, ingress_channel_id  # already present in base_extra

        if not chunks:
            return None

        ch = getattr(force_reply_target, "channel", None) or getattr(message, "channel", None)
        reply_target = force_reply_target
        scope_case = "forced"
        if reply_target is None:
            scope_case = "fallback"
            try:
                if _is_thread_channel(ch):
                    reply_target, _ = await resolve_thread_reply_target(self, message, self.config)
                    scope_case = "thread"
                elif getattr(message, "reference", None) is not None:
                    reply_target = message
                    scope_case = "reply"
                else:
                    reply_target = message
                    scope_case = "plain"
            except (AttributeError, TypeError, discord.HTTPException, discord.NotFound):
                reply_target = message
                scope_case = "reply"

        # Scope + target breadcrumbs
        try:
            self.logger.info(
                "scope_resolved",
                extra={
                    **base_extra,
                    "subsys": "route",
                    "event": "scope_resolved",
                    "detail": {"case": scope_case, "scope": getattr(ch, "id", None)},
                },
            )
            if reply_target is not None:
                self.logger.info(
                    "reply_target_ok",
                    extra={
                        **base_extra,
                        "subsys": self.config.get("MEM_LOG_SUBSYS", "mem.force"),
                        "event": "reply_target_ok",
                        "detail": {
                            "id": getattr(reply_target, "id", None),
                            "reason": "trigger_message",
                        },
                    },
                )
        except (AttributeError, TypeError, ValueError) as exc:
            self.logger.debug(f"scope/reply_target logging failed: {exc}")

        with suppress(Exception):
            self.logger.info(
                "reply.target",
                extra={
                    **base_extra,
                    "event": "dispatch.reply_target",
                    "detail": {
                        "channel_id": base_extra.get("channel_id"),
                        "thread_id": base_extra.get("thread_id"),
                        "trigger_message_id": dispatch_meta.get("trigger_message_id"),
                    },
                },
            )

        # Any existing placeholder should be removed; multi-part replies always target the user message directly.
        if target_message is not None:
            with suppress(Exception):
                await self._call_with_discord_retry("delete_message", target_message.delete, base_extra=base_extra)

        # Resolve recipient and ping strategy once, then fan out to chunks.
        recipient = None
        recipient_reason = "no_target"
        try:
            if reply_target is not None and hasattr(reply_target, "author"):
                recipient = getattr(reply_target, "author", None)
                recipient_reason = "target_author"
            if recipient is not None and (getattr(recipient, "bot", False) or getattr(recipient, "id", None) == getattr(self.user, "id", None)):
                recipient = message.author if not getattr(message.author, "bot", False) else None
                recipient_reason = "fallback_human" if recipient else "no_human"
        except (AttributeError, TypeError):
            recipient = None
            recipient_reason = "no_human"

        ping_mode = "none"
        explicit_mention = False
        try:
            if recipient is not None:
                if reply_target is not None and getattr(reply_target, "author", None) is recipient and not getattr(recipient, "bot", False):
                    ping_mode = "reply_ping"
                else:
                    ping_mode = "explicit_mention"
                    explicit_mention = True
            else:
                ping_mode = "none"
        except (AttributeError, TypeError):
            ping_mode = "none"
            explicit_mention = False

        try:
            self.logger.info(
                "recipient_resolved",
                extra={
                    **base_extra,
                    "subsys": "mention",
                    "event": "recipient_resolved",
                    "detail": {
                        "user": getattr(recipient, "id", None),
                        "reason": recipient_reason,
                    },
                },
            )
            self.logger.info(
                "ping_strategy",
                extra={
                    **base_extra,
                    "subsys": "mention",
                    "event": "ping_strategy",
                    "detail": {"mode": ping_mode},
                },
            )
        except (AttributeError, TypeError, ValueError) as exc:
            self.logger.debug(f"recipient/ping logging failed: {exc}")

        try:
            if ping_mode == "reply_ping":
                allowed_first = discord.AllowedMentions(everyone=False, users=[], roles=False, replied_user=True)
                allowed_followups = discord.AllowedMentions(everyone=False, users=[], roles=False, replied_user=False)
            elif ping_mode == "explicit_mention" and recipient is not None:
                allowed_first = discord.AllowedMentions(
                    everyone=False,
                    users=[recipient],
                    roles=False,
                    replied_user=False,
                )
                allowed_followups = discord.AllowedMentions(everyone=False, users=[], roles=False, replied_user=False)
            else:
                allowed_first = discord.AllowedMentions(everyone=False, users=[], roles=False, replied_user=False)
                allowed_followups = allowed_first
        except (discord.HTTPException, discord.NotFound, discord.Forbidden, AttributeError):
            allowed_first = None
            allowed_followups = None

        mention_prefix = None
        try:
            if explicit_mention and recipient is not None:
                mention_prefix = getattr(recipient, "mention", None)
        except (AttributeError, TypeError):
            mention_prefix = None

        total_parts = len(chunks)
        last_sent: discord.Message | None = None

        # Fence prefix/suffix per raw chunk, computed once up front from the
        # UNSANITIZED chunks so fence state tracks the actual ``` occurrences.
        # Applied around each part's sanitized text below -- a code block split
        # across parts still renders as code in every part, without changing
        # what `chunks` itself means for the mismatch/oversize checks upstream. [REH]
        fence_markers = fence_wrap_markers(chunks)

        for part_idx, base_chunk in enumerate(chunks, start=1):
            is_first_part = part_idx == 1
            fence_prefix, fence_suffix = fence_markers[part_idx - 1]
            chunk_content = sanitize_public_text(base_chunk)
            if chunk_content:
                chunk_content = f"{fence_prefix}{chunk_content}{fence_suffix}"

            # Apply explicit mention only to the first part.
            try:
                if is_first_part and mention_prefix and not (chunk_content or "").lstrip().startswith(str(mention_prefix)):
                    chunk_content = f"{mention_prefix} {chunk_content}" if chunk_content else f"{mention_prefix}"
            except (AttributeError, TypeError, ValueError) as exc:
                self.logger.debug(f"explicit_mention chunk prefix failed: {exc}")

            part_embeds = action.embeds if is_first_part else []
            part_files = files if (is_first_part and files) else None
            part_allowed_mentions = allowed_first if is_first_part else allowed_followups

            if not (chunk_content or part_embeds or part_files):
                continue

            # Per-part attempt logging
            try:
                part_preview = (chunk_content or "").replace("\n", " ")[:120]
                self.logger.info(
                    f'dispatch:attempt | part={part_idx}/{total_parts} content_len={len(chunk_content or "")} preview="{part_preview}" embeds={len(part_embeds)} files={len(part_files) if part_files else 0}',
                    extra={
                        **base_extra,
                        "event": "dispatch.send.attempt",
                        "part": part_idx,
                        "parts": total_parts,
                    },
                )
            except (AttributeError, TypeError, ValueError) as exc:
                self.logger.debug(f"dispatch.attempt logging failed: {exc}")

            # Resolve channel for sending.
            send_channel = ch or message.channel

            # Small delay between continuation chunks to respect Discord rate limits
            # (5 messages / 5 seconds per channel). This ensures ordered delivery and
            # avoids silent drops from hitting the limit too hard [REH][PA].
            if not is_first_part:
                await asyncio.sleep(0.3)

            try:
                if is_first_part and reply_target is not None:
                    # First chunk: actual reply to the user's message.
                    sent = await self._call_with_discord_retry(
                        "reply_send",
                        lambda: reply_target.reply(
                            content=chunk_content,
                            embeds=part_embeds,
                            files=part_files,
                            mention_author=False,
                            allowed_mentions=part_allowed_mentions,
                        ),
                        base_extra=base_extra,
                    )
                else:
                    # Continuation chunks: plain channel messages (no reply chain).
                    sent = await self._call_with_discord_retry(
                        "channel_send",
                        lambda: send_channel.send(
                            content=chunk_content,
                            embeds=part_embeds,
                            files=part_files,
                            allowed_mentions=part_allowed_mentions,
                        ),
                        base_extra=base_extra,
                    )

                # Ensure strict ordering: each chunk is fully sent before the next.
                last_sent = sent
                with suppress(Exception):
                    self.logger.info(
                        f"dispatch:ok | part={part_idx}/{total_parts} discord_msg_id={getattr(sent, 'id', None)} embeds={len(part_embeds)} files={len(part_files) if part_files else 0}",
                        extra={
                            **base_extra,
                            "event": "dispatch.send.ok",
                            "part": part_idx,
                            "parts": total_parts,
                        },
                    )

                if self.enhanced_context_manager and sent:
                    await self.enhanced_context_manager.append_message(sent, role="bot")
            except discord.errors.HTTPException as e:
                with suppress(Exception):
                    self.logger.error(
                        f"dispatch:error | part={part_idx}/{total_parts} code={e.code} status={getattr(e, 'status', 'n/a')} details={e!s}",
                        extra={
                            **base_extra,
                            "event": "dispatch.send.error",
                            "part": part_idx,
                            "parts": total_parts,
                        },
                        exc_info=True,
                    )
                break

        return last_sent

    def _chunk_message_content(self, content: str) -> list[str]:
        """Split a text payload into Discord-safe chunks.

        Thin delegate to the shared splitter so every outbound path -- this one and
        ``bot.core.output`` -- applies the same policy and limit. [CA]
        """
        return split_for_discord(content, _DISCORD_MAX_CONTENT_LEN)

    def _is_long_running_admin_command(self, message: discord.Message) -> bool:
        """Check if this is a long-running admin command that should run out-of-band."""
        if not message.content:
            return False

        content = message.content.strip().lower()
        prefix = getattr(self, "command_prefix", "!")

        # List of long-running admin commands that should not block user queues
        long_running_commands = [
            f"{prefix}rag bootstrap",
            f"{prefix}rag refresh",
            f"{prefix}rag update",
            f"{prefix}rag scan",
            f"{prefix}archive-sync",
            f"{prefix}archive-sync-channel",
            # Add other potentially long-running commands here
        ]

        return any(content.startswith(cmd) for cmd in long_running_commands)

    async def _message_is_command(self, message: discord.Message) -> bool:
        """Determine if a message should be treated as a command fallback candidate."""
        if not message.content:
            return False

        try:
            prefixes = await self.get_prefix(message)
            if isinstance(prefixes, (list, tuple)):
                return any(prefix and message.content.startswith(prefix) for prefix in prefixes)
            if prefixes:
                return message.content.startswith(prefixes)
        except (discord.HTTPException, discord.NotFound, discord.Forbidden, AttributeError) as e:
            self.logger.debug(f"command_prefix_check_failed | {e}")
        return False

    def _generate_task_id(self, message: discord.Message) -> str:
        """Generate a unique task ID for tracking."""
        return f"{message.author.id}_{message.id}_{message.content.split()[0] if message.content else 'unknown'}"

    def _register_long_running_task(self, task_id: str, task: asyncio.Task, message: discord.Message, command: str) -> None:
        """Register a long-running task for tracking and cancellation."""
        self._active_long_running_tasks[task_id] = task
        self._task_metadata[task_id] = {
            "user_id": message.author.id,
            "channel_id": message.channel.id,
            "guild_id": message.guild.id if message.guild else None,
            "command": command,
            "started_at": asyncio.get_event_loop().time(),
            "message_id": message.id,
        }

        # Add callback to clean up when task completes
        def cleanup_task(future) -> None:
            # [BUGFIX] Use threadsafe deletion from done callback to avoid race with cancel_task
            self._active_long_running_tasks.pop(task_id, None)
            self._task_metadata.pop(task_id, None)

        task.add_done_callback(cleanup_task)

        self.logger.info(f"Registered long-running task: {task_id} for command: {command}")

    def get_active_tasks_for_user(self, user_id: int) -> list[tuple[str, dict[str, Any]]]:
        """Get all active long-running tasks for a specific user."""
        user_tasks = []
        for task_id, metadata in self._task_metadata.items():
            if metadata["user_id"] == user_id:
                user_tasks.append((task_id, metadata))
        return user_tasks

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a long-running task gracefully."""
        if task_id not in self._active_long_running_tasks:
            return False

        task = self._active_long_running_tasks[task_id]
        metadata = self._task_metadata.get(task_id, {})

        self.logger.info(f"Cancelling long-running task: {task_id} (command: {metadata.get('command', 'unknown')})")

        # Cancel the task
        task.cancel()

        try:
            # Wait a bit for graceful cancellation
            await asyncio.wait_for(task, timeout=5.0)
        except (TimeoutError, asyncio.CancelledError):
            # Expected - task was cancelled or timed out
            pass
        except (RuntimeError, OSError) as e:
            self.logger.warning(f"Error during task cancellation: {e}")

        # Clean up tracking
        if task_id in self._active_long_running_tasks:
            del self._active_long_running_tasks[task_id]
        if task_id in self._task_metadata:
            del self._task_metadata[task_id]

        return True

    async def _execute_out_of_band_command(self, message: discord.Message) -> None:
        """Execute a long-running command outside the user's message queue."""
        try:
            guild_info = "DM" if isinstance(message.channel, discord.DMChannel) else f"guild:{message.guild.id}"
            self.logger.info(f"Executing out-of-band command: msg_id:{message.id} author:{message.author.id} in:{guild_info} cmd:{message.content[:50]}...")

            # Process the command directly without queuing
            await self._process_single_message(message)

        except Exception as e:
            self.logger.error(
                f"Error in out-of-band command execution for {message.id}: {e}",
                exc_info=True,
            )
            # Send error message to user
            try:
                await message.reply(
                    f"❌ Error executing command: {str(e)[:100]}...",
                    mention_author=True,
                )
            except (discord.HTTPException, discord.NotFound, discord.Forbidden) as exc:
                self.logger.debug(f"Error reply failed: {exc}")

    async def on_message(self, message: discord.Message) -> None:
        """Delegate early filtering to MessageProcessor, then handle
        SSOT gate / readiness / long-running commands before enqueue.
        """
        # Archive DMs for dashboard (best-effort, non-blocking) — before
        # the early-return gate so DMs are captured even if filtered out
        # by the message processor (empty content, alert-suppressed, etc.)
        if isinstance(message.channel, discord.DMChannel):
            with suppress(Exception):
                self._track_background_task(asyncio.create_task(self._record_dm_message(message)))

        # Early filtering returns False for messages that should be dropped
        if not await self.message_processor.on_message(message):
            return

        # --- SSOT gate: check gating before enqueuing heavy work [IV] ---
        try:
            if self.router is not None and not self._is_long_running_admin_command(message):
                is_command_msg = False
                try:
                    prefixes = await self.get_prefix(message)
                    if isinstance(prefixes, (list, tuple)):
                        is_command_msg = any(prefix and message.content.startswith(prefix) for prefix in prefixes)
                    elif prefixes:
                        is_command_msg = message.content.startswith(prefixes)
                except (discord.HTTPException, discord.NotFound, discord.Forbidden, AttributeError) as e:
                    self.logger.debug(f"prefix_check_failed | {e}")

                if not is_command_msg:
                    gate_allowed = self.router._should_process_message(message)
                    if gate_allowed:
                        self.router.record_gate_hint(message.id, True)
                    else:
                        reason = self.router.pop_gate_denied_reason(message.id) or "blocked"
                        with suppress(Exception):
                            self.logger.info(
                                f"gate.drop | reason={reason} msg_id:{message.id}",
                                extra={
                                    "event": "gate.drop",
                                    "reason": reason,
                                    "msg_id": message.id,
                                    "user_id": getattr(message.author, "id", None),
                                },
                            )
                        return
        except (AttributeError, TypeError, ValueError, RuntimeError) as e:
            self.logger.warning(f"Gate check failed for msg_id:{message.id}: {e}")

        await self._is_ready.wait()  # Wait until the bot is ready

        # Check if this is a long-running admin command
        if self._is_long_running_admin_command(message):
            task_id = self._generate_task_id(message)
            command = message.content.split()[0] if message.content else "unknown"
            task = asyncio.create_task(self._execute_out_of_band_command(message))
            self._register_long_running_task(task_id, task, message, command)
            return

        # Delegate regular message queueing to MessageProcessor
        await self.message_processor.enqueue(message)

    async def load_profiles(self) -> None:
        """Load user and server memory profiles."""
        try:
            self.logger.info("Loading memory profiles...")
            self.user_profiles, self.server_profiles = load_all_profiles()
            self.logger.info(f"Loaded {len(self.user_profiles)} user and {len(self.server_profiles)} server profiles.")
        except Exception as e:
            self.logger.error(f"Failed to load profiles: {e}", exc_info=True)

    def setup_background_tasks(self) -> None:
        """Set up background tasks for the bot."""
        try:
            from bot.tasks import setup_memory_save_task
            from bot.tasks_registry import get_registry

            registry = get_registry()

            # --- Curated memory service ---
            memory_enabled = bool(self.config.get("PERSISTENT_MEMORY_ENABLE", True))
            if memory_enabled:
                memory_task = asyncio.create_task(start_memory_service(self), name="memory_service")
                registry.register(memory_task, name="memory_service", feature="memory")
                self._track_background_task(memory_task)
                self.logger.info("Curated memory service background worker started")
            else:
                self.logger.info("Curated memory service disabled — skipping background worker")

            # --- Memory distiller ---
            distiller_enabled = bool(self.config.get("MEMORY_DISTILLER_ENABLED", False))
            if distiller_enabled:
                distiller_task = asyncio.create_task(start_memory_distiller(self), name="memory_distiller")
                registry.register(distiller_task, name="memory_distiller", feature="memory")
                self._track_background_task(distiller_task)
                self.logger.info("Memory distiller background worker started")
            else:
                self.logger.info("Memory distiller disabled — skipping background worker")

            # --- Context autosave ---
            self.memory_save_task = setup_memory_save_task(self)
            self.memory_save_task.start()

            # --- Server archive ingest ---
            archive_enabled = bool(
                self.config.get(
                    "SERVER_ARCHIVE_ENABLED",
                    self.config.get("SERVER_ARCHIVE_ENABLE", False),
                ),
            )
            if archive_enabled:
                archive_start_task = asyncio.create_task(start_server_archive_service(self), name="server_archive_start")

                def _log_archive_start_failure(task) -> None:
                    try:
                        self.archive_service = task.result()
                    except Exception as exc:
                        self.logger.error(f"Server archive startup failed: {exc}", exc_info=True)

                archive_start_task.add_done_callback(_log_archive_start_failure)
                self._track_background_task(archive_start_task)
                registry.register(archive_start_task, name="server_archive", feature="server_archive")
            else:
                self.logger.info("Server archive disabled — skipping ingest workers")

            # --- Dashboard server ---
            self._track_background_task(asyncio.create_task(self._start_dashboard(), name="dashboard_start"))
        except Exception as e:
            self.logger.error(f"Failed to set up background tasks: {e}", exc_info=True)

    async def _start_dashboard(self) -> None:
        """Start the dashboard server if enabled. Called as a background task."""
        try:
            from bot.dashboard import AuditStore, DashboardServer, DMStore, MessageStore, load_dashboard_config
            from bot.dashboard.backfill import BackfillJobStore, BackfillService
            from bot.dashboard.services import DashboardServices

            cfg = load_dashboard_config()
            if not cfg.enabled:
                self.logger.debug("Dashboard not enabled, skipping")
                return

            audit_store = AuditStore(db_path=cfg.audit_db_path, retention_days=cfg.audit_retention_days)
            await audit_store.initialize()

            dm_store = DMStore(
                db_path="./data/dashboard_dms.db",
                retention_days=cfg.dm_retention_days,
            )
            if cfg.dm_archive_enabled:
                await dm_store.initialize()
            else:
                dm_store._initialized = True  # Skip DM operations

            services = DashboardServices(
                bot=self,
                config=cfg,
                audit_store=audit_store,
                dm_store=dm_store,
            )

            message_store = MessageStore(
                db_path=cfg.message_db_path or "./data/dashboard_messages.db",
                retention_days=cfg.message_retention_days or cfg.dm_retention_days,
            )
            if cfg.dm_archive_enabled or cfg.guild_archive_enabled:
                await message_store.initialize()

            backfill_store = BackfillJobStore(
                db_path=cfg.backfill_db_path or "./data/dashboard_backfill.db",
            )
            await backfill_store.initialize()

            backfill_service = BackfillService(
                bot=self,
                message_store=message_store,
                job_store=backfill_store,
                audit_store=audit_store,
                sleep_between_channels=cfg.backfill_sleep_ms / 1000.0 if cfg.backfill_sleep_ms else 0.5,
            )

            server = DashboardServer(
                config=cfg,
                services=services,
                audit_store=audit_store,
                dm_store=dm_store,
                message_store=message_store,
                backfill_store=backfill_store,
                backfill_service=backfill_service,
            )
            await server.start()
            self._dashboard_server = server

            # Record dashboard start event
            await audit_store.record(
                event_type="dashboard.start",
                result="success",
                metadata={"host": cfg.host, "port": cfg.port, "public_bind": cfg.public_bind},
            )
        except Exception as e:
            self.logger.error(f"Dashboard startup failed: {e}", exc_info=True)

    async def _record_dm_message(self, message) -> None:
        """Record an incoming DM to the bot via dashboard DM store."""
        server = getattr(self, "_dashboard_server", None)
        if server is None:
            return
        services = getattr(server, "_services", None) if server.app else None
        if services:
            await services.record_dm_message(message)

    async def setup_tts(self) -> None:
        """Set up TTS manager if configured."""
        try:
            from bot.tts.interface import TTSManager

            self.tts_manager = TTSManager(self)
            self.logger.info("TTS manager initialized")
        except Exception as e:
            self.logger.error(f"Failed to set up TTS: {e}", exc_info=True)

    async def setup_router(self) -> None:
        """Set up message router and vision orchestrator."""
        try:
            # Create single VisionOrchestrator instance first (idempotent) [CA]
            vision_enabled = bool(self.config.get("VISION_ENABLED", True))
            if vision_enabled:
                try:
                    from bot.vision import VisionOrchestrator

                    if not getattr(self, "vision_orchestrator", None):
                        self.vision_orchestrator = VisionOrchestrator(self.config)
                        self.logger.info("VisionOrchestrator: created")
                    # Queue non-blocking start at boot; lazy start remains as safety net
                    try:
                        import asyncio

                        loop = asyncio.get_running_loop()
                        if loop and loop.is_running() and not getattr(self.vision_orchestrator, "_started", False):
                            self._track_background_task(
                                asyncio.create_task(
                                    self.vision_orchestrator.start(),
                                    name="vision_startup",
                                ),
                            )
                            self.logger.info("VisionOrchestrator: start queued")
                    except RuntimeError:
                        # No running loop; fall back to direct start
                        try:
                            await self.vision_orchestrator.start()
                        except Exception as e:
                            self.logger.exception(f"Failed to start VisionOrchestrator: {e}")
                except ImportError:
                    self.logger.warning("Vision module not available")
                    self.vision_orchestrator = None
                except Exception as e:
                    self.logger.exception(f"Failed to initialize VisionOrchestrator: {e}")
                    self.vision_orchestrator = None
            else:
                self.vision_orchestrator = None

            # Initialize router (will adopt bot.vision_orchestrator or create fallback)
            from bot.router import Router

            self.router = Router(self)  # Pass bot instance, not config dict
            self.logger.debug("✅ Message router initialized successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize message router: {e}", exc_info=True)
            raise

    async def setup_rag(self) -> None:
        """Set up RAG system to enable eager loading if configured."""
        try:
            # Skip expensive RAG bootstrap in unit tests
            if os.getenv("PYTEST_CURRENT_TEST"):
                self.logger.debug("⏭️ Skipping RAG setup during tests")
                return

            # Check if RAG is configured and eager loading is enabled
            rag_enabled = self.config.get("rag_enabled", True)
            if not rag_enabled:
                self.logger.info("⚠️  RAG system disabled via configuration")
                return

            # Import RAG config and check eager loading setting
            from bot.rag.config import get_rag_config

            rag_config = get_rag_config()

            if rag_config.eager_vector_load:
                self.logger.info("🚀 RAG eager loading enabled - initializing RAG system at startup")

                # Initialize the RAG system to trigger eager loading
                from bot.rag.hybrid_search import get_hybrid_search

                await get_hybrid_search()

                self.logger.info("✅ RAG system initialized with eager vector loading")
            else:
                self.logger.info("⏱️  RAG lazy loading enabled - deferring initialization until first use")
        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as e:
            # RAG initialization failure should not crash the bot
            self.logger.warning(f"⚠️  RAG system initialization failed (bot will continue without RAG): {e}")
            if self.config.get("debug", False):
                self.logger.debug("RAG initialization traceback:", exc_info=True)

    async def load_extensions(self) -> None:
        """Load command extensions using Rich visual reporting."""
        import importlib

        # Define command modules and their corresponding cog classes
        module_definitions = [
            ("test_cmds", "TestCommands"),
            ("memory_cmds", "MemoryCommands"),
            ("tts_cmds", "TTSCommands"),
            ("config_commands", "ConfigCommands"),
            ("operator_commands", "OperatorCommands"),
            ("janitor_commands", "JanitorCommands"),
            ("video_commands", "VideoCommands"),
            ("rag_commands", "RAGCommands"),
            ("search_commands", "SearchCommands"),
            ("news_commands", "NewsCommands"),
            ("screenshot_commands", "ScreenshotCommands"),
            ("image_upgrade_commands", "ImageUpgradeCommands"),
            ("admin_alert_commands", "AdminAlertCommands"),
            ("archive_commands", "ArchiveCommands"),
            ("memory_extended_cmds", "ExtendedMemoryCommands"),
        ]

        command_modules = []  # List of (module_name, success_status)
        command_cogs = []  # List of (cog_name, success_status)
        loaded_modules = {}  # Store successfully loaded modules

        # Phase 1: Import command modules
        for module_name, cog_class_name in module_definitions:
            try:
                self.logger.debug(f"Importing {module_name}...")
                module = importlib.import_module(f"bot.commands.{module_name}")
                loaded_modules[cog_class_name] = module
                command_modules.append((module_name, True))
                self.logger.debug(f"✅ Successfully imported {module_name}")
            except Exception as import_error:
                command_modules.append((module_name, False))
                command_cogs.append((cog_class_name, False))
                self.logger.error(f"❌ Failed to import {module_name}: {import_error}", exc_info=True)

        # Phase 2: Load and register cogs
        for cog_class_name, module in loaded_modules.items():
            try:
                # Check if cog is already loaded to avoid duplicates
                if self.get_cog(cog_class_name):
                    self.logger.debug(f"Skipping already loaded cog: {cog_class_name}")
                    command_cogs.append((cog_class_name, True))
                    continue

                self.logger.debug(f"Loading {cog_class_name} cog...")

                # Check if module has setup function
                if hasattr(module, "setup"):
                    await module.setup(self)

                    # Verify the cog was actually loaded
                    if self.get_cog(cog_class_name):
                        command_cogs.append((cog_class_name, True))
                        self.logger.debug(f"✅ {cog_class_name} loaded successfully")
                    else:
                        command_cogs.append((cog_class_name, False))
                        self.logger.error(f"❌ {cog_class_name} setup completed but cog not found")
                else:
                    command_cogs.append((cog_class_name, False))
                    self.logger.error(f"❌ {cog_class_name} module missing setup function")

            except Exception as cog_error:
                command_cogs.append((cog_class_name, False))
                self.logger.error(f"❌ Failed to load {cog_class_name}: {cog_error}", exc_info=True)

        # Count total registered commands across all cogs
        total_commands = 0
        for cog in self.cogs.values():
            total_commands += len(list(cog.get_commands()))

        # Generate Rich visual report
        from bot.utils.logging_helper import log_commands_setup

        log_commands_setup(self.console, command_modules, command_cogs, total_commands)

        # Log summary at appropriate level
        successful_modules = sum(1 for _, success in command_modules if success)
        failed_modules = sum(1 for _, success in command_modules if not success)
        successful_cogs = sum(1 for _, success in command_cogs if success)
        failed_cogs = sum(1 for _, success in command_cogs if not success)

        if failed_modules > 0 or failed_cogs > 0:
            self.logger.warning(f"⚠️ Command setup completed with failures: {failed_modules + failed_cogs} failed")
        else:
            self.logger.info(f"✅ Command setup completed successfully: {successful_modules + successful_cogs} loaded")

    async def connect(self, *, reconnect: bool = True) -> None:
        """Connect to Discord."""
        try:
            await super().connect(reconnect=reconnect)
        except discord.ConnectionClosed as e:
            self.logger.exception(f"Connection closed: {e}")
            # Attempt to reconnect after a delay
            await asyncio.sleep(5)
            await self.connect(reconnect=reconnect)

    async def close(self) -> None:
        """Clean up resources before shutdown."""
        self.logger.info("Bot is shutting down...")

        try:
            # CRITICAL: Close Discord connection FIRST to stop heartbeat thread
            # This prevents the "Event loop is closed" error from heartbeat thread
            if not self.is_closed():
                self.logger.info("Closing Discord connection...")
                try:
                    # Use a timeout to prevent hanging on Discord close
                    await asyncio.wait_for(super().close(), timeout=8.0)
                    self.logger.info("Discord connection closed successfully")
                except TimeoutError:
                    # Flipping the internal flag alone does not actually tear
                    # down a wedged socket/heartbeat -- observed in production
                    # leaving a zombie process that required a manual SIGKILL.
                    # The important state (profiles, context) was already
                    # persisted by the shutdown steps before close() runs, so
                    # it's safe to just end the process here. [REH]
                    self.logger.critical("Discord close timed out; connection is wedged -- forcing process exit instead of leaving a zombie for SIGKILL")
                    if hasattr(self, "_closed"):
                        self._closed = True
                    os._exit(1)
                except (RuntimeError, OSError) as e:
                    self.logger.warning(f"Error closing Discord connection: {e}")

            # Cancel user message processors via MessageProcessor
            if self.message_processor:
                await self.message_processor.shutdown()

            # Cancel memory save task
            if self.memory_save_task:
                try:
                    # Handle both Task and Loop objects
                    if hasattr(self.memory_save_task, "done"):
                        # It's an asyncio.Task
                        if not self.memory_save_task.done():
                            self.memory_save_task.cancel()
                            with suppress(TimeoutError, asyncio.CancelledError):
                                await asyncio.wait_for(self.memory_save_task, timeout=2.0)
                    elif hasattr(self.memory_save_task, "is_being_cancelled"):
                        # It's a tasks.Loop
                        if not self.memory_save_task.is_being_cancelled():
                            self.memory_save_task.cancel()
                            self.logger.debug("Cancelled memory save task loop")
                    else:
                        # Unknown type, try to cancel anyway
                        self.memory_save_task.cancel()
                except (RuntimeError, OSError, asyncio.CancelledError) as e:
                    self.logger.warning(f"Error cancelling memory save task: {e}")

            # Cancel tracked background tasks
            for task in list(self._background_tasks):
                if not task.done():
                    task.cancel()

            # Wait for background tasks to complete with timeout
            if self._background_tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*self._background_tasks, return_exceptions=True),
                        timeout=3.0,
                    )
                except TimeoutError:
                    self.logger.warning("Background tasks did not complete within timeout")

            # Stop memory distiller service
            try:
                await stop_memory_distiller()
            except (RuntimeError, OSError, AttributeError) as e:
                self.logger.warning(f"Error stopping memory distiller service: {e}")

            # Stop curated memory service
            try:
                await stop_memory_service()
            except (RuntimeError, OSError, AttributeError) as e:
                self.logger.warning(f"Error stopping curated memory service: {e}")

            # Stop server archive service
            try:
                await stop_server_archive_service()
            except (RuntimeError, OSError, AttributeError) as e:
                self.logger.warning(f"Error stopping server archive service: {e}")

            # Stop dashboard server
            try:
                dashboard = getattr(self, "_dashboard_server", None)
                if dashboard:
                    await dashboard.stop()
            except (RuntimeError, OSError, AttributeError) as e:
                self.logger.warning(f"Error stopping dashboard server: {e}")

            # Close Vision Orchestrator (if initialized either on bot or via router fallback)
            try:
                vo = getattr(self, "vision_orchestrator", None)
                if not vo and getattr(self, "router", None):
                    vo = getattr(self.router, "_vision_orchestrator", None)
                if vo:
                    try:
                        await asyncio.wait_for(vo.close(), timeout=5.0)
                        self.logger.info("VisionOrchestrator: closed")
                    except TimeoutError:
                        self.logger.warning("VisionOrchestrator close timed out")
                    except (RuntimeError, OSError, AttributeError) as e:
                        self.logger.warning(f"VisionOrchestrator close error: {e}")
            except (RuntimeError, OSError, AttributeError) as exc:
                self.logger.debug(f"VisionOrchestrator shutdown error: {exc}")

            # Close TTS manager
            if self.tts_manager:
                try:
                    await asyncio.wait_for(self.tts_manager.close(), timeout=2.0)
                except TimeoutError:
                    self.logger.warning("TTS manager close timed out")

            # Close global ollama client first
            try:
                ollama_client = None
                for module_name in ("bot.ollama", "bot.core.ollama"):
                    module = sys.modules.get(module_name)
                    if not module:
                        continue
                    candidate = getattr(module, "ollama_client", None)
                    if candidate:
                        ollama_client = candidate
                        break

                if ollama_client and hasattr(ollama_client, "close"):
                    self.logger.debug("Closing global ollama client")
                    await asyncio.wait_for(ollama_client.close(), timeout=2.0)
                else:
                    self.logger.debug("No global ollama client to close (module not loaded or disabled)")
            except (RuntimeError, OSError, AttributeError) as e:
                self.logger.debug(f"Error closing global ollama client: {e}")

            # Close aiohttp sessions
            await self._close_all_aiohttp_sessions()

            # Close shared HTTP client
            await cleanup_http_client()

            # Close web extraction service
            from bot.web_extraction_service import web_extractor

            await web_extractor.aclose()

            # Close the database connection
            if hasattr(self, "db") and self.db.is_connected:
                self.db.close()

            # Close any other resources
            if hasattr(self, "rag_system"):
                try:
                    await asyncio.wait_for(self.rag_system.close(), timeout=2.0)
                except TimeoutError:
                    self.logger.warning("RAG system close timed out")

            # Cancel remaining tasks with aggressive timeout
            current_task = asyncio.current_task()
            tasks = [t for t in asyncio.all_tasks() if t is not current_task and not t.done() and not t.cancelled()]

            if tasks:
                self.logger.info(f"Cancelling {len(tasks)} remaining background tasks")
                for task in tasks:
                    task.cancel()

                # Very short timeout for remaining tasks
                try:
                    await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=2.0)
                except TimeoutError:
                    self.logger.warning("Some tasks did not cancel within final timeout")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}", exc_info=True)
        finally:
            self.logger.info("Bot shutdown complete")

    async def _close_all_aiohttp_sessions(self) -> None:
        """Close all aiohttp sessions to prevent shutdown warnings."""
        try:
            self.logger.debug("Closing all aiohttp sessions...")

            # Close bot's own session if it exists
            if hasattr(self, "session") and self.session and not self.session.closed:
                self.logger.debug("Closing bot session")
                await self.session.close()
                await asyncio.sleep(0.05)  # Small delay for cleanup

            # Close ollama backend session
            if hasattr(self, "router") and self.router and hasattr(self.router, "ollama_backend") and self.router.ollama_backend:
                self.logger.debug("Closing ollama backend session")
                await self.router.ollama_backend.close()
                await asyncio.sleep(0.05)

            # Close any HTTP client sessions in various modules
            modules_to_check = [
                "ollama",  # Global ollama client
                "utils",  # Any utility HTTP clients
                "rag",  # RAG system HTTP clients
            ]

            for module_name in modules_to_check:
                try:
                    if hasattr(self, module_name):
                        module = getattr(self, module_name)
                        if hasattr(module, "close"):
                            self.logger.debug(f"Closing {module_name} HTTP sessions")
                            await module.close()
                            await asyncio.sleep(0.05)
                except (RuntimeError, OSError, AttributeError) as e:
                    self.logger.debug(f"Error closing {module_name} sessions: {e}")

            # Find and close any remaining aiohttp sessions using garbage collection
            import gc

            import aiohttp

            # Force garbage collection to expose any remaining sessions
            gc.collect()

            # Find any remaining ClientSession objects
            remaining_sessions = []
            for obj in gc.get_objects():
                if isinstance(obj, aiohttp.ClientSession) and not obj.closed:
                    remaining_sessions.append(obj)

            if remaining_sessions:
                self.logger.warning(f"Found {len(remaining_sessions)} unclosed aiohttp sessions, closing them")
                for session in remaining_sessions:
                    try:
                        await session.close()
                        await asyncio.sleep(0.01)
                    except (RuntimeError, OSError, AttributeError) as e:
                        self.logger.debug(f"Error closing remaining session: {e}")

            # Final garbage collection
            gc.collect()

            # Give more time for all sessions to properly close
            await asyncio.sleep(0.3)

            self.logger.debug("All aiohttp sessions closed")
        except (RuntimeError, OSError, AttributeError) as e:
            self.logger.warning(f"Error closing aiohttp sessions: {e}")
