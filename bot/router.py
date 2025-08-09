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
from typing import Any, Callable, Coroutine, Dict, Optional, TYPE_CHECKING, List, Awaitable
import time
import discord
import httpx
from html import unescape

# Import new modality system
from .modality import InputModality, InputItem, collect_input_items, map_item_to_modality
from .multimodal_retry import run_with_retries
from .result_aggregator import ResultAggregator
from .x_api_client import XApiClient
from .enhanced_retry import EnhancedRetryManager, ProviderConfig, get_retry_manager
from .brain import brain_infer
from .contextual_brain import contextual_brain_infer
import re
from . import web
from discord import Message, DMChannel, Embed, File
from .search.factory import get_search_provider
from .search.types import SearchQueryParams, SafeSearch, SearchResult

if TYPE_CHECKING:
    from bot.core.bot import LLMBot as DiscordBot
    from bot.metrics import Metrics
    from .command_parser import ParsedCommand

logger = get_logger(__name__)

# Local application imports
from .action import BotAction, ResponseMessage
from .command_parser import Command, parse_command
from .exceptions import DispatchEmptyError, DispatchTypeError, APIError
from .hear import hear_infer, hear_infer_from_url
from .pdf_utils import PDFProcessor
from .see import see_infer
from .web import process_url
from .utils.mention_utils import ensure_single_mention
from .web_extraction_service import web_extractor
from .utils.file_utils import download_file

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
        # Lazy-initialized X API client
        self._x_api_client: Optional[XApiClient] = None
        # Tweet syndication cache and locks [CA][PA]
        self._syn_cache: Dict[str, Dict[str, Any]] = {}
        self._syn_locks: Dict[str, asyncio.Lock] = {}
        try:
            self._syn_ttl_s: float = float(self.config.get("X_SYNDICATION_TTL_S", 900))
        except Exception:
            self._syn_ttl_s = 900.0

    async def _get_x_api_client(self) -> Optional[XApiClient]:
        """Create or return a cached XApiClient based on config. [CA][IV]"""
        cfg = self.config
        if not cfg.get("X_API_ENABLED", False):
            return None
        token = cfg.get("X_API_BEARER_TOKEN")
        if not token:
            return None
        if self._x_api_client is None:
            try:
                self._x_api_client = XApiClient(
                    bearer_token=token,
                    timeout_ms=int(cfg.get("X_API_TIMEOUT_MS", 8000)),
                    default_tweet_fields=cfg.get("X_TWEET_FIELDS", []),
                    default_expansions=cfg.get("X_EXPANSIONS", []),
                    default_media_fields=cfg.get("X_MEDIA_FIELDS", []),
                    default_user_fields=cfg.get("X_USER_FIELDS", []),
                    default_poll_fields=cfg.get("X_POLL_FIELDS", []),
                    default_place_fields=cfg.get("X_PLACE_FIELDS", []),
                )
            except Exception as e:
                self.logger.error(f"Failed to initialize XApiClient: {e}")
                self._x_api_client = None
        return self._x_api_client

    async def _get_tweet_via_syndication(self, tweet_id: str) -> Optional[Dict[str, Any]]:
        """Fetch tweet via X/Twitter syndication CDN with TTL cache and per-ID concurrency.
        Endpoint shape: https://cdn.syndication.twimg.com/widgets/tweet?id={id}
        Returns parsed JSON dict on success or None on failure. [PA][REH]
        """
        if not tweet_id:
            return None
        # Check cache
        now = time.time()
        cached = self._syn_cache.get(tweet_id)
        if cached and (now - float(cached.get("ts", 0))) < self._syn_ttl_s:
            if cached.get("neg"):
                self._metric_inc("x.syndication.neg_cache_hit", None)
                return None
            self._metric_inc("x.syndication.cache_hit", None)
            return cached.get("data")

        # Per-ID lock to avoid thundering herd
        lock = self._syn_locks.get(tweet_id)
        if lock is None:
            lock = asyncio.Lock()
            self._syn_locks[tweet_id] = lock
        async with lock:
            # Check cache again inside lock
            cached = self._syn_cache.get(tweet_id)
            if cached and (now - float(cached.get("ts", 0))) < self._syn_ttl_s:
                if cached.get("neg"):
                    self._metric_inc("x.syndication.neg_cache_hit_locked", None)
                    return None
                self._metric_inc("x.syndication.cache_hit_locked", None)
                return cached.get("data")

            timeout_ms = int(self.config.get("X_SYNDICATION_TIMEOUT_MS", 4000))
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
                ),
                "Accept": "application/json, text/javascript;q=0.9, */*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://platform.twitter.com/",
            }
            params_variants = [
                ("widgets", {"id": tweet_id, "lang": "en"}),
                ("tweet-result", {"id": tweet_id, "lang": "en"}),
                ("widgets", {"id": tweet_id, "lang": "en", "dnt": "false"}),
            ]
            base = "https://cdn.syndication.twimg.com/"
            data = None
            try:
                async with httpx.AsyncClient(timeout=timeout_ms / 1000.0) as client:
                    for endpoint, params in params_variants:
                        url = base + ("widgets/tweet" if endpoint == "widgets" else "tweet-result")
                        self._metric_inc("x.syndication.fetch", {"endpoint": endpoint})
                        resp = await client.get(url, headers=headers, params=params)
                        if resp.status_code != 200:
                            self.logger.info(
                                "Syndication non-200",
                                extra={"detail": {"tweet_id": tweet_id, "status": resp.status_code, "endpoint": endpoint}},
                            )
                            self._metric_inc("x.syndication.non_200", {"status": str(resp.status_code), "endpoint": endpoint})
                            continue
                        try:
                            data = resp.json()
                        except Exception:
                            self._metric_inc("x.syndication.invalid_json", {"endpoint": endpoint})
                            continue
                        # Found a candidate
                        break
                    # If no usable JSON with text/full_text, try oEmbed as last resort
                    if not (isinstance(data, dict) and (data.get("text") or data.get("full_text"))):
                        oembed_url = "https://publish.twitter.com/oembed"
                        oembed_params = {
                            "url": f"https://twitter.com/i/status/{tweet_id}",
                            "dnt": "false",
                            "omit_script": "true",
                            "hide_thread": "true",
                            "lang": "en",
                        }
                        self._metric_inc("x.syndication.fetch", {"endpoint": "oembed"})
                        resp = await client.get(oembed_url, headers=headers, params=oembed_params)
                        if resp.status_code == 200:
                            try:
                                obj = resp.json()
                            except Exception:
                                obj = None
                            if isinstance(obj, dict):
                                html = obj.get("html")
                                if html:
                                    # Very light HTML → text conversion
                                    txt = re.sub(r"<br\\s*/?>", "\n", html)
                                    txt = re.sub(r"<[^>]+>", "", txt)
                                    txt = unescape(txt).strip()
                                    if txt:
                                        data = {
                                            "text": txt,
                                            "user": {"name": obj.get("author_name")},
                                        }
                        # Try x.com oembed variant if still no data
                        if not (isinstance(data, dict) and (data.get("text") or data.get("full_text"))):
                            oembed_params_x = dict(oembed_params)
                            oembed_params_x["url"] = f"https://x.com/i/status/{tweet_id}"
                            self._metric_inc("x.syndication.fetch", {"endpoint": "oembed_x"})
                            resp2 = await client.get(oembed_url, headers=headers, params=oembed_params_x)
                            if resp2.status_code == 200:
                                try:
                                    obj2 = resp2.json()
                                except Exception:
                                    obj2 = None
                                if isinstance(obj2, dict):
                                    html2 = obj2.get("html")
                                    if html2:
                                        txt2 = re.sub(r"<br\\s*/?>", "\n", html2)
                                        txt2 = re.sub(r"<[^>]+>", "", txt2)
                                        txt2 = unescape(txt2).strip()
                                        if txt2:
                                            data = {
                                                "text": txt2,
                                                "user": {"name": obj2.get("author_name")},
                                            }
            except Exception as e:
                self.logger.info(
                    "Syndication fetch failed",
                    extra={"detail": {"tweet_id": tweet_id, "error": str(e)}},
                )
                self._metric_inc("x.syndication.error", None)
                return None

            # Minimal validation: require text field
            if not isinstance(data, dict) or not (data.get("text") or data.get("full_text")):
                self._metric_inc("x.syndication.invalid", None)
                # Negative cache to avoid repeated hits for unavailable/blocked tweets
                self._syn_cache[tweet_id] = {"neg": True, "ts": time.time()}
                self._metric_inc("x.syndication.neg_store", None)
                return None

            # Cache and return
            self._syn_cache[tweet_id] = {"data": data, "ts": time.time()}
            self._metric_inc("x.syndication.success", None)
            return data

    def _format_syndication_result(self, syn_data: Dict[str, Any], url: str) -> str:
        """Format Syndication JSON tweet into concise text similar to API format. [PA]"""
        try:
            text = (syn_data.get("text") or syn_data.get("full_text") or "").strip()
            user = syn_data.get("user") or {}
            username = user.get("screen_name") or user.get("name")
            created_at = syn_data.get("created_at") or syn_data.get("date_created")
            photos = syn_data.get("photos") or []
            media_hint = f" • media:{len(photos)}" if photos else ""
            prefix = f"@{username}" if username else "Tweet"
            stamp = f" • {created_at}" if created_at else ""
            body = text if len(text) <= 4000 else (text[:3990] + "…")
            return f"{prefix}{stamp}{media_hint} → {url}\n{body}"
        except Exception:
            return f"Tweet → {url}\n{str(syn_data)[:4000]}"

    @staticmethod
    def _is_twitter_url(url: str) -> bool:
        try:
            u = str(url).lower()
        except Exception:
            return False
        return any(d in u for d in ["twitter.com/", "x.com/", "vxtwitter.com/", "fxtwitter.com/"])

    def _format_x_tweet_result(self, api_data: Dict[str, Any], url: str) -> str:
        """Format X API tweet response into concise text. [PA][IV]"""
        try:
            data = api_data.get("data") or {}
            text = (data.get("text") or "").strip()
            created_at = data.get("created_at")
            author_id = data.get("author_id")
            username = None
            includes = api_data.get("includes") or {}
            for u in includes.get("users", []) or []:
                if u.get("id") == author_id:
                    username = u.get("username") or u.get("name")
                    break
            # Minimal media hint
            media = includes.get("media") or []
            media_hint = f" • media:{len(media)}" if media else ""
            prefix = f"@{username}" if username else "Tweet"
            stamp = f" • {created_at}" if created_at else ""
            body = text if len(text) <= 4000 else (text[:3990] + "…")
            return f"{prefix}{stamp}{media_hint} → {url}\n{body}"
        except Exception:
            # Fallback to raw dump if unexpected structure
            return f"Tweet → {url}\n{str(api_data)[:4000]}"

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
        """Single source-of-truth gate: decide if this message should be processed.
        Cheap, synchronous, and config-driven. No network or heavy CPU allowed here.
        """
        cfg = self.config
        owners: list[int] = cfg.get("OWNER_IDS", [])
        triggers: list[str] = cfg.get("REPLY_TRIGGERS", [
            "dm", "mention", "reply", "bot_threads", "owner", "command_prefix"
        ])

        # Master switch: if disabled, allow everything (legacy behavior)
        if not cfg.get("BOT_SPEAKS_ONLY_WHEN_SPOKEN_TO", True):
            self.logger.debug(
                f"gate.allow | reason=master_switch_off msg_id={message.id}",
                extra={"event": "gate.allow", "reason": "master_switch_off", "msg_id": message.id},
            )
            self._metric_inc("gate.allowed", {"reason": "master_switch_off"})
            return True

        content = (message.content or "").strip()
        is_dm = isinstance(message.channel, DMChannel)
        is_mentioned = self.bot.user in message.mentions if hasattr(message, "mentions") else False
        is_reply = self._is_reply_to_bot(message)
        is_owner = message.author.id in owners if getattr(message, "author", None) else False

        in_bot_thread = False
        try:
            if isinstance(message.channel, discord.Thread):
                # Cheap ownership check only; do not fetch history here.
                in_bot_thread = (getattr(message.channel, "owner_id", None) == self.bot.user.id)
        except Exception:
            in_bot_thread = False

        # Prefix command detection (strip leading mention if present)
        command_prefix = cfg.get("COMMAND_PREFIX", "!")
        if content:
            mention_pattern = fr'^<@!?{self.bot.user.id}>\s*'
            clean_content = re.sub(mention_pattern, "", content)
        else:
            clean_content = ""
        has_prefix = bool(clean_content.startswith(command_prefix)) if clean_content else False

        # Evaluate triggers
        if is_owner and "owner" in triggers:
            self.logger.info(
                f"gate.allow | reason=owner_override msg_id={message.id}",
                extra={"event": "gate.allow", "reason": "owner_override", "user_id": message.author.id, "msg_id": message.id},
            )
            self._metric_inc("gate.allowed", {"reason": "owner_override"})
            return True

        if is_dm and "dm" in triggers:
            self.logger.debug(
                f"gate.allow | reason=dm msg_id={message.id}",
                extra={"event": "gate.allow", "reason": "dm", "msg_id": message.id},
            )
            self._metric_inc("gate.allowed", {"reason": "dm"})
            return True

        if is_mentioned and "mention" in triggers:
            self.logger.debug(
                f"gate.allow | reason=mention msg_id={message.id}",
                extra={"event": "gate.allow", "reason": "mention", "msg_id": message.id},
            )
            self._metric_inc("gate.allowed", {"reason": "mention"})
            return True

        if is_reply and "reply" in triggers:
            self.logger.debug(
                f"gate.allow | reason=reply_to_bot msg_id={message.id}",
                extra={"event": "gate.allow", "reason": "reply_to_bot", "msg_id": message.id},
            )
            self._metric_inc("gate.allowed", {"reason": "reply_to_bot"})
            return True

        if in_bot_thread and "bot_threads" in triggers:
            self.logger.debug(
                f"gate.allow | reason=bot_thread msg_id={message.id}",
                extra={"event": "gate.allow", "reason": "bot_thread", "msg_id": message.id},
            )
            self._metric_inc("gate.allowed", {"reason": "bot_thread"})
            return True

        if has_prefix and "command_prefix" in triggers:
            self.logger.debug(
                f"gate.allow | reason=command_prefix msg_id={message.id}",
                extra={"event": "gate.allow", "reason": "command_prefix", "msg_id": message.id},
            )
            self._metric_inc("gate.allowed", {"reason": "command_prefix"})
            return True

        # Do NOT allow on mere presence of URLs (e.g., twitter) – must be addressed first
        self.logger.info(
            f"gate.block | reason=not_addressed msg_id={message.id}",
            extra={
                "event": "gate.block",
                "reason": "not_addressed",
                "msg_id": message.id,
                "guild_id": getattr(message.guild, 'id', None),
                "is_dm": is_dm,
            },
        )
        self._metric_inc("gate.blocked", {"reason": "not_addressed"})
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
            # 1. Quick pre-filter: Only parse commands for messages that start with '!' to avoid unnecessary parsing
            content = message.content.strip()
            
            # Remove bot mention to check for command pattern
            mention_pattern = fr'^<@!?{self.bot.user.id}>\s*'
            clean_content = re.sub(mention_pattern, '', content)
            
            # Only parse if it looks like a command (starts with '!')
            if clean_content.startswith('!'):
                parsed_command = parse_command(message, self.bot)
                
                # 2. If a command is found, delegate it to the command processor (cogs).
                if parsed_command:
                    self.logger.info(f"Found command '{parsed_command.command.name}', delegating to cog. (msg_id: {message.id})")
                    return BotAction(meta={'delegated_to_cog': True})
                # If it starts with '!' but isn't a known command, let it continue to normal processing
                self.logger.debug(f"Unknown command pattern ignored: {clean_content.split()[0] if clean_content else '(empty)'} (msg_id: {message.id})")

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
                result_action = await self._process_multimodal_message_internal(message, context_str)
                return result_action  # Return the actual processing result

        except Exception as e:
            self.logger.error(f"❌ Error in router dispatch: {e} (msg_id: {message.id})", exc_info=True)
            return BotAction(content="I encountered an error while processing your message.", error=True)

    async def _process_multimodal_message_internal(self, message: Message, context_str: str) -> Optional[BotAction]:
        """
        Process all input items from a message sequentially with result aggregation.
        Follows the 1 IN → 1 OUT rule by combining all results into a single response.
        Returns the BotAction instead of executing it directly.
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
        
        # Resolve inline [search(...)] directives inside the remaining text
        try:
            original_text = await self._resolve_inline_searches(original_text, message)
        except Exception as e:
            self.logger.error(f"Inline search resolution failed: {e} (msg_id: {message.id})", exc_info=True)
        
        # If no items found, process as text-only
        if not items:
            # No actionable items found, treat as text-only
            response_action = await self._invoke_text_flow(original_text, message, context_str)
            if response_action and response_action.has_payload:
                self.logger.info(f"✅ Text-only response generated successfully (msg_id: {message.id})")
                return response_action
            else:
                self.logger.warning(f"No response generated from text-only flow (msg_id: {message.id})")
                return None
        
        self.logger.info(f"🚶 Processing {len(items)} input items SEQUENTIALLY for deterministic order (msg_id: {message.id})")
        
        # Initialize result aggregator and retry manager
        aggregator = ResultAggregator()
        retry_manager = get_retry_manager()
        
        # Per-item budgets
        # LLM/vision tasks can be shorter; media (yt-dlp/transcribe) needs more time. [PA]
        LLM_PER_ITEM_BUDGET = float(os.environ.get('MULTIMODAL_PER_ITEM_BUDGET', '30.0'))
        MEDIA_PER_ITEM_BUDGET = float(os.environ.get('MEDIA_PER_ITEM_BUDGET', '120.0'))
        
        # Process items strictly sequentially for determinism [CA]
        start_time = time.time()
        for i, item in enumerate(items, start=1):
            modality = await map_item_to_modality(item)
            # Create description for logging
            if item.source_type == "attachment":
                description = f"{item.payload.filename}"
            elif item.source_type == "url":
                description = f"URL: {item.payload[:30]}{'...' if len(item.payload) > 30 else ''}"
            else:
                description = f"{item.source_type}"

            self.logger.info(f"📋 Starting item {i}: {modality.name} - {description}")

            # Determine modality type for retry manager and per-item budget
            if modality in [InputModality.SINGLE_IMAGE, InputModality.MULTI_IMAGE]:
                retry_modality = "vision"
                selected_budget = LLM_PER_ITEM_BUDGET
            elif modality in [InputModality.VIDEO_URL, InputModality.AUDIO_VIDEO_FILE]:
                retry_modality = "media"
                selected_budget = MEDIA_PER_ITEM_BUDGET
            elif modality in [InputModality.PDF_DOCUMENT, InputModality.PDF_OCR]:
                retry_modality = "media"
                selected_budget = MEDIA_PER_ITEM_BUDGET
            else:
                retry_modality = "text"
                selected_budget = LLM_PER_ITEM_BUDGET

            # Create coroutine factory for this item
            def create_handler_coro(provider_config: ProviderConfig):
                async def handler_coro():
                    return await self._handle_item_with_provider(item, modality, provider_config)
                return handler_coro

            try:
                result = await retry_manager.run_with_fallback(
                    modality=retry_modality,
                    coro_factory=create_handler_coro,
                    per_item_budget=selected_budget,
                )

                if result.success:
                    self.logger.info(f"✅ Item {i} completed successfully ({result.total_time:.2f}s)")
                    success = True
                    result_text = result.result
                    duration = result.total_time
                    attempts = result.attempts
                else:
                    msg = f"❌ Failed after {result.attempts} attempts: {result.error}"
                    if result.fallback_occurred:
                        msg += " (fallback attempted)"
                    self.logger.warning(f"❌ Item {i} failed ({result.total_time:.2f}s)")
                    success = False
                    result_text = msg
                    duration = result.total_time
                    attempts = result.attempts
            except Exception as e:
                self.logger.error(f"❌ Item {i} exception: {e}")
                success = False
                result_text = f"❌ Exception: {e}"
                duration = 0.0
                attempts = 0

            aggregator.add_result(
                item_index=i,
                item=item,
                modality=modality,
                result_text=result_text,
                success=success,
                duration=duration,
                attempts=attempts,
            )

        total_time = time.time() - start_time
        # Generate aggregated prompt and send single response
        aggregated_prompt = aggregator.get_aggregated_prompt(original_text)

        # Log summary statistics
        stats = aggregator.get_summary_stats()
        successful_items = stats.get('successful_items', 0)
        total_items = stats.get('total_items', 0)
        self.logger.info(
            f"📦 SEQUENTIAL MULTIMODAL COMPLETE: {successful_items}/{total_items} successful, total: {total_time:.1f}s"
        )

        # Generate single aggregated response through text flow (1 IN → 1 OUT)
        if aggregated_prompt.strip():
            response_action = await self._invoke_text_flow(aggregated_prompt, message, context_str)
            if response_action and response_action.has_payload:
                self.logger.info(f"✅ Multimodal response generated successfully (msg_id: {message.id})")
                return response_action
            else:
                self.logger.warning(f"No response generated from text flow (msg_id: {message.id})")
                return None
        else:
            self.logger.warning(f"No content to process after multimodal aggregation (msg_id: {message.id})")
            return None

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
                # Prefer embed.image.url, then embed.thumbnail.url, but only if valid [IV]
                try:
                    img = getattr(item.payload, 'image', None)
                    if img and getattr(img, 'url', None):
                        return await self._process_image_from_url(img.url)
                    thumb = getattr(item.payload, 'thumbnail', None)
                    if thumb and getattr(thumb, 'url', None):
                        return await self._process_image_from_url(thumb.url)
                except Exception as _e:
                    self.logger.debug(f"Embed URL extraction failed: {_e}")
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
                # Prefer embed.image.url, then embed.thumbnail.url, but only if valid [IV]
                try:
                    img = getattr(item.payload, 'image', None)
                    if img and getattr(img, 'url', None):
                        return await self._process_image_from_url(img.url)
                    thumb = getattr(item.payload, 'thumbnail', None)
                    if thumb and getattr(thumb, 'url', None):
                        return await self._process_image_from_url(thumb.url)
                except Exception as _e:
                    self.logger.debug(f"Embed URL extraction failed (override): {_e}")
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
        """Process image from URL using screenshot API + vision analysis."""
        from .utils.external_api import external_screenshot
        from .see import see_infer
        
        try:
            # Validate URL before attempting screenshot [IV]
            if not url or not isinstance(url, str) or not re.match(r'^https?://', url):
                self.logger.warning(f"⚠️ Skipping screenshot: invalid URL: {url}")
                return "⚠️ Skipping screenshot: invalid or missing image URL."

            # Take screenshot using the configured screenshot API
            self.logger.info(f"📸 Taking screenshot of URL: {url}")
            screenshot_path = await external_screenshot(url)
            
            if not screenshot_path:
                self.logger.error(f"❌ Failed to capture screenshot of URL: {url}")
                return f"⚠️ Failed to capture screenshot of URL: {url}"
            
            # Process the screenshot with vision model
            self.logger.info(f"👁️ Processing screenshot with vision model: {screenshot_path}")
            vision_result = await see_infer(image_path=screenshot_path, prompt="Describe the contents of this screenshot")
            
            if vision_result and hasattr(vision_result, 'content') and vision_result.content:
                analysis = vision_result.content
                self.logger.info(f"✅ Screenshot analysis completed: {len(analysis)} chars")
                return f"Screenshot analysis of {url}: {analysis}"
            else:
                self.logger.warning(f"⚠️ Vision analysis returned empty result for: {screenshot_path}")
                return f"⚠️ Screenshot captured but vision analysis failed for: {url}"
                
        except Exception as e:
            self.logger.error(f"❌ Error in screenshot + vision processing: {e}", exc_info=True)
            return f"⚠️ Failed to process screenshot of URL: {url} (Error: {str(e)})"
    
    async def _vl_describe_image_from_url(self, image_url: str, *, prompt: Optional[str] = None, model_override: Optional[str] = None) -> Optional[str]:
        """
        Download an image from a direct URL and run VL inference. Returns text or None.
        [IV][RM][REH]
        """
        if not image_url or not isinstance(image_url, str) or not re.match(r'^https?://', image_url):
            self.logger.warning(f"⚠️ Invalid image URL for VL: {image_url}")
            return None
        suffix = ".jpg"
        try:
            # Infer extension if present
            m = re.search(r"\.(jpg|jpeg|png|webp)(?:\?|$)", image_url, re.IGNORECASE)
            if m:
                ext = m.group(1).lower()
                suffix = f".{ext if ext != 'jpeg' else 'jpg'}"
        except Exception:
            pass
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_path = tmp_file.name
            ok = await download_file(image_url, Path(tmp_path))
            if not ok:
                self.logger.error(f"❌ Failed to download image for VL: {image_url}")
                return None
            vl_prompt = prompt or "Describe this image in detail. Focus on salient objects, text, and context."
            res = await see_infer(image_path=tmp_path, prompt=vl_prompt, model_override=model_override)
            if res and getattr(res, 'content', None):
                return str(res.content).strip()
            self.logger.warning(f"⚠️ VL returned empty content for: {image_url}")
            return None
        except Exception as e:
            self.logger.error(f"❌ VL describe failed for {image_url}: {e}", exc_info=True)
            return None
        finally:
            try:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception:
                pass
    
    async def _handle_video_url(self, item: InputItem) -> str:
        """
        Handle video URL input items (YouTube, TikTok, etc.).
        For Twitter/X URLs: tries yt-dlp first, routes non-video posts to the tiered WebExtractionService (no auto-screenshot).
        Returns transcribed text for further processing.
        """
        from .video_ingest import VideoIngestError
        from .exceptions import InferenceError
        
        url = item.payload
        self.logger.info(f"🎥 Processing video URL: {url}")
        
        # For Twitter/X URLs, implement fallback logic
        is_twitter = re.match(r'https?://(?:www\.)?(?:twitter|x|fxtwitter|vxtwitter)\.com/', url)
        
        try:
            # Try video/audio extraction first
            result = await hear_infer_from_url(url)
            if result and result.get('transcription'):
                transcription = result['transcription']
                metadata = result.get('metadata', {})
                title = metadata.get('title', 'Unknown')
                
                return f"Video transcription from {url} ('{title}'): {transcription}"
            else:
                return f"Could not transcribe audio from video: {url}"
            
        except VideoIngestError as ve:
            error_str = str(ve).lower()
            
            # For Twitter URLs with no media, route to tiered web extractor instead of screenshot [SST]
            if is_twitter and (
                "no video or audio content found" in error_str or
                "no video could be found" in error_str or
                "failed to download video" in error_str
            ):
                self.logger.info(f"🐦 No video in Twitter URL; routing to tiered extractor: {url}")
                extract_res = await web_extractor.extract(url)
                if extract_res.success:
                    return f"Twitter post content:\n{extract_res.to_message()}"
                else:
                    return "🔍 No video or audio content found in this URL, and text extraction was unsuccessful."
            
            # For non-Twitter URLs, provide user-friendly message  
            self.logger.info(f"ℹ️ Video processing: {ve}")
            return f"⚠️ {str(ve)}"
            
        except InferenceError as ie:
            # InferenceError already has user-friendly messages
            self.logger.info(f"ℹ️ Video inference: {ie}")
            return f"⚠️ {str(ie)}"
            
        except Exception as e:
            # Handle any other unexpected errors gracefully
            error_str = str(e).lower()
            self.logger.error(f"❌ Unexpected video processing error: {e}", exc_info=True)
            
            # For Twitter URLs, attempt tiered extractor (no screenshot fallback)
            if is_twitter:
                self.logger.info(f"🐦 Attempting tiered extractor due to unexpected error: {url}")
                extract_res = await web_extractor.extract(url)
                if extract_res.success:
                    return f"Twitter post content:\n{extract_res.to_message()}"
                else:
                    return "⚠️ Could not process this Twitter URL as video; text extraction also failed."
            
            return f"⚠️ Video processing failed: {str(e)}"

    async def _handle_audio_video_file(self, item: InputItem) -> str:
        """
        Handle audio/video file attachments.
        Returns transcribed text for further processing.
        """
        from .video_ingest import VideoIngestError
        from .exceptions import InferenceError
        
        attachment = item.payload
        self.logger.info(f"🎵 Processing audio/video file: {attachment.filename}")
        
        try:
            result = await hear_infer(attachment)
            return result
        except VideoIngestError as ve:
            self.logger.error(f"❌ Audio/video file ingestion failed: {ve}")
            return f"⚠️ {str(ve)}"
        except InferenceError as ie:
            self.logger.error(f"❌ Audio/video inference failed: {ie}")
            return f"⚠️ {str(ie)}"
        except Exception as e:
            self.logger.error(f"❌ Audio/video file processing failed: {e}", exc_info=True)
            return f"⚠️ Could not process this audio/video file: {str(e)}"
    
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
        No auto-screenshot fallback here; screenshots require explicit !ss command.
        """
        try:
            if item.source_type != "url":
                return f"URL handler received non-URL item: {item.source_type}"
            
            url = item.payload
            self.logger.info(f"🌐 Processing general URL: {url}")

            # API-first for Twitter/X posts [CA][SFT]
            if self._is_twitter_url(url):
                cfg = self.config
                require_api = bool(cfg.get("X_API_REQUIRE_API_FOR_TWITTER", False))
                allow_fallback_5xx = bool(cfg.get("X_API_ALLOW_FALLBACK_ON_5XX", True))
                syndication_enabled = bool(cfg.get("X_SYNDICATION_ENABLED", True))
                # Keep API-first by default; can opt-in to syndication-first
                syndication_first = bool(cfg.get("X_SYNDICATION_FIRST", False))
                tweet_id = XApiClient.extract_tweet_id(str(url))
                x_client = await self._get_x_api_client()

                # Tier 1: Syndication JSON (cache + concurrency) when allowed and preferred [PA][REH]
                if tweet_id and syndication_enabled and not require_api and (syndication_first or x_client is None):
                    syn = await self._get_tweet_via_syndication(tweet_id)
                    if syn and syn.get("text"):
                        self._metric_inc("x.syndication.hit", None)
                        # Basic media handling: photos → optional VL
                        photos = syn.get("photos") or []
                        base = self._format_syndication_result(syn, url)
                        if not photos:
                            return base
                        # Respect photo-to-VL flag
                        route_photos = bool(cfg.get("X_API_ROUTE_PHOTOS_TO_VL", False))
                        if not route_photos:
                            return f"{base}\nPhotos: {len(photos)}"
                        descriptions: List[str] = []
                        successes = 0
                        failures = 0
                        for idx, p in enumerate(photos, start=1):
                            purl = p.get("url") or p.get("image_url") or p.get("src")
                            if not purl:
                                failures += 1
                                descriptions.append(f"📷 Photo {idx}/{len(photos)} — no URL available")
                                continue
                            self._metric_inc("x.photo_to_vl.attempt", {"idx": str(idx)})
                            prompt = (
                                f"This is photo {idx} of {len(photos)} from a tweet: {url}. "
                                f"Describe it clearly and succinctly, including any visible text."
                            )
                            desc = await self._vl_describe_image_from_url(purl, prompt=prompt)
                            if desc:
                                successes += 1
                                self._metric_inc("x.photo_to_vl.success", {"idx": str(idx)})
                                descriptions.append(f"📷 Photo {idx}/{len(photos)}\n{desc}")
                            else:
                                failures += 1
                                self._metric_inc("x.photo_to_vl.failure", {"idx": str(idx)})
                                descriptions.append(f"📷 Photo {idx}/{len(photos)} — analysis unavailable")
                        header = f"{base}\nPhotos analyzed: {successes}/{len(photos)}"
                        return f"{header}\n\n" + "\n\n".join(descriptions)

                # Tier 2 (optionally before API if syndication_first): X API [SFT]
                if tweet_id and x_client is not None:
                    try:
                        api_data = await x_client.get_tweet_by_id(tweet_id)
                        includes = api_data.get("includes") or {}
                        media_list = includes.get("media") or []
                        media_types = {m.get("type") for m in media_list if isinstance(m, dict)}

                        if {"video", "animated_gif"} & media_types:
                            try:
                                stt_res = await hear_infer_from_url(url)
                                if stt_res and stt_res.get("transcription"):
                                    transcription = stt_res["transcription"]
                                    return f"Video/audio content from {url}: {transcription}"
                                return f"Detected media in {url} but transcription failed."
                            except Exception as stt_err:
                                self.logger.error(
                                    f"X media STT route failed for {url}: {stt_err}",
                                    extra={"detail": {"url": url}},
                                )
                                return f"Detected media in {url} but could not process it right now."

                        if media_types == {"photo"} or ("photo" in media_types and len(media_types) == 1):
                            route_photos = bool(cfg.get("X_API_ROUTE_PHOTOS_TO_VL", False))
                            photos = [m for m in media_list if isinstance(m, dict) and m.get("type") == "photo"]
                            base = self._format_x_tweet_result(api_data, url)
                            if not route_photos:
                                photo_count = len(photos)
                                self._metric_inc("x.photo_to_vl.skipped", {"enabled": "false"})
                                return f"{base}\nPhotos: {photo_count}"

                            self.logger.info(
                                "🖼️🐦 Routing X photos to VL",
                                extra={
                                    "event": "x.photo_to_vl.start",
                                    "detail": {
                                        "url": url,
                                        "photo_count": len(photos),
                                    },
                                },
                            )
                            self._metric_inc("x.photo_to_vl.enabled", None)
                            photo_urls: List[str] = []
                            for m in photos:
                                u = m.get("url") or m.get("preview_image_url")
                                if u:
                                    photo_urls.append(u)

                            if not photo_urls:
                                self.logger.warning("No photo URLs present in X API media; falling back to count")
                                self._metric_inc("x.photo_to_vl.no_urls", None)
                                return f"{base}\nPhotos: {len(photos)}"

                            descriptions: List[str] = []
                            successes = 0
                            failures = 0
                            for idx, purl in enumerate(photo_urls, start=1):
                                self.logger.info(f"🔎 VL analyzing X photo {idx}/{len(photo_urls)}: {purl}")
                                self._metric_inc("x.photo_to_vl.attempt", {"idx": str(idx)})
                                prompt = (
                                    f"This is photo {idx} of {len(photo_urls)} from a tweet: {url}. "
                                    f"Describe it clearly and succinctly, including any visible text."
                                )
                                desc = await self._vl_describe_image_from_url(purl, prompt=prompt)
                                if desc:
                                    successes += 1
                                    self._metric_inc("x.photo_to_vl.success", {"idx": str(idx)})
                                    descriptions.append(f"📷 Photo {idx}/{len(photo_urls)}\n{desc}")
                                else:
                                    failures += 1
                                    self._metric_inc("x.photo_to_vl.failure", {"idx": str(idx)})
                                    descriptions.append(f"📷 Photo {idx}/{len(photo_urls)} — analysis unavailable")

                            self.logger.info(
                                "🧩 X photo VL complete",
                                extra={
                                    "event": "x.photo_to_vl.done",
                                    "detail": {
                                        "url": url,
                                        "total": len(photo_urls),
                                        "ok": successes,
                                        "fail": failures,
                                    },
                                },
                            )
                            agg = "\n\n".join(descriptions)
                            header = f"{base}\nPhotos analyzed: {successes}/{len(photo_urls)}"
                            return f"{header}\n\n{agg}"

                        return self._format_x_tweet_result(api_data, url)
                    except APIError as e:
                        emsg = str(e)
                        if any(tok in emsg for tok in ["access denied", "not found", "deleted (", "unexpected status: 401", "unexpected status: 403", "unexpected status: 404", "unexpected status: 410"]):
                            self.logger.info("X API denied or content missing; not scraping due to policy", extra={"detail": {"url": url, "error": emsg}})
                            return "⚠️ This X post cannot be accessed via API (private/removed). Per policy, scraping is disabled."
                        if ("429" in emsg or "server error" in emsg) and (not require_api) and allow_fallback_5xx:
                            self.logger.warning("X API transient issue, falling back to generic extractor", extra={"detail": {"url": url, "error": emsg}})
                            # fall through to generic handling below
                        else:
                            self.logger.info("X API error without fallback; returning policy message", extra={"detail": {"url": url, "error": emsg}})
                            return "⚠️ Temporary issue accessing X API for this post. Please try again later."
                else:
                    if require_api:
                        return "⚠️ X posts require API access and cannot be scraped. Configure X_API_BEARER_TOKEN to enable."
                    # else fall through to generic handling
            
            # Use existing URL processing logic - process_url returns a dict
            url_result = await process_url(url)
            
            # Handle errors
            if not url_result or url_result.get('error'):
                return f"Could not extract content from URL: {url}"
            
            # Check if smart routing detected media and should route to yt-dlp
            route_to_ytdlp = url_result.get('route_to_ytdlp', False)
            if route_to_ytdlp:
                self.logger.info(f"🎥 Smart routing detected media in {url}, routing to yt-dlp flow")
                
                try:
                    # Process through yt-dlp flow
                    transcription_result = await hear_infer_from_url(url)
                    
                    if transcription_result and transcription_result.get('transcription'):
                        transcription = transcription_result['transcription']
                        metadata = transcription_result.get('metadata', {})
                        title = metadata.get('title', 'Unknown')
                        
                        return f"Video/audio content from {url} ('{title}'): {transcription}"
                    else:
                        return f"Successfully detected media in {url} but transcription failed"
                        
                except Exception as e:
                    self.logger.error(f"yt-dlp processing failed for {url}: {e}")
                    return f"Successfully detected media in {url} but could not process it: {str(e)}"
            
            # Prefer text from process_url when available.
            content = url_result.get('text', '')
            if content and content.strip():
                return f"Web content from {url}: {content}"

            # If no text was extracted (and no media route), use tiered extractor (no screenshots)
            self.logger.info(f"🧭 Falling back to tiered extractor for {url} (no auto-screenshot)")
            extract_res = await web_extractor.extract(url)
            if extract_res.success:
                return f"Web content from {extract_res.canonical_url or url}:\n{extract_res.to_message()}"
            else:
                return f"Could not extract content from URL: {url}"
            
        except Exception as e:
            self.logger.error(f"Error processing general URL: {e}", exc_info=True)
            return f"Failed to process URL: {item.payload}"
    
    async def _handle_screenshot_url(self, item: InputItem, progress_cb: Optional[Callable[[str, int], Awaitable[None]]] = None) -> str:
        """
        Handle URLs that need screenshot fallback.
        Returns screenshot analysis for further processing.
        Screenshots are explicitly command-gated (e.g., !ss).
        """
        try:
            if item.source_type != "url":
                return f"Screenshot handler received non-URL item: {item.source_type}"
            
            url = item.payload
            self.logger.info(f"📸 Taking screenshot of URL: {url}")
            if progress_cb:
                await progress_cb("validate", 1)
            # Lazy-import to avoid circular deps and keep import costs off hot paths
            from .utils.external_api import external_screenshot
            # Preparation phase (network/client setup, throttling checks, etc.)
            if progress_cb:
                await progress_cb("prepare", 2)
            
            if progress_cb:
                await progress_cb("capture", 3)
            screenshot_path = await external_screenshot(url)
            if not screenshot_path:
                self.logger.warning(f"⚠️ Screenshot API did not return an image for {url}")
                return f"⚠️ Could not capture a screenshot for: {url}. Please try again later."

            if progress_cb:
                await progress_cb("saved", 4)
            self.logger.info(f"🖼️ Screenshot saved at: {screenshot_path}. Sending to VL.")
            try:
                # Use VL to analyze the screenshot content
                if progress_cb:
                    await progress_cb("analyze", 5)
                analysis = await see_infer(
                    image_path=screenshot_path,
                    prompt=(
                        f"Analyze this screenshot from {url}. Summarize the main content, visible text, "
                        f"and any important details. Be concise."
                    ),
                )
                if analysis:
                    if progress_cb:
                        await progress_cb("done", 6)
                    return f"Screenshot content from {url}: {analysis}"
                else:
                    if progress_cb:
                        await progress_cb("done", 6)
                    return f"✅ Captured screenshot from {url}, but vision analysis returned no content."
            except Exception as vl_err:
                self.logger.error(f"❌ Vision analysis failed for {screenshot_path}: {vl_err}", exc_info=True)
                if progress_cb:
                    await progress_cb("done", 6)
                return f"✅ Captured screenshot from {url}, but could not analyze it right now."
            
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

        # Check for video URLs using comprehensive patterns from video_ingest.py
        try:
            from .video_ingest import SUPPORTED_PATTERNS
            self.logger.debug(f"🎥 Testing {len(SUPPORTED_PATTERNS)} video patterns against: {message.content}")
            
            for pattern in SUPPORTED_PATTERNS:
                if re.search(pattern, message.content):
                    self.logger.info(f"✅ Video URL detected: {message.content} matched pattern: {pattern}")
                    return InputModality.VIDEO_URL
                    
            self.logger.debug(f"❌ No video patterns matched for: {message.content}")
        except ImportError as e:
            self.logger.warning(f"Could not import SUPPORTED_PATTERNS from video_ingest: {e}, using fallback patterns")
            # Fallback patterns (original limited set)
            fallback_patterns = [
                r'https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+',
                r'https?://youtu\.be/[\w-]+',
                r'https?://(?:www\.)?tiktok\.com/@[\w.-]+/video/\d+',
                r'https?://(?:www\.)?tiktok\.com/t/[\w-]+',
                r'https?://(?:m|vm)\.tiktok\.com/[\w-]+',
            ]
            
            for pattern in fallback_patterns:
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
        
        # 1. RAG Integration - Search vector database concurrently for speed
        rag_task = None
        if os.getenv("ENABLE_RAG", "true").lower() == "true":
            try:
                from bot.rag.hybrid_search import get_hybrid_search
                max_results = int(os.getenv("RAG_MAX_VECTOR_RESULTS", "5"))
                self.logger.debug(f"🔍 RAG: Starting concurrent search for: '{content[:50]}...' [msg_id={message.id if message else 'N/A'}]")
                
                # Start RAG search concurrently - don't await here
                async def rag_search_task():
                    search_engine = await get_hybrid_search()
                    if search_engine:
                        return await search_engine.search(query=content, max_results=max_results)
                    return None
                
                rag_task = asyncio.create_task(rag_search_task())
            except Exception as e:
                self.logger.error(f"❌ RAG: Failed to start concurrent search: {e} [msg_id={message.id if message else 'N/A'}]", exc_info=True)
                rag_task = None
        
        # 2. Wait for RAG search to complete and process results
        if rag_task:
            try:
                # Add timeout to prevent hanging [REH]
                rag_results = await asyncio.wait_for(rag_task, timeout=5.0)
                if rag_results:
                    self.logger.debug(f"📊 RAG: Search completed, found {len(rag_results)} results")
                    
                    # Extract relevant content from search results (List[HybridSearchResult])
                    rag_context_parts = []
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
                    
                    if rag_context_parts:
                        rag_context = "\n\n".join(rag_context_parts)
                        enhanced_context = f"{context}\n\n=== Relevant Knowledge ===\n{rag_context}\n=== End Knowledge ===\n" if context else f"=== Relevant Knowledge ===\n{rag_context}\n=== End Knowledge ===\n"
                        self.logger.debug(f"✅ RAG: Enhanced context with {len(rag_context_parts)} knowledge chunks")
                    else:
                        self.logger.debug(f"⚠️ RAG: Search returned results but all chunks were empty")
                else:
                    self.logger.debug(f"🚫 RAG: No relevant results found")
            except Exception as e:
                self.logger.error(f"❌ RAG: Concurrent search failed: {e}")

        # 3. Use contextual brain inference if enhanced context manager is available and message is provided
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
        
        # 4. Fallback to basic brain inference with enhanced context (including RAG)
        return await brain_infer(content, context=enhanced_context)

    # ===== Inline [search(...)] directive handling =====
    def _extract_inline_search_queries(self, text: str) -> list[tuple[tuple[int, int], str]]:
        """
        Extract inline search directives of the form [search(<query>)] from text.
        Returns list of ((start, end), query) for replacement.
        """
        if not text:
            return []
        pattern = re.compile(r"\[search\s*\((.*?)\)\]", re.IGNORECASE | re.DOTALL)
        matches = []
        for m in pattern.finditer(text):
            query = (m.group(1) or '').strip()
            if query:
                matches.append(((m.start(), m.end()), query))
        return matches

    async def _resolve_inline_searches(self, text: str, message: Message) -> str:
        """
        Find and execute inline search directives in text, replacing each directive
        with a compact, formatted markdown block of results.
        """
        directives = self._extract_inline_search_queries(text)
        if not directives:
            return text

        self.logger.info(f"🔎 Found {len(directives)} inline search directive(s) (msg_id: {message.id})")

        # Config [IV]: pull from self.config with safe defaults
        provider_name = str(self.config.get("SEARCH_PROVIDER", "ddg"))
        max_results = int(self.config.get("SEARCH_MAX_RESULTS", 5))
        locale = self.config.get("SEARCH_LOCALE") or None
        safe_str = str(self.config.get("SEARCH_SAFE", "moderate")).lower()
        try:
            safesearch = SafeSearch(safe_str)
        except Exception:
            safesearch = SafeSearch.MODERATE
        timeout_ms = int(self.config.get("DDG_TIMEOUT_MS", 5000)) if provider_name == "ddg" else int(self.config.get("CUSTOM_SEARCH_TIMEOUT_MS", 8000))
        max_concurrency = int(os.getenv("SEARCH_INLINE_MAX_CONCURRENCY", "3"))

        provider = get_search_provider()

        # Execute searches with bounded concurrency [PA]
        sem = asyncio.Semaphore(max(1, max_concurrency))
        async def run_search(q: str):
            async with sem:
                params = SearchQueryParams(
                    query=q,
                    max_results=max_results,
                    safesearch=safesearch,
                    locale=locale,
                    timeout_ms=timeout_ms,
                )
                try:
                    self.logger.debug(f"[InlineSearch] Executing: '{q[:80]}'")
                    return await provider.search(params)
                except Exception as e:
                    self.logger.error(f"[InlineSearch] provider error for '{q}': {e}", exc_info=True)
                    return e

        tasks = [run_search(q) for _, q in directives]
        results_list = await asyncio.gather(*tasks, return_exceptions=False)

        # Build replacements
        pieces: list[str] = []
        cursor = 0
        for ((start, end), query), results in zip(directives, results_list):
            # Append text before directive
            if cursor < start:
                pieces.append(text[cursor:start])

            # Format replacement
            if isinstance(results, Exception):
                replacement = f"❌ Search failed for '{query}': please try again later."
            else:
                replacement = self._format_inline_search_block(query, results, provider_name, safesearch)

            pieces.append(replacement)
            cursor = end

        # Append trailing text
        pieces.append(text[cursor:])
        new_text = "".join(pieces)
        self.logger.debug(f"[InlineSearch] Rewrote text with {len(directives)} replacement(s). New length={len(new_text)}")
        return new_text

    def _format_inline_search_block(self, query: str, results: List[SearchResult], provider_name: str, safesearch: SafeSearch) -> str:
        """Format search results into a compact markdown block to inline into the prompt."""
        # Truncation limits aligned with Discord embed norms but adapted for text [PA]
        TITLE_LIMIT = 120
        SNIPPET_LIMIT = 240
        MAX_ITEMS = min(5, len(results))

        def trunc(s: str, limit: int) -> str:
            s = s or ""
            return s if len(s) <= limit else s[: max(0, limit - 1)] + "…"

        header = f"🔎 Search: `{trunc(query, 256)}`\n"
        lines: list[str] = [header]

        if not results:
            lines.append("No results found.")
        else:
            for idx, r in enumerate(results[:MAX_ITEMS], start=1):
                title = trunc(r.title or r.url, TITLE_LIMIT)
                snippet = trunc(r.snippet or "", SNIPPET_LIMIT)
                # Minimal, readable line per result
                lines.append(f"{idx}. {title}\n{r.url}")
                if snippet:
                    lines.append(f"    {snippet}")
                lines.append("")

        lines.append(f"Provider: {provider_name} • Safe: {safesearch.value}")
        return "\n".join(lines).strip()

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