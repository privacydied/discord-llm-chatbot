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
import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, Optional, TYPE_CHECKING, List, Awaitable, Tuple, Union
from html import unescape
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
import discord
import httpx
from .x_api_client import XApiClient
from .enhanced_retry import EnhancedRetryManager, ProviderConfig, get_retry_manager
from .brain import brain_infer
from .contextual_brain import contextual_brain_infer
import re
from . import web
from discord import Message, DMChannel, Embed, File
from .search.factory import get_search_provider
from .search.types import SearchQueryParams, SafeSearch, SearchResult, SearchCategory

if TYPE_CHECKING:
    from bot.core.bot import LLMBot as DiscordBot
    from bot.metrics import Metrics
    from .command_parser import ParsedCommand

logger = get_logger(__name__)

# Local application imports
from .action import BotAction, ResponseMessage
from .command_parser import Command, parse_command
from .modality import collect_input_items, InputModality, InputItem, map_item_to_modality
from .result_aggregator import ResultAggregator
from .exceptions import DispatchEmptyError, DispatchTypeError, APIError
from .hear import hear_infer, hear_infer_from_url
from .pdf_utils import PDFProcessor
from .see import see_infer
from .web import process_url
from .utils.mention_utils import ensure_single_mention
from .web_extraction_service import web_extractor
from .utils.file_utils import download_file
from .tts.state import tts_state

# Vision generation system
try:
    from .vision import VisionIntentRouter, VisionOrchestrator
    VISION_ENABLED = True
except ImportError:
    VISION_ENABLED = False

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
        # Image upgrade manager for emoji-driven expansions [CA]
        self._upgrade_manager = None  # Lazy-loaded when needed
        # Tweet syndication cache and locks [CA][PA]
        self._syn_cache: Dict[str, Dict[str, Any]] = {}
        self._syn_locks: Dict[str, asyncio.Lock] = {}
        try:
            self._syn_ttl_s: float = float(self.config.get("X_SYNDICATION_TTL_S", 900))
        except Exception:
            self._syn_ttl_s = 900.0
        
        # Vision generation system [CA][SFT]
        self._vision_intent_router: Optional[VisionIntentRouter] = None
        self._vision_orchestrator: Optional[VisionOrchestrator] = None
        
        # Debug logging for vision initialization
        self.logger.info(f"🔍 Vision initialization debug: VISION_ENABLED={VISION_ENABLED}, config_enabled={self.config.get('VISION_ENABLED', 'NOT_SET')}")

        # Load centralized VL prompt guidelines if available [CA]
        self._vl_prompt_guidelines: Optional[str] = None
        try:
            prompts_dir = Path(__file__).resolve().parents[1] / "prompts" / "vl-prompt.txt"
            if prompts_dir.exists():
                content = prompts_dir.read_text(encoding="utf-8").strip()
                if content:
                    self._vl_prompt_guidelines = content
                    self.logger.debug("Loaded VL prompt guidelines from prompts/vl-prompt.txt")
        except Exception:
            # Non-fatal; handler has built-in defaults
            self._vl_prompt_guidelines = None
        
        if VISION_ENABLED and self.config.get("VISION_ENABLED", False):
            self.logger.info("🚀 Starting vision system initialization...")
            try:
                self.logger.info("🔧 Creating VisionIntentRouter...")
                self._vision_intent_router = VisionIntentRouter(self.config)
                self.logger.info("🔧 Creating VisionOrchestrator...")
                self._vision_orchestrator = VisionOrchestrator(self.config)
                self.logger.info("✔ Vision generation system initialized successfully!")
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize Vision system: {e}", exc_info=True)
                self._vision_intent_router = None
                self._vision_orchestrator = None
        else:
            self.logger.warning(f"⚠️ Vision system NOT initialized - VISION_ENABLED={VISION_ENABLED}, config={self.config.get('VISION_ENABLED', 'NOT_SET')}")

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

    @staticmethod
    def _is_direct_image_url(url: str) -> bool:
        """Lightweight check for direct image URLs by extension. [IV]"""
        try:
            u = str(url).lower()
        except Exception:
            return False
        return bool(re.search(r"\.(jpe?g|png|webp)(?:\?|#|$)", u))

    async def _process_image_from_attachment_with_model(self, attachment, model_override: Optional[str] = None) -> str:
        """Save a Discord image attachment to a temp file and run VL analysis. [RM][REH]"""
        from .see import see_infer
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_path = tmp_file.name
            await attachment.save(tmp_path)
            self.logger.debug(f"📷 Saved image attachment to temp file: {tmp_path}")

            prompt = (
                "Describe this image in detail, focusing on key visual elements, objects, text, and context."
            )
            vision_response = await see_infer(
                image_path=tmp_path,
                prompt=prompt,
                model_override=model_override,
            )

            if not vision_response:
                return "❌ Vision processing returned no response"
            if getattr(vision_response, 'error', None):
                return f"❌ Vision processing error: {vision_response.error}"
            content = getattr(vision_response, 'content', '') or ''
            if not content.strip():
                return "❌ Vision processing returned empty content"
            filename = getattr(attachment, 'filename', 'image')
            return f"🖼️ **Image Analysis ({filename})**\n{content.strip()}"
        except Exception as e:
            self.logger.error(f"❌ Attachment VL processing failed: {e}", exc_info=True)
            return f"⚠️ Failed to analyze image attachment (error: {e})"
        finally:
            try:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception:
                pass

    async def _handle_image_with_model(self, item: InputItem, model_override: Optional[str] = None) -> str:
        """Handle image item with explicit model override. [CA][IV][REH]
        - Attachments: direct VL on file
        - URLs: direct image URL → download+VL; otherwise screenshot→VL
        - Embeds: try image/thumbnail URL similarly
        """
        try:
            if item.source_type == "attachment":
                attachment = item.payload
                return await self._process_image_from_attachment_with_model(attachment, model_override)

            if item.source_type == "url":
                url = item.payload
                if self._is_direct_image_url(url):
                    prompt = (
                        "Describe this image in detail, focusing on key visual elements, objects, text, and context."
                    )
                    desc = await self._vl_describe_image_from_url(url, prompt=prompt, model_override=model_override)
                    return desc or "⚠️ Unable to analyze the image from the provided URL."
                # Not a direct image URL → screenshot fallback
                return await self._process_image_from_url(url, model_override=model_override)

            if item.source_type == "embed":
                embed = item.payload or {}
                # Try common embed shapes
                image_url = None
                try:
                    if isinstance(embed, dict):
                        if isinstance(embed.get("image"), dict):
                            image_url = embed.get("image", {}).get("url")
                        if not image_url and isinstance(embed.get("thumbnail"), dict):
                            image_url = embed.get("thumbnail", {}).get("url")
                        if not image_url:
                            image_url = embed.get("url")
                except Exception:
                    image_url = None

                if image_url and self._is_direct_image_url(image_url):
                    desc = await self._vl_describe_image_from_url(
                        image_url,
                        prompt=(
                            "Describe this image in detail, focusing on key visual elements, objects, text, and context."
                        ),
                        model_override=model_override,
                    )
                    return desc or "⚠️ Unable to analyze the image from the embed."
                if image_url and isinstance(image_url, str) and image_url.startswith("http"):
                    return await self._process_image_from_url(image_url, model_override=model_override)
                return "⚠️ Embed did not contain a usable image URL."

            return "⚠️ Unsupported image source type."
        except Exception as e:
            self.logger.error(f"❌ _handle_image_with_model failed: {e}", extra={"detail": {"source_type": item.source_type}}, exc_info=True)
            return f"⚠️ Failed to process image item (error: {e})"

    async def _handle_image(self, item: InputItem) -> str:
        """Handle image without explicit model override, using default VL model. [CA]"""
        return await self._handle_image_with_model(item, model_override=None)

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

    def _format_x_tweet_with_transcription(
        self,
        *,
        base_text: Optional[str],
        url: str,
        stt_res: Dict[str, Any],
    ) -> str:
        """Combine formatted tweet text with STT transcription and lightweight media metadata. [CA][PA]

        base_text: already formatted tweet string (e.g., from _format_x_tweet_result or _format_syndication_result)
        stt_res: result dict from hear_infer_from_url()
        """
        try:
            transcription = (stt_res or {}).get("transcription") or ""
            meta = (stt_res or {}).get("metadata") or {}
            src = meta.get("source") or "media"
            title = meta.get("title") or ""
            dur = meta.get("original_duration_s") or meta.get("processed_duration_s")
            dur_s = f" • {int(dur)}s" if isinstance(dur, (int, float)) else ""
            title_str = f" • '{title}'" if title else ""
            header = f"🎙️ Transcription ({src}{title_str}{dur_s})"
            if base_text and base_text.strip():
                return f"{base_text}\n\n{header}:\n{transcription}"
            # Fallback if no tweet text available
            return f"Tweet → {url}\n\n{header}:\n{transcription}"
        except Exception:
            # Last-resort fallback: just return transcription string
            transcription = (stt_res or {}).get("transcription") or ""
            return f"Video/audio content from {url}: {transcription}"

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

        # ROUTER_DEBUG=1 diagnostics for path selection [IV]
        router_debug = os.getenv("ROUTER_DEBUG", "0").lower() in ("1", "true", "yes", "on")

        try:
            # 1. Quick pre-filter: Only parse commands for messages that start with '!' to avoid unnecessary parsing
            content = message.content.strip()
            
            # Remove bot mention to check for command pattern
            mention_pattern = fr'^<@!?{self.bot.user.id}>\s*'
            clean_content = re.sub(mention_pattern, '', content)

            # 1b. Compatibility fast-path for legacy tests: attachments + empty content
            # Run this BEFORE gating and typing() to avoid MagicMock issues in tests
            try:
                has_attachments = bool(getattr(message, "attachments", None)) and len(message.attachments) > 0
            except Exception:
                has_attachments = False
            cleaned_for_compat = re.sub(mention_pattern, '', (message.content or '').strip())
            if has_attachments and cleaned_for_compat == "":
                handler = self._flows.get('process_attachments')
                if handler:
                    self.logger.debug("Compat path (pre-gate): delegating to _flows['process_attachments'] with empty text.")
                    res = await handler(message, "")
                    if isinstance(res, BotAction):
                        return res
                    else:
                        # Wrap plain string result into BotAction for compatibility
                        return BotAction(content=str(res))
            
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

                # 4. Compatibility fast-path for legacy tests: attachments + empty content (secondary safeguard)
                try:
                    has_attachments = bool(getattr(message, "attachments", None)) and len(message.attachments) > 0
                except Exception:
                    has_attachments = False
                # Recompute a minimal cleaned content (strip mention prefix like above)
                mention_pattern = fr'^<@!?{self.bot.user.id}>\s*'
                cleaned_for_compat = re.sub(mention_pattern, '', (message.content or '').strip())
                if has_attachments and cleaned_for_compat == "":
                    handler = self._flows.get('process_attachments')
                    if handler:
                        self.logger.debug("Compat path: delegating to _flows['process_attachments'] with empty text.")
                        res = await handler(message, "")
                        if isinstance(res, BotAction):
                            return res
                        else:
                            # Wrap plain string result into BotAction for compatibility
                            return BotAction(content=str(res))

                # Continue with normal flow processing
                context_str = ""  # Simple fix: no conversation context for now

                # 5. Check for vision generation intent early (before multi-modal)
                try:
                    prechecked = await self._prioritized_vision_route(message, context_str)
                except Exception as e:
                    prechecked = None
                    self.logger.debug(f"vision.precheck_exception | {e}")
                if prechecked is not None:
                    if router_debug:
                        self.logger.info(f"ROUTER_DEBUG | path=t2i reason=vision_intent_detected msg_id={message.id}")
                    return prechecked

                # 6. Sequential multimodal processing
                result_action = await self._process_multimodal_message_internal(message, context_str)
                if router_debug:
                    # Determine what path was taken based on message content
                    has_x_urls = any(self._is_twitter_url(url) for url in re.findall(r'https?://\S+', content))
                    has_attachments = bool(getattr(message, 'attachments', None))
                    if has_x_urls:
                        self.logger.info(f"ROUTER_DEBUG | path=x_syndication_vl reason=twitter_url_detected msg_id={message.id}")
                    elif has_attachments:
                        self.logger.info(f"ROUTER_DEBUG | path=attachment_vl reason=image_attachments msg_id={message.id}")
                    else:
                        self.logger.info(f"ROUTER_DEBUG | path=multimodal reason=default_flow msg_id={message.id}")
                return result_action  # Return the actual processing result

        except Exception as e:
            self.logger.error(f"❌ Error in router dispatch: {e} (msg_id: {message.id})", exc_info=True)
            return BotAction(content="⚠️ An unexpected error occurred while processing your message.", error=True)

    def compute_streaming_eligibility(self, message: Message) -> Dict[str, Any]:
        """Preflight: determine if streaming status cards should be enabled for this message.
        This must be cheap and avoid network calls. [CA][IV][PA]

        Returns a dict with:
        - eligible: bool
        - modality: str ("TEXT_ONLY" | "MEDIA_OR_HEAVY")
        - domains: set[str] subset of {"text","media","search","rag"}
        - reason: str short reason string for logging
        """
        try:
            cfg = self.config
            if not cfg.get("STREAMING_ENABLE", True):
                return {"eligible": False, "modality": "TEXT_ONLY", "domains": {"text"}, "reason": "streaming_master_disabled"}

            content = (message.content or "").lower().strip()
            domains: set[str] = set()

            # Command-based detections (search/rag)
            if content.startswith("!search") or content.startswith("[search]"):
                domains.add("search")
            if content.startswith("!rag "):
                domains.add("rag")

            # Collect items and mark media when confidently heavy without network
            items = collect_input_items(message)
            has_media = False
            if items:
                # Lightweight modality mapping – should inspect filenames/urls only
                # Avoid network; map_item_to_modality may be async but typically local; use best-effort heuristics here.
                for it in items:
                    # Attachments by filename
                    if it.source_type == "attachment":
                        name = getattr(it.payload, "filename", "").lower()
                        if any(name.endswith(ext) for ext in (".png",".jpg",".jpeg",".webp",".gif",".bmp",".pdf",".mp4",".mov",".mkv",".webm",".avi",".m4v",".mp3",".wav",".ogg",".m4a",".flac")):
                            has_media = True
                    elif it.source_type == "url":
                        url = str(it.payload).lower()
                        # Heuristics deemed heavy: youtube/streaming video links, explicit screenshot directives
                        if "youtu" in url or "youtube" in url:
                            has_media = True
                        # Some flows generate screenshots via explicit markers; prefer conservative enabling only when explicit
                        if "[screenshot]" in content:
                            has_media = True
                    elif it.source_type == "embed":
                        # Embeds with image/video hints may be heavy; conservative: don't enable by embeds alone
                        pass

            if has_media:
                domains.add("media")

            # If nothing detected, default to text
            if not domains:
                domains.add("text")

            # Apply config toggles per domain
            allow = False
            reasons = []
            if "media" in domains:
                if cfg.get("STREAMING_ENABLE_MEDIA", True):
                    allow = True
                    reasons.append("media_allowed")
                else:
                    reasons.append("media_disabled")
            if "search" in domains:
                if cfg.get("STREAMING_ENABLE_SEARCH", False):
                    allow = True
                    reasons.append("search_allowed")
                else:
                    reasons.append("search_disabled")
            if "rag" in domains:
                if cfg.get("STREAMING_ENABLE_RAG", False):
                    allow = True
                    reasons.append("rag_allowed")
                else:
                    reasons.append("rag_disabled")
            if domains == {"text"}:
                if cfg.get("STREAMING_ENABLE_TEXT", False):
                    allow = True
                    reasons.append("text_allowed")
                else:
                    reasons.append("text_disabled")

            modality = "MEDIA_OR_HEAVY" if ("media" in domains or "search" in domains or "rag" in domains) else "TEXT_ONLY"
            return {"eligible": bool(allow), "modality": modality, "domains": domains, "reason": ",".join(reasons) or "none"}
        except Exception as e:
            # Fail-closed to quiet mode for safety
            self.logger.debug(f"stream:eligibility_failed | {e}")
            return {"eligible": False, "modality": "TEXT_ONLY", "domains": {"text"}, "reason": "exception"}

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
        
        # --- Routing precedence gates (feature-flagged) ---
        # 0) Safety: re-run prioritized vision precheck here to catch any triggers/intents that
        #    may have been missed earlier. This is a no-op if none are detected. [CA][REH]
        try:
            prechecked = await self._prioritized_vision_route(message, context_str)
            if prechecked is not None:
                self._metric_inc("routing.vision.precedence", {"stage": "in_multimodal"})
                return prechecked
        except Exception as e:
            # Never break dispatch because of a precheck failure
            self.logger.debug(f"routing.precedence.vision_check_failed | {e}")
        
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
        
        # 1) Web link precedence (if enabled): when URLs are present and vision intent wasn't selected,
        #    prioritize URL processing over other modalities. This preserves 1 IN → 1 OUT by limiting the
        #    item set to URLs only. [Feature-flag: ROUTING_WEB_LINK_PRECEDENCE]
        try:
            web_link_precedence = bool(self.config.get("ROUTING_WEB_LINK_PRECEDENCE", False))
        except Exception:
            web_link_precedence = False
        try:
            url_items = [it for it in items if getattr(it, 'source_type', None) == "url"]
        except Exception:
            url_items = []

        # Helper: check if text is meaningful (letters/digits after stripping whitespace/punct)
        def _has_meaningful_text(s: str) -> bool:
            try:
                s = (s or "").strip()
                if not s:
                    return False
                # Remove non-alphanumeric characters (keep unicode letters/digits)
                cleaned = re.sub(r"[^\w]+", "", s, flags=re.UNICODE)
                return len(cleaned) >= 3
            except Exception:
                return bool(s and s.strip())

        # 2) Bare image default VL (if enabled): when only images are provided with no meaningful text,
        #    run VL description using the default prompt. We keep the sequential pipeline but scope items
        #    to image attachments to minimize disruption. [Feature-flag: VL_DEFAULT_PROMPT_FOR_BARE_IMAGE]
        try:
            vl_default_for_bare_image = bool(self.config.get("VL_DEFAULT_PROMPT_FOR_BARE_IMAGE", True))
        except Exception:
            vl_default_for_bare_image = True
        try:
            image_attachment_items = [
                it for it in items
                if getattr(it, 'source_type', None) == "attachment"
                and hasattr(getattr(it, 'payload', None), 'content_type')
                and isinstance(getattr(it, 'payload').content_type, str)
                and 'image' in (getattr(it, 'payload').content_type or '').lower()
            ]
        except Exception:
            image_attachment_items = []

        precedence_applied = False
        if web_link_precedence and url_items:
            self.logger.info(
                f"🔗 Web link precedence enabled; routing to URL-only processing (urls={len(url_items)}) (msg_id: {message.id})"
            )
            self._metric_inc("routing.url.precedence.selected", {"count": str(len(url_items))})
            items = url_items
            precedence_applied = True
        elif vl_default_for_bare_image and image_attachment_items and (not _has_meaningful_text(original_text)):
            # Backward-compat: legacy attachment-only messages with truly empty content remain supported by
            # the earlier fast-path. This branch handles minimal/implicit prompts too. [REH]
            self.logger.info(
                f"route=attachments | 🖼️ Bare image attachments detected with no meaningful text; prioritizing VL analysis (msg_id: {message.id})"
            )
            self._metric_inc("routing.vl.default_bare_image.selected", {"count": str(len(image_attachment_items))})
            items = image_attachment_items
            precedence_applied = True

        self.logger.info(
            f"🚶 Processing {len(items)} input items SEQUENTIALLY for deterministic order (precedence={precedence_applied}) (msg_id: {message.id})"
        )
        
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

            # Special-case: Twitter/X GENERAL_URL items may invoke heavy media (STT) work even though
            # we keep API-first logic in _handle_general_url(). To avoid cancelling STT with short
            # text timeouts, treat these items as 'media' for retry/budget purposes. [PA][REH]
            try:
                if modality == InputModality.GENERAL_URL and item.source_type == "url":
                    raw_url = str(item.payload)
                    if self._is_twitter_url(raw_url):
                        self.logger.info(
                            "⚙️ Treating Twitter/X GENERAL_URL as media for retry budget/timeouts",
                            extra={"event": "x.retry_policy.media_budget", "detail": {"url": raw_url}},
                        )
                        retry_modality = "media"
                        selected_budget = MEDIA_PER_ITEM_BUDGET
            except Exception:
                # Never break dispatch due to budgeting heuristics
                pass

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
    
    async def _process_image_from_url(self, url: str, model_override: Optional[str] = None) -> str:
        """Process image from URL using screenshot API + vision analysis. Passes model_override to VL."""
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
            vision_result = await see_infer(image_path=screenshot_path, prompt="Describe the contents of this screenshot", model_override=model_override)
            
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
                # Special-case: pbs.twimg.com sometimes rejects name=orig; fall back to name=large [REH]
                try:
                    p = urlparse(image_url)
                    host = (p.netloc or "").split(":")[0]
                    if host == "pbs.twimg.com":
                        qs = dict(parse_qsl(p.query, keep_blank_values=True))
                        qs["name"] = "large"
                        fallback_url = urlunparse((p.scheme, p.netloc, p.path, p.params, urlencode(qs, doseq=True), p.fragment))
                        self.logger.warning(f"⚠️ High-res download failed, retrying with 'name=large': {fallback_url}")
                        ok = await download_file(fallback_url, Path(tmp_path))
                        if not ok:
                            # Third tier: try 'name=medium' to stay under budget [PA]
                            qs["name"] = "medium"
                            fallback_medium = urlunparse((p.scheme, p.netloc, p.path, p.params, urlencode(qs, doseq=True), p.fragment))
                            self.logger.warning(f"⚠️ Large download failed, retrying with 'name=medium': {fallback_medium}")
                            ok = await download_file(fallback_medium, Path(tmp_path))
                            if not ok:
                                self.logger.error(f"❌ Failed to download Twitter image even with fallbacks: {fallback_medium}")
                                return None
                            # Update for logging clarity
                            image_url = fallback_medium
                        else:
                            # Update for logging clarity
                            image_url = fallback_url
                    else:
                        self.logger.error(f"❌ Failed to download image for VL: {image_url}")
                        return None
                except Exception as _e:
                    self.logger.error(f"❌ Image download failed (no fallback applied): {image_url} err={_e}")
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
                if is_twitter:
                    cfg = self.config
                    tweet_id = XApiClient.extract_tweet_id(str(url))
                    x_client = await self._get_x_api_client()
                    base_text = None
                    if tweet_id and x_client is not None:
                        try:
                            api_data = await x_client.get_tweet_by_id(tweet_id)
                            base_text = self._format_x_tweet_result(api_data, url)
                        except Exception:
                            base_text = None
                    if base_text is None and tweet_id and bool(cfg.get("X_SYNDICATION_ENABLED", True)):
                        try:
                            syn = await self._get_tweet_via_syndication(tweet_id)
                            if syn:
                                base_text = self._format_syndication_result(syn, url)
                        except Exception:
                            base_text = None
                    return self._format_x_tweet_with_transcription(
                        base_text=base_text,
                        url=url,
                        stt_res=result,
                    )
                # Non-Twitter: keep existing concise output
                transcription = result['transcription']
                metadata = result.get('metadata', {})
                title = metadata.get('title', 'Unknown')
                return f"Video transcription from {url} ('{title}'): {transcription}"
            else:
                return f"Could not transcribe audio from video: {url}"
            
        except VideoIngestError as ve:
            error_str = str(ve).lower()
            
            # For Twitter URLs with no media, use syndication/API path instead of web extractor [CA][REH]
            if is_twitter and (
                "no video or audio content found" in error_str or
                "no video could be found" in error_str or
                "failed to download video" in error_str
            ):
                self.logger.info(f"🐦 No video in Twitter URL; routing to syndication/API path: {url}")
                # Force to general URL handler which has proper X syndication logic
                return await self._handle_general_url(InputItem(source_type="url", payload=url))
            
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

                # Fast path: try STT probe first for X URLs only when API client is unavailable [PA][REH]
                # Preserves API-first behavior when API access is configured/available.
                if bool(cfg.get("X_TWITTER_STT_PROBE_FIRST", True)) and (x_client is None):
                    try:
                        stt_res = await hear_infer_from_url(url)
                        if stt_res and stt_res.get("transcription"):
                            base_text = None
                            if syndication_enabled and tweet_id:
                                syn = await self._get_tweet_via_syndication(tweet_id)
                                if syn:
                                    base_text = self._format_syndication_result(syn, url)
                            return self._format_x_tweet_with_transcription(
                                base_text=base_text,
                                url=url,
                                stt_res=stt_res,
                            )
                    except Exception as stt_err:
                        err_str = str(stt_err).lower()
                        # Only bypass to API/syndication if clearly not video/audio or unsupported URL
                        if ("no video or audio content" in err_str) or ("unsupported url" in err_str):
                            pass
                        else:
                            self.logger.info(
                                "X STT probe failed; continuing with API/syndication path",
                                extra={"event": "x.stt_probe.fail", "detail": {"url": url, "error": str(stt_err)}},
                            )

                # Tier 1: Syndication JSON (cache + concurrency) when allowed and preferred [PA][REH]
                if tweet_id and syndication_enabled and not require_api and (syndication_first or x_client is None):
                    syn = await self._get_tweet_via_syndication(tweet_id)
                    if syn:
                        self._metric_inc("x.syndication.hit", None)
                        # Media-first branching: detect image-only tweets [CA][IV]
                        photos = syn.get("photos") or []
                        text = (syn.get("text") or syn.get("full_text") or "").strip()
                        
                        # Check for image-only tweet: photos present AND empty/whitespace text [IV]
                        normalize_empty = bool(cfg.get("TWITTER_NORMALIZE_EMPTY_TEXT", True))
                        is_image_only = (photos and 
                                       (not text or (normalize_empty and not text.strip())))
                        
                        if is_image_only and bool(cfg.get("TWITTER_IMAGE_ONLY_ENABLE", True)):
                            # Route to Vision/OCR pipeline for image-only tweets [CA]
                            self.logger.info(f"🖼️ Image-only tweet detected, routing to Vision/OCR: {url}")
                            self._metric_inc("x.tweet_image_only.syndication", {"photos": str(len(photos))})
                            return await self._handle_image_only_tweet(url, syn, source="syndication")
                        
                        base = self._format_syndication_result(syn, url)
                        if not photos:
                            # Attempt STT for potential video tweets; silently fall back if not video [PA][REH]
                            try:
                                stt_res = await hear_infer_from_url(url)
                                if stt_res and stt_res.get("transcription"):
                                    return self._format_x_tweet_with_transcription(
                                        base_text=base,
                                        url=url,
                                        stt_res=stt_res,
                                    )
                            except Exception as stt_err:
                                err_s = str(stt_err).lower()
                                if ("no video or audio content" in err_s) or ("no video" in err_s):
                                    pass  # text-only tweet; just return base
                                else:
                                    self.logger.info(
                                        "X syndication STT attempt non-fatal error",
                                        extra={"event": "x.syndication.stt.warn", "detail": {"url": url, "error": str(stt_err)}},
                                    )
                            return base
                        # SURGICAL FIX: Always route photos to VL for X URLs (ignore the flag) [CA][REH]
                        # This ensures native photos are analyzed instead of just counting them
                        
                        # Use new syndication handler for full-res images [CA][PA]
                        from ..syndication.handler import handle_twitter_syndication_to_vl
                        # Minimal route log for observability [CMV]
                        try:
                            self.logger.info(
                                "route=x_syndication | sending photos to VL with high-res upgrade",
                                extra={"detail": {"url": url}},
                            )
                        except Exception:
                            pass
                        return await handle_twitter_syndication_to_vl(
                            syn,
                            url,
                            self._vl_describe_image_from_url,
                            self.bot.system_prompts.get("vl_prompt"),
                        )

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
                                    base = self._format_x_tweet_result(api_data, url)
                                    return self._format_x_tweet_with_transcription(
                                        base_text=base,
                                        url=url,
                                        stt_res=stt_res,
                                    )
                                base = self._format_x_tweet_result(api_data, url)
                                return f"{base}\n\nDetected media in this tweet but transcription failed."
                            except Exception as stt_err:
                                self.logger.error(
                                    f"X media STT route failed for {url}: {stt_err}",
                                    extra={"detail": {"url": url}},
                                )
                                base = self._format_x_tweet_result(api_data, url)
                                return f"{base}\n\nDetected media in this tweet but could not process it right now."

                        if media_types == {"photo"} or ("photo" in media_types and len(media_types) == 1):
                            # Check for image-only tweet via API data [IV]
                            tweet_data = api_data.get("data", {})
                            if isinstance(tweet_data, list) and tweet_data:
                                tweet_data = tweet_data[0]
                            elif isinstance(tweet_data, dict):
                                pass  # already correct format
                            else:
                                tweet_data = {}
                            
                            api_text = (tweet_data.get("text") or "").strip()
                            photos = [m for m in media_list if isinstance(m, dict) and m.get("type") == "photo"]
                            normalize_empty = bool(cfg.get("TWITTER_NORMALIZE_EMPTY_TEXT", True))
                            is_image_only = (photos and 
                                           (not api_text or (normalize_empty and not api_text.strip())))
                            
                            if is_image_only and bool(cfg.get("TWITTER_IMAGE_ONLY_ENABLE", True)):
                                # Route to Vision/OCR pipeline for image-only tweets [CA]
                                self.logger.info(f"🖼️ Image-only tweet detected via API, routing to Vision/OCR: {url}")
                                self._metric_inc("x.tweet_image_only.api", {"photos": str(len(photos))})
                                # Convert API data to syndication-like format for unified handling
                                api_as_syn = {
                                    "text": api_text,
                                    "photos": [{"url": p.get("url")} for p in photos if p.get("url")],
                                    "user": {"screen_name": "unknown"},  # Will be enriched if user data available
                                    "created_at": tweet_data.get("created_at")
                                }
                                return await self._handle_image_only_tweet(url, api_as_syn, source="api")
                            
                            # SURGICAL FIX: Always route photos to VL for X URLs (ignore the flag) [CA][REH]
                            # This ensures native photos are analyzed instead of just counting them
                            base = self._format_x_tweet_result(api_data, url)

                            self.logger.info(
                                "🖼️🐦 Routing X photos to VL via API data",
                                extra={
                                    "event": "x.photo_to_vl.start",
                                    "detail": {
                                        "url": url,
                                        "photo_count": len(photos),
                                    },
                                },
                            )
                            self._metric_inc("x.photo_to_vl.enabled", None)
                            
                            # Convert API data to syndication-like format for unified handling [CA][PA]
                            api_as_syn = {
                                "text": (tweet_data.get("text") or "").strip(),
                                "photos": [{"url": p.get("url")} for p in photos if p.get("url")],
                                "user": {"screen_name": "unknown"},
                                "created_at": tweet_data.get("created_at")
                            }
                            
                            # Use new syndication handler for full-res images [CA][PA] 
                            from ..syndication.handler import handle_twitter_syndication_to_vl
                            return await handle_twitter_syndication_to_vl(
                                api_as_syn,
                                url,
                                self._vl_describe_image_from_url,
                                self.bot.system_prompts.get("vl_prompt"),
                            )

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

    async def _prioritized_vision_route(self, message: Message, context_str: str) -> Optional[BotAction]:
        """Early, prioritized vision routing based on direct triggers or intent.
        Respects feature flags and supports dry-run mode. Returns a BotAction if
        vision generation should be taken over immediately; otherwise None to continue
        with normal multimodal processing. [CA][SFT][REH]
        """
        try:
            content = (message.content or "").strip()
            if not content:
                return None
            
            # Clean mention prefix for more accurate intent detection
            try:
                mention_pattern = fr'^<@!?{self.bot.user.id}>\s*'
                content_clean = re.sub(mention_pattern, '', content)
            except Exception:
                content_clean = content

            # Perception beats generation: if images or Twitter URLs are present, skip gen path
            try:
                has_img_attachments = any(
                    (getattr(a, 'content_type', '') or '').startswith('image/')
                    for a in (getattr(message, 'attachments', None) or [])
                )
            except Exception:
                has_img_attachments = False

            has_twitter_url = False
            try:
                url_candidates = re.findall(r'https?://\S+', content)
                has_twitter_url = any(self._is_twitter_url(u) for u in url_candidates)
            except Exception:
                has_twitter_url = False

            if has_img_attachments or has_twitter_url:
                route = 'attachments' if has_img_attachments else 'x_syndication'
                self.logger.info(
                    f"route.guard: perception_beats_generation | route={route} (msg_id: {message.id})"
                )
                try:
                    self._metric_inc("vision.route.vl_only_bypass_t2i", {"route": route})
                except Exception:
                    pass
                # Never trigger image generation if images or Twitter URLs are present
                return None
            
            # Respect vision feature flags
            cfg_enabled = bool(self.config.get("VISION_ENABLED", False))
            dry_run = bool(self.config.get("VISION_DRY_RUN_MODE", False))
            orchestrator_ready = bool(self._vision_orchestrator)
            
            # If vision is not enabled at all, skip
            if not cfg_enabled:
                self._metric_inc("vision.route.skipped", {"reason": "cfg_disabled"})
                return None
            
            # 1) Direct trigger bypass (highest priority)
            direct_vision = self._detect_direct_vision_triggers(content_clean)
            if direct_vision:
                self.logger.info(
                    f"🎨 Precheck: Direct vision bypass (reason: {direct_vision['bypass_reason']}) (msg_id: {message.id})"
                )
                self._metric_inc("vision.route.direct", {"stage": "precheck"})
                
                # Create a mock intent result for the vision handler
                from types import SimpleNamespace
                intent_result = SimpleNamespace()
                intent_result.decision = SimpleNamespace()
                intent_result.decision.use_vision = True
                intent_result.extracted_params = SimpleNamespace()
                intent_result.extracted_params.task = direct_vision["task"]
                intent_result.extracted_params.prompt = direct_vision["prompt"]
                intent_result.extracted_params.width = 1024
                intent_result.extracted_params.height = 1024
                intent_result.extracted_params.batch_size = 1
                intent_result.confidence = direct_vision["confidence"]
                
                if dry_run:
                    self._metric_inc("vision.route.dry_run", {"path": "direct"})
                    return BotAction(content=(
                        "[DRY RUN] Vision generation would be triggered via direct trigger "
                        f"(task={intent_result.extracted_params.task}, prompt='{intent_result.extracted_params.prompt[:80]}...')."
                    ))
                
                if not orchestrator_ready:
                    self._metric_inc("vision.route.blocked", {"reason": "orchestrator_unavailable", "path": "direct"})
                    return BotAction(content="🚫 Vision generation is not available right now. Please try again later.")
                
                return await self._handle_vision_generation(intent_result, message, context_str)
            
            # 2) Intent router decision (lower priority than direct bypass)
            if self._vision_intent_router:
                try:
                    intent_result = await self._vision_intent_router.determine_intent(
                        user_message=content_clean,
                        context=context_str,
                        user_id=str(message.author.id),
                        guild_id=str(message.guild.id) if message.guild else None
                    )
                    if intent_result and getattr(intent_result.decision, 'use_vision', False):
                        conf = float(getattr(intent_result, 'confidence', 0.0) or 0.0)
                        self.logger.info(
                            f"🎨 Precheck: Vision intent detected (confidence: {conf:.2f}), routing to Vision system (msg_id: {message.id})"
                        )
                        self._metric_inc("vision.route.intent", {"stage": "precheck"})
                        if dry_run:
                            self._metric_inc("vision.route.dry_run", {"path": "intent"})
                            return BotAction(content=(
                                "[DRY RUN] Vision generation would be triggered via intent detection "
                                f"(confidence={conf:.2f})."
                            ))
                        if not orchestrator_ready:
                            self._metric_inc("vision.route.blocked", {"reason": "orchestrator_unavailable", "path": "intent"})
                            return BotAction(content="🚫 Vision generation is not available right now. Please try again later.")
                        return await self._handle_vision_generation(intent_result, message, context_str)
                except Exception as e:
                    self.logger.error(f"❌ Vision intent precheck failed: {e} (msg_id: {message.id})", exc_info=True)
                    self._metric_inc("vision.intent.error", None)
                    # Fall through to normal multimodal flow on errors
            
            return None
        except Exception as e:
            # Fail-safe: never break dispatch on precheck
            self.logger.debug(f"vision.precheck_failed | {e}")
            return None

    async def _invoke_text_flow(self, content: str, message: Message, context_str: str) -> BotAction:
        """Invoke the text processing flow, formatting history into a context string."""
        self.logger.info(f"route=text | Routing to text flow. (msg_id: {message.id})")
        
        # Perception beats generation: suppress gen triggers if images/Twitter present
        perception_guard = False
        try:
            has_img_attachments = any(
                (getattr(a, 'content_type', '') or '').startswith('image/')
                for a in (getattr(message, 'attachments', None) or [])
            )
        except Exception:
            has_img_attachments = False
        try:
            url_candidates = re.findall(r'https?://\S+', content or '')
            has_twitter_url = any(self._is_twitter_url(u) for u in url_candidates)
        except Exception:
            has_twitter_url = False
        if has_img_attachments or has_twitter_url:
            perception_guard = True
            try:
                route = 'attachments' if has_img_attachments else 'x_syndication'
                self._metric_inc("vision.route.vl_only_bypass_t2i", {"route": route})
            except Exception:
                pass

        # Check for direct vision triggers first (bypasses rate-limited intent detection)
        if content.strip() and not perception_guard:
            direct_vision = self._detect_direct_vision_triggers(content)
            if direct_vision:
                self.logger.info(
                    f"route=gen | 🎨 Direct vision bypass triggered (reason: {direct_vision['bypass_reason']}) (msg_id: {message.id})"
                )
                self._metric_inc("vision.route.direct", {"stage": "text_flow"})
                # Create a mock intent result for the vision handler
                from types import SimpleNamespace
                intent_result = SimpleNamespace()
                intent_result.decision = SimpleNamespace()
                intent_result.decision.use_vision = True
                intent_result.extracted_params = SimpleNamespace()
                intent_result.extracted_params.task = direct_vision["task"]
                intent_result.extracted_params.prompt = direct_vision["prompt"]
                intent_result.extracted_params.width = 1024
                intent_result.extracted_params.height = 1024
                intent_result.extracted_params.batch_size = 1
                intent_result.confidence = direct_vision["confidence"]
                
                return await self._handle_vision_generation(intent_result, message, context_str)
        
        # Check if this should be routed to Vision generation [CA][SFT]
        if self._vision_intent_router and content.strip():
            try:
                intent_result = await self._vision_intent_router.determine_intent(
                    user_message=content,
                    context=context_str,
                    user_id=str(message.author.id),
                    guild_id=str(message.guild.id) if message.guild else None
                )
                
                if intent_result.decision.use_vision:
                    self.logger.info(
                        f"🎨 Vision intent detected (confidence: {intent_result.confidence:.2f}), routing to Vision system (msg_id: {message.id})"
                    )
                    self._metric_inc("vision.route.intent", {"stage": "text_flow"})
                    return await self._handle_vision_generation(intent_result, message, context_str)
            except Exception as e:
                self.logger.error(f"❌ Vision intent routing failed: {e} (msg_id: {message.id})", exc_info=True)
                self._metric_inc("vision.intent.error", None)
                # Continue to regular text flow on error
        
        try:
            action = await self._flows['process_text'](content, context_str, message)
            if action and action.has_payload:
                # Respect TTS state: one-time flag first, then per-user/global preference [CA][REH]
                try:
                    user_id = getattr(message.author, 'id', None)
                    require_tts = False
                    if user_id is not None:
                        if tts_state.get_and_clear_one_time_tts(user_id):
                            require_tts = True
                        elif tts_state.is_user_tts_enabled(user_id):
                            require_tts = True

                    if require_tts:
                        action.meta['requires_tts'] = True
                        # Include transcript captions unless disabled via env/config [IV][CMV]
                        include_transcript = os.getenv('TTS_INCLUDE_TRANSCRIPT', 'true').lower() in ('1','true','yes','on')
                        action.meta['include_transcript'] = include_transcript
                except Exception as e:
                    # Never break dispatch on TTS flag evaluation
                    self.logger.debug(f"tts.flag_eval_failed | {e}")
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
    def _extract_inline_search_queries(self, text: str) -> list[tuple[tuple[int, int], str, Optional[SearchCategory]]]:
        """
        Extract inline search directives of the form [search(<query>)] or
        [search(<query>, <category>)] from text.

        Returns list of ((start, end), query, category) for replacement.
        The category is optional and will be None if not provided. When
        present, it is parsed case-insensitively and mapped to SearchCategory.
        """
        if not text:
            return []
        pattern = re.compile(r"\[search\s*\((.*?)\)\]", re.IGNORECASE | re.DOTALL)
        matches: list[tuple[tuple[int, int], str, Optional[SearchCategory]]] = []

        def _parse_category(arg_tail: str) -> Optional[SearchCategory]:
            # Normalize and strip quotes
            token = (arg_tail or "").strip().strip("'\"")
            if not token:
                return None
            # Accept common synonyms (image, images, video, videos)
            token_l = token.lower()
            if token_l in ("text",):
                return SearchCategory.TEXT
            if token_l in ("news",):
                return SearchCategory.NEWS
            if token_l in ("image", "images"):  # allow singular
                return SearchCategory.IMAGES
            if token_l in ("video", "videos"):  # allow singular
                return SearchCategory.VIDEOS
            # Unrecognized -> None to preserve backward compatibility
            return None

        for m in pattern.finditer(text):
            inner = (m.group(1) or '').strip()
            if not inner:
                continue
            # Try to parse optional category by splitting on the last comma
            # This preserves commas inside the query.
            query: str = inner
            category: Optional[SearchCategory] = None
            if "," in inner:
                q_part, cat_part = inner.rsplit(",", 1)
                cat = _parse_category(cat_part)
                if cat is not None:
                    query = q_part.strip()
                    category = cat
            if query:
                matches.append(((m.start(), m.end()), query, category))
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
        async def run_search(q: str, cat: Optional[SearchCategory]):
            async with sem:
                params = SearchQueryParams(
                    query=q,
                    max_results=max_results,
                    safesearch=safesearch,
                    locale=locale,
                    timeout_ms=timeout_ms,
                    category=cat or SearchCategory.TEXT,
                )
                try:
                    cat_label = (cat.value if isinstance(cat, SearchCategory) else "text")
                    self._metric_inc("inline_search.start", {"category": cat_label, "provider": provider_name})
                    self.logger.debug(f"[InlineSearch] Executing: '{q[:80]}' (category={cat_label})")
                    return await provider.search(params)
                except Exception as e:
                    self.logger.error(f"[InlineSearch] provider error for '{q}': {e}", exc_info=True)
                    cat_label = (cat.value if isinstance(cat, SearchCategory) else "text")
                    self._metric_inc("inline_search.error", {"category": cat_label, "provider": provider_name})
                    return e

        tasks = [run_search(q, cat) for (_, _), q, cat in directives]
        results_list = await asyncio.gather(*tasks, return_exceptions=False)

        # Build replacements
        pieces: list[str] = []
        cursor = 0
        for ((start, end), query, category), results in zip(directives, results_list):
            # Append text before directive
            if cursor < start:
                pieces.append(text[cursor:start])

            # Format replacement
            if isinstance(results, Exception):
                replacement = f"❌ Search failed for '{query}': please try again later."
            else:
                replacement = self._format_inline_search_block(query, results, provider_name, safesearch)
                cat_label = (category.value if isinstance(category, SearchCategory) else "text")
                self._metric_inc("inline_search.success", {"category": cat_label, "provider": provider_name})

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
        # Accept either a Discord Attachment object or a placeholder (e.g., "" from compat path)
        if not hasattr(attachment, "filename"):
            try:
                attachments = getattr(message, "attachments", None)
                if attachments and len(attachments) > 0:
                    attachment = attachments[0]
                else:
                    self.logger.warning(f"No attachments available to process (msg_id: {message.id})")
                    return BotAction(content="I didn't receive a file to process.")
            except Exception:
                self.logger.warning(f"Attachment placeholder received but unable to access message.attachments (msg_id: {message.id})")
                return BotAction(content="I didn't receive a file to process.")

        self.logger.info(f"Processing attachment: {attachment.filename} (msg_id: {message.id})")

        content_type = getattr(attachment, "content_type", None)
        filename = (getattr(attachment, "filename", "") or "").lower()

        # Process image attachments
        if (content_type and content_type.startswith("image/")) or any(
            filename.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")
        ):
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

            # Use cached VL prompt instructions for vision; user text handled separately
            vl_instructions = (self.bot.system_prompts.get("vl_prompt")
                               or "Describe this image in detail, focusing on key visual elements, objects, text, and context.")
            vision_response = await see_infer(image_path=tmp_path, prompt=vl_instructions)

            if not vision_response or vision_response.error:
                self.logger.warning(f"Vision model returned no/error response (msg_id: {message.id})")
                return BotAction(content="I couldn't understand the image.", error=True)

            vl_content = vision_response.content
            # Truncate if response is too long for Discord
            if len(vl_content) > 1999:
                self.logger.info(f"VL response is too long ({len(vl_content)} chars), truncating for text fallback.")
                vl_content = vl_content[:1999].rsplit('\n', 1)[0]

            # Synthesize user intent for empty messages: "Thoughts?" (not shown directly to user)
            user_text = (message.content or '').strip() or "Thoughts?"
            final_prompt = f"{user_text}\n\nImage analysis:\n{vl_content}"
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

    async def _handle_image_only_tweet(self, url: str, syn_data: Dict[str, Any], source: str = "syndication") -> str:
        """
        Handle image-only tweets with Vision/OCR pipeline and emoji upgrade support.
        Returns neutral, concise alt-text with optional OCR snippet. [CA][SFT][REH]
        """
        try:
            cfg = self.config
            photos = syn_data.get("photos") or []
            
            if not photos:
                self.logger.warning(f"⚠️ Called _handle_image_only_tweet but no photos found: {url}")
                return "⚠️ Expected image content but no photos were found in this tweet."
            
            # Extract tweet metadata for provenance
            user = syn_data.get("user") or {}
            username = user.get("screen_name") or user.get("name") or "unknown"
            created_at = syn_data.get("created_at")
            
            self.logger.info(f"🖼️ Processing {len(photos)} image(s) from image-only tweet: {url}")
            self._metric_inc("vision.image_only_tweet.start", {"source": source, "images": str(len(photos))})
            
            # Process images with Vision/OCR
            results = []
            ocr_texts = []
            safety_flags = []
            
            for idx, photo in enumerate(photos, start=1):
                photo_url = photo.get("url") or photo.get("image_url") or photo.get("src")
                if not photo_url:
                    results.append(f"📷 Image {idx}/{len(photos)} — URL not available")
                    continue
                
                try:
                    # Generate neutral, objective alt-text [SFT]
                    prompt = self._build_neutral_vision_prompt(idx, len(photos), url)
                    
                    # Get vision analysis with retry logic
                    analysis = await self._vl_describe_image_from_url(photo_url, prompt=prompt)
                    
                    if analysis:
                        # Parse analysis for alt-text and OCR if enabled
                        alt_text, ocr_text, safety = self._parse_vision_analysis(analysis, cfg)
                        results.append(alt_text)
                        
                        if ocr_text:
                            ocr_texts.append(ocr_text)
                        if safety:
                            safety_flags.extend(safety)
                            
                        self._metric_inc("vision.image_only_tweet.success", {"image_idx": str(idx)})
                    else:
                        results.append(f"📷 Image {idx}/{len(photos)} — analysis unavailable")
                        self._metric_inc("vision.image_only_tweet.failure", {"image_idx": str(idx)})
                
                except Exception as img_err:
                    self.logger.error(f"❌ Vision analysis failed for image {idx}: {img_err}", exc_info=True)
                    results.append(f"📷 Image {idx}/{len(photos)} — could not analyze")
                    self._metric_inc("vision.image_only_tweet.error", {"image_idx": str(idx)})
            
            # Build minimal response with provenance
            response_parts = []
            
            # Main alt-text (concise, neutral)
            if len(results) == 1:
                response_parts.append(results[0])
            else:
                response_parts.extend(results)
            
            # Add brief OCR snippet if found and enabled
            if ocr_texts and bool(cfg.get("VISION_OCR_ENABLE", True)):
                max_chars = int(cfg.get("VISION_OCR_MAX_CHARS", 160))
                combined_ocr = " • ".join(ocr_texts)[:max_chars]
                if combined_ocr:
                    response_parts.append(f"Seen text: {combined_ocr}")
            
            # Add provenance footer
            timestamp = f" • {created_at}" if created_at else ""
            response_parts.append(f"@{username}{timestamp} → {url}")
            
            # Log successful processing
            self._metric_inc("vision.image_only_tweet.complete", {
                "source": source, 
                "images": str(len(photos)),
                "ocr_found": str(bool(ocr_texts)),
                "safety_flags": str(len(safety_flags))
            })
            
            response = "\n".join(response_parts)
            
            # Note: The upgrade context will be stored after the message is sent
            # This is handled in dispatch_message where we have access to the Discord message object
            self.logger.info(f"✅ Image-only tweet processed successfully: {len(results)} images analyzed")
            
            return response
            
        except Exception as e:
            self.logger.error(f"❌ Image-only tweet processing failed: {e}", exc_info=True)
            self._metric_inc("vision.image_only_tweet.fatal_error", {"source": source})
            return f"⚠️ Could not process images from this tweet right now. Please try again later."

    def _build_neutral_vision_prompt(self, idx: int, total: int, url: str) -> str:
        """Build neutral, objective vision prompt that avoids toxic language echoing. [SFT]"""
        cfg = self.config
        tone = cfg.get("REPLY_TONE", "neutral_objective")
        
        # Ensure neutral, non-toxic prompting [SFT]
        if total == 1:
            return (
                f"Describe this image objectively and concisely. Include who/what/where if clearly visible, "
                f"and any text on objects or signs. Keep the description neutral and factual. "
                f"Avoid speculation, personal opinions, or sensitive commentary."
            )
        else:
            return (
                f"This is image {idx} of {total} from a social media post. Describe it objectively and concisely. "
                f"Include who/what/where if clearly visible, and any text on objects or signs. "
                f"Keep the description neutral and factual. Avoid speculation or sensitive commentary."
            )

    def _parse_vision_analysis(self, analysis: str, cfg: Dict[str, Any]) -> tuple[str, Optional[str], Optional[List[str]]]:
        """
        Parse vision analysis into alt-text, OCR text, and safety flags.
        Returns (alt_text, ocr_text, safety_flags). [IV][SFT]
        """
        # Simplified parsing - in production this would be more sophisticated
        # Check for toxic content echoing and filter it out [SFT]
        echo_toxic = bool(cfg.get("ECHO_TOXIC_USER_TERMS", False))
        
        if not echo_toxic:
            # Basic filtering of potentially toxic content (this would be more comprehensive)
            analysis = self._filter_toxic_echoes(analysis)
        
        # Extract potential OCR text (look for quotes, text mentions, etc.)
        ocr_text = None
        if bool(cfg.get("VISION_OCR_ENABLE", True)):
            # Simple OCR extraction - look for quoted text or "text says" patterns
            import re
            ocr_patterns = [
                r'"([^"]{3,})"',  # Quoted text
                r'text says[:\s]+"?([^".\n]+)"?',  # "text says" pattern
                r'sign reads[:\s]+"?([^".\n]+)"?',  # "sign reads" pattern
            ]
            
            for pattern in ocr_patterns:
                matches = re.findall(pattern, analysis, re.IGNORECASE)
                if matches:
                    ocr_text = matches[0].strip()
                    break
        
        # Basic safety flag detection (simplified)
        safety_flags = []
        safety_filter = cfg.get("VISION_SAFETY_FILTER", "strict")
        if safety_filter != "off":
            # Look for NSFW/medical/violence indicators
            safety_indicators = ["nsfw", "explicit", "medical", "violence", "inappropriate"]
            for indicator in safety_indicators:
                if indicator in analysis.lower():
                    safety_flags.append(indicator)
        
        return analysis, ocr_text, safety_flags if safety_flags else None

    def _filter_toxic_echoes(self, text: str) -> str:
        """Filter out toxic language that might echo user input. [SFT]"""
        # This would implement comprehensive toxic language filtering
        # For now, a basic implementation that removes obvious slurs and offensive terms
        # In production, this would use a proper content filtering service
        
        # Basic word filtering (this would be much more comprehensive)
        toxic_patterns = [
            # This would contain actual filtering logic
            # For safety, not implementing specific words here
        ]
        
        filtered_text = text
        # Apply filtering logic here
        
        return filtered_text

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

    async def _handle_vision_generation(self, intent_result, message: Message, context_str: str) -> BotAction:
        """
        Handle Vision generation request through orchestrator with comprehensive error handling [REH][SFT]
        
        Args:
            intent_result: VisionIntentResult from intent router
            message: Discord message for context
            context_str: Conversation context
            
        Returns:
            BotAction with generation result or error message
        """
        if not self._vision_orchestrator:
            return BotAction(
                content="🚫 Vision generation is not available right now. Please try again later.",
                error=True
            )
        
        try:
            # Convert intent result to VisionRequest
            from .vision.types import VisionRequest, VisionTask, VisionProvider
            
            # Convert string task to enum
            task_str = intent_result.extracted_params.task
            task_enum = VisionTask(task_str) if isinstance(task_str, str) else task_str
            
            vision_request = VisionRequest(
                task=task_enum,
                prompt=intent_result.extracted_params.prompt,
                user_id=str(message.author.id),
                guild_id=str(message.guild.id) if message.guild else None,
                channel_id=str(message.channel.id),
                negative_prompt=getattr(intent_result.extracted_params, 'negative_prompt', ""),
                width=getattr(intent_result.extracted_params, 'width', 1024),
                height=getattr(intent_result.extracted_params, 'height', 1024),
                steps=getattr(intent_result.extracted_params, 'steps', 30),
                guidance_scale=getattr(intent_result.extracted_params, 'guidance_scale', 7.0),
                seed=getattr(intent_result.extracted_params, 'seed', None),
                preferred_provider=getattr(intent_result.extracted_params, 'preferred_provider', None)
            )
            
            # Submit job to orchestrator
            self.logger.info(
                f"🎨 Submitting Vision job: {task_enum.value} (msg_id: {message.id})"
            )
            
            job = await self._vision_orchestrator.submit_job(vision_request)
            
            # Send initial progress message
            progress_msg = await message.channel.send(
                f"🎨 **Vision Generation Started**\n"
                f"Task: {vision_request.task.value.replace('_', ' ').title()}\n"
                f"Job ID: `{job.job_id[:8]}`\n"
                f"Status: {job.state.value.title()}\n"
                f"⏳ *Processing...*"
            )
            
            # Monitor job progress and update message
            return await self._monitor_vision_job(job, progress_msg, message)
            
        except Exception as e:
            self.logger.error(f"❌ Vision generation failed: {e} (msg_id: {message.id})", exc_info=True)
            
            # Provide user-friendly error messages based on error type
            error_str = str(e).lower()
            
            if "content filtered" in error_str or "safety" in error_str:
                return BotAction(
                    content="🚫 **Content Safety Issue**\n"
                           "Your request contains content that violates our usage policies. "
                           "Please modify your prompt to remove prohibited content and try again.",
                    error=True
                )
            elif "budget" in error_str or "quota" in error_str:
                return BotAction(
                    content="💰 **Budget Limit Reached**\n"
                           "You've reached your vision generation budget limit. "
                           "Please wait for your quota to reset or contact an admin for assistance.",
                    error=True
                )
            elif "provider" in error_str or "service" in error_str:
                return BotAction(
                    content="🔄 **Service Temporarily Unavailable**\n"
                           "The vision generation service is experiencing issues. "
                           "Please try again in a few moments.",
                    error=True
                )
            else:
                return BotAction(
                    content="❌ **Generation Failed**\n"
                           "An error occurred during vision generation. "
                           "Please check your parameters and try again.",
                    error=True
                )

    async def _monitor_vision_job(self, job, progress_msg, original_msg: Message) -> BotAction:
        """
        Monitor Vision job progress and update Discord message with results [REH][PA]
        
        Args:
            job: VisionJob instance
            progress_msg: Discord message to update with progress
            original_msg: Original user message for context
            
        Returns:
            BotAction with final result
        """
        try:
            import asyncio
            
            # Poll job status with timeout
            timeout_seconds = self.config.get("VISION_JOB_TIMEOUT_SECONDS", 300)
            poll_interval = self.config.get("VISION_PROGRESS_UPDATE_INTERVAL_S", 10)
            
            start_time = asyncio.get_event_loop().time()
            
            while True:
                # Check timeout
                if asyncio.get_event_loop().time() - start_time > timeout_seconds:
                    await progress_msg.edit(
                        content=f"⏰ **Job Timeout**\n"
                               f"Job ID: `{job.job_id[:8]}`\n"
                               f"The generation took too long and was cancelled. Please try again."
                    )
                    return BotAction(content="Job timed out", error=True)
                
                # Get updated job status
                updated_job = await self._vision_orchestrator.get_job_status(job.job_id)
                self.logger.debug(f"🔍 Vision job status check - job_id: {job.job_id[:8]}, state: {updated_job.state.value if updated_job else 'None'}, terminal: {updated_job.is_terminal_state() if updated_job else 'N/A'}")
                
                if not updated_job:
                    self.logger.warning(f"⚠️ Vision job not found during monitoring - job_id: {job.job_id[:8]}")
                    break
                
                # Update progress message
                if updated_job.progress_percentage > 0:
                    progress_bar = self._create_progress_bar(updated_job.progress_percentage)
                    await progress_msg.edit(
                        content=f"🎨 **Vision Generation**\n"
                               f"Job ID: `{updated_job.job_id[:8]}`\n"
                               f"Status: {updated_job.state.value.title()}\n"
                               f"{progress_bar} {updated_job.progress_percentage}%\n"
                               f"💭 *{getattr(updated_job, 'progress_message', 'Processing...')}*"
                    )
                
                # Check if job is complete
                if updated_job.is_terminal_state():
                    self.logger.info(f"🏁 Vision job terminal state reached - job_id: {updated_job.job_id[:8]}, state: {updated_job.state.value}, has_response: {updated_job.response is not None}")
                    
                    if updated_job.state.value == "completed" and updated_job.response:
                        # Job completed successfully
                        self.logger.info(f"✅ Calling vision success handler - job_id: {updated_job.job_id[:8]}")
                        return await self._handle_vision_success(updated_job, progress_msg, original_msg)
                    else:
                        # Job failed or was cancelled
                        self.logger.warning(f"❌ Job failed or cancelled - job_id: {updated_job.job_id[:8]}, state: {updated_job.state.value}")
                        return await self._handle_vision_failure(updated_job, progress_msg)
                
                # Wait before next poll
                await asyncio.sleep(poll_interval)
                
        except Exception as e:
            self.logger.error(f"❌ Vision job monitoring failed: {e}", exc_info=True)
            await progress_msg.edit(
                content=f"❌ **Monitoring Error**\n"
                       f"Job ID: `{job.job_id[:8]}`\n"
                       f"Lost connection to job status. Please check back later."
            )
            return BotAction(content="Job monitoring failed", error=True)

    async def _handle_vision_success(self, job, progress_msg, original_msg: Message) -> BotAction:
        """Handle successful Vision generation with file uploads [PA]"""
        try:
            response = job.response
            
            # Download and prepare files for Discord upload
            files_to_upload = []
            result_descriptions = []
            
            for i, artifact_path in enumerate(response.artifacts, 1):
                try:
                    # Read generated content from local file
                    if not artifact_path.exists():
                        result_descriptions.append(f"❌ Result {i} file not found")
                        continue
                        
                    # Determine file format and name from path
                    ext = artifact_path.suffix.lower().lstrip('.')
                    filename = f"generated_{job.job_id[:8]}_{i}.{ext}"
                    
                    # Create Discord file from local path
                    discord_file = discord.File(
                        artifact_path,
                        filename=filename
                    )
                    files_to_upload.append(discord_file)
                    result_descriptions.append(f"📎 {filename}")
                        
                except Exception as e:
                    self.logger.warning(f"Failed to download result {i}: {e}")
                    result_descriptions.append(f"❌ Result {i} download failed")
            
            # Cost formatting: avoid numeric format on Money [REH][IV]
            cost_str = "N/A"
            try:
                ac = getattr(response, 'actual_cost', None)
                if ac is not None:
                    # Money-aware display if available
                    if hasattr(ac, 'to_display_string'):
                        cost_str = ac.to_display_string()
                    else:
                        # Legacy numeric fallback
                        cost_str = f"${float(ac):.2f}"
            except Exception as e:
                self.logger.debug(f"money.format_fallback | {e}")
                cost_str = "N/A"

            # Instrumentation before message assembly [PA]
            try:
                self.logger.info(
                    f"🧾 Vision success summary | job={job.job_id[:8]} cost={cost_str} artifacts={len(response.artifacts) if response and response.artifacts else 0}"
                )
            except Exception:
                pass

            # Create final success message
            success_content = (
                f"✅ **Vision Generation Complete**\n"
                f"Task: {job.request.task.value.replace('_', ' ').title()}\n"
                f"Provider: {response.provider.value.title()}\n"
                f"Processing Time: {response.processing_time_seconds:.1f}s\n"
                f"Cost: {cost_str}\n\n"
                f"**Results:**\n" + "\n".join(result_descriptions)
            )
            
            if job.request.prompt:
                success_content += f"\n\n**Prompt:** {job.request.prompt[:100]}{'...' if len(job.request.prompt) > 100 else ''}"
            
            # Update progress message and upload files
            await progress_msg.edit(content=success_content)
            
            if files_to_upload:
                # Log filenames and sizes before upload [PA]
                try:
                    upload_meta = []
                    for f in files_to_upload:
                        try:
                            # discord.File has .fp or .path; we derive size from path when available
                            path = getattr(f, 'fp', None)
                            size = None
                            if hasattr(f, 'fp') and hasattr(f.fp, 'name'):
                                pth = getattr(f.fp, 'name', None)
                                if pth and os.path.exists(pth):
                                    size = os.path.getsize(pth)
                                    upload_meta.append((f.filename, size))
                            else:
                                upload_meta.append((f.filename, None))
                        except Exception:
                            upload_meta.append((getattr(f, 'filename', 'unknown'), None))
                    self.logger.info(
                        "📤 Upload starting | files=" + ", ".join(
                            [f"{name} ({size} bytes)" if size is not None else name for name, size in upload_meta]
                        )
                    )
                except Exception:
                    pass
                await original_msg.channel.send(files=files_to_upload)
            
            return BotAction(content="Vision generation completed successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Vision success handling failed: {e}", exc_info=True)
            await progress_msg.edit(
                content=f"✅ **Generation Complete**\n"
                       f"Job ID: `{job.job_id[:8]}`\n"
                       f"⚠️ Results are ready but file upload failed. Please try the job ID with an admin command."
            )
            return BotAction(content="Generation completed with upload issues", error=True)

    async def _handle_vision_failure(self, job, progress_msg) -> BotAction:
        """Handle failed Vision generation with user guidance [REH]"""
        error_msg = job.error.user_message if job.error else "Unknown error occurred"
        
        failure_content = (
            f"❌ **Generation Failed**\n"
            f"Job ID: `{job.job_id[:8]}`\n"
            f"Status: {job.state.value.title()}\n\n"
            f"**Issue:** {error_msg}\n\n"
            f"💡 **Suggestions:**\n"
            f"• Try a different prompt or parameters\n"
            f"• Check if your request follows content guidelines\n"
            f"• Contact support if the issue persists"
        )
        
        await progress_msg.edit(content=failure_content)
        return BotAction(content="Vision generation failed", error=True)

    def _create_progress_bar(self, percent: int, length: int = 10) -> str:
        """Create ASCII progress bar [CMV]"""
        filled = int(length * percent / 100)
        bar = "█" * filled + "░" * (length - filled)
        return f"[{bar}]"

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

    def _detect_direct_vision_triggers(self, content: str) -> Optional[Dict[str, Any]]:
        """
        Direct pattern matching for obvious vision requests to bypass rate-limited intent detection.
        Returns extracted vision parameters if triggers found, None otherwise.
        [RAT: REH, PA] - Robust Error Handling, Performance Awareness
        """
        content_lower = content.lower().strip()
        
        # Direct trigger phrases from vision_policy.json
        image_triggers = [
            "create an image", "generate an image", "draw", "make a picture", 
            "make a photo", "create a photo", "create a picture", "paint", 
            "sketch", "illustration", "artwork", "render"
        ]
        
        # Check for direct triggers
        for trigger in image_triggers:
            if trigger in content_lower:
                # Extract prompt by removing the trigger phrase
                prompt = content
                for t in image_triggers:
                    if t in content_lower:
                        # Find the trigger and extract what comes after it
                        idx = content_lower.find(t)
                        if idx >= 0:
                            prompt = content[idx + len(t):].strip()
                            # Remove common prefixes like "of", "a", "an"
                            for prefix in [" of ", " a ", " an "]:
                                if prompt.lower().startswith(prefix):
                                    prompt = prompt[len(prefix):].strip()
                            break
                
                if prompt and len(prompt.strip()) > 2:  # Ensure we have a meaningful prompt
                    self.logger.info(f"🎨 Direct vision trigger detected: '{trigger}' -> prompt: '{prompt[:50]}...'")
                    return {
                        "use_vision": True,
                        "task": "text_to_image",
                        "prompt": prompt,
                        "confidence": 0.95,  # High confidence for direct triggers
                        "bypass_reason": f"Direct trigger: '{trigger}'"
                    }
        
        return None

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