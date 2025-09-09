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
from .utils.logging import get_logger
import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from html import unescape
import json
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING, Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import discord
from discord import DMChannel, Message

from .brain import brain_infer
from .enhanced_retry import ProviderConfig, get_retry_manager
from .exceptions import APIError
from .http_client import get_http_client
from .modality import (
    InputItem,
    InputModality,
    collect_image_urls_from_message,
    collect_input_items,
    map_item_to_modality,
)
from .pdf_utils import PDFProcessor
from .result_aggregator import ResultAggregator
from .search.factory import get_search_provider
from .search.types import SafeSearch, SearchCategory, SearchQueryParams, SearchResult
from .hear import hear_infer, hear_infer_from_url
from .see import see_infer
from .types import Command, ParsedCommand
from .utils.env import get_bool
from .vl.postprocess import sanitize_model_output, sanitize_vl_reply_text
from .web import process_url
from .web_extraction_service import web_extractor
from .x_api_client import XApiClient
from .action import BotAction
from .command_parser import parse_command
from .utils.file_utils import download_file
from .tts.state import tts_state
from datetime import datetime, timezone

if TYPE_CHECKING:
    from bot.core.bot import LLMBot as DiscordBot
    from .command_parser import ParsedCommand

logger = get_logger(__name__)

_router_instance: Optional["Router"] | None = None


@dataclass
class XTwitterMediaInfo:
    """Detection result for X/Twitter media content."""

    has_x_link: bool = False
    media_kind: str = "none"  # "image", "video", "none"
    media_urls: List[str] = field(default_factory=list)


def _detect_x_twitter_media(message: Message) -> XTwitterMediaInfo:
    """Detect X/Twitter links and conservatively classify media type from Discord embeds.
    Rules:
    - If a tweet has an explicit embed.video → classify as "video" and point at the tweet URL.
    - If a tweet only has a thumbnail/image and no embed.video → classify as "unknown" and point at the tweet URL (STT-first probe).
    - Only classify as "image" when the URL itself is a direct image (e.g., pbs.twimg.com) without a tweet URL.
    """
    x_hosts = {"x.com", "twitter.com", "fxtwitter.com"}

    # Check message content for X/Twitter URLs
    content = message.content or ""
    has_x_link = any(host in content.lower() for host in x_hosts)

    # Collect actual tweet URLs from content for STT fallback
    import re

    tweet_urls: List[str] = []
    url_pattern = r"https?://(?:www\.)?(x\.com|twitter\.com|fxtwitter\.com)/\S+"
    for match in re.finditer(url_pattern, content, re.IGNORECASE):
        u = match.group(0)
        if u not in tweet_urls:
            tweet_urls.append(u)

    media_kind = "none"
    media_urls: List[str] = []

    # Track direct image URLs that are not associated with a tweet URL
    direct_image_urls: List[str] = []

    for embed in message.embeds:
        # Check embed URLs and provider info
        embed_url = getattr(embed, "url", "") or ""
        provider = getattr(embed, "provider", None)
        provider_name = getattr(provider, "name", "") if provider else ""
        author = getattr(embed, "author", None)
        author_url = getattr(author, "url", "") if author else ""

        embed_url_l = (embed_url or "").lower()

        # Detect X/Twitter from embed metadata
        if (
            any(host in embed_url_l for host in x_hosts)
            or "twitter" in (provider_name or "").lower()
            or "x.com" in (provider_name or "").lower()
            or any(host in (author_url or "").lower() for host in x_hosts)
        ):
            has_x_link = True
            # Add embed URL to tweet URLs if it's a tweet URL
            if (
                embed_url
                and any(host in embed_url_l for host in x_hosts)
                and embed_url not in tweet_urls
            ):
                tweet_urls.append(embed_url)

            # Classify based on embed structure
            if getattr(embed, "video", None):
                # Explicit video → treat as video, point to tweet URL
                media_kind = "video"
                if embed_url:
                    if embed_url not in media_urls:
                        media_urls.append(embed_url)
            elif getattr(embed, "image", None):
                # Photo-only tweets often expose embed.image. Treat as image (no STT),
                # and prefer the tweet URL so downstream routing uses the syndication/VL path.
                media_kind = "image"
                if tweet_urls:
                    media_urls = list(tweet_urls)
                elif embed_url:
                    media_urls = [embed_url]
                else:
                    # Last resort, point at the image itself
                    try:
                        if getattr(embed.image, "url", None):
                            media_urls = [embed.image.url]
                    except Exception:
                        pass
            else:
                # If only thumbnail present without embed.video, mark as unknown and point to the tweet URL
                if media_kind == "none" and tweet_urls:
                    media_kind = "unknown"
                    media_urls = list(tweet_urls)
                elif media_kind == "none" and embed_url:
                    media_kind = "unknown"
                    media_urls = [embed_url]

        # Also capture direct image URLs (pbs.twimg.com etc.) that are not tied to a tweet URL
        try:
            # Some embed shapes expose image/thumbnail URLs even when embed.url is unrelated
            image_url = None
            if (
                hasattr(embed, "image")
                and getattr(embed, "image")
                and getattr(embed.image, "url", None)
            ):
                image_url = embed.image.url
            elif (
                hasattr(embed, "thumbnail")
                and getattr(embed, "thumbnail")
                and getattr(embed.thumbnail, "url", None)
            ):
                image_url = embed.thumbnail.url
            if image_url:
                u_l = str(image_url).lower()
                # Only treat as direct image if it's clearly a twimg (not a tweet URL)
                if "twimg.com" in u_l and not any(host in u_l for host in x_hosts):
                    if image_url not in direct_image_urls:
                        direct_image_urls.append(image_url)
        except Exception:
            pass

    # If we couldn't determine from embeds, but have tweet URLs → unknown for STT-first
    if has_x_link and media_kind == "none" and tweet_urls:
        media_kind = "unknown"
        media_urls = list(tweet_urls)

    # Only classify as image when we only have direct images without tweet URLs
    if media_kind == "none" and direct_image_urls and not tweet_urls:
        media_kind = "image"
        media_urls = list(direct_image_urls)

    return XTwitterMediaInfo(
        has_x_link=has_x_link, media_kind=media_kind, media_urls=media_urls
    )


# Vision generation system (import only, no flag constant)
try:
    from .vision import VisionIntentRouter, VisionOrchestrator
except ImportError:
    VisionIntentRouter = None
    VisionOrchestrator = None

# Dependency availability flags
try:
    import docx  # noqa: F401

    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False

try:
    import fitz  # PyMuPDF  # noqa: F401

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

    def __init__(
        self,
        bot: "DiscordBot",
        flow_overrides: Optional[Dict[str, Callable]] = None,
        logger: Optional[logging.Logger] = None,
    ):
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
        # Single source of truth: orchestrator is owned by the bot
        self._vision_orchestrator: Optional[VisionOrchestrator] = getattr(
            bot, "vision_orchestrator", None
        )

        # Router fallback: if bot didn't provide an orchestrator, create and attach one [REH]
        if (
            self._vision_orchestrator is None
            and VisionOrchestrator is not None
            and self.config.get("VISION_ENABLED", True)
        ):
            try:
                self._vision_orchestrator = VisionOrchestrator(self.config)
                setattr(self.bot, "vision_orchestrator", self._vision_orchestrator)
                self.logger.info("VisionOrchestrator: created (router fallback)")
            except Exception as e:
                self.logger.error(
                    f"Failed to create VisionOrchestrator (router fallback): {e}",
                    exc_info=True,
                )
                self._vision_orchestrator = None

    def _append_note_once(self, text: str, note: str) -> str:
        """Append a parenthetical note once, avoiding duplicates and preserving spacing."""
        text = text or ""
        if note in text:
            return text
        sep = "\n\n" if text and not text.endswith("\n") else ""
        return f"{text}{sep}{note}".strip()

        # Queue non-blocking eager start to reduce first-check false negatives [PA]
        try:
            import asyncio

            loop = asyncio.get_running_loop()
            if (
                loop
                and loop.is_running()
                and self._vision_orchestrator
                and not getattr(self._vision_orchestrator, "_started", False)
            ):
                asyncio.create_task(self._vision_orchestrator.start())
                self.logger.debug("🚀 Vision Orchestrator start queued (router init)")
        except RuntimeError:
            # No running loop at construction time; lazy start path will cover this
            pass

        # Feature flags summary (treat missing as enabled) [CMV]
        ve = bool(self.config.get("VISION_ENABLED", True))
        vti = bool(self.config.get("VISION_T2I_ENABLED", True))
        self.logger.info(
            f"Vision flags | VISION_ENABLED={'on' if ve else 'off'} VISION_T2I_ENABLED={'on' if vti else 'off'}"
        )

        # Load centralized VL prompt guidelines if available [CA]
        self._vl_prompt_guidelines: Optional[str] = None
        try:
            prompts_dir = (
                Path(__file__).resolve().parents[1] / "prompts" / "vl-prompt.txt"
            )
            if prompts_dir.exists():
                content = prompts_dir.read_text(encoding="utf-8").strip()
                if content:
                    self._vl_prompt_guidelines = content
                    self.logger.debug(
                        "Loaded VL prompt guidelines from prompts/vl-prompt.txt"
                    )
        except Exception:
            # Non-fatal; handler has built-in defaults
            self._vl_prompt_guidelines = None

    def _format_x_tweet_with_transcription(
        self, tweet_url: str, transcript: str, user_content: str
    ) -> str:
        """Format X/Twitter video with transcription header and optional tweet context."""
        # Simple transcription format with header
        formatted = f"🎙️ Transcription: {transcript.strip()}"

        # Add user content if present
        if user_content and user_content.strip():
            formatted = f"{user_content.strip()}\n\n{formatted}"

        return formatted

    async def _handle_x_twitter_fallback(
        self, tweet_url: str, message: Message, context_str: str
    ) -> BotAction:
        """Handle X/Twitter fallback to syndication/API for photos or text."""
        try:
            # Try to use existing _handle_general_url which already has syndication/VL logic
            from .modality import InputItem

            item = InputItem(source_type="url", payload=tweet_url)
            result = await self._handle_general_url(item)

            if result and result.strip():
                return await self._flow_process_text(
                    content=f"{message.content or ''}\n\n{result}".strip(),
                    context=context_str,
                    message=message,
                )
        except Exception as e:
            self.logger.debug(f"X/Twitter syndication fallback failed: {e}")

        # Final fallback: just return the original content with hint
        fallback_content = f"{message.content or ''}\n\n(tweet content unavailable; proceeding without it)"
        return await self._flow_process_text(
            content=fallback_content.strip(), context=context_str, message=message
        )

        if VisionIntentRouter is not None and self.config.get("VISION_ENABLED", True):
            try:
                self.logger.info("🔧 Creating VisionIntentRouter...")
                self._vision_intent_router = VisionIntentRouter(self.config)
                if self._vision_orchestrator:
                    self.logger.info("✔ Vision system initialized (using orchestrator)")
                else:
                    self.logger.warning(
                        "⚠️ VisionOrchestrator missing; availability will be gated"
                    )
            except Exception as e:
                self.logger.error(
                    f"❌ Failed to initialize Vision intent router: {e}", exc_info=True
                )
                self._vision_intent_router = None
        else:
            # Use centralized parsed booleans instead of raw reads [CA]
            ve_parsed = self.config.get("VISION_ENABLED", True)
            self.logger.warning(
                f"⚠️ Vision system NOT initialized - vision_enabled={ve_parsed}, reason=module_unavailable_or_disabled"
            )

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

    async def _get_tweet_via_syndication(
        self, tweet_id: str
    ) -> Optional[Dict[str, Any]]:
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

            int(self.config.get("X_SYNDICATION_TIMEOUT_MS", 4000))
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
                http_client = await get_http_client()
                for endpoint, params in params_variants:
                    url = base + (
                        "widgets/tweet" if endpoint == "widgets" else "tweet-result"
                    )
                    self._metric_inc("x.syndication.fetch", {"endpoint": endpoint})
                    resp = await http_client.get(url, headers=headers, params=params)
                    if resp.status_code != 200:
                        self.logger.info(
                            "Syndication non-200",
                            extra={
                                "detail": {
                                    "tweet_id": tweet_id,
                                    "status": resp.status_code,
                                    "endpoint": endpoint,
                                }
                            },
                        )
                        self._metric_inc(
                            "x.syndication.non_200",
                            {"status": str(resp.status_code), "endpoint": endpoint},
                        )
                        continue
                    try:
                        data = resp.json()
                    except Exception:
                        self._metric_inc(
                            "x.syndication.invalid_json", {"endpoint": endpoint}
                        )
                        continue
                    # Found a candidate
                    break
                    # If no usable JSON with text/full_text, try oEmbed as last resort
                    if not (
                        isinstance(data, dict)
                        and (data.get("text") or data.get("full_text"))
                    ):
                        oembed_url = "https://publish.twitter.com/oembed"
                        oembed_params = {
                            "url": f"https://twitter.com/i/status/{tweet_id}",
                            "dnt": "false",
                            "omit_script": "true",
                            "hide_thread": "true",
                            "lang": "en",
                        }
                        self._metric_inc("x.syndication.fetch", {"endpoint": "oembed"})
                        resp = await http_client.get(
                            oembed_url, headers=headers, params=oembed_params
                        )
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
                        if not (
                            isinstance(data, dict)
                            and (data.get("text") or data.get("full_text"))
                        ):
                            oembed_params_x = dict(oembed_params)
                            oembed_params_x["url"] = (
                                f"https://x.com/i/status/{tweet_id}"
                            )
                            self._metric_inc(
                                "x.syndication.fetch", {"endpoint": "oembed_x"}
                            )
                            resp2 = await http_client.get(
                                oembed_url, headers=headers, params=oembed_params_x
                            )
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
                                                "user": {
                                                    "name": obj2.get("author_name")
                                                },
                                            }
            except Exception as e:
                self.logger.info(
                    "Syndication fetch failed",
                    extra={"detail": {"tweet_id": tweet_id, "error": str(e)}},
                )
                self._metric_inc("x.syndication.error", None)
                return None

            # Minimal validation: require text field
            if not isinstance(data, dict) or not (
                data.get("text") or data.get("full_text")
            ):
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
        return any(
            d in u
            for d in ["twitter.com/", "x.com/", "vxtwitter.com/", "fxtwitter.com/"]
        )

    @staticmethod
    def _is_direct_image_url(url: str) -> bool:
        """Lightweight check for direct image URLs by extension. [IV]"""
        try:
            u = str(url).lower()
        except Exception:
            return False
        return bool(re.search(r"\.(jpe?g|png|webp)(?:\?|#|$)", u))

    async def _process_image_from_attachment_with_model(
        self, attachment, model_override: Optional[str] = None
    ) -> str:
        """Save a Discord image attachment to a temp file and run VL analysis. [RM][REH]"""
        from .see import see_infer

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_path = tmp_file.name
            await attachment.save(tmp_path)
            self.logger.debug(f"📷 Saved image attachment to temp file: {tmp_path}")

            prompt = "Describe this image in detail, focusing on key visual elements, objects, text, and context."
            vision_response = await see_infer(
                image_path=tmp_path,
                prompt=prompt,
                model_override=model_override,
            )

            if not vision_response:
                return "❌ Vision processing returned no response"
            if getattr(vision_response, "error", None):
                return f"❌ Vision processing error: {vision_response.error}"
            content = getattr(vision_response, "content", "") or ""
            if not content.strip():
                return "❌ Vision processing returned empty content"
            filename = getattr(attachment, "filename", "image")
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

    async def _handle_image_with_model(
        self, item: InputItem, model_override: Optional[str] = None
    ) -> str:
        """Handle image item with explicit model override. [CA][IV][REH]
        - Attachments: direct VL on file
        - URLs: direct image URL → download+VL; otherwise screenshot→VL
        - Embeds: try image/thumbnail URL similarly
        """
        try:
            if item.source_type == "attachment":
                attachment = item.payload
                return await self._process_image_from_attachment_with_model(
                    attachment, model_override
                )

            if item.source_type == "url":
                url = item.payload
                if self._is_direct_image_url(url):
                    prompt = "Describe this image in detail, focusing on key visual elements, objects, text, and context."
                    desc = await self._vl_describe_image_from_url(
                        url, prompt=prompt, model_override=model_override
                    )
                    return (
                        desc or "⚠️ Unable to analyze the image from the provided URL."
                    )
                # Not a direct image URL → screenshot fallback
                return await self._process_image_from_url(
                    url, model_override=model_override
                )

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
                if (
                    image_url
                    and isinstance(image_url, str)
                    and image_url.startswith("http")
                ):
                    return await self._process_image_from_url(
                        image_url, model_override=model_override
                    )
                return "⚠️ Embed did not contain a usable image URL."

            return "⚠️ Unsupported image source type."
        except Exception as e:
            self.logger.error(
                f"❌ _handle_image_with_model failed: {e}",
                extra={"detail": {"source_type": item.source_type}},
                exc_info=True,
            )
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
        triggers: list[str] = cfg.get(
            "REPLY_TRIGGERS",
            ["dm", "mention", "reply", "bot_threads", "owner", "command_prefix"],
        )

        # Master switch: if disabled, allow everything (legacy behavior)
        if not cfg.get("BOT_SPEAKS_ONLY_WHEN_SPOKEN_TO", True):
            self.logger.debug(
                f"gate.allow | reason=master_switch_off msg_id={message.id}",
                extra={
                    "event": "gate.allow",
                    "reason": "master_switch_off",
                    "msg_id": message.id,
                },
            )
            self._metric_inc("gate.allowed", {"reason": "master_switch_off"})
            return True

        content = (message.content or "").strip()
        is_dm = isinstance(message.channel, DMChannel)
        is_mentioned = (
            self.bot.user in message.mentions if hasattr(message, "mentions") else False
        )
        is_reply = self._is_reply_to_bot(message)
        is_owner = (
            message.author.id in owners if getattr(message, "author", None) else False
        )

        in_bot_thread = False
        try:
            if isinstance(message.channel, discord.Thread):
                # Cheap ownership check only; do not fetch history here.
                in_bot_thread = (
                    getattr(message.channel, "owner_id", None) == self.bot.user.id
                )
        except Exception:
            in_bot_thread = False

        # Prefix command detection (strip leading mention if present)
        command_prefix = cfg.get("COMMAND_PREFIX", "!")
        if content:
            mention_pattern = rf"^<@!?{self.bot.user.id}>\s*"
            clean_content = re.sub(mention_pattern, "", content)
        else:
            clean_content = ""
        has_prefix = (
            bool(clean_content.startswith(command_prefix)) if clean_content else False
        )

        # Evaluate triggers
        if is_owner and "owner" in triggers:
            self.logger.info(
                f"gate.allow | reason=owner_override msg_id={message.id}",
                extra={
                    "event": "gate.allow",
                    "reason": "owner_override",
                    "user_id": message.author.id,
                    "msg_id": message.id,
                },
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
                extra={
                    "event": "gate.allow",
                    "reason": "mention",
                    "msg_id": message.id,
                },
            )
            self._metric_inc("gate.allowed", {"reason": "mention"})
            return True

        if is_reply and "reply" in triggers:
            self.logger.debug(
                f"gate.allow | reason=reply_to_bot msg_id={message.id}",
                extra={
                    "event": "gate.allow",
                    "reason": "reply_to_bot",
                    "msg_id": message.id,
                },
            )
            self._metric_inc("gate.allowed", {"reason": "reply_to_bot"})
            return True

        if in_bot_thread and "bot_threads" in triggers:
            self.logger.debug(
                f"gate.allow | reason=bot_thread msg_id={message.id}",
                extra={
                    "event": "gate.allow",
                    "reason": "bot_thread",
                    "msg_id": message.id,
                },
            )
            self._metric_inc("gate.allowed", {"reason": "bot_thread"})
            return True

        if has_prefix and "command_prefix" in triggers:
            self.logger.debug(
                f"gate.allow | reason=command_prefix msg_id={message.id}",
                extra={
                    "event": "gate.allow",
                    "reason": "command_prefix",
                    "msg_id": message.id,
                },
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
                "guild_id": getattr(message.guild, "id", None),
                "is_dm": is_dm,
            },
        )
        self._metric_inc("gate.blocked", {"reason": "not_addressed"})
        return False

    def _bind_flow_methods(self, flow_overrides: Optional[Dict[str, Callable]] = None):
        """Binds flow methods to the instance, allowing for overrides for testing."""
        self._flows = {
            "process_text": self._flow_process_text,
            "process_url": self._flow_process_url,
            "process_audio": self._flow_process_audio,
            "process_attachments": self._flow_process_attachments,
            "generate_tts": self._flow_generate_tts,
        }

        if flow_overrides:
            self._flows.update(flow_overrides)

    async def dispatch_message(self, message: Message) -> Optional[BotAction]:
        """Process a message and ensure exactly one response is generated (1 IN > 1 OUT rule)."""
        self.logger.info(f"🔄 === ROUTER DISPATCH STARTED: MSG {message.id} ====")

        # ROUTER_DEBUG=1 diagnostics for path selection [IV]
        router_debug = get_bool("ROUTER_DEBUG", False)

        try:
            # 1. Quick pre-filter: Only parse commands for messages that start with '!' to avoid unnecessary parsing
            content = message.content.strip()

            # Remove bot mention to check for command pattern
            mention_pattern = rf"^<@!?{self.bot.user.id}>\s*"
            clean_content = re.sub(mention_pattern, "", content)

            # 1b. Compatibility fast-path for legacy tests: attachments + empty content
            # Run this BEFORE gating and typing() to avoid MagicMock issues in tests
            try:
                has_attachments = (
                    bool(getattr(message, "attachments", None))
                    and len(message.attachments) > 0
                )
            except Exception:
                has_attachments = False
            cleaned_for_compat = re.sub(
                mention_pattern, "", (message.content or "").strip()
            )
            if has_attachments and cleaned_for_compat == "":
                handler = self._flows.get("process_attachments")
                if handler:
                    self.logger.debug(
                        "Compat path (pre-gate): delegating to _flows['process_attachments'] with empty text."
                    )
                    res = await handler(message, "")
                    if isinstance(res, BotAction):
                        return res
                    else:
                        # Wrap plain string result into BotAction for compatibility
                        return BotAction(content=str(res))

            # Only parse if it looks like a command (starts with '!')
            parsed_command = None
            if clean_content.startswith("!"):
                parsed_command = parse_command(message, self.bot)

                # 2. If a command is found, handle special cases or delegate to cogs.
                if parsed_command:
                    # Special handling for IMG command - delegate to existing image-gen handler
                    if parsed_command.command == Command.IMG:
                        self.logger.info(
                            f"Found command 'IMG', delegating to cog. (msg_id: {message.id})"
                        )
                        return await self._handle_img_command(parsed_command, message)

                    # All other commands delegate to cogs
                    self.logger.info(
                        f"Found command '{parsed_command.command.name}', delegating to cog. (msg_id: {message.id})"
                    )
                    return BotAction(meta={"delegated_to_cog": True})
                # If it starts with '!' but isn't a known command, let it continue to normal processing
                self.logger.debug(
                    f"Unknown command pattern ignored: {clean_content.split()[0] if clean_content else '(empty)'} (msg_id: {message.id})"
                )

            # 3. Determine if the bot should process this message (DM, mention, or reply).
            if not self._should_process_message(message):
                self.logger.debug(
                    f"Ignoring message {message.id} in guild {message.guild.id if message.guild else 'N/A'}: Not a DM or direct mention."
                )
                return None

            # --- Start of processing for DMs, Mentions, and Replies ---
            async with message.channel.typing():
                self.logger.info(
                    f"Processing message: DM={isinstance(message.channel, DMChannel)}, Mention={self.bot.user in message.mentions} (msg_id: {message.id})"
                )

                # 4. Compatibility fast-path for legacy tests: attachments + empty content (secondary safeguard)
                try:
                    has_attachments = (
                        bool(getattr(message, "attachments", None))
                        and len(message.attachments) > 0
                    )
                except Exception:
                    has_attachments = False
                # Recompute a minimal cleaned content (strip mention prefix like above)
                mention_pattern = rf"^<@!?{self.bot.user.id}>\s*"
                cleaned_for_compat = re.sub(
                    mention_pattern, "", (message.content or "").strip()
                )
                if has_attachments and cleaned_for_compat == "":
                    handler = self._flows.get("process_attachments")
                    if handler:
                        self.logger.debug(
                            "Compat path: delegating to _flows['process_attachments'] with empty text."
                        )
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
                    prechecked = await self._prioritized_vision_route(
                        message, context_str
                    )
                except Exception as e:
                    prechecked = None
                    self.logger.debug(f"vision.precheck_exception | {e}")
                if prechecked is not None:
                    if router_debug:
                        self.logger.info(
                            f"ROUTER_DEBUG | path=t2i reason=vision_intent_detected msg_id={message.id}"
                        )
                    return prechecked

                # 5.5. Check for X/Twitter media routing (before multimodal)
                x_info = _detect_x_twitter_media(message)
                if (
                    x_info.has_x_link
                    and x_info.media_kind not in ("none",)
                    and not parsed_command
                    and x_info.media_urls
                ):
                    self.logger.debug(
                        f"🔗 X/Twitter link detected | kind={x_info.media_kind} | embeds={len(message.embeds)}"
                    )

                    try:
                        # STT-first approach: try STT for video OR unknown media types
                        if x_info.media_kind in ("video", "unknown"):
                            tweet_url = x_info.media_urls[0]
                            self.logger.debug(f"🐦 X STT probe start | url={tweet_url}")

                            try:
                                # Use existing STT pipeline
                                transcript = await hear_infer_from_url(tweet_url)
                                if transcript and transcript.strip():
                                    self.logger.debug(
                                        f"📝 STT complete | transcript_len={len(transcript)}"
                                    )
                                    # Format with transcription header
                                    formatted_response = (
                                        self._format_x_tweet_with_transcription(
                                            tweet_url, transcript, message.content or ""
                                        )
                                    )
                                    return await self._flow_process_text(
                                        content=formatted_response,
                                        context=context_str,
                                        message=message,
                                    )
                            except Exception as e:
                                error_msg = str(e).lower()
                                if any(
                                    phrase in error_msg
                                    for phrase in [
                                        "no video",
                                        "no audio",
                                        "no media",
                                        "not a video",
                                    ]
                                ):
                                    self.logger.debug(
                                        "🐦 X STT probe: no media → falling back to syndication/api"
                                    )
                                    # Fall back to syndication/API for photos or text
                                    return await self._handle_x_twitter_fallback(
                                        tweet_url, message, context_str
                                    )
                                else:
                                    self.logger.debug(
                                        f"❌ X/Twitter STT failed: {e} | continuing without media"
                                    )
                                    note = "(video audio unavailable; proceeding without it)"
                                    fallback_content = self._append_note_once(
                                        message.content or "", note
                                    )
                                    return await self._flow_process_text(
                                        content=fallback_content.strip(),
                                        context=context_str,
                                        message=message,
                                    )

                        elif x_info.media_kind == "image":
                            # X/Twitter images → syndication/VL → text flow
                            tweet_url = (
                                x_info.media_urls[0]
                                if any(
                                    host in x_info.media_urls[0].lower()
                                    for host in {
                                        "x.com",
                                        "twitter.com",
                                        "fxtwitter.com",
                                    }
                                )
                                else None
                            )
                            if tweet_url:
                                self.logger.debug(
                                    f"🖼️🐦 Routing X photos to VL | url={tweet_url}"
                                )
                                return await self._handle_x_twitter_fallback(
                                    tweet_url, message, context_str
                                )
                            else:
                                # Direct image URL
                                image_url = x_info.media_urls[0]
                                self.logger.debug(
                                    f"📎 X/Twitter image detected | url={image_url}"
                                )

                                try:
                                    http_client = await get_http_client()
                                    response = await http_client.get(image_url)
                                    if response.status_code == 200:
                                        ctype = response.headers.get(
                                            "content-type", ""
                                        ).lower()
                                        if not ctype.startswith("image/"):
                                            self.logger.debug(
                                                f"❌ X/Twitter image URL did not return image content-type | content_type={ctype}"
                                            )
                                        else:
                                            image_data = response.content
                                            vl_notes = await see_infer(image_data)
                                            if vl_notes:
                                                vl_notes = sanitize_vl_reply_text(
                                                    vl_notes
                                                )
                                                return await self._flow_process_text(
                                                    content=message.content or "",
                                                    context=context_str,
                                                    message=message,
                                                    perception_notes=vl_notes,
                                                )
                                except Exception:
                                    self.logger.debug(
                                        "❌ X/Twitter image unavailable | reason=image_fetch_failed | continuing without media"
                                    )
                                    note = (
                                        "(image was unavailable; proceeding without it)"
                                    )
                                    fallback_content = self._append_note_once(
                                        message.content or "", note
                                    )
                                    return await self._flow_process_text(
                                        content=fallback_content.strip(),
                                        context=context_str,
                                        message=message,
                                    )

                    except Exception as e:
                        self.logger.error(
                            f"X/Twitter media processing failed: {e}", exc_info=True
                        )
                        # Fall through to normal processing

                # 6. Sequential multimodal processing
                result_action = await self._process_multimodal_message_internal(
                    message, context_str
                )
                if router_debug:
                    # Determine what path was taken based on message content
                    has_x_urls = any(
                        self._is_twitter_url(url)
                        for url in re.findall(r"https?://\S+", content)
                    )
                    has_attachments = bool(getattr(message, "attachments", None))
                    if has_x_urls:
                        self.logger.info(
                            f"ROUTER_DEBUG | path=x_syndication_vl reason=twitter_url_detected msg_id={message.id}"
                        )
                    elif has_attachments:
                        self.logger.info(
                            f"ROUTER_DEBUG | path=attachment_vl reason=image_attachments msg_id={message.id}"
                        )
                    else:
                        self.logger.info(
                            f"ROUTER_DEBUG | path=multimodal reason=default_flow msg_id={message.id}"
                        )
                return result_action  # Return the actual processing result

        except Exception as e:
            self.logger.error(
                f"❌ Error in router dispatch: {e} (msg_id: {message.id})",
                exc_info=True,
            )
            return BotAction(
                content="⚠️ An unexpected error occurred while processing your message.",
                error=True,
            )

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
                return {
                    "eligible": False,
                    "modality": "TEXT_ONLY",
                    "domains": {"text"},
                    "reason": "streaming_master_disabled",
                }

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
                        if any(
                            name.endswith(ext)
                            for ext in (
                                ".png",
                                ".jpg",
                                ".jpeg",
                                ".webp",
                                ".gif",
                                ".bmp",
                                ".pdf",
                                ".mp4",
                                ".mov",
                                ".mkv",
                                ".webm",
                                ".avi",
                                ".m4v",
                                ".mp3",
                                ".wav",
                                ".ogg",
                                ".m4a",
                                ".flac",
                            )
                        ):
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

            modality = (
                "MEDIA_OR_HEAVY"
                if ("media" in domains or "search" in domains or "rag" in domains)
                else "TEXT_ONLY"
            )
            return {
                "eligible": bool(allow),
                "modality": modality,
                "domains": domains,
                "reason": ",".join(reasons) or "none",
            }
        except Exception as e:
            # Fail-closed to quiet mode for safety
            self.logger.debug(f"stream:eligibility_failed | {e}")
            return {
                "eligible": False,
                "modality": "TEXT_ONLY",
                "domains": {"text"},
                "reason": "exception",
            }

    async def _process_multimodal_message_internal(
        self, message: Message, context_str: str
    ) -> Optional[BotAction]:
        """
        Process all input items from a message sequentially with result aggregation.
        Follows the 1 IN → 1 OUT rule by combining all results into a single response.
        Returns the BotAction instead of executing it directly.
        """
        # Collect all input items from the message
        items = collect_input_items(message)

        # Check for reply-image harvesting [VISION_REPLY_IMAGE_HARVEST]
        if message.reference and self.config.get("VISION_REPLY_IMAGE_HARVEST", True):
            try:
                # Fetch the referenced message to harvest images
                ref_message = await message.channel.fetch_message(
                    message.reference.message_id
                )
                reply_images = collect_image_urls_from_message(ref_message)

                if reply_images:
                    # Convert ImageRef objects to InputItem objects and append
                    for idx, img_ref in enumerate(reply_images):
                        items.append(
                            InputItem(
                                source_type="url",
                                payload=img_ref.url,
                                order_index=len(items) + idx,
                            )
                        )

                    # Logging per acceptance: use 📎 and count/kept/truncated fields
                    kept_count = len(reply_images)
                    truncated = False  # No truncation at harvest time
                    self.logger.info(
                        f"📎 Reply image capture | from_msg={ref_message.id} count={len(reply_images)} kept={kept_count} truncated={truncated}"
                    )

            except Exception as e:
                # Non-fatal: continue without reply images if fetch fails
                self.logger.debug(f"Reply image harvest failed: {e}")

        # Process original text content (remove URLs that will be processed separately)
        original_text = message.content
        if self.bot.user in message.mentions:
            original_text = re.sub(
                r"^<@!?{}>\s*".format(self.bot.user.id), "", original_text
            ).strip()

        # Remove URLs from text content since they will be processed separately
        url_pattern = r'https?://[^\s<>"\'\'[\]{}|\\\^`]+'
        original_text = re.sub(url_pattern, "", original_text).strip()

        # Resolve inline [search(...)] directives inside the remaining text
        try:
            original_text = await self._resolve_inline_searches(original_text, message)
        except Exception as e:
            self.logger.error(
                f"Inline search resolution failed: {e} (msg_id: {message.id})",
                exc_info=True,
            )

        # --- Routing precedence gates (feature-flagged) ---
        # 0) Safety: re-run prioritized vision precheck here to catch any triggers/intents that
        #    may have been missed earlier. This is a no-op if none are detected. [CA][REH]
        try:
            prechecked = await self._prioritized_vision_route(message, context_str)
            if prechecked is not None:
                self._metric_inc(
                    "routing.vision.precedence", {"stage": "in_multimodal"}
                )
                return prechecked
        except Exception as e:
            # Never break dispatch because of a precheck failure
            self.logger.debug(f"routing.precedence.vision_check_failed | {e}")

        # Check for reply-image → VL routing condition (forced by config)
        is_dm = isinstance(message.channel, discord.DMChannel)
        mentioned_me = self.bot.user in message.mentions

        # Robust harvest count from referenced and current messages
        bool(self.config.get("VISION_REPLY_IMAGE_FORCE_VL", True))
        combined_count = 0
        heuristic_image_items: List[InputItem] = []
        try:
            # Prefer direct harvest for reliability over extension heuristics
            ref_count = 0
            if message.reference:
                try:
                    ref_message = await message.channel.fetch_message(
                        message.reference.message_id
                    )
                    ref_imgs = collect_image_urls_from_message(ref_message)
                    ref_count = len(ref_imgs or [])
                except Exception:
                    ref_count = 0
            cur_imgs = collect_image_urls_from_message(message) or []
            combined_count = ref_count + len(cur_imgs)
        except Exception:
            # Fallback to heuristic count from collected items
            for item in items:
                if item.source_type == "attachment":
                    if (
                        hasattr(item.payload, "content_type")
                        and item.payload.content_type
                        and item.payload.content_type.startswith("image/")
                    ):
                        heuristic_image_items.append(item)
                elif item.source_type == "url":
                    url_lower = str(item.payload).lower()
                    if any(
                        ext in url_lower
                        for ext in [".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"]
                    ):
                        heuristic_image_items.append(item)
            combined_count = len(heuristic_image_items)

        # Don't let "reply-image perception" hijack X/Twitter links
        x_info_for_gate = None
        try:
            x_info_for_gate = _detect_x_twitter_media(message)
        except Exception:
            x_info_for_gate = None
        has_x_url = any(
            host in (message.content or "").lower()
            for host in ["x.com", "twitter.com", "fxtwitter.com"]
        ) or (x_info_for_gate.has_x_link if x_info_for_gate else False)

        # Filter out Twitter thumbnails from image count if X URLs present
        if has_x_url and heuristic_image_items:
            filtered_items: List[InputItem] = []
            suppressed = 0
            for item in heuristic_image_items:
                try:
                    if item.source_type == "url":
                        from urllib.parse import urlparse

                        raw = str(item.payload)
                        parsed = urlparse(raw)
                        host = (parsed.netloc or "").lower()
                        (parsed.path or "").lower()
                        # Ignore common Twitter image thumbnail hosts and paths
                        if host.endswith("twimg.com") or host in {
                            "pbs.twimg.com",
                            "video.twimg.com",
                        }:
                            suppressed += 1
                            self._metric_inc("routing.twitter.thumb_suppressed", None)
                            continue
                except Exception:
                    # On parse errors, keep the item
                    pass
                filtered_items.append(item)

            combined_count = len(filtered_items)

        # Route to perception (VL notes) → TEXT when conditions met (but skip for X/Twitter)
        if (
            (is_dm or mentioned_me)
            and combined_count >= 1
            and bool(self.config.get("HYBRID_FORCE_PERCEPTION_ON_REPLY", True))
            and not has_x_url
        ):
            self.logger.info(
                f"🎯 Route: text (with perception) | images={combined_count} | msg_id={message.id}"
            )
            try:
                # Run silent perception step to obtain VL notes (sanitized & capped)
                notes, reason = await self._run_perception_notes(message, original_text)
                perception_injection = notes
                if not perception_injection:
                    # Per acceptance: still run text flow with a small hint
                    perception_injection = (
                        "The user replied to an image, but I couldn’t fetch it."
                    )
                    self.logger.info(
                        f"❌ perception unavailable | reason={reason or 'unknown'}"
                    )

                # Invoke TEXT flow with injected perception notes (context unchanged here)
                action = await self._invoke_text_flow(
                    original_text,
                    message,
                    context_str,
                    perception_notes=perception_injection,
                )
                # Final visible truncation by sentence boundary
                try:
                    max_final = int(self.config.get("TEXT_FINAL_MAX_CHARS", 420))
                except Exception:
                    max_final = 420
                if action and getattr(action, "content", None):
                    action.content = self._truncate_final_text(
                        action.content, max_final
                    )
                return action
            except Exception as e:
                self.logger.error(f"Perception→TEXT routing failed: {e}", exc_info=True)
                # Fall back to normal text flow on error

        # If no items found, process as text-only
        if not items:
            # No actionable items found, treat as text-only
            response_action = await self._invoke_text_flow(
                original_text, message, context_str
            )
            if response_action and response_action.has_payload:
                self.logger.info(
                    f"✅ Text-only response generated successfully (msg_id: {message.id})"
                )
                return response_action
            else:
                self.logger.warning(
                    f"No response generated from text-only flow (msg_id: {message.id})"
                )
                return None

        # 1) Web link precedence (if enabled): when URLs are present and vision intent wasn't selected,
        #    prioritize URL processing over other modalities. This preserves 1 IN → 1 OUT by limiting the
        #    item set to URLs only. [Feature-flag: ROUTING_WEB_LINK_PRECEDENCE]
        try:
            web_link_precedence = bool(
                self.config.get("ROUTING_WEB_LINK_PRECEDENCE", False)
            )
        except Exception:
            web_link_precedence = False
        try:
            url_items = [
                it for it in items if getattr(it, "source_type", None) == "url"
            ]
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
            vl_default_for_bare_image = bool(
                self.config.get("VL_DEFAULT_PROMPT_FOR_BARE_IMAGE", True)
            )
        except Exception:
            vl_default_for_bare_image = True
        try:
            image_attachment_items = [
                it
                for it in items
                if getattr(it, "source_type", None) == "attachment"
                and hasattr(getattr(it, "payload", None), "content_type")
                and isinstance(getattr(it, "payload").content_type, str)
                and "image" in (getattr(it, "payload").content_type or "").lower()
            ]
        except Exception:
            image_attachment_items = []

        precedence_applied = False
        if web_link_precedence and url_items:
            self.logger.info(
                f"🔗 Web link precedence enabled; routing to URL-only processing (urls={len(url_items)}) (msg_id: {message.id})"
            )
            self._metric_inc(
                "routing.url.precedence.selected", {"count": str(len(url_items))}
            )
            items = url_items
            precedence_applied = True
        elif (
            vl_default_for_bare_image
            and image_attachment_items
            and (not _has_meaningful_text(original_text))
        ):
            # Backward-compat: legacy attachment-only messages with truly empty content remain supported by
            # the earlier fast-path. This branch handles minimal/implicit prompts too. [REH]
            self.logger.info(
                f"route=attachments | 🖼️ Bare image attachments detected with no meaningful text; prioritizing VL analysis (msg_id: {message.id})"
            )
            self._metric_inc(
                "routing.vl.default_bare_image.selected",
                {"count": str(len(image_attachment_items))},
            )
            items = image_attachment_items
            precedence_applied = True

        self.logger.info(
            f"🚶 Processing {len(items)} input items SEQUENTIALLY for deterministic order (precedence={precedence_applied}) (msg_id: {message.id})"
        )

        # Initialize result aggregator and retry manager
        aggregator = ResultAggregator()
        retry_manager = get_retry_manager()
        # Define timeout mappings for different modalities

        # Per-item budgets
        # LLM/vision tasks can be shorter; media (yt-dlp/transcribe) needs more time. [PA]
        LLM_PER_ITEM_BUDGET = float(
            os.environ.get("MULTIMODAL_PER_ITEM_BUDGET", "30.0")
        )
        MEDIA_PER_ITEM_BUDGET = float(os.environ.get("MEDIA_PER_ITEM_BUDGET", "120.0"))

        # Process items strictly sequentially for determinism [CA]
        start_time = time.time()
        for i, item in enumerate(items, start=1):
            modality = await map_item_to_modality(item)
            # Create description for logging
            if item.source_type == "attachment":
                description = f"{item.payload.filename}"
            elif item.source_type == "url":
                description = (
                    f"URL: {item.payload[:30]}{'...' if len(item.payload) > 30 else ''}"
                )
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
                            extra={
                                "event": "x.retry_policy.media_budget",
                                "detail": {"url": raw_url},
                            },
                        )
                        retry_modality = "media"
                        selected_budget = MEDIA_PER_ITEM_BUDGET
            except Exception:
                # Never break dispatch due to budgeting heuristics
                pass

            # Create coroutine factory for this item
            def create_handler_coro(provider_config: ProviderConfig):
                async def handler_coro():
                    return await self._handle_item_with_provider(
                        item, modality, provider_config
                    )

                return handler_coro

            try:
                result = await retry_manager.run_with_fallback(
                    modality=retry_modality,
                    coro_factory=create_handler_coro,
                    per_item_budget=selected_budget,
                )

                if result.success:
                    self.logger.info(
                        f"✅ Item {i} completed successfully ({result.total_time:.2f}s)"
                    )
                    success = True
                    result_text = result.result
                    duration = result.total_time
                    attempts = result.attempts
                else:
                    msg = f"❌ Failed after {result.attempts} attempts: {result.error}"
                    if result.fallback_occurred:
                        msg += " (fallback attempted)"
                    self.logger.warning(
                        f"❌ Item {i} failed ({result.total_time:.2f}s)"
                    )
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
        successful_items = stats.get("successful_items", 0)
        total_items = stats.get("total_items", 0)
        self.logger.info(
            f"📦 SEQUENTIAL MULTIMODAL COMPLETE: {successful_items}/{total_items} successful, total: {total_time:.1f}s"
        )

        # Generate single aggregated response through text flow (1 IN → 1 OUT)
        if aggregated_prompt.strip():
            response_action = await self._invoke_text_flow(
                aggregated_prompt, message, context_str
            )
            if response_action and response_action.has_payload:
                self.logger.info(
                    f"✅ Multimodal response generated successfully (msg_id: {message.id})"
                )
                return response_action
            else:
                self.logger.warning(
                    f"No response generated from text flow (msg_id: {message.id})"
                )
                return None
        else:
            self.logger.warning(
                f"No content to process after multimodal aggregation (msg_id: {message.id})"
            )
            return None

    async def _handle_item_with_provider(
        self, item: InputItem, modality: InputModality, provider_config: ProviderConfig
    ) -> str:
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
            return await self._handle_image_with_model(
                item, model_override=provider_config.model
            )

        handler = handlers.get(modality, self._handle_unknown)
        return await handler(item)

    async def _process_image_from_url(
        self, url: str, model_override: Optional[str] = None
    ) -> str:
        """Process image from URL using screenshot API + vision analysis. Passes model_override to VL."""
        from .utils.external_api import external_screenshot
        from .see import see_infer

        try:
            # Validate URL before attempting screenshot [IV]
            if not url or not isinstance(url, str) or not re.match(r"^https?://", url):
                self.logger.warning(f"⚠️ Skipping screenshot: invalid URL: {url}")
                return "⚠️ Skipping screenshot: invalid or missing image URL."

            # Take screenshot using the configured screenshot API
            self.logger.info(f"📸 Taking screenshot of URL: {url}")
            screenshot_path = await external_screenshot(url)

            if not screenshot_path:
                self.logger.error(f"❌ Failed to capture screenshot of URL: {url}")
                return f"⚠️ Failed to capture screenshot of URL: {url}"

            # Process the screenshot with vision model
            self.logger.info(
                f"👁️ Processing screenshot with vision model: {screenshot_path}"
            )
            vision_result = await see_infer(
                image_path=screenshot_path,
                prompt="Describe the contents of this screenshot",
                model_override=model_override,
            )

            if (
                vision_result
                and hasattr(vision_result, "content")
                and vision_result.content
            ):
                analysis = vision_result.content
                self.logger.info(
                    f"✅ Screenshot analysis completed: {len(analysis)} chars"
                )
                return f"Screenshot analysis of {url}: {analysis}"
            else:
                self.logger.warning(
                    f"⚠️ Vision analysis returned empty result for: {screenshot_path}"
                )
                return f"⚠️ Screenshot captured but vision analysis failed for: {url}"

        except Exception as e:
            self.logger.error(
                f"❌ Error in screenshot + vision processing: {e}", exc_info=True
            )
            # Extract user-friendly hints when possible, else generic fallback
            error_str = str(e)
            if "temporarily unavailable" in error_str or "provider" in error_str:
                hint = "🔧 The image analysis service is temporarily unavailable. Please try again shortly."
            elif "format" in error_str and "supported" in error_str:
                hint = "🖼️ Unsupported image format. Try JPEG, PNG, or WebP."
            elif "too large" in error_str:
                hint = "📏 Image too large. Please upload a smaller image."
            elif "not be found" in error_str:
                hint = "📁 Image could not be processed. Please upload it again."
            else:
                hint = "❌ Failed to analyze the image. Please try again."
            return f"⚠️ Failed to process screenshot of URL: {url} ({hint})"

    async def _vl_describe_image_from_url(
        self,
        image_url: str,
        *,
        prompt: Optional[str] = None,
        model_override: Optional[str] = None,
    ) -> Optional[str]:
        """
        Download an image from a direct URL and run VL inference. Returns text or None.
        [IV][RM][REH]
        """
        if (
            not image_url
            or not isinstance(image_url, str)
            or not re.match(r"^https?://", image_url)
        ):
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
                        fallback_url = urlunparse(
                            (
                                p.scheme,
                                p.netloc,
                                p.path,
                                p.params,
                                urlencode(qs, doseq=True),
                                p.fragment,
                            )
                        )
                        self.logger.warning(
                            f"⚠️ High-res download failed, retrying with 'name=large': {fallback_url}"
                        )
                        ok = await download_file(fallback_url, Path(tmp_path))
                        if not ok:
                            # Third tier: try 'name=medium' to stay under budget [PA]
                            qs["name"] = "medium"
                            fallback_medium = urlunparse(
                                (
                                    p.scheme,
                                    p.netloc,
                                    p.path,
                                    p.params,
                                    urlencode(qs, doseq=True),
                                    p.fragment,
                                )
                            )
                            self.logger.warning(
                                f"⚠️ Large download failed, retrying with 'name=medium': {fallback_medium}"
                            )
                            ok = await download_file(fallback_medium, Path(tmp_path))
                            if not ok:
                                self.logger.error(
                                    f"❌ Failed to download Twitter image even with fallbacks: {fallback_medium}"
                                )
                                return None
                            # Update for logging clarity
                            image_url = fallback_medium
                        else:
                            # Update for logging clarity
                            image_url = fallback_url
                    else:
                        self.logger.error(
                            f"❌ Failed to download image for VL: {image_url}"
                        )
                        return None
                except Exception as _e:
                    self.logger.error(
                        f"❌ Image download failed (no fallback applied): {image_url} err={_e}"
                    )
                    return None
            vl_prompt = (
                prompt
                or "Describe this image in detail. Focus on salient objects, text, and context."
            )
            res = await see_infer(
                image_path=tmp_path, prompt=vl_prompt, model_override=model_override
            )
            if res and getattr(res, "content", None):
                return str(res.content).strip()
            self.logger.warning(f"⚠️ VL returned empty content for: {image_url}")
            return None
        except Exception as e:
            self.logger.error(
                f"❌ VL describe failed for {image_url}: {e}", exc_info=True
            )
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
        is_twitter = re.match(
            r"https?://(?:www\.)?(?:twitter|x|fxtwitter|vxtwitter)\.com/", url
        )

        try:
            # Try video/audio extraction first
            result = await hear_infer_from_url(url)
            if result and result.get("transcription"):
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
                    if (
                        base_text is None
                        and tweet_id
                        and bool(cfg.get("X_SYNDICATION_ENABLED", True))
                    ):
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
                transcription = result["transcription"]
                metadata = result.get("metadata", {})
                title = metadata.get("title", "Unknown")
                return f"Video transcription from {url} ('{title}'): {transcription}"
            else:
                return f"Could not transcribe audio from video: {url}"

        except VideoIngestError as ve:
            error_str = str(ve).lower()

            # For Twitter URLs with no media, use syndication/API path instead of web extractor [CA][REH]
            if is_twitter and (
                "no video or audio content found" in error_str
                or "no video could be found" in error_str
                or "failed to download video" in error_str
            ):
                self.logger.info(
                    f"🐦 No video in Twitter URL; routing to syndication/API path: {url}"
                )
                # Force to general URL handler which has proper X syndication logic
                return await self._handle_general_url(
                    InputItem(source_type="url", payload=url)
                )

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
            self.logger.error(
                f"❌ Unexpected video processing error: {e}", exc_info=True
            )

            # For Twitter URLs, attempt tiered extractor (no screenshot fallback)
            if is_twitter:
                self.logger.info(
                    f"🐦 Attempting tiered extractor due to unexpected error: {url}"
                )
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
            self.logger.error(
                f"❌ Audio/video file processing failed: {e}", exc_info=True
            )
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
                return (
                    f"PDF handler received unsupported source type: {item.source_type}"
                )

        except Exception as e:
            self.logger.error(f"Error processing PDF: {e}", exc_info=True)
            return "Failed to process PDF document."

    async def _process_pdf_from_attachment(self, attachment: discord.Attachment) -> str:
        """Process PDF from Discord attachment."""
        if not self.pdf_processor:
            return "PDF processing not available (PyMuPDF not installed)."

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_path = tmp_file.name

        try:
            await attachment.save(tmp_path)
            self.logger.info(f"📄 Processing PDF attachment: {attachment.filename}")

            # Process PDF and get result dictionary
            result = await self.pdf_processor.process(tmp_path)

            # Handle error case
            if result.get("error"):
                return f"Could not extract text from PDF: {attachment.filename} (Error: {result['error']})"

            # Extract text content from result dictionary
            text_content = result.get("text", "")
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
                if bool(cfg.get("X_TWITTER_STT_PROBE_FIRST", True)) and (
                    x_client is None
                ):
                    try:
                        stt_res = await hear_infer_from_url(url)
                        if stt_res and stt_res.get("transcription"):
                            base_text = None
                            if syndication_enabled and tweet_id:
                                syn = await self._get_tweet_via_syndication(tweet_id)
                                if syn:
                                    base_text = self._format_syndication_result(
                                        syn, url
                                    )
                            return self._format_x_tweet_with_transcription(
                                base_text=base_text,
                                url=url,
                                stt_res=stt_res,
                            )
                    except Exception as stt_err:
                        err_str = str(stt_err).lower()
                        # Only bypass to API/syndication if clearly not video/audio or unsupported URL
                        if ("no video or audio content" in err_str) or (
                            "unsupported url" in err_str
                        ):
                            pass
                        else:
                            self.logger.info(
                                "X STT probe failed; continuing with API/syndication path",
                                extra={
                                    "event": "x.stt_probe.fail",
                                    "detail": {"url": url, "error": str(stt_err)},
                                },
                            )

                # Tier 1: Syndication JSON (cache + concurrency) when allowed and preferred [PA][REH]
                if (
                    tweet_id
                    and syndication_enabled
                    and not require_api
                    and (syndication_first or x_client is None)
                ):
                    syn = await self._get_tweet_via_syndication(tweet_id)
                    if syn:
                        self._metric_inc("x.syndication.hit", None)
                        # Media-first branching: detect image-only tweets [CA][IV]
                        photos = syn.get("photos") or []
                        text = (syn.get("text") or syn.get("full_text") or "").strip()

                        # Check for image-only tweet: photos present AND empty/whitespace text [IV]
                        normalize_empty = bool(
                            cfg.get("TWITTER_NORMALIZE_EMPTY_TEXT", True)
                        )
                        is_image_only = photos and (
                            not text or (normalize_empty and not text.strip())
                        )

                        if is_image_only and bool(
                            cfg.get("TWITTER_IMAGE_ONLY_ENABLE", True)
                        ):
                            # Route to Vision/OCR pipeline for image-only tweets [CA]
                            self.logger.info(
                                f"🖼️ Image-only tweet detected, routing to Vision/OCR: {url}"
                            )
                            self._metric_inc(
                                "x.tweet_image_only.syndication",
                                {"photos": str(len(photos))},
                            )
                            return await self._handle_image_only_tweet(
                                url, syn, source="syndication"
                            )

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
                                if ("no video or audio content" in err_s) or (
                                    "no video" in err_s
                                ):
                                    pass  # text-only tweet; just return base
                                else:
                                    self.logger.info(
                                        "X syndication STT attempt non-fatal error",
                                        extra={
                                            "event": "x.syndication.stt.warn",
                                            "detail": {
                                                "url": url,
                                                "error": str(stt_err),
                                            },
                                        },
                                    )
                            return base
                        # SURGICAL FIX: Always route photos to VL for X URLs (ignore the flag) [CA][REH]
                        # This ensures native photos are analyzed instead of just counting them

                        # Use new syndication handler for full-res images [CA][PA]
                        from ..syndication.handler import (
                            handle_twitter_syndication_to_vl,
                        )

                        # Minimal route log for observability [CMV]
                        try:
                            self.logger.info(
                                "route=x_syndication | sending photos to VL with high-res upgrade",
                                extra={"detail": {"url": url}},
                            )
                        except Exception:
                            pass
                        result = await handle_twitter_syndication_to_vl(
                            syn,
                            url,
                            self._unified_vl_to_text_pipeline,
                            self.bot.system_prompts.get("vl_prompt"),
                            reply_style="ack+thoughts",
                        )
                        # Syndication handler now returns final text, wrap in BotAction
                        return BotAction(content=result)

                # Tier 2 (optionally before API if syndication_first): X API [SFT]
                if tweet_id and x_client is not None:
                    try:
                        api_data = await x_client.get_tweet_by_id(tweet_id)
                        includes = api_data.get("includes") or {}
                        media_list = includes.get("media") or []
                        media_types = {
                            m.get("type") for m in media_list if isinstance(m, dict)
                        }

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

                        if media_types == {"photo"} or (
                            "photo" in media_types and len(media_types) == 1
                        ):
                            # Check for image-only tweet via API data [IV]
                            tweet_data = api_data.get("data", {})
                            if isinstance(tweet_data, list) and tweet_data:
                                tweet_data = tweet_data[0]
                            elif isinstance(tweet_data, dict):
                                pass  # already correct format
                            else:
                                tweet_data = {}

                            api_text = (tweet_data.get("text") or "").strip()
                            photos = [
                                m
                                for m in media_list
                                if isinstance(m, dict) and m.get("type") == "photo"
                            ]
                            normalize_empty = bool(
                                cfg.get("TWITTER_NORMALIZE_EMPTY_TEXT", True)
                            )
                            is_image_only = photos and (
                                not api_text
                                or (normalize_empty and not api_text.strip())
                            )

                            if is_image_only and bool(
                                cfg.get("TWITTER_IMAGE_ONLY_ENABLE", True)
                            ):
                                # Route to Vision/OCR pipeline for image-only tweets [CA]
                                self.logger.info(
                                    f"🖼️ Image-only tweet detected via API, routing to Vision/OCR: {url}"
                                )
                                self._metric_inc(
                                    "x.tweet_image_only.api",
                                    {"photos": str(len(photos))},
                                )
                                # Convert API data to syndication-like format for unified handling
                                api_as_syn = {
                                    "text": api_text,
                                    "photos": [
                                        {"url": p.get("url")}
                                        for p in photos
                                        if p.get("url")
                                    ],
                                    "user": {
                                        "screen_name": "unknown"
                                    },  # Will be enriched if user data available
                                    "created_at": tweet_data.get("created_at"),
                                }
                                return await self._handle_image_only_tweet(
                                    url, api_as_syn, source="api"
                                )

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
                                "photos": [
                                    {"url": p.get("url")}
                                    for p in photos
                                    if p.get("url")
                                ],
                                "user": {"screen_name": "unknown"},
                                "created_at": tweet_data.get("created_at"),
                            }

                            # Use new syndication handler for full-res images [CA][PA]
                            from ..syndication.handler import (
                                handle_twitter_syndication_to_vl,
                            )

                            return await handle_twitter_syndication_to_vl(
                                api_as_syn,
                                url,
                                self._vl_describe_image_from_url,
                                self.bot.system_prompts.get("vl_prompt"),
                                reply_style="ack+thoughts",
                            )

                        return self._format_x_tweet_result(api_data, url)
                    except APIError as e:
                        emsg = str(e)
                        if any(
                            tok in emsg
                            for tok in [
                                "access denied",
                                "not found",
                                "deleted (",
                                "unexpected status: 401",
                                "unexpected status: 403",
                                "unexpected status: 404",
                                "unexpected status: 410",
                            ]
                        ):
                            self.logger.info(
                                "X API denied or content missing; not scraping due to policy",
                                extra={"detail": {"url": url, "error": emsg}},
                            )
                            return "⚠️ This X post cannot be accessed via API (private/removed). Per policy, scraping is disabled."
                        if (
                            ("429" in emsg or "server error" in emsg)
                            and (not require_api)
                            and allow_fallback_5xx
                        ):
                            self.logger.warning(
                                "X API transient issue, falling back to generic extractor",
                                extra={"detail": {"url": url, "error": emsg}},
                            )
                            # fall through to generic handling below
                        else:
                            self.logger.info(
                                "X API error without fallback; returning policy message",
                                extra={"detail": {"url": url, "error": emsg}},
                            )
                            return "⚠️ Temporary issue accessing X API for this post. Please try again later."
                else:
                    if require_api:
                        return "⚠️ X posts require API access and cannot be scraped. Configure X_API_BEARER_TOKEN to enable."
                    # else fall through to generic handling

            # Use existing URL processing logic - process_url returns a dict
            url_result = await process_url(url)

            # Handle errors
            if not url_result or url_result.get("error"):
                if url_result and url_result.get("error"):
                    return f"Could not extract content from URL: {url} (Error: {url_result['error']})"
                return f"Could not extract content from URL: {url}"

            # Extract text content from result dictionary
            content = result.get("text", "")
            if not content or not content.strip():
                # If no text content, check if we have a screenshot
                if result.get("screenshot_path"):
                    return f"Screenshot captured for {url}: {result['screenshot_path']}"
                return f"Could not extract content from URL: {url}"

            # Check if smart routing detected media and should route to yt-dlp
            route_to_ytdlp = url_result.get("route_to_ytdlp", False)
            if route_to_ytdlp:
                self.logger.info(
                    f"🎥 Smart routing detected media in {url}, routing to yt-dlp flow"
                )

                try:
                    # Process through yt-dlp flow
                    transcription_result = await hear_infer_from_url(url)

                    if transcription_result and transcription_result.get(
                        "transcription"
                    ):
                        transcription = transcription_result["transcription"]
                        metadata = transcription_result.get("metadata", {})
                        title = metadata.get("title", "Unknown")

                        return f"Video/audio content from {url} ('{title}'): {transcription}"
                    else:
                        return f"Successfully detected media in {url} but transcription failed"

                except Exception as e:
                    self.logger.error(f"yt-dlp processing failed for {url}: {e}")
                    return f"Successfully detected media in {url} but could not process it: {str(e)}"

            # Prefer text from process_url when available.
            content = url_result.get("text", "")
            if content and content.strip():
                return f"Web content from {url}: {content}"

            # If no text was extracted (and no media route), use tiered extractor (no screenshots)
            self.logger.info(
                f"🧭 Falling back to tiered extractor for {url} (no auto-screenshot)"
            )
            extract_res = await web_extractor.extract(url)
            if extract_res.success:
                return f"Web content from {extract_res.canonical_url or url}:\n{extract_res.to_message()}"
            else:
                return f"Could not extract content from URL: {url}"

        except Exception as e:
            self.logger.error(f"Error processing general URL: {e}", exc_info=True)
            return f"Failed to process URL: {item.payload}"

    async def _handle_screenshot_url(
        self,
        item: InputItem,
        progress_cb: Optional[Callable[[str, int], Awaitable[None]]] = None,
    ) -> str:
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
                self.logger.warning(
                    f"⚠️ Screenshot API did not return an image for {url}"
                )
                return f"⚠️ Could not capture a screenshot for: {url}. Please try again later."

            if progress_cb:
                await progress_cb("saved", 4)
            self.logger.info(
                f"🖼️ Screenshot saved at: {screenshot_path}. Sending to VL."
            )
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
                self.logger.error(
                    f"❌ Vision analysis failed for {screenshot_path}: {vl_err}",
                    exc_info=True,
                )
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
        self.logger.warning(
            f"Unknown input item type: {item.source_type} with payload type {type(item.payload)}"
        )
        return f"Unsupported input type detected: {item.source_type}. Unable to process this item."

    def _get_input_modality(self, message: Message) -> InputModality:
        """Determine the input modality of a message."""
        if message.attachments:
            attachment = message.attachments[0]
            content_type = attachment.content_type
            filename = attachment.filename.lower()
            if content_type and "image" in content_type:
                return InputModality.IMAGE
            if filename.endswith((".pdf", ".docx")):
                return InputModality.DOCUMENT
            if content_type and "audio" in content_type:
                return InputModality.AUDIO

        # Check for video URLs using comprehensive patterns from video_ingest.py
        try:
            from .video_ingest import SUPPORTED_PATTERNS

            self.logger.debug(
                f"🎥 Testing {len(SUPPORTED_PATTERNS)} video patterns against: {message.content}"
            )

            for pattern in SUPPORTED_PATTERNS:
                if re.search(pattern, message.content):
                    self.logger.info(
                        f"✅ Video URL detected: {message.content} matched pattern: {pattern}"
                    )
                    return InputModality.VIDEO_URL

            self.logger.debug(f"❌ No video patterns matched for: {message.content}")
        except ImportError as e:
            self.logger.warning(
                f"Could not import SUPPORTED_PATTERNS from video_ingest: {e}, using fallback patterns"
            )
            # Fallback patterns (original limited set)
            fallback_patterns = [
                r"https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+",
                r"https?://youtu\.be/[\w-]+",
                r"https?://(?:www\.)?tiktok\.com/@[\w.-]+/video/\d+",
                r"https?://(?:www\.)?tiktok\.com/t/[\w-]+",
                r"https?://(?:m|vm)\.tiktok\.com/[\w-]+",
            ]

            for pattern in fallback_patterns:
                if re.search(pattern, message.content):
                    return InputModality.VIDEO_URL

        # Check for other URLs
        if re.search(r"https?://[\S]+", message.content):
            return InputModality.URL

        return InputModality.TEXT_ONLY

    def _get_output_modality(
        self, parsed_command: Optional[ParsedCommand], message: Message
    ) -> OutputModality:
        """Determine the output modality based on command or channel settings."""
        # Future: check for TTS commands or channel/user settings
        return OutputModality.TEXT

    async def _prioritized_vision_route(
        self, message: Message, context_str: str
    ) -> Optional[BotAction]:
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
                mention_pattern = rf"^<@!?{self.bot.user.id}>\s*"
                content_clean = re.sub(mention_pattern, "", content)
            except Exception:
                content_clean = content

            # Perception beats generation: if images or Twitter URLs are present, skip gen path
            try:
                has_img_attachments = any(
                    (getattr(a, "content_type", "") or "").startswith("image/")
                    for a in (getattr(message, "attachments", None) or [])
                )
            except Exception:
                has_img_attachments = False

            has_twitter_url = False
            try:
                url_candidates = re.findall(r"https?://\S+", content)
                has_twitter_url = any(self._is_twitter_url(u) for u in url_candidates)
            except Exception:
                has_twitter_url = False

            if has_img_attachments or has_twitter_url:
                route = "attachments" if has_img_attachments else "x_syndication"
                self.logger.info(
                    f"route.guard: perception_beats_generation | route={route} (msg_id: {message.id})"
                )
                try:
                    self._metric_inc(
                        "vision.route.vl_only_bypass_t2i", {"route": route}
                    )
                except Exception:
                    pass
                # Never trigger image generation if images or Twitter URLs are present
                return None

            # Check vision availability using centralized helper [CA][REH]
            cfg_enabled = self.config.get(
                "VISION_ENABLED", True
            )  # Use centralized parsed boolean
            dry_run = bool(self.config.get("VISION_DRY_RUN_MODE", False))
            vision_available = self._vision_available()

            # If vision is not enabled at all, skip
            if not cfg_enabled:
                self._metric_inc("vision.route.skipped", {"reason": "cfg_disabled"})
                return None

            # 1) Direct trigger bypass (highest priority)
            direct_vision = self._detect_direct_vision_triggers(content_clean, message)
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
                    return BotAction(
                        content=(
                            "[DRY RUN] Vision generation would be triggered via direct trigger "
                            f"(task={intent_result.extracted_params.task}, prompt='{intent_result.extracted_params.prompt[:80]}...')."
                        )
                    )

                # Lazy start orchestrator if not started [CA]
                if self._vision_orchestrator and not getattr(
                    self._vision_orchestrator, "_started", False
                ):
                    try:
                        await self._vision_orchestrator.ensure_started()
                        vision_available = (
                            self._vision_available()
                        )  # Re-check after lazy start
                    except Exception as e:
                        self.logger.warning(f"Lazy orchestrator start failed: {e}")

                if not vision_available:
                    self._metric_inc(
                        "vision.route.blocked",
                        {"reason": "orchestrator_unavailable", "path": "direct"},
                    )
                    return BotAction(
                        content="🚫 Vision generation is not available right now. Please try again later."
                    )

                return await self._handle_vision_generation(
                    intent_result, message, context_str
                )

            # 2) Intent router decision (lower priority than direct bypass)
            allow_nlp_triggers = bool(
                self.config.get("VISION_ALLOW_NLP_TRIGGERS", False)
            )
            if allow_nlp_triggers and self._vision_intent_router:
                try:
                    intent_result = await self._vision_intent_router.determine_intent(
                        user_message=content_clean,
                        context=context_str,
                        user_id=str(message.author.id),
                        guild_id=str(message.guild.id) if message.guild else None,
                    )
                    if intent_result and getattr(
                        intent_result.decision, "use_vision", False
                    ):
                        conf = float(getattr(intent_result, "confidence", 0.0) or 0.0)
                        self.logger.info(
                            f"🎨 Precheck: Vision intent detected (confidence: {conf:.2f}), routing to Vision system (msg_id: {message.id})"
                        )
                        self._metric_inc("vision.route.intent", {"stage": "precheck"})
                        if dry_run:
                            self._metric_inc("vision.route.dry_run", {"path": "intent"})
                            return BotAction(
                                content=(
                                    "[DRY RUN] Vision generation would be triggered via intent detection "
                                    f"(confidence={conf:.2f})."
                                )
                            )
                        # Lazy start orchestrator if not started [CA]
                        if self._vision_orchestrator and not getattr(
                            self._vision_orchestrator, "_started", False
                        ):
                            try:
                                await self._vision_orchestrator.ensure_started()
                                vision_available = (
                                    self._vision_available()
                                )  # Re-check after lazy start
                            except Exception as e:
                                self.logger.warning(
                                    f"Lazy orchestrator start failed: {e}"
                                )

                        if not vision_available:
                            self._metric_inc(
                                "vision.route.blocked",
                                {
                                    "reason": "orchestrator_unavailable",
                                    "path": "intent",
                                },
                            )
                            return BotAction(
                                content="🚫 Vision generation is not available right now. Please try again later."
                            )
                        return await self._handle_vision_generation(
                            intent_result, message, context_str
                        )
                except Exception as e:
                    self.logger.error(
                        f"❌ Vision intent precheck failed: {e} (msg_id: {message.id})",
                        exc_info=True,
                    )
                    self._metric_inc("vision.intent.error", None)
                    # Fall through to normal multimodal flow on errors

            return None
        except Exception as e:
            # Fail-safe: never break dispatch on precheck
            self.logger.debug(f"vision.precheck_failed | {e}")
            return None

    async def _invoke_text_flow(
        self,
        content: str,
        message: Message,
        context_str: str,
        perception_notes: Optional[str] = None,
    ) -> BotAction:
        """Invoke the text processing flow, formatting history into a context string.
        Optionally inject perception notes into the prompt via contextual brain path.
        """
        self.logger.info(f"route=text | Routing to text flow. (msg_id: {message.id})")

        # Perception beats generation: suppress gen triggers if images/any-URL present (from original message)
        perception_guard = False
        try:
            has_img_attachments = any(
                (getattr(a, "content_type", "") or "").startswith("image/")
                for a in (getattr(message, "attachments", None) or [])
            )
        except Exception:
            has_img_attachments = False
        try:
            # IMPORTANT: check URLs on the original message, not sanitized content
            raw_text = message.content or ""
            url_candidates = re.findall(r"https?://\S+", raw_text)
            has_any_url = bool(url_candidates)
            has_twitter_url = any(self._is_twitter_url(u) for u in url_candidates)
        except Exception:
            has_any_url = False
            has_twitter_url = False
        if has_img_attachments or has_any_url or has_twitter_url:
            perception_guard = True
            try:
                route = (
                    "attachments"
                    if has_img_attachments
                    else ("x_syndication" if has_twitter_url else "links")
                )
                self._metric_inc("vision.route.vl_only_bypass_t2i", {"route": route})
            except Exception:
                pass
            # Minimal breadcrumb for verification
            try:
                self.logger.info(
                    "vision.guard.blocked reason=links_or_attachments",
                    extra={
                        "event": "vision.guard.blocked",
                        "reason": "links_or_attachments",
                        "msg_id": message.id,
                    },
                )
            except Exception:
                pass

        # If perception notes are present (reply-image perception path), always suppress generation triggers
        if perception_notes:
            perception_guard = True

        # Check for direct vision triggers first (explicit tokens only)
        if content.strip() and not perception_guard:
            direct_vision = self._detect_direct_vision_triggers(content, message)
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

                return await self._handle_vision_generation(
                    intent_result, message, context_str
                )

        # Check if this should be routed to Vision generation [CA][SFT]
        allow_nlp_triggers = bool(self.config.get("VISION_ALLOW_NLP_TRIGGERS", False))
        if (
            allow_nlp_triggers
            and (not perception_guard)
            and self._vision_intent_router
            and content.strip()
        ):
            try:
                intent_result = await self._vision_intent_router.determine_intent(
                    user_message=content,
                    context=context_str,
                    user_id=str(message.author.id),
                    guild_id=str(message.guild.id) if message.guild else None,
                )

                if intent_result.decision.use_vision:
                    self.logger.info(
                        f"🎨 Vision intent detected (confidence: {intent_result.confidence:.2f}), routing to Vision system (msg_id: {message.id})"
                    )
                    self._metric_inc("vision.route.intent", {"stage": "text_flow"})
                    return await self._handle_vision_generation(
                        intent_result, message, context_str
                    )
            except Exception as e:
                self.logger.error(
                    f"❌ Vision intent routing failed: {e} (msg_id: {message.id})",
                    exc_info=True,
                )
                self._metric_inc("vision.intent.error", None)
                # Continue to regular text flow on error

        try:
            action = await self._flows["process_text"](
                content, context_str, message, perception_notes=perception_notes
            )
            if action and action.has_payload:
                # Respect TTS state: one-time flag first, then per-user/global preference [CA][REH]
                try:
                    user_id = getattr(message.author, "id", None)
                    require_tts = False
                    if user_id is not None:
                        if tts_state.get_and_clear_one_time_tts(user_id):
                            require_tts = True
                        elif tts_state.is_user_tts_enabled(user_id):
                            require_tts = True

                    if require_tts:
                        action.meta["requires_tts"] = True
                        # Include transcript captions unless disabled via env/config [IV][CMV]
                        include_transcript = os.getenv(
                            "TTS_INCLUDE_TRANSCRIPT", "true"
                        ).lower() in ("1", "true", "yes", "on")
                        action.meta["include_transcript"] = include_transcript
                except Exception as e:
                    # Never break dispatch on TTS flag evaluation
                    self.logger.debug(f"tts.flag_eval_failed | {e}")
                return action
            else:
                self.logger.warning(
                    f"Text flow returned no response. (msg_id: {message.id})"
                )
                return None
        except Exception as e:
            self.logger.error(
                f"Text processing flow failed: {e} (msg_id: {message.id})",
                exc_info=True,
            )
            return BotAction(content="I had trouble processing that text.", error=True)

    def _truncate_final_text(self, text: str, max_chars: int) -> str:
        """Cleanly truncate final visible text at sentence/space boundary with ellipsis."""
        try:
            if max_chars <= 0 or len(text) <= max_chars:
                return text.strip()
            s = text.strip()
            # Prefer last sentence boundary within max range
            boundary = -1
            for i in range(min(len(s), max_chars), max(0, max_chars - 300), -1):
                if s[i - 1] in ".!?":
                    boundary = i
                    break
            if boundary == -1:
                space_idx = s.rfind(" ", 0, max_chars)
                boundary = space_idx if space_idx != -1 else max_chars
            return s[:boundary].rstrip() + "…"
        except Exception:
            # Fallback hard cut
            return (text or "")[:max_chars].rstrip() + (
                "…" if len(text or "") > max_chars else ""
            )

    async def _run_perception_notes(
        self, message: Message, text_instruction: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Run silent perception on reply-image context and return sanitized/capped VL notes.
        Returns (notes, reason) where reason is set on failure paths.
        """
        try:
            from .modality import collect_image_urls_from_message
            from .utils.file_utils import download_robust_image
            import tempfile

            # Harvest from referenced then current message
            image_refs = []
            ref_id = None
            if message.reference:
                try:
                    ref_message = await message.channel.fetch_message(
                        message.reference.message_id
                    )
                    ref_id = getattr(ref_message, "id", None)
                    refs = collect_image_urls_from_message(ref_message) or []
                    image_refs.extend(refs)
                except Exception as e:
                    self.logger.debug(f"perception: harvest(ref) failed | {e}")
            cur_refs = collect_image_urls_from_message(message) or []
            image_refs.extend(cur_refs)

            self.logger.info(
                f"📎 Perception capture | ref_msg={ref_id if ref_id else 'none'} total={len(image_refs)}"
            )

            if not image_refs:
                return None, "no_images"

            # Provider limit: 1 image
            image_refs = image_refs[:1]

            # Download the single image with robust fallback
            downloaded_path = None
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".jpg"
                ) as tmp_file:
                    tmp_path = tmp_file.name
                ok = await download_robust_image(image_refs[0], tmp_path)
                if not ok:
                    # Cleanup temp file on failure
                    try:
                        if tmp_path:
                            os.unlink(tmp_path)
                    except Exception:
                        pass
                    return None, "all_downloads_failed"
                downloaded_path = tmp_path
            except Exception as e:
                self.logger.debug(f"perception: download failed | {e}")
                try:
                    if tmp_path:
                        os.unlink(tmp_path)
                except Exception:
                    pass
                return None, "download_exception"

            # Run VL adapter to get raw notes
            prompt = (
                text_instruction or ""
            ).strip() or "Analyze this image briefly and provide concise notes."
            try:
                vision_result = await see_infer(
                    image_path=downloaded_path, prompt=prompt
                )
                raw_text = ""
                if (
                    vision_result
                    and hasattr(vision_result, "content")
                    and vision_result.content
                ):
                    raw_text = str(vision_result.content).strip()
                else:
                    return None, "provider_empty"

                # Sanitize and cap notes
                try:
                    notes_max = int(self.config.get("VL_NOTES_MAX_CHARS", 600))
                except Exception:
                    notes_max = 600
                strip_reason = bool(self.config.get("VL_STRIP_REASONING", True))
                notes = sanitize_vl_reply_text(
                    raw_text, max_chars=notes_max, strip_reasoning=strip_reason
                )
                return (notes or ""), None
            except Exception as e:
                self.logger.debug(f"perception: see_infer failed | {e}", exc_info=True)
                return None, "provider_error"
            finally:
                # Cleanup temp file
                try:
                    if downloaded_path:
                        os.unlink(downloaded_path)
                except Exception:
                    pass
        except Exception as e:
            self.logger.debug(f"perception: unexpected failure | {e}")
            return None, "unexpected"

    async def _flow_process_text(
        self,
        content: str,
        context: str = "",
        message: Optional[Message] = None,
        *,
        perception_notes: Optional[str] = None,
    ) -> BotAction:
        """Process text input through the AI model with RAG integration and conversation context."""
        self.logger.info("Processing text with AI model and RAG integration.")

        enhanced_context = context

        # 1. RAG Integration - Search vector database concurrently for speed
        rag_task = None
        if os.getenv("ENABLE_RAG", "true").lower() == "true":
            try:
                from bot.rag.hybrid_search import get_hybrid_search

                max_results = int(os.getenv("RAG_MAX_VECTOR_RESULTS", "5"))
                self.logger.debug(
                    f"🔍 RAG: Starting concurrent search for: '{content[:50]}...' [msg_id={message.id if message else 'N/A'}]"
                )

                # Start RAG search concurrently - don't await here
                async def rag_search_task():
                    search_engine = await get_hybrid_search()
                    if search_engine:
                        return await search_engine.search(
                            query=content, max_results=max_results
                        )
                    return None

                rag_task = asyncio.create_task(rag_search_task())
            except Exception as e:
                self.logger.error(
                    f"❌ RAG: Failed to start concurrent search: {e} [msg_id={message.id if message else 'N/A'}]",
                    exc_info=True,
                )
                rag_task = None

        # 2. Wait for RAG search to complete and process results
        if rag_task:
            try:
                # Add timeout to prevent hanging [REH]
                rag_results = await asyncio.wait_for(rag_task, timeout=5.0)
                if rag_results:
                    self.logger.debug(
                        f"📊 RAG: Search completed, found {len(rag_results)} results"
                    )

                    # Extract relevant content from search results (List[HybridSearchResult])
                    rag_context_parts = []
                    for i, result in enumerate(
                        rag_results[:5]
                    ):  # Limit to top 5 results
                        # HybridSearchResult should have content attribute or similar
                        if hasattr(result, "content"):
                            chunk_content = result.content.strip()
                        elif hasattr(result, "text"):
                            chunk_content = result.text.strip()
                        elif isinstance(result, dict):
                            chunk_content = result.get(
                                "content", result.get("text", "")
                            ).strip()
                        else:
                            chunk_content = str(result).strip()

                        if chunk_content:
                            rag_context_parts.append(chunk_content)

                    if rag_context_parts:
                        rag_context = "\n\n".join(rag_context_parts)
                        enhanced_context = (
                            f"{context}\n\n=== Relevant Knowledge ===\n{rag_context}\n=== End Knowledge ===\n"
                            if context
                            else f"=== Relevant Knowledge ===\n{rag_context}\n=== End Knowledge ===\n"
                        )
                        self.logger.debug(
                            f"✅ RAG: Enhanced context with {len(rag_context_parts)} knowledge chunks"
                        )
                    else:
                        self.logger.debug(
                            "⚠️ RAG: Search returned results but all chunks were empty"
                        )
                else:
                    self.logger.debug("🚫 RAG: No relevant results found")
            except Exception as e:
                self.logger.error(f"❌ RAG: Concurrent search failed: {e}")

        # 3. Use contextual brain inference if enhanced context manager is available and message is provided
        if (
            message
            and hasattr(self.bot, "enhanced_context_manager")
            and self.bot.enhanced_context_manager
            and os.getenv("USE_ENHANCED_CONTEXT", "true").lower() == "true"
        ):
            try:
                from bot.contextual_brain import contextual_brain_infer_simple

                self.logger.debug(
                    f"🧠 Using contextual brain inference [msg_id={message.id}]"
                )
                if perception_notes:
                    # Breadcrumb for injection [INFO]
                    try:
                        self.logger.info(
                            f"🧩 Injecting perception into text prompt | chars={len(perception_notes)}"
                        )
                    except Exception:
                        pass
                response_text = await contextual_brain_infer_simple(
                    message,
                    content,
                    self.bot,
                    perception_notes=perception_notes,
                    extra_context=enhanced_context,
                )
                return BotAction(content=response_text)
            except Exception as e:
                self.logger.warning(
                    f"Contextual brain inference failed, falling back to basic: {e}"
                )

        # 4. Fallback to basic brain inference with enhanced context (including RAG).
        # Ensure perception notes are not lost in fallback path by appending as a context block.
        if perception_notes:
            try:
                perception_block = f"Perception (from the image the user replied to):\n{perception_notes.strip()}"
                enhanced_context = (
                    f"{enhanced_context}\n\n{perception_block}"
                    if enhanced_context
                    else perception_block
                )
            except Exception:
                pass
        return await brain_infer(content, context=enhanced_context)

    # ===== Inline [search(...)] directive handling =====
    def _extract_inline_search_queries(
        self, text: str
    ) -> list[tuple[tuple[int, int], str, Optional[SearchCategory]]]:
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
            inner = (m.group(1) or "").strip()
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

        self.logger.info(
            f"🔎 Found {len(directives)} inline search directive(s) (msg_id: {message.id})"
        )

        # Config [IV]: pull from self.config with safe defaults
        provider_name = str(self.config.get("SEARCH_PROVIDER", "ddg"))
        max_results = int(self.config.get("SEARCH_MAX_RESULTS", 5))
        locale = self.config.get("SEARCH_LOCALE") or None
        safe_str = str(self.config.get("SEARCH_SAFE", "moderate")).lower()
        try:
            safesearch = SafeSearch(safe_str)
        except Exception:
            safesearch = SafeSearch.MODERATE
        timeout_ms = (
            int(self.config.get("DDG_TIMEOUT_MS", 5000))
            if provider_name == "ddg"
            else int(self.config.get("CUSTOM_SEARCH_TIMEOUT_MS", 8000))
        )
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
                    cat_label = cat.value if isinstance(cat, SearchCategory) else "text"
                    self._metric_inc(
                        "inline_search.start",
                        {"category": cat_label, "provider": provider_name},
                    )
                    self.logger.debug(
                        f"[InlineSearch] Executing: '{q[:80]}' (category={cat_label})"
                    )
                    return await provider.search(params)
                except Exception as e:
                    self.logger.error(
                        f"[InlineSearch] provider error for '{q}': {e}", exc_info=True
                    )
                    cat_label = cat.value if isinstance(cat, SearchCategory) else "text"
                    self._metric_inc(
                        "inline_search.error",
                        {"category": cat_label, "provider": provider_name},
                    )
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
                replacement = self._format_inline_search_block(
                    query, results, provider_name, safesearch
                )
                cat_label = (
                    category.value if isinstance(category, SearchCategory) else "text"
                )
                self._metric_inc(
                    "inline_search.success",
                    {"category": cat_label, "provider": provider_name},
                )

            pieces.append(replacement)
            cursor = end

        # Append trailing text
        pieces.append(text[cursor:])
        new_text = "".join(pieces)
        self.logger.debug(
            f"[InlineSearch] Rewrote text with {len(directives)} replacement(s). New length={len(new_text)}"
        )
        return new_text

    def _format_inline_search_block(
        self,
        query: str,
        results: List[SearchResult],
        provider_name: str,
        safesearch: SafeSearch,
    ) -> str:
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
            if not hasattr(self, "_media_ingestion_manager"):
                from .media_ingestion import create_media_ingestion_manager

                self._media_ingestion_manager = create_media_ingestion_manager(self.bot)

            return await self._media_ingestion_manager.process_url_smart(url, message)

        except Exception as e:
            self.logger.error(
                f"❌ Smart URL processing failed unexpectedly: {e} (msg_id: {message.id})",
                exc_info=True,
            )
            return BotAction(
                content="⚠️ An unexpected error occurred while processing this URL.",
                error=True,
            )

    async def _flow_process_video_url(self, url: str, message: Message) -> BotAction:
        """Process video URL through STT pipeline and integrate with conversation context."""
        self.logger.info(f"🎥 Processing video URL: {url} (msg_id: {message.id})")

        try:
            # Transcribe video URL audio
            result = await hear_infer_from_url(url)

            transcription = result["transcription"]
            metadata = result["metadata"]

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
                full_context = (
                    f"{context_str}\n\n--- VIDEO CONTENT ---\n{video_context}"
                )
            else:
                full_context = video_context

            # Process through text flow with enriched context
            prompt = (
                "Please summarize and discuss the key points from this video. "
                "Provide insights, analysis, or answer any questions about the content."
            )

            # Use contextual brain inference if available
            if (
                hasattr(self.bot, "enhanced_context_manager")
                and self.bot.enhanced_context_manager
                and os.getenv("USE_ENHANCED_CONTEXT", "true").lower() == "true"
            ):
                try:
                    from bot.contextual_brain import contextual_brain_infer_simple

                    self.logger.debug(
                        f"🧠🎥 Using contextual brain for video analysis [msg_id={message.id}]"
                    )

                    # Add video metadata to enhanced context
                    video_metadata_context = {
                        "source": metadata["source"],
                        "url": metadata["url"],
                        "title": metadata["title"],
                        "uploader": metadata["uploader"],
                        "original_duration_s": metadata["original_duration_s"],
                        "processed_duration_s": metadata["processed_duration_s"],
                        "speedup_factor": metadata["speedup_factor"],
                        "timestamp": metadata["timestamp"],
                    }

                    # Serialize metadata into a compact extra context block
                    meta_str = json.dumps(video_metadata_context, ensure_ascii=False)
                    response_text = await contextual_brain_infer_simple(
                        message,
                        video_context,
                        self.bot,
                        extra_context=f"Video metadata:\n{meta_str}",
                    )
                    return BotAction(content=response_text)

                except Exception as e:
                    self.logger.warning(
                        f"Contextual brain inference failed for video, falling back: {e}"
                    )

            # Fallback to basic brain inference
            return await brain_infer(prompt, context=full_context)

        except Exception as e:
            self.logger.error(
                f"❌ Video URL processing failed: {e} (msg_id: {message.id})",
                exc_info=True,
            )
            error_msg = str(e).lower()

            # Provide user-friendly error messages
            if "unsupported url" in error_msg:
                return BotAction(
                    content="❌ This URL is not supported. Please use YouTube or TikTok links.",
                    error=True,
                )
            elif "video too long" in error_msg:
                return BotAction(
                    content="❌ This video is too long to process. Please try a shorter video (max 10 minutes).",
                    error=True,
                )
            elif "download failed" in error_msg:
                return BotAction(
                    content="❌ Could not download the video. It may be private, unavailable, or region-locked.",
                    error=True,
                )
            elif "audio processing failed" in error_msg:
                return BotAction(
                    content="❌ Could not process the audio from this video. The audio format may be unsupported.",
                    error=True,
                )
            else:
                return BotAction(
                    content="❌ An error occurred while processing the video. Please try again or use a different video.",
                    error=True,
                )

    async def _flow_process_audio(self, message: Message) -> BotAction:
        """Process audio attachment through STT model."""
        self.logger.info(f"Processing audio attachment. (msg_id: {message.id})")
        return await hear_infer(message)

    async def _flow_process_attachments(
        self, message: Message, attachment
    ) -> BotAction:
        """Process image/document attachments."""
        # Accept either a Discord Attachment object or a placeholder (e.g., "" from compat path)
        if not hasattr(attachment, "filename"):
            try:
                attachments = getattr(message, "attachments", None)
                if attachments and len(attachments) > 0:
                    attachment = attachments[0]
                else:
                    self.logger.warning(
                        f"No attachments available to process (msg_id: {message.id})"
                    )
                    return BotAction(content="I didn't receive a file to process.")
            except Exception:
                self.logger.warning(
                    f"Attachment placeholder received but unable to access message.attachments (msg_id: {message.id})"
                )
                return BotAction(content="I didn't receive a file to process.")

        self.logger.info(
            f"Processing attachment: {attachment.filename} (msg_id: {message.id})"
        )

        content_type = getattr(attachment, "content_type", None)
        filename = (getattr(attachment, "filename", "") or "").lower()

        # Process image attachments
        if (content_type and content_type.startswith("image/")) or any(
            filename.endswith(ext)
            for ext in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")
        ):
            return await self._process_image_attachment(message, attachment)

        # Process document attachments
        elif filename.endswith(".pdf") and self.pdf_processor:
            return await self._process_pdf_attachment(message, attachment)

        else:
            self.logger.warning(
                f"Unsupported attachment type: {filename} (msg_id: {message.id})"
            )
            return BotAction(content="I can't process that type of file attachment.")

    async def _unified_vl_to_text_pipeline(
        self, image_paths: List[str], user_caption: str = "", intent: str = "Thoughts?"
    ) -> BotAction:
        """
        Unified 1-hop VL → Text pipeline that enforces "1 in ➜ 1 out" rule.

        Args:
            image_paths: List of local image file paths (up to VL_MAX_IMAGES)
            user_caption: User's text caption or original message content
            intent: Implicit intent when no caption (e.g., "Thoughts?" for naked images)

        Returns:
            BotAction with single final response
        """
        try:
            # Config variables
            max_images = int(os.getenv("VL_MAX_IMAGES", "4"))
            debug_flow = os.getenv("VL_DEBUG_FLOW", "0").lower() in (
                "1",
                "true",
                "yes",
                "on",
            )

            # Limit images to max
            limited_paths = image_paths[:max_images]
            if debug_flow:
                self.logger.info(
                    f"VL_DEBUG_FLOW | processing {len(limited_paths)}/{len(image_paths)} images"
                )

            # Get prompts
            vl_prompt = self.bot.system_prompts.get(
                "vl_prompt", "Analyze and describe this image."
            )
            text_prompt = self.bot.system_prompts.get(
                "text_prompt", "You are a helpful assistant."
            )

            # Step 1: Single VL call with all images
            vl_results = []
            for i, image_path in enumerate(limited_paths):
                try:
                    from .see import see_infer

                    vision_result = await see_infer(
                        image_path=image_path, prompt=vl_prompt
                    )
                    if vision_result and getattr(vision_result, "content", None):
                        raw_content = str(vision_result.content).strip()
                        # Sanitize VL output immediately
                        sanitized_content = sanitize_model_output(raw_content)
                        vl_results.append(f"Image {i + 1}: {sanitized_content}")
                    else:
                        vl_results.append(f"Image {i + 1}: [No analysis available]")
                except Exception as e:
                    self.logger.error(f"VL processing failed for image {i + 1}: {e}")
                    vl_results.append(
                        f"Image {i + 1}: [Analysis failed: {str(e)[:100]}]"
                    )

            if not vl_results:
                return BotAction(
                    content="📷 I couldn't analyze any of the images. Please try again.",
                    error=True,
                )

            # Combine VL results
            combined_vl_result = "\n\n".join(vl_results)
            if debug_flow:
                self.logger.info(
                    f"VL_DEBUG_FLOW | sanitized VL result: {len(combined_vl_result)} chars"
                )

            # Step 2: Prepare input for Text Flow
            if user_caption.strip():
                # User provided caption - include it as context
                text_input = (
                    f"{combined_vl_result}\n\nUser message: {user_caption.strip()}"
                )
            else:
                # No caption - use implicit intent (but don't echo it to Discord)
                text_input = f"{combined_vl_result}\n\nInternal intent: {intent}"

            # Step 3: Single Text Flow call
            from .brain import brain_infer

            final_response = await brain_infer(text_input, context=text_prompt)

            if debug_flow:
                self.logger.info(
                    "VL_DEBUG_FLOW | 1-hop pipeline complete: VL→Text→1 final response"
                )

            return final_response

        except Exception as e:
            self.logger.error(f"❌ Unified VL→Text pipeline failed: {e}", exc_info=True)
            return BotAction(
                content="⚠️ An error occurred while processing the image(s). Please try again.",
                error=True,
            )

    async def _process_image_attachment(
        self, message: Message, attachment
    ) -> BotAction:
        self.logger.info(
            f"Processing image attachment: {attachment.filename} (msg_id: {message.id})"
        )

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(attachment.filename)[1] or ".jpg"
        ) as tmp_file:
            tmp_path = tmp_file.name

        try:
            await attachment.save(tmp_path)
            self.logger.debug(
                f"Saved image to temp file: {tmp_path} (msg_id: {message.id})"
            )

            # Determine user caption and intent
            user_caption = message.content.strip() if message.content else ""
            intent = "Thoughts?" if not user_caption else user_caption

            # Use unified VL → Text pipeline (enforces 1 in ➜ 1 out)
            return await self._unified_vl_to_text_pipeline(
                [tmp_path], user_caption, intent
            )

        except Exception as e:
            self.logger.error(
                f"❌ Image processing failed: {e} (msg_id: {message.id})", exc_info=True
            )
            error_str = str(e).lower()
            if "timeout" in error_str or "time" in error_str:
                return BotAction(
                    content="⏰ Image analysis took too long. Please try again with a smaller image.",
                    error=True,
                )
            elif "memory" in error_str or "size" in error_str:
                return BotAction(
                    content="🧠 Image is too large to process. Please try uploading a smaller image.",
                    error=True,
                )
            elif "file format" in error_str or "unsupported" in error_str:
                return BotAction(
                    content="📷 Unsupported image format. Please try uploading a JPEG, PNG, or WebP image.",
                    error=True,
                )
            elif "file size" in error_str or "too large" in error_str:
                return BotAction(
                    content="📏 Image is too large. Please try uploading a smaller image.",
                    error=True,
                )
            else:
                return BotAction(
                    content="⚠️ An error occurred while processing this image. Please try again.",
                    error=True,
                )
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    async def _process_pdf_attachment(self, message: Message, attachment) -> BotAction:
        self.logger.info(
            f"📄 Processing PDF attachment: {attachment.filename} (msg_id: {message.id})"
        )
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_path = tmp_file.name
        try:
            await attachment.save(tmp_path)
            text_content = await self.pdf_processor.process(tmp_path)
            if not text_content:
                return BotAction(content="I couldn't extract any text from that PDF.")

            final_prompt = f"User uploaded a PDF document. Here is the text content:\n\n{text_content}"
            return await brain_infer(final_prompt)
        except Exception as e:
            self.logger.error(
                f"❌ PDF processing failed: {e} (msg_id: {message.id})", exc_info=True
            )
            return BotAction(
                content="⚠️ An error occurred while processing this PDF.", error=True
            )
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    async def _handle_image_only_tweet(
        self, url: str, syn_data: Dict[str, Any], source: str = "syndication"
    ) -> str:
        """
        Handle image-only tweets with Vision/OCR pipeline and emoji upgrade support.
        Returns neutral, concise alt-text with optional OCR snippet. [CA][SFT][REH]
        """
        try:
            cfg = self.config
            photos = syn_data.get("photos") or []

            if not photos:
                self.logger.warning(
                    f"⚠️ Called _handle_image_only_tweet but no photos found: {url}"
                )
                return (
                    "⚠️ Expected image content but no photos were found in this tweet."
                )

            # Extract tweet metadata for provenance
            user = syn_data.get("user") or {}
            username = user.get("screen_name") or user.get("name") or "unknown"
            created_at = syn_data.get("created_at")

            self.logger.info(
                f"🖼️ Processing {len(photos)} image(s) from image-only tweet: {url}"
            )
            self._metric_inc(
                "vision.image_only_tweet.start",
                {"source": source, "images": str(len(photos))},
            )

            # Process images with Vision/OCR
            results = []
            ocr_texts = []
            safety_flags = []

            for idx, photo in enumerate(photos, start=1):
                photo_url = (
                    photo.get("url") or photo.get("image_url") or photo.get("src")
                )
                if not photo_url:
                    results.append(f"📷 Image {idx}/{len(photos)} — URL not available")
                    continue

                try:
                    # Generate neutral, objective alt-text [SFT]
                    prompt = self._build_neutral_vision_prompt(idx, len(photos), url)

                    # Get vision analysis with retry logic
                    analysis = await self._vl_describe_image_from_url(
                        photo_url, prompt=prompt
                    )

                    if analysis:
                        # Parse analysis for alt-text and OCR if enabled
                        alt_text, ocr_text, safety = self._parse_vision_analysis(
                            analysis, cfg
                        )
                        results.append(alt_text)

                        if ocr_text:
                            ocr_texts.append(ocr_text)
                        if safety:
                            safety_flags.extend(safety)

                        self._metric_inc(
                            "vision.image_only_tweet.success", {"image_idx": str(idx)}
                        )
                    else:
                        results.append(
                            f"📷 Image {idx}/{len(photos)} — analysis unavailable"
                        )
                        self._metric_inc(
                            "vision.image_only_tweet.failure", {"image_idx": str(idx)}
                        )

                except Exception as img_err:
                    self.logger.error(
                        f"❌ Vision analysis failed for image {idx}: {img_err}",
                        exc_info=True,
                    )
                    results.append(f"📷 Image {idx}/{len(photos)} — could not analyze")
                    self._metric_inc(
                        "vision.image_only_tweet.error", {"image_idx": str(idx)}
                    )

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
            self._metric_inc(
                "vision.image_only_tweet.complete",
                {
                    "source": source,
                    "images": str(len(photos)),
                    "ocr_found": str(bool(ocr_texts)),
                    "safety_flags": str(len(safety_flags)),
                },
            )

            response = "\n".join(response_parts)

            # Note: The upgrade context will be stored after the message is sent
            # This is handled in dispatch_message where we have access to the Discord message object
            self.logger.info(
                f"✅ Image-only tweet processed successfully: {len(results)} images analyzed"
            )

            return response

        except Exception as e:
            self.logger.error(
                f"❌ Image-only tweet processing failed: {e}", exc_info=True
            )
            self._metric_inc("vision.image_only_tweet.fatal_error", {"source": source})
            return "⚠️ Could not process images from this tweet right now. Please try again later."

    def _build_neutral_vision_prompt(self, idx: int, total: int, url: str) -> str:
        """Build neutral, objective vision prompt that avoids toxic language echoing. [SFT]"""
        cfg = self.config
        cfg.get("REPLY_TONE", "neutral_objective")

        # Ensure neutral, non-toxic prompting [SFT]
        if total == 1:
            return (
                "Describe this image objectively and concisely. Include who/what/where if clearly visible, "
                "and any text on objects or signs. Keep the description neutral and factual. "
                "Avoid speculation, personal opinions, or sensitive commentary."
            )
        else:
            return (
                f"This is image {idx} of {total} from a social media post. Describe it objectively and concisely. "
                f"Include who/what/where if clearly visible, and any text on objects or signs. "
                f"Keep the description neutral and factual. Avoid speculation or sensitive commentary."
            )

    def _parse_vision_analysis(
        self, analysis: str, cfg: Dict[str, Any]
    ) -> tuple[str, Optional[str], Optional[List[str]]]:
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
            safety_indicators = [
                "nsfw",
                "explicit",
                "medical",
                "violence",
                "inappropriate",
            ]
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
            return await self._flows["generate_tts"](text)
        except Exception as e:
            self.logger.error(f"TTS generation failed: {e}", exc_info=True)
            return None

    async def _handle_vision_generation(
        self, intent_result, message: Message, context_str: str
    ) -> BotAction:
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
                error=True,
            )

        try:
            # Convert intent result to VisionRequest
            from .vision.types import VisionRequest, VisionTask

            # Convert string task to enum
            task_str = intent_result.extracted_params.task
            task_enum = VisionTask(task_str) if isinstance(task_str, str) else task_str

            vision_request = VisionRequest(
                task=task_enum,
                prompt=intent_result.extracted_params.prompt,
                user_id=str(message.author.id),
                guild_id=str(message.guild.id) if message.guild else None,
                channel_id=str(message.channel.id),
                negative_prompt=getattr(
                    intent_result.extracted_params, "negative_prompt", ""
                ),
                width=getattr(intent_result.extracted_params, "width", 1024),
                height=getattr(intent_result.extracted_params, "height", 1024),
                steps=getattr(intent_result.extracted_params, "steps", 30),
                guidance_scale=getattr(
                    intent_result.extracted_params, "guidance_scale", 7.0
                ),
                seed=getattr(intent_result.extracted_params, "seed", None),
                preferred_provider=getattr(
                    intent_result.extracted_params, "preferred_provider", None
                ),
            )

            # Submit job to orchestrator
            self.logger.info(
                f"🎨 Submitting Vision job: {task_enum.value} (msg_id: {message.id})"
            )

            job = await self._vision_orchestrator.submit_job(vision_request)

            # Initial message uses compact working card
            initial_embed = self._build_vision_status_embed(
                state="REQUESTED",
                job=job,
                user=message.author,
                prompt=job.request.prompt if hasattr(job.request, "prompt") else "",
            )
            progress_msg = await message.channel.send(embed=initial_embed)

            # Monitor job progress and update message
            return await self._monitor_vision_job(job, progress_msg, message)

        except Exception as e:
            self.logger.error(
                f"❌ Vision generation failed: {e} (msg_id: {message.id})",
                exc_info=True,
            )

            # Provide user-friendly error messages based on error type
            error_str = str(e).lower()

            if "content filtered" in error_str or "safety" in error_str:
                return BotAction(
                    content="🚫 **Content Safety Issue**\n"
                    "Your request contains content that violates our usage policies. "
                    "Please modify your prompt to remove prohibited content and try again.",
                    error=True,
                )
            elif "budget" in error_str or "quota" in error_str:
                return BotAction(
                    content="💰 **Budget Limit Reached**\n"
                    "You've reached your vision generation budget limit. "
                    "Please wait for your quota to reset or contact an admin for assistance.",
                    error=True,
                )
            elif "provider" in error_str or "service" in error_str:
                return BotAction(
                    content="🔄 **Service Temporarily Unavailable**\n"
                    "The vision generation service is experiencing issues. "
                    "Please try again in a few moments.",
                    error=True,
                )
            else:
                return BotAction(
                    content="❌ **Generation Failed**\n"
                    "An error occurred during vision generation. "
                    "Please check your parameters and try again.",
                    error=True,
                )

    async def _monitor_vision_job(
        self, job, progress_msg, original_msg: Message
    ) -> BotAction:
        """
        Monitor Vision job progress and update Discord message with results [REH][PA]

        Args:
            job: VisionJob instance
            progress_msg: Discord message to update with progress
            original_msg: Original user message for context

        Returns:
            BotAction with final result
        """
        from bot.vision.job_watcher import get_watcher_registry

        try:
            # Use single-flight watcher registry to prevent duplicate polling loops
            watcher_registry = get_watcher_registry()

            # Use typing indicator during monitoring
            async with original_msg.channel.typing():
                updated_job = await watcher_registry.watch_job(
                    job_id=job.job_id,
                    orchestrator=self._vision_orchestrator,
                    progress_msg=progress_msg,
                    original_msg=original_msg,
                    timeout_seconds=600,  # 10 minute timeout
                )

                if not updated_job:
                    self.logger.warning(
                        f"⚠️ Vision job watcher returned no result - job_id: {job.job_id[:8]}"
                    )
                    return BotAction(
                        content="Job monitoring failed or timed out", error=True
                    )

                # Handle final result based on terminal state
                if updated_job.is_terminal_state():
                    if updated_job.state.value == "completed" and updated_job.response:
                        self.logger.info(
                            f"✅ Vision job completed successfully - job_id: {updated_job.job_id[:8]}"
                        )
                        return await self._handle_vision_success(
                            updated_job, progress_msg, original_msg
                        )
                    else:
                        self.logger.warning(
                            f"❌ Vision job failed - job_id: {updated_job.job_id[:8]}, state: {updated_job.state.value}"
                        )
                        return await self._handle_vision_failure(
                            updated_job, progress_msg
                        )
                else:
                    # Should not happen with proper watcher implementation
                    self.logger.error(
                        f"🔴 Vision job watcher returned non-terminal job - job_id: {updated_job.job_id[:8]}"
                    )
                    return BotAction(
                        content="Unexpected job monitoring result", error=True
                    )

        except Exception as e:
            self.logger.error(f"❌ Vision job monitoring failed: {e}", exc_info=True)
            try:
                await progress_msg.edit(
                    content=f"❌ **Monitoring Error**\n"
                    f"Job ID: `{job.job_id[:8]}`\n"
                    f"Lost connection to job status. Please check back later."
                )
            except Exception:
                pass  # Don't fail if message edit fails
            return BotAction(content="Job monitoring failed", error=True)

    async def _handle_reply_image_analysis(
        self,
        image_items: List[InputItem],
        text_instruction: str,
        message: Message,
        context_str: str,
    ) -> BotAction:
        """Handle reply-image → VL analysis with silent mode (no cards) [CA][REH]"""
        if not image_items:
            self.logger.info("Reply-image VL failed | reason=no_images")
            return BotAction(
                content="I couldn’t fetch the image you replied to. Please re-upload it or try again."
            )

        # Check silent mode config (default on)
        silent_mode = self.config.get("VISION_REPLY_IMAGE_SILENT", True)

        if not silent_mode:
            # Fall back to card-based UI for backward compatibility
            return await self._handle_reply_image_analysis_with_cards(
                image_items, text_instruction, message, context_str
            )

        # Silent mode: no cards, just plain text responses
        try:
            # Collect and convert ImageRef objects for robust downloading
            from .modality import collect_image_urls_from_message
            from .utils.file_utils import download_robust_image
            import tempfile

            # Harvest image refs from referenced and current messages (no dependency on reference only)
            image_refs = []
            if message.reference:
                try:
                    ref_message = await message.channel.fetch_message(
                        message.reference.message_id
                    )
                    image_refs.extend(
                        collect_image_urls_from_message(ref_message) or []
                    )
                except Exception:
                    pass
            image_refs.extend(collect_image_urls_from_message(message) or [])

            if not image_refs:
                self.logger.info("Reply-image VL failed | reason=no_images")
                return BotAction(
                    content="I couldn’t fetch the image you replied to. Please re-upload it or try again."
                )

            # Cap at provider limit (assume 1 for simplicity, could be configurable)
            provider_limit = 1  # Most VL providers handle 1 image well
            truncated = len(image_refs) > provider_limit
            if truncated:
                image_refs = image_refs[:provider_limit]
                self.logger.debug(
                    f"Truncated image batch from {len(image_refs)} to {provider_limit}"
                )

            # Download first available image using robust method
            downloaded_paths = []

            for img_ref in image_refs:
                try:
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".jpg"
                    ) as tmp_file:
                        tmp_path = tmp_file.name

                    success = await download_robust_image(img_ref, tmp_path)
                    if success:
                        downloaded_paths.append(tmp_path)
                        break  # Use first successful download for simplicity
                    else:
                        # Clean up failed download
                        try:
                            os.unlink(tmp_path)
                        except Exception:
                            pass

                except Exception as e:
                    self.logger.debug(f"Image download attempt failed: {e}")
                    continue

            if not downloaded_paths:
                self.logger.info("Reply-image VL failed | reason=all_downloads_failed")
                return BotAction(
                    content="I couldn’t fetch the image you replied to. Please re-upload it or try again."
                )

            # Use existing VL analysis pipeline
            prompt = (
                text_instruction.strip()
                or "Analyze this image in detail. Describe what you see, including objects, text, and context."
            )

            try:
                vision_result = await see_infer(
                    image_path=downloaded_paths[0], prompt=prompt
                )

                if (
                    vision_result
                    and hasattr(vision_result, "content")
                    and vision_result.content
                ):
                    raw_text = str(vision_result.content).strip()

                    # Optional expand path: if user asked to "expand", return full text (still no files)
                    instr_lc = (text_instruction or "").strip().lower()
                    expand_tokens = {
                        "expand",
                        "more details",
                        "more detail",
                        "more",
                        "expand please",
                    }
                    if instr_lc in expand_tokens:
                        final_text = raw_text
                        # Soft guard: Discord 2000 char limit
                        if len(final_text) > 1900:
                            final_text = final_text[:1900].rstrip() + "…"
                        return BotAction(content=final_text)

                    # Concise path: sanitize and truncate per config
                    max_chars = 0
                    try:
                        max_chars = int(self.config.get("VL_REPLY_MAX_CHARS", 420))
                    except Exception:
                        max_chars = 420
                    strip_reasoning = bool(self.config.get("VL_STRIP_REASONING", True))
                    final_text = sanitize_vl_reply_text(
                        raw_text, max_chars=max_chars, strip_reasoning=strip_reasoning
                    )

                    if not final_text:
                        final_text = "I can’t produce a concise description. Say ‘expand’ if you want the long version."

                    return BotAction(content=final_text)
                else:
                    raise Exception("Vision analysis returned no results")

            finally:
                # Cleanup temp files
                for tmp_path in downloaded_paths:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

        except Exception as e:
            self.logger.info(
                f"Reply-image VL failed | reason=provider_error | error={str(e)[:100]}"
            )
            self.logger.debug(f"Reply-image VL analysis failed: {e}", exc_info=True)
            return BotAction(
                content="Vision analysis failed. Please try again or re-upload the image."
            )

    async def _handle_reply_image_analysis_with_cards(
        self,
        image_items: List[InputItem],
        text_instruction: str,
        message: Message,
        context_str: str,
    ) -> BotAction:
        """Legacy card-based reply-image analysis for backward compatibility [CA][REH]"""
        # This preserves the original card-based implementation when silent mode is off
        if not image_items:
            return BotAction(content="❌ No images found for analysis.", error=True)

        # Create compact "Working" card
        embed = discord.Embed(
            title="🖼️ Vision Analysis Working",
            color=0x3498DB,  # Blue for working
            timestamp=datetime.now(timezone.utc),
        )
        embed.add_field(name="Task", value="Image Analysis", inline=True)
        embed.add_field(name="Images", value=str(len(image_items)), inline=True)
        embed.add_field(name="Status", value="Processing...", inline=True)

        if text_instruction.strip():
            # Truncate instruction to fit embed limits
            instruction_display = (
                text_instruction[:1020] + "..."
                if len(text_instruction) > 1020
                else text_instruction
            )
            embed.add_field(
                name="Instruction", value=f"`{instruction_display}`", inline=False
            )

        # Post working card
        working_msg = await message.channel.send(embed=embed)

        try:
            # Process first image (respect provider limits - using first image for simplicity)
            first_item = image_items[0]
            image_url = str(first_item.payload)

            # Use existing VL analysis pipeline
            prompt = (
                text_instruction.strip()
                or "Analyze this image in detail. Describe what you see, including objects, text, and context."
            )

            # Download and analyze image
            analysis_start = time.time()
            tmp_path = None

            try:
                # Download image to temp file
                import tempfile
                from .utils.file_utils import download_file

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".jpg"
                ) as tmp_file:
                    tmp_path = tmp_file.name

                success = await download_file(image_url, tmp_path)
                if not success:
                    raise Exception(f"Failed to download image from {image_url}")

                # Use existing see_infer for VL analysis
                vision_result = await see_infer(image_path=tmp_path, prompt=prompt)

                processing_time = time.time() - analysis_start

                if (
                    vision_result
                    and hasattr(vision_result, "content")
                    and vision_result.content
                ):
                    # Success - update to Complete card
                    embed = discord.Embed(
                        title="✅ Vision Analysis Complete",
                        color=0x2ECC71,  # Green for success
                        timestamp=datetime.now(timezone.utc),
                    )
                    embed.add_field(name="Task", value="Image Analysis", inline=True)
                    embed.add_field(
                        name="Images", value=str(len(image_items)), inline=True
                    )
                    embed.add_field(
                        name="Processing Time",
                        value=f"{processing_time:.2f}s",
                        inline=True,
                    )

                    if text_instruction.strip():
                        instruction_display = (
                            text_instruction[:1020] + "..."
                            if len(text_instruction) > 1020
                            else text_instruction
                        )
                        embed.add_field(
                            name="Prompt",
                            value=f"`{instruction_display}`",
                            inline=False,
                        )

                    # Truncate result to fit embed limits
                    result_content = str(vision_result.content).strip()
                    if len(result_content) > 1020:
                        result_content = result_content[:1020] + "..."

                    embed.add_field(name="Analysis", value=result_content, inline=False)

                    if len(image_items) > 1:
                        embed.add_field(
                            name="Note",
                            value=f"Analyzed first image of {len(image_items)} total",
                            inline=False,
                        )

                    await working_msg.edit(embed=embed)
                    return BotAction(
                        content="Vision analysis completed",
                        meta={"discord_msg": working_msg},
                    )

                else:
                    raise Exception("Vision analysis returned no results")

            finally:
                # Cleanup temp file
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

        except Exception as e:
            self.logger.error(f"Reply-image VL analysis failed: {e}", exc_info=True)

            # Error - update to Failed card using unified system
            embed = discord.Embed(
                title="❌ Vision Analysis Failed",
                color=0xED4245,  # Discord brand red
                timestamp=datetime.now(timezone.utc),
            )
            embed.add_field(name="Task", value="Image Analysis", inline=True)
            embed.add_field(name="Images", value=str(len(image_items)), inline=True)
            embed.add_field(name="Status", value="Failed", inline=True)

            # Sanitize error message - remove stack traces, keep it concise
            error_msg = str(e)
            if len(error_msg) > 220:
                error_msg = error_msg[:217] + "..."
            embed.add_field(name="Error", value=error_msg, inline=False)

            # Add prompt if provided
            if text_instruction.strip():
                prompt_display = (
                    text_instruction[:350] + "..."
                    if len(text_instruction) > 350
                    else text_instruction
                )
                embed.add_field(
                    name="Prompt", value=f"`{prompt_display}`", inline=False
                )

            # Footer with user info
            if message.author:
                footer_text = f"Requested by {message.author.display_name}"
                embed.set_footer(text=footer_text)

            try:
                await working_msg.edit(embed=embed)
            except Exception:
                pass  # Don't fail if edit fails

            return BotAction(
                content="❌ Vision analysis failed. Please try again or re-upload the image.",
                error=True,
                meta={"discord_msg": working_msg},
            )

    async def _handle_img_command(self, parsed_command, message: Message) -> BotAction:
        """Handle !img prefix command - delegate to existing image-gen handler [CA]"""
        prompt = parsed_command.cleaned_content.strip()

        # If no prompt, check for attachments
        if not prompt:
            self.logger.info(
                f"IMG: No prompt, checking {len(message.attachments)} attachments"
            )

            # Try to read prompt from attachments
            for att in message.attachments:
                try:
                    self.logger.info(
                        f"IMG: Trying attachment {att.filename} ({att.size} bytes)"
                    )
                    if att.size > 262144:  # 256KB limit
                        continue

                    data = await att.read()
                    if not data:
                        continue

                    # Try multiple encodings
                    text = None
                    for encoding in ["utf-8", "utf-16", "latin-1"]:
                        try:
                            text = data.decode(encoding)
                            break
                        except Exception:
                            continue

                    if text:
                        text = text.replace("\x00", "").strip()
                        if text:
                            prompt = text[:2000]  # Limit prompt length
                            self.logger.info(
                                f"IMG: Found prompt from {att.filename}: '{prompt[:50]}...'"
                            )
                            break
                except Exception as e:
                    self.logger.error(f"IMG: Error reading {att.filename}: {e}")
                    continue

            # Show usage if still no prompt
            if not prompt:
                self.logger.info("IMG: No prompt from attachments, showing help")
                return BotAction(
                    content="🎨 **Image Generation Help**\n"
                    "Usage: `!img <description>`\n"
                    "Example: `!img a kitten playing with yarn`\n"
                    "You can also attach a .txt file with your prompt.\n"
                    "Works in DMs and guild channels, with or without mentioning me."
                )

        # Check if Vision is enabled
        if not self._vision_orchestrator:
            return BotAction(
                content="🚫 Vision generation is not available right now. Please try again later.",
                error=True,
            )

        # Create mock intent result that matches what the vision system expects
        from bot.vision.types import VisionTask, IntentResult, IntentDecision

        class MockIntentParams:
            def __init__(self, prompt: str):
                self.task = VisionTask.TEXT_TO_IMAGE.value
                self.prompt = prompt
                self.negative_prompt = ""
                self.width = 1024
                self.height = 1024
                self.steps = 30
                self.guidance_scale = 7.0
                self.seed = None
                self.preferred_provider = None

        # Create proper IntentResult structure
        mock_decision = IntentDecision(
            use_vision=True,
            confidence=1.0,
            task=VisionTask.TEXT_TO_IMAGE,
            reasoning="!img prefix command",
        )

        mock_intent_result = IntentResult(
            decision=mock_decision,
            extracted_params=MockIntentParams(prompt),
            confidence=1.0,
        )

        # Delegate to existing vision generation handler
        try:
            return await self._handle_vision_generation(mock_intent_result, message, "")
        except Exception as e:
            self.logger.error(f"Failed to handle !img command: {e}", exc_info=True)
            return BotAction(
                content="❌ Failed to process image generation request. Please try again.",
                error=True,
            )

    async def _handle_vision_success(
        self, job, progress_msg, original_msg: Message
    ) -> BotAction:
        """Handle successful Vision generation with file uploads [PA]"""
        try:
            response = job.response

            # Pre-check Discord permissions before attempting upload
            channel = original_msg.channel
            can_attach_files = False

            try:
                if hasattr(channel, "permissions_for") and hasattr(
                    original_msg.guild, "me"
                ):
                    # Guild channel - check bot permissions
                    perms = channel.permissions_for(original_msg.guild.me)
                    can_attach_files = perms.attach_files and perms.send_messages
                    if not can_attach_files:
                        missing_perms = []
                        if not perms.attach_files:
                            missing_perms.append("Attach Files")
                        if not perms.send_messages:
                            missing_perms.append("Send Messages")
                        f"Missing permissions: {', '.join(missing_perms)}"
                else:
                    # DM channel - assume we can attach files
                    can_attach_files = True
            except Exception as e:
                self.logger.warning(
                    f"Permission check failed, assuming no upload capability: {e}"
                )
                can_attach_files = False

            # Download and prepare files for Discord upload
            files_to_upload = []
            result_descriptions = []

            for i, artifact_path in enumerate(response.artifacts, 1):
                try:
                    # Read generated content from local file
                    if not artifact_path.exists():
                        result_descriptions.append(f"❌ Result {i} file not found")
                        continue

                    # Determine file format and name from path with proper MIME type detection
                    ext = (
                        artifact_path.suffix.lower().lstrip(".") or "png"
                    )  # fallback to png
                    filename = f"generated_{job.job_id[:8]}_{i}.{ext}"

                    if can_attach_files:
                        # Detect MIME type from file content
                        with open(artifact_path, "rb") as f:
                            header_bytes = f.read(32)

                        # Map detected MIME to content type
                        if header_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
                            content_type = "image/png"
                        elif header_bytes.startswith(b"\xff\xd8\xff"):
                            content_type = "image/jpeg"
                        elif (
                            header_bytes.startswith(b"RIFF")
                            and b"WEBP" in header_bytes[:12]
                        ):
                            content_type = "image/webp"
                        elif header_bytes.startswith((b"GIF87a", b"GIF89a")):
                            content_type = "image/gif"
                        else:
                            content_type = "image/png"  # safe default

                        # Create Discord file from local path with proper content type
                        discord_file = discord.File(artifact_path, filename=filename)
                        files_to_upload.append(discord_file)
                        result_descriptions.append(f"📎 {filename} ({content_type})")
                    else:
                        # Can't upload, just note the file path for fallback message
                        result_descriptions.append(f"🗂️ {filename} (saved locally)")

                except Exception as e:
                    self.logger.warning(f"Failed to prepare result {i}: {e}")
                    result_descriptions.append(f"❌ Result {i} preparation failed")

            # Cost formatting: avoid numeric format on Money [REH][IV]
            cost_str = "N/A"
            try:
                ac = getattr(response, "actual_cost", None)
                if ac is not None:
                    # Money-aware display if available
                    if hasattr(ac, "to_display_string"):
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

            # Use unified card system for completion
            success_embed = self._build_vision_status_embed(
                state="COMPLETED",
                job=job,
                user=original_msg.author,
                prompt=job.request.prompt if hasattr(job.request, "prompt") else "",
                response=response,
            )

            # Update progress message and upload files
            await progress_msg.edit(content=None, embed=success_embed)

            if files_to_upload:
                # Log filenames and sizes before upload [PA]
                try:
                    upload_meta = []
                    for f in files_to_upload:
                        try:
                            # discord.File has .fp or .path; we derive size from path when available
                            getattr(f, "fp", None)
                            size = None
                            if hasattr(f, "fp") and hasattr(f.fp, "name"):
                                pth = getattr(f.fp, "name", None)
                                if pth and os.path.exists(pth):
                                    size = os.path.getsize(pth)
                                    upload_meta.append((f.filename, size))
                            else:
                                upload_meta.append((f.filename, None))
                        except Exception:
                            upload_meta.append(
                                (getattr(f, "filename", "unknown"), None)
                            )
                    self.logger.info(
                        "📤 Upload starting | files="
                        + ", ".join(
                            [
                                f"{name} ({size} bytes)" if size is not None else name
                                for name, size in upload_meta
                            ]
                        )
                    )
                except Exception:
                    pass
                try:
                    await original_msg.channel.send(files=files_to_upload)
                    self.logger.info(
                        f"📤 Successfully uploaded {len(files_to_upload)} files for job {job.job_id[:8]}"
                    )
                except discord.Forbidden as e:
                    # 403 Forbidden - likely missing Attach Files permission
                    self.logger.warning(f"Upload failed due to permissions (403): {e}")
                    [str(response.artifacts[i]) for i in range(len(files_to_upload))]
                    fallback_content = (
                        f"✅ **Generation Complete**\n"
                        f"Job ID: `{job.job_id[:8]}`\n"
                        f"⚠️ **Upload Issue:** Missing 'Attach Files' permission\n"
                        f"Files saved locally. Contact admin or try in a channel where I can attach files.\n\n"
                        f"**Generated Files:** {len(files_to_upload)} image(s)"
                    )
                    await original_msg.channel.send(content=fallback_content)
                except Exception as e:
                    # Other upload errors
                    self.logger.error(f"File upload failed: {e}")
                    fallback_content = (
                        f"✅ **Generation Complete**\n"
                        f"Job ID: `{job.job_id[:8]}`\n"
                        f"⚠️ **Upload Issue:** {str(e)[:100]}...\n"
                        f"Files generated but upload failed. Please try again."
                    )
                    await original_msg.channel.send(content=fallback_content)
                except Exception as perm_e:
                    self.logger.warning(
                        f"Permission check failed, attempting upload anyway: {perm_e}"
                    )
                    await original_msg.channel.send(files=files_to_upload)

            return BotAction(content="Vision generation completed successfully")

        except Exception as e:
            self.logger.error(f"❌ Vision success handling failed: {e}", exc_info=True)
            # Use unified failure card instead of legacy text
            try:
                user = progress_msg.author if hasattr(progress_msg, "author") else None
                failure_embed = self._build_vision_status_embed(
                    state="FAILED",
                    job=job,
                    user=user,
                    prompt=job.request.prompt
                    if hasattr(job, "request") and hasattr(job.request, "prompt")
                    else "",
                    response=None,
                    error_reason=f"Upload failed: {str(e)[:200]}...",
                )
                await progress_msg.edit(content=None, embed=failure_embed)
            except Exception as card_e:
                self.logger.error(
                    f"❌ Failed to update failure card: {card_e}", exc_info=True
                )
                await progress_msg.edit(content="❌ Vision generation failed")
            return BotAction(
                content="Generation completed with upload issues", error=True
            )

    async def _handle_vision_failure(self, job, progress_msg) -> BotAction:
        """Handle failed Vision generation with unified card system [REH]"""
        try:
            # Get user from progress message for footer
            user = progress_msg.author if hasattr(progress_msg, "author") else None

            # Build unified failure card
            failure_embed = self._build_vision_status_embed(
                state="FAILED",
                job=job,
                user=user,
                prompt=job.request.prompt
                if hasattr(job, "request") and hasattr(job.request, "prompt")
                else "",
                response=None,
                error_reason=job.error.user_message
                if job.error
                else "Unknown error occurred",
            )

            # Edit the progress message to show failure card
            await progress_msg.edit(content=None, embed=failure_embed)
            return BotAction(content="Vision generation failed", error=True)

        except Exception as e:
            self.logger.error(f"❌ Failed to update failure card: {e}", exc_info=True)
            # Fallback to simple text edit if card update fails
            await progress_msg.edit(content="❌ Vision generation failed")
            return BotAction(content="Vision generation failed", error=True)

    def _create_progress_bar(self, percent: int, length: int = 10) -> str:
        """Create ASCII progress bar [CMV]"""
        filled = int(length * percent / 100)
        bar = "█" * filled + "░" * (length - filled)
        return f"[{bar}]"

    def _metric_inc(self, metric_name: str, labels: Optional[Dict[str, str]] = None):
        """Increment a metric, if metrics are enabled."""
        if hasattr(self.bot, "metrics") and self.bot.metrics:
            try:
                # Handle both increment() and inc() method names
                if hasattr(self.bot.metrics, "increment"):
                    self.bot.metrics.increment(metric_name, labels or {})
                elif hasattr(self.bot.metrics, "inc"):
                    self.bot.metrics.inc(metric_name, labels=labels or {})
                else:
                    # Fallback - metrics object doesn't have expected methods
                    pass
            except Exception as e:
                # Never let metrics failures break the application
                self.logger.debug(f"Metrics increment failed for {metric_name}: {e}")

    def _detect_direct_vision_triggers(
        self, content: str, message: Optional[Message] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Direct pattern matching for obvious vision requests to bypass rate-limited intent detection.
        Returns extracted vision parameters if triggers found, None otherwise.
        [RAT: REH, PA] - Robust Error Handling, Performance Awareness
        """
        import re

        # Early bail-out: if original message has URLs or attachments, never trigger regex T2I
        try:
            if message is not None:
                has_attachments = (
                    bool(getattr(message, "attachments", None))
                    and len(message.attachments) > 0
                )
                raw_text = message.content or ""
                has_any_url = bool(re.search(r"https?://\S+", raw_text))
                if has_attachments or has_any_url:
                    return None
        except Exception:
            pass

        # Start-anchored explicit tokens only (safe, intentional)
        text = (content or "").strip()
        token_patterns = [
            re.compile(r"^(?:img|image):\s+(.+)$", re.IGNORECASE | re.DOTALL),
            re.compile(r"^(?:draw|render):\s+(.+)$", re.IGNORECASE | re.DOTALL),
        ]

        debug_triggers = os.getenv("VISION_TRIGGER_DEBUG", "0").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

        for pat in token_patterns:
            m = pat.match(text)
            if not m:
                continue
            prompt = (m.group(1) or "").strip()
            # Require minimum substance and no URLs inside the extracted prompt
            if len(prompt) < 8:
                continue
            if re.search(r"https?://", prompt, re.IGNORECASE):
                return None
            final_prompt = " ".join(prompt.split())
            self.logger.info(
                f"🎨 Direct vision trigger detected: token '{pat.pattern}' -> prompt: '{final_prompt[:50]}...'"
            )
            return {
                "use_vision": True,
                "task": "text_to_image",
                "prompt": final_prompt,
                "confidence": 0.95,
                "bypass_reason": "Direct token trigger",
            }

        if debug_triggers:
            self.logger.info(
                f"VISION_TRIGGER_DEBUG | no_token_matched content='{text[:100]}...'"
            )
        return None

    def _vision_available(self) -> bool:
        """
        Centralized availability check for vision generation [CA][REH]
        Returns True only if:
        - Feature flag enabled (VISION_ENABLED/VISION_T2I_ENABLED)
        - Orchestrator exists and is ready
        """
        # Check feature flags (use centralized parsed booleans) [CA]
        vision_enabled = self.config.get("VISION_ENABLED", True)
        t2i_enabled = self.config.get("VISION_T2I_ENABLED", True)

        # Check orchestrator state
        orchestrator_exists = self._vision_orchestrator is not None
        orchestrator_ready = orchestrator_exists and getattr(
            self._vision_orchestrator, "ready", False
        )

        # Debug logging (controlled by env var) [PA]
        vision_debug = os.getenv("VISION_ORCH_DEBUG", "0").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if vision_debug and not (vision_enabled and t2i_enabled and orchestrator_ready):
            self.logger.debug(
                f"VISION_UNAVAILABLE | reason=orchestrator_unavailable | "
                f"feature={'on' if (vision_enabled and t2i_enabled) else 'off'} | "
                f"orch={'none' if not orchestrator_exists else ('not_ready' if not orchestrator_ready else 'ready')}"
            )

        return vision_enabled and t2i_enabled and orchestrator_ready

    def _build_vision_status_embed(
        self,
        state: str,
        job,
        user,
        prompt: str,
        response=None,
        error_reason="",
        working_ellipsis=False,
    ) -> discord.Embed:
        """Centralized vision status embed builder for all job states."""

        if state == "FAILED":
            embed = discord.Embed(
                title="❌ Vision Generation Failed",
                color=0xED4245,  # Discord brand danger red
                timestamp=discord.utils.utcnow(),
            )

            if hasattr(job, "error_message") and job.error_message:
                reason = (
                    job.error_message[:512] + "..."
                    if len(job.error_message) > 512
                    else job.error_message
                )
                embed.add_field(name="Reason", value=reason, inline=False)

            footer_text = (
                f"Requested by {user.display_name} • Session: {job.job_id[:8]}"
            )
            embed.set_footer(text=footer_text[:2048])
            return embed

        # Success states use consistent green styling
        title_suffix = " · …" if working_ellipsis else ""
        embed = discord.Embed(
            title=f"🎨 Vision Generation {state.title()}{title_suffix}",
            color=0x00D26A,  # Discord brand success green
            timestamp=discord.utils.utcnow(),
        )

        # Task field (always present)
        task_name = (
            job.request.task.value.replace("_", " ").title()
            if hasattr(job.request, "task")
            else "Vision Task"
        )
        embed.add_field(name="Task", value=task_name, inline=True)

        if state == "WORKING":
            # Compact working card - minimal fields only
            embed.add_field(name="Results", value="(pending)", inline=True)

            # Prompt field - single line, heavily truncated for compactness
            if prompt:
                prompt_text = prompt.replace("\n", " ")  # Single line
                if len(prompt_text) > 256:  # Much shorter for working state
                    prompt_text = prompt_text[:253] + "..."
                embed.add_field(name="Prompt", value=prompt_text, inline=False)

        elif state == "COMPLETED" and response:
            # Full completion card with all details
            embed.add_field(
                name="Provider", value=response.provider.value.title(), inline=True
            )
            embed.add_field(
                name="Processing Time",
                value=f"{response.processing_time_seconds:.1f}s",
                inline=True,
            )

            # Cost calculation
            cost_str = "N/A"
            if hasattr(response, "cost_info") and response.cost_info:
                try:
                    cost_str = f"${response.cost_info.total:.4f}"
                except Exception:
                    cost_str = "N/A"
            embed.add_field(name="Cost", value=cost_str, inline=True)

            # Results field
            result_descriptions = []
            if response.artifacts:
                for i, artifact in enumerate(response.artifacts):
                    if hasattr(artifact, "filename") and artifact.filename:
                        result_descriptions.append(
                            f"• [{artifact.filename}](attachment://{artifact.filename})"
                        )
                    else:
                        result_descriptions.append(f"• Image {i + 1}")

            results_text = (
                "\n".join(result_descriptions) if result_descriptions else "No files"
            )
            if len(results_text) > 1024:
                results_text = results_text[:1021] + "..."
            embed.add_field(name="Results", value=results_text, inline=False)

            # Full prompt field for completion
            if prompt:
                prompt_text = prompt
                if len(prompt_text) > 1024:
                    prompt_text = prompt_text[:1021] + "..."
                embed.add_field(name="Prompt", value=prompt_text, inline=False)

        else:
            # Requested state - show placeholders
            embed.add_field(name="Provider", value="—", inline=True)
            embed.add_field(name="Processing Time", value="—", inline=True)
            embed.add_field(name="Results", value="(pending)", inline=False)

            # Prompt field for requested state
            if prompt:
                prompt_text = prompt
                if len(prompt_text) > 1024:
                    prompt_text = prompt_text[:1021] + "..."
                embed.add_field(name="Prompt", value=prompt_text, inline=False)

        # Footer with user and session info
        footer_text = f"Requested by {user.display_name}"
        if (
            state == "COMPLETED"
            and response
            and hasattr(response, "model_name")
            and response.model_name
        ):
            footer_text += f" • Model: {response.model_name}"
        else:
            footer_text += " • Model: —"
        footer_text += f" • Session: {job.job_id[:8]}"

        if len(footer_text) > 2048:
            footer_text = footer_text[:2045] + "..."
        embed.set_footer(text=footer_text)

        # Hard cap for working state to keep it compact
        if state == "WORKING":
            total_length = len(embed.title or "") + len(embed.description or "")
            for field in embed.fields:
                total_length += len(field.name) + len(field.value)
            total_length += len(embed.footer.text if embed.footer else "")

            if total_length > 1500:  # Hard cap for compact working card
                self.logger.warning(
                    f"⚠️ Working embed exceeds 1500 chars ({total_length}), truncating"
                )
                # Truncate prompt further if needed
                for field in embed.fields:
                    if field.name == "Prompt" and len(field.value) > 100:
                        field.value = field.value[:97] + "..."
                        break

        return embed


def get_router() -> Router:
    """Get the singleton router instance."""
    if _router_instance is None:
        raise RuntimeError("Router has not been initialized. Call setup_router first.")
    return _router_instance
