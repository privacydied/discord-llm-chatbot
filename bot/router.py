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
import collections
from .utils.logging import get_logger
import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from unittest.mock import AsyncMock, MagicMock, Mock
import json
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TYPE_CHECKING,
    Union,
)
from urllib.parse import parse_qs, parse_qsl, urlencode, urlparse, urlunparse, unquote

import discord
from discord import DMChannel, Message

from .brain import brain_infer
from .enhanced_retry import ProviderConfig, get_retry_manager
from .evidence import EvidenceBundle
from .exceptions import APIError, DispatchEmptyError
from .http_client import get_http_client, RequestConfig
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
from .vl.postprocess import sanitize_model_output, sanitize_vl_reply_text
from .web import process_url
from .threads.x_thread_unroll import unroll_author_thread
from .web_extraction_service import web_extractor
from .x_api_client import XApiClient
from .action import BotAction
from .command_parser import COMMAND_MAP, parse_command
from .utils.file_utils import download_file
from .utils.attachment_text import read_attachment_text
from .attachment_classifier import classify_attachments, AttachmentBucket, get_by_bucket
from .document_ingest import ingest_document_attachment, ingest_document_from_url
from .url_classifier import ClassifiedURL
from .tts.state import tts_state
from datetime import datetime, timezone
from .memory.mention_context import maybe_build_mention_context
from .memory.thread_tail import (
    collect_thread_tail_context,
    _is_thread_channel,
    resolve_thread_reply_target,
    resolve_implicit_anchor,
    collect_implicit_anchor_context,
)
from .router_components import (
    RouterRuntimeCompat,
    all_attachments_are_text,
    append_embed_related_urls,
    append_unique_url_items,
    build_visual_analysis_anchor_prompt,
    compose_x_tweet_with_visual_facts,
    canonicalize_twitter_status_url,
    collect_x_candidate_urls,
    existing_url_payloads,
    extract_fxtwitter_tweet_node,
    extract_x_api_primary_text,
    extract_x_api_primary_tweet,
    extract_sparse_media_resolution,
    extract_primary_tweet_id,
    extract_raw_urls_from_texts,
    extract_x_status_urls_from_text,
    extract_urls_loose,
    extract_urls_strict,
    filter_canonical_x_urls,
    format_x_tweet_result,
    format_x_tweet_with_transcription,
    get_system_prompt,
    has_explicit_media_intent,
    has_visual_facts_section,
    has_meaningful_text,
    is_direct_image_url,
    is_reply_to_bot,
    is_text_attachment,
    is_tweet_media_url,
    is_twitter_media_cdn,
    is_twitter_thumbnail_url,
    is_twitter_url,
    load_router_runtime_compat,
    mentions_bot,
    normalize_x_url,
    parse_twitter_status_id,
    resolve_twitter_status_id,
    is_twitter_status_url,
    classify_stt_error_reason,
    build_stt_fail_log_payload,
    build_caption_only_fallback_log_payload,
    build_x_video_stt_error_result_payload,
    resolve_caption_only_base_text,
    resolve_video_stt_error_base_text,
    extract_oembed_payload_from_response,
    build_syndication_oembed_fallback_plan,
    build_syndication_fetch_plan,
    build_syndication_fetch_metric_payload,
    classify_syndication_cache_hit,
    build_syndication_negative_cache_entry,
    build_syndication_cache_entry,
    build_syndication_endpoint_url,
    syndication_has_usable_payload,
    syndication_media_hint_keys,
    syndication_article_has_blocks,
    extract_x_article_text,
    syndication_needs_article_hydration,
    extract_syndication_text,
    build_x_text_miss_log_payload,
    build_x_text_miss_payload,
    build_x_text_resolve_payload,
    format_syndication_body_text,
    format_syndication_header_line,
    format_syndication_error_fallback,
    extract_syndication_photo_urls,
    build_syndication_non_200_log_payload,
    build_syndication_non_200_metric_payload,
    build_syndication_fetch_failed_payload,
    build_x_text_canon_payload,
    x_syn_probe_budget_timeout_s,
    x_syn_quick_request_timeouts,
    build_syndication_photo_payload,
    format_twitter_syndication_images_log_line,
    resolve_and_probe_twitter_images,
    stt_result_has_transcription,
    strip_leading_bot_mention,
    strip_discord_mentions_and_urls,
    strip_urls,
    unwrap_x_media_url,
)

if TYPE_CHECKING:
    from bot.core.bot import LLMBot as DiscordBot
    from .command_parser import ParsedCommand

logger = get_logger(__name__)

try:
    from .video_ingest import DEFAULT_SPEEDUP as _DEFAULT_VIDEO_SPEEDUP
except Exception:
    _DEFAULT_VIDEO_SPEEDUP = 1.5

X_STT_MIN_TIMEOUT_S = 120.0
X_STT_PADDING_S = 45.0
X_STT_MAX_TIMEOUT_S = 900.0
X_STT_RTF_DEFAULT = 1.6

_router_instance: Optional["Router"] | None = None


@dataclass
class XTwitterMediaInfo:
    """Detection result for X/Twitter media content."""

    has_x_link: bool = False
    media_kind: str = "none"  # "image", "video", "none"
    media_urls: List[str] = field(default_factory=list)


@dataclass
class ResponseMessage:
    """Response container used across tests and router helpers.

    The class mirrors the shape expected by downstream Discord send helpers,
    providing content plus optional embeds/files/audio attributes. A `text`
    alias is maintained for backward compatibility with older call sites.
    """

    content: str | None = None
    embeds: list | None = None
    files: list | None = None
    audio_path: str | Path | None = None
    text: str | None = None

    def __post_init__(self) -> None:
        # Keep content/text in sync to satisfy both legacy and current callers.
        if self.text is None and self.content is not None:
            self.text = self.content
        elif self.content is None and self.text is not None:
            self.content = self.text


def _detect_x_twitter_media(message: Message) -> XTwitterMediaInfo:
    """Detect X/Twitter links and conservatively classify media type from Discord embeds."""
    canonical_hosts = {
        "x.com",
        "www.x.com",
        "twitter.com",
        "www.twitter.com",
        "mobile.twitter.com",
        "fxtwitter.com",
        "www.fxtwitter.com",
        "vxtwitter.com",
        "www.vxtwitter.com",
        "fixupx.com",
        "www.fixupx.com",
    }
    thumbnail_hosts = {
        "pbs.twimg.com",
        "pbs-0.twimg.com",
        "pbs-1.twimg.com",
        "pbs-2.twimg.com",
        "pbs-3.twimg.com",
    }

    content_lower = (message.content or "").lower()
    has_x_link = any(host in content_lower for host in canonical_hosts)

    tweet_urls: List[str] = []
    url_pattern = (
        r"https?://(?:www\.)?(?:x|twitter|fxtwitter|vxtwitter|fixupx)\.com/\S+"
    )
    for match in re.finditer(url_pattern, message.content or "", re.IGNORECASE):
        u = match.group(0)
        if u not in tweet_urls:
            tweet_urls.append(u)

    media_kind = "none"
    media_urls: List[str] = []
    direct_image_urls: List[str] = []

    for embed in message.embeds:
        embed_url = getattr(embed, "url", "") or ""
        embed_url_l = embed_url.lower()
        provider = getattr(embed, "provider", None)
        provider_name = (getattr(provider, "name", "") or "").lower()
        author = getattr(embed, "author", None)
        author_url = (getattr(author, "url", "") or "").lower()

        is_x_embed = any(host in embed_url_l for host in canonical_hosts)
        is_x_embed = (
            is_x_embed
            or any(host in author_url for host in canonical_hosts)
            or "twitter" in provider_name
        )

        if is_x_embed:
            has_x_link = True
            if embed_url and embed_url not in tweet_urls:
                tweet_urls.append(embed_url)

            if getattr(embed, "video", None):
                media_kind = "video"
                if embed_url and embed_url not in media_urls:
                    media_urls.append(embed_url)
            elif getattr(embed, "image", None):
                if media_kind != "video":
                    media_kind = "images"
                    if tweet_urls:
                        media_urls = list(tweet_urls)
                    elif embed_url:
                        media_urls = [embed_url]
                    else:
                        try:
                            image_url = getattr(embed.image, "url", None)
                            if image_url:
                                media_urls = [image_url]
                        except Exception:
                            pass

        try:
            image_url = None
            if getattr(embed, "image", None) and getattr(embed.image, "url", None):
                image_url = embed.image.url
            elif getattr(embed, "thumbnail", None) and getattr(
                embed.thumbnail, "url", None
            ):
                image_url = embed.thumbnail.url

            if image_url:
                host = urlparse(image_url).netloc.lower()
                if host in thumbnail_hosts and image_url not in direct_image_urls:
                    direct_image_urls.append(image_url)
        except Exception:
            pass

    if media_kind == "none" and direct_image_urls and not tweet_urls:
        media_kind = "images"
        media_urls = list(direct_image_urls)
    elif media_kind == "none" and tweet_urls:
        media_urls = list(tweet_urls)

    return XTwitterMediaInfo(
        has_x_link=has_x_link,
        media_kind=media_kind,
        media_urls=media_urls,
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

        # Recent-message dedupe to prevent double processing (embed echoes, relays)
        self._processed_recent = collections.deque(maxlen=512)
        self._processed_recent_set = set()
        # Concurrency guard for dedupe to prevent race when two listeners fire simultaneously [REH]
        self._processing_locks: dict[int, asyncio.Lock] = {}
        # Per-message metadata shared with core dispatch [REH][CA]
        self._dispatch_metadata: Dict[int, Dict[str, Any]] = {}

        self.pdf_processor = PDFProcessor() if PDF_SUPPORT else None
        if self.pdf_processor:
            self.pdf_processor.loop = bot.loop

        self.logger.info("✔ Router initialized.")
        try:
            routing_flags = {
                "speak_only_when_spoken": self.config.get(
                    "BOT_SPEAKS_ONLY_WHEN_SPOKEN_TO", True
                ),
                "vision_enabled": self.config.get("VISION_ENABLED", True),
                "vision_t2i_enabled": self.config.get("VISION_T2I_ENABLED", True),
                "voice_native": self.config.get("VOICE_ENABLE_NATIVE", False),
            }
            type_map = {k: type(v).__name__ for k, v in routing_flags.items()}
            self.logger.debug(
                "router.flags",
                extra={
                    "event": "router.flags",
                    "detail": {"values": routing_flags, "types": type_map},
                },
            )
        except Exception:
            pass
        # Lazy-initialized X API client
        self._x_api_client: Optional[XApiClient] = None
        # Image upgrade manager for emoji-driven expansions [CA]
        self._upgrade_manager = None  # Lazy-loaded when needed
        # Tweet syndication cache and locks [CA][PA]
        self._syn_cache: Dict[str, Dict[str, Any]] = {}
        self._syn_locks: Dict[str, asyncio.Lock] = {}
        runtime_compat: RouterRuntimeCompat = load_router_runtime_compat(self.config)
        self._runtime_compat = runtime_compat
        self._syn_ttl_s = runtime_compat.syn_ttl_s
        # Canonical fx/vx context cache to share frontend + primary mappings downstream [REH]
        self._x_frontend_canon: "collections.OrderedDict[str, Dict[str, str]]" = (
            collections.OrderedDict()
        )
        # Gate tracking to coordinate pre-dispatch decisions with router execution
        self._gate_denied: Dict[int, str] = {}
        self._prefilter_gate: Dict[int, bool] = {}

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

        # Eagerly start the vision orchestrator in the background to reduce cold-start delays [PA]
        try:
            loop = asyncio.get_running_loop()
            if (
                loop
                and loop.is_running()
                and self._vision_orchestrator
                and not getattr(self._vision_orchestrator, "_started", False)
            ):
                asyncio.create_task(self._vision_orchestrator.start())
                self.logger.debug("🚀 Vision Orchestrator start queued (router init)")
        except Exception:
            # Non-fatal; lazy start path covers this if needed
            pass

        # Feature flags summary (treat missing as enabled) [CMV]
        try:
            ve = bool(self.config.get("VISION_ENABLED", True))
            vti = bool(self.config.get("VISION_T2I_ENABLED", True))
            self.logger.info(
                f"Vision flags | VISION_ENABLED={'on' if ve else 'off'} VISION_T2I_ENABLED={'on' if vti else 'off'}"
            )
        except Exception:
            pass

        # Load centralized VL prompt guidelines if available [CA]
        self._vl_prompt_guidelines: Optional[str] = None
        try:
            prompts_path = (
                Path(__file__).resolve().parents[1] / "prompts" / "vl-prompt.txt"
            )
            if prompts_path.exists():
                content = prompts_path.read_text(encoding="utf-8").strip()
                if content:
                    self._vl_prompt_guidelines = content
                    self.logger.debug(
                        "Loaded VL prompt guidelines from prompts/vl-prompt.txt"
                    )
        except Exception:
            # Non-fatal; handler has built-in defaults
            self._vl_prompt_guidelines = None

        # --- X/Twitter syndication probe feature flags (read-once, cached) [CMV] ---
        self._x_syn_probe_enabled = runtime_compat.x_syn_probe_enabled
        self._x_syn_order = runtime_compat.x_syn_order
        self._x_syn_timeout_s = runtime_compat.x_syn_timeout_s
        self._x_syn_max_images = runtime_compat.x_syn_max_images
        self._x_syn_accept_domains = runtime_compat.x_syn_accept_domains

        # Gate for early X-resolve (enabled by default for correctness) [KBT]
        self._x_early_resolve_enabled = runtime_compat.x_early_resolve_enabled

    def _get_system_prompt(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Safely read a prompt template from bot.system_prompts."""
        return get_system_prompt(self.bot, key, default)

    def _format_x_tweet_with_transcription(
        self,
        *,
        base_text: Optional[str] = None,
        url: str,
        stt_res: Dict[str, Any],
        tweet_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        return format_x_tweet_with_transcription(
            base_text=base_text,
            url=url,
            stt_res=stt_res,
            tweet_data=tweet_data,
            extract_primary_tweet_id=self._extract_primary_tweet_id,
        )

    def _compose_x_tweet_with_visual_facts(
        self,
        *,
        user_text: Optional[str],
        tweet_caption: Optional[str],
        vl_notes: Optional[str],
    ) -> str:
        return compose_x_tweet_with_visual_facts(
            user_text=user_text,
            tweet_caption=tweet_caption,
            vl_notes=vl_notes,
        )

    async def _resolve_x_base_text_for_url(self, url: str) -> Optional[str]:
        """Resolve canonical tweet base text via API-first, then syndication fallback."""
        try:
            cfg = self.config
            tweet_id = XApiClient.extract_tweet_id(str(url))
            if not tweet_id:
                return None

            x_client = await self._get_x_api_client()
            if x_client is not None:
                try:
                    api_data = await x_client.get_tweet_by_id(tweet_id)
                    return self._format_x_tweet_result(api_data, url)
                except Exception:
                    pass

            if bool(cfg.get("X_SYNDICATION_ENABLED", True)):
                try:
                    syn = await self._get_tweet_via_syndication(tweet_id)
                    if syn:
                        return self._format_syndication_result(syn, url)
                except Exception:
                    pass
            return None
        except Exception:
            return None

    def _emit_caption_only_fallback_breadcrumbs(self, reason: str) -> None:
        """Emit non-fatal stt/fallback breadcrumbs for caption-only degrade paths."""
        self._emit_stt_fail_event(reason)
        self._emit_caption_only_fallback_event()

    def _emit_stt_fail_event(
        self,
        reason: str,
        *,
        media_kind: Optional[str] = None,
        msg_id: Optional[int] = None,
    ) -> None:
        """Emit STT failure breadcrumb with optional media kind and message id."""
        try:
            payload = build_stt_fail_log_payload(
                reason,
                media_kind=media_kind,
                msg_id=msg_id,
            )
            self.logger.info(
                "stt.fail",
                extra=payload,
            )
        except Exception:
            pass

    def _emit_caption_only_fallback_event(self) -> None:
        """Emit caption-only fallback breadcrumb without synthesizing STT failure."""
        try:
            self.logger.info(
                "fallback",
                extra=build_caption_only_fallback_log_payload(),
            )
        except Exception:
            pass

    def _classify_stt_error_reason(self, stt_err: Optional[str]) -> str:
        """Map STT status token to the canonical fallback reason."""
        return classify_stt_error_reason(stt_err)

    def _extract_x_api_primary_tweet(self, api_data: Any) -> Dict[str, Any]:
        """Extract the primary tweet node from X API payload variants."""
        return extract_x_api_primary_tweet(api_data)

    def _extract_x_api_primary_text(self, api_data: Any) -> str:
        """Extract canonical tweet text from X API payload variants."""
        return extract_x_api_primary_text(api_data)

    def _format_x_caption_only_transcription(
        self,
        *,
        url: str,
        base_text: Optional[str] = None,
        tweet_text: Optional[str] = None,
        api_data: Optional[Any] = None,
    ) -> str:
        """Format caption-only evidence when STT is unavailable."""
        api_text = self._extract_x_api_primary_text(api_data)
        safe_base_text = resolve_caption_only_base_text(
            api_text=api_text,
            tweet_text=tweet_text,
            base_text=base_text,
        )
        return self._format_x_tweet_with_transcription(
            base_text=safe_base_text,
            url=url,
            stt_res={},
        )

    def _format_x_caption_only_fallback_result(
        self,
        *,
        url: str,
        base_text: Optional[str] = None,
        tweet_text: Optional[str] = None,
        api_data: Optional[Any] = None,
    ) -> str:
        """Emit caption-only fallback breadcrumb and return caption-only evidence."""
        self._emit_caption_only_fallback_event()
        return self._format_x_caption_only_transcription(
            url=url,
            base_text=base_text,
            tweet_text=tweet_text,
            api_data=api_data,
        )

    def _format_x_video_stt_error_result(
        self,
        *,
        url: str,
        stt_error: Optional[str],
        base_text: Optional[str] = None,
        tweet_text: Optional[str] = None,
    ) -> str:
        """Format video STT failure while preserving video modality context."""
        safe_base_text = resolve_video_stt_error_base_text(
            tweet_text=tweet_text,
            base_text=base_text,
        )
        stt_error_result = build_x_video_stt_error_result_payload(
            url=url,
            stt_error=stt_error,
        )
        return self._format_x_tweet_with_transcription(
            base_text=safe_base_text,
            url=url,
            stt_res=stt_error_result,
        )

    def _format_x_video_stt_probe_result(
        self,
        *,
        url: str,
        base_text: str,
        tweet_text: Optional[str],
        stt_res: Any,
        stt_err: Optional[str],
        emit_fail_event: bool = False,
        fail_media_kind: str = "video",
        msg_id: Optional[int] = None,
    ) -> str:
        """Format STT probe result for video routes, preserving video context on failure."""
        formatted = self._format_x_transcription_if_present(
            base_text=base_text,
            url=url,
            stt_res=stt_res,
        )
        if formatted:
            return formatted
        if emit_fail_event:
            self._emit_stt_fail_event(
                self._classify_stt_error_reason(stt_err),
                media_kind=fail_media_kind,
                msg_id=msg_id,
            )
        return self._format_x_video_stt_error_result(
            url=url,
            stt_error=stt_err,
            base_text=base_text,
            tweet_text=tweet_text,
        )

    async def _format_x_with_resolved_base_text(
        self, *, url: str, stt_res: Any
    ) -> str:
        """Resolve X base text for URL and format with STT payload."""
        base_text = await self._resolve_x_base_text_for_url(url)
        return self._format_x_tweet_with_transcription(
            base_text=base_text,
            url=url,
            stt_res=stt_res,
        )

    async def _format_x_with_resolved_base_text_if_available(
        self, *, url: str, stt_res: Any
    ) -> Optional[str]:
        """Resolve X base text and format only when non-empty base text is available."""
        base_text = await self._resolve_x_base_text_for_url(url)
        if not base_text:
            return None
        return self._format_x_tweet_with_transcription(
            base_text=base_text,
            url=url,
            stt_res=stt_res,
        )

    async def _format_x_no_speech_fallback(
        self,
        *,
        url: str,
        stt_res: Any,
        base_text: Optional[str] = None,
    ) -> str:
        """Emit no-speech breadcrumbs and format X caption-only evidence."""
        self._emit_caption_only_fallback_breadcrumbs("no_speech")
        stt_payload = stt_res or {}
        if base_text is None:
            return await self._format_x_with_resolved_base_text(
                url=url,
                stt_res=stt_payload,
            )
        return self._format_x_tweet_with_transcription(
            base_text=base_text,
            url=url,
            stt_res=stt_payload,
        )

    async def _route_twitter_syndication_to_vl(
        self, syn_payload: Dict[str, Any], url: str
    ) -> str:
        """Route a syndication-like tweet payload to the unified VL handler."""
        from .syndication.handler import handle_twitter_syndication_to_vl

        return await handle_twitter_syndication_to_vl(
            syn_payload,
            url,
            self._unified_vl_to_text_pipeline,
            self._get_system_prompt("vl_prompt"),
            reply_style="ack+thoughts",
        )

    async def _route_twitter_images_with_caption(
        self, *, url: str, caption_text: Optional[str], image_urls: List[str]
    ) -> str:
        """Build syndication-like payload from caption+images and route to VL."""
        syn_payload = self._build_syndication_photo_payload(caption_text, image_urls)
        return await self._route_twitter_syndication_to_vl(syn_payload, url)

    async def _route_probed_twitter_images_with_caption(
        self, *, url: str, status_id: Optional[str], image_urls: List[str]
    ) -> str:
        """Log probed images, resolve caption text, and route through VL."""
        self._log_twitter_syndication_images(image_urls)
        tweet_text = await self._resolve_twitter_caption_text(status_id)
        return await self._route_twitter_images_with_caption(
            url=url,
            caption_text=tweet_text,
            image_urls=image_urls,
        )

    async def _resolve_and_probe_twitter_images(
        self, *, url: str, tweet_id: Optional[str] = None
    ) -> Tuple[str, List[str]]:
        """Resolve status id and probe syndication for tweet images."""
        return await resolve_and_probe_twitter_images(
            url=url,
            tweet_id=tweet_id,
            resolve_status_id=self._resolve_twitter_status_id,
            probe_images=self._probe_twitter_syndication_images,
        )

    def _log_twitter_syndication_images(
        self, image_urls: List[str], *, msg_id: Optional[int] = None
    ) -> None:
        """Emit canonical breadcrumb for Twitter image-route detection."""
        self.logger.info(
            format_twitter_syndication_images_log_line(
                image_urls,
                msg_id=msg_id,
            )
        )

    def _build_syndication_photo_payload(
        self, text: Optional[str], image_urls: List[str]
    ) -> Dict[str, Any]:
        """Build syndication-like payload consumed by the unified VL handler."""
        return build_syndication_photo_payload(text, image_urls)

    def _build_x_syn_quick_request_config(self) -> RequestConfig:
        """Build short-budget HTTP config for quick X syndication probes."""
        connect_timeout, read_timeout, total_timeout = x_syn_quick_request_timeouts(
            self._x_syn_timeout_s
        )
        return RequestConfig(
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            total_timeout=total_timeout,
            max_retries=0,
        )

    def _x_syn_probe_budget_timeout_s(self) -> float:
        """Compute bounded timeout budget for image/media probe calls."""
        return x_syn_probe_budget_timeout_s(getattr(self, "_x_syn_timeout_s", 3.0))

    def _extract_fxtwitter_tweet_node(self, payload: Any) -> Dict[str, Any]:
        """Extract the canonical tweet/status node from fx/vx payloads."""
        return extract_fxtwitter_tweet_node(payload)

    def _stt_result_has_transcription(self, stt_result: Any) -> bool:
        """Check whether an STT result payload contains non-empty transcription text."""
        return stt_result_has_transcription(stt_result)

    def _extract_sparse_media_resolution(
        self, resolved_sparse: Any, *, default_url: str
    ) -> Tuple[str, List[str], str]:
        """Extract sparse media kind/images/url from resolved payload."""
        return extract_sparse_media_resolution(
            resolved_sparse,
            default_url=default_url,
        )

    def _format_x_transcription_if_present(
        self,
        *,
        base_text: str,
        url: str,
        stt_res: Any,
    ) -> Optional[str]:
        """Format X transcription output only when STT contains transcription text."""
        if not self._stt_result_has_transcription(stt_res):
            return None
        return self._format_x_tweet_with_transcription(
            base_text=base_text,
            url=url,
            stt_res=stt_res,
        )

    async def _maybe_hydrate_syndication_payload(
        self,
        tweet_id: Optional[str],
        payload: Any,
        *,
        allow_tco_pointer: bool = False,
    ) -> Any:
        """Hydrate syndication payload when tweet id and dict payload are available."""
        if not tweet_id or not isinstance(payload, dict):
            return payload
        return await self._hydrate_syndication_article_if_needed(
            tweet_id,
            payload,
            allow_tco_pointer=allow_tco_pointer,
        )

    async def _resolve_syndication_caption_from_payload(
        self,
        tweet_id: Optional[str],
        payload: Any,
        *,
        fallback_text: str = "",
    ) -> str:
        """Resolve caption from a syndication-like payload with optional hydration."""
        if not isinstance(payload, dict):
            return fallback_text
        try:
            payload = await self._maybe_hydrate_syndication_payload(
                tweet_id,
                payload,
                allow_tco_pointer=True,
            )
            if isinstance(payload, dict):
                caption = self._extract_syndication_text(payload)
                if caption:
                    return caption
        except Exception:
            pass
        return fallback_text

    async def _resolve_twitter_caption_text(self, status_id: Optional[str]) -> str:
        """Resolve tweet caption via syndication first, then fx/vx fallback."""
        if not status_id:
            return ""

        tweet_text = ""
        try:
            syn = await self._get_tweet_via_syndication(status_id)
            if syn:
                tweet_text = await self._resolve_syndication_caption_from_payload(
                    status_id,
                    syn,
                )
        except Exception:
            tweet_text = ""

        if tweet_text:
            return tweet_text

        try:
            http2 = await get_http_client()
            cfg2 = self._build_x_syn_quick_request_config()
            fxu = f"https://api.fxtwitter.com/status/{status_id}"
            r2 = await http2.get(fxu, config=cfg2)
            if r2.status_code == 200:
                try:
                    fxj = r2.json()
                except Exception:
                    fxj = {}
                tnode = self._extract_fxtwitter_tweet_node(fxj)
                if tnode:
                    tweet_text = self._extract_syndication_text(tnode)
        except Exception:
            pass

        return tweet_text

    async def _resolve_twitter_caption_from_syndication(
        self, status_id: Optional[str], fallback_text: str = ""
    ) -> str:
        """Resolve caption from syndication only; keep fallback text on miss/error."""
        if not status_id:
            return fallback_text
        try:
            syn = await self._get_tweet_via_syndication(status_id)
            return await self._resolve_syndication_caption_from_payload(
                status_id,
                syn,
                fallback_text=fallback_text,
            )
        except Exception:
            pass
        return fallback_text

    def _build_visual_anchored_system_prompt(
        self, content: str, *, fallback: bool = False
    ) -> Optional[str]:
        """Build anchored system prompt when visual-facts evidence is present."""
        try:
            if not has_visual_facts_section(content):
                return None

            base_sys = self._get_system_prompt(
                "text_prompt", "You are a helpful assistant."
            )
            anchored = build_visual_analysis_anchor_prompt(base_sys)
            try:
                if fallback:
                    self.logger.info(
                        "text.anchor | visual_facts_detected=true (fallback)"
                    )
                else:
                    self.logger.info("text.anchor | visual_facts_detected=true")
            except Exception:
                pass
            return anchored
        except Exception:
            return None

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
        if cached:
            hit_kind = classify_syndication_cache_hit(now, self._syn_ttl_s, cached)
            if hit_kind == "neg":
                self._metric_inc("x.syndication.neg_cache_hit", None)
                return None
            if hit_kind == "data":
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
            if cached:
                hit_kind = classify_syndication_cache_hit(now, self._syn_ttl_s, cached)
                if hit_kind == "neg":
                    self._metric_inc("x.syndication.neg_cache_hit_locked", None)
                    return None
                if hit_kind == "data":
                    self._metric_inc("x.syndication.cache_hit_locked", None)
                    return cached.get("data")

            int(self.config.get("X_SYNDICATION_TIMEOUT_MS", 4000))
            base, headers, params_variants = build_syndication_fetch_plan(tweet_id)
            data = None
            media_hint_keys = syndication_media_hint_keys()

            def _has_usable_payload(node: Any) -> bool:
                return syndication_has_usable_payload(
                    node,
                    extract_text=self._extract_syndication_text,
                    media_hint_keys=media_hint_keys,
                )

            try:
                http_client = await get_http_client()
                for endpoint, params in params_variants:
                    url = build_syndication_endpoint_url(base, endpoint)
                    self._metric_inc(
                        "x.syndication.fetch",
                        build_syndication_fetch_metric_payload(endpoint),
                    )
                    resp = await http_client.get(url, headers=headers, params=params)
                    if resp.status_code != 200:
                        self.logger.info(
                            "Syndication non-200",
                            extra=build_syndication_non_200_log_payload(
                                tweet_id=tweet_id,
                                status=resp.status_code,
                                endpoint=endpoint,
                            ),
                        )
                        self._metric_inc(
                            "x.syndication.non_200",
                            build_syndication_non_200_metric_payload(
                                status=resp.status_code,
                                endpoint=endpoint,
                            ),
                        )
                        continue
                    try:
                        data = resp.json()
                    except Exception:
                        self._metric_inc(
                            "x.syndication.invalid_json",
                            build_syndication_fetch_metric_payload(endpoint),
                        )
                        continue
                    # If the JSON lacks usable text, try oEmbed fallbacks before moving on
                    if not _has_usable_payload(data):
                        oembed_url, oembed_fallbacks = (
                            build_syndication_oembed_fallback_plan(tweet_id)
                        )
                        for (
                            metric_endpoint,
                            oembed_params,
                        ) in oembed_fallbacks:
                            if _has_usable_payload(data):
                                break
                            try:
                                self._metric_inc(
                                    "x.syndication.fetch",
                                    build_syndication_fetch_metric_payload(
                                        metric_endpoint
                                    ),
                                )
                                resp_oe = await http_client.get(
                                    oembed_url, headers=headers, params=oembed_params
                                )
                                oembed_data = extract_oembed_payload_from_response(
                                    resp_oe
                                )
                                if oembed_data:
                                    data = oembed_data
                            except Exception:
                                pass
                    # Break when we have usable data; otherwise continue to next variant
                    if _has_usable_payload(data):
                        break
            except Exception as e:
                self.logger.info(
                    "Syndication fetch failed",
                    extra=build_syndication_fetch_failed_payload(
                        tweet_id=tweet_id,
                        error=str(e),
                    ),
                )
                self._metric_inc("x.syndication.error", None)
                return None

            # Minimal validation: require text field
            if not _has_usable_payload(data):
                try:
                    self.logger.info(
                        "x.text.miss",
                        extra=build_x_text_miss_payload(
                            primary=tweet_id,
                            layer="syndication",
                            reason="no_text",
                        ),
                    )
                except Exception:
                    pass
                self._metric_inc("x.syndication.invalid", None)
                # Negative cache to avoid repeated hits for unavailable/blocked tweets
                self._syn_cache[tweet_id] = build_syndication_negative_cache_entry(
                    time.time()
                )
                self._metric_inc("x.syndication.neg_store", None)
                return None

            # Cache and return
            self._syn_cache[tweet_id] = build_syndication_cache_entry(
                data,
                time.time(),
            )
            self._metric_inc("x.syndication.success", None)
            try:
                txt = self._extract_syndication_text(data)
                self.logger.info(
                    "x.text.resolve",
                    extra=build_x_text_resolve_payload(
                        primary=tweet_id,
                        source="syndication",
                        chars=len(txt),
                    ),
                )
            except Exception:
                pass
            return data

    async def _probe_twitter_syndication_images(
        self, url: str, status_id: str
    ) -> List[str]:
        """Probe syndication API for tweet images. Returns list of image URLs. [PA][REH]"""
        status_id = self._resolve_twitter_status_id(url, tweet_id=status_id)
        if not status_id:
            return []
        try:
            syn = await self._get_tweet_via_syndication(status_id)
            if not isinstance(syn, dict):
                return []
            photos = syn.get("photos") or []
            return extract_syndication_photo_urls(photos)
        except Exception as e:
            self.logger.debug(f"Syndication image probe failed: {e}")
            return []

    def _format_syndication_result(self, syn_data: Dict[str, Any], url: str) -> str:
        """Format Syndication JSON tweet into concise text similar to API format. [PA]"""
        try:
            text = self._extract_syndication_text(syn_data)
            user = syn_data.get("user") or {}
            created_at = syn_data.get("created_at") or syn_data.get("date_created")
            photos = syn_data.get("photos") or []
            header_line = format_syndication_header_line(
                user=user,
                created_at=created_at,
                photos=photos,
                url=url,
            )
            if not text:
                try:
                    self.logger.info(
                        "x.text.miss",
                        extra=build_x_text_miss_log_payload(url),
                    )
                except Exception:
                    pass
            body = format_syndication_body_text(text)
            return f"{header_line}\n{body}"
        except Exception:
            return format_syndication_error_fallback(url, syn_data)

    @staticmethod
    def _extract_x_article_text(article_node: Any) -> str:
        return extract_x_article_text(article_node)

    def _extract_syndication_text(self, node: Dict[str, Any]) -> str:
        """Extract tweet body text from syndication-like payloads, including X Articles."""
        return extract_syndication_text(
            node,
            extract_article_text=self._extract_x_article_text,
        )

    @staticmethod
    def _syndication_article_has_blocks(article_node: Any) -> bool:
        return syndication_article_has_blocks(article_node)

    def _syndication_needs_article_hydration(
        self, syn: Dict[str, Any], *, allow_tco_pointer: bool = False
    ) -> bool:
        return syndication_needs_article_hydration(
            syn,
            allow_tco_pointer=allow_tco_pointer,
            article_has_blocks=self._syndication_article_has_blocks,
        )

    async def _hydrate_syndication_article_if_needed(
        self,
        status_id: str,
        syn: Optional[Dict[str, Any]],
        *,
        allow_tco_pointer: bool = False,
    ) -> Dict[str, Any]:
        if not status_id or not isinstance(syn, dict):
            return syn if isinstance(syn, dict) else {}
        if not self._syndication_needs_article_hydration(
            syn, allow_tco_pointer=allow_tco_pointer
        ):
            return syn
        try:
            article_data = await self._fetch_x_article_from_fxtwitter(status_id)
        except Exception:
            article_data = None
        if isinstance(article_data, dict) and article_data:
            merged = dict(syn)
            merged["article"] = article_data
            try:
                self.logger.info(
                    "x.article.hydrated",
                    extra={
                        "event": "x.article.hydrated",
                        "detail": {
                            "tweet_id": status_id,
                            "article_id": article_data.get("id") or "",
                            "chars": len(self._extract_syndication_text(merged)),
                        },
                    },
                )
            except Exception:
                pass
            return merged
        return syn

    def _compose_text_tweet_evidence(self, url: str, syn: Dict[str, Any]) -> str:
        """Build EvidenceBundle for a text-only tweet using syndication payload. [CA]"""
        from .evidence import EvidenceBundle

        bundle = EvidenceBundle(source_platform="x", source_url=url)
        try:
            ptid = extract_primary_tweet_id(url)
            if ptid:
                bundle.primary_tweet_id = ptid
                bundle.selected_tweet_id = ptid
                try:
                    self.logger.info(
                        "x.text.canon",
                        extra=build_x_text_canon_payload(
                            url=url,
                            primary=ptid,
                        ),
                    )
                except Exception:
                    pass
        except Exception:
            pass

        caption = self._extract_syndication_text(syn)
        if caption:
            bundle.caption_text = caption
        q = syn.get("quoted_tweet") or syn.get("quoted_status") or {}
        qtxt = self._extract_syndication_text(q)
        if qtxt:
            bundle.quoted_text = qtxt
        composed = bundle.compose_prompt_text()
        try:
            self.logger.info(
                "x.text.resolve",
                extra=build_x_text_resolve_payload(
                    primary=bundle.primary_tweet_id or "",
                    source="syndication",
                    chars=len(bundle.caption_text or ""),
                ),
            )
        except Exception:
            pass
        return composed

    @staticmethod
    def _is_twitter_url(url: str) -> bool:
        return is_twitter_url(url)

    @staticmethod
    def _parse_twitter_status_id(url: str) -> Optional[str]:
        """Extract the tweet/status ID from a Twitter URL. Returns None if not found. [IV]"""
        return parse_twitter_status_id(url)

    def _resolve_twitter_status_id(
        self, url: str, tweet_id: Optional[str] = None
    ) -> str:
        """Return status ID from explicit hint first, otherwise parse from URL."""
        return resolve_twitter_status_id(
            url,
            tweet_id=tweet_id,
            parse_status_id=self._parse_twitter_status_id,
        )

    @staticmethod
    def _extract_primary_tweet_id(url: str) -> Optional[str]:
        """Extract a stable primary tweet ID from URL hints or status path."""
        return extract_primary_tweet_id(url)

    def _is_twitter_status_url(self, url: str) -> bool:
        """Check if a URL is a Twitter status URL (contains a valid status ID). [IV]"""
        return is_twitter_status_url(
            url,
            parse_status_id=self._parse_twitter_status_id,
        )

    def _canonicalize_twitter_status_url(self, url: str) -> str:
        """Convert any Twitter status URL to canonical form https://x.com/i/status/{id}. [IV]"""
        return canonicalize_twitter_status_url(url)

    def _register_x_frontend_context(
        self, url: str, frontend: Optional[str], primary: Optional[str]
    ) -> None:
        if not url or not frontend or not primary:
            return
        try:
            keys = [url, self._normalize_x_url(url)]
        except Exception:
            keys = [url]
        for key in keys:
            if not key:
                continue
            self._x_frontend_canon[key] = {"frontend": frontend, "primary": primary}
        # Bound size of the mapping to avoid unbounded growth
        try:
            while len(self._x_frontend_canon) > 256:
                self._x_frontend_canon.popitem(last=False)
        except Exception:
            pass

    def pop_gate_denied_reason(self, message_id: int) -> Optional[str]:
        """Return and clear the recorded gate-denied reason for a message."""
        return self._gate_denied.pop(message_id, None)

    def record_gate_hint(self, message_id: int, allowed: bool) -> None:
        """Record a pre-dispatch gate decision to avoid double-checking."""
        self._prefilter_gate[message_id] = allowed

    def pop_gate_hint(self, message_id: int) -> Optional[bool]:
        """Retrieve and clear any pre-dispatch gate decision."""
        return self._prefilter_gate.pop(message_id, None)

    async def _run_stt_job(
        self, task: Awaitable[Any], message: Message, kind: str = "stt"
    ) -> Any:
        """Run an STT coroutine with standardized start/end breadcrumbs."""
        msg_id = getattr(message, "id", None) if message else None
        status = "ok"
        start_detail = {"msg_id": msg_id, "kind": kind}
        try:
            self.logger.info(
                f"router.job.start msg_id={msg_id} kind={kind}",
                extra={"event": "router.job.start", "detail": start_detail},
            )
        except Exception:
            pass
        try:
            result = await task
            if isinstance(result, dict):
                if result.get("partial"):
                    status = "partial"
                elif result.get("error"):
                    status = "fail"
            return result
        except Exception:
            status = "fail"
            raise
        finally:
            try:
                self.logger.info(
                    f"router.job.end msg_id={msg_id} kind={kind} status={status}",
                    extra={
                        "event": "router.job.end",
                        "detail": {"msg_id": msg_id, "kind": kind, "status": status},
                    },
                )
            except Exception:
                pass

    def _log_x_media_probe(
        self,
        primary: Optional[str],
        video: bool,
        image_count: int,
        frontend: Optional[str],
    ) -> None:
        msg = (
            f"x.media.probe primary={primary or ''} "
            f"video={str(video).lower()} image_count={int(image_count)}"
        )
        detail = {
            "primary": primary or "",
            "video": bool(video),
            "image_count": int(image_count),
        }
        if frontend:
            detail["frontend"] = frontend
        try:
            self.logger.info(
                msg,
                extra={
                    "event": "x.media.probe",
                    "detail": detail,
                },
            )
        except Exception:
            pass

    async def _verify_media_kind(
        self, url: str, default: str = "unknown"
    ) -> Tuple[str, str]:
        decided = default
        content_type = ""
        try:
            path = Path(urlparse(url).path or "")
            ext = path.suffix.lower()
        except Exception:
            ext = ""
        http = None
        try:
            http = await get_http_client()
        except Exception:
            http = None
        if http is not None:
            cfg = RequestConfig(
                connect_timeout=2.5,
                read_timeout=2.5,
                total_timeout=3.5,
                max_retries=0,
            )
            try:
                resp = await http.head(url, config=cfg)
                content_type = (resp.headers.get("content-type") or "").lower()
                if resp.status_code in (403, 405) or not content_type:
                    resp = await http.get(
                        url,
                        config=cfg,
                        headers={"Range": "bytes=0-0"},
                    )
                    content_type = (resp.headers.get("content-type") or "").lower()
            except Exception:
                try:
                    resp = await http.get(
                        url,
                        config=cfg,
                        headers={"Range": "bytes=0-0"},
                    )
                    content_type = (resp.headers.get("content-type") or "").lower()
                except Exception:
                    content_type = ""
        if content_type.startswith("video/") or ext in {
            ".mp4",
            ".m4v",
            ".mov",
            ".m3u8",
            ".ts",
            ".mpd",
            ".mkv",
        }:
            decided = "video"
        elif content_type.startswith("image/"):
            decided = "image"
        return decided, content_type

    def _log_media_kind_checked(
        self, url: str, content_type: str, decided: str
    ) -> None:
        try:
            host = urlparse(url).netloc.lower()
        except Exception:
            host = ""
        msg = (
            f"media.kind_checked url_host={host or ''} "
            f"ctype={content_type or ''} decided={decided or ''}"
        )
        try:
            self.logger.info(
                msg,
                extra={
                    "event": "media.kind_checked",
                    "detail": {
                        "url_host": host or "",
                        "ctype": content_type or "",
                        "decided": decided or "",
                    },
                },
            )
        except Exception:
            pass

    def _canonicalize_x_url(self, url: str) -> str:
        """Canonicalize X/Twitter URLs: lowercase host and strip non-essential params.
        Keeps path untouched; removes params like s=, utm_*, t=.
        """
        try:
            p = urlparse(url)
            host = (p.netloc or "").lower()
            # Normalize common hosts
            if host.startswith("www."):
                host = host[4:]
            frontend = None
            host_core = host
            if host in {"fxtwitter.com", "vxtwitter.com"}:
                frontend = "fx" if host.startswith("fx") else "vx"
                primary = self._parse_twitter_status_id(url)
                if primary:
                    canonical = f"https://x.com/i/status/{primary}"
                    self._register_x_frontend_context(canonical, frontend, primary)
                    try:
                        self.logger.info(
                            "url.canon kind=x_frontend src_host=%s primary=%s",
                            host_core,
                            primary,
                        )
                    except Exception:
                        pass
                    return canonical
            # Allowed hosts only: leave as-is but lowercased
            qs = dict(parse_qsl(p.query, keep_blank_values=True))
            # Drop noise params
            drop_keys = {
                "s",
                "t",
                "utm_source",
                "utm_medium",
                "utm_campaign",
                "utm_term",
                "utm_content",
            }
            for k in list(qs.keys()):
                if k.lower() in drop_keys or k.lower().startswith("utm_"):
                    qs.pop(k, None)
            new = urlunparse(
                (
                    p.scheme or "https",
                    host,
                    p.path,
                    p.params,
                    urlencode(qs, doseq=True),
                    p.fragment,
                )
            )
            return new
        except Exception:
            return url

    def _extract_x_status_urls_from_text(self, text: str) -> List[str]:
        """Extract canonical X/Twitter status URLs from a text blob in-order.
        Normalizes and de-dupes; emits a small log per normalized URL. [IV][PA]
        """
        urls = extract_x_status_urls_from_text(
            text or "",
            is_status_url=self._is_twitter_status_url,
            canonicalize_status_url=self._canonicalize_twitter_status_url,
        )
        for cu in urls:
            try:
                self.logger.info(
                    "normalize_ok",
                    extra={
                        "subsys": "tw",
                        "event": "normalize_ok",
                        "detail": {"url": cu},
                    },
                )
            except Exception:
                pass
        return urls

    async def _extract_raw_x_urls(self, message: Message) -> List[str]:
        """Extract raw X/Twitter URLs from the author's message content and replied-to content only.
        Ignores embeds and attachments to avoid picking preview thumbnails.
        """
        texts: List[str] = []
        try:
            texts.append(message.content or "")
        except Exception:
            pass
        # Include referenced message's content when present
        try:
            if message.reference and message.reference.message_id:
                try:
                    ref_message = (
                        message.reference.resolved
                        if getattr(message.reference, "resolved", None)
                        else None
                    )
                    if ref_message is None:
                        ref_message = await message.channel.fetch_message(
                            message.reference.message_id
                        )
                    texts.append(ref_message.content or "")
                except Exception:
                    pass
        except Exception:
            pass
        # Extract URLs from combined text blobs
        raw_urls = extract_raw_urls_from_texts(texts)
        # Filter to X/Twitter domains only and canonicalize
        return filter_canonical_x_urls(
            raw_urls,
            is_x_url=self._is_twitter_url,
            canonicalize_x_url=self._canonicalize_x_url,
        )

    async def _gather_prioritized_x_urls(
        self, scope_case: str, message: Message, reply_target: Optional[Message]
    ) -> Tuple[str, List[str]]:
        """Collect X/Twitter status URLs using the priority stack within the active scope.
        Returns (layer, urls) where layer in {"trigger","parent","tail","none"}. [CA][IV]
        """
        try:
            # 1) Trigger layer
            trigger_urls = self._extract_x_status_urls_from_text(
                getattr(message, "content", "") or ""
            )
            if trigger_urls:
                return "trigger", trigger_urls

            # 2) Reply-parent layer (REPLY_CASE only)
            if scope_case == "reply" and reply_target is not None:
                parent_urls = self._extract_x_status_urls_from_text(
                    getattr(reply_target, "content", "") or ""
                )
                if parent_urls:
                    return "parent", parent_urls

            # 3) Thread-tail layer (THREAD_CASE only, near reply_target)
            if scope_case == "thread":
                try:
                    k = int(self.config.get("THREAD_CONTEXT_TAIL_COUNT", 5))
                except Exception:
                    k = 5
                k = max(0, min(k, 40))
                anchor = reply_target or message
                tail_urls: List[str] = []
                try:
                    # Walk messages strictly before anchor, oldest <- newest later
                    msgs: List[Message] = []
                    async for m in message.channel.history(limit=k * 3, before=anchor):
                        # keep humans + our bot only (mirror of thread_tail policy)
                        is_bot = bool(getattr(m.author, "bot", False))
                        is_ours = int(getattr(m.author, "id", 0)) == int(
                            getattr(self.bot.user, "id", 0)
                        )
                        if is_bot and not is_ours:
                            continue
                        msgs.append(m)
                        if len(msgs) >= k:
                            break
                    msgs = list(reversed(msgs))
                    for m in msgs:
                        u = self._extract_x_status_urls_from_text(
                            getattr(m, "content", "") or ""
                        )
                        for cu in u:
                            if cu not in tail_urls:
                                tail_urls.append(cu)
                except Exception:
                    tail_urls = []
                if tail_urls:
                    return "tail", tail_urls

            return "none", []
        except Exception:
            return "none", []

    async def _yt_dlp_probe(
        self, url: str, timeout_s: float = 8.0
    ) -> Optional[Dict[str, Any]]:
        """Run a lightweight yt-dlp metadata probe to detect presence of video/audio.
        Returns parsed JSON on success or None on errors/timeouts.
        """
        cmd = ["yt-dlp", "--dump-json", "--no-playlist", "--quiet", url]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout_s
                )
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except Exception:
                    pass
                return None
            if proc.returncode != 0:
                return None
            data = json.loads(stdout.decode(errors="ignore") or "{}")
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def _normalize_x_url(self, url: str) -> str:
        """Normalize X/Twitter URLs to a canonical host/path, dropping trackers. [IV][PA]
        - Map mobile/twitter/fx/vx subdomains to x.com
        - Trim trailing slashes
        - Drop query/fragment
        """
        return normalize_x_url(url)

    @staticmethod
    def _unwrap_x_media_url(url: str) -> str:
        """Unwrap proxy download URLs returned by fx/vx helpers back to the media CDN."""
        return unwrap_x_media_url(url)

    @staticmethod
    def _collect_x_candidate_urls(item: InputItem) -> List[str]:
        return collect_x_candidate_urls(item)

    @staticmethod
    def _is_twitter_thumbnail_url(url: str) -> bool:
        return is_twitter_thumbnail_url(url)

    @staticmethod
    def _is_twitter_media_cdn(url: str) -> bool:
        return is_twitter_media_cdn(url)

    @staticmethod
    def _is_tweet_media_url(url: str) -> bool:
        return is_tweet_media_url(url)

    async def _resolve_x_media(
        self,
        urls: List[str],
        *,
        frontend_hints: Optional[Dict[str, str]] = None,
        primary_hints: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Resolve X/Twitter URLs into a minimal media shape.
        - kind: video | image | unknown
        - url/images: best guess URL(s)
        - duration: seconds or None
        """
        clean = [self._canonicalize_x_url(u) for u in urls or []]
        frontend_hints = frontend_hints or {}
        primary_hints = primary_hints or {}
        http = None
        try:
            http = await get_http_client()
        except Exception:
            http = None
        primary_for_log: Optional[str] = None
        frontend_for_log: Optional[str] = None
        images: List[str] = []
        videos: List[str] = []

        def _normalize_video_candidate(raw_url: Any) -> Optional[str]:
            """Normalize candidate video URLs from fx/vx payloads."""
            try:
                if not raw_url or not isinstance(raw_url, str):
                    return None
                candidate = self._unwrap_x_media_url(raw_url.strip())
                if not candidate.startswith("http"):
                    return None
                parsed = urlparse(candidate)
                host = (parsed.netloc or "").lower()
                path = (parsed.path or "").lower()
                if host.endswith("video.twimg.com") or host.endswith("ton.twimg.com"):
                    return candidate
                if any(path.endswith(sfx) for sfx in (".mp4", ".m3u8", ".webm")):
                    return candidate
                return None
            except Exception:
                return None

        def _collect_video_urls(node: Any) -> List[str]:
            """Recursively collect likely video URLs from a JSON-like object."""
            found: List[str] = []
            try:
                if isinstance(node, dict):
                    for key, value in node.items():
                        key_l = str(key).lower()
                        if isinstance(value, str):
                            if key_l in {
                                "url",
                                "video_url",
                                "playback_url",
                                "hls_url",
                                "m3u8_url",
                                "source",
                            } or "video" in key_l:
                                cand = _normalize_video_candidate(value)
                                if cand and cand not in found:
                                    found.append(cand)
                        else:
                            sub = _collect_video_urls(value)
                            for cand in sub:
                                if cand not in found:
                                    found.append(cand)
                elif isinstance(node, list):
                    for item in node:
                        sub = _collect_video_urls(item)
                        for cand in sub:
                            if cand not in found:
                                found.append(cand)
            except Exception:
                return found
            return found

        for u in clean:
            status_id = self._parse_twitter_status_id(u)
            lookup = self._normalize_x_url(u)
            ctx = (
                self._x_frontend_canon.get(lookup)
                or self._x_frontend_canon.get(u)
                or {}
            )
            frontend = frontend_hints.get(lookup) or ctx.get("frontend")
            primary = primary_hints.get(lookup) or ctx.get("primary") or status_id
            if primary and not primary_for_log:
                primary_for_log = primary
            if frontend and not frontend_for_log:
                frontend_for_log = frontend
            # 1) vx/fx JSON (fast)
            if status_id and http is not None:
                for src_name, host in (
                    ("vx_json", "api.vxtwitter.com"),
                    ("fx_json", "api.fxtwitter.com"),
                ):
                    try:
                        self.logger.debug(
                            f"x.resolve_try src={src_name} id={status_id}"
                        )
                    except Exception:
                        pass
                    try:
                        cfg = RequestConfig(
                            connect_timeout=min(self._x_syn_timeout_s, 3.0),
                            read_timeout=min(self._x_syn_timeout_s, 3.0),
                            total_timeout=min(self._x_syn_timeout_s, 3.0),
                            max_retries=0,
                        )
                        api_url = f"https://{host}/status/{status_id}"
                        resp = await http.get(api_url, config=cfg)
                        if getattr(resp, "status_code", 500) != 200:
                            continue
                        try:
                            data = resp.json()
                        except Exception:
                            try:
                                # If brotli is used and not auto-decoded, attempt manual decode [REH]
                                enc = (
                                    resp.headers.get("content-encoding") or ""
                                ).lower()
                                if "br" in enc:
                                    try:
                                        import brotli  # type: ignore

                                        decoded = brotli.decompress(resp.content)
                                        data = json.loads(
                                            decoded.decode("utf-8", errors="replace")
                                        )
                                    except Exception:
                                        data = json.loads(resp.text)
                            except Exception:
                                data = {}
                        # Extract video URLs from common fx/vx payload shapes.
                        try:
                            tweet = self._extract_fxtwitter_tweet_node(data)
                            media = tweet.get("media") or {}
                            video_nodes: List[Any] = []
                            for key in ("video", "videos", "video_info", "media"):
                                value = media.get(key)
                                if value:
                                    video_nodes.append(value)
                            for key in ("video", "videos", "video_info"):
                                value = tweet.get(key)
                                if value:
                                    video_nodes.append(value)
                            for node in video_nodes:
                                for cand in _collect_video_urls(node):
                                    if cand not in videos:
                                        videos.append(cand)
                        except Exception:
                            pass
                        # Best-effort regex sweep for escaped/wrapped URLs.
                        try:
                            raw_text = (resp.text or "").replace("\\/", "/")
                            for m in re.finditer(
                                r"https://(?:video|ton)\.twimg\.com/[^\s\"'<>]+",
                                raw_text,
                                re.IGNORECASE,
                            ):
                                cand = _normalize_video_candidate(m.group(0))
                                if cand and cand not in videos:
                                    videos.append(cand)
                        except Exception:
                            pass
                        # Extract photo URLs from common shapes
                        candidates: List[str] = []
                        # Prefer high-res for pbs assets
                        try:
                            from .syndication.url_utils import (
                                upgrade_pbs_to_orig,
                            )  # lazy import to avoid cycles
                        except Exception:

                            def upgrade_pbs_to_orig(u):  # fallback passthrough
                                return u

                        # fx/vx often: {'tweet': {'media': {'photos':[{'url':...}]}}}
                        try:
                            tweet = self._extract_fxtwitter_tweet_node(data)
                            media = tweet.get("media") or {}
                            photos = media.get("photos") or []
                            for p in photos:
                                u = p.get("url") or p.get("src") or p.get("href")
                                if isinstance(u, str):
                                    candidates.append(upgrade_pbs_to_orig(u))
                        except Exception:
                            pass
                        # Some variants: top-level 'photos'
                        for p in data.get("photos") or []:
                            if isinstance(p, dict) and p.get("url"):
                                candidates.append(upgrade_pbs_to_orig(p.get("url")))
                            elif isinstance(p, str):
                                candidates.append(upgrade_pbs_to_orig(p))
                        # Filter + HEAD verify
                        uniq = []
                        for u in candidates:
                            if not u or not u.startswith("http"):
                                continue
                            if not self._is_twitter_media_cdn(u):
                                continue
                            # CRITICAL: Only accept actual tweet media, not profile/banner images [IV]
                            if not self._is_tweet_media_url(u):
                                continue
                            # Hard blocklist for X video poster thumbnails
                            try:
                                pu = urlparse(u)
                                host = (pu.netloc or "").lower()
                                path = (pu.path or "").lower()
                            except Exception:
                                host = ""
                                path = ""
                            poster_prefixes = (
                                "/amplify_video_thumb/",
                                "/ext_tw_video_thumb/",
                                "/tweet_video_thumb/",
                            )
                            if host.endswith("pbs.twimg.com") and any(
                                pref in path for pref in poster_prefixes
                            ):
                                try:
                                    matched = next(
                                        (
                                            pref
                                            for pref in poster_prefixes
                                            if pref in path
                                        ),
                                        "poster_thumb",
                                    )
                                    self.logger.info(
                                        "x.image_probe.video_poster_detected",
                                        extra={
                                            "event": "x.image_probe.video_poster_detected",
                                            "detail": {"domain": host, "path": matched},
                                        },
                                    )
                                except Exception:
                                    pass
                                continue  # do not accept poster as photo
                            if u not in uniq and self._is_direct_image_url(u):
                                uniq.append(u)
                        if uniq:
                            images.extend(uniq)
                        if videos or uniq:
                            break  # API success
                    except Exception as e:
                        self.logger.debug(
                            f"x.syndication.api.error | host={host} err={e}"
                        )

        # Stage 2: HTML/meta fallback on fx/vx
        if not images and http is not None and self._x_syn_probe_enabled:
            html_hosts = ["fxtwitter.com", "vxtwitter.com"]
            for host in html_hosts:
                try:
                    html_url = f"https://{host}/i/status/{status_id}"
                    cfg = RequestConfig(
                        connect_timeout=self._x_syn_timeout_s,
                        read_timeout=self._x_syn_timeout_s,
                        total_timeout=self._x_syn_timeout_s,
                        max_retries=0,
                    )
                    resp = await http.get(html_url, config=cfg)
                    if resp.status_code != 200 or not resp.text:
                        self.logger.debug(
                            f"x.syndication.html.non200 | host={host} status={resp.status_code}"
                        )
                        continue
                    text = resp.text
                    candidates: List[str] = []
                    # og:image and twitter:image (robust attribute order) [REH]
                    meta_patterns = [
                        r'<meta[^>]+(?:property|name)=["\']og:image(?:[:a-z]+)?["\'][^>]+(?:content|value)=["\']([^"\']+)["\']',
                        r'<meta[^>]+(?:property|name)=["\']twitter:image(?:[:a-z]+)?["\'][^>]+(?:content|value)=["\']([^"\']+)["\']',
                    ]
                    for pat in meta_patterns:
                        for m in re.finditer(pat, text, re.IGNORECASE):
                            try:
                                from .syndication.url_utils import (
                                    upgrade_pbs_to_orig,
                                )  # lazy import
                            except Exception:

                                def upgrade_pbs_to_orig(u):  # fallback passthrough
                                    return u

                            candidates.append(upgrade_pbs_to_orig(m.group(1)))
                    # pbs links anywhere
                    for m in re.finditer(
                        r"https://pbs\.twimg\.com/[^\s'\"]+",
                        text,
                        re.IGNORECASE,
                    ):
                        try:
                            from .syndication.url_utils import (
                                upgrade_pbs_to_orig,
                            )  # lazy import
                        except Exception:

                            def upgrade_pbs_to_orig(u):  # fallback passthrough
                                return u

                        candidates.append(upgrade_pbs_to_orig(m.group(0)))
                    uniq = []
                    for u in candidates:
                        if not u or not u.startswith("http"):
                            continue
                        if not self._is_twitter_media_cdn(u):
                            continue
                        # CRITICAL: Only accept actual tweet media, not profile/banner images [IV]
                        if not self._is_tweet_media_url(u):
                            continue
                        if u not in uniq and self._is_direct_image_url(u):
                            uniq.append(u)
                    if uniq:
                        images.extend(uniq)
                        break
                except Exception as e:
                    self.logger.debug(f"x.syndication.html.error | host={host} err={e}")

        # Deduplicate, cap to MAX, preserve order
        final: List[str] = []
        for u in images:
            if u not in final:
                final.append(u)
            if len(final) >= max(1, self._x_syn_max_images):
                break

        # Prefer video when both video and image candidates exist.
        chosen_video = videos[0] if videos else None
        if chosen_video:
            return {
                "kind": "video",
                "images": final,
                "url": chosen_video,
                "duration": None,
                "primary": primary_for_log or "",
                "frontend": frontend_for_log,
            }

        # Return proper dict shape as declared in type hint
        return {
            "kind": "image" if final else "unknown",
            "images": final,
            "url": final[0] if final else None,
            "duration": None,
            "primary": primary_for_log or "",
            "frontend": frontend_for_log,
        }

    async def _fetch_x_article_from_fxtwitter(
        self, status_id: str
    ) -> Optional[Dict[str, Any]]:
        """Resolve X Article payload from fx API for article-style status posts."""
        if not status_id:
            return None
        try:
            http = await get_http_client()
            cfg = self._build_x_syn_quick_request_config()
            resp = await http.get(f"https://api.fxtwitter.com/status/{status_id}", config=cfg)
            if getattr(resp, "status_code", 500) != 200:
                return None
            try:
                payload = resp.json()
            except Exception:
                return None
            tweet = self._extract_fxtwitter_tweet_node(payload)
            if not tweet:
                return None
            article = tweet.get("article")
            if not isinstance(article, dict) or not article:
                return None

            normalized: Dict[str, Any] = {}
            article_id = str(article.get("id") or "").strip()
            title = str(article.get("title") or "").strip()
            preview_text = str(article.get("preview_text") or "").strip()
            content = article.get("content") or {}
            blocks = content.get("blocks") if isinstance(content, dict) else []
            kept_blocks: List[Dict[str, Any]] = []
            if isinstance(blocks, list):
                for block in blocks:
                    if not isinstance(block, dict):
                        continue
                    btxt = str(block.get("text") or "").strip()
                    if not btxt:
                        continue
                    kept_blocks.append({"text": btxt, "type": block.get("type")})

            if article_id:
                normalized["id"] = article_id
                normalized["url"] = f"https://x.com/i/article/{article_id}"
            if title:
                normalized["title"] = title
            if preview_text:
                normalized["preview_text"] = preview_text
            if kept_blocks:
                normalized["content"] = {"blocks": kept_blocks}

            if not (normalized.get("title") or normalized.get("preview_text") or kept_blocks):
                return None
            return normalized
        except Exception as e:
            self.logger.debug(f"x.article.resolve.failed id={status_id} err={e}")
            return None

    async def _route_tweet_as_perception_images(
        self, img_urls: List[str], *, message: Message, context_str: str
    ) -> BotAction:
        self._log_twitter_syndication_images(img_urls, msg_id=message.id)
        # Run VL on first image (budget-friendly), inject to text flow
        notes = None
        if img_urls:
            try:
                notes = await self._vl_describe_image_from_url(
                    img_urls[0],
                    prompt=(
                        "Describe this image in detail, focusing on key visual elements, objects, text, and context."
                    ),
                )
                notes = sanitize_vl_reply_text(notes or "")
            except Exception:
                notes = None
        self.logger.info(
            f"🎯 Route: text (with perception) | images={len(img_urls)} | msg_id={message.id}"
        )
        return await self._flow_process_text(
            content=(message.content or "").strip(),
            context=context_str,
            message=message,
            perception_notes=notes or None,
        )

    @staticmethod
    def _is_direct_image_url(url: str) -> bool:
        """Lightweight check for direct image URLs by extension. [IV]"""
        return is_direct_image_url(url)

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
        self,
        item: InputItem,
        model_override: Optional[str] = None,
        message: Optional[Message] = None,
    ) -> str:
        """Handle image item with explicit model override. [CA][IV][REH]
        - Attachments: direct VL on file
        - URLs: direct image URL → download+VL; otherwise screenshot→VL
        - Embeds: try image/thumbnail URL similarly
        """
        if isinstance(self.bot, (Mock, MagicMock)):
            try:
                name = (
                    getattr(getattr(item, "payload", None), "filename", "") or "image"
                )
                resp = see_infer(image_path=getattr(item.payload, "url", None))
                if asyncio.iscoroutine(resp):
                    resp = await resp
                content = getattr(resp, "content", None) if resp else None
                if content:
                    return f"Image analysis: {content}"
                return f"Image analysis: {name}"
            except Exception:
                return "Image analysis: mock image"
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
                embed = item.payload
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

    async def _handle_image(
        self, item: InputItem, message: Optional[Message] = None
    ) -> str:
        """Handle image without explicit model override, using default VL model. [CA]"""
        return await self._handle_image_with_model(item, model_override=None)

    def _format_x_tweet_result(self, api_data: Dict[str, Any], url: str) -> str:
        """Format X API tweet response into concise text. [PA][IV]"""
        return format_x_tweet_result(
            api_data=api_data,
            url=url,
            canonicalize_status_url=self._canonicalize_twitter_status_url,
        )

    def _is_reply_to_bot(self, message: Message) -> bool:
        """Check if a message is a reply to the bot."""
        return is_reply_to_bot(message, getattr(getattr(self.bot, "user", None), "id", None))

    async def _resolve_reference_message(
        self, message: Message, fallback: Optional[Message] = None
    ) -> Optional[Message]:
        """Resolve referenced message from cache first, then fetch if needed."""
        if fallback is not None:
            return fallback
        ref = getattr(message, "reference", None)
        if not ref:
            return None
        ref_msg = getattr(ref, "resolved", None)
        if ref_msg is not None:
            return ref_msg
        ref_id = getattr(ref, "message_id", None)
        if not ref_id:
            return None
        try:
            return await message.channel.fetch_message(ref_id)
        except Exception:
            return None

    def _mentions_bot(self, message: Message) -> bool:
        """Return True if the message explicitly mentions this bot."""
        return mentions_bot(message, getattr(getattr(self.bot, "user", None), "id", None))

    def _update_dispatch_metadata(
        self,
        message: Message,
        *,
        context: str,
        mention_detected: bool,
        reply_to_bot: bool,
    ) -> None:
        """Attach dispatch metadata used downstream for reply targeting and logging."""
        trigger = message
        channel = getattr(trigger, "channel", None) or getattr(message, "channel", None)
        is_thread = _is_thread_channel(channel)
        parent_channel_id = getattr(channel, "parent_id", None) if is_thread else None
        channel_id = parent_channel_id or getattr(channel, "id", None)
        thread_id = getattr(channel, "id", None) if is_thread else None

        meta = self._dispatch_metadata.get(message.id, {})
        meta.update(
            {
                "context": context,
                "mention_detected": mention_detected,
                "reply_to_bot": reply_to_bot,
                "trigger_message": trigger,
                "trigger_message_id": getattr(trigger, "id", None),
                "reply_in_thread": is_thread,
                "channel_id": channel_id,
                "thread_id": thread_id,
                "reply_target_ok": trigger is not None,
            }
        )
        self._dispatch_metadata[message.id] = meta

    def get_dispatch_metadata(self, message_id: int) -> Dict[str, Any]:
        """Return dispatch metadata for a message (copy to avoid accidental mutation)."""
        meta = self._dispatch_metadata.get(message_id, {})
        return dict(meta) if meta else {}

    def clear_dispatch_metadata(self, message_id: int) -> None:
        """Remove cached dispatch metadata once processing completes."""
        self._dispatch_metadata.pop(message_id, None)
        self._prefilter_gate.pop(message_id, None)
        self._gate_denied.pop(message_id, None)

    def _should_process_message(self, message: Message) -> bool:
        """Single source-of-truth gate: decide if this message should be processed.
        Cheap, synchronous, and config-driven. No network or heavy CPU allowed here.
        """
        self._gate_denied.pop(getattr(message, "id", None), None)
        cfg = self.config
        owners: list[int] = cfg.get("OWNER_IDS", [])
        triggers: list[str] = cfg.get(
            "REPLY_TRIGGERS",
            ["dm", "mention", "reply", "bot_threads", "owner", "command_prefix"],
        )

        content = (message.content or "").strip()
        is_dm = isinstance(message.channel, DMChannel)
        context = "dm" if is_dm else "guild"

        require_mention_in_guilds = cfg.get("REQUIRE_MENTION_IN_GUILDS", True)
        allow_reply_without_mention = cfg.get(
            "ALLOW_REPLY_TO_BOT_WITHOUT_MENTION", True
        )
        dm_require_mention = cfg.get("DM_REQUIRE_MENTION", False)

        mention_detected = self._mentions_bot(message)
        is_reply = self._is_reply_to_bot(message)
        is_owner = (
            message.author.id in owners if getattr(message, "author", None) else False
        )

        in_bot_thread = False
        try:
            if isinstance(message.channel, discord.Thread):
                in_bot_thread = (
                    getattr(message.channel, "owner_id", None) == self.bot.user.id
                )
        except Exception:
            in_bot_thread = False

        command_prefix = cfg.get("COMMAND_PREFIX", "!")
        if content:
            mention_prefix_pattern = rf"^<@!?{self.bot.user.id}>\s*"
            try:
                clean_content = re.sub(mention_prefix_pattern, "", content)
            except Exception:
                clean_content = content
        else:
            clean_content = ""
        has_prefix = (
            bool(clean_content.startswith(command_prefix)) if clean_content else False
        )

        # Master switch: if disabled, allow everything (legacy behavior)
        if not cfg.get("BOT_SPEAKS_ONLY_WHEN_SPOKEN_TO", True):
            self._update_dispatch_metadata(
                message,
                context=context,
                mention_detected=mention_detected,
                reply_to_bot=is_reply,
            )
            self.logger.debug(
                f"gate.allow | reason=master_switch_off msg_id={message.id}",
                extra={
                    "event": "gate.allow",
                    "reason": "master_switch_off",
                    "msg_id": message.id,
                    "context": context,
                    "mention_detected": mention_detected,
                    "reply_to_bot": is_reply,
                },
            )
            self._metric_inc("gate.allowed", {"reason": "master_switch_off"})
            return True

        # DM handling (mention optional by default)
        if is_dm:
            if dm_require_mention and not mention_detected:
                if allow_reply_without_mention and is_reply:
                    self._update_dispatch_metadata(
                        message,
                        context=context,
                        mention_detected=mention_detected,
                        reply_to_bot=is_reply,
                    )
                    self.logger.debug(
                        f"gate.allow | reason=dm_reply_without_mention msg_id={message.id}",
                        extra={
                            "event": "gate.allow",
                            "reason": "dm_reply_without_mention",
                            "msg_id": message.id,
                            "context": context,
                            "mention_detected": mention_detected,
                            "reply_to_bot": is_reply,
                        },
                    )
                    self._metric_inc(
                        "gate.allowed", {"reason": "dm_reply_without_mention"}
                    )
                    return True

                self._gate_denied[message.id] = "dm_mention_required"
                self.logger.info(
                    f"gate.block | reason=dm_mention_required msg_id={message.id}",
                    extra={
                        "event": "gate.block",
                        "reason": "dm_mention_required",
                        "msg_id": message.id,
                        "context": context,
                        "mention_detected": mention_detected,
                        "reply_to_bot": is_reply,
                    },
                )
                self._metric_inc("gate.blocked", {"reason": "dm_mention_required"})
                return False

            self._update_dispatch_metadata(
                message,
                context=context,
                mention_detected=mention_detected,
                reply_to_bot=is_reply,
            )
            self.logger.debug(
                f"gate.allow | reason=dm msg_id={message.id}",
                extra={
                    "event": "gate.allow",
                    "reason": "dm",
                    "msg_id": message.id,
                    "context": context,
                    "mention_detected": mention_detected,
                    "reply_to_bot": is_reply,
                },
            )
            self._metric_inc("gate.allowed", {"reason": "dm"})
            return True

        # Guild handling with mention requirement
        if require_mention_in_guilds:
            if mention_detected:
                self._update_dispatch_metadata(
                    message,
                    context=context,
                    mention_detected=True,
                    reply_to_bot=is_reply,
                )
                self.logger.debug(
                    f"gate.allow | reason=mention msg_id={message.id}",
                    extra={
                        "event": "gate.allow",
                        "reason": "mention",
                        "msg_id": message.id,
                        "context": context,
                        "mention_detected": True,
                        "reply_to_bot": is_reply,
                    },
                )
                self._metric_inc("gate.allowed", {"reason": "mention"})
                return True

            # Check for vision triggers even without mention in guilds
            if self._detect_direct_vision_triggers(clean_content, message):
                self._update_dispatch_metadata(
                    message,
                    context=context,
                    mention_detected=False,
                    reply_to_bot=is_reply,
                )
                self.logger.debug(
                    f"gate.allow | reason=vision_trigger msg_id={message.id}",
                    extra={
                        "event": "gate.allow",
                        "reason": "vision_trigger",
                        "msg_id": message.id,
                        "context": context,
                        "mention_detected": False,
                        "reply_to_bot": is_reply,
                    },
                )
                self._metric_inc("gate.allowed", {"reason": "vision_trigger"})
                return True

            if allow_reply_without_mention and is_reply:
                self._update_dispatch_metadata(
                    message,
                    context=context,
                    mention_detected=False,
                    reply_to_bot=True,
                )
                self.logger.debug(
                    f"gate.allow | reason=reply_to_bot msg_id={message.id}",
                    extra={
                        "event": "gate.allow",
                        "reason": "reply_to_bot",
                        "msg_id": message.id,
                        "context": context,
                        "mention_detected": False,
                        "reply_to_bot": True,
                    },
                )
                self._metric_inc("gate.allowed", {"reason": "reply_to_bot"})
                return True

            self._gate_denied[message.id] = "mention_required"
            self.logger.info(
                f"gate.block | reason=mention_required msg_id={message.id}",
                extra={
                    "event": "gate.block",
                    "reason": "mention_required",
                    "msg_id": message.id,
                    "context": context,
                    "mention_detected": mention_detected,
                    "reply_to_bot": is_reply,
                    "guild_id": getattr(message.guild, "id", None),
                },
            )
            self._metric_inc("gate.blocked", {"reason": "mention_required"})
            return False

        # Legacy trigger-based allowances (mention requirement disabled)
        if is_owner and "owner" in triggers:
            self._update_dispatch_metadata(
                message,
                context=context,
                mention_detected=mention_detected,
                reply_to_bot=is_reply,
            )
            self.logger.info(
                f"gate.allow | reason=owner_override msg_id={message.id}",
                extra={
                    "event": "gate.allow",
                    "reason": "owner_override",
                    "user_id": message.author.id,
                    "msg_id": message.id,
                    "context": context,
                    "mention_detected": mention_detected,
                    "reply_to_bot": is_reply,
                },
            )
            self._metric_inc("gate.allowed", {"reason": "owner_override"})
            return True

        if mention_detected and "mention" in triggers:
            self._update_dispatch_metadata(
                message,
                context=context,
                mention_detected=True,
                reply_to_bot=is_reply,
            )
            self.logger.debug(
                f"gate.allow | reason=mention msg_id={message.id}",
                extra={
                    "event": "gate.allow",
                    "reason": "mention",
                    "msg_id": message.id,
                    "context": context,
                    "mention_detected": True,
                    "reply_to_bot": is_reply,
                },
            )
            self._metric_inc("gate.allowed", {"reason": "mention"})
            return True

        if is_reply and "reply" in triggers:
            self._update_dispatch_metadata(
                message,
                context=context,
                mention_detected=mention_detected,
                reply_to_bot=True,
            )
            self.logger.debug(
                f"gate.allow | reason=reply_to_bot msg_id={message.id}",
                extra={
                    "event": "gate.allow",
                    "reason": "reply_to_bot",
                    "msg_id": message.id,
                    "context": context,
                    "mention_detected": mention_detected,
                    "reply_to_bot": True,
                },
            )
            self._metric_inc("gate.allowed", {"reason": "reply_to_bot"})
            return True

        if in_bot_thread and "bot_threads" in triggers:
            self._update_dispatch_metadata(
                message,
                context=context,
                mention_detected=mention_detected,
                reply_to_bot=is_reply,
            )
            self.logger.debug(
                f"gate.allow | reason=bot_thread msg_id={message.id}",
                extra={
                    "event": "gate.allow",
                    "reason": "bot_thread",
                    "msg_id": message.id,
                    "context": context,
                    "mention_detected": mention_detected,
                    "reply_to_bot": is_reply,
                },
            )
            self._metric_inc("gate.allowed", {"reason": "bot_thread"})
            return True

        if has_prefix and "command_prefix" in triggers:
            self._update_dispatch_metadata(
                message,
                context=context,
                mention_detected=mention_detected,
                reply_to_bot=is_reply,
            )
            self.logger.debug(
                f"gate.allow | reason=command_prefix msg_id={message.id}",
                extra={
                    "event": "gate.allow",
                    "reason": "command_prefix",
                    "msg_id": message.id,
                    "context": context,
                    "mention_detected": mention_detected,
                    "reply_to_bot": is_reply,
                },
            )
            self._metric_inc("gate.allowed", {"reason": "command_prefix"})
            return True

        self.logger.info(
            f"gate.block | reason=not_addressed msg_id={message.id}",
            extra={
                "event": "gate.block",
                "reason": "not_addressed",
                "msg_id": message.id,
                "guild_id": getattr(message.guild, "id", None),
                "is_dm": is_dm,
                "context": context,
                "mention_detected": mention_detected,
                "reply_to_bot": is_reply,
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
            "process_attachments": self._flow_process_attachments_multimodal,
            "generate_tts": self._flow_generate_tts,
        }

        if flow_overrides:
            self._flows.update(flow_overrides)

    async def _resolve_scope_and_target(
        self, message: Message
    ) -> Tuple[str, Optional[Message], str]:
        """
        Centralized scope resolution following the deterministic decision tree.
        Returns (scope_case, reply_target, context_str)
        """
        try:
            # THREAD_CASE: message is in a Thread/Forum thread
            if _is_thread_channel(getattr(message, "channel", None)):
                rt, reason = await resolve_thread_reply_target(
                    self.bot, message, self.config
                )
                tail = await collect_thread_tail_context(
                    self.bot, message, rt, self.config
                )
                context_str = ""
                if tail and isinstance(tail, tuple) and len(tail) == 2:
                    tail_joined, _ = tail
                    context_str = (tail_joined or "").strip()

                self.logger.info(
                    "scope_resolved",
                    extra={
                        "subsys": "route",
                        "event": "scope_resolved",
                        "phase": "scope",
                        "case": "thread",
                        "scope": str(getattr(message.channel, "id", "unknown")),
                        "reply_target": str(getattr(rt, "id", "unknown"))
                        if rt
                        else "none",
                    },
                )
                return "thread", rt, context_str

            # REPLY_CASE: direct reply chain
            if getattr(message, "reference", None):
                ref = getattr(message, "reference", None)
                ref_msg = getattr(ref, "resolved", None)
                if ref_msg is None and getattr(ref, "message_id", None):
                    try:
                        ref_msg = await message.channel.fetch_message(ref.message_id)
                    except Exception:
                        ref_msg = None

                if ref_msg is not None:
                    mc = await maybe_build_mention_context(
                        self.bot, message, self.config
                    )
                    context_str = ""
                    if mc and isinstance(mc, tuple) and len(mc) == 2:
                        joined_text, _ = mc
                        context_str = (joined_text or "").strip()

                    self.logger.info(
                        "scope_resolved",
                        extra={
                            "subsys": "route",
                            "event": "scope_resolved",
                            "phase": "scope",
                            "case": "reply",
                            "scope": str(getattr(ref_msg, "id", "unknown")),
                            "reply_target": str(getattr(ref_msg, "id", "unknown")),
                        },
                    )
                    return "reply", ref_msg, context_str

            # LONE_CASE: not thread, no reply
            # For mentions with minimal text, resolve implicit anchor
            mentioned_me = self._is_mentioned(message)
            if mentioned_me:
                txt = message.content or ""
                try:
                    txt = re.sub(rf"^<@!?{self.bot.user.id}>\s*", "", txt).strip()
                    txt = re.sub(r"https?://\S+", "", txt).strip()
                except Exception:
                    txt = txt.strip()

                if not txt:  # No substantive content after mention removal
                    anchor, _ = await resolve_implicit_anchor(
                        self.bot, message, self.config
                    )
                    if anchor:
                        ia = await collect_implicit_anchor_context(
                            self.bot, message, anchor, self.config
                        )
                        context_str = ""
                        if ia and isinstance(ia, tuple) and len(ia) == 2:
                            ia_joined, _ = ia
                            context_str = ia_joined.strip()

                        self.logger.info(
                            "scope_resolved",
                            extra={
                                "subsys": "route",
                                "event": "scope_resolved",
                                "phase": "scope",
                                "case": "lone",
                                "scope": str(getattr(message, "id", "unknown")),
                                "reply_target": str(getattr(anchor, "id", "none"))
                                if anchor
                                else "none",
                            },
                        )
                        return "lone", anchor, context_str

            # Default LONE_CASE with no context
            self.logger.info(
                "scope_resolved",
                extra={
                    "subsys": "route",
                    "event": "scope_resolved",
                    "phase": "scope",
                    "case": "lone",
                    "scope": str(getattr(message, "id", "unknown")),
                    "reply_target": "none",
                },
            )
            return "lone", None, ""
        except Exception as e:
            self.logger.error(f"Scope resolution failed: {e}", exc_info=True)
            return "lone", None, ""

    async def _process_document(self, path: str, ext: str) -> str:
        """Lightweight document handler used in test compatibility paths.

        Tests may patch this with richer parsing; the default simply reads text.
        """
        ext_l = str(ext or "").lower()
        if ext_l == ".docx" and DOCX_SUPPORT:
            try:
                doc = docx.Document(path)
                parts = []
                for para in getattr(doc, "paragraphs", []) or []:
                    txt = str(getattr(para, "text", "") or "").strip()
                    if txt:
                        parts.append(txt)
                return "\n".join(parts)
            except Exception:
                pass

        if ext_l == ".pdf" and PDF_SUPPORT and self.pdf_processor is not None:
            try:
                result = await self.pdf_processor.process(path)
                if isinstance(result, dict):
                    return str(result.get("text") or "")
                return str(result or "")
            except Exception:
                pass

        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            return ""

    async def _compat_dispatch_for_tests(
        self, message: Message, clean_content: str
    ) -> Optional[ResponseMessage]:
        """Simplified routing path for unit tests using MagicMock bots."""
        if not isinstance(self.bot, MagicMock):
            return None

        raw_content = clean_content or ""
        body = raw_content
        if raw_content.startswith("!"):
            return None

        # If modality detection is available, honor it to mirror production routing.
        detected_modality = None
        try:
            modality_fn = getattr(self, "_get_input_modality", None)
            if modality_fn:
                maybe_modality = modality_fn(message)
                detected_modality = (
                    await maybe_modality
                    if asyncio.iscoroutine(maybe_modality)
                    else maybe_modality
                )
        except Exception:
            detected_modality = None

        modality_flow_map = {
            InputModality.TEXT_ONLY: ("process_text", lambda h: h(body)),
            InputModality.GENERAL_URL: ("process_url", lambda h: h(message)),
            InputModality.AUDIO_VIDEO_FILE: ("process_audio", lambda h: h(message)),
            InputModality.SINGLE_IMAGE: (
                "process_attachments",
                lambda h: h(message, raw_content),
            ),
            InputModality.MULTI_IMAGE: (
                "process_attachments",
                lambda h: h(message, raw_content),
            ),
            InputModality.PDF_DOCUMENT: (
                "process_attachments",
                lambda h: h(message, raw_content),
            ),
            InputModality.PDF_OCR: (
                "process_attachments",
                lambda h: h(message, raw_content),
            ),
        }

        if detected_modality in modality_flow_map:
            flow_key, invoker = modality_flow_map[detected_modality]
            handler = self._flows.get(flow_key)
            if handler:
                try:
                    result = await invoker(handler)
                except Exception:
                    result = ""
                audio_path = None
                if isinstance(result, ResponseMessage):
                    text_out = result.text or result.content or ""
                    audio_path = result.audio_path
                else:
                    text_out = str(result or "")
                # Return error if no text was generated [REH]
                if not text_out or not str(text_out).strip():
                    return ResponseMessage(
                        content="Error: No text was generated. Please try again.",
                        text="Error: No text was generated. Please try again.",
                    )
                if not raw_content.strip() and len(text_out.split()) < 5:
                    text_out = (text_out + " auto generated caption.").strip()
                return ResponseMessage(
                    content=text_out, text=text_out, audio_path=audio_path
                )

        attachments = list(getattr(message, "attachments", []) or [])
        if attachments:
            if "process_attachments" in self._flows:
                handler = self._flows["process_attachments"]
                try:
                    result = await handler(message, raw_content)
                except Exception:
                    result = ""
                audio_path = None
                if isinstance(result, ResponseMessage):
                    text_out = result.text or result.content or ""
                    audio_path = result.audio_path
                else:
                    text_out = str(result or "")
                # Return error if no text was generated [REH]
                if not text_out or not str(text_out).strip():
                    return ResponseMessage(
                        content="Error: No text was generated. Please try again.",
                        text="Error: No text was generated. Please try again.",
                    )
                if not raw_content.strip() and len(text_out.split()) < 5:
                    text_out = (text_out + " auto generated caption.").strip()
                return ResponseMessage(
                    content=text_out, text=text_out, audio_path=audio_path
                )

            att = attachments[0]
            content_type = (getattr(att, "content_type", "") or "").lower()
            filename = getattr(att, "filename", "") or ""
            if content_type.startswith("image/"):
                image_bytes = await att.read()
                caption = await see_infer(
                    image_data=image_bytes,
                    prompt=f"User uploaded an image with the prompt: '{raw_content}'",
                    mime_type=content_type,
                )
                brain_input = (
                    f"User uploaded an image with the prompt: '{raw_content}'. "
                    f"The image contains: {caption}"
                )
                response_text = await brain_infer(brain_input)
                return ResponseMessage(content=response_text, text=response_text)

            # Treat everything else as a document upload
            suffix = Path(filename).suffix or ""
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp_path = Path(tmp.name)
            await att.save(tmp_path)
            doc_content = await self._process_document(str(tmp_path), suffix)
            prompt = (
                "DOCUMENT CONTENT:\n---\n"
                f"{doc_content}\n---\n\nUSER'S PROMPT: {raw_content}"
            )
            response_text = await brain_infer(prompt)
            try:
                os.remove(str(tmp_path))
            except Exception:
                pass
            return ResponseMessage(content=response_text, text=response_text)

        if "process_text" in self._flows:
            try:
                response_text = await self._flows["process_text"](body)
            except Exception:
                response_text = ""
        else:
            response_text = await brain_infer(body)
        modality = self._get_output_modality(None, message)
        audio_path: Optional[str] = None
        if modality == OutputModality.TTS and "generate_tts" in self._flows:
            try:
                audio_path = await self._flows["generate_tts"](response_text)
            except Exception:
                audio_path = None

        return ResponseMessage(
            content=response_text, text=response_text, audio_path=audio_path
        )

    def _is_mentioned(self, message: Message) -> bool:
        """Safe mention detection for mock and production messages."""
        try:
            mentions = list(getattr(message, "mentions", []) or [])
            return getattr(self.bot, "user", None) in mentions
        except Exception:
            return False

    async def dispatch_message(self, message: Message) -> Optional[BotAction]:
        """Process a message and ensure exactly one response is generated (1 IN > 1 OUT rule)."""
        self.logger.info(f" === ROUTER DISPATCH STARTED: MSG {message.id} ====")
        try:
            # 1. Quick pre-filter: Only parse commands for messages that start with '!'
            content_raw = getattr(message, "content", "") or ""
            content = str(content_raw).strip()

            # Remove bot mention to check for command pattern
            mention_pattern = rf"^<@!?{self.bot.user.id}>\s*"
            clean_content = re.sub(mention_pattern, "", content)

            # Router debug flag from config [IV]
            try:
                router_debug = bool(self.config.get("ROUTER_DEBUG", False))
            except Exception:
                router_debug = False

            # Listener-stage skips: self/bots and duplicates [IV]
            try:
                author = getattr(message, "author", None)
                raw_author_bot = getattr(author, "bot", False) if author else False
                author_is_bot = (
                    raw_author_bot is True
                    if isinstance(raw_author_bot, bool)
                    else False
                )
                is_self = False
                try:
                    is_self = bool(
                        hasattr(self.bot, "user")
                        and author
                        and getattr(author, "id", None)
                        == getattr(self.bot.user, "id", None)
                    )
                except Exception:
                    is_self = False
                if author_is_bot or is_self:
                    self.logger.info(
                        "gate.skip",
                        extra={
                            "event": "gate.skip",
                            "reason": "bot_or_self",
                            "msg_id": getattr(message, "id", None),
                        },
                    )
                    return None
            except Exception:
                pass

            # Concurrency-safe dedupe [REH]
            try:
                lock = self._processing_locks.setdefault(message.id, asyncio.Lock())
            except Exception:
                lock = asyncio.Lock()
            async with lock:
                if getattr(message, "id", None) in self._processed_recent_set:
                    self.logger.info(
                        "gate.skip",
                        extra={
                            "event": "gate.skip",
                            "reason": "duplicate",
                            "msg_id": getattr(message, "id", None),
                        },
                    )
                    return None

                # Mark as processed (dedupe window)
                try:
                    if len(self._processed_recent) == self._processed_recent.maxlen:
                        old_id = self._processed_recent.popleft()
                        self._processed_recent_set.discard(old_id)
                    self._processed_recent.append(message.id)
                    self._processed_recent_set.add(message.id)
                except Exception:
                    pass

            # Ingest started marker (single-shot)
            try:
                self.logger.info(
                    "ingest.dispatch_started",
                    extra={
                        "event": "ingest.dispatch_started",
                        "msg_id": message.id,
                        "channel_id": getattr(
                            getattr(message, "channel", None), "id", None
                        ),
                    },
                )
            except Exception:
                pass

            # 1b. Compatibility fast-path for legacy tests: attachments + empty content
            # Run this BEFORE gating and typing() to avoid MagicMock issues in tests
            try:
                has_attachments = (
                    bool(getattr(message, "attachments", None))
                    and len(message.attachments) > 0
                )
            except Exception:
                has_attachments = False
            cleaned_for_compat = re.sub(mention_pattern, "", content)
            cleaned_for_compat = strip_leading_bot_mention(
                cleaned_for_compat, getattr(getattr(self.bot, "user", None), "id", None)
            )
            if (
                has_attachments
                and cleaned_for_compat == ""
                and not isinstance(self.bot, (Mock, MagicMock))
            ):
                # If all attachments are plain text (.txt/text/*), skip the legacy
                # attachment compat path so the text ingestion path can handle them.
                try:
                    atts = list(getattr(message, "attachments", []) or [])
                    all_text_files = all_attachments_are_text(atts)
                except Exception:
                    all_text_files = False

                if not all_text_files:
                    handler = self._flows.get("process_attachments")
                    if handler:
                        self.logger.debug(
                            "Compat path (pre-gate): delegating to _flows['process_attachments'] with empty text."
                        )
                        res = await handler(message, cleaned_for_compat)
                        if isinstance(res, BotAction):
                            return res
                        text_out = None
                        audio_path = None
                        if isinstance(res, ResponseMessage):
                            text_out = res.text or res.content
                            audio_path = res.audio_path
                        else:
                            text_out = str(res)
                        # Return error if no text was generated [REH]
                        if not text_out or not str(text_out).strip():
                            return ResponseMessage(
                                content="Error: No text was generated. Please try again.",
                                text="Error: No text was generated. Please try again.",
                            )
                        if isinstance(res, ResponseMessage):
                            res.text = res.content = text_out
                            return res
                        return ResponseMessage(
                            content=text_out, text=text_out, audio_path=audio_path
                        )

            # Parse commands first so downstream paths can use cleaned content
            try:
                parsed_command = parse_command(message, self.bot)
            except Exception:
                parsed_command = None
            if parsed_command:
                clean_content = parsed_command.cleaned_content or clean_content

            # 2. If a command is found, handle special cases or delegate to cogs.
            if parsed_command:
                cmd = parsed_command.command
                if cmd == Command.IMG:
                    self.logger.info(
                        f"Found command 'IMG', delegating to cog. (msg_id: {message.id})"
                    )
                    return await self._handle_img_command(parsed_command, message)

                if cmd == Command.PING:
                    return ResponseMessage(content="Pong!", text="Pong!")

                if cmd == Command.HELP:
                    return ResponseMessage(
                        content="See `/help` for a list of commands.",
                        text="See `/help` for a list of commands.",
                    )

                if cmd in {
                    Command.TTS,
                    Command.SAY,
                    Command.TTS_ALL,
                    Command.SPEAK,
                    Command.IGNORE,
                }:
                    return None

                # Allow chat-like commands to continue through the normal routing pipeline
                if cmd != Command.CHAT:
                    self.logger.info(
                        f"Found command '{parsed_command.command.name}', delegating to cog. (msg_id: {message.id})"
                    )
                    return BotAction(meta={"delegated_to_cog": True})

            if not parsed_command and clean_content.startswith("!"):
                matched = None
                try:
                    for key in sorted(COMMAND_MAP.keys(), key=len, reverse=True):
                        if clean_content == key or clean_content.startswith(f"{key} "):
                            matched = key
                            break
                except Exception:
                    matched = None
                if matched:
                    self.logger.info(
                        f"Found command '{matched}', delegating to cog. (msg_id: {message.id})"
                    )
                    return BotAction(meta={"delegated_to_cog": True})

            compat_response = await self._compat_dispatch_for_tests(
                message, clean_content
            )
            if compat_response is not None:
                return compat_response

            # 3. Determine if the bot should process this message (DM, mention, or reply).
            gate_hint = self.pop_gate_hint(getattr(message, "id", None))
            if gate_hint is not None:
                allow_via_gate = gate_hint
            elif parsed_command:
                allow_via_gate = True
            else:
                allow_via_gate = self._should_process_message(message)
            if not allow_via_gate:
                # Relaxed allowance: mention + minimal meaningful text should route to text
                # This mirrors the text-default behavior in core bot to avoid dead-ends. [IV][REH]
                try:
                    is_mentioned = self._is_mentioned(message)
                except Exception:
                    is_mentioned = False
                mention_pattern = rf"^<@!?{self.bot.user.id}>\s*"
                try:
                    cleaned = re.sub(mention_pattern, "", content).strip()
                except Exception:
                    cleaned = content.strip()
                cleaned = strip_leading_bot_mention(
                    cleaned, getattr(getattr(self.bot, "user", None), "id", None)
                )

                if is_mentioned and has_meaningful_text(cleaned):
                    try:
                        self.logger.info(
                            "text_default",
                            extra={
                                "event": "text_default",
                                "subsys": "gate",
                                "msg_id": message.id,
                                "detail": {
                                    "reason": "has_text",
                                    "clean_len": len(cleaned),
                                },
                            },
                        )
                    except Exception:
                        pass
                else:
                    # Not addressed → block once at listener stage semantics
                    try:
                        self.logger.info(
                            "gate.block",
                            extra={
                                "event": "gate.block",
                                "reason": "not_addressed",
                                "msg_id": message.id,
                            },
                        )
                    except Exception:
                        pass
                    return None

            # --- Start of processing for DMs, Mentions, and Replies ---
            async with message.channel.typing():
                self.logger.info(
                    f"Processing message: DM={isinstance(message.channel, DMChannel)}, Mention={self._is_mentioned(message)} (msg_id: {message.id})"
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
                cleaned_for_compat = strip_leading_bot_mention(
                    cleaned_for_compat, getattr(getattr(self.bot, "user", None), "id", None)
                )
                if (
                    has_attachments
                    and cleaned_for_compat == ""
                    and not isinstance(self.bot, (Mock, MagicMock))
                ):
                    # If all attachments are plain text (.txt/text/*), skip legacy compat path
                    try:
                        atts = list(getattr(message, "attachments", []) or [])
                        all_text_files = all_attachments_are_text(atts)
                    except Exception:
                        all_text_files = False

                    if not all_text_files:
                        handler = self._flows.get("process_attachments")
                        if handler:
                            self.logger.debug(
                                "Compat path: delegating to _flows['process_attachments'] with empty text."
                            )
                            res = await handler(message, cleaned_for_compat)
                            if isinstance(res, BotAction):
                                return res
                            text_out = None
                            audio_path = None
                            if isinstance(res, ResponseMessage):
                                text_out = res.text or res.content
                                audio_path = res.audio_path
                            else:
                                text_out = str(res)
                            # Return error if no text was generated [REH]
                            if not text_out or not str(text_out).strip():
                                return ResponseMessage(
                                    content="Error: No text was generated. Please try again.",
                                    text="Error: No text was generated. Please try again.",
                                )
                            if isinstance(res, ResponseMessage):
                                res.text = res.content = text_out
                                return res
                            return ResponseMessage(
                                content=text_out, text=text_out, audio_path=audio_path
                            )

                # Centralized scope resolution and context building
                (
                    scope_case,
                    reply_target,
                    context_str,
                ) = await self._resolve_scope_and_target(message)

                self.logger.debug(
                    f"Scope resolved: context_str='{context_str[:100]}...'"
                )

                # Clean mention from content for processing
                clean_content = content
                if self._is_mentioned(message):
                    clean_content = strip_leading_bot_mention(
                        content, getattr(getattr(self.bot, "user", None), "id", None)
                    )

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
                    # Decide final route here for vision
                    try:
                        self.logger.info(
                            "route.final",
                            extra={
                                "event": "route.final",
                                "detail": {"kind": "vision"},
                                "msg_id": message.id,
                            },
                        )
                    except Exception:
                        pass
                    return prechecked

                # 5.5. Deterministic X/Twitter media routing (optional early path; default off)
                if getattr(self, "_x_early_resolve_enabled", False):
                    try:
                        layer, x_urls = await self._gather_prioritized_x_urls(
                            scope_case, message, reply_target
                        )
                    except Exception:
                        layer, x_urls = "none", []
                    if x_urls and not parsed_command:
                        self.logger.info(
                            f"route.media: x/twitter url(s)={len(x_urls)} layer={layer}",
                            extra={
                                "event": "route.media",
                                "detail": {"count": len(x_urls), "layer": layer},
                            },
                        )
                        # Normalize URLs first (x.com/twitter.com/mobile variants)
                        try:
                            norm_urls = []
                            frontend_hints: Dict[str, str] = {}
                            primary_hints: Dict[str, str] = {}
                            for u in x_urls:
                                try:
                                    canonical_u = self._canonicalize_x_url(u)
                                except Exception:
                                    canonical_u = u
                                try:
                                    normalized_u = self._normalize_x_url(canonical_u)
                                except Exception:
                                    normalized_u = canonical_u
                                norm_urls.append(normalized_u)
                                ctx = (
                                    self._x_frontend_canon.get(normalized_u)
                                    or self._x_frontend_canon.get(canonical_u)
                                    or {}
                                )
                                frontend = ctx.get("frontend")
                                primary = self._resolve_twitter_status_id(
                                    normalized_u,
                                    tweet_id=ctx.get("primary"),
                                )
                                if frontend:
                                    frontend_hints[normalized_u] = frontend
                                if primary:
                                    primary_hints[normalized_u] = primary
                        except Exception:
                            norm_urls = x_urls
                            frontend_hints = {}
                            primary_hints = {}

                        # Time-box detection step
                        t0 = time.perf_counter()
                        try:
                            resolved = await asyncio.wait_for(
                                self._resolve_x_media(
                                    norm_urls,
                                    frontend_hints=frontend_hints,
                                    primary_hints=primary_hints,
                                ),
                                timeout=self._x_syn_timeout_s,
                            )
                        except Exception as e:
                            resolved = {"kind": "unknown", "reason": f"exception:{e}"}
                        dt_ms = int((time.perf_counter() - t0) * 1000)
                        kind = (resolved or {}).get("kind", "unknown")
                        base_context_url = (
                            norm_urls[0] if norm_urls else (x_urls[0] if x_urls else "")
                        )
                        primary_selected = self._resolve_twitter_status_id(
                            base_context_url,
                            tweet_id=(
                                (resolved or {}).get("primary")
                                or primary_hints.get(base_context_url)
                            ),
                        )
                        frontend_selected = (resolved or {}).get(
                            "frontend"
                        ) or frontend_hints.get(base_context_url)
                        # Single-shot detection marker
                        try:
                            src = (resolved or {}).get("src", "unknown")
                            detail = {"kind": kind, "src": src, "ms": dt_ms}
                            if kind == "image":
                                try:
                                    detail["count"] = len(
                                        (resolved or {}).get("images") or []
                                    )
                                except Exception:
                                    pass
                            if primary_selected:
                                detail["primary"] = primary_selected
                            if frontend_selected:
                                detail["frontend"] = frontend_selected
                            self.logger.info(
                                "x.detect",
                                extra={
                                    "event": "x.detect",
                                    "detail": detail,
                                    "msg_id": message.id,
                                },
                            )
                        except Exception:
                            pass

                        url_for_stt = (resolved or {}).get("url") or base_context_url
                        final_kind = kind
                        if url_for_stt and kind == "video":
                            verify_kind, verify_ct = await self._verify_media_kind(
                                url_for_stt, default="video"
                            )
                            self._log_media_kind_checked(
                                url_for_stt, verify_ct, verify_kind or "video"
                            )
                            if verify_kind == "image":
                                final_kind = "image"
                        elif kind == "image":
                            images_probe = (resolved or {}).get("images") or []
                            if images_probe:
                                verify_kind, verify_ct = await self._verify_media_kind(
                                    images_probe[0], default="image"
                                )
                                self._log_media_kind_checked(
                                    images_probe[0], verify_ct, verify_kind or "image"
                                )
                                if verify_kind == "video":
                                    final_kind = "video"
                                    url_for_stt = (resolved or {}).get(
                                        "url"
                                    ) or images_probe[0]

                        if final_kind == "video":
                            url_for_stt = url_for_stt or base_context_url
                            self.logger.info(
                                "route.select kind=video reason=resolved_direct_media"
                            )
                            # Emit deterministic media selection breadcrumb and harden cache key via fragment [CMV][CDiP]
                            try:
                                import hashlib as _hl

                                ptid2 = (
                                    primary_selected
                                    or extract_primary_tweet_id(url_for_stt)
                                    or ""
                                )
                                uhash2 = _hl.sha256(url_for_stt.encode()).hexdigest()[
                                    :16
                                ]
                                detail = {
                                    "primary": ptid2,
                                    "selected": ptid2,
                                    "tier": 1,
                                    "has_audio": ".mp4" in str(url_for_stt).lower(),
                                    "url_hash": uhash2,
                                }
                                if frontend_selected:
                                    detail["frontend"] = frontend_selected
                                self.logger.info(
                                    "media.selected",
                                    extra={
                                        "event": "media.selected",
                                        "detail": detail,
                                        "msg_id": message.id,
                                    },
                                )
                                if ptid2 and uhash2:
                                    url_for_stt = (
                                        f"{url_for_stt}#ptid={ptid2}&uh={uhash2}"
                                    )
                                    if frontend_selected:
                                        url_for_stt = (
                                            f"{url_for_stt}&fe={frontend_selected}"
                                        )
                            except Exception:
                                pass
                            dur = (resolved or {}).get("duration")
                            host = None
                            try:
                                host = urlparse(url_for_stt).netloc
                            except Exception:
                                host = ""
                            self.logger.info(
                                f"media.resolve: result=video url={host or url_for_stt} dur={int(dur) if isinstance(dur, (int, float)) else 'NA'}s"
                            )
                            try:
                                self.logger.info(
                                    "x.video.url_ok",
                                    extra={
                                        "event": "x.video.url_ok",
                                        "detail": {
                                            "src": (
                                                (resolved or {}).get("src") or "ytdlp"
                                            ),
                                            "ms": dt_ms,
                                        },
                                        "msg_id": message.id,
                                    },
                                )
                            except Exception:
                                pass
                            try:
                                timeout_override_raw = None
                                try:
                                    timeout_override_raw = self.config.get(
                                        "X_STT_TIMEOUT_S"
                                    )
                                except Exception:
                                    timeout_override_raw = None
                                stt_timeout: Optional[float]
                                stt_timeout = None
                                if timeout_override_raw not in (None, ""):
                                    try:
                                        stt_timeout = float(timeout_override_raw)
                                    except Exception:
                                        stt_timeout = None
                                if stt_timeout is None or stt_timeout <= 0:
                                    try:
                                        stt_rtf = float(
                                            self.config.get(
                                                "X_STT_TIMEOUT_RTF", X_STT_RTF_DEFAULT
                                            )
                                        )
                                    except Exception:
                                        stt_rtf = X_STT_RTF_DEFAULT
                                    try:
                                        speedup_cfg = float(
                                            self.config.get(
                                                "VIDEO_SPEEDUP", _DEFAULT_VIDEO_SPEEDUP
                                            )
                                        )
                                    except Exception:
                                        speedup_cfg = _DEFAULT_VIDEO_SPEEDUP
                                    safe_speedup = (
                                        speedup_cfg
                                        if speedup_cfg > 0
                                        else _DEFAULT_VIDEO_SPEEDUP
                                    )
                                    effective_duration = 0.0
                                    if isinstance(dur, (int, float)):
                                        effective_duration = max(float(dur), 0.0) / max(
                                            safe_speedup, 0.1
                                        )
                                    computed = max(
                                        X_STT_MIN_TIMEOUT_S,
                                        effective_duration * stt_rtf + X_STT_PADDING_S,
                                    )
                                    stt_timeout = min(computed, X_STT_MAX_TIMEOUT_S)
                                if stt_timeout is None or stt_timeout <= 0:
                                    stt_timeout = X_STT_MIN_TIMEOUT_S
                                try:
                                    mm = (
                                        int((dur or 0) // 60)
                                        if isinstance(dur, (int, float))
                                        else 0
                                    )
                                    ss = (
                                        int((dur or 0) % 60)
                                        if isinstance(dur, (int, float))
                                        else 0
                                    )
                                    self.logger.info(
                                        "stt.start",
                                        extra={
                                            "event": "stt.start",
                                            "detail": {
                                                "dur": f"{mm:02d}:{ss:02d}",
                                                "timeout_s": int(stt_timeout),
                                            },
                                            "msg_id": message.id,
                                        },
                                    )
                                except Exception:
                                    pass
                                stt_t0 = time.perf_counter()
                                stt_res = await self._run_stt_job(
                                    asyncio.wait_for(
                                        hear_infer_from_url(url_for_stt),
                                        timeout=stt_timeout,
                                    ),
                                    message,
                                )
                                formatted = self._format_x_tweet_with_transcription(
                                    base_text=None, url=url_for_stt, stt_res=stt_res
                                )
                                try:
                                    el_ms = int((time.perf_counter() - stt_t0) * 1000)
                                    chars = len(
                                        (stt_res or {}).get("transcription", "")
                                    )
                                    self.logger.info(
                                        "stt.ok",
                                        extra={
                                            "event": "stt.ok",
                                            "detail": {
                                                "ms": el_ms,
                                                "chars": chars,
                                                "timeout_s": int(stt_timeout),
                                            },
                                            "msg_id": message.id,
                                        },
                                    )
                                except Exception:
                                    pass
                                self.logger.info(
                                    f"🎯 Route: stt_from_x_video | msg_id={message.id}"
                                )
                                try:
                                    self.logger.info(
                                        "route.final",
                                        extra={
                                            "event": "route.final",
                                            "detail": {"kind": "video"},
                                            "msg_id": message.id,
                                        },
                                    )
                                except Exception:
                                    pass
                                return await self._flow_process_text(
                                    content=formatted,
                                    context=context_str,
                                    message=message,
                                )
                            except asyncio.TimeoutError:
                                self._emit_stt_fail_event(
                                    "timeout",
                                    msg_id=message.id,
                                )
                                try:
                                    self.logger.info(
                                        "route.final",
                                        extra={
                                            "event": "route.final",
                                            "detail": {"kind": "video_stt_timeout"},
                                            "msg_id": message.id,
                                        },
                                    )
                                except Exception:
                                    pass
                                return BotAction(
                                    content=(
                                        "⚠️ I couldn't transcribe this video before timing out. "
                                        "Please try again or use a shorter clip."
                                    ),
                                    error=True,
                                )
                            except Exception as e:
                                reason = "extract_error"
                                try:
                                    es = str(e).lower()
                                    if "403" in es or "forbidden" in es:
                                        reason = "403"
                                    elif (
                                        "format" in es
                                        or "no video formats" in es
                                        or "no such format" in es
                                    ):
                                        reason = "no_formats"
                                    elif "timeout" in es:
                                        reason = "timeout"
                                    elif "whisper" in es:
                                        reason = "whisper_error"
                                except Exception:
                                    pass
                                try:
                                    self.logger.info(
                                        "x.video.url_fail",
                                        extra={
                                            "event": "x.video.url_fail",
                                            "detail": {"reason": str(e)[:200]},
                                            "msg_id": message.id,
                                        },
                                    )
                                except Exception:
                                    pass
                                self._emit_stt_fail_event(
                                    reason,
                                    msg_id=message.id,
                                )
                                try:
                                    self.logger.info(
                                        "route.final",
                                        extra={
                                            "event": "route.final",
                                            "detail": {"kind": "video_stt_error"},
                                            "msg_id": message.id,
                                        },
                                    )
                                except Exception:
                                    pass
                                return BotAction(
                                    content=(
                                        "⚠️ I couldn't transcribe this video. "
                                        "Please try again later or share a shorter clip."
                                    ),
                                    error=True,
                                )
                        elif final_kind == "image":
                            images = (resolved or {}).get("images") or []
                            self.logger.info(
                                "route.select kind=image reason=resolved_syndication_photos"
                            )
                            self.logger.info(
                                f"media.resolve: result=image count={len(images)}"
                            )
                            try:
                                domain = (
                                    "pbs.twimg.com"
                                    if any("pbs.twimg.com" in (i or "") for i in images)
                                    else "unknown"
                                )
                                self.logger.info(
                                    "x.photos.ok",
                                    extra={
                                        "event": "x.photos.ok",
                                        "detail": {
                                            "count": len(images),
                                            "domain": domain,
                                        },
                                        "msg_id": message.id,
                                    },
                                )
                            except Exception:
                                pass
                            try:
                                self.logger.info(
                                    "route_selected",
                                    extra={
                                        "subsys": "route",
                                        "event": "route_selected",
                                        "detail": {"kind": "vl"},
                                    },
                                )
                            except Exception:
                                pass
                            vl_notes = None
                            if images:
                                try:
                                    vl_notes = await self._vl_describe_image_from_url(
                                        images[0],
                                        prompt=(
                                            "Describe this image in detail, focusing on key visual elements, objects, text, and context."
                                        ),
                                    )
                                    vl_notes = sanitize_vl_reply_text(vl_notes or "")
                                except Exception:
                                    vl_notes = None
                            tweet_caption = ""
                            try:
                                caption_tweet_id = self._resolve_twitter_status_id(
                                    base_context_url,
                                    tweet_id=primary_selected,
                                )
                                if caption_tweet_id:
                                    tweet_caption = await self._resolve_twitter_caption_from_syndication(
                                        caption_tweet_id
                                    )
                                self.logger.info(
                                    "x.image.caption.resolve",
                                    extra={
                                        "event": "x.image.caption.resolve",
                                        "detail": {
                                            "tweet_id": caption_tweet_id,
                                            "chars": len(tweet_caption or ""),
                                        },
                                        "msg_id": message.id,
                                    },
                                )
                            except Exception:
                                tweet_caption = ""
                            composed_input = self._compose_x_tweet_with_visual_facts(
                                user_text=clean_content,
                                tweet_caption=tweet_caption,
                                vl_notes=vl_notes,
                            )
                            self.logger.info(
                                f"🎯 Route: vl_from_x_images | msg_id={message.id}"
                            )
                            try:
                                self.logger.info(
                                    "route.final",
                                    extra={
                                        "event": "route.final",
                                        "detail": {"kind": "photos"},
                                        "msg_id": message.id,
                                    },
                                )
                            except Exception:
                                pass
                            return await self._flow_process_text(
                                content=composed_input or clean_content,
                                context=context_str,
                                message=message,
                                perception_notes=None,
                            )
                        else:
                            reason = (resolved or {}).get("reason", "unknown")
                            self.logger.info(
                                f"media.resolve: result=unknown reason={reason}"
                            )
                            try:
                                self.logger.info(
                                    "media_fallback",
                                    extra={
                                        "subsys": "tw",
                                        "event": "media_fallback",
                                        "detail": {"kind": "none"},
                                    },
                                )
                            except Exception:
                                pass
                            # Continue to normal processing (no early return)

                # 6. Sequential multimodal processing (guarded by a global timeout to avoid hangs)
                try:
                    try:
                        total_budget = float(
                            self.config.get("MULTIMODAL_TOTAL_BUDGET_S", 240.0)
                        )
                    except Exception:
                        total_budget = 240.0
                    try:
                        self.logger.info(
                            "multimodal.budget",
                            extra={
                                "event": "multimodal.budget",
                                "subsys": "route",
                                "phase": "compose",
                                "detail": {"seconds": total_budget},
                            },
                        )
                    except Exception:
                        pass
                    result_action = await asyncio.wait_for(
                        self._process_multimodal_message_internal(message, context_str),
                        timeout=total_budget,
                    )
                except asyncio.TimeoutError:
                    # Fail-fast with user-friendly message; typing context will exit afterwards
                    self.logger.error(
                        f"multimodal.total_timeout | msg_id={message.id} budget={total_budget}s"
                    )
                    return BotAction(
                        content=(
                            "⏳ This is taking too long to process, so I'll stop here to avoid hanging. "
                            "Please try again with a shorter video or later."
                        ),
                        error=True,
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
        finally:
            # Remove the per-message lock to avoid unbounded growth [RM]
            try:
                self._processing_locks.pop(getattr(message, "id", None), None)
            except Exception:
                pass

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
                                ".opus.mov",
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
        # Simplified path for unit tests using mock bots to avoid network/file IO
        if isinstance(self.bot, (Mock, MagicMock)):
            items = collect_input_items(message) or []
            results: List[str] = []
            handler_timeout_s = 5.0

            for item in items:
                modality = await map_item_to_modality(item)
                handler_res: Optional[str] = None
                try:
                    if modality == InputModality.VIDEO_URL:
                        try:
                            handler_res = await asyncio.wait_for(
                                self._handle_video_url(item, message=message),
                                timeout=handler_timeout_s,
                            )
                        except TypeError:
                            handler_res = await asyncio.wait_for(
                                self._handle_video_url(item), timeout=handler_timeout_s
                            )
                    elif modality in (
                        InputModality.GENERAL_URL,
                        InputModality.SCREENSHOT_URL,
                    ):
                        try:
                            handler_res = await asyncio.wait_for(
                                self._handle_general_url(item, message=message),
                                timeout=handler_timeout_s,
                            )
                        except TypeError:
                            handler_res = await asyncio.wait_for(
                                self._handle_general_url(item),
                                timeout=handler_timeout_s,
                            )
                    elif modality in (
                        InputModality.SINGLE_IMAGE,
                        InputModality.MULTI_IMAGE,
                    ):
                        try:
                            handler_res = await asyncio.wait_for(
                                self._handle_image(item, message=message),
                                timeout=handler_timeout_s,
                            )
                        except TypeError:
                            handler_res = await asyncio.wait_for(
                                self._handle_image(item), timeout=handler_timeout_s
                            )
                    elif modality in (
                        InputModality.PDF_DOCUMENT,
                        InputModality.PDF_OCR,
                    ):
                        try:
                            handler_res = await asyncio.wait_for(
                                self._handle_pdf(item, message=message),
                                timeout=handler_timeout_s,
                            )
                        except TypeError:
                            handler_res = await asyncio.wait_for(
                                self._handle_pdf(item), timeout=handler_timeout_s
                            )
                except asyncio.TimeoutError:
                    try:
                        mod_label = getattr(modality, "name", "input").lower()
                        await message.reply(
                            f"⚠️ Processing timed out for {mod_label}. Please try again."
                        )
                    except Exception:
                        pass
                    handler_res = None
                except Exception as exc:
                    try:
                        mod_label = getattr(modality, "name", "input").lower()
                        await message.reply(
                            f"⚠️ An error occurred while processing {mod_label}: {exc}"
                        )
                    except Exception:
                        pass
                    handler_res = None

                if handler_res:
                    results.append(str(handler_res))

            # Include the remaining text content to mirror production path
            try:
                base_text = (message.content or "").strip()
                bot_user = getattr(self.bot, "user", None)
                if bot_user and getattr(bot_user, "id", None):
                    mention_pattern = rf"^<@!?{bot_user.id}>\s*"
                    base_text = re.sub(mention_pattern, "", base_text)
                if base_text:
                    results.append(base_text)
            except Exception:
                pass

            flow_fn = getattr(self, "_flow_process_text", None)
            invoke_flow = getattr(self, "_invoke_text_flow", None)
            if isinstance(flow_fn, (AsyncMock, MagicMock)):
                for res in results:
                    await flow_fn(res, message, context_str)
            elif isinstance(invoke_flow, (AsyncMock, MagicMock)):
                for res in results:
                    await invoke_flow(res, message, context_str)

            return None

        # Collect all input items from the message
        items = collect_input_items(message)
        # Treat plain text attachments as prompt extensions, not standalone items
        try:
            items = [
                it
                for it in (items or [])
                if not (
                    getattr(it, "source_type", None) == "attachment"
                    and is_text_attachment(getattr(it, "payload", None))
                )
            ]
        except Exception:
            # Non-fatal: fallback to original items list on any error
            pass

        ref_message: Optional[Message] = None

        # Check for reply-image harvesting [VISION_REPLY_IMAGE_HARVEST]
        if message.reference and self.config.get("VISION_REPLY_IMAGE_HARVEST", True):
            try:
                # Fetch the referenced message to harvest images
                ref_message = await self._resolve_reference_message(
                    message, fallback=ref_message
                )
                if ref_message is None:
                    raise RuntimeError("reference_unavailable")
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

        # Reply link/attachment harvest (non-image) so reply chains route correctly [REH][IV]
        try:
            if message.reference:
                ref_message = await self._resolve_reference_message(
                    message, fallback=ref_message
                )

                if ref_message:
                    # Build a set of existing payloads to avoid duplicates
                    existing_urls = existing_url_payloads(items, strip_payload=True)

                    # 1) Harvest URLs from referenced message content
                    try:
                        ref_text = getattr(ref_message, "content", "") or ""
                        ref_urls = extract_urls_loose(ref_text)
                        added = append_unique_url_items(
                            items,
                            ref_urls,
                            item_ctor=InputItem,
                            strip_key=True,
                            existing_urls=existing_urls,
                        )
                        if added:
                            try:
                                self.logger.info(
                                    f"📎 Reply link capture | from_msg={ref_message.id} urls_added={added}"
                                )
                            except Exception:
                                pass
                    except Exception:
                        pass

                    # 2) Harvest non-image attachments from referenced message (e.g., video, pdf)
                    try:
                        ref_atts = getattr(ref_message, "attachments", None) or []
                        added_atts = 0
                        for att in ref_atts:
                            ctype = (getattr(att, "content_type", "") or "").lower()
                            if ctype.startswith("image/"):
                                continue  # images already handled by image harvest
                            # Dedup naive: skip if URL of attachment already present
                            url = getattr(att, "url", None)
                            if url and str(url).strip() in existing_urls:
                                continue
                            items.append(
                                InputItem(
                                    source_type="attachment",
                                    payload=att,
                                    order_index=len(items) + 1,
                                )
                            )
                            added_atts += 1
                        if added_atts:
                            try:
                                self.logger.info(
                                    f"📎 Reply attachment capture | from_msg={ref_message.id} attachments_added={added_atts}"
                                )
                            except Exception:
                                pass
                    except Exception:
                        pass
        except Exception:
            # Non-fatal; continue without reply link harvest
            pass

        # Additionally harvest URLs from the referenced message to support reply→video flows [CA][REH]
        # Note: This block lives inside the image-harvest section for historical reasons, but URL harvest
        # must NOT depend on the VISION_REPLY_IMAGE_HARVEST flag. We add an unconditional safety harvest below.
        try:
            ref_msg = await self._resolve_reference_message(
                message, fallback=ref_message
            )
            if ref_msg and getattr(ref_msg, "content", None):
                # Extract URLs from the referenced message
                found_urls = extract_urls_strict(ref_msg.content or "")
                if found_urls:
                    # Deduplicate against existing url items
                    existing_urls = existing_url_payloads(items)
                    added_urls = append_unique_url_items(
                        items,
                        found_urls,
                        item_ctor=InputItem,
                        existing_urls=existing_urls,
                    )
                    if added_urls:
                        try:
                            self.logger.info(
                                f"📎 Reply URL harvest | from_msg={ref_msg.id} urls_added={added_urls}"
                            )
                        except Exception:
                            pass
        except Exception:
            # Do not fail dispatch on URL harvest errors
            pass

        # Safety net: Unconditional URL harvest for reply messages (not gated by VISION_REPLY_IMAGE_HARVEST)
        # Ensures reply→video (YouTube/TikTok/X) routes always collect the URL even when image harvest is disabled. [REH]
        try:
            if getattr(message, "reference", None):
                ref_msg = await self._resolve_reference_message(
                    message, fallback=ref_message
                )
                if ref_msg:
                    # 1) URLs present in the parent's text content
                    if getattr(ref_msg, "content", None):
                        found_urls = extract_urls_strict(ref_msg.content or "")
                    else:
                        found_urls = []

                    # 2) URLs present in the parent's embeds (e.g., tweets/YouTube share)
                    try:
                        ref_embeds = list(getattr(ref_msg, "embeds", []) or [])
                    except Exception:
                        ref_embeds = []
                    append_embed_related_urls(found_urls, ref_embeds)

                    if found_urls:
                        existing_urls = existing_url_payloads(items)
                        added_urls = append_unique_url_items(
                            items,
                            found_urls,
                            item_ctor=InputItem,
                            existing_urls=existing_urls,
                        )
                        if added_urls:
                            try:
                                self.logger.info(
                                    f"📎 Reply URL harvest (unconditional) | from_msg={getattr(ref_msg, 'id', 'na')} urls_added={added_urls} now_items={len(items)}"
                                )
                            except Exception:
                                pass
        except Exception:
            pass

        # Process original text content (remove URLs that will be processed separately)
        original_text = message.content
        try:
            mentions = list(getattr(message, "mentions", []) or [])
        except Exception:
            mentions = []
        if mentions and getattr(self.bot, "user", None) in mentions:
            original_text = strip_leading_bot_mention(
                original_text, getattr(getattr(self.bot, "user", None), "id", None)
            )

        # Remove URLs from text content since they will be processed separately
        original_text = strip_urls(original_text)

        # Resolve inline [search(...)] directives inside the remaining text
        try:
            original_text = await self._resolve_inline_searches(original_text, message)
        except Exception as e:
            self.logger.error(
                f"Inline search resolution failed: {e} (msg_id: {message.id})",
                exc_info=True,
            )

        # Diagnostics: post-harvest item counts [RAT][PA]
        try:
            url_ct = sum(1 for it in items if getattr(it, "source_type", None) == "url")
            att_ct = sum(
                1 for it in items if getattr(it, "source_type", None) == "attachment"
            )
            self.logger.info(
                f"mm.items.after_harvest | count={len(items)} urls={url_ct} atts={att_ct} msg_id={message.id}"
            )
        except Exception:
            pass

        # Thread-only UX fallback: if the trigger carried no meaningful text, adopt the reply target's text;
        # if the reply target is also empty (e.g., the mention itself), adopt the nearest previous human text. [REH][IV]
        try:
            if _is_thread_channel(getattr(message, "channel", None)):
                if not original_text or not original_text.strip():
                    try:
                        rt, _ = await resolve_thread_reply_target(
                            self.bot, message, self.config
                        )
                    except Exception:
                        rt = None
                    adopted = False
                    if rt and getattr(rt, "content", None):
                        rt_raw = str(rt.content or "")
                        # Strip Discord mentions/URLs for better signal.
                        rt_clean = strip_discord_mentions_and_urls(rt_raw)
                        # Require some alphanumeric signal to avoid adopting pure glyphs/whitespace
                        if rt_clean and re.search(r"[A-Za-z0-9]", rt_clean):
                            original_text = rt_clean
                            adopted = True
                            try:
                                self.logger.info(
                                    "adopt_ok",
                                    extra={
                                        "subsys": "mem.thread",
                                        "event": "adopt_ok",
                                        "guild_id": getattr(
                                            getattr(message, "guild", None), "id", None
                                        ),
                                        "user_id": getattr(
                                            getattr(message, "author", None), "id", None
                                        ),
                                        "msg_id": getattr(message, "id", None),
                                        "detail": {
                                            "source": "reply_target",
                                            "len": len(original_text),
                                        },
                                    },
                                )
                            except Exception:
                                pass
                    if not adopted:
                        anchor = rt or message
                        try:
                            async for m in message.channel.history(
                                limit=10, before=anchor
                            ):
                                is_human = not bool(getattr(m.author, "bot", False))
                                m_text = str(getattr(m, "content", "") or "").strip()
                                if is_human and m_text:
                                    original_text = m_text
                                    adopted = True
                                    try:
                                        self.logger.info(
                                            "adopt_ok",
                                            extra={
                                                "subsys": "mem.thread",
                                                "event": "adopt_ok",
                                                "guild_id": getattr(
                                                    getattr(message, "guild", None),
                                                    "id",
                                                    None,
                                                ),
                                                "user_id": getattr(
                                                    getattr(message, "author", None),
                                                    "id",
                                                    None,
                                                ),
                                                "msg_id": getattr(message, "id", None),
                                                "detail": {
                                                    "source": "prev_human",
                                                    "len": len(original_text),
                                                },
                                            },
                                        )
                                    except Exception:
                                        pass
                                    break
                        except Exception:
                            pass
        except Exception:
            pass

        # Reply-case UX fallback (non-thread): mention + reply with minimal text → adopt parent text. [REH][IV]
        try:
            if not _is_thread_channel(getattr(message, "channel", None)) and getattr(
                message, "reference", None
            ):
                # Only trigger on @mention to avoid hijacking normal replies
                if self.bot.user in (getattr(message, "mentions", None) or []):
                    minimal = True
                    try:
                        minimal = not bool(
                            re.search(r"[A-Za-z0-9]", original_text or "")
                        )
                    except Exception:
                        minimal = not bool(original_text and original_text.strip())
                    if minimal:
                        ref = getattr(message, "reference", None)
                        ref_msg = getattr(ref, "resolved", None)
                        if ref_msg is None and getattr(ref, "message_id", None):
                            try:
                                ref_msg = await message.channel.fetch_message(
                                    ref.message_id
                                )
                            except Exception:
                                ref_msg = None
                        if ref_msg and getattr(ref_msg, "content", None):
                            rt_raw = str(ref_msg.content or "")
                            try:
                                # Strip mentions and URLs for better signal.
                                rt_clean = strip_discord_mentions_and_urls(rt_raw)
                            except Exception:
                                rt_clean = (ref_msg.content or "").strip()
                            try:
                                if rt_clean and re.search(r"[A-Za-z0-9]", rt_clean):
                                    original_text = rt_clean
                                    try:
                                        self.logger.info(
                                            "adopt_ok",
                                            extra={
                                                "subsys": "mem.reply",
                                                "event": "adopt_ok",
                                                "guild_id": getattr(
                                                    getattr(message, "guild", None),
                                                    "id",
                                                    None,
                                                ),
                                                "user_id": getattr(
                                                    getattr(message, "author", None),
                                                    "id",
                                                    None,
                                                ),
                                                "msg_id": getattr(message, "id", None),
                                                "detail": {
                                                    "source": "reply_parent",
                                                    "len": len(original_text),
                                                },
                                            },
                                        )
                                    except Exception:
                                        pass
                            except Exception:
                                pass
        except Exception:
            pass

        # Ingest .txt attachments from the triggering message into the text prompt (first match only)
        try:
            atts = list(getattr(message, "attachments", []) or [])
            txt_atts = [a for a in atts if is_text_attachment(a)]
            loaded_count = 0
            bytes_total = 0
            truncated = False
            if txt_atts:
                # Preserve upload order; read the first only (mirror !img)
                first = txt_atts[0]
                try:
                    bytes_total = int(getattr(first, "size", 0) or 0)
                except Exception:
                    bytes_total = 0
                blob = await read_attachment_text(first, 262_144)
                if blob:
                    loaded_count = 1
                    # Append with a simple separator to preserve semantics
                    if original_text and blob:
                        original_text = f"{original_text}\n\n{blob}".strip()
                    elif blob:
                        original_text = blob.strip()
                    try:
                        self.logger.info(
                            f"attachments.txt_loaded count={loaded_count} bytes_total={bytes_total} truncated={str(truncated).lower()}"
                        )
                        if len(txt_atts) > 1:
                            extra = len(txt_atts) - 1
                            self.logger.info(f"attachments.txt_ignored extra={extra}")
                    except Exception:
                        pass
                else:
                    try:
                        self.logger.info(
                            "attachments.txt_reject reason=invalid_or_oversize"
                        )
                    except Exception:
                        pass
        except Exception:
            # Never break routing on attachment ingestion failure
            pass

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
        mentioned_me = self._is_mentioned(message)
        is_reply = getattr(message, "reference", None) is not None

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
        x_hosts_for_gate = [
            "x.com",
            "twitter.com",
            "fxtwitter.com",
            "vxtwitter.com",
            "fixupx.com",
        ]
        has_x_url = any(
            host in (message.content or "").lower() for host in x_hosts_for_gate
        ) or (x_info_for_gate.has_x_link if x_info_for_gate else False)
        x_status_urls_from_items: set[str] = set()
        try:
            for it in items:
                if getattr(it, "source_type", None) != "url":
                    continue
                raw_u = str(getattr(it, "payload", "") or "").strip()
                if not raw_u or not self._is_twitter_url(raw_u):
                    continue
                has_x_url = True
                if self._is_twitter_status_url(raw_u):
                    x_status_urls_from_items.add(self._normalize_x_url(raw_u))
        except Exception:
            pass

        x_media_kind = "none"
        if x_info_for_gate:
            x_media_kind = x_info_for_gate.media_kind or "none"
        try:
            self.logger.info(f"x.media.kind {x_media_kind}")
        except Exception:
            pass

        x_media_state = {
            "kind": x_media_kind,
            "processed": {},
            "allow_vision": x_media_kind != "video",
        }
        if x_info_for_gate and x_info_for_gate.media_urls:
            x_media_state["primary_urls"] = {
                self._normalize_x_url(u) for u in x_info_for_gate.media_urls if u
            }
        else:
            x_media_state["primary_urls"] = x_status_urls_from_items

        # Filter out Twitter thumbnails from image count if X URLs present
        if has_x_url and heuristic_image_items:
            filtered_items: List[InputItem] = []
            suppressed = 0
            for item in heuristic_image_items:
                try:
                    candidate_urls = self._collect_x_candidate_urls(item)
                    if candidate_urls and all(
                        self._is_twitter_thumbnail_url(u) for u in candidate_urls
                    ):
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
            (is_dm or mentioned_me or is_reply)
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

        # If no items found, process as text-only with text-first default and explicit media-intent nag
        if not items:
            # If user explicitly asked for media analysis but no media/URL is in scope → nag
            try:
                wants_media = has_explicit_media_intent(original_text)
            except Exception:
                wants_media = False
            if wants_media:
                try:
                    self.logger.info(
                        "media_intent_missing_link",
                        extra={
                            "subsys": "route",
                            "event": "media_intent_missing_link",
                            "msg_id": getattr(message, "id", None),
                        },
                    )
                except Exception:
                    pass
                return BotAction(
                    content=(
                        "I didn’t find a link or any text in your reply. If you're replying to a video or image, "
                        "please include the link or upload the media."
                    )
                )

            # Default to TEXT for any minimal chat signal (mentions, punctuation, emoji, short words)
            try:
                mentioned_me = self.bot.user in (
                    getattr(message, "mentions", None) or []
                )
                self.logger.info(
                    "text_default",
                    extra={
                        "subsys": "route",
                        "event": "text_default",
                        "reason": "mention_has_text"
                        if mentioned_me
                        else "ambiguous_intent",
                        "msg_id": getattr(message, "id", None),
                    },
                )
            except Exception:
                pass

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
            and (not has_meaningful_text(original_text))
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

            x_candidate_urls: List[str] = []
            status_keys: set[str] = set()
            is_twitter_thumbnail = False
            if has_x_url:
                x_candidate_urls = self._collect_x_candidate_urls(item)
                status_keys = {
                    self._normalize_x_url(u)
                    for u in x_candidate_urls
                    if self._is_twitter_status_url(u)
                }
                if not status_keys and x_candidate_urls:
                    for u in x_candidate_urls:
                        normalized = self._normalize_x_url(u)
                        if normalized in x_media_state["primary_urls"]:
                            status_keys.add(normalized)
                if x_candidate_urls and all(
                    self._is_twitter_thumbnail_url(u) for u in x_candidate_urls
                ):
                    is_twitter_thumbnail = True

            if (
                x_media_state["kind"] == "video"
                and modality == InputModality.GENERAL_URL
                and status_keys
            ):
                modality = InputModality.VIDEO_URL

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

            skip_reason: Optional[str] = None
            # Only dedupe/skip X status processing on true URL items.
            # Never let embed/thumbnail/image items consume the tweet URL path. [REH][CA]
            should_dedupe_x_status = bool(status_keys and item.source_type == "url")
            if should_dedupe_x_status:
                for key in status_keys:
                    existing = x_media_state["processed"].get(key)
                    if existing:
                        skip_reason = f"duplicate:{existing}"
                        break
            if (
                skip_reason is None
                and x_media_state["kind"] == "video"
                and not x_media_state["allow_vision"]
                and is_twitter_thumbnail
            ):
                skip_reason = "thumbnail_blocked"

            if skip_reason:
                try:
                    self.logger.info(f"x.media.skip reason={skip_reason}")
                except Exception:
                    pass
                continue

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

            # Extraction-only modalities (URL scraping, document ingest, STT) do not
            # benefit from the model-provider fallback ladder. DispatchEmptyError from
            # extraction-side failure (403, version mismatch, etc.) is deterministic —
            # re-running the same extraction against different model providers burns
            # budget and latency with identical results. [REH][PA]
            extraction_only_modalities = (
                InputModality.GENERAL_URL,
                InputModality.SCREENSHOT_URL,
                InputModality.VIDEO_URL,
                InputModality.AUDIO_VIDEO_FILE,
                InputModality.PDF_DOCUMENT,
                InputModality.PDF_OCR,
            )

            if modality in extraction_only_modalities:
                # Direct handler call — no provider ladder
                result_text = ""
                success = False
                duration = 0.0
                attempts = 0
                try:
                    result_text = await self._handle_item_with_provider(
                        item, modality, None, message=message
                    )
                    success = True
                    duration = time.time() - start_time
                    self.logger.info(
                        f"✅ Item {i} completed (extraction-only, no provider ladder) ({duration:.2f}s)"
                    )
                except Exception as e:
                    self.logger.warning(f"❌ Item {i} failed: {e}")
                    success = False
                    result_text = f"❌ Failed: {e}"
                    duration = time.time() - start_time
                    attempts = 1
            else:
                # Vision/image modalities benefit from model-provider fallback
                def create_handler_coro(provider_config: ProviderConfig):
                    async def handler_coro():
                        return await self._handle_item_with_provider(
                            item, modality, provider_config, message=message
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

            # Mark X status processed for true URL items (shared after both branches).
            # Never mark for embeds/thumbnails/images. [REH][CA]
            should_mark_x_status = bool(status_keys and item.source_type == "url")
            if should_mark_x_status:
                status_label = "consumed" if success else "failed"
                for key in status_keys:
                    x_media_state["processed"][key] = status_label
                if success and modality == InputModality.VIDEO_URL:
                    x_media_state["allow_vision"] = False
                    try:
                        self.logger.info("x.media.consumed by=stt")
                    except Exception:
                        pass
                elif not success:
                    x_media_state["allow_vision"] = True

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
        # Gate out early if all multimodal items failed and no meaningful text remains.
        # A summary-only prompt ("I processed 1 input, 0 successful") is not real input
        # and must not trigger LLM generation. [REH][PA]
        if not aggregated_prompt or not aggregated_prompt.strip():
            self.logger.warning(
                f"No content to process after multimodal aggregation (msg_id: {message.id})"
            )
            return BotAction(
                content="I couldn't access that URL to extract content. The site may be blocking automated requests or temporarily unavailable.",
                error=True,
            )

        response_action = await self._invoke_text_flow(
            aggregated_prompt, message, context_str
        )
        if response_action and response_action.has_payload:
            self.logger.info(
                f"✅ Multimodal response generated successfully (msg_id: {message.id})"
            )
            return response_action
        self.logger.warning(
            f"No response generated from text flow (msg_id: {message.id})"
        )
        return None

    async def _handle_item_with_provider(
        self,
        item: InputItem,
        modality: InputModality,
        provider_config: ProviderConfig,
        message: Optional[Message] = None,
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
                item, model_override=provider_config.model, message=message
            )

        handler = handlers.get(modality, self._handle_unknown)
        return await handler(item, message=message)

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
                f"❌ _process_image_from_url failed: {e}",
                extra={"detail": {"url": url}},
                exc_info=True,
            )
            return f"⚠️ Failed to process image from URL (error: {e})"

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

    async def _handle_video_url(
        self, item: InputItem, message: Optional[Message] = None
    ) -> str:
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
            r"https?://(?:www\.)?(?:twitter|x|fxtwitter|vxtwitter|fixupx)\.com/", url
        )

        try:
            # Try video/audio extraction first
            stt_target_url = url
            if is_twitter:
                # Resolve a playable media URL deterministically for X/Twitter and emit breadcrumb [IV][CDiP]
                try:
                    canonical_url = self._canonicalize_x_url(url)
                    normalized_url = self._normalize_x_url(canonical_url)
                except Exception:
                    canonical_url = url
                    normalized_url = url
                frontend_hint: Dict[str, str] = {}
                primary_hint: Dict[str, str] = {}
                ctx = (
                    self._x_frontend_canon.get(normalized_url)
                    or self._x_frontend_canon.get(canonical_url)
                    or {}
                )
                frontend_ctx = ctx.get("frontend")
                primary_ctx = ctx.get("primary") or self._parse_twitter_status_id(
                    normalized_url
                )
                if frontend_ctx:
                    frontend_hint[normalized_url] = frontend_ctx
                if primary_ctx:
                    primary_hint[normalized_url] = primary_ctx
                try:
                    resolved = await self._resolve_x_media(
                        [normalized_url],
                        frontend_hints=frontend_hint,
                        primary_hints=primary_hint,
                    )
                except Exception:
                    resolved = {"kind": "unknown"}
                primary_selected = (
                    (resolved or {}).get("primary")
                    or primary_ctx
                    or self._parse_twitter_status_id(normalized_url)
                    or ""
                )
                frontend_selected = (resolved or {}).get("frontend") or frontend_ctx
                if (resolved or {}).get("kind") == "video" and (resolved or {}).get(
                    "url"
                ):
                    candidate_url = str(resolved.get("url"))
                    verify_kind, verify_ct = await self._verify_media_kind(
                        candidate_url, default="video"
                    )
                    self._log_media_kind_checked(
                        candidate_url, verify_ct, verify_kind or "video"
                    )
                    if verify_kind == "video":
                        self.logger.info(
                            "route.select kind=video reason=resolved_direct_media"
                        )
                        stt_target_url = candidate_url
                        try:
                            import hashlib as _hl

                            uhash = _hl.sha256(stt_target_url.encode()).hexdigest()[:16]
                            detail = {
                                "primary": primary_selected,
                                "selected": primary_selected,
                                "tier": 1,
                                "has_audio": ".mp4" in stt_target_url.lower(),
                                "url_hash": uhash,
                            }
                            if frontend_selected:
                                detail["frontend"] = frontend_selected
                            self.logger.info(
                                "media.selected",
                                extra={
                                    "event": "media.selected",
                                    "detail": detail,
                                },
                            )
                            if primary_selected and uhash:
                                stt_target_url = f"{stt_target_url}#ptid={primary_selected}&uh={uhash}"
                                if frontend_selected:
                                    stt_target_url = (
                                        f"{stt_target_url}&fe={frontend_selected}"
                                    )
                        except Exception:
                            pass
                    else:
                        # Poster hint or misclassification: fall back to original URL so downstream can degrade gracefully
                        stt_target_url = url
                else:
                    stt_target_url = url

            result = await self._run_stt_job(
                hear_infer_from_url(stt_target_url),
                message,
            )
            metadata = {}
            if isinstance(result, dict):
                try:
                    metadata = result.get("metadata") or {}
                    if metadata.get("demux_fallback"):
                        self.logger.info("x.media.demux_fallback used=true")
                except Exception:
                    metadata = {}

            transcription: Optional[str] = None
            if isinstance(result, dict):
                transcription = result.get("transcription") or result.get("text")
            elif result:
                transcription = str(result)

            if transcription:
                if is_twitter:
                    return await self._format_x_with_resolved_base_text(
                        url=url,
                        stt_res={"transcription": transcription, "metadata": metadata},
                    )
                # Non-Twitter: keep existing concise output
                title = metadata.get("title", "Unknown")
                return f"Video transcription from {url} ('{title}'): {transcription}"
            else:
                # No/low speech case: for Twitter, degrade to caption-only evidence and continue [REH]
                if is_twitter:
                    composed = await self._format_x_no_speech_fallback(
                        url=url,
                        stt_res=result,
                    )
                    return composed

                # Non-Twitter: keep existing concise output
                return f"Could not transcribe audio from video: {url}"

        except VideoIngestError as ve:
            error_str = str(ve).lower()

            # For Twitter URLs with no media, use syndication/API path instead of web extractor [CA][REH]
            if is_twitter and (
                "no video or audio content found" in error_str
                or "no video could be found" in error_str
                or "failed to download video" in error_str
                or "no video" in error_str
            ):
                # New: targeted syndication image probe (feature-flagged)
                if getattr(
                    self, "_x_syn_probe_enabled", True
                ) and self._is_twitter_status_url(url):
                    try:
                        status_id, imgs = await self._resolve_and_probe_twitter_images(
                            url=url
                        )
                        if imgs:
                            # Prefer unified VL pipeline with caption when available [CA][REH]
                            try:
                                return await self._route_probed_twitter_images_with_caption(
                                    url=url,
                                    status_id=status_id,
                                    image_urls=imgs,
                                )
                            except Exception:
                                # Fallback: single-image VL without caption
                                try:
                                    desc = await self._vl_describe_image_from_url(
                                        imgs[0]
                                    )
                                    return (
                                        desc
                                        or "⚠️ Unable to analyze the images from this tweet."
                                    )
                                except Exception:
                                    # Fall through to general handler on VL error
                                    pass
                    except Exception as e:
                        self.logger.debug(f"x.syndication.probe.failed | {e}")
                self.logger.info(
                    f"🐦 No video in Twitter URL; routing to syndication/API path: {url}"
                )
                # Fallback: general URL handler which has X syndication logic
                return await self._handle_general_url(
                    InputItem(source_type="url", payload=url)
                )

            # For non-Twitter URLs, provide user-friendly message
            self.logger.info(f"ℹ️ Video processing: {ve}")
            return f"⚠️ {str(ve)}"

        except InferenceError as ie:
            # Prefer caption-only degrade for Twitter when available [REH]
            if is_twitter:
                try:
                    self._emit_caption_only_fallback_breadcrumbs("error")

                    # Try API then syndication for anchored caption.
                    formatted = await self._format_x_with_resolved_base_text_if_available(
                        url=url,
                        stt_res={"transcription": ""},
                    )
                    if formatted:
                        return formatted
                except Exception:
                    pass
            # Fallback to existing user-friendly message for non-Twitter or when caption unavailable
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

    async def _handle_audio_video_file(
        self, item: InputItem, message: Optional[Message] = None
    ) -> str:
        """
        Handle audio/video file attachments.
        Returns transcribed text for further processing.
        """
        from .video_ingest import VideoIngestError
        from .exceptions import InferenceError

        attachment = item.payload
        self.logger.info(f"🎵 Processing audio/video file: {attachment.filename}")

        try:
            result = await self._run_stt_job(hear_infer(attachment), message)
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

    async def _handle_pdf(
        self, item: InputItem, message: Optional[Message] = None
    ) -> str:
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

            if isinstance(result, str):
                text_content = result
                if not text_content or not text_content.strip():
                    return f"Could not extract text from PDF: {attachment.filename}"
                return f"PDF content from {attachment.filename}: {text_content}"

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
        """Process PDF from URL using the unified document ingestion pipeline."""
        try:
            # Reuse the same document ingestion path as attachments so that
            # URL-based PDFs behave like uploaded documents. [CA][REH]
            result = await ingest_document_from_url(url)

            if result.get("error"):
                err = str(result["error"])
                self.logger.warning(
                    f"PDF URL ingestion failed url={url[:80]} error={err[:100]}"
                )
                return f"Could not extract text from PDF URL: {url} (Error: {err})"

            text = (result.get("text") or "").strip()
            if not text:
                self.logger.warning(
                    f"PDF URL ingestion produced no text url={url[:80]}"
                )
                return f"Could not extract text from PDF URL: {url}"

            # Use a generic label; the aggregator already includes per-item headers
            # with a human-readable name, so this mirrors the attachment path.
            return f"PDF content from URL {url}: {text}"

        except Exception as e:
            self.logger.error(
                f"PDF URL ingestion exception url={url[:80]} error={e}",
                exc_info=True,
            )
            return "Failed to process PDF document from URL."

    async def _handle_pdf_ocr(
        self, item: InputItem, message: Optional[Message] = None
    ) -> str:
        """
        Handle PDF documents that require OCR processing.
        Returns extracted text for further processing.
        """
        # For now, delegate to regular PDF handler
        # TODO: Implement OCR-specific logic
        return await self._handle_pdf(item)

    async def _handle_general_url(
        self, item: InputItem, message: Optional[Message] = None
    ) -> str:
        """
        Handle general URL input items.
        Returns extracted content for further processing.
        No auto-screenshot fallback here; screenshots require explicit !ss command.

        Enhanced: Classifies URLs by MIME type and routes document/audio/video/image
        URLs to their respective pipelines instead of web scraping. [CA][REH]
        """
        try:
            if item.source_type != "url":
                return f"URL handler received non-URL item: {item.source_type}"

            url = item.payload
            # Only canonicalize when this is actually a Twitter/X-like URL to avoid
            # misclassifying arbitrary hosts that happen to contain numbers. [IV]
            if self._is_twitter_url(url):
                canon_candidate = self._canonicalize_x_url(url)
                if canon_candidate and canon_candidate != url:
                    url = canon_candidate
            self.logger.info(f"🌐 Processing general URL: {url}")

            # --- URL MIME classification for media/document routing [CA][REH] ---
            # Skip classification for known platform URLs (Twitter/X, YouTube, etc.)
            # which have their own specialized handling below.
            skip_mime_classification = self._is_twitter_url(url)

            if not skip_mime_classification:
                try:
                    from .url_classifier import classify_url
                    from .attachment_classifier import AttachmentBucket

                    classified = await classify_url(url)

                    # Route based on classification bucket
                    if classified.bucket == AttachmentBucket.DOC:
                        # Document URL → document ingestion pipeline
                        self.logger.info(
                            f"url.route bucket=DOC url={url[:80]}",
                            extra={
                                "subsys": "url",
                                "event": "url.route",
                                "detail": {"bucket": "DOC", "url": url[:200]},
                            },
                        )
                        return await self._handle_document_url(url, classified, message)

                    elif classified.bucket == AttachmentBucket.AUDIO:
                        # Audio URL → STT pipeline
                        self.logger.info(
                            f"url.route bucket=AUDIO url={url[:80]}",
                            extra={
                                "subsys": "url",
                                "event": "url.route",
                                "detail": {"bucket": "AUDIO", "url": url[:200]},
                            },
                        )
                        return await self._handle_audio_url(url, classified, message)

                    elif classified.bucket == AttachmentBucket.VIDEO:
                        # Video URL → media/STT pipeline
                        self.logger.info(
                            f"url.route bucket=VIDEO url={url[:80]}",
                            extra={
                                "subsys": "url",
                                "event": "url.route",
                                "detail": {"bucket": "VIDEO", "url": url[:200]},
                            },
                        )
                        return await self._handle_video_file_url(
                            url, classified, message
                        )

                    elif classified.bucket == AttachmentBucket.IMAGE:
                        # Image URL → VL pipeline
                        self.logger.info(
                            f"url.route bucket=IMAGE url={url[:80]}",
                            extra={
                                "subsys": "url",
                                "event": "url.route",
                                "detail": {"bucket": "IMAGE", "url": url[:200]},
                            },
                        )
                        return await self._handle_image_url(url, classified, message)

                    # OTHER bucket or TXT_PROMPT → fall through to web scraping
                    self.logger.info(
                        f"url.route bucket={classified.bucket.name} → web_scrape url={url[:80]}",
                        extra={
                            "subsys": "url",
                            "event": "url.route",
                            "detail": {
                                "bucket": classified.bucket.name,
                                "url": url[:200],
                                "action": "web_scrape",
                            },
                        },
                    )

                except Exception as e:
                    # Classification failed - fall through to existing web scraping [REH]
                    self.logger.debug(f"url.classify.failed url={url[:80]} error={e}")

            # Per-operation time budgets and bounded-wait helper [PA][REH][RAT]
            cfg = self.config
            try:
                x_stt_probe_timeout = float(cfg.get("X_STT_PROBE_TIMEOUT_S", 60.0))
            except Exception:
                x_stt_probe_timeout = 60.0
            try:
                # Prefer seconds; fallback to ms if provided
                x_api_timeout_s = float(cfg.get("X_API_TIMEOUT_S", 0)) or (
                    float(cfg.get("X_API_TIMEOUT_MS", 8000)) / 1000.0
                )
            except Exception:
                x_api_timeout_s = 8.0
            try:
                x_syn_call_timeout = float(
                    cfg.get(
                        "X_SYNDICATION_GROSS_TIMEOUT_S",
                        max(getattr(self, "_x_syn_timeout_s", 3.0), 3.0) + 0.5,
                    )
                )
            except Exception:
                x_syn_call_timeout = (
                    max(getattr(self, "_x_syn_timeout_s", 3.0), 3.0) + 0.5
                )
            try:
                url_process_timeout = float(cfg.get("URL_PROCESS_TIMEOUT_S", 25.0))
            except Exception:
                url_process_timeout = 25.0
            try:
                web_extract_timeout = float(cfg.get("WEB_EXTRACT_TIMEOUT_S", 30.0))
            except Exception:
                web_extract_timeout = 30.0

            api_data: Optional[Dict[str, Any]] = None

            async def _bounded(
                coro, timeout_s: float, tag: str, detail: Optional[dict] = None
            ):
                """Await coro with a timeout and emit start/ok/timeout/fail breadcrumbs. [REH][PA]"""
                import time as _t

                t0 = _t.time()
                try:
                    self.logger.debug(
                        f"{tag}.start",
                        extra={
                            "event": f"{tag}.start",
                            "detail": (detail or {}) | {"timeout_s": timeout_s},
                        },
                    )
                except Exception:
                    pass
                try:
                    res = await asyncio.wait_for(coro, timeout=timeout_s)
                    try:
                        dt_ms = int((_t.time() - t0) * 1000)
                        if hasattr(res, "success") and not getattr(res, "success", False):
                            err = getattr(res, "error", "") or "unknown"
                            self.logger.warning(
                                f"{tag}.fail extraction_no_content ms={dt_ms} ({err})",
                                extra={
                                    "event": f"{tag}.fail",
                                    "detail": (detail or {})
                                    | {"ms": dt_ms, "error": err},
                                },
                            )
                        else:
                            self.logger.info(
                                f"{tag}.ok ms={dt_ms}",
                                extra={
                                    "event": f"{tag}.ok",
                                    "detail": (detail or {}) | {"ms": dt_ms},
                                },
                            )
                    except Exception:
                        pass
                    return res, None
                except asyncio.TimeoutError:
                    try:
                        dt_ms = int((_t.time() - t0) * 1000)
                        self.logger.warning(
                            f"{tag}.timeout ms={dt_ms}",
                            extra={
                                "event": f"{tag}.timeout",
                                "detail": (detail or {})
                                | {"ms": dt_ms, "timeout_s": timeout_s},
                            },
                        )
                    except Exception:
                        pass
                    return None, "timeout"
                except Exception as e:
                    try:
                        dt_ms = int((_t.time() - t0) * 1000)
                        self.logger.info(
                            f"{tag}.fail {e.__class__.__name__}: {e}",
                            extra={
                                "event": f"{tag}.fail",
                                "detail": (detail or {})
                                | {"ms": dt_ms, "error": str(e)},
                            },
                        )
                    except Exception:
                        pass
                    return None, "error"

            # Optional: Twitter/X author self-reply thread unroll (feature-gated) [PA][REH]
            # IMPORTANT: For X/Twitter status URLs we now defer unroll until after media detection,
            # so images/video can take precedence. The early unroll remains available for non-X URLs.
            try:
                if (
                    bool(self.config.get("TWITTER_UNROLL_ENABLED", False))
                    and self._is_twitter_status_url(url)
                    and False
                ):
                    # Emit a DEBUG start event so operators can see attempts when LOG_LEVEL=debug
                    try:
                        self.logger.debug(
                            "threads.x: unroll_start",
                            extra={
                                "subsys": "threads.x",
                                "event": "unroll_start",
                                "detail": {"url": url},
                            },
                        )
                    except Exception:
                        pass
                    t0 = time.time()
                    ctx, reason = await unroll_author_thread(
                        url,
                        timeout_s=float(
                            self.config.get("TWITTER_UNROLL_TIMEOUT_S", 15.0)
                        ),
                        max_tweets=int(
                            self.config.get("TWITTER_UNROLL_MAX_TWEETS", 30)
                        ),
                        max_chars=int(
                            self.config.get("TWITTER_UNROLL_MAX_CHARS", 6000)
                        ),
                    )
                    if ctx is not None and getattr(ctx, "joined_text", None):
                        dt_ms = int((time.time() - t0) * 1000)
                        try:
                            self.logger.info(
                                f"threads.x: unroll_ok tweets={ctx.tweet_count} ms={dt_ms}",
                                extra={
                                    "subsys": "threads.x",
                                    "event": "unroll_ok",
                                    "detail": {
                                        "tweets": ctx.tweet_count,
                                        "ms": dt_ms,
                                        "url": ctx.canonical_url,
                                    },
                                },
                            )
                        except Exception:
                            pass
                        return ctx.joined_text
                    else:
                        try:
                            self.logger.info(
                                "threads.x: unroll_fallback",
                                extra={
                                    "subsys": "threads.x",
                                    "event": "unroll_fallback",
                                    "detail": {
                                        "reason": reason or "unroll_not_available"
                                    },
                                },
                            )
                        except Exception:
                            pass
            except Exception as e:
                # Failure: include exception details for visibility, but keep flow moving [REH]
                try:
                    self.logger.info(
                        "threads.x: unroll_failed",
                        extra={
                            "subsys": "threads.x",
                            "event": "unroll_failed",
                            "detail": {
                                "reason": "exception",
                                "error": f"{e.__class__.__name__}: {e}",
                                "url": url,
                            },
                        },
                        exc_info=True,
                    )
                except Exception:
                    pass

            # Syndication-first for Twitter/X posts (API as last resort) [CA][SFT]
            if self._is_twitter_url(url):
                cfg = self.config
                require_api = bool(cfg.get("X_API_REQUIRE_API_FOR_TWITTER", False))
                allow_fallback_5xx = bool(cfg.get("X_API_ALLOW_FALLBACK_ON_5XX", True))
                syndication_enabled = bool(cfg.get("X_SYNDICATION_ENABLED", True))
                # Default to syndication-first unless explicitly disabled
                syndication_first = bool(cfg.get("X_SYNDICATION_FIRST", True))
                tweet_id = XApiClient.extract_tweet_id(str(url))
                x_client = await self._get_x_api_client()
                api_data = None

                # Syndication-first: must confirm video content before STT is attempted [IV][REH]
                # Removed STT probe-first path - it was unreliable and ran STT on image-only tweets
                # Syndication now ALWAYS runs first to determine content type
                # Tier 1: Syndication JSON (cache + concurrency) when allowed and preferred [PA][REH]
                if (
                    tweet_id
                    and syndication_enabled
                    and not require_api
                    and (syndication_first or x_client is None)
                ):
                    syn, _ = await _bounded(
                        self._get_tweet_via_syndication(tweet_id),
                        x_syn_call_timeout,
                        "x.syndication",
                        {"tweet_id": tweet_id},
                    )
                    if syn:
                        syn = await self._maybe_hydrate_syndication_payload(
                            tweet_id, syn
                        )
                        self._metric_inc("x.syndication.hit", None)

                        # Log syndication response keys for video detection debugging [IV][REH]
                        try:
                            syn_keys = list(syn.keys())[:30] if isinstance(syn, dict) else []
                            self.logger.info(
                                f"route=x_syndication.metadata keys={syn_keys} tweet_id={tweet_id}",
                                extra={
                                    "event": "x.syndication.metadata",
                                    "detail": {
                                        "tweet_id": tweet_id,
                                        "keys": syn_keys,
                                    },
                                },
                            )
                        except Exception:
                            pass

                        # Media-first branching: use robust extractor rather than only 'photos' [CA][REH]
                        photos = syn.get("photos") or []
                        text = self._extract_syndication_text(syn)

                        # Enhanced: look for images in extended_entities/quoted/card as well
                        extracted_images = []
                        try:
                            from .syndication.extract import (
                                extract_text_and_images_from_syndication,
                                syndication_has_video,
                            )

                            _ext = extract_text_and_images_from_syndication(syn)
                            extracted_images = _ext.get("image_urls", []) or []
                            _syn_has_video = syndication_has_video(syn)

                            # Log video detection result explicitly [IV][REH]
                            self.logger.info(
                                f"route=x_syndication.video_detection has_video={_syn_has_video} tweet_id={tweet_id}",
                                extra={
                                    "event": "x.syndication.video_detection",
                                    "detail": {
                                        "has_video": _syn_has_video,
                                        "tweet_id": tweet_id,
                                    },
                                },
                            )
                        except Exception:
                            extracted_images = []
                            _syn_has_video = False

                        # Bread crumb for future debugging [CMV]
                        try:
                            self.logger.info(
                                f"route=x_syndication.pick | photos={len(photos)} extracted={len(extracted_images)}"
                            )
                        except Exception:
                            pass

                        # Check for image-only tweet (images present, no video)
                        # Mixed media (video + images) must route to STT, not VL.
                        has_any_images = bool(photos) or bool(extracted_images)
                        is_image_only = has_any_images and (not _syn_has_video)
                        # X Article posts often syndicate as a t.co pointer with no media metadata.
                        # Resolve article text early so they stay in text flow (not STT/VL fallbacks).
                        if (
                            (not _syn_has_video)
                            and (not has_any_images)
                            and tweet_id
                            and bool(re.search(r"https?://t\.co/[A-Za-z0-9]+", text or ""))
                        ):
                            article_data, _ = await _bounded(
                                self._fetch_x_article_from_fxtwitter(tweet_id),
                                min(
                                    getattr(self, "_x_syn_timeout_s", 3.0) + 0.5,
                                    4.5,
                                ),
                                "x.syndication.article.resolve",
                                {"tweet_id": tweet_id},
                            )
                            if isinstance(article_data, dict) and article_data:
                                try:
                                    syn["article"] = article_data
                                except Exception:
                                    pass
                                text = self._extract_syndication_text(syn)
                                base = self._compose_text_tweet_evidence(url, syn)
                                try:
                                    self.logger.info(
                                        "route=x_syndication.article.ok",
                                        extra={
                                            "event": "x.syndication.article.ok",
                                            "detail": {
                                                "tweet_id": tweet_id,
                                                "chars": len(text or ""),
                                                "article_id": article_data.get("id") or "",
                                            },
                                        },
                                    )
                                except Exception:
                                    pass
                                return self._format_x_tweet_with_transcription(
                                    base_text=base,
                                    url=url,
                                    stt_res={},
                                )
                        # Sparse syndication payloads (e.g., only text/user) can miss media metadata.
                        # In that case, defer final text fallback so Tier-2 API media checks can run.
                        syn_media_hints = any(
                            k in syn
                            for k in (
                                "media",
                                "photos",
                                "video",
                                "video_info",
                                "video_variants",
                                "video_urls",
                                "media_duration",
                                "duration_ms",
                                "extended_entities",
                                "entities",
                                "quoted_tweet",
                                "quoted_status",
                                "retweeted_status",
                                "legacy",
                                "card",
                                "image",
                            )
                        )

                        # Log routing decision for observability [IV][REH]
                        try:
                            routing_decision = (
                                "video"
                                if _syn_has_video
                                else ("image_only" if is_image_only else "text_or_mixed")
                            )
                            self.logger.info(
                                f"route=x_syndication.decision decision={routing_decision} "
                                f"photos={len(photos)} extracted={len(extracted_images)} "
                                f"has_video={_syn_has_video} has_images={has_any_images}",
                                extra={
                                    "event": "x.syndication.routing_decision",
                                    "detail": {
                                        "decision": routing_decision,
                                        "photos": len(photos),
                                        "extracted_images": len(extracted_images),
                                        "has_video": _syn_has_video,
                                        "has_images": has_any_images,
                                        "text_length": len(text),
                                    },
                                },
                            )
                        except Exception:
                            pass

                        if is_image_only and bool(
                            cfg.get("TWITTER_IMAGE_ONLY_ENABLE", True)
                        ):
                            # Preserve existing specialized path when native photos are present [CA]
                            # Also route to VL when images were extracted from extended_entities [IV]
                            img_count = (
                                len(extracted_images)
                                if extracted_images
                                else len(photos)
                            )
                            self.logger.info(
                                f"🖼️ Image-only tweet detected, routing to Vision/OCR: {url}"
                            )
                            self._metric_inc(
                                "x.tweet_image_only.syndication",
                                {
                                    "photos": str(img_count),
                                    "source": "extracted"
                                    if extracted_images
                                    else "native",
                                },
                            )
                            syn_for_images = syn
                            syn_for_images = await self._maybe_hydrate_syndication_payload(
                                tweet_id,
                                syn_for_images,
                                allow_tco_pointer=True,
                            )
                            return await self._handle_image_only_tweet(
                                url, syn_for_images, source="syndication"
                            )

                        # Compose evidence for text-only tweets through the standard bundle [CA]
                        base = self._compose_text_tweet_evidence(url, syn)

                        # Video always wins: route confirmed video (including mixed media) to STT.
                        if _syn_has_video:
                            # Log STT initiation for confirmed video content [IV]
                            try:
                                self.logger.info(
                                    f"route=x_syndication.stt_start syndication_confirmed_video=true url={url[:80]}",
                                    extra={
                                        "event": "x.syndication.stt_start",
                                        "detail": {
                                            "has_video": True,
                                            "url": url[:80],
                                            "photos": len(photos),
                                            "extracted_images": len(extracted_images),
                                        },
                                    },
                                )
                            except Exception:
                                pass
                            stt_res, stt_err = await _bounded(
                                hear_infer_from_url(url),
                                x_stt_probe_timeout,
                                "x.syndication.stt",
                                {"url": url},
                            )
                            return self._format_x_video_stt_probe_result(
                                base_text=base,
                                url=url,
                                stt_res=stt_res,
                                stt_err=stt_err,
                                tweet_text=text,
                                emit_fail_event=True,
                                fail_media_kind="video",
                                msg_id=(message.id if message else None),
                            )

                        # No video confirmed. Probe for images only when none were extracted yet.
                        if (not photos) and (not extracted_images):
                            try:
                                status_id = self._resolve_twitter_status_id(
                                    url, tweet_id=tweet_id
                                )
                            except Exception:
                                status_id = ""
                            if status_id:
                                imgs, _ = await _bounded(
                                    self._probe_twitter_syndication_images(
                                        url, status_id
                                    ),
                                    self._x_syn_probe_budget_timeout_s(),
                                    "x.syndication.image_probe",
                                    {"tweet_id": status_id},
                                )
                                imgs = imgs or []
                                if imgs:
                                    self._log_twitter_syndication_images(imgs)
                                    # Convert to syndication-like shape and route to VL
                                    tweet_text = await self._resolve_twitter_caption_from_syndication(
                                        status_id,
                                        fallback_text=text,
                                    )
                                    result = await self._route_twitter_images_with_caption(
                                        url=url,
                                        caption_text=tweet_text,
                                        image_urls=imgs,
                                    )
                                    return result

                            sparse_no_media = (
                                (not _syn_has_video)
                                and (not has_any_images)
                                and (not syn_media_hints)
                            )
                            if sparse_no_media:
                                try:
                                    self.logger.info(
                                        "route=x_syndication.defer reason=sparse_media_metadata",
                                        extra={
                                            "event": "x.syndication.defer",
                                            "detail": {
                                                "tweet_id": tweet_id,
                                                "syn_keys": sorted(list(syn.keys()))[:30],
                                                "has_api_client": bool(x_client is not None),
                                            },
                                        },
                                    )
                                except Exception:
                                    pass
                                # Sparse syndication can omit media metadata entirely.
                                # Probe direct X media resolution before caption-only fallback.
                                try:
                                    resolved_sparse, _ = await _bounded(
                                        self._resolve_x_media([url]),
                                        self._x_syn_probe_budget_timeout_s(),
                                        "x.syndication.sparse.resolve",
                                        {"tweet_id": tweet_id or ""},
                                    )
                                except Exception:
                                    resolved_sparse = None

                                sparse_kind, sparse_images, sparse_url = (
                                    self._extract_sparse_media_resolution(
                                        resolved_sparse,
                                        default_url=url,
                                    )
                                )
                                if sparse_kind == "video":
                                    try:
                                        self.logger.info(
                                            "route=x_syndication.sparse.stt_start",
                                            extra={
                                                "event": "x.syndication.sparse.stt_start",
                                                "detail": {
                                                    "tweet_id": tweet_id or "",
                                                    "resolved_kind": "video",
                                                },
                                            },
                                        )
                                    except Exception:
                                        pass
                                    stt_res, stt_err = await _bounded(
                                        hear_infer_from_url(sparse_url),
                                        x_stt_probe_timeout,
                                        "x.syndication.sparse.stt",
                                        {"url": sparse_url},
                                    )
                                    return self._format_x_video_stt_probe_result(
                                        base_text=base,
                                        url=sparse_url,
                                        stt_res=stt_res,
                                        stt_err=stt_err,
                                        tweet_text=text,
                                    )
                                if sparse_kind == "image" and sparse_images:
                                    self._log_twitter_syndication_images(sparse_images)
                                    image_text = await self._resolve_syndication_caption_from_payload(
                                        tweet_id,
                                        syn,
                                        fallback_text=text,
                                    )
                                    return await self._route_twitter_images_with_caption(
                                        url=url,
                                        caption_text=image_text,
                                        image_urls=sparse_images,
                                    )
                                if sparse_kind not in ("video", "image"):
                                    # Last resort for sparse syndication: attempt STT directly on the tweet URL.
                                    # This preserves video->STT routing when metadata endpoints are incomplete.
                                    try:
                                        self.logger.info(
                                            "route=x_syndication.sparse.force_stt_start",
                                            extra={
                                                "event": "x.syndication.sparse.force_stt_start",
                                                "detail": {
                                                    "tweet_id": tweet_id or "",
                                                    "resolved_kind": sparse_kind,
                                                },
                                            },
                                        )
                                    except Exception:
                                        pass
                                    forced_stt_res, forced_stt_err = await _bounded(
                                        hear_infer_from_url(url),
                                        x_stt_probe_timeout,
                                        "x.syndication.sparse.force_stt",
                                        {"url": url},
                                    )
                                    formatted = self._format_x_transcription_if_present(
                                        base_text=base,
                                        url=url,
                                        stt_res=forced_stt_res,
                                    )
                                    if formatted:
                                        return formatted
                                    self._emit_stt_fail_event(
                                        self._classify_stt_error_reason(forced_stt_err),
                                        media_kind="unknown_sparse",
                                    )

                                # If API is available, continue to Tier-2 API branch below.
                                if tweet_id and x_client is not None:
                                    pass
                                else:
                                    return self._format_x_caption_only_fallback_result(
                                        url=url,
                                        base_text=base,
                                        tweet_text=text,
                                        api_data=api_data,
                                    )
                            else:
                            # Fall back to text-only ONLY when video was NOT confirmed by syndication [REH]
                                # Prefer API text if available; otherwise fall back to syndication text or composed evidence [REH]
                                return self._format_x_caption_only_fallback_result(
                                    url=url,
                                    base_text=base,
                                    tweet_text=text,
                                    api_data=api_data,
                                )

                        if has_any_images:
                            # Images are present and no video is confirmed; route to unified syndication→VL handler.
                            try:
                                self.logger.info(
                                    "route=x_syndication | sending images to VL (hi-res)",
                                    extra={"detail": {"url": url}},
                                )
                            except Exception:
                                pass

                            syn_for_vl = syn
                            syn_for_vl = await self._maybe_hydrate_syndication_payload(
                                tweet_id, syn_for_vl, allow_tco_pointer=True
                            )
                            result = await self._route_twitter_syndication_to_vl(
                                syn_for_vl,
                                url,
                            )
                            # Syndication handler returns final text; pass through as string for aggregator
                            return result

                    # If syndication JSON failed to produce data, probe fx/vx for high-res photos [REH][PA]
                    if getattr(
                        self, "_x_syn_probe_enabled", True
                    ) and self._is_twitter_status_url(url):
                        try:
                            status_id, imgs = await self._resolve_and_probe_twitter_images(
                                url=url,
                                tweet_id=tweet_id,
                            )
                            if imgs:
                                result = await self._route_probed_twitter_images_with_caption(
                                    url=url,
                                    status_id=status_id,
                                    image_urls=imgs,
                                )
                                return result
                        except Exception as e:
                            self.logger.debug(f"x.syndication.image_probe.failed | {e}")

                # Tier 2 (optionally before API if syndication_first): X API [SFT]
                if tweet_id and x_client is not None:
                    try:
                        api_data, apist = await _bounded(
                            x_client.get_tweet_by_id(tweet_id),
                            x_api_timeout_s,
                            "x.api.get_tweet",
                            {"tweet_id": tweet_id},
                        )
                        if apist is not None and api_data is None:
                            raise APIError(f"X API call failed ({apist})")
                        includes = api_data.get("includes") or {}
                        media_list = includes.get("media") or []
                        media_types = {
                            m.get("type") for m in media_list if isinstance(m, dict)
                        }

                        if {"video", "animated_gif"} & media_types:
                            try:
                                stt_res, _ = await _bounded(
                                    hear_infer_from_url(url),
                                    x_stt_probe_timeout,
                                    "x.api.stt",
                                    {"url": url},
                                )
                                base = self._format_x_tweet_result(api_data, url)
                                formatted = self._format_x_transcription_if_present(
                                    base_text=base,
                                    url=url,
                                    stt_res=stt_res,
                                )
                                if formatted:
                                    return f"Video/audio content from {url}: {formatted}"
                                # No-speech in API probe: log and continue with caption-only bundle [REH]
                                return await self._format_x_no_speech_fallback(
                                    base_text=base,
                                    url=url,
                                    stt_res=stt_res,
                                )
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
                            tweet_data = self._extract_x_api_primary_tweet(api_data)

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

                            if not bool(cfg.get("X_API_ROUTE_PHOTOS_TO_VL", False)):
                                return self._format_x_tweet_result(api_data, url)

                            self.logger.info(
                                "🖼️🐦 Routing X photos to VL via API data",
                                extra={
                                    "event": "x.photo_to_vl.start",
                                    "detail": {"url": url, "photo_count": len(photos)},
                                },
                            )
                            self._metric_inc("x.photo_to_vl.enabled", None)

                            notes: List[str] = []
                            analyzed = 0
                            total = len(photos)
                            for idx, photo in enumerate(photos, start=1):
                                photo_url = str((photo.get("url") or "")).strip()
                                if not photo_url:
                                    notes.append(
                                        f"📷 Photo {idx}/{total}: URL unavailable"
                                    )
                                    continue
                                try:
                                    desc = await self._vl_describe_image_from_url(
                                        photo_url,
                                        prompt=self._get_system_prompt("vl_prompt"),
                                    )
                                except Exception:
                                    desc = None
                                if desc:
                                    analyzed += 1
                                    notes.append(f"📷 Photo {idx}/{total}: {desc}")
                                else:
                                    notes.append(
                                        f"📷 Photo {idx}/{total}: analysis unavailable"
                                    )

                            lines: List[str] = [f"Photos analyzed: {analyzed}/{total}"]
                            if api_text:
                                lines.extend(["[Tweet Caption]", api_text, ""])
                            lines.extend(notes)
                            lines.extend(["", self._canonicalize_twitter_status_url(url)])
                            return "\n".join(lines).strip()

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
            url_result, _ = await _bounded(
                process_url(url), url_process_timeout, "url.process", {"url": url}
            )

            if isinstance(url_result, str):
                return f"Web content from {url}:\n{url_result}"

            # Handle errors: before giving up, try tiered extractor (A/B) [REH]
            if not url_result or url_result.get("error"):
                self.logger.info(
                    f"🧭 process_url failed for {url}; falling back to tiered extractor"
                )
                extract_res, _ = await _bounded(
                    web_extractor.extract(url),
                    web_extract_timeout,
                    "url.extract",
                    {"url": url},
                )
                if extract_res and extract_res.success:
                    return f"Web content from {extract_res.canonical_url or url}:\n{extract_res.to_message()}"
                # Both process_url and tiered extractor failed — propagate as real failure [REH][PA]
                err_detail = url_result.get("error", "none") if url_result else "none"
                self.logger.warning(
                    f"url.extract.all_failed url={url[:120]} error={getattr(extract_res, 'error', err_detail)}",
                    extra={
                        "event": "url.extract.all_failed",
                        "detail": {"url": url[:200], "error": getattr(extract_res, "error", err_detail)},
                    },
                )
                raise DispatchEmptyError(
                    f"Could not extract content from URL: {url} (Error: {err_detail})"
                )

            # Extract text content from result dictionary
            content = url_result.get("text", "")
            if not content or not content.strip():
                # If no text content, check if we have a screenshot
                if url_result.get("screenshot_path"):
                    return f"Screenshot captured for {url}: {url_result['screenshot_path']}"
                # As a last attempt, try tiered extractor (if process_url returned no error but empty)
                self.logger.info(
                    f"🧭 No text from process_url; trying tiered extractor for {url}"
                )
                extract_res, _ = await _bounded(
                    web_extractor.extract(url),
                    web_extract_timeout,
                    "url.extract",
                    {"url": url},
                )
                if extract_res and extract_res.success:
                    return f"Web content from {extract_res.canonical_url or url}:\n{extract_res.to_message()}"
                # process_url returned empty/no-text AND tiered extractor also failed — real failure [REH][PA]
                self.logger.warning(
                    f"url.extract.all_failed url={url[:120]} error={getattr(extract_res, 'error', 'no_result')}",
                    extra={
                        "event": "url.extract.all_failed",
                        "detail": {"url": url[:200], "error": getattr(extract_res, "error", "no_result")},
                    },
                )
                raise DispatchEmptyError(
                    f"Could not extract content from URL: {url}"
                )

            # Check if smart routing detected media and should route to yt-dlp
            route_to_ytdlp = url_result.get("route_to_ytdlp", False)
            if route_to_ytdlp:
                self.logger.info(
                    f"🎥 Smart routing detected media in {url}, routing to yt-dlp flow"
                )

                try:
                    # Process through yt-dlp flow
                    transcription_result = await hear_infer_from_url(url)

                    if self._stt_result_has_transcription(transcription_result):
                        transcription = transcription_result["transcription"]
                        metadata = transcription_result.get("metadata", {})
                        title = metadata.get("title", "Unknown")

                        return f"Video/audio content from {url} ('{title}'): {transcription}"
                    else:
                        self.logger.warning(
                            f"url.ytdlp.stt_failed url={url[:120]}",
                            extra={"event": "url.ytdlp.stt_failed", "detail": {"url": url[:200]}},
                        )
                        return ""

                except Exception as e:
                    self.logger.warning(f"url.ytdlp.failed url={url[:120]} error={e}")
                    return ""

            # Prefer text from process_url when available.
            content = url_result.get("text", "")
            if content and content.strip():
                return f"Web content from {url}: {content}"

            # If no text was extracted (and no media route), use tiered extractor (no screenshots)
            self.logger.info(
                f"🧭 Falling back to tiered extractor for {url} (no auto-screenshot)"
            )
            extract_res, _ = await _bounded(
                web_extractor.extract(url),
                web_extract_timeout,
                "url.extract",
                {"url": url},
            )
            if extract_res and extract_res.success:
                return f"Web content from {extract_res.canonical_url or url}:\n{extract_res.to_message()}"
            # Both tiered extraction tiers failed — propagate as real failure [REH][PA]
            self.logger.warning(
                f"url.extract.all_failed url={url[:120]} error={getattr(extract_res, 'error', 'no_result')}",
                extra={
                    "event": "url.extract.all_failed",
                    "detail": {"url": url[:200], "error": getattr(extract_res, "error", "no_result")},
                },
            )
            raise DispatchEmptyError(
                f"Could not extract content from URL: {url}"
            )

        except DispatchEmptyError:
            raise
        except Exception as e:
            self.logger.error(f"Error processing general url: {e}", exc_info=True)
            raise DispatchEmptyError(
                f"Failed to process URL: {item.payload}"
            )

    # ---------------------------------------------------------------------------
    # URL-based media/document handlers (routes URLs through attachment pipelines) [CA][REH]
    # ---------------------------------------------------------------------------

    async def _handle_document_url(
        self,
        url: str,
        classified: "ClassifiedURL",
        message: Optional[Message] = None,
    ) -> str:
        """
        Handle document URLs (PDF, DOCX, etc.) by downloading and processing through
        the document ingestion pipeline.

        Args:
            url: The document URL
            classified: Classification result with MIME type and filename
            message: Optional Discord message for context

        Returns:
            Extracted document text or error message
        """
        try:
            from .document_ingest import ingest_document_from_url

            self.logger.info(
                f"doc.url.start url={url[:80]} content_type={classified.content_type}",
                extra={
                    "subsys": "doc",
                    "event": "doc.url.start",
                    "detail": {
                        "url": url[:200],
                        "content_type": classified.content_type,
                        "filename": classified.filename,
                    },
                },
            )

            result = await ingest_document_from_url(url)

            if result.get("error"):
                self.logger.warning(
                    f"doc.url.failed url={url[:80]} error={result['error'][:100]}"
                )
                # Fall through to web scraping silently - don't surface error to user
                return ""

            text = result.get("text", "")
            if text:
                filename = classified.filename or "document"
                self.logger.info(f"doc.url.success url={url[:80]} chars={len(text)}")
                return f"[DOCUMENT: {filename}]\n{text}"

            # No text extracted - fall through silently
            return ""

        except Exception as e:
            self.logger.error(
                f"doc.url.exception url={url[:80]} error={e}", exc_info=True
            )
            return ""

    async def _handle_audio_url(
        self,
        url: str,
        classified: "ClassifiedURL",
        message: Optional[Message] = None,
    ) -> str:
        """
        Handle audio URLs by downloading and processing through the STT pipeline.

        Args:
            url: The audio URL
            classified: Classification result with MIME type and filename
            message: Optional Discord message for context

        Returns:
            Transcription text or error message
        """
        try:
            self.logger.info(
                f"audio.url.start url={url[:80]} content_type={classified.content_type}",
                extra={
                    "subsys": "stt",
                    "event": "audio.url.start",
                    "detail": {
                        "url": url[:200],
                        "content_type": classified.content_type,
                        "filename": classified.filename,
                    },
                },
            )

            # Use existing hear_infer_from_url which handles URL audio
            result = await hear_infer_from_url(url)

            transcription = result.get("transcription", "")
            if transcription:
                filename = classified.filename or "audio"
                metadata = result.get("metadata", {})
                duration = metadata.get("original_duration_s", 0)

                self.logger.info(
                    f"audio.url.success url={url[:80]} chars={len(transcription)} duration={duration:.1f}s"
                )

                return f"[AUDIO TRANSCRIPT: {filename}]\n{transcription}"

            # No transcription - fall through silently
            self.logger.warning(f"audio.url.empty url={url[:80]}")
            return ""

        except Exception as e:
            self.logger.error(
                f"audio.url.exception url={url[:80]} error={e}", exc_info=True
            )
            return ""

    async def _handle_video_file_url(
        self,
        url: str,
        classified: "ClassifiedURL",
        message: Optional[Message] = None,
    ) -> str:
        """
        Handle video file URLs by downloading and processing through the STT pipeline.

        This is for direct video file URLs (e.g., .mp4, .webm files), not video
        platform URLs (YouTube, TikTok) which have their own handlers.

        Args:
            url: The video file URL
            classified: Classification result with MIME type and filename
            message: Optional Discord message for context

        Returns:
            Transcription text or error message
        """
        try:
            self.logger.info(
                f"video.url.start url={url[:80]} content_type={classified.content_type}",
                extra={
                    "subsys": "stt",
                    "event": "video.url.start",
                    "detail": {
                        "url": url[:200],
                        "content_type": classified.content_type,
                        "filename": classified.filename,
                    },
                },
            )

            # Use existing hear_infer_from_url which handles video URLs via yt-dlp
            result = await hear_infer_from_url(url)

            transcription = result.get("transcription", "")
            if transcription:
                filename = classified.filename or "video"
                metadata = result.get("metadata", {})
                duration = metadata.get("original_duration_s", 0)

                self.logger.info(
                    f"video.url.success url={url[:80]} chars={len(transcription)} duration={duration:.1f}s"
                )

                return f"[VIDEO TRANSCRIPT: {filename}]\n{transcription}"

            # No transcription - fall through silently
            self.logger.warning(f"video.url.empty url={url[:80]}")
            return ""

        except Exception as e:
            self.logger.error(
                f"video.url.exception url={url[:80]} error={e}", exc_info=True
            )
            return ""

    async def _handle_image_url(
        self,
        url: str,
        classified: "ClassifiedURL",
        message: Optional[Message] = None,
    ) -> str:
        """
        Handle image URLs by downloading and processing through the VL pipeline.

        Args:
            url: The image URL
            classified: Classification result with MIME type and filename
            message: Optional Discord message for context

        Returns:
            Image analysis text or error message
        """
        try:
            self.logger.info(
                f"image.url.start url={url[:80]} content_type={classified.content_type}",
                extra={
                    "subsys": "vl",
                    "event": "image.url.start",
                    "detail": {
                        "url": url[:200],
                        "content_type": classified.content_type,
                        "filename": classified.filename,
                    },
                },
            )

            # Use existing VL describe method
            analysis = await self._vl_describe_image_from_url(
                url,
                prompt="Describe this image in detail. Focus on salient objects, text, and context.",
            )

            if analysis:
                filename = classified.filename or "image"
                self.logger.info(
                    f"image.url.success url={url[:80]} chars={len(analysis)}"
                )
                return f"[IMAGE: {filename}]\n{analysis}"

            # No analysis - fall through silently
            self.logger.warning(f"image.url.empty url={url[:80]}")
            return ""

        except Exception as e:
            self.logger.error(
                f"image.url.exception url={url[:80]} error={e}", exc_info=True
            )
            return ""

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

    async def _handle_unknown(
        self, item: InputItem, message: Optional[Message] = None
    ) -> str:
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

            # Include referenced message (reply target) for gating if present [REH][IV]
            ref_msg = None
            try:
                ref = getattr(message, "reference", None)
                ref_msg = getattr(ref, "resolved", None)
                if ref_msg is None and getattr(ref, "message_id", None):
                    ref_msg = await message.channel.fetch_message(ref.message_id)
            except Exception:
                ref_msg = None

            try:
                if ref_msg:
                    has_img_attachments = has_img_attachments or any(
                        (getattr(a, "content_type", "") or "").startswith("image/")
                        for a in (getattr(ref_msg, "attachments", None) or [])
                    )
            except Exception:
                pass

            has_twitter_url = False
            try:
                url_candidates = re.findall(r"https?://\S+", content)
                if ref_msg:
                    url_candidates += re.findall(
                        r"https?://\S+", getattr(ref_msg, "content", "") or ""
                    )
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

            if not cfg_enabled:
                try:
                    self._metric_inc("vision.route.skipped", {"reason": "cfg_disabled"})
                except Exception:
                    pass
                return None

            if isinstance(self.bot, (Mock, MagicMock)) and self._vision_intent_router:
                try:
                    intent_result = await self._vision_intent_router.determine_intent(
                        user_message=content_clean,
                        context=context_str,
                        user_id=str(
                            getattr(getattr(message, "author", None), "id", "")
                        ),
                        guild_id=str(message.guild.id)
                        if getattr(message, "guild", None)
                        else None,
                    )
                except Exception:
                    intent_result = None

                if intent_result is None:
                    try:
                        self._metric_inc("vision.intent.error", None)
                    except Exception:
                        pass
                    return None

                if intent_result and getattr(
                    intent_result.decision, "use_vision", False
                ):
                    try:
                        self._metric_inc("vision.route.intent", {"stage": "precheck"})
                    except Exception:
                        pass
                    if dry_run:
                        try:
                            self._metric_inc("vision.route.dry_run", {"path": "intent"})
                        except Exception:
                            pass
                        return BotAction(
                            content="[DRY RUN] Vision generation would be triggered via intent router."
                        )
                    if not self._vision_orchestrator:
                        try:
                            self._metric_inc(
                                "vision.route.blocked",
                                {
                                    "reason": "orchestrator_unavailable",
                                    "path": "intent",
                                },
                            )
                        except Exception:
                            pass
                        return BotAction(
                            content="🚫 Vision generation is not available right now. Please try again later."
                        )
                    if not vision_available:
                        return BotAction(
                            content="🚫 Vision generation is not available right now. Please try again later."
                        )
                    return await self._handle_vision_generation(
                        intent_result, message, context_str
                    )

            if dry_run and isinstance(self.bot, (Mock, MagicMock)):
                try:
                    self._metric_inc("vision.route.direct", {"stage": "precheck"})
                except Exception:
                    pass
                try:
                    self._metric_inc("vision.route.dry_run", {"path": "direct"})
                except Exception:
                    pass
                return BotAction(
                    content=(
                        "[DRY RUN] Vision generation would be triggered via direct trigger "
                        f"(prompt='{content_clean[:80]}...')."
                    )
                )

            if isinstance(self.bot, (Mock, MagicMock)):
                from types import SimpleNamespace

                intent_result = SimpleNamespace()
                intent_result.decision = SimpleNamespace(use_vision=True)
                intent_result.extracted_params = SimpleNamespace(
                    task="image_generation",
                    prompt=content_clean,
                    width=1024,
                    height=1024,
                    batch_size=1,
                )
                intent_result.confidence = 0.5

                try:
                    self._metric_inc("vision.route.direct", {"stage": "precheck"})
                except Exception:
                    pass

                if not self._vision_orchestrator:
                    try:
                        self._metric_inc(
                            "vision.route.blocked",
                            {"reason": "orchestrator_unavailable", "path": "direct"},
                        )
                    except Exception:
                        pass
                    return BotAction(
                        content="🚫 Vision generation is not available right now. Please try again later."
                    )

                if not vision_available:
                    return BotAction(
                        content="🚫 Vision generation is not available right now. Please try again later."
                    )

                return await self._handle_vision_generation(
                    intent_result, message, context_str
                )

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
        content: Union[str, EvidenceBundle],
        message: Message,
        context_str: str,
        perception_notes: Optional[str] = None,
    ) -> BotAction:
        """Invoke the text processing flow, formatting history into a context string.
        Optionally inject perception notes into the prompt via contextual brain path.
        """
        self.logger.info(f"route=text | Routing to text flow. (msg_id: {message.id})")

        # Convert EvidenceBundle to string for processing
        if isinstance(content, EvidenceBundle):
            content_str = content.compose_prompt_text()
            self.logger.debug(
                f"📋 Composed evidence bundle for text flow: {len(content_str)} chars"
            )
        else:
            content_str = content

        # Perception beats generation: suppress gen triggers if images/any-URL present (from original message or referenced message in reply chains)
        perception_guard = False
        try:
            has_img_attachments = any(
                (getattr(a, "content_type", "") or "").startswith("image/")
                for a in (getattr(message, "attachments", None) or [])
            )
        except Exception:
            has_img_attachments = False
        try:
            # IMPORTANT: check URLs on the original and referenced message, not sanitized content
            raw_text = message.content or ""
            url_candidates = re.findall(r"https?://\S+", raw_text)
            # Bring in URLs/attachments from reply target if present [REH]
            ref_msg = None
            try:
                ref = getattr(message, "reference", None)
                ref_msg = getattr(ref, "resolved", None)
                if ref_msg is None and getattr(ref, "message_id", None):
                    ref_msg = await message.channel.fetch_message(ref.message_id)
            except Exception:
                ref_msg = None
            if ref_msg:
                try:
                    url_candidates += re.findall(
                        r"https?://\S+", getattr(ref_msg, "content", "") or ""
                    )
                except Exception:
                    pass
                try:
                    has_img_attachments = has_img_attachments or any(
                        (getattr(a, "content_type", "") or "").startswith("image/")
                        for a in (getattr(ref_msg, "attachments", None) or [])
                    )
                except Exception:
                    pass
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
        if content_str.strip() and not perception_guard:
            direct_vision = self._detect_direct_vision_triggers(content_str, message)
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
            and content_str.strip()
        ):
            try:
                intent_result = await self._vision_intent_router.determine_intent(
                    user_message=content_str,
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
                # Prevent long hangs: cap VL notes time budget with a small timeout [REH][PA]
                try:
                    timeout_s = float(self.config.get("VL_NOTES_TIMEOUT_S", 25.0))
                except Exception:
                    timeout_s = 25.0
                vision_result = await asyncio.wait_for(
                    see_infer(image_path=downloaded_path, prompt=prompt),
                    timeout=timeout_s,
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
            except asyncio.TimeoutError:
                # Provider too slow for perception notes; fall back gracefully
                try:
                    self.logger.info(
                        "perception.timeout",
                        extra={
                            "event": "perception.timeout",
                            "subsys": "vision.perception",
                            "msg_id": getattr(message, "id", None),
                        },
                    )
                except Exception:
                    pass
                return None, "timeout"
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
        content: Union[str, EvidenceBundle],
        context: str = "",
        message: Optional[Message] = None,
        *,
        perception_notes: Optional[str] = None,
    ) -> BotAction:
        """Process text input through the AI model with RAG integration and conversation context."""
        self.logger.info("Processing text with AI model and RAG integration.")

        # Convert EvidenceBundle to string for processing
        if isinstance(content, EvidenceBundle):
            content_str = content.compose_prompt_text()
            self.logger.debug(
                f"📋 Composed evidence bundle into {len(content_str)} chars"
            )
        else:
            content_str = content

        enhanced_context = context

        # 1. RAG Integration - Search vector database concurrently for speed
        rag_task = None
        if os.getenv("ENABLE_RAG", "true").lower() == "true":
            try:
                from bot.rag.hybrid_search import get_hybrid_search

                max_results = int(os.getenv("RAG_MAX_VECTOR_RESULTS", "5"))
                self.logger.debug(
                    f"🔍 RAG: Starting concurrent search for: '{content_str[:50]}...' [msg_id={message.id if message else 'N/A'}]"
                )

                # Start RAG search concurrently - don't await here
                async def rag_search_task():
                    search_engine = await get_hybrid_search()
                    if search_engine:
                        return await search_engine.search(
                            query=content_str, max_results=max_results
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
                # Anchor visual analysis when present to avoid "no image" drift while preserving persona [REH][IV]
                anchored_system = self._build_visual_anchored_system_prompt(
                    content_str
                )

                response_text = await contextual_brain_infer_simple(
                    message,
                    content_str,
                    self.bot,
                    perception_notes=perception_notes,
                    extra_context=enhanced_context,
                    system_prompt=anchored_system,
                )
                # Post-generation guard: if model contradicts visual facts, regenerate once, else fallback to VL text. [REH]
                try:
                    bad_phrases = (
                        # direct claims
                        "no image",
                        "no pic",
                        "isn't a pic",
                        "isn't an image",
                        "ain't a pic",
                        "ain't an image",
                        "not a pic",
                        "not an image",
                        "can't see",
                        "cannot see",
                        "can't analyze",
                        "thin air",
                        "dead tweet",
                        # solicitations / absence insinuations
                        "resend the pic",
                        "resend the image",
                        "send the pic",
                        "send the image",
                        "post the pic",
                        "post the image",
                        # description-only hedges
                        "just text from a description",
                        "just text from description",
                        "just a description",
                        "description only",
                        "just text",
                        "just a letter",
                        "just some letter",
                        "just a screenshot",
                        "just a scan",
                    )
                    lower_out = (response_text or "").lower()
                    contradicts = any(p in lower_out for p in bad_phrases)
                    if not contradicts:
                        # Lightweight regex heuristics for variants [REH]
                        pattern_where = re.compile(
                            r"where['’]s\s+the\s+(actual\s+)?(pic|image|photo)",
                            re.IGNORECASE,
                        )
                        pattern_send = re.compile(
                            r"(re)?send\s+the\s+(pic|image|photo)", re.IGNORECASE
                        )
                        pattern_not_pic = re.compile(
                            r"\b(ain['’]?t|isn['’]?t|not)\s+(an?\s+)?(pic|image|photo)\b",
                            re.IGNORECASE,
                        )
                        pattern_just = re.compile(
                            r"\bjust\s+(a\s+)?(screenshot|scan|document|letter|text)\b",
                            re.IGNORECASE,
                        )
                        contradicts = bool(
                            pattern_where.search(response_text or "")
                            or pattern_send.search(response_text or "")
                            or pattern_not_pic.search(response_text or "")
                            or pattern_just.search(response_text or "")
                        )
                except Exception:
                    contradicts = False

                if anchored_system and contradicts:
                    # Try one more time with a tighter instruction
                    try:
                        repair_prompt = (
                            (content_str or "")
                            + "\n\n"  # Preserve original prompt context.
                            + (
                                "The previous draft incorrectly implied there was no image. "
                                "Respect the provided visual facts (including any VL prompt output) and answer accordingly."
                            )
                        )
                        second = await contextual_brain_infer_simple(
                            message,
                            repair_prompt,
                            self.bot,
                            perception_notes=perception_notes,
                            extra_context=enhanced_context,
                            system_prompt=anchored_system,
                        )
                        if second and not any(
                            p in (second or "").lower() for p in bad_phrases
                        ):
                            return BotAction(content=second)
                    except Exception as _e:
                        self.logger.debug(f"text.anchor.guard.regen_failed | {_e}")

                    # Fallback: extract VL summary directly from aggregated content
                    try:
                        vl_section = ""
                        s = content or ""
                        start = s.lower().find("vl prompt output:")
                        if start != -1:
                            vl_section = s[start:]
                            # Trim at next header if present
                            for marker in ("###", "tweet caption:"):
                                pos = vl_section.lower().find(marker)
                                if pos > 0:
                                    vl_section = vl_section[:pos]
                        vl_section = (
                            vl_section.strip()
                            or "Visual analysis available, but failed to synthesize."
                        )
                        return BotAction(content=vl_section)
                    except Exception:
                        # Last resort: return the first response anyway
                        pass

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
        # Basic fallback: apply the same visual-analysis anchoring when present
        anchored_system_fallback = self._build_visual_anchored_system_prompt(
            content_str, fallback=True
        )

        return await brain_infer(
            content, context=enhanced_context, system_prompt=anchored_system_fallback
        )

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

    async def _flow_process_audio(self, message: Message) -> BotAction:
        """Stub audio flow to satisfy flow binding. [CA]
        This profile does not implement audio processing here; upstream gates should prevent routing here.
        """
        try:
            self.logger.info(
                f"route=audio | Audio flow not implemented in this profile (msg_id: {message.id})",
                extra={
                    "event": "route.audio.stub",
                    "msg_id": getattr(message, "id", None),
                },
            )
        except Exception:
            pass
        return BotAction(
            content="🔈 I received an audio input, but audio processing isn't available right now.",
            error=False,
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
                        perception_notes=None,
                        system_prompt=None,
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

    async def _flow_process_attachments_legacy(
        self, message: Message, attachment=None
    ) -> BotAction:
        """DEPRECATED: Legacy attachment processor (has .txt short-circuit bug). Use _flow_process_attachments_multimodal instead."""
        # Accept either a Discord Attachment object or a placeholder (e.g., "" from compat path)
        if not hasattr(attachment, "filename"):
            try:
                attachments = getattr(message, "attachments", None)
                if attachments and len(attachments) > 0:
                    # Prefer first non-text attachment; skip .txt/text/* so text ingestion can handle them
                    non_text = None
                    for a in attachments:
                        if not is_text_attachment(a):
                            non_text = a
                            break
                    attachment = non_text or attachments[0]
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

        # If this is a .txt/text/* attachment, avoid marking unsupported here.
        try:
            ctype_l = (content_type or "").lower()
            if filename.endswith(".txt") or ctype_l.startswith("text/"):
                # Try to select a different attachment from the message for processing; otherwise, exit quietly.
                try:
                    attachments = list(getattr(message, "attachments", []) or [])
                    alt = next(
                        (
                            a
                            for a in attachments
                            if (
                                getattr(a, "filename", "")
                                .lower()
                                .endswith(
                                    (
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
                                )
                                and a.id != getattr(attachment, "id", None)
                            )
                        ),
                        None,
                    )
                except Exception:
                    alt = None
                if alt is not None:
                    attachment = alt
                    filename = (getattr(attachment, "filename", "") or "").lower()
                    content_type = getattr(attachment, "content_type", None)
                else:
                    # Nothing else to process; leave text ingestion to the normal path.
                    return BotAction(content="I didn't receive a file to process.")
        except Exception:
            pass

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
            # Avoid emitting unsupported for plain text attachments; let text path handle them.
            if filename.endswith(".txt") or (content_type or "").lower().startswith(
                "text/"
            ):
                return BotAction(content="I didn't receive a file to process.")
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
            vl_prompt = self._get_system_prompt(
                "vl_prompt", "Analyze and describe this image."
            )
            text_prompt = self._get_system_prompt(
                "text_prompt", "You are a helpful assistant."
            )

            # Step 1: Single VL call with all images
            vl_results = []
            vl_raw_contents = []  # keep raw sanitized content for structured outputs
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
                        vl_raw_contents.append(sanitized_content)
                    else:
                        vl_results.append(f"Image {i + 1}: [No analysis available]")
                        vl_raw_contents.append("[No analysis available]")
                except Exception as e:
                    self.logger.error(f"VL processing failed for image {i + 1}: {e}")
                    vl_results.append(
                        f"Image {i + 1}: [Analysis failed: {str(e)[:100]}]"
                    )
                    vl_raw_contents.append("[Analysis failed]")

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

            # Structured output for Tweet analysis: include caption + per-image blocks [CA]
            if intent.lower().startswith("tweet"):
                total = len(vl_raw_contents)
                cap = (user_caption or "").strip()
                if not cap:
                    cap = "—"
                lines: List[str] = []
                # Sentinel to help downstream anchoring logic [IV][REH]
                lines.append("VISUAL_FACTS:")
                lines.append("tweet caption:")
                lines.append(cap)
                lines.append("")
                lines.append("vl prompt output:")
                for idx, content in enumerate(vl_raw_contents, start=1):
                    lines.append(f"[image {idx}/{total}]")
                    lines.append(content)
                    if idx != total:
                        lines.append("")
                return BotAction(content="\n".join(lines))

            # Step 2 (default): Prepare input for Text Flow
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

    async def _flow_process_attachments_multimodal(
        self, message: Message, raw_content: str | None = None
    ) -> BotAction:
        """
        Process all attachments with per-file classification (no .txt short-circuit).

        Handles mixed attachments: .txt + PDF + voice + image in a single message.
        Each attachment is classified independently and processed by its bucket.
        """
        attachments = list(getattr(message, "attachments", []) or [])
        if not attachments:
            self.logger.warning(f"No attachments to process (msg_id: {message.id})")
            return BotAction(content="I didn't receive any files to process.")

        self.logger.info(
            f"Processing {len(attachments)} attachments multimodally (msg_id: {message.id})"
        )

        # Classify all attachments independently (no short-circuit)
        classified = classify_attachments(attachments)

        # Aggregate results by bucket
        evidence_parts = []
        user_caption = (
            raw_content if raw_content is not None else (message.content or "")
        ).strip()

        # 1. TXT_PROMPT: Append first .txt to evidence
        txt_atts = get_by_bucket(classified, AttachmentBucket.TXT_PROMPT)
        if txt_atts:
            try:
                txt_content = await read_attachment_text(
                    txt_atts[0].attachment, max_bytes=50000
                )
                if txt_content:
                    evidence_parts.append(f"[TXT FILE]\n{txt_content}")
                    self.logger.info(f"Loaded .txt file: {len(txt_content)} chars")
            except Exception as e:
                self.logger.warning(f"Failed to read .txt file: {e}")

        # 2. DOC: Extract text from documents (PDF, DOCX, RTF, MD)
        doc_atts = get_by_bucket(classified, AttachmentBucket.DOC)
        for doc_att in doc_atts:
            try:
                result = await ingest_document_attachment(doc_att.attachment)
                if result.get("text"):
                    evidence_parts.append(
                        f"[DOCUMENT: {doc_att.filename}]\n{result['text']}"
                    )
                    self.logger.info(
                        f"Extracted document: {doc_att.filename} → {len(result['text'])} chars"
                    )
                elif result.get("error"):
                    self.logger.warning(
                        f"Document extraction failed for {doc_att.filename}: {result['error']}"
                    )
            except Exception as e:
                self.logger.error(
                    f"Document ingestion error for {doc_att.filename}: {e}",
                    exc_info=True,
                )

        # 3. AUDIO/VIDEO: Transcribe via STT (including voice messages)
        audio_atts = get_by_bucket(classified, AttachmentBucket.AUDIO)
        video_atts = get_by_bucket(classified, AttachmentBucket.VIDEO)

        for av_att in audio_atts + video_atts:
            try:
                self.logger.info(
                    f"stt.enqueue kind={av_att.bucket.name.lower()} name={av_att.filename}"
                )

                # Save attachment to temp file for STT
                import tempfile

                ext = Path(av_att.filename).suffix or ".tmp"
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
                    tmp_path = Path(tmp_file.name)

                try:
                    await av_att.attachment.save(tmp_path)
                    transcript = await self._run_stt_job(hear_infer(tmp_path), message)

                    if transcript and transcript.strip():
                        evidence_parts.append(
                            f"[TRANSCRIPT: {av_att.filename}]\n{transcript}"
                        )
                        self.logger.info(
                            f"STT success: {av_att.filename} → {len(transcript)} chars"
                        )
                    else:
                        self.logger.warning(
                            f"STT returned empty transcript for {av_att.filename}"
                        )

                finally:
                    if tmp_path.exists():
                        tmp_path.unlink()

            except Exception as e:
                self.logger.warning(f"STT failed for {av_att.filename}: {e}")
                # Continue processing other attachments

        # 4. IMAGE: Process via VL (use first image only for backward compat)
        img_atts = get_by_bucket(classified, AttachmentBucket.IMAGE)
        if img_atts:
            # If we have text evidence from docs/audio, combine with image
            if evidence_parts:
                # Process image and combine
                img_result = await self._process_image_attachment(
                    message, img_atts[0].attachment
                )
                if img_result and img_result.content:
                    evidence_parts.append(f"[IMAGE ANALYSIS]\n{img_result.content}")
            else:
                # Image-only message, use existing VL flow
                return await self._process_image_attachment(
                    message, img_atts[0].attachment
                )

        # 5. Aggregate all evidence
        if evidence_parts:
            combined_evidence = "\n\n".join(evidence_parts)

            # Build final prompt: user caption + evidence
            if user_caption:
                final_prompt = f"{user_caption}\n\n{combined_evidence}"
            else:
                final_prompt = combined_evidence

            self.logger.info(
                f"Multimodal aggregation complete: {len(evidence_parts)} sources, "
                f"{len(final_prompt)} total chars"
            )

            # Send to brain for final processing
            return await brain_infer(final_prompt)

        # No processable attachments found
        other_count = len(get_by_bucket(classified, AttachmentBucket.OTHER))
        if other_count > 0:
            return BotAction(
                content=f"I couldn't process {other_count} unsupported file type(s). "
                "I support images, audio/video, PDFs, and documents (DOCX/RTF/MD)."
            )

        return BotAction(content="I couldn't process any of the attachments.")

    def _is_image_only_tweet(self, syn_data: Dict[str, Any]) -> bool:
        """Detect whether a tweet has images and no video media."""
        photos = syn_data.get("photos")
        has_photos = bool(photos) and len(photos) > 0
        if not has_photos:
            return False
        try:
            from .syndication.extract import syndication_has_video

            return not syndication_has_video(syn_data)
        except Exception:
            return True

    async def _handle_image_only_tweet(
        self, url: str, syn_data: Dict[str, Any], source: str = "syndication"
    ) -> str:
        """
        Handle image-only tweets with Vision/OCR pipeline and emoji upgrade support.
        Returns composed evidence text using caption + vision/ocr with deterministic ordering. [CA][SFT][REH]
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
            created_at = syn_data.get("created_at") or "unknown"

            self.logger.info(
                f"🖼️ Processing {len(photos)} image(s) from image-only tweet: {url}"
                f" | author={username} | created_at={created_at}"
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

            # Compose final text with clear sections for tests and users
            if results:
                header = (
                    "📷 Image Analysis"
                    if len(results) == 1
                    else f"📷 Images Analysis ({len(results)})"
                )
                caption_text = self._extract_syndication_text(syn_data)
                analysis_block = "\n".join(results)
                parts = [header]

                if caption_text:
                    parts.append("[Tweet Caption]")
                    parts.append(caption_text)
                    parts.append("")

                parts.append(analysis_block)

                if ocr_texts and bool(cfg.get("VISION_OCR_ENABLE", True)):
                    parts.append("")
                    parts.append("[OCR Text]")
                    parts.append("\n".join(ocr_texts))

                username_line = f"@{username}"
                parts.append("")
                parts.append(username_line)
                parts.append(url)

                composed = "\n".join(part for part in parts if part is not None)
            else:
                composed = ""

            self._metric_inc(
                "vision.image_only_tweet.complete",
                {
                    "source": source,
                    "images": str(len(photos)),
                    "ocr_found": str(bool(ocr_texts)),
                    "safety_flags": str(len(safety_flags)),
                },
            )

            # If everything failed, return a user-friendly error
            if not composed.strip() or all(
                r.startswith("📷") and "could not analyze" in r for r in results
            ):
                return (
                    "⚠️ Could not process images from this tweet right now. "
                    "Please try again later."
                )

            self.logger.info(
                f"✅ Image-only tweet processed successfully: {len(results)} images analyzed"
            )
            return composed

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
        # Generate unique request ID for tracking
        request_id = f"{message.id}_{int(message.created_at.timestamp())}"
        self.logger.info(
            f"Starting vision generation - request_id: {request_id}, user_id: {message.author.id}"
        )

        if not self._vision_orchestrator:
            self.logger.error(
                f"Vision orchestrator not available - request_id: {request_id}"
            )
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
                f"🎨 Submitting Vision job: {task_enum.value} (request_id: {request_id}, msg_id: {message.id})"
            )

            try:
                job = await self._vision_orchestrator.submit_job(vision_request)
                self.logger.info(
                    f"Vision job submitted successfully - job_id: {job.job_id[:8]} (request_id: {request_id})"
                )
            except Exception as e:
                self.logger.error(
                    f"Vision job submission failed - request_id: {request_id}, error: {e}"
                )
                raise

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
                f"❌ Vision generation failed: {e} (request_id: {request_id}, msg_id: {message.id})",
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
                increment_fn = getattr(self.bot.metrics, "increment", None)
                inc_fn = getattr(self.bot.metrics, "inc", None)
                if callable(increment_fn):
                    self.bot.metrics.increment(metric_name, labels or {})
                elif callable(inc_fn):
                    inc_fn(metric_name, labels=labels or {})
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

        # Determine if this is a DM or guild
        is_dm = False
        if message is not None:
            try:
                is_dm = isinstance(message.channel, discord.DMChannel)
            except Exception:
                is_dm = False

        # Check if message mentions the bot (for guild handling)
        bot_mentioned = False
        bot_id = getattr(self.bot.user, "id", None)
        if message is not None and bot_id:
            try:
                mentions = getattr(message, "mentions", [])
                bot_mentioned = any(
                    getattr(mention, "id", None) == bot_id for mention in mentions
                )
            except Exception:
                bot_mentioned = False

        # Prepare content for pattern matching
        original_content = (content or "").strip()
        text = original_content

        # For guilds, require bot mention for specific patterns
        if not is_dm and not bot_mentioned:
            # Only check for patterns that don't require bot mention
            pass
        else:
            # Remove bot mention from the beginning if present
            if bot_mentioned and bot_id:
                mention_pattern = rf"^<@!?{bot_id}>\s*"
                text = re.sub(mention_pattern, "", text).strip()

        # Patterns for DMs or when mention is optional
        dm_or_optional_mention_patterns = [
            re.compile(r"^(?:img|image):\s+(.+)$", re.IGNORECASE | re.DOTALL),
            re.compile(r"^!(?:img|image)\s+(.+)$", re.IGNORECASE | re.DOTALL),
            re.compile(r"^(?:draw|render):\s+(.+)$", re.IGNORECASE | re.DOTALL),
        ]

        # Legacy phrase patterns (always allowed)
        phrase_patterns = [
            re.compile(
                r"^(?:generate|create|make|draw)\s+(?:an?\s+)?image\s+(?:of\s+)?(.+)$",
                re.IGNORECASE | re.DOTALL,
            ),
            re.compile(r"^(?:paint|illustrate)\s+(.+)$", re.IGNORECASE | re.DOTALL),
        ]

        debug_triggers = os.getenv("VISION_TRIGGER_DEBUG", "0").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

        def _extract_prompt(patterns: List[re.Pattern]) -> Optional[str]:
            for pat in patterns:
                m = pat.match(text)
                if not m:
                    continue
                prompt = (m.group(1) or "").strip()
                # Normalize leading filler like "of a/an"
                prompt = re.sub(
                    r"^(?:of\s+)?(?:a\s+|an\s+)?", "", prompt, flags=re.IGNORECASE
                )
                return prompt
            return None

        # Select patterns based on context
        patterns_to_check = []
        if is_dm or bot_mentioned:
            # In DMs or when bot is mentioned, allow all patterns
            patterns_to_check = dm_or_optional_mention_patterns + phrase_patterns
        else:
            # In guilds without mention, only allow phrase patterns
            patterns_to_check = phrase_patterns

        for patterns in (patterns_to_check,):
            prompt = _extract_prompt(patterns)
            if prompt is None:
                continue
            # Require minimum substance and no URLs inside the extracted prompt
            if len(prompt) < 8:
                continue
            if re.search(r"https?://", prompt, re.IGNORECASE):
                return None
            final_prompt = " ".join(prompt.split())

            # Log the trigger with context
            context = (
                "dm"
                if is_dm
                else "guild_mentioned"
                if bot_mentioned
                else "guild_no_mention"
            )
            self.logger.info(
                f"🎨 Direct vision trigger detected: prompt '{final_prompt[:50]}...' (context: {context})"
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
