"""
Change Summary:
- Created new modality.py module to support sequential multimodal processing
- Introduced InputItem dataclass for unified input handling across attachments, URLs, and embeds
- Implemented collect_input_items() to gather ALL inputs from a message in textual order
- Implemented map_item_to_modality() for robust per-item modality detection
- Extended InputModality enum with additional modalities for comprehensive coverage
- Added comprehensive logging and error handling for modality detection

This replaces the single-shot modality detection with a multi-pass system that processes
each input item sequentially, enabling full multimodal support.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import lru_cache
from typing import List, Literal, Union, TYPE_CHECKING, Optional, Pattern

import discord
from .util.logging import get_logger

if TYPE_CHECKING:
    from discord import Message

logger = get_logger(__name__)

# Pre-compile and cache regex patterns for performance [PA]
_URL_PATTERN = re.compile(r'https?://[^\s<>"\'\'\[\]{}|\\^`]+')
_IMAGE_EXT_PATTERN = re.compile(r"\.(jpg|jpeg|png|gif|webp|bmp)(\?.*)?$", re.IGNORECASE)
_PDF_EXT_PATTERN = re.compile(r"\.pdf(\?.*)?$", re.IGNORECASE)

# Cache for video patterns - loaded once and reused [PA]
_VIDEO_PATTERNS: Optional[List[Pattern[str]]] = None
_FALLBACK_PATTERNS = [
    re.compile(r"https?://(?:www\.)?youtube\.com/watch\?(?:.*&)?v=[\w-]+"),
    re.compile(r"https?://youtu\.be/[\w-]+"),
    re.compile(r"https?://(?:www\.)?tiktok\.com/@[\w.-]+/video/\d+"),
    re.compile(r"https?://(?:www\.)?tiktok\.com/t/[\w-]+"),
    re.compile(r"https?://(?:m|vm)\.tiktok\.com/[\w-]+"),
]


class InputModality(Enum):
    """Defines the type of input the bot is processing."""

    TEXT_ONLY = auto()
    SINGLE_IMAGE = auto()
    MULTI_IMAGE = auto()
    VIDEO_URL = auto()  # YouTube/TikTok URLs for audio transcription
    AUDIO_VIDEO_FILE = auto()  # Audio/video file attachments
    PDF_DOCUMENT = auto()
    PDF_OCR = auto()  # Image-only PDFs requiring OCR
    GENERAL_URL = auto()
    SCREENSHOT_URL = auto()  # URLs that need screenshot fallback
    UNKNOWN = auto()  # Fallback for unrecognized inputs


@dataclass
class InputItem:
    """Unified abstraction for all input types (attachments, URLs, embeds)."""

    source_type: Literal["attachment", "url", "embed"]
    payload: Union[discord.Attachment, str, discord.Embed]
    order_index: int  # Preserve original message order

    def __post_init__(self):
        """Validate input item data [IV]."""
        if self.source_type not in ("attachment", "url", "embed"):
            raise ValueError(f"Invalid source_type: {self.source_type}")
        if self.order_index < 0:
            raise ValueError(f"Invalid order_index: {self.order_index}")


def collect_input_items(message: Message) -> List[InputItem]:
    """
    Collect all input items from a message in textual order.

    Args:
        message: Discord message to process

    Returns:
        List of InputItem objects in order of appearance
    """
    items = []
    order_index = 0

    # Input validation [IV]
    if not message or not hasattr(message, "content"):
        logger.warning("Invalid message object provided")
        return []

    # Extract URLs from message content in order of appearance [PA]
    urls = _URL_PATTERN.findall(message.content)

    # Add URLs first (they appear in text content)
    for url in urls:
        items.append(InputItem(source_type="url", payload=url, order_index=order_index))
        order_index += 1

    # Add attachments (they appear after text content)
    for attachment in message.attachments:
        items.append(
            InputItem(
                source_type="attachment", payload=attachment, order_index=order_index
            )
        )
        order_index += 1

    # Add embeds (they appear last)
    for embed in message.embeds:
        # Deduplicate: skip embeds that are previews of video URLs already in message content [IV][PA]
        try:
            embed_url = getattr(embed, "url", None)
            if embed_url:
                # If the embed points to a known video platform URL that is already present in text, skip it
                video_hosts = (
                    "tiktok.com",
                    "vm.tiktok.com",
                    "m.tiktok.com",
                    "youtube.com",
                    "youtu.be",
                    "twitter.com",
                    "x.com",
                    "fxtwitter.com",
                    "vxtwitter.com",
                )

                def _host_match(u: str) -> bool:
                    return any(h in u for h in video_hosts)

                if _host_match(embed_url) and any(_host_match(u) for u in urls):
                    logger.info(
                        "🧹 Skipping video preview embed; corresponding video URL present in message"
                    )
                    continue
        except Exception as _e:
            # Non-fatal: proceed to include the embed if dedupe check fails [REH]
            logger.debug(f"Embed dedupe check failed: {_e}")

        items.append(
            InputItem(source_type="embed", payload=embed, order_index=order_index)
        )
        order_index += 1

    logger.info(f"📋 Collected {len(items)} input items from message {message.id}")
    return items


@dataclass
class ImageRef:
    """Reference to an image with metadata for high-res harvesting."""

    url: str
    filename: Optional[str] = None
    content_type: Optional[str] = None
    fallback_urls: List[str] = field(default_factory=list)


def collect_image_urls_from_message(message: Message) -> List[ImageRef]:
    """
    Harvest high-resolution images from a message (attachments + image embeds) with fallback candidates.

    Args:
        message: Discord message to harvest images from

    Returns:
        List of ImageRef objects with deduplication and fallback URLs
    """
    images = []
    seen_urls = set()

    # Input validation [IV]
    if not message or not hasattr(message, "attachments"):
        logger.warning("Invalid message object provided for image harvesting")
        return []

    # 1. Attachments first (highest priority - full resolution)
    for attachment in message.attachments:
        if attachment.content_type and attachment.content_type.startswith("image/"):
            if attachment.url not in seen_urls:
                # Build candidate URLs for robust fetching
                candidates = [attachment.url]  # Full CDN URL first

                # Add proxy URL as fallback
                if hasattr(attachment, "proxy_url") and attachment.proxy_url:
                    candidates.append(attachment.proxy_url)

                # Add 4096 variant for media.discordapp.net
                if (
                    "media.discordapp.net" in attachment.url
                    and "?format=" not in attachment.url
                ):
                    candidates.append(f"{attachment.url}?format=png&size=4096")

                images.append(
                    ImageRef(
                        url=candidates[0],  # Primary URL
                        filename=attachment.filename,
                        content_type=attachment.content_type,
                        fallback_urls=candidates[1:] if len(candidates) > 1 else [],
                    )
                )
                seen_urls.add(attachment.url)

    # 2. Image embeds (use .image.url for full size, with proxy fallback)
    for embed in message.embeds:
        if embed.type in ("image", "rich") and embed.image and embed.image.url:
            if embed.image.url not in seen_urls:
                candidates = [embed.image.url]

                # Add thumbnail proxy as fallback if available
                if embed.thumbnail and embed.thumbnail.proxy_url:
                    candidates.append(embed.thumbnail.proxy_url)

                images.append(
                    ImageRef(
                        url=candidates[0],
                        filename=None,  # Embeds usually don't have filenames
                        content_type="image/*",  # Generic type
                        fallback_urls=candidates[1:] if len(candidates) > 1 else [],
                    )
                )
                seen_urls.add(embed.image.url)

    return images


async def map_item_to_modality(item: InputItem) -> InputModality:
    """
    Map an input item to its appropriate modality.

    Args:
        item: InputItem to analyze

    Returns:
        InputModality enum value
    """
    try:
        if item.source_type == "attachment":
            return await _map_attachment_to_modality(item.payload)
        elif item.source_type == "url":
            return await _map_url_to_modality(item.payload)
        elif item.source_type == "embed":
            return await _map_embed_to_modality(item.payload)
        else:
            logger.warning(f"Unknown source_type: {item.source_type}")
            return InputModality.UNKNOWN

    except Exception as e:
        logger.error(f"Error mapping item to modality: {e}", exc_info=True)
        return InputModality.UNKNOWN


async def _map_attachment_to_modality(attachment: discord.Attachment) -> InputModality:
    """Map attachment to modality based on content type and filename."""
    # Cache string operations [PA]
    content_type = attachment.content_type or ""
    filename_lower = attachment.filename.lower()

    # Image attachments
    if content_type.startswith("image/"):
        return InputModality.SINGLE_IMAGE

    # PDF documents
    if filename_lower.endswith(".pdf"):
        # For now, assume all PDFs are text-based
        # TODO: Implement OCR detection logic
        return InputModality.PDF_DOCUMENT

    # Audio/video files
    if content_type.startswith(("audio/", "video/")) or filename_lower.endswith(
        (".mp3", ".wav", ".mp4", ".avi", ".mov", ".mkv", ".webm")
    ):
        return InputModality.AUDIO_VIDEO_FILE

    # Document files (future expansion)
    if filename_lower.endswith((".docx", ".txt", ".rtf")):
        return InputModality.PDF_DOCUMENT  # Reuse document processing

    logger.warning(f"Unrecognized attachment type: {filename_lower} ({content_type})")
    return InputModality.UNKNOWN


async def _map_url_to_modality(url: str) -> InputModality:
    """Map URL to modality based on domain and pattern matching [PA]."""
    global _VIDEO_PATTERNS

    # Twitter/X status posts should go through API-first general URL path [SFT][CA]
    # but allow broadcasts (Spaces/live) to be handled as video-capable URLs.
    if re.search(r"https?://(?:www\.)?(?:twitter|x)\.com/.+/status/\d+", url):
        logger.info(
            f"➡️ Routing Twitter/X status URL to GENERAL_URL for API-first: {url}"
        )
        return InputModality.GENERAL_URL

    # Load and cache video patterns once [PA]
    if _VIDEO_PATTERNS is None:
        try:
            from .video_ingest import SUPPORTED_PATTERNS

            # Pre-compile patterns for performance [PA]
            _VIDEO_PATTERNS = [re.compile(pattern) for pattern in SUPPORTED_PATTERNS]
            logger.debug(
                f"🎥 Loaded and compiled {len(_VIDEO_PATTERNS)} video patterns"
            )
        except ImportError as e:
            logger.warning(
                f"Could not import SUPPORTED_PATTERNS from video_ingest: {e}, using fallback patterns"
            )
            _VIDEO_PATTERNS = _FALLBACK_PATTERNS

    # Test video patterns with compiled regex [PA]
    for pattern in _VIDEO_PATTERNS:
        if pattern.search(url):  # Fixed: use .search() consistently [REH]
            logger.info(f"✅ Video URL modality detected: {url}")
            return InputModality.VIDEO_URL

    # Image URLs [PA]
    if _IMAGE_EXT_PATTERN.search(url):
        return InputModality.SINGLE_IMAGE

    # PDF URLs [PA]
    if _PDF_EXT_PATTERN.search(url):
        return InputModality.PDF_DOCUMENT

    # General URLs (will try oEmbed first, fallback to screenshot)
    return InputModality.GENERAL_URL


async def _map_embed_to_modality(embed: discord.Embed) -> InputModality:
    """Map embed to modality based on embed type and content."""
    # Image embeds
    if embed.image or embed.thumbnail:
        return InputModality.SINGLE_IMAGE

    # Video embeds
    if embed.video:
        return InputModality.VIDEO_URL

    # General embeds (treat as URL content)
    return InputModality.GENERAL_URL


async def detect_modality(message: Message) -> InputModality:
    """
    Legacy function for backward compatibility.
    Returns the modality of the first detected input item.

    Args:
        message: Discord message to analyze

    Returns:
        InputModality of first item, or TEXT_ONLY if no items
    """
    items = collect_input_items(message)
    if not items:
        return InputModality.TEXT_ONLY

    return await map_item_to_modality(items[0])


# New async bulk processing functions [PA]
async def map_items_to_modalities_concurrent(
    items: List[InputItem],
) -> List[InputModality]:
    """
    Map multiple items to modalities concurrently for performance [PA].

    Args:
        items: List of InputItem objects to process

    Returns:
        List of InputModality enum values in same order
    """
    if not items:
        return []

    # Process items concurrently [PA]
    tasks = [map_item_to_modality(item) for item in items]
    modalities = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle any exceptions [REH]
    results = []
    for i, modality in enumerate(modalities):
        if isinstance(modality, Exception):
            logger.error(f"Error processing item {i}: {modality}", exc_info=True)
            results.append(InputModality.UNKNOWN)
        else:
            results.append(modality)

    return results


@lru_cache(maxsize=1000)  # Cache frequent pattern checks [PA]
def _cached_url_check(url: str, pattern_type: str) -> bool:
    """
    Cached URL pattern checking for frequently accessed URLs [PA].

    Args:
        url: URL to check
        pattern_type: Type of pattern ('image', 'pdf')

    Returns:
        True if URL matches pattern
    """
    if pattern_type == "image":
        return bool(_IMAGE_EXT_PATTERN.search(url))
    elif pattern_type == "pdf":
        return bool(_PDF_EXT_PATTERN.search(url))
    return False
