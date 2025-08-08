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

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Literal, Union, TYPE_CHECKING

import discord
from .util.logging import get_logger

if TYPE_CHECKING:
    from discord import Message, Attachment, Embed

logger = get_logger(__name__)


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
    
    # Extract URLs from message content in order of appearance
    url_pattern = r'https?://[^\s<>"\'\[\]{}|\\^`]+'
    urls = re.findall(url_pattern, message.content)
    
    # Add URLs first (they appear in text content)
    for url in urls:
        items.append(InputItem(
            source_type="url",
            payload=url,
            order_index=order_index
        ))
        order_index += 1
    
    # Add attachments (they appear after text content)
    for attachment in message.attachments:
        items.append(InputItem(
            source_type="attachment", 
            payload=attachment,
            order_index=order_index
        ))
        order_index += 1
    
    # Add embeds (they appear last)
    for embed in message.embeds:
        items.append(InputItem(
            source_type="embed",
            payload=embed,
            order_index=order_index
        ))
        order_index += 1
    
    logger.info(f"📋 Collected {len(items)} input items from message {message.id}")
    return items


def map_item_to_modality(item: InputItem) -> InputModality:
    """
    Map an input item to its appropriate modality.
    
    Args:
        item: InputItem to analyze
        
    Returns:
        InputModality enum value
    """
    try:
        if item.source_type == "attachment":
            return _map_attachment_to_modality(item.payload)
        elif item.source_type == "url":
            return _map_url_to_modality(item.payload)
        elif item.source_type == "embed":
            return _map_embed_to_modality(item.payload)
        else:
            logger.warning(f"Unknown source_type: {item.source_type}")
            return InputModality.UNKNOWN
            
    except Exception as e:
        logger.error(f"Error mapping item to modality: {e}", exc_info=True)
        return InputModality.UNKNOWN


def _map_attachment_to_modality(attachment: discord.Attachment) -> InputModality:
    """Map attachment to modality based on content type and filename."""
    content_type = attachment.content_type or ""
    filename = attachment.filename.lower()
    
    # Image attachments
    if content_type.startswith("image/"):
        return InputModality.SINGLE_IMAGE
    
    # PDF documents
    if filename.endswith('.pdf'):
        # For now, assume all PDFs are text-based
        # TODO: Implement OCR detection logic
        return InputModality.PDF_DOCUMENT
    
    # Audio/video files
    if (content_type.startswith(("audio/", "video/")) or 
        filename.endswith(('.mp3', '.wav', '.mp4', '.avi', '.mov', '.mkv', '.webm'))):
        return InputModality.AUDIO_VIDEO_FILE
    
    # Document files (future expansion)
    if filename.endswith(('.docx', '.txt', '.rtf')):
        return InputModality.PDF_DOCUMENT  # Reuse document processing
    
    logger.warning(f"Unrecognized attachment type: {filename} ({content_type})")
    return InputModality.UNKNOWN


def _map_url_to_modality(url: str) -> InputModality:
    """Map URL to modality based on domain and pattern matching."""
    # Video URLs - use comprehensive patterns from video_ingest.py
    try:
        from .video_ingest import SUPPORTED_PATTERNS
        logger.debug(f"🎥 Testing {len(SUPPORTED_PATTERNS)} video patterns for modality mapping: {url}")
        
        for pattern in SUPPORTED_PATTERNS:
            if re.search(pattern, url):
                logger.info(f"✅ Video URL modality detected: {url} matched pattern: {pattern}")
                return InputModality.VIDEO_URL
                
        logger.debug(f"❌ No video patterns matched for modality mapping: {url}")
    except ImportError as e:
        logger.warning(f"Could not import SUPPORTED_PATTERNS from video_ingest: {e}, using fallback patterns")
        # Fallback to original limited patterns
        video_patterns = [
            r'https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+',
            r'https?://youtu\.be/[\w-]+',
            r'https?://(?:www\.)?tiktok\.com/@[\w.-]+/video/\d+',
            r'https?://(?:vm\.)?tiktok\.com/[\w-]+',
        ]
        
        for pattern in video_patterns:
            if re.match(pattern, url):
                return InputModality.VIDEO_URL
    
    # Image URLs
    if re.search(r'\.(jpg|jpeg|png|gif|webp|bmp)(\?.*)?$', url, re.IGNORECASE):
        return InputModality.SINGLE_IMAGE
    
    # PDF URLs
    if re.search(r'\.pdf(\?.*)?$', url, re.IGNORECASE):
        return InputModality.PDF_DOCUMENT
    
    # General URLs (will try oEmbed first, fallback to screenshot)
    return InputModality.GENERAL_URL


def _map_embed_to_modality(embed: discord.Embed) -> InputModality:
    """Map embed to modality based on embed type and content."""
    # Image embeds
    if embed.image or embed.thumbnail:
        return InputModality.SINGLE_IMAGE
    
    # Video embeds
    if embed.video:
        return InputModality.VIDEO_URL
    
    # General embeds (treat as URL content)
    return InputModality.GENERAL_URL


def detect_modality(message: Message) -> InputModality:
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
    
    return map_item_to_modality(items[0])
