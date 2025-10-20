"""
Per-file attachment classification for robust multimodal handling.

This module provides attachment bucketing that doesn't short-circuit on .txt files,
allowing proper handling of mixed attachments (e.g., .txt + PDF + voice note).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, List, Tuple

from .utils.logging import get_logger

if TYPE_CHECKING:
    import discord

logger = get_logger(__name__)


class AttachmentBucket(Enum):
    """Classification buckets for attachments."""
    
    AUDIO = auto()  # Audio files and voice messages → STT
    VIDEO = auto()  # Video files → STT
    DOC = auto()  # Documents (PDF, DOCX, RTF, MD) → Document ingestion
    IMAGE = auto()  # Images → VL
    TXT_PROMPT = auto()  # .txt files → Append to prompt
    OTHER = auto()  # Unsupported types


@dataclass
class ClassifiedAttachment:
    """An attachment with its assigned bucket."""
    
    attachment: discord.Attachment
    bucket: AttachmentBucket
    filename: str
    content_type: str
    size: int


def classify_attachment(attachment: discord.Attachment) -> ClassifiedAttachment:
    """
    Classify a single attachment into its processing bucket.
    
    Args:
        attachment: Discord attachment to classify
        
    Returns:
        ClassifiedAttachment with assigned bucket
        
    Rules (first matching wins):
    - AUDIO: audio/* MIME or voice_message flag or known audio extensions/containers
    - VIDEO: video/* MIME or known video extensions
    - DOC: PDF, DOCX, DOC, RTF, MD, ODT
    - IMAGE: image/* MIME or image extensions
    - TXT_PROMPT: .txt files or text/plain
    - OTHER: Everything else
    """
    # Extract metadata with safe defaults [REH]
    try:
        filename = (getattr(attachment, "filename", "") or "").lower()
        content_type = (getattr(attachment, "content_type", "") or "").lower()
        size = getattr(attachment, "size", 0)
        
        # Check for Discord voice message flag (if available)
        is_voice_message = getattr(attachment, "voice_message", False) or getattr(
            attachment, "is_voice_message", False
        )
    except Exception as e:
        logger.warning(f"Error extracting attachment metadata: {e}")
        filename = ""
        content_type = ""
        size = 0
        is_voice_message = False
    
    # Determine bucket using first-match rules
    bucket = _determine_bucket(filename, content_type, is_voice_message)
    
    # Log classification for observability
    logger.info(
        f"attach.classify name={filename} size={size} mime={content_type} "
        f"bucket={bucket.name} voice={is_voice_message}",
        extra={
            "subsys": "attach",
            "event": "attach.classify",
            "detail": {
                "filename": filename[:100],
                "content_type": content_type,
                "size": size,
                "bucket": bucket.name,
                "is_voice_message": is_voice_message,
            },
        },
    )
    
    return ClassifiedAttachment(
        attachment=attachment,
        bucket=bucket,
        filename=filename,
        content_type=content_type,
        size=size,
    )


def _determine_bucket(
    filename: str, content_type: str, is_voice_message: bool
) -> AttachmentBucket:
    """
    Determine the bucket for an attachment based on filename, MIME, and flags.
    
    First matching rule wins.
    """
    mime_root = content_type.split(";", 1)[0].strip()

    # AUDIO bucket (broadest audio support including voice messages)
    if is_voice_message:
        return AttachmentBucket.AUDIO
    
    # Audio MIME types (including Opus containers)
    if mime_root.startswith("audio/"):
        return AttachmentBucket.AUDIO
    
    # Opus in ogg/webm containers (Discord voice notes)
    if mime_root in {
        "application/ogg",
        "audio/ogg",
        "audio/webm",
        "video/webm",  # Can contain Opus audio
    }:
        return AttachmentBucket.AUDIO
    
    # Audio extensions (comprehensive list)
    audio_exts = {
        ".mp3", ".wav", ".ogg", ".opus", ".m4a", ".aac", 
        ".flac", ".wma", ".webm", ".oga"
    }
    if any(filename.endswith(ext) for ext in audio_exts):
        return AttachmentBucket.AUDIO
    
    # VIDEO bucket
    if mime_root.startswith("video/"):
        return AttachmentBucket.VIDEO
    
    video_exts = {
        ".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v",
        ".flv", ".wmv", ".mpg", ".mpeg"
    }
    if any(filename.endswith(ext) for ext in video_exts):
        return AttachmentBucket.VIDEO
    
    # DOC bucket (documents for text extraction)
    # PDF
    if filename.endswith(".pdf") or mime_root == "application/pdf":
        return AttachmentBucket.DOC
    
    # Microsoft Office formats
    doc_mimes_ms = {
        "application/msword",  # .doc
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
        "application/vnd.ms-excel",  # .xls
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # .xlsx
    }
    if mime_root in doc_mimes_ms:
        return AttachmentBucket.DOC
    
    # RTF
    if filename.endswith(".rtf") or mime_root in {"text/rtf", "application/rtf"}:
        return AttachmentBucket.DOC
    
    # Markdown
    if filename.endswith((".md", ".markdown")) or mime_root == "text/markdown":
        return AttachmentBucket.DOC
    
    # Office extensions (case-insensitive handled by .lower() on filename)
    doc_exts = {
        ".docx", ".doc", ".odt", ".rtf", ".md", ".markdown"
    }
    if any(filename.endswith(ext) for ext in doc_exts):
        return AttachmentBucket.DOC
    
    # IMAGE bucket
    if mime_root.startswith("image/"):
        return AttachmentBucket.IMAGE
    
    image_exts = {
        ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", 
        ".svg", ".ico", ".tiff", ".tif"
    }
    if any(filename.endswith(ext) for ext in image_exts):
        return AttachmentBucket.IMAGE
    
    # TXT_PROMPT bucket (.txt for prompt extension)
    if filename.endswith(".txt"):
        return AttachmentBucket.TXT_PROMPT
    
    # text/plain but not .txt (might be misidentified; safer to treat as doc)
    if mime_root.startswith("text/plain") and not filename.endswith(".txt"):
        # Check if it's actually an audio/video file with wrong MIME
        if any(filename.endswith(ext) for ext in {".mp3", ".wav", ".ogg", ".mp4", ".webm"}):
            # Re-classify based on extension
            if any(filename.endswith(ext) for ext in {".mp3", ".wav", ".ogg"}):
                return AttachmentBucket.AUDIO
            elif any(filename.endswith(ext) for ext in {".mp4", ".webm"}):
                return AttachmentBucket.VIDEO
        return AttachmentBucket.TXT_PROMPT
    
    # text/* (but not text/plain, which was handled above)
    if mime_root.startswith("text/"):
        return AttachmentBucket.DOC
    
    # application/octet-stream is ambiguous; use extension
    if mime_root == "application/octet-stream":
        # Try to infer from extension
        if any(filename.endswith(ext) for ext in audio_exts):
            return AttachmentBucket.AUDIO
        elif any(filename.endswith(ext) for ext in video_exts):
            return AttachmentBucket.VIDEO
        elif any(filename.endswith(ext) for ext in image_exts):
            return AttachmentBucket.IMAGE
        elif any(filename.endswith(ext) for ext in doc_exts):
            return AttachmentBucket.DOC
        elif filename.endswith(".txt"):
            return AttachmentBucket.TXT_PROMPT
    
    # Default: OTHER (unsupported)
    return AttachmentBucket.OTHER


def classify_attachments(
    attachments: List[discord.Attachment],
) -> List[ClassifiedAttachment]:
    """
    Classify multiple attachments independently (no short-circuiting).
    
    Args:
        attachments: List of Discord attachments
        
    Returns:
        List of ClassifiedAttachment objects in same order
    """
    if not attachments:
        return []
    
    classified = [classify_attachment(att) for att in attachments]
    
    # Log summary for observability
    bucket_counts = {}
    for c in classified:
        bucket_counts[c.bucket.name] = bucket_counts.get(c.bucket.name, 0) + 1
    
    logger.info(
        f"attach.summary total={len(classified)} "
        + " ".join(f"{k.lower()}={v}" for k, v in bucket_counts.items()),
        extra={
            "subsys": "attach",
            "event": "attach.summary",
            "detail": {"total": len(classified), "buckets": bucket_counts},
        },
    )
    
    return classified


def get_by_bucket(
    classified: List[ClassifiedAttachment], bucket: AttachmentBucket
) -> List[ClassifiedAttachment]:
    """Filter classified attachments by bucket."""
    return [c for c in classified if c.bucket == bucket]
