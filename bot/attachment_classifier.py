"""Per-file attachment classification for robust multimodal handling.

This module provides attachment bucketing that doesn't short-circuit on .txt files,
allowing proper handling of mixed attachments (e.g., .txt + PDF + voice note).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

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
    """Classify a single attachment into its processing bucket.

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

        def _resolve_flag(val) -> bool:
            try:
                if callable(val):
                    return bool(val())
            except (TypeError, AttributeError):
                return False
            return bool(val)

        voice_attr = getattr(attachment, "voice_message", None)
        voice_method = getattr(attachment, "is_voice_message", None)
        is_voice_message = _resolve_flag(voice_attr) or _resolve_flag(voice_method)
    except (AttributeError, TypeError) as e:
        logger.warning(f"Error extracting attachment metadata: {e}")
        filename = ""
        content_type = ""
        size = 0
        is_voice_message = False

    # Determine bucket using first-match rules
    bucket = _determine_bucket(filename, content_type, is_voice_message)

    # Log classification for observability
    logger.info(
        f"attach.classify name={filename} size={size} mime={content_type} bucket={bucket.name} voice={is_voice_message}",
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


def _determine_bucket(filename: str, content_type: str, is_voice_message: bool) -> AttachmentBucket:
    """Determine the bucket for an attachment based on filename, MIME, and flags.

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
        ".mp3",
        ".wav",
        ".ogg",
        ".opus",
        ".m4a",
        ".aac",
        ".flac",
        ".wma",
        ".webm",
        ".oga",
    }
    if any(filename.endswith(ext) for ext in audio_exts):
        return AttachmentBucket.AUDIO

    # VIDEO bucket
    if mime_root.startswith("video/"):
        return AttachmentBucket.VIDEO

    video_exts = {
        ".mp4",
        ".mov",
        ".mkv",
        ".webm",
        ".avi",
        ".m4v",
        ".flv",
        ".wmv",
        ".mpg",
        ".mpeg",
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
    doc_exts = {".docx", ".doc", ".odt", ".rtf", ".md", ".markdown"}
    if any(filename.endswith(ext) for ext in doc_exts):
        return AttachmentBucket.DOC

    # IMAGE bucket
    if mime_root.startswith("image/"):
        return AttachmentBucket.IMAGE

    image_exts = {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".webp",
        ".bmp",
        ".svg",
        ".ico",
        ".tiff",
        ".tif",
    }
    if any(filename.endswith(ext) for ext in image_exts):
        return AttachmentBucket.IMAGE

    # TXT_PROMPT bucket (.txt for prompt extension)
    if filename.endswith(".txt"):
        return AttachmentBucket.TXT_PROMPT

    # text/plain but not .txt (might be misidentified; safer to treat as doc)
    if mime_root.startswith("text/plain") and not filename.endswith(".txt"):
        # Check if it's actually an audio/video file with wrong MIME
        if any(filename.endswith(ext) for ext in (".mp3", ".wav", ".ogg", ".mp4", ".webm")):
            # Re-classify based on extension
            if any(filename.endswith(ext) for ext in (".mp3", ".wav", ".ogg")):
                return AttachmentBucket.AUDIO
            if any(filename.endswith(ext) for ext in (".mp4", ".webm")):
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
        if any(filename.endswith(ext) for ext in video_exts):
            return AttachmentBucket.VIDEO
        if any(filename.endswith(ext) for ext in image_exts):
            return AttachmentBucket.IMAGE
        if any(filename.endswith(ext) for ext in doc_exts):
            return AttachmentBucket.DOC
        if filename.endswith(".txt"):
            return AttachmentBucket.TXT_PROMPT

    # Default: OTHER (unsupported)
    return AttachmentBucket.OTHER


def classify_attachments(
    attachments: list[discord.Attachment],
) -> list[ClassifiedAttachment]:
    """Classify multiple attachments independently (no short-circuiting).

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
        f"attach.summary total={len(classified)} " + " ".join(f"{k.lower()}={v}" for k, v in bucket_counts.items()),
        extra={
            "subsys": "attach",
            "event": "attach.summary",
            "detail": {"total": len(classified), "buckets": bucket_counts},
        },
    )

    return classified


def get_by_bucket(classified: list[ClassifiedAttachment], bucket: AttachmentBucket) -> list[ClassifiedAttachment]:
    """Filter classified attachments by bucket."""
    return [c for c in classified if c.bucket == bucket]


# ---------------------------------------------------------------------------
# Unified MIME classification for both attachments and URLs [CA][CMV]
# ---------------------------------------------------------------------------

# MIME type → bucket mappings (first match wins within category)
_AUDIO_MIMES = frozenset(
    {
        "audio/mpeg",
        "audio/mp3",
        "audio/wav",
        "audio/x-wav",
        "audio/flac",
        "audio/x-flac",
        "audio/webm",
        "audio/ogg",
        "audio/opus",
        "audio/aac",
        "audio/m4a",
        "audio/x-m4a",
        "audio/mp4",
        "application/ogg",
    },
)

_VIDEO_MIMES = frozenset(
    {
        "video/mp4",
        "video/webm",
        "video/x-matroska",
        "video/quicktime",
        "video/x-msvideo",
        "video/avi",
        "video/x-flv",
        "video/x-ms-wmv",
        "video/mpeg",
        "video/ogg",
    },
)

_IMAGE_MIMES = frozenset(
    {
        "image/png",
        "image/jpeg",
        "image/jpg",
        "image/webp",
        "image/gif",
        "image/bmp",
        "image/tiff",
        "image/svg+xml",
        "image/x-icon",
    },
)

_DOC_MIMES = frozenset(
    {
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/rtf",
        "text/rtf",
        "text/markdown",
        "text/x-markdown",
        "application/vnd.oasis.opendocument.text",
    },
)

_WEB_PAGE_MIMES = frozenset(
    {
        "text/html",
        "application/xhtml+xml",
    },
)

# Extension → bucket mappings (lowercase, with leading dot)
_AUDIO_EXTS = frozenset(
    {
        ".mp3",
        ".wav",
        ".ogg",
        ".opus",
        ".m4a",
        ".aac",
        ".flac",
        ".wma",
        ".webm",
        ".oga",
    },
)

_VIDEO_EXTS = frozenset(
    {
        ".mp4",
        ".mov",
        ".mkv",
        ".webm",
        ".avi",
        ".m4v",
        ".flv",
        ".wmv",
        ".mpg",
        ".mpeg",
    },
)

_IMAGE_EXTS = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".webp",
        ".bmp",
        ".svg",
        ".ico",
        ".tiff",
        ".tif",
    },
)

_DOC_EXTS = frozenset(
    {
        ".pdf",
        ".doc",
        ".docx",
        ".odt",
        ".rtf",
        ".md",
        ".markdown",
        ".xls",
        ".xlsx",
    },
)

_TXT_EXTS = frozenset({".txt"})


def classify_mime_and_extension(
    content_type: str | None,
    filename_or_path: str | None,
) -> AttachmentBucket:
    """Classify content into a processing bucket based on MIME type and/or filename extension.

    This is a unified helper for both Discord attachments and URL-fetched content.

    Args:
        content_type: MIME type string (may include charset, e.g. "text/html; charset=utf-8")
        filename_or_path: Filename or URL path for extension-based fallback

    Returns:
        AttachmentBucket indicating the appropriate processing pipeline

    Rules (first matching wins):
    - WEB_PAGE: text/html or application/xhtml+xml → handled by web scraper (not attachment pipeline)
    - AUDIO: audio/* MIME or known audio extensions
    - VIDEO: video/* MIME or known video extensions
    - DOC: PDF, DOCX, etc. MIME or document extensions
    - IMAGE: image/* MIME or image extensions
    - TXT_PROMPT: .txt files
    - OTHER: Everything else

    """
    # Normalize inputs
    mime_root = ""
    if content_type:
        mime_root = content_type.split(";", 1)[0].strip().lower()

    ext = ""
    if filename_or_path:
        # Extract extension from filename or URL path
        path_lower = filename_or_path.lower()
        # Handle query strings in URLs
        if "?" in path_lower:
            path_lower = path_lower.split("?", 1)[0]
        # Find last dot for extension
        dot_idx = path_lower.rfind(".")
        if dot_idx != -1:
            ext = path_lower[dot_idx:]

    # 1. Web page detection (special case - not an "attachment" bucket but useful for routing)
    if mime_root in _WEB_PAGE_MIMES:
        return AttachmentBucket.OTHER  # Signal to use web scraper, not attachment pipeline

    # 2. Audio detection
    if mime_root in _AUDIO_MIMES or mime_root.startswith("audio/"):
        return AttachmentBucket.AUDIO
    if ext in _AUDIO_EXTS:
        return AttachmentBucket.AUDIO

    # 3. Video detection (but not video/webm which might be audio-only)
    if mime_root in _VIDEO_MIMES or mime_root.startswith("video/"):
        # video/webm could be audio-only (Opus), check extension
        if mime_root == "video/webm" and ext in _AUDIO_EXTS:
            return AttachmentBucket.AUDIO
        return AttachmentBucket.VIDEO
    if ext in _VIDEO_EXTS:
        # .webm could be audio
        if ext == ".webm" and mime_root and mime_root.startswith("audio/"):
            return AttachmentBucket.AUDIO
        return AttachmentBucket.VIDEO

    # 4. Document detection
    if mime_root in _DOC_MIMES:
        return AttachmentBucket.DOC
    if ext in _DOC_EXTS:
        return AttachmentBucket.DOC

    # 5. Image detection
    if mime_root in _IMAGE_MIMES or mime_root.startswith("image/"):
        return AttachmentBucket.IMAGE
    if ext in _IMAGE_EXTS:
        return AttachmentBucket.IMAGE

    # 6. Plain text (.txt) for prompt extension
    if ext in _TXT_EXTS:
        return AttachmentBucket.TXT_PROMPT
    if mime_root == "text/plain":
        # text/plain without .txt extension - treat as doc
        return AttachmentBucket.DOC

    # 7. application/octet-stream - rely on extension
    if mime_root == "application/octet-stream":
        if ext in _AUDIO_EXTS:
            return AttachmentBucket.AUDIO
        if ext in _VIDEO_EXTS:
            return AttachmentBucket.VIDEO
        if ext in _IMAGE_EXTS:
            return AttachmentBucket.IMAGE
        if ext in _DOC_EXTS:
            return AttachmentBucket.DOC
        if ext in _TXT_EXTS:
            return AttachmentBucket.TXT_PROMPT

    # 8. Default: unsupported
    return AttachmentBucket.OTHER


def is_web_page_mime(content_type: str | None) -> bool:
    """Check if the content type indicates an HTML web page.

    Args:
        content_type: MIME type string (may include charset)

    Returns:
        True if this is an HTML page that should go through web scraping

    """
    if not content_type:
        return False
    mime_root = content_type.split(";", 1)[0].strip().lower()
    return mime_root in _WEB_PAGE_MIMES
