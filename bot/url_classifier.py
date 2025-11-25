"""
URL classification and content-type detection for unified media/document handling.

This module provides URL classification that routes URLs to the appropriate
processing pipeline (document, audio, video, image, or web scraper) based on
HTTP Content-Type headers and URL path extensions.

It treats URLs as another source of "attachments" rather than a separate feature,
reusing existing pipelines for document parsing, STT, VL, etc.
"""

from __future__ import annotations

import asyncio
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple
from urllib.parse import urlparse

import httpx

from .attachment_classifier import (
    AttachmentBucket,
    classify_mime_and_extension,
    is_web_page_mime,
)
from .utils.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants [CMV]
# ---------------------------------------------------------------------------

# Timeout for HEAD requests to detect content type (keep short to avoid blocking)
URL_HEAD_TIMEOUT_S = 8.0

# Maximum file size to download for document/media processing (25MB, matches Discord premium)
URL_MAX_DOWNLOAD_BYTES = 25 * 1024 * 1024

# Timeout for full file download
URL_DOWNLOAD_TIMEOUT_S = 60.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ClassifiedURL:
    """A URL with its classification result."""
    
    url: str
    bucket: AttachmentBucket
    content_type: Optional[str]  # From HTTP headers, if available
    filename: Optional[str]  # Extracted from URL path
    content_length: Optional[int]  # From HTTP headers, if available
    detection_method: str  # "head_request", "extension_only", or "fallback"


# ---------------------------------------------------------------------------
# URL content-type detection [REH][PA]
# ---------------------------------------------------------------------------

async def detect_url_content_type(url: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Detect the content type of a URL using a HEAD request.
    
    Args:
        url: HTTP(S) URL to probe
        
    Returns:
        Tuple of (content_type, content_length) or (None, None) on failure
        
    Notes:
        - Uses HEAD request to avoid downloading full content
        - Falls back gracefully on network errors or timeouts
        - Only handles http:// and https:// URLs
    """
    # Validate URL scheme
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            logger.debug(f"url.probe skip non-http scheme={parsed.scheme} url={url[:80]}")
            return None, None
    except Exception:
        return None, None
    
    try:
        async with httpx.AsyncClient(
            timeout=URL_HEAD_TIMEOUT_S,
            follow_redirects=True,
            max_redirects=5,
        ) as client:
            response = await client.head(url)
            
            content_type = response.headers.get("content-type")
            content_length_str = response.headers.get("content-length")
            content_length = int(content_length_str) if content_length_str else None
            
            logger.info(
                f"url.probe ok url={url[:80]} content_type={content_type} "
                f"content_length={content_length}",
                extra={
                    "subsys": "url",
                    "event": "url.probe",
                    "detail": {
                        "url": url[:200],
                        "content_type": content_type,
                        "content_length": content_length,
                        "status": response.status_code,
                    },
                },
            )
            
            return content_type, content_length
            
    except httpx.TimeoutException:
        logger.debug(f"url.probe timeout url={url[:80]}")
        return None, None
    except httpx.RequestError as e:
        logger.debug(f"url.probe error url={url[:80]} error={e}")
        return None, None
    except Exception as e:
        logger.debug(f"url.probe unexpected error url={url[:80]} error={e}")
        return None, None


def _extract_filename_from_url(url: str) -> Optional[str]:
    """Extract filename from URL path."""
    try:
        parsed = urlparse(url)
        path = parsed.path
        if not path or path == "/":
            return None
        # Get last path segment
        segments = [s for s in path.split("/") if s]
        if segments:
            filename = segments[-1]
            # Strip query params if accidentally included
            if "?" in filename:
                filename = filename.split("?", 1)[0]
            return filename
        return None
    except Exception:
        return None


async def classify_url(url: str) -> ClassifiedURL:
    """
    Classify a URL into a processing bucket.
    
    This determines whether a URL should be processed as:
    - DOC: Document (PDF, DOCX, etc.) → document parsing pipeline
    - AUDIO: Audio file → STT pipeline
    - VIDEO: Video file → media/STT pipeline
    - IMAGE: Image file → VL pipeline
    - OTHER: Web page or unsupported → existing web scraper
    
    Args:
        url: HTTP(S) URL to classify
        
    Returns:
        ClassifiedURL with bucket assignment and metadata
    """
    filename = _extract_filename_from_url(url)
    
    # Try HEAD request first for accurate content-type detection
    content_type, content_length = await detect_url_content_type(url)
    
    if content_type:
        # Check if it's a web page - these go to web scraper, not attachment pipeline
        if is_web_page_mime(content_type):
            logger.info(
                f"url.classify bucket=WEB_PAGE url={url[:80]} method=head_request",
                extra={
                    "subsys": "url",
                    "event": "url.classify",
                    "detail": {
                        "url": url[:200],
                        "bucket": "WEB_PAGE",
                        "content_type": content_type,
                        "method": "head_request",
                    },
                },
            )
            return ClassifiedURL(
                url=url,
                bucket=AttachmentBucket.OTHER,  # OTHER signals web scraper path
                content_type=content_type,
                filename=filename,
                content_length=content_length,
                detection_method="head_request",
            )
        
        # Use unified MIME classification
        bucket = classify_mime_and_extension(content_type, filename)
        
        logger.info(
            f"url.classify bucket={bucket.name} url={url[:80]} method=head_request",
            extra={
                "subsys": "url",
                "event": "url.classify",
                "detail": {
                    "url": url[:200],
                    "bucket": bucket.name,
                    "content_type": content_type,
                    "filename": filename,
                    "method": "head_request",
                },
            },
        )
        
        return ClassifiedURL(
            url=url,
            bucket=bucket,
            content_type=content_type,
            filename=filename,
            content_length=content_length,
            detection_method="head_request",
        )
    
    # Fallback: classify by extension only
    if filename:
        bucket = classify_mime_and_extension(None, filename)
        
        logger.info(
            f"url.classify bucket={bucket.name} url={url[:80]} method=extension_only",
            extra={
                "subsys": "url",
                "event": "url.classify",
                "detail": {
                    "url": url[:200],
                    "bucket": bucket.name,
                    "filename": filename,
                    "method": "extension_only",
                },
            },
        )
        
        return ClassifiedURL(
            url=url,
            bucket=bucket,
            content_type=None,
            filename=filename,
            content_length=None,
            detection_method="extension_only",
        )
    
    # No content-type and no recognizable extension - treat as web page
    logger.info(
        f"url.classify bucket=OTHER url={url[:80]} method=fallback",
        extra={
            "subsys": "url",
            "event": "url.classify",
            "detail": {
                "url": url[:200],
                "bucket": "OTHER",
                "method": "fallback",
            },
        },
    )
    
    return ClassifiedURL(
        url=url,
        bucket=AttachmentBucket.OTHER,
        content_type=None,
        filename=None,
        content_length=None,
        detection_method="fallback",
    )


# ---------------------------------------------------------------------------
# URL download helper [REH][RM]
# ---------------------------------------------------------------------------

async def download_url_to_temp(
    url: str,
    max_bytes: int = URL_MAX_DOWNLOAD_BYTES,
    timeout: float = URL_DOWNLOAD_TIMEOUT_S,
    suffix: str = "",
) -> Tuple[Optional[Path], Optional[str]]:
    """
    Download a URL to a temporary file.
    
    Args:
        url: HTTP(S) URL to download
        max_bytes: Maximum file size to download
        timeout: Download timeout in seconds
        suffix: File extension for temp file (e.g. ".pdf")
        
    Returns:
        Tuple of (temp_file_path, error_message)
        - On success: (Path, None)
        - On failure: (None, error_string)
        
    Notes:
        - Caller is responsible for cleaning up the temp file
        - Respects Content-Length header to avoid downloading oversized files
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return None, f"Unsupported URL scheme: {parsed.scheme}"
    except Exception as e:
        return None, f"Invalid URL: {e}"
    
    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            max_redirects=5,
        ) as client:
            # Stream the download to handle large files
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                
                # Check content length before downloading
                content_length_str = response.headers.get("content-length")
                if content_length_str:
                    content_length = int(content_length_str)
                    if content_length > max_bytes:
                        return None, f"File too large: {content_length} bytes (max {max_bytes})"
                
                # Create temp file
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix or ".tmp",
                ) as tmp_file:
                    tmp_path = Path(tmp_file.name)
                    
                    downloaded = 0
                    async for chunk in response.aiter_bytes(chunk_size=65536):
                        downloaded += len(chunk)
                        if downloaded > max_bytes:
                            # Clean up and abort
                            tmp_path.unlink(missing_ok=True)
                            return None, f"Download exceeded max size: {downloaded} bytes"
                        tmp_file.write(chunk)
                
                logger.info(
                    f"url.download ok url={url[:80]} bytes={downloaded}",
                    extra={
                        "subsys": "url",
                        "event": "url.download",
                        "detail": {
                            "url": url[:200],
                            "bytes": downloaded,
                            "path": str(tmp_path),
                        },
                    },
                )
                
                return tmp_path, None
                
    except httpx.HTTPStatusError as e:
        return None, f"HTTP error {e.response.status_code}: {e}"
    except httpx.TimeoutException:
        return None, f"Download timeout after {timeout}s"
    except httpx.RequestError as e:
        return None, f"Network error: {e}"
    except Exception as e:
        return None, f"Download failed: {e}"
