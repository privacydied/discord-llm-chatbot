"""Document ingestion wrapper for Discord attachments.

Handles PDF (with OCR fallback), DOCX, RTF, and Markdown files.
Integrates with existing RAG document parsers and PDF processor.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .utils.logging import get_logger

if TYPE_CHECKING:
    import discord

logger = get_logger(__name__)


async def ingest_document_attachment(
    attachment: discord.Attachment,
) -> dict[str, Any]:
    """Ingest a document attachment and extract its text content.

    Supports: PDF (with OCR fallback), DOCX, DOC, RTF, MD, ODT

    Args:
        attachment: Discord attachment to process

    Returns:
        Dict with keys: text, metadata, error (if any)

    """
    filename = getattr(attachment, "filename", "unknown")
    ext = Path(filename).suffix.lower()

    logger.info(
        f"doc.parse kind={ext} name={filename}",
        extra={
            "subsys": "doc",
            "event": "doc.parse.start",
            "detail": {"filename": filename, "extension": ext},
        },
    )

    # Save attachment to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext or ".tmp") as tmp_file:
        tmp_path = Path(tmp_file.name)

    try:
        await attachment.save(tmp_path)

        # Route to appropriate parser based on extension
        if ext == ".pdf":
            result = await _ingest_pdf(tmp_path, filename)
        elif ext in {".docx", ".doc", ".odt"}:
            result = await _ingest_docx(tmp_path, filename)
        elif ext == ".rtf":
            result = await _ingest_rtf(tmp_path, filename)
        elif ext in {".md", ".markdown"}:
            result = await _ingest_markdown(tmp_path, filename)
        else:
            result = {
                "text": "",
                "metadata": {},
                "error": f"Unsupported document type: {ext}",
            }

        # Log result
        if result.get("text"):
            chars = len(result["text"])
            pages = result.get("metadata", {}).get("page_count", "-")
            ocr_used = result.get("metadata", {}).get("extraction_method") == "ocr"

            logger.info(
                f"doc.parse kind={ext} pages={pages} chars={chars} ocr_used={ocr_used}",
                extra={
                    "subsys": "doc",
                    "event": "doc.parse.success",
                    "detail": {
                        "kind": ext,
                        "pages": pages,
                        "chars": chars,
                        "ocr_used": ocr_used,
                    },
                },
            )
        elif result.get("error"):
            logger.warning(
                f"doc.parse.failed kind={ext} error={result['error'][:100]}",
                extra={
                    "subsys": "doc",
                    "event": "doc.parse.failed",
                    "detail": {"kind": ext, "error": result["error"][:200]},
                },
            )

        return result

    except Exception as e:
        logger.error(f"Document ingestion failed for {filename}: {e}", exc_info=True)
        return {
            "text": "",
            "metadata": {},
            "error": f"Failed to process document: {e!s}",
        }

    finally:
        # Cleanup temp file
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except (OSError, PermissionError) as e:
            logger.debug(f"Failed to cleanup temp file {tmp_path}: {e}")


async def _ingest_pdf(tmp_path: Path, filename: str) -> dict[str, Any]:
    """Ingest PDF with OCR fallback.

    Uses existing PDFProcessor which already has OCR fallback logic.
    """
    try:
        from .pdf_utils import PDFProcessor

        processor = PDFProcessor()

        if not processor.supported:
            return {
                "text": "",
                "metadata": {},
                "error": "PDF processing not available (PyMuPDF not installed)",
            }

        # Process PDF (this already includes OCR fallback)
        result = await processor.process(tmp_path)

        if isinstance(result, dict):
            # New API returns dict
            if result.get("error") and not result.get("text"):
                return {
                    "text": "",
                    "metadata": result.get("metadata", {}),
                    "error": result["error"],
                }
            return {
                "text": result.get("text", ""),
                "metadata": result,
                "error": None,
            }
        # Legacy API returns string
        return {
            "text": str(result),
            "metadata": {},
            "error": None if result else "No text extracted",
        }

    except Exception as e:
        logger.error(f"PDF ingestion failed: {e}", exc_info=True)
        return {
            "text": "",
            "metadata": {},
            "error": str(e),
        }


async def _ingest_docx(tmp_path: Path, filename: str) -> dict[str, Any]:
    """Ingest DOCX/DOC/ODT files using RAG document parser."""
    try:
        from .rag.document_parsers import DocumentParserFactory

        parser = DocumentParserFactory()
        content, metadata = await parser.parse_document(tmp_path)

        return {
            "text": content,
            "metadata": metadata,
            "error": None,
        }

    except ImportError as e:
        return {
            "text": "",
            "metadata": {},
            "error": f"DOCX processing not available: {e}",
        }
    except Exception as e:
        logger.error(f"DOCX ingestion failed: {e}", exc_info=True)
        return {
            "text": "",
            "metadata": {},
            "error": str(e),
        }


async def _ingest_rtf(tmp_path: Path, filename: str) -> dict[str, Any]:
    """Ingest RTF files."""
    try:
        # Try to use RAG parser if available
        from .rag.document_parsers import DocumentParserFactory

        parser = DocumentParserFactory()

        # Check if RTF is supported
        if ".rtf" not in parser.get_supported_extensions():
            # Fall back to simple text extraction
            return await _ingest_as_text(tmp_path, filename)

        content, metadata = await parser.parse_document(tmp_path)

        return {
            "text": content,
            "metadata": metadata,
            "error": None,
        }
    except (AttributeError, TypeError, ValueError, OSError, UnicodeDecodeError) as e:
        logger.warning(f"Markdown parsing failed, trying text fallback: {e}")
        return await _ingest_as_text(tmp_path, filename)


async def _ingest_markdown(tmp_path: Path, filename: str) -> dict[str, Any]:
    """Ingest Markdown files using RAG document parser."""
    try:
        from .rag.document_parsers import DocumentParserFactory

        parser = DocumentParserFactory()
        content, metadata = await parser.parse_document(tmp_path)

        return {
            "text": content,
            "metadata": metadata,
            "error": None,
        }

    except Exception as e:
        logger.warning(f"Markdown parsing failed, trying text fallback: {e}")
        return await _ingest_as_text(tmp_path, filename)


async def _ingest_as_text(tmp_path: Path, filename: str) -> dict[str, Any]:
    """Fallback: Read file as plain text."""
    try:
        # Try UTF-8 first
        content = tmp_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Fall back to latin-1
        try:
            content = tmp_path.read_text(encoding="latin-1")
        except (UnicodeDecodeError, OSError, ValueError) as e:
            return {
                "text": "",
                "metadata": {},
                "error": f"Could not decode file: {e}",
            }

    return {
        "text": content,
        "metadata": {
            "parser_type": "text_fallback",
            "char_count": len(content),
        },
        "error": None,
    }


# ---------------------------------------------------------------------------
# URL-based document ingestion [CA][REH]
# ---------------------------------------------------------------------------


async def ingest_document_from_url(url: str) -> dict[str, Any]:
    """Ingest a document from a URL and extract its text content.

    Downloads the document to a temp file and processes it using the same
    pipeline as Discord attachments.

    Supports: PDF (with OCR fallback), DOCX, DOC, RTF, MD, ODT

    Args:
        url: HTTP(S) URL pointing to a document file

    Returns:
        Dict with keys: text, metadata, error (if any)

    """
    from .url_classifier import _extract_filename_from_url, download_url_to_temp

    filename = _extract_filename_from_url(url) or "document"
    ext = Path(filename).suffix.lower()

    # Ensure we have a valid extension for temp file
    if not ext:
        ext = ".tmp"

    logger.info(
        f"doc.url.parse kind={ext} url={url[:80]}",
        extra={
            "subsys": "doc",
            "event": "doc.url.parse.start",
            "detail": {"url": url[:200], "extension": ext},
        },
    )

    # Download to temp file
    tmp_path, error = await download_url_to_temp(url, suffix=ext)

    if error:
        logger.warning(
            f"doc.url.download.failed url={url[:80]} error={error[:100]}",
            extra={
                "subsys": "doc",
                "event": "doc.url.download.failed",
                "detail": {"url": url[:200], "error": error[:200]},
            },
        )
        return {
            "text": "",
            "metadata": {"source_url": url},
            "error": f"Failed to download document: {error}",
        }

    try:
        # Route to appropriate parser based on extension
        if ext == ".pdf":
            result = await _ingest_pdf(tmp_path, filename)
        elif ext in {".docx", ".doc", ".odt"}:
            result = await _ingest_docx(tmp_path, filename)
        elif ext == ".rtf":
            result = await _ingest_rtf(tmp_path, filename)
        elif ext in {".md", ".markdown"}:
            result = await _ingest_markdown(tmp_path, filename)
        elif ext == ".txt":
            result = await _ingest_as_text(tmp_path, filename)
        else:
            result = {
                "text": "",
                "metadata": {},
                "error": f"Unsupported document type: {ext}",
            }

        # Add source URL to metadata
        result.setdefault("metadata", {})
        result["metadata"]["source_url"] = url

        # Log result
        if result.get("text"):
            chars = len(result["text"])
            pages = result.get("metadata", {}).get("page_count", "-")
            ocr_used = result.get("metadata", {}).get("extraction_method") == "ocr"

            logger.info(
                f"doc.url.parse kind={ext} pages={pages} chars={chars} ocr_used={ocr_used}",
                extra={
                    "subsys": "doc",
                    "event": "doc.url.parse.success",
                    "detail": {
                        "kind": ext,
                        "pages": pages,
                        "chars": chars,
                        "ocr_used": ocr_used,
                        "url": url[:200],
                    },
                },
            )
        elif result.get("error"):
            logger.warning(
                f"doc.url.parse.failed kind={ext} error={result['error'][:100]}",
                extra={
                    "subsys": "doc",
                    "event": "doc.url.parse.failed",
                    "detail": {
                        "kind": ext,
                        "error": result["error"][:200],
                        "url": url[:200],
                    },
                },
            )

        return result

    except Exception as e:
        logger.error(f"Document URL ingestion failed for {url}: {e}", exc_info=True)
        return {
            "text": "",
            "metadata": {"source_url": url},
            "error": f"Failed to process document: {e!s}",
        }

    finally:
        # Cleanup temp file
        try:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink()
        except (OSError, PermissionError) as e:
            logger.debug(f"Failed to cleanup temp file {tmp_path}: {e}")
