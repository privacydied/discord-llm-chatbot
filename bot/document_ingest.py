"""
Document ingestion wrapper for Discord attachments.

Handles PDF (with OCR fallback), DOCX, RTF, and Markdown files.
Integrates with existing RAG document parsers and PDF processor.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Any, Optional

from .utils.logging import get_logger

if TYPE_CHECKING:
    import discord

logger = get_logger(__name__)


async def ingest_document_attachment(
    attachment: discord.Attachment,
) -> Dict[str, Any]:
    """
    Ingest a document attachment and extract its text content.
    
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
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=ext or ".tmp"
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
    
    try:
        await attachment.save(tmp_path)
        
        # Route to appropriate parser based on extension
        if ext == ".pdf":
            result = await _ingest_pdf(tmp_path, filename)
        elif ext in {".docx", ".doc", ".odt"}:
            result = await _ingest_docx(tmp_path, filename)
        elif ext in {".rtf"}:
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
            "error": f"Failed to process document: {str(e)}",
        }
    
    finally:
        # Cleanup temp file
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception as e:
            logger.debug(f"Failed to cleanup temp file {tmp_path}: {e}")


async def _ingest_pdf(tmp_path: Path, filename: str) -> Dict[str, Any]:
    """
    Ingest PDF with OCR fallback.
    
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
        else:
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


async def _ingest_docx(tmp_path: Path, filename: str) -> Dict[str, Any]:
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


async def _ingest_rtf(tmp_path: Path, filename: str) -> Dict[str, Any]:
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
    
    except Exception as e:
        logger.warning(f"RTF parsing failed, trying text fallback: {e}")
        return await _ingest_as_text(tmp_path, filename)


async def _ingest_markdown(tmp_path: Path, filename: str) -> Dict[str, Any]:
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


async def _ingest_as_text(tmp_path: Path, filename: str) -> Dict[str, Any]:
    """Fallback: Read file as plain text."""
    try:
        # Try UTF-8 first
        content = tmp_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Fall back to latin-1
        try:
            content = tmp_path.read_text(encoding="latin-1")
        except Exception as e:
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
