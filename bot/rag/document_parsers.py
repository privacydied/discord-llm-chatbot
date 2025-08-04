"""
Document parsers for various file formats in the RAG system.
Supports PDF, MOBI, EPUB, HTML, DOCX, TXT, and MD files.
"""
import os
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod

from ..util.logging import get_logger

logger = get_logger(__name__)


class DocumentParser(ABC):
    """Abstract base class for document parsers."""
    
    @abstractmethod
    def can_parse(self, file_path: Path) -> bool:
        """Check if this parser can handle the given file."""
        pass
    
    @abstractmethod
    def parse(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """
        Parse the document and return content and metadata.
        
        Returns:
            Tuple of (content_text, metadata_dict)
        """
        pass
    
    def _get_base_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Get basic file metadata."""
        stat = file_path.stat()
        return {
            "file_size": stat.st_size,
            "modified_time": stat.st_mtime,
            "file_extension": file_path.suffix.lower(),
            "filename": file_path.name,
            "parser_type": self.__class__.__name__
        }


class TextParser(DocumentParser):
    """Parser for plain text files."""
    
    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in {'.txt', '.text'}
    
    def parse(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Parse plain text file."""
        try:
            # Try different encodings
            for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
                try:
                    content = file_path.read_text(encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("Could not decode text file with any supported encoding")
            
            metadata = self._get_base_metadata(file_path)
            metadata.update({
                "content_type": "text/plain",
                "line_count": len(content.splitlines()),
                "char_count": len(content)
            })
            
            return content.strip(), metadata
            
        except Exception as e:
            logger.error(f"Error parsing text file {file_path}: {e}")
            raise


class MarkdownParser(DocumentParser):
    """Parser for Markdown files."""
    
    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in {'.md', '.markdown', '.mdown', '.mkd'}
    
    def parse(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Parse Markdown file."""
        try:
            content = file_path.read_text(encoding='utf-8')
            
            metadata = self._get_base_metadata(file_path)
            metadata.update({
                "content_type": "text/markdown",
                "line_count": len(content.splitlines()),
                "char_count": len(content)
            })
            
            # Extract markdown-specific metadata
            headers = re.findall(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE)
            if headers:
                metadata["header_count"] = len(headers)
                metadata["top_level_headers"] = [h[1] for h in headers if len(h[0]) == 1][:5]
            
            return content.strip(), metadata
            
        except Exception as e:
            logger.error(f"Error parsing markdown file {file_path}: {e}")
            raise


class PDFParser(DocumentParser):
    """Parser for PDF files using process isolation to prevent blocking the main thread."""
    
    # Maximum time (in seconds) to wait for the entire PDF processing
    PROCESS_TIMEOUT = 30
    
    # Maximum number of pages to process
    MAX_PAGES = 100
    
    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == '.pdf'
    
    def parse(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Parse PDF file using process isolation to prevent blocking the main thread."""
        import multiprocessing
        import tempfile
        import json
        import os
        import time
        import subprocess
        import sys
        
        # Create base metadata
        metadata = self._get_base_metadata(file_path)
        metadata["content_type"] = "application/pdf"
        
        # Create a temporary file to store the extraction results
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_file:
            temp_path = temp_file.name
        
        # Define the script path outside the try block so it's available in finally
        script_path = temp_path + '.py'
        
        try:
            # Define the PDF extraction script as a string
            pdf_script = """
#!/usr/bin/env python3
import sys
import json
import traceback
from pathlib import Path

def extract_pdf_text(pdf_path, output_path, max_pages=100):
    try:
        import PyPDF2
        result = {
            "success": False,
            "content": "",
            "metadata": {},
            "error": None
        }
        
        # Basic metadata
        result["metadata"] = {
            "filename": Path(pdf_path).name,
            "file_size": Path(pdf_path).stat().st_size,
            "content_type": "application/pdf"
        }
        
        try:
            content_parts = []
            with open(pdf_path, 'rb') as file:
                try:
                    pdf_reader = PyPDF2.PdfReader(file)
                    
                    # Extract PDF metadata
                    if hasattr(pdf_reader, 'metadata') and pdf_reader.metadata:
                        pdf_meta = pdf_reader.metadata
                        safe_metadata = {}
                        for key in ['/Title', '/Author', '/Subject', '/Creator', '/Producer']:
                            try:
                                value = pdf_meta.get(key, '')
                                if value and isinstance(value, (str, int, float, bool)):
                                    safe_metadata[f"pdf_{key.lower()[1:]}"] = value
                            except Exception:
                                pass
                        result["metadata"].update(safe_metadata)
                    
                    # Get page count
                    try:
                        page_count = len(pdf_reader.pages)
                        result["metadata"]["page_count"] = page_count
                    except Exception as e:
                        result["metadata"]["page_count"] = 0
                    
                    # Extract text from pages (limited by max_pages)
                    pages_to_process = min(result["metadata"].get("page_count", 0), max_pages)
                    for page_idx in range(pages_to_process):
                        try:
                            page = pdf_reader.pages[page_idx]
                            page_text = page.extract_text()
                            if page_text and page_text.strip():
                                content_parts.append(f"[Page {page_idx + 1}]\n{page_text.strip()}")
                        except Exception as e:
                            continue
                    
                    # Join content parts
                    content = "\n\n".join(content_parts)
                    result["content"] = content
                    result["metadata"]["char_count"] = len(content)
                    result["metadata"]["pages_extracted"] = len(content_parts)
                    result["success"] = True
                    
                except Exception as e:
                    result["error"] = f"Error reading PDF: {str(e)}"
        except Exception as e:
            result["error"] = f"Error opening PDF file: {str(e)}"
        
        # Write results to output file
        with open(output_path, 'w') as f:
            json.dump(result, f)
            
    except Exception as e:
        # Catch-all error handler
        with open(output_path, 'w') as f:
            error_info = {
                "success": False,
                "content": "",
                "metadata": {
                    "filename": Path(pdf_path).name,
                    "content_type": "application/pdf"
                },
                "error": f"Unhandled error: {str(e)}\n{traceback.format_exc()}"
            }
            json.dump(error_info, f)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: script.py <pdf_path> <output_json_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_path = sys.argv[2]
    extract_pdf_text(pdf_path, output_path, {self.MAX_PAGES})
    sys.exit(0)
"""
            
            # Write the extraction script to a temporary file
            with open(script_path, 'w') as f:
                f.write(pdf_script)
            
            # Make the script executable
            os.chmod(script_path, 0o755)
            
            # Run the PDF extraction in a separate process with timeout
            logger.info(f"Starting isolated PDF processing for {file_path}")
            start_time = time.time()
            
            try:
                # Use subprocess with timeout instead of multiprocessing to ensure complete isolation
                process = subprocess.Popen(
                    [sys.executable, script_path, str(file_path), temp_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                # Wait for the process to complete with timeout
                try:
                    stdout, stderr = process.communicate(timeout=self.PROCESS_TIMEOUT)
                    exit_code = process.returncode
                    
                    if exit_code != 0:
                        logger.warning(f"PDF extraction process exited with code {exit_code}")
                        if stderr:
                            logger.warning(f"PDF extraction stderr: {stderr.decode('utf-8', errors='replace')}")
                    
                except subprocess.TimeoutExpired:
                    logger.warning(f"PDF extraction timed out after {self.PROCESS_TIMEOUT} seconds")
                    process.kill()
                    stdout, stderr = process.communicate()
                    metadata["extraction_timeout"] = True
                    metadata["extraction_success"] = False
                    return "", metadata
                
            except Exception as e:
                logger.error(f"Error running PDF extraction process: {e}")
                metadata["extraction_error"] = str(e)
                metadata["extraction_success"] = False
                return "", metadata
            
            # Read the results from the temporary file
            try:
                if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                    with open(temp_path, 'r') as f:
                        result = json.load(f)
                    
                    # Update metadata and get content
                    if result.get("success", False):
                        content = result.get("content", "")
                        result_metadata = result.get("metadata", {})
                        metadata.update(result_metadata)
                        metadata["extraction_success"] = True
                        metadata["extraction_time"] = time.time() - start_time
                        
                        logger.info(f"PDF extraction completed in {metadata['extraction_time']:.2f} seconds")
                        return content, metadata
                    else:
                        error = result.get("error", "Unknown error")
                        logger.warning(f"PDF extraction failed: {error}")
                        metadata["extraction_error"] = error
                        metadata["extraction_success"] = False
                        return "", metadata
                else:
                    logger.warning(f"PDF extraction result file not found or empty")
                    metadata["extraction_error"] = "Result file not found or empty"
                    metadata["extraction_success"] = False
                    return "", metadata
                    
            except Exception as e:
                logger.error(f"Error reading PDF extraction results: {e}")
                metadata["extraction_error"] = str(e)
                metadata["extraction_success"] = False
                return "", metadata
            
        except Exception as e:
            logger.error(f"Error in PDF parsing process: {e}")
            metadata["extraction_error"] = str(e)
            metadata["extraction_success"] = False
            return "", metadata
            
        finally:
            # Clean up temporary files
            for path in [temp_path, script_path]:
                try:
                    if os.path.exists(path):
                        os.unlink(path)
                except Exception as e:
                    logger.warning(f"Error cleaning up temporary file {path}: {e}")



class HTMLParser(DocumentParser):
    """Parser for HTML files using BeautifulSoup."""
    
    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in {'.html', '.htm', '.xhtml'}
    
    def parse(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Parse HTML file."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("beautifulsoup4 is required for HTML parsing. Install with: pip install beautifulsoup4")
        
        try:
            content = file_path.read_text(encoding='utf-8')
            soup = BeautifulSoup(content, 'html.parser')
            
            metadata = self._get_base_metadata(file_path)
            metadata["content_type"] = "text/html"
            
            # Extract HTML metadata
            title_tag = soup.find('title')
            if title_tag:
                metadata["html_title"] = title_tag.get_text().strip()
            
            meta_tags = soup.find_all('meta')
            for meta in meta_tags:
                if meta.get('name') == 'description':
                    metadata["html_description"] = meta.get('content', '')
                elif meta.get('name') == 'keywords':
                    metadata["html_keywords"] = meta.get('content', '')
                elif meta.get('name') == 'author':
                    metadata["html_author"] = meta.get('content', '')
            
            # Extract text content, preserving some structure
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text with some structure preservation
            text_content = soup.get_text(separator='\n', strip=True)
            
            # Clean up excessive whitespace
            text_content = re.sub(r'\n\s*\n\s*\n', '\n\n', text_content)
            text_content = re.sub(r'[ \t]+', ' ', text_content)
            
            metadata["char_count"] = len(text_content)
            
            return text_content.strip(), metadata
            
        except Exception as e:
            logger.error(f"Error parsing HTML file {file_path}: {e}")
            raise


class DOCXParser(DocumentParser):
    """Parser for DOCX files using python-docx."""
    
    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in {'.docx', '.docm'}
    
    def parse(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Parse DOCX file."""
        try:
            from docx import Document
        except ImportError:
            raise ImportError("python-docx is required for DOCX parsing. Install with: pip install python-docx")
        
        try:
            doc = Document(file_path)
            
            metadata = self._get_base_metadata(file_path)
            metadata["content_type"] = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            
            # Extract document properties
            if doc.core_properties:
                props = doc.core_properties
                metadata.update({
                    "docx_title": props.title or '',
                    "docx_author": props.author or '',
                    "docx_subject": props.subject or '',
                    "docx_keywords": props.keywords or '',
                    "docx_comments": props.comments or '',
                    "docx_created": str(props.created) if props.created else '',
                    "docx_modified": str(props.modified) if props.modified else ''
                })
            
            # Extract text from paragraphs
            content_parts = []
            for paragraph in doc.paragraphs:
                text = paragraph.text.strip()
                if text:
                    content_parts.append(text)
            
            # Extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        cell_text = cell.text.strip()
                        if cell_text:
                            row_text.append(cell_text)
                    if row_text:
                        content_parts.append(" | ".join(row_text))
            
            content = "\n\n".join(content_parts)
            metadata.update({
                "paragraph_count": len(doc.paragraphs),
                "table_count": len(doc.tables),
                "char_count": len(content)
            })
            
            return content.strip(), metadata
            
        except Exception as e:
            logger.error(f"Error parsing DOCX file {file_path}: {e}")
            raise


class EPUBParser(DocumentParser):
    """Parser for EPUB files using ebooklib."""
    
    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == '.epub'
    
    def parse(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Parse EPUB file."""
        try:
            import ebooklib
            from ebooklib import epub
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("ebooklib and beautifulsoup4 are required for EPUB parsing. Install with: pip install ebooklib beautifulsoup4")
        
        try:
            book = epub.read_epub(str(file_path))
            
            metadata = self._get_base_metadata(file_path)
            metadata["content_type"] = "application/epub+zip"
            
            # Extract EPUB metadata
            metadata.update({
                "epub_title": book.get_metadata('DC', 'title')[0][0] if book.get_metadata('DC', 'title') else '',
                "epub_author": book.get_metadata('DC', 'creator')[0][0] if book.get_metadata('DC', 'creator') else '',
                "epub_language": book.get_metadata('DC', 'language')[0][0] if book.get_metadata('DC', 'language') else '',
                "epub_publisher": book.get_metadata('DC', 'publisher')[0][0] if book.get_metadata('DC', 'publisher') else '',
                "epub_identifier": book.get_metadata('DC', 'identifier')[0][0] if book.get_metadata('DC', 'identifier') else ''
            })
            
            # Extract text content from all items
            content_parts = []
            chapter_count = 0
            
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    chapter_count += 1
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()
                    
                    chapter_text = soup.get_text(separator='\n', strip=True)
                    if chapter_text.strip():
                        content_parts.append(f"[Chapter {chapter_count}]\n{chapter_text}")
            
            content = "\n\n".join(content_parts)
            metadata.update({
                "chapter_count": chapter_count,
                "char_count": len(content)
            })
            
            return content.strip(), metadata
            
        except Exception as e:
            logger.error(f"Error parsing EPUB file {file_path}: {e}")
            raise


class MOBIParser(DocumentParser):
    """Parser for MOBI files using kindle-unpack."""
    
    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in {'.mobi', '.azw', '.azw3'}
    
    def parse(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Parse MOBI file."""
        try:
            import mobidedrm
        except ImportError:
            logger.warning("mobidedrm not available for MOBI parsing. Attempting basic extraction...")
            # Fallback to basic text extraction
            return self._parse_mobi_basic(file_path)
        
        try:
            # This is a placeholder - MOBI parsing is complex and requires specialized libraries
            # For now, we'll use a basic approach
            return self._parse_mobi_basic(file_path)
            
        except Exception as e:
            logger.error(f"Error parsing MOBI file {file_path}: {e}")
            raise
    
    def _parse_mobi_basic(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Basic MOBI parsing fallback."""
        metadata = self._get_base_metadata(file_path)
        metadata["content_type"] = "application/x-mobipocket-ebook"
        
        # For now, return empty content with a warning
        logger.warning(f"MOBI parsing not fully implemented for {file_path}. Consider converting to EPUB or PDF.")
        
        return "", metadata


class DocumentParserFactory:
    """Factory for creating appropriate document parsers."""
    
    def __init__(self):
        self.parsers = [
            TextParser(),
            MarkdownParser(),
            PDFParser(),
            HTMLParser(),
            DOCXParser(),
            EPUBParser(),
            MOBIParser()
        ]
    
    def get_parser(self, file_path: Path) -> Optional[DocumentParser]:
        """Get the appropriate parser for a file."""
        for parser in self.parsers:
            if parser.can_parse(file_path):
                return parser
        return None
    
    def get_supported_extensions(self) -> set:
        """Get all supported file extensions."""
        extensions = set()
        test_files = [
            Path("test.txt"), Path("test.md"), Path("test.pdf"), 
            Path("test.html"), Path("test.docx"), Path("test.epub"), 
            Path("test.mobi"), Path("test.azw"), Path("test.azw3")
        ]
        
        for test_file in test_files:
            for parser in self.parsers:
                if parser.can_parse(test_file):
                    extensions.add(test_file.suffix.lower())
                    break
        
        return extensions
    
    def parse_document(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Parse a document using the appropriate parser."""
        parser = self.get_parser(file_path)
        if not parser:
            raise ValueError(f"No parser available for file type: {file_path.suffix}")
        
        return parser.parse(file_path)


# Global parser factory instance
document_parser_factory = DocumentParserFactory()
