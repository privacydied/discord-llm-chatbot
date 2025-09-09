"""
PDF processing module with text extraction, OCR fallback, and metadata handling.
"""
import asyncio
import re
import hashlib
import logging
from typing import Dict, Tuple, Union
from datetime import datetime

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import magic
import requests
from cachetools import TTLCache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache for storing processed PDFs (1 hour TTL)
pdf_cache = TTLCache(maxsize=100, ttl=3600)

class PDFProcessor:
    """Handles PDF text extraction with OCR fallback and metadata processing."""
    
    def __init__(self, ocr_lang: str = 'eng'):
        """Initialize the PDF processor.
        
        Args:
            ocr_lang: Language code for Tesseract OCR (default: 'eng')
        """
        self.ocr_lang = ocr_lang
        self.mime = magic.Magic(mime=True)
    
    def _get_cache_key(self, source: str) -> str:
        """Generate a cache key for the PDF source."""
        return hashlib.md5(source.encode()).hexdigest()
    
    def _is_pdf(self, file_path_or_url: str) -> bool:
        """Check if the input is a PDF file or URL."""
        if file_path_or_url.lower().endswith('.pdf'):
            return True
        
        try:
            mime_type = self.mime.from_file(file_path_or_url)
            return mime_type == 'application/pdf'
        except (magic.MagicException, FileNotFoundError):
            pass
            
        return False
    
    def _download_pdf(self, url: str, timeout: int = 30) -> bytes:
        """Download PDF from URL with error handling."""
        try:
            response = requests.get(url, stream=True, timeout=timeout)
            response.raise_for_status()
            return response.content
        except requests.RequestException as e:
            logger.error(f"Failed to download PDF from {url}: {e}")
            raise
    
    def _extract_metadata(self, doc: fitz.Document) -> Dict[str, str]:
        """Extract metadata from PDF document."""
        meta = {}
        info = doc.metadata
        
        # Standard metadata fields
        meta['title'] = info.get('title', '').strip() or 'Untitled'
        meta['author'] = info.get('author', '').strip() or 'Unknown'
        meta['subject'] = info.get('subject', '').strip()
        meta['keywords'] = info.get('keywords', '').strip()
        meta['creator'] = info.get('creator', '').strip()
        meta['producer'] = info.get('producer', '').strip()
        
        # Format dates
        for date_field in ['creationDate', 'modDate']:
            if date_field in info and info[date_field]:
                try:
                    # Convert PDF date format to ISO format
                    date_str = info[date_field].replace("'", "")  # Remove single quotes
                    if date_str.startswith('D:'):
                        date_str = date_str[2:]
                    date_obj = datetime.strptime(date_str[:14], '%Y%m%d%H%M%S')
                    meta[date_field] = date_obj.isoformat()
                except (ValueError, IndexError):
                    meta[date_field] = info[date_field]
        
        return meta
    
    async def _ocr_image(self, img: Image.Image) -> str:
        """Run OCR on an image in a thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,  # Uses default ThreadPoolExecutor
            lambda: pytesseract.image_to_string(img, lang=self.ocr_lang)
        )
    
    async def _extract_text_with_pymupdf(self, doc: fitz.Document) -> Tuple[str, bool]:
        """Extract text from PDF using PyMuPDF.
        
        Returns:
            Tuple of (extracted_text, is_scanned)
        """
        text = []
        is_scanned = True
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Try text extraction first
            page_text = page.get_text("text")
            
            # If little to no text, it might be a scanned document
            if len(page_text.strip()) < 50:  # Threshold for considering as scanned
                # Try OCR as fallback
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                page_text = await self._ocr_image(img)
            else:
                is_scanned = False
            
            if page_text.strip():
                text.append(page_text.strip())
        
        return "\n\n".join(text), is_scanned
    
    def _clean_text(self, text: str) -> str:
        """Clean and format extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common header/footer patterns
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)  # Page numbers
        text = re.sub(r'\n\s*[A-Z0-9\s]+\n\s*\n', '\n', text)  # All-caps headers
        
        # Remove consecutive blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    async def _process_large_pdf_async(self, doc: fitz.Document, max_pages: int = 40) -> str:
        """Process large PDFs by extracting key sections asynchronously."""
        # Get table of contents
        toc = doc.get_toc()
        
        # If no TOC, just take first and last few pages
        if not toc or len(toc) < 3:
            num_pages = len(doc)
            pages_to_extract = list(range(min(5, num_pages)))
            if num_pages > 10:
                pages_to_extract.extend(range(num_pages - 5, num_pages))
            
            text_parts = []
            for page_num in pages_to_extract:
                page = doc.load_page(page_num)
                text_parts.append(page.get_text("text").strip())
            
            return "\n\n".join(text_parts)
        
        # Otherwise, extract sections from TOC using async extraction
        text, _ = await self._extract_text_with_pymupdf(doc)
        return text
    
    async def process_pdf(self, source: Union[str, bytes], is_url: bool = False) -> Dict:
        """Process a PDF file or URL and extract text and metadata asynchronously.
        
        Args:
            source: Path to PDF file, URL, or PDF content as bytes
            is_url: Whether source is a URL
            
        Returns:
            Dict with text, metadata, page count, and source
        """
        # Check cache first
        cache_key = self._get_cache_key(source if isinstance(source, str) else 'bytes')
        if cache_key in pdf_cache:
            return pdf_cache[cache_key]
        
        result = {
            'text': '',
            'metadata': {},
            'pages': 0,
            'source': source if isinstance(source, str) else 'uploaded_file',
            'is_scanned': False
        }
        
        try:
            # Load PDF content
            if is_url or (isinstance(source, str) and source.startswith(('http://', 'https://'))):
                pdf_content = self._download_pdf(source)
                doc = fitz.open(stream=pdf_content, filetype='pdf')
            elif isinstance(source, bytes):
                doc = fitz.open(stream=source, filetype='pdf')
            else:  # File path
                doc = fitz.open(source)
            
            # Extract metadata
            result['metadata'] = self._extract_metadata(doc)
            result['pages'] = len(doc)
            
            # Process content based on size asynchronously
            if len(doc) > 40:  # Large PDF
                result['text'] = await self._process_large_pdf_async(doc)
                result['is_large'] = True
            else:  # Small to medium PDF
                result['text'], result['is_scanned'] = await self._extract_text_with_pymupdf(doc)
            
            # Clean up text
            result['text'] = self._clean_text(result['text'])
            
            # Cache the result
            pdf_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}", exc_info=True)
            result['error'] = str(e)
            return result
            
        finally:
            if 'doc' in locals():
                doc.close()

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pdf_processor.py <pdf_file_or_url>")
        sys.exit(1)
    
    processor = PDFProcessor()
    source = sys.argv[1]
    is_url = source.startswith(('http://', 'https://'))
    
    result = processor.process_pdf(source, is_url=is_url)
    
    print("\n=== PDF Metadata ===")
    for key, value in result.get('metadata', {}).items():
        print(f"{key}: {value}")
    
    print("\n=== Extracted Text (first 1000 chars) ===")
    print(result.get('text', '')[:1000] + '...')
    
    print("\n=== Processing Info ===")
    print(f"Pages: {result.get('pages', 0)}")
    print(f"Is Scanned: {result.get('is_scanned', False)}")
    if 'error' in result:
        print(f"Error: {result['error']}")
