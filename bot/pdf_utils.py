"""
PDF processing utilities for the Discord bot, using PyMuPDF (fitz).
"""
import io
import logging
import asyncio
import shutil
from pathlib import Path
from typing import Dict, Union, BinaryIO, Optional

# PyMuPDF
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    fitz = None
    PYMUPDF_AVAILABLE = False
    logging.warning("PyMuPDF (fitz) is not installed. PDF processing is disabled.")

# Try to import OCR libraries
try:
    import pytesseract
    from PIL import Image
    # Check if tesseract executable is available in PATH
    TESSERACT_EXECUTABLE = shutil.which("tesseract")
    TESSERACT_AVAILABLE = TESSERACT_EXECUTABLE is not None
    
    if not TESSERACT_AVAILABLE:
        logging.warning("Tesseract OCR executable not found in PATH. Image-based PDFs will not be processed.")
    else:
        logging.debug(f"Found Tesseract OCR at: {TESSERACT_EXECUTABLE}")
        
except ImportError:
    pytesseract = None
    Image = None
    TESSERACT_AVAILABLE = False
    logging.warning("Tesseract OCR Python libraries not installed. Image-based PDFs will not be processed.")


class PDFProcessor:
    """Class to handle PDF processing operations using PyMuPDF."""

    def __init__(self):
        self.supported = PYMUPDF_AVAILABLE
        self.loop = None

    def is_pdf(self, file_path: Union[str, Path, BinaryIO]) -> bool:
        """Check if a file is a PDF by its magic number."""
        try:
            if hasattr(file_path, 'read'):
                pos = file_path.tell()
                header = file_path.read(4)
                file_path.seek(pos)
                return header == b'%PDF'
            else:
                with open(file_path, 'rb') as f:
                    return f.read(4) == b'%PDF'
        except Exception as e:
            logging.error(f"Error checking if file is PDF: {e}")
            return False

    def _open_pdf(self, file_path: Union[str, Path, BinaryIO]) -> 'fitz.Document':
        """Helper to open a PDF from a path or file-like object."""
        if hasattr(file_path, 'read'):
            # It's a file-like object
            file_path.seek(0)
            stream = file_path.read()
            return fitz.open(stream=stream, filetype="pdf")
        else:
            # It's a path
            return fitz.open(file_path)

    def extract_text(self, doc: 'fitz.Document') -> str:
        """Extract all text from a PyMuPDF document."""
        try:
            return "".join(page.get_text() for page in doc)
        except Exception as e:
            logging.error(f"Error extracting text with PyMuPDF: {e}")
            return ""

    def extract_text_from_image_pdf(self, doc: 'fitz.Document', dpi: int = 300) -> str:
        """Extract text from a scanned/image-based PDF using OCR."""
        if not TESSERACT_AVAILABLE:
            logging.warning("Cannot perform OCR: Tesseract is not available.")
            return ""
        
        text_parts = []
        try:
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=dpi)
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                page_text = pytesseract.image_to_string(img)
                text_parts.append(page_text)
            return "\n\n".join(text_parts)
        except Exception as e:
            logging.error(f"Error during OCR extraction: {e}")
            return ""

    def get_metadata(self, doc: 'fitz.Document') -> Dict[str, any]:
        """Extract metadata from a PyMuPDF document."""
        if not self.supported:
            return {}
        try:
            metadata = doc.metadata
            metadata['page_count'] = doc.page_count
            return metadata
        except Exception as e:
            logging.error(f"Error extracting PDF metadata with PyMuPDF: {e}")
            return {}

    def is_scanned_pdf(self, doc: 'fitz.Document') -> bool:
        """Check if a PDF is scanned/image-based by checking for text content."""
        text = self.extract_text(doc)
        # If the text is very short, it's likely scanned.
        return len(text.strip()) < 100

    def extract_all(self, file_path: Union[str, Path, BinaryIO], extract_images: bool = False) -> Dict[str, any]:
        """Extract all available information from a PDF using PyMuPDF."""
        result = {
            'text': '',
            'metadata': {},
            'is_scanned': False,
            'page_count': 0,
            'extraction_method': 'none',
            'error': None
        }

        if not self.supported:
            result['error'] = 'PyMuPDF (fitz) is not installed.'
            return result

        try:
            if not self.is_pdf(file_path):
                result['error'] = 'Not a valid PDF file'
                return result

            doc = self._open_pdf(file_path)
            
            result['metadata'] = self.get_metadata(doc)
            result['page_count'] = doc.page_count
            
            if self.is_scanned_pdf(doc):
                result['is_scanned'] = True
                if TESSERACT_AVAILABLE:
                    result['text'] = self.extract_text_from_image_pdf(doc)
                    result['extraction_method'] = 'pymupdf_ocr'
                else:
                    result['error'] = 'Scanned PDF detected but OCR is not available'
            else:
                result['text'] = self.extract_text(doc)
                result['extraction_method'] = 'pymupdf'

            if not result['text'].strip() and not result['error']:
                result['error'] = 'Could not extract text from PDF'

            doc.close()
            return result

        except Exception as e:
            logging.error(f"Error processing PDF with PyMuPDF: {e}", exc_info=True)
            result['error'] = str(e)
            return result

    async def process(self, file_path: Union[str, Path, BinaryIO], extract_images: bool = False) -> Dict[str, any]:
        """
        Asynchronously process a PDF file by running the synchronous extract_all
        method in an executor to avoid blocking the event loop.
        """
        loop = self.loop or asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,  # Use the default executor
            self.extract_all,
            file_path,
            extract_images
        )
