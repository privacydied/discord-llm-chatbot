"""
PDF processing utilities for the Discord bot.
"""
import os
import io
import logging
import re
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union, BinaryIO

# Import config
from .config import load_config

# Load configuration
config = load_config()

# Try to import PyPDF2
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    logging.warning("PyPDF2 is not installed. PDF processing will be limited.")

# Try to import pdfminer.six for better text extraction
try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
    from pdfminer.pdfdocument import PDFDocument, PDFTextExtractionNotAllowed
    from pdfminer.pdfparser import PDFParser
    from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
    from pdfminer.converter import TextConverter
    from pdfminer.layout import LAParams
    from pdfminer.pdfpage import PDFPage
    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False
    logging.warning("pdfminer.six is not installed. PDF text extraction will be limited.")

# Try to import OCR libraries
try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logging.warning("Tesseract OCR is not installed. Image-based PDFs will not be processed.")

class PDFProcessor:
    """Class to handle PDF processing operations."""

    def __init__(self):
        self.supported = PYPDF2_AVAILABLE or PDFMINER_AVAILABLE
        # We need the bot's event loop to run sync functions in an executor
        self.loop = None
    
    def is_pdf(self, file_path: Union[str, Path, BinaryIO]) -> bool:
        """Check if a file is a PDF by its magic number."""
        try:
            if hasattr(file_path, 'read'):
                # Handle file-like object
                pos = file_path.tell()
                header = file_path.read(4)
                file_path.seek(pos)  # Reset file pointer
                return header == b'%PDF'
            else:
                # Handle file path
                with open(file_path, 'rb') as f:
                    return f.read(4) == b'%PDF'
        except Exception as e:
            logging.error(f"Error checking if file is PDF: {e}")
            return False
    
    def extract_text_with_pdfminer(self, file_path: Union[str, Path, BinaryIO]) -> str:
        """Extract text from a PDF using pdfminer.six."""
        if not PDFMINER_AVAILABLE:
            return ""
        
        try:
            if hasattr(file_path, 'read'):
                # Handle file-like object
                return pdfminer_extract_text(file_path)
            else:
                # Handle file path
                return pdfminer_extract_text(file_path)
        except PDFTextExtractionNotAllowed:
            logging.warning("PDF does not allow text extraction.")
            return ""
        except Exception as e:
            logging.error(f"Error extracting text with pdfminer: {e}")
            return ""
    
    def extract_text_with_pypdf2(self, file_path: Union[str, Path, BinaryIO]) -> str:
        """Extract text from a PDF using PyPDF2."""
        if not PYPDF2_AVAILABLE:
            return ""
        
        try:
            if hasattr(file_path, 'read'):
                # Handle file-like object
                pdf_reader = PyPDF2.PdfReader(file_path)
            else:
                # Handle file path
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
            
            text_parts = []
            for page in pdf_reader.pages:
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                except Exception as e:
                    logging.warning(f"Error extracting text from PDF page: {e}")
                    continue
            
            return "\n\n".join(text_parts)
        except Exception as e:
            logging.error(f"Error extracting text with PyPDF2: {e}")
            return ""
    
    def extract_text(self, file_path: Union[str, Path, BinaryIO]) -> str:
        """Extract text from a PDF using the best available method."""
        if not self.is_pdf(file_path):
            return ""
        
        # First try pdfminer for better text extraction
        if PDFMINER_AVAILABLE:
            text = self.extract_text_with_pdfminer(file_path)
            if text.strip():
                return text
        
        # Fall back to PyPDF2 if pdfminer fails or is not available
        if PYPDF2_AVAILABLE:
            text = self.extract_text_with_pypdf2(file_path)
            if text.strip():
                return text
        
        return ""
    
    def extract_text_from_image_pdf(self, file_path: Union[str, Path], dpi: int = 300) -> str:
        """
        Extract text from a scanned/image-based PDF using OCR.
        
        Args:
            file_path: Path to the PDF file
            dpi: DPI to use for image conversion (higher is better quality but slower)
            
        Returns:
            Extracted text from the PDF
        """
        if not TESSERACT_AVAILABLE or not PYPDF2_AVAILABLE:
            return ""
        
        try:
            # Open the PDF
            if hasattr(file_path, 'read'):
                # Handle file-like object
                pdf_reader = PyPDF2.PdfReader(file_path)
            else:
                # Handle file path
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
            
            text_parts = []
            
            # Process each page
            for page_num in range(len(pdf_reader.pages)):
                try:
                    # Convert PDF page to image
                    page = pdf_reader.pages[page_num]
                    
                    # Create a PDF writer with just this page
                    pdf_writer = PyPDF2.PdfWriter()
                    pdf_writer.add_page(page)
                    
                    # Write to a bytes buffer
                    pdf_bytes = io.BytesIO()
                    pdf_writer.write(pdf_bytes)
                    pdf_bytes.seek(0)
                    
                    # Convert PDF to image (requires pdf2image)
                    try:
                        from pdf2image import convert_from_bytes
                        images = convert_from_bytes(
                            pdf_bytes.getvalue(),
                            dpi=dpi,
                            fmt='jpeg',
                            thread_count=1
                        )
                        
                        # Extract text from each image
                        for img in images:
                            text = pytesseract.image_to_string(img)
                            if text.strip():
                                text_parts.append(text.strip())
                    except ImportError:
                        logging.warning("pdf2image is not installed. Cannot process image-based PDFs.")
                        return ""
                    
                except Exception as e:
                    logging.warning(f"Error processing PDF page {page_num + 1}: {e}")
                    continue
            
            return "\n\n".join(text_parts)
        
        except Exception as e:
            logging.error(f"Error extracting text from image PDF: {e}")
            return ""
    
    def get_metadata(self, file_path: Union[str, Path, BinaryIO]) -> Dict[str, str]:
        """Extract metadata from a PDF file."""
        metadata = {
            'title': '',
            'author': '',
            'subject': '',
            'keywords': '',
            'creator': '',
            'producer': '',
            'creation_date': '',
            'modification_date': '',
            'page_count': 0
        }
        
        if not self.is_pdf(file_path):
            return metadata
        
        try:
            if hasattr(file_path, 'read'):
                # Handle file-like object
                pdf_reader = PyPDF2.PdfReader(file_path)
            else:
                # Handle file path
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
            
            # Get document info
            doc_info = pdf_reader.metadata or {}
            
            # Map PDF metadata to our format
            metadata.update({
                'title': str(doc_info.get('/Title', '')),
                'author': str(doc_info.get('/Author', '')),
                'subject': str(doc_info.get('/Subject', '')),
                'keywords': str(doc_info.get('/Keywords', '')),
                'creator': str(doc_info.get('/Creator', '')),
                'producer': str(doc_info.get('/Producer', '')),
                'creation_date': str(doc_info.get('/CreationDate', '')),
                'modification_date': str(doc_info.get('/ModDate', '')),
                'page_count': len(pdf_reader.pages)
            })
            
            # Clean up the values
            for key in metadata:
                if key != 'page_count':
                    # Remove PDF object markers and clean up the string
                    metadata[key] = re.sub(r'^[^a-zA-Z0-9\s\-\.,;:!?]', '', metadata[key])
                    metadata[key] = ' '.join(metadata[key].split())
        
        except Exception as e:
            logging.error(f"Error extracting PDF metadata: {e}")
        
        return metadata
    
    def is_scanned_pdf(self, file_path: Union[str, Path, BinaryIO]) -> bool:
        """Check if a PDF is scanned/image-based."""
        if not self.is_pdf(file_path):
            return False
        
        # Extract text using both methods
        text_miner = self.extract_text_with_pdfminer(file_path) if PDFMINER_AVAILABLE else ""
        text_pypdf = self.extract_text_with_pypdf2(file_path) if PYPDF2_AVAILABLE else ""
        
        # If both methods return very little text, it's likely a scanned PDF
        return (len(text_miner.strip()) < 100 and len(text_pypdf.strip()) < 100)
    
    def extract_all(self, file_path: Union[str, Path, BinaryIO], 
                   extract_images: bool = False) -> Dict[str, any]:
        """
        Extract all available information from a PDF.
        
        Args:
            file_path: Path to the PDF file or file-like object
            extract_images: Whether to extract images (not implemented)
            
        Returns:
            Dictionary with extracted text, metadata, and other information
        """
        result = {
            'text': '',
            'metadata': {},
            'is_scanned': False,
            'page_count': 0,
            'extraction_method': 'none',
            'error': None
        }
        
        try:
            # Check if the file is a PDF
            if not self.is_pdf(file_path):
                result['error'] = 'Not a valid PDF file'
                return result
            
            # Get metadata first
            result['metadata'] = self.get_metadata(file_path)
            result['page_count'] = result['metadata'].get('page_count', 0)
            
            # Check if it's a scanned/image-based PDF
            if self.is_scanned_pdf(file_path):
                result['is_scanned'] = True
                if TESSERACT_AVAILABLE:
                    result['text'] = self.extract_text_from_image_pdf(file_path)
                    result['extraction_method'] = 'ocr'
                else:
                    result['error'] = 'Scanned PDF detected but OCR is not available'
            else:
                # Try pdfminer first
                if PDFMINER_AVAILABLE:
                    result['text'] = self.extract_text_with_pdfminer(file_path)
                    if result['text'].strip():
                        result['extraction_method'] = 'pdfminer'
                
                # Fall back to PyPDF2 if pdfminer didn't work
                if not result['text'].strip() and PYPDF2_AVAILABLE:
                    result['text'] = self.extract_text_with_pypdf2(file_path)
                    if result['text'].strip():
                        result['extraction_method'] = 'pypdf2'
                
                if not result['text'].strip():
                    result['error'] = 'Could not extract text from PDF'
            
            return result
        
        except Exception as e:
            logging.error(f"Error processing PDF: {e}", exc_info=True)
            result['error'] = str(e)
            return result

    async def process(self, file_path: Union[str, Path, BinaryIO], 
                      extract_images: bool = False) -> Dict[str, any]:
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
