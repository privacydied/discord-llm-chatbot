"""OCR (Optical Character Recognition) utilities."""

import logging
import shutil
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)

# Global flag for OCR availability
OCR_READY = False
OCR_WARNING_SHOWN = False

def check_ocr_dependencies() -> Tuple[bool, Dict[str, bool]]:
    """
    Check if OCR dependencies are available.
    
    Returns:
        Tuple of (OCR_READY, details_dict)
    """
    global OCR_READY
    
    # Check for tesseract binary
    tesseract_binary = shutil.which('tesseract') is not None
    
    # Check for pytesseract Python bindings
    pytesseract_available = False
    try:
        import pytesseract
        pytesseract_available = True
    except ImportError:
        pass
    
    # Set global OCR_READY flag - both binary and Python bindings must be present
    OCR_READY = tesseract_binary and pytesseract_available
    
    if not OCR_READY:
        logger.warning(
            "Tesseract OCR dependencies not fully available. Image-based PDFs will not be processed.",
            extra={
                'subsys': 'ocr', 
                'event': 'init.missing_dependencies',
                'tesseract_binary': tesseract_binary,
                'pytesseract': pytesseract_available
            }
        )
    else:
        logger.info(
            "Tesseract OCR dependencies available.",
            extra={
                'subsys': 'ocr', 
                'event': 'init.dependencies_available',
                'tesseract_binary': tesseract_binary,
                'pytesseract': pytesseract_available
            }
        )
    
    return OCR_READY, {
        'tesseract_binary': tesseract_binary,
        'pytesseract': pytesseract_available
    }

def get_ocr_status_message() -> str:
    """
    Get a user-friendly message about OCR status.
    
    Returns:
        A message suitable for displaying to users
    """
    global OCR_WARNING_SHOWN
    
    if OCR_READY:
        return "OCR is available for processing image-based PDFs."
    
    # Mark warning as shown
    OCR_WARNING_SHOWN = True
    
    return ("⚠ OCR is not available on this server. "
            "Please install Tesseract + pytesseract to process scanned PDFs.")

def is_ocr_warning_needed() -> bool:
    """
    Check if an OCR warning needs to be shown.
    
    Returns:
        True if warning should be shown, False otherwise
    """
    global OCR_WARNING_SHOWN
    
    if OCR_READY or OCR_WARNING_SHOWN:
        return False
    
    return True
