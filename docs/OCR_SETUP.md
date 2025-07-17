# OCR Setup Guide

This document provides instructions for setting up Optical Character Recognition (OCR) functionality in the Discord bot.

## Overview

The bot uses Tesseract OCR to extract text from images and scanned PDFs. This enables the following capabilities:

1. Processing image-based PDFs
2. Extracting text from images sent in Discord
3. Converting scanned documents to text for LLM processing

## Requirements

- Tesseract OCR binary (system package)
- pytesseract Python package (Python binding)

## Installation

### Arch Linux

```bash
# Install Tesseract OCR binary
sudo pacman -Sy tesseract tesseract-data-eng

# Install additional language data (optional)
# sudo pacman -Sy tesseract-data-deu tesseract-data-fra tesseract-data-spa

# Install Python binding in virtual environment
source .venv/bin/activate
uv pip install pytesseract
```

### Debian/Ubuntu

```bash
# Install Tesseract OCR binary
sudo apt-get update
sudo apt-get install -y tesseract-ocr

# Install additional language data (optional)
# sudo apt-get install -y tesseract-ocr-deu tesseract-ocr-fra tesseract-ocr-spa

# Install Python binding in virtual environment
source .venv/bin/activate
uv pip install pytesseract
```

### macOS

```bash
# Install Tesseract OCR binary
brew install tesseract

# Install additional language data (optional)
# brew install tesseract-lang

# Install Python binding in virtual environment
source .venv/bin/activate
uv pip install pytesseract
```

## Verification

To verify that OCR is properly set up, run:

```bash
source .venv/bin/activate
python -c "from bot.ocr_utils import check_ocr_dependencies; print(check_ocr_dependencies())"
```

This should output `(True, {'tesseract_binary': True, 'pytesseract': True})` if OCR is properly configured.

## Troubleshooting

### OCR Not Available Warning

If you see the warning "OCR is not available on this server" when uploading image-based PDFs:

1. Check if Tesseract is installed:
   ```bash
   which tesseract
   tesseract --version
   ```

2. Check if pytesseract is installed:
   ```bash
   source .venv/bin/activate
   python -c "import pytesseract; print(pytesseract.__version__)"
   ```

3. Verify the Tesseract path is correctly set:
   ```bash
   source .venv/bin/activate
   python -c "import pytesseract; print(pytesseract.get_tesseract_cmd())"
   ```

### Poor OCR Quality

If OCR quality is poor:

1. Ensure you have the appropriate language data installed
2. Try preprocessing images before OCR (the bot does this automatically)
3. Consider using a more recent version of Tesseract (5.x+ recommended)

## Advanced Configuration

### Custom Tesseract Path

If Tesseract is installed in a non-standard location, you can set the path in your `.env` file:

```
TESSERACT_CMD=/path/to/tesseract
```

### Additional Languages

To support OCR for additional languages:

1. Install the appropriate language data for Tesseract
2. Set the language in your `.env` file:

```
TESSERACT_LANG=eng+deu+fra
```

## Monitoring

The bot logs OCR-related events:

- INFO level: OCR initialization and availability
- WARNING level: Missing OCR dependencies
- DEBUG level: OCR processing details and timing

Monitor these logs to diagnose any issues with OCR functionality.
