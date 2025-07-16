# TTS and OCR Integration Guide

This guide explains how to use and configure the Text-to-Speech (TTS) and Optical Character Recognition (OCR) features in the Discord bot.

## TTS Features

The bot uses kokoro-onnx as its primary TTS engine, which provides high-quality speech synthesis.

### Environment Variables

The TTS system supports the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `TTS_BACKEND` | TTS engine to use | `kokoro-onnx` |
| `TTS_MODEL_PATH` | Path to the ONNX model file | `models/kokoro.onnx` |
| `TTS_VOICES_PATH` | Path to the voices file | `models/voices.npz` |
| `TTS_VOICE` | Voice to use | `default` |
| `TTS_PHONEMISER` | Override default phonemizer selection | Language-specific default |

> **Note:** The older variables `TTS_MODEL_FILE` and `TTS_VOICE_FILE` are still supported for backward compatibility but are deprecated. Please use `TTS_MODEL_PATH` and `TTS_VOICES_PATH` instead.

### Tokenization Methods

The TTS system supports multiple tokenization methods:

- **PHONEME_ENCODE**: Uses the model's built-in tokenizer's `encode` method
- **PHONEME_TO_ID**: Uses the model's built-in tokenizer's `phoneme_to_id` method
- **ESPEAK**: Uses espeak-ng for phonemization (best for English and most languages)
- **PHONEMIZER**: Uses the Python phonemizer library
- **MISAKI**: Uses the Misaki phonemizer (best for Japanese)

The system automatically selects the best tokenization method based on the language and available dependencies.

### System Requirements

For full TTS functionality, the following dependencies are required:

- **Python packages**:
  - kokoro-onnx (>=0.4.9)
  - numpy (>=2.0.2)
  - librosa (>=0.11.0)
  - phonemizer (>=3.2.0)
  - soundfile (>=0.12.1)

- **System dependencies**:
  - espeak-ng: For phonemization (recommended for best quality)
  - libsndfile: For audio file handling

## OCR Features

The bot includes OCR capabilities for extracting text from image-based PDFs.

### System Requirements

For OCR functionality, the following dependencies are required:

- **Python packages**:
  - PyMuPDF (>=1.23.0)

- **System dependencies**:
  - tesseract-ocr: For text extraction from images
  - tesseract-data-eng: English language data for Tesseract (or other language packs as needed)

### OCR Soft Dependency

The OCR functionality is implemented as a soft dependency. If Tesseract is not available:

1. The system will still function normally for text-based PDFs
2. A warning will be logged when an image-based PDF is encountered
3. Only the text that can be extracted without OCR will be returned

## Troubleshooting

### TTS Issues

- **Silent or zero audio output**: The system now detects and rejects silent audio output. If this occurs, check:
  - Model and voice file compatibility
  - Input text format and language
  - Speaker embedding configuration

- **Tokenization errors**: If you see tokenization errors, ensure:
  - espeak-ng is installed for your language
  - The appropriate language model is available
  - Try setting `TTS_PHONEMISER` to override the default selection

### OCR Issues

- **Missing text from PDFs**: If text is missing from PDFs, check:
  - Tesseract is installed and in your PATH
  - Appropriate language data is installed
  - The PDF contains actual images (not just formatted text)

## Installation

All dependencies can be installed using the `fix_deps.sh` script:

```bash
# Activate virtual environment
source .venv/bin/activate

# Run the dependency fix script
./scripts/fix_deps.sh
```

For system dependencies on Arch Linux:

```bash
sudo pacman -S tesseract tesseract-data-eng espeak-ng
```

## Testing

The TTS and OCR functionality can be tested using the included test suite:

```bash
# Run TTS tests
python -m pytest tests/test_tts_*.py -v
```
