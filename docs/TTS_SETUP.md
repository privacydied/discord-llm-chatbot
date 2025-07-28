# Text-to-Speech Setup Guide

This document provides instructions for setting up and troubleshooting the Text-to-Speech (TTS) functionality in the Discord bot.

## Overview

The bot uses an engine-based TTS architecture with KokoroEngine as the default engine. This enables four canonical flows:

1. TEXT→TEXT: Standard text responses
2. TEXT→TTS: Text responses converted to voice notes
3. IMAGE/DOC→TEXT: Vision/document processing with text responses
4. IMAGE/DOC→TTS: Vision/document processing with voice note responses

## Requirements

- Python 3.11
- Virtual environment with `uv` for dependency management
- Internet connection for initial model downloads

## Environment Setup

1. Ensure your `.env` file contains the necessary TTS configuration:

```
# TTS voice ID to use for speech synthesis (must exist in voices-v1.0.bin)
TTS_VOICE=en-US-GuyNeural

# TTS language code for phoneme generation
TTS_LANGUAGE=en-US

# TTS backend to use (currently only kokoro-onnx is supported)
TTS_BACKEND=kokoro-onnx

# Directory for TTS cache files
TTS_CACHE_DIR=tts_cache

# Paths for TTS model and voice assets
TTS_ASSET_DIR=tts/assets
TTS_MODEL_PATH=tts/onnx
TTS_VOICES_PATH=tts/voices

# URLs for downloading TTS model files
TTS_VOICE_BIN_URL=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
TTS_MODEL_URL=https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/onnx/model.onnx?download=true

# SHA256 checksums for TTS model files
TTS_MODEL_SHA256=c5b230b4529e5c15b6d810f64424e9e7e5b54f2c6214f8f5c9a9813d8f7ad3f2
TTS_VOICE_BIN_SHA256=7c6c5a3ab28b21c6f3afa9563f9ea6d3c4d5d9e1c8f7a6b5c4d3e2f1a0b9c8d7
```

2. Install dependencies:

```bash
source .venv/bin/activate
uv pip install -r requirements.txt
```

3. For better English TTS quality, install one of these phonetic tokenizers:

```bash
# Option 1: Install phonemizer (Python package)
source .venv/bin/activate
uv pip install phonemizer

# Option 2: Install espeak-ng (system package)
sudo pacman -Sy espeak-ng  # For Arch Linux
# OR
sudo apt install espeak-ng  # For Debian/Ubuntu

# Option 3: Install g2p_en (Python package, English only)
source .venv/bin/activate
uv pip install g2p_en
```

Without a phonetic tokenizer, the TTS quality may be reduced as it will fall back to grapheme tokenization.

## Note on Gruut Fork

This project uses a fork of the Gruut package and its English language plugin to support NumPy 2.0. The fork is available at:
- gruut: `https://github.com/<org>/gruut@1.4.0-np2`
- gruut-lang-en: `https://github.com/<org>/gruut-lang-en@2.0.0-np2`

We have submitted a pull request to the upstream repository. Once merged, we will revert to the official packages.

## Automatic Asset Management

The bot includes an automatic asset management system that:

1. Checks for required TTS model files on startup
2. Downloads missing files automatically
3. Verifies file integrity using SHA256 checksums
4. Implements retry logic with exponential backoff
5. Validates voice bin format and structure

To manually trigger asset verification and download:

```bash
python -m bot.tts_utils_enhanced
```

## TTS Commands

The bot supports the following TTS commands:

1. `@bot !tts on` - Enable TTS responses for the user
2. `@bot !tts off` - Disable TTS responses for the user
3. `@bot !tts-all on` - (Admin only) Enable TTS responses for all users
4. `@bot !tts-all off` - (Admin only) Disable TTS responses for all users
5. `!speak <message>` - Generate a one-time TTS response
6. `!say <text>` - Speak the exact text as a voice note

## Troubleshooting

### TTS Not Working

1. Check logs for TTS-related errors:
   - Look for messages from `TTSManager`, `KokoroEngine`, or `tts_utils_enhanced`
   - Debug logs will show engine loading, TTS generation, and file operations

2. Verify model files:
   - Ensure `tts/onnx/model.onnx` exists and has the correct checksum
   - Ensure `tts/voices/voices-v1.0.bin` exists and has the correct checksum
   - Run `python -m bot.tts_utils_enhanced` to verify and download files

3. Check environment variables:
   - Ensure `TTS_VOICE` is set to a voice that exists in the voice bin
   - Ensure `TTS_LANGUAGE` is set correctly (e.g., `en-US`)

4. Check permissions:
   - Ensure the bot has write permissions to the TTS directories
   - Ensure the bot has permissions to send voice messages in Discord

### Common Issues

1. **Same word repeated in TTS output**:
   - Check if the style vector is being loaded correctly
   - Verify the voice bin file integrity
   - Try a different voice ID

2. **No audio output**:
   - Check if the audio file was generated successfully
   - Verify Discord permissions for sending voice messages
   - Check if the audio duration is too short (Discord has minimum length requirements)

3. **Slow TTS generation**:
   - TTS generation is CPU-intensive and may be slow on low-end hardware
   - Consider enabling the TTS cache to avoid regenerating the same text

## Testing

Run the TTS tests to verify functionality:

```bash
python -m pytest tests/test_tts_assets.py tests/tts/engines/test_kokoro.py -v
```

This will test:
- Asset downloading and verification
- Environment validation
- Voice bin structure validation
- TTS synthesis
- KokoroEngine functionality

## Advanced Configuration

### Custom Voice Styles

The default voice bin includes several voices. To use a different voice:

1. Set `TTS_VOICE` to one of the available voice IDs (e.g., `en-US-JennyNeural`)
2. Restart the bot

### Custom Model

To use a custom ONNX model:

1. Place your model file at the path specified by `KOKORO_MODEL_PATH`
2. Update the `KOKORO_MODEL_SHA256` checksum in your `.env` file
3. Restart the bot

## Monitoring

The bot includes extensive logging for TTS operations:

- Debug logs for voice loading and selection
- Info logs for TTS generation and caching
- Warning logs for retries and fallbacks
- Error logs for failures

Monitor these logs to diagnose any issues with TTS functionality.
