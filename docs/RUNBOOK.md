# Discord LLM Chatbot Runbook

This document provides clear, step-by-step instructions for setting up, testing, and running the Discord LLM Chatbot.

## 1. Project Setup

These steps will guide you through setting up the project environment from scratch.

### Prerequisites

- Python 3.11
- `uv` (the project's package manager)

### Installation

1.  **Clone the Repository**

    ```bash
    git clone <repository-url>
    cd discord-llm-chatbot
    ```

2.  **Create Virtual Environment**

    This project uses a `.venv` virtual environment managed by `uv`.

    ```bash
    # Create the virtual environment with uv
    uv venv --python 3.11

    # Activate the virtual environment
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install Dependencies**

    Install all required packages using `uv` with prerelease support for kokoro-onnx.

    ```bash
    # Install kokoro-onnx with prerelease support first
    uv pip install "kokoro-onnx>=0.3.3" --prerelease=allow
    
    # Sync all other dependencies from requirements.txt
    uv pip sync requirements.txt
    
    # Install test dependencies
    uv pip install pytest pytest-asyncio pytest-cov
    ```

4.  **Configure Environment Variables**

    Copy the sample environment file and fill in your specific API keys and configuration details.

    ```bash
    cp .env-sample .env
    ```

    Now, edit the `.env` file with your credentials:
    - `DISCORD_TOKEN`: Your Discord bot token.
    - `TEXT_BACKEND`: The text generation backend (`openai` or `ollama`).
    - `KOKORO_MODEL_PATH`: Path to the Kokoro TTS model.
    - `KOKORO_VOICE_PACK_PATH`: Path to the Kokoro voice pack.
    - Any other required API keys (e.g., `OPENAI_API_KEY`).

## 2. Running the Tests

To ensure all components are working correctly, run the test suite using `pytest`. Note that we need to override the default pytest options defined in pyproject.toml.

```bash
# Make sure your virtual environment is activated
source .venv/bin/activate

# Run all tests (overriding default options)
python -m pytest tests/ -o addopts=

# Run a specific test
python -m pytest tests/test_tts_assets.py::TestTTSAssets::test_validate_voice_bin -v -o addopts=

# Run tests with coverage report
python -m pytest tests/ --cov=bot --cov-report=term-missing
```

Note: Some tests may require specific environment variables or mock data to be set up correctly. Check the test files for specific requirements.

## 3. Running the Bot

Once setup and testing are complete, you can start the bot:

```bash
# Make sure your virtual environment is activated
source .venv/bin/activate

# Start the bot from the project's root directory
python run.py
```

## Streaming Status Cards: Labeled Phases

The bot shows a streaming status card (placeholder embed) while it works on long-running tasks. These cards now include labeled phases with a spinner and an i/N footer.

-  The title shows the current phase with a spinner (e.g., `⠋ Vision analysis`).
-  The footer shows progress `i/N`. When a plan is inferred, `N` equals the number of phases; otherwise `N = STREAMING_MAX_STEPS`.
-  At completion, the placeholder is finalized and the final message either edits it (no files/audio) or replaces it (files/audio present) to preserve the 1 IN → 1 OUT rule.

### Configuration

- `STREAMING_ENABLE` (default: true)
- `STREAMING_EMBED_STYLE` (e.g., `compact`)
- `STREAMING_TICK_MS` (update cadence in ms)
- `STREAMING_MAX_STEPS` (fallback step count when no plan inferred)

### Phase Detection Rules (Heuristics)

The bot infers a plan from the message content and attachments:

-  **Online Search** (`!search`, `[search]`)
   - Parsing query → Hitting provider → Collecting results → Ranking & dedupe → Generating response
-  **Multi-Image** (2+ image attachments)
   - Collecting images → Pre-processing (hash/resize) → Vision analysis → Fusing context → Generating response
-  **Video URLs (yt-dlp)** (URL contains `youtu`)
   - Processing link → Fetching metadata → Extracting audio → Transcribing audio → Generating response
-  **Audio/Video Files** (audio/video attachments)
   - Validating file → Extracting audio → Transcribing audio → Generating response
-  **Single Image** (exactly 1 image attachment)
   - Uploading/validating → Vision analysis → Generating response
-  **PDF Documents** (PDF attachment)
   - Parsing PDF → Chunking pages → Extracting text → Summarizing → Generating response
-  **PDF Documents (OCR)** (PDF + content mentions `ocr`)
   - Rasterizing pages → OCR → Text cleanup → Summarizing → Generating response
-  **General URLs** (non-YouTube URL in content)
   - Fetching page → Extracting content → De-boilerplating → Summarizing → Generating response
-  **RAG commands**
   - `!rag bootstrap`: Discovering sources → Chunking → Embedding → Indexing → Ready
   - `!rag scan`: Scanning changes → Chunking → Embedding → Indexing → Updated
   - `!rag wipe`: Stopping jobs → Dropping index → Clearing cache → Verifying → Done

If no plan can be inferred, the bot falls back to generic phases labeled “Working…” with the configured step count.

## Memory and Context Management

This project supports configurable memory systems to balance context quality, persistence, and resource usage. See the dedicated guide at `docs/MEMORY_AND_CONTEXT.md`.

### Quick Overview

-  **Conversation Context (ephemeral):**
   - `MAX_CONTEXT_MESSAGES` controls how many recent messages are kept in-model for each reply.
   - `IN_MEMORY_CONTEXT_ONLY` forces context to be maintained only in RAM (no disk persistence of conversation logs).

-  **User Memory (persistent per user):**
   - `MAX_USER_MEMORY` limits the number of salient “memories” per user (e.g., preferences, recurring details).
   - Saved periodically per `MEMORY_SAVE_INTERVAL`.

-  **Server Memory (persistent per guild/server):**
   - `MAX_SERVER_MEMORY` caps the number of global server-level memories.
   - Useful for shared knowledge (channels, norms, roles).

-  **Optional paths and limits:**
   - `USER_PROFILE_DIR`, `SERVER_PROFILE_DIR`, `DM_LOGS_DIR`, `TTS_PREFS_FILE`, `MAX_CONVERSATION_LOG_SIZE`, `DEBUG`.

Example `.env`:

```env
# Context & Memory
MAX_CONTEXT_MESSAGES=30
IN_MEMORY_CONTEXT_ONLY=false
MAX_USER_MEMORY=5
MAX_SERVER_MEMORY=100
MEMORY_SAVE_INTERVAL=30
# Optional paths and limits
#USER_PROFILE_DIR=var/data/users
#SERVER_PROFILE_DIR=var/data/servers
#DM_LOGS_DIR=var/log/dm
#TTS_PREFS_FILE=var/data/tts_prefs.json
#MAX_CONVERSATION_LOG_SIZE=10000
```

## 4. Troubleshooting

### Dependency Issues

If you encounter dependency conflicts, especially with kokoro-onnx and numpy versions:

```bash
# Reinstall kokoro-onnx with prerelease support
uv pip install "kokoro-onnx>=0.3.3" --prerelease=allow

# Sync dependencies
uv pip sync requirements.txt
```

### TTS Voice Validation Issues

If TTS voice validation fails:

1. Check that the voice bin file exists at the path specified in your .env file
2. Ensure the voice bin file is in the correct format (npz archive)
3. Run the specific test to validate:
   ```bash
   python -m pytest tests/test_tts_assets.py::TestTTSAssets::test_validate_voice_bin -v -o addopts=
   ```

The bot should now be online and responding to commands in your Discord server.

## 5. RAG System Commands

The bot includes a Retrieval Augmented Generation (RAG) system that allows it to search through and reference documents in its knowledge base. Here are the available RAG commands:

### RAG Command Overview

All RAG commands require administrator permissions and use the `!rag` prefix:

```
!rag - Shows all available RAG commands
```

### Status and Information Commands

```
!rag status - Shows the current status of the RAG system
!rag stats - Displays search statistics
!rag config - Shows the current RAG configuration
```

### Knowledge Base Management

```
!rag bootstrap [--force] - Initialize the knowledge base from files in the kb/ directory
                           Use --force to rebuild the entire knowledge base
                           Supports pdf, txt, md, html, docx, and other document formats
```

The bootstrap command supports multiple file formats:
- PDF (.pdf)
- Markdown (.md)
- Text (.txt)
- HTML (.html)
- EPUB (.epub)
- MOBI (.mobi)
- Word Documents (.docx)

### Testing and Search

```
!rag search <query> - Test the RAG search with a specific query
!rag test - Run system tests to verify RAG functionality
```

### Troubleshooting RAG Issues

If you encounter issues with the RAG system:

1. Check that all required files exist in the `kb/` directory
2. Verify that the ChromaDB backend is properly initialized
3. Run `!rag status` to check system health
4. Run `!rag test` to perform diagnostic tests
5. Try rebuilding the knowledge base with `!rag bootstrap --force`
