# Discord LLM Chatbot Dockerfile
# Multi-stage build for smaller final image

FROM python:3.11-slim-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip wheel setuptools && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# =============================================================================
# Final runtime image
# =============================================================================
FROM python:3.11-slim-bookworm

# Labels
LABEL maintainer="pry"
LABEL description="Discord LLM Chatbot"
LABEL version="0.2.0"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/venv/bin:$PATH" \
    HOME=/app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # FFmpeg for audio/video processing
    ffmpeg \
    # Tesseract OCR
    tesseract-ocr \
    tesseract-ocr-eng \
    # Poppler for PDF processing
    poppler-utils \
    # espeak-ng for phonemizer/TTS
    espeak-ng \
    libespeak-ng1 \
    # libmagic for file type detection
    libmagic1 \
    # Sound libraries
    libsndfile1 \
    # Additional utilities
    curl \
    ca-certificates \
    # Clean up
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Install Playwright browsers (Chromium only for smaller size)
RUN playwright install chromium && \
    playwright install-deps chromium

# Create app user for security (uid 1026 to match Synology NAS pry user)
RUN useradd --create-home --shell /bin/bash --uid 1026 --gid 100 botuser

# Create app directory structure
WORKDIR /app

# Create necessary directories
RUN mkdir -p \
    /app/cache \
    /app/chroma_db \
    /app/kb \
    /app/logs \
    /app/prompts \
    /app/tts \
    /app/user_logs \
    /app/user_profiles \
    /app/server_profiles \
    /app/vision_data \
    /app/vision_artifacts \
    /app/temp \
    /app/stt \
    /app/tts_cache \
    && chown -R 1026:100 /app

# Copy application code
COPY --chown=1026:100 bot/ /app/bot/
COPY --chown=1026:100 utils/ /app/utils/
COPY --chown=1026:100 configs/ /app/configs/
COPY --chown=1026:100 run.py /app/
COPY --chown=1026:100 pyproject.toml /app/

# Switch to non-root user
USER botuser

# Expose Prometheus metrics port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/metrics || exit 1

# Default command
CMD ["python", "run.py"]
