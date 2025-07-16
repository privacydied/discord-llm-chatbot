#!/usr/bin/env bash
# Dependency conflict resolution script for discord-llm-chatbot
# Handles the TTS/NumPy conflict by using vendored patched gruut wheel
set -e

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="${PROJECT_ROOT}/vendor"

echo "🔧 Setting up virtual environment..."
#uv venv --python 3.11
python -m venv .venv
source .venv/bin/activate

# Create vendor directory if it doesn't exist
mkdir -p "${VENDOR_DIR}"

# Build patched gruut wheel if it doesn't exist
if [[ ! -f "${VENDOR_DIR}"/gruut-2.2.3.post1-*.whl ]]; then
    echo "🔨 Building patched gruut wheel..."
    "${PROJECT_ROOT}/scripts/build_gruut_patch.sh"
fi

echo "📦 Installing dependencies..."
# First install numpy 2.x and other core dependencies
uv pip install --upgrade "numpy>=2.0.2,<2.3" "librosa>=0.11.0" "numba>=0.60" "kokoro-onnx==0.4.9" "soundfile>=0.12.1" "pytest>=7.0.0"

echo "🔄 Installing TTS with patched gruut..."
# Try different TTS versions in order of preference
for TTS_VERSION in "0.22.0" "0.21.1" "0.21.0"; do
    echo "Trying TTS==${TTS_VERSION}..."
    if uv pip install "TTS==${TTS_VERSION}" --find-links "${VENDOR_DIR}" 2>/dev/null; then
        echo "✅ Successfully installed TTS ${TTS_VERSION}"
        break
    else
        echo "❌ Failed to install TTS ${TTS_VERSION}, trying next version..."
    fi
done

echo "🧩 Installing remaining dependencies from pyproject.toml..."
# Install remaining dependencies from pyproject.toml
uv pip install --no-deps -e .

# Install additional runtime dependencies needed for tests
echo "📚 Installing additional runtime dependencies..."
uv pip install "python-dotenv>=1.0.0" "discord.py>=2.3.0" "python-json-logger>=2.0.7" "cachetools>=5.3.0" "faster-whisper>=0.10.0" "beautifulsoup4>=4.12.0" "PyMuPDF>=1.23.0" "trafilatura>=2.0.0" "lxml>=4.9.0" "aiohttp>=3.8.0"

echo "✅ Dependencies installed successfully!"

# Verify numpy version
echo "🔍 Verifying numpy version:"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"

# Verify TTS is importable
echo "🔍 Verifying TTS is importable:"
python -c "import TTS; print(f'TTS version: {TTS.__version__}')" || echo "Failed to import TTS"

# Verify kokoro_onnx is importable
echo "🔍 Verifying kokoro_onnx is importable:"
python -c "import kokoro_onnx; print('kokoro_onnx is available')" || echo "Failed to import kokoro_onnx"

# Verify gruut is importable
echo "🔍 Verifying gruut is importable:"
python -c "import gruut; print(f'gruut version: {gruut.__version__}')" || echo "Failed to import gruut"

# Run TTS smoke test if it exists
if [[ -f "${PROJECT_ROOT}/tests/test_tts_smoke.py" ]]; then
    echo "🔊 Running TTS smoke test..."
    cd "${PROJECT_ROOT}"
    # Run pytest with -k to select only the test_tts_smoke.py file and override addopts
    PYTHONPATH="${PROJECT_ROOT}" python -m pytest tests/test_tts_smoke.py -xvs -o addopts=
fi
