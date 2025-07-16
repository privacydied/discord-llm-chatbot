#!/usr/bin/env bash
# Build patched gruut wheel with NumPy 2.x compatibility
set -e

echo "🔧 Building patched gruut wheel for NumPy 2.x compatibility..."

# Set up temporary build directory
BUILD_DIR="$(mktemp -d)"
VENDOR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/vendor"
mkdir -p "${VENDOR_DIR}"

# Clean up on exit
trap 'rm -rf "${BUILD_DIR}"' EXIT

# Install wheel package needed for building
pip install wheel

# Download gruut source
echo "📥 Downloading gruut-2.2.3 source..."
cd "${BUILD_DIR}"

# Use pip to download the source distribution
pip download gruut==2.2.3 --no-binary=:all: --no-deps

# Find the downloaded tarball
GRUUT_TARBALL=$(find . -name "gruut-2.2.3.tar.gz")
if [ -z "${GRUUT_TARBALL}" ]; then
    echo "❌ Failed to download gruut-2.2.3.tar.gz"
    exit 1
fi

# Extract the tarball
echo "📦 Extracting gruut source..."
mkdir -p gruut-src
tar -xzf "${GRUUT_TARBALL}" -C gruut-src --strip-components=1

# Check if pyproject.toml exists
if [ ! -f "gruut-src/pyproject.toml" ]; then
    echo "❌ pyproject.toml not found in extracted source"
    find gruut-src -type f -name "*.toml" -o -name "setup.py"
    # Look for setup.py instead
    if [ -f "gruut-src/setup.py" ]; then
        echo "📝 Found setup.py, patching that instead..."
        sed -i 's/"numpy>=1.19.0,<2.0.0"/"numpy>=1.19.0"/' gruut-src/setup.py
    else
        echo "❌ No setup files found to patch"
        exit 1
    fi
else
    # Patch pyproject.toml to remove NumPy version ceiling
    echo "🔄 Patching NumPy dependency constraint in pyproject.toml..."
    sed -i 's/numpy>=1.19.0,<2.0.0/numpy>=1.19.0/' gruut-src/pyproject.toml
fi

# Build wheel
echo "🏗️ Building patched wheel..."
cd gruut-src
pip install -e .
pip wheel . --no-deps -w "${VENDOR_DIR}"

# Rename wheel to indicate it's patched
cd "${VENDOR_DIR}"
WHEEL_NAME=$(find . -name "gruut-2.2.3-*.whl")
if [ -n "${WHEEL_NAME}" ]; then
    NEW_NAME=$(echo "${WHEEL_NAME}" | sed 's/gruut-2.2.3-/gruut-2.2.3.post1-/')
    mv "${WHEEL_NAME}" "${NEW_NAME}"
    echo "✅ Successfully built patched wheel: ${NEW_NAME}"
else
    echo "❌ Failed to find built wheel"
    exit 1
fi
