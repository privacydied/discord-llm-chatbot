#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# CONFIG: Toggle layers (1=include, 0=exclude)
# =========================================================
INCLUDE_ML=1         # torch / transformers
INCLUDE_STT=1        # faster-whisper
INCLUDE_OCR=1        # HTML/PDF extraction + OCR
INCLUDE_TIME=       # date/time locale helpers
INCLUDE_DEV=0        # dev tools (tests, lint)
LOCK=1               # 1=create/update requirements.lock using uv pip compile
PYTHON_VERSION="${PYTHON_VERSION:-3.13.5}"  # for uv venv
VENV_DIR=".venv"

# =========================================================
# Prepare layered requirement source files (.in)
# =========================================================
mkdir -p requirements

# --- Base (minimal bot) ---
cat > requirements/base.in <<'EOF'
discord.py==2.5.2
openai==1.97.0
httpx==0.28.1
pydantic==2.11.7
python-dotenv==1.1.1
python-json-logger==3.3.0
rich==14.0.0
regex==2024.11.6
typing-extensions==4.14.1
EOF

# --- ML (models / local processing) ---
cat > requirements/extras-ml.in <<'EOF'
torch==2.7.1
transformers==4.53.2
tokenizers==0.21.2
safetensors==0.5.3
tqdm==4.67.1
numpy==2.3.1
EOF

# --- STT (speech to text) ---
cat > requirements/extras-stt.in <<'EOF'
faster-whisper==1.1.1
ctranslate2==4.6.0
onnxruntime==1.22.1
EOF

# --- OCR / Web / PDF ---
cat > requirements/extras-ocr.in <<'EOF'
beautifulsoup4==4.13.4
lxml==5.4.0
lxml-html-clean==0.4.2
trafilatura==2.0.0
courlan==1.3.2
justext==3.0.2
htmldate==1.9.3
tld==0.13.1
pymupdf==1.26.3
pytesseract==0.3.13
EOF

# --- Time / locale ---
cat > requirements/extras-time.in <<'EOF'
dateparser==1.2.2
babel==2.17.0
tzlocal==5.3.1
EOF

# --- Dev / tooling ---
cat > requirements/dev.in <<'EOF'
pytest==8.2.0
ruff
black
pipdeptree
EOF

echo "✅ Layer .in files written."

# =========================================================
# Assemble combined requirements.txt (sorted unique)
# =========================================================
tmp_combined="$(mktemp)"

cat requirements/base.in >> "$tmp_combined"
[ "$INCLUDE_ML"   = "1" ] && cat requirements/extras-ml.in   >> "$tmp_combined"
[ "$INCLUDE_STT"  = "1" ] && cat requirements/extras-stt.in  >> "$tmp_combined"
[ "$INCLUDE_OCR"  = "1" ] && cat requirements/extras-ocr.in  >> "$tmp_combined"
[ "$INCLUDE_TIME" = "1" ] && cat requirements/extras-time.in >> "$tmp_combined"
[ "$INCLUDE_DEV"  = "1" ] && cat requirements/dev.in         >> "$tmp_combined"

# Normalize: strip blanks, sort, unique
grep -Ev '^\s*$' "$tmp_combined" | sort -u > requirements.txt
rm -f "$tmp_combined"

echo "✅ Generated requirements.txt (layers applied)."

# =========================================================
# (Optional) Produce a LOCK file with hashes using uv pip compile
# =========================================================
if [ "$LOCK" = "1" ]; then
  echo "🔒 Compiling lock (requirements.lock) with uv..."
  # uv pip compile reads constraints and emits a fully pinned file.
  # We feed the already combined requirements.txt
  uv pip compile requirements.txt -o requirements.lock
  echo "✅ Created requirements.lock"
fi

# =========================================================
# Recreate virtual environment using uv (fast)
# =========================================================
if [ -d "$VENV_DIR" ]; then
  echo "⚠️  Removing existing venv: $VENV_DIR"
  rm -rf "$VENV_DIR"
fi

echo "🛠  Creating venv ($PYTHON_VERSION) via uv..."
uv venv --python "$PYTHON_VERSION" "$VENV_DIR"

# Activate
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# =========================================================
# Install dependencies with uv
# Prefer lock if present (reproducible)
# =========================================================
if [ -f requirements.lock ]; then
  echo "📥 Syncing from lock file..."
  uv pip sync requirements.lock
else
  echo "📥 Installing from requirements.txt (no lock)..."
  uv pip install -r requirements.txt
fi

echo "🔍 Verifying core imports..."
python - <<'PY'
import importlib, sys
mods = ["discord", "openai", "httpx", "pydantic"]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        missing.append((m, str(e)))
if missing:
    print("❌ Missing imports:", missing)
    sys.exit(1)
print("✅ Core imports OK.")
PY

echo "📄 Top installed packages:"
uv pip list | head -n 40

cat <<'MSG'

🎉 Done.

Files:
  requirements/base.in          (core)
  requirements/extras-*.in      (layer sources)
  requirements.txt              (assembled)
  requirements.lock             (fully pinned, if LOCK=1)

To modify layers:
  - Edit INCLUDE_* flags at top of script
  - Rerun script (it will rebuild venv + lock)

To add a package:
  - Append to appropriate .in file
  - Rerun script

To skip lock creation next run:
  - Set LOCK=0

MSG

