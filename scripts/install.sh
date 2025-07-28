#!/usr/bin/env bash
set -euo pipefail

### --------------------------------------------
### CONFIG: Choose which extras to include
### Set to 1 to include, 0 to skip
### --------------------------------------------
INCLUDE_ML=1
INCLUDE_STT=0
INCLUDE_OCR=0
INCLUDE_TIME=0
INCLUDE_DEV=0

### --------------------------------------------
### Create requirements directory
### --------------------------------------------
mkdir -p requirements

### Base requirements (minimal bot + logging + OpenAI)
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

### ML extras (LLM local / embeddings etc.)
cat > requirements/extras-ml.in <<'EOF'
torch==2.7.1
transformers==4.53.2
tokenizers==0.21.2
safetensors==0.5.3
tqdm==4.67.1
numpy==2.3.1
EOF

### STT (speech-to-text)
cat > requirements/extras-stt.in <<'EOF'
faster-whisper==1.1.1
ctranslate2==4.6.0
onnxruntime==1.22.1
EOF

### OCR / Web extraction / PDF
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

### Time / locale utilities (optional)
cat > requirements/extras-time.in <<'EOF'
dateparser==1.2.2
babel==2.17.0
tzlocal==5.3.1
EOF

### Dev / tooling
cat > requirements/dev.in <<'EOF'
pytest==8.2.0
pipdeptree
black
ruff
EOF

echo "✅ Wrote layered requirement .in files."

### --------------------------------------------
### Assemble combined requirements.txt
### Only include chosen extras
### --------------------------------------------
tmp_combined=$(mktemp)

cat requirements/base.in >> "$tmp_combined"

if [ "$INCLUDE_ML" = "1" ];  then cat requirements/extras-ml.in  >> "$tmp_combined"; fi
if [ "$INCLUDE_STT" = "1" ]; then cat requirements/extras-stt.in >> "$tmp_combined"; fi
if [ "$INCLUDE_OCR" = "1" ]; then cat requirements/extras-ocr.in >> "$tmp_combined"; fi
if [ "$INCLUDE_TIME" = "1" ]; then cat requirements/extras-time.in >> "$tmp_combined"; fi
if [ "$INCLUDE_DEV" = "1" ];  then cat requirements/dev.in       >> "$tmp_combined"; fi

# Normalize (unique, sorted). Keep version pins.
sort -u "$tmp_combined" > requirements.txt
rm -f "$tmp_combined"

echo "✅ Generated combined requirements.txt (based on selected extras)."

### --------------------------------------------
### Rebuild virtual environment (optional)
### Comment this block out if you do NOT want auto recreate
### --------------------------------------------
if [ -d ".venv" ]; then
  echo "⚠️  Removing existing .venv ..."
  rm -rf .venv
fi

echo "📦 Creating fresh venv..."
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip
echo "📥 Installing selected requirements..."
pip install -r requirements.txt

echo "🔍 Verifying a few core imports..."
python - <<'PY'
mods = ["discord","openai","httpx","pydantic"]
import importlib, sys
missing = []
for m in mods:
    try: importlib.import_module(m)
    except Exception as e: missing.append((m,str(e)))
if missing:
    print("❌ Missing imports:", missing)
    sys.exit(1)
print("✅ Core imports OK.")
PY

echo "📄 Final dependency list (truncated):"
pip list --format=columns | head -n 40

echo ""
echo "🎉 Done. Adjust INCLUDE_* flags and rerun to regenerate."
echo "   - Base file: requirements/base.in"
echo "   - Combined:  requirements.txt"
echo "   - Add extras by setting INCLUDE_* = 1 and rerunning."


