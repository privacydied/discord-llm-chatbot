#!/usr/bin/env python3
from __future__ import annotations

# Simple sanity import test for native voice message integration
# This script just imports the modules to ensure there are no ImportErrors

import sys
from pathlib import Path

# Ensure project root is on sys.path when run from utils/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import bot.voice.publisher as publisher
    import bot.commands.tts_cmds as tts_cmds

    print("OK: Imports succeeded:", publisher.__name__, tts_cmds.__name__)
    sys.exit(0)
except Exception as e:
    # Print the error and exit non-zero for visibility in CI or local checks
    print(f"IMPORT_ERROR: {e}")
    raise
