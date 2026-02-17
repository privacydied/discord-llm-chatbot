"""ffmpeg runtime helper utilities for STT pipeline."""

from __future__ import annotations

import os
import re
import subprocess
from typing import List


def ffmpeg_candidates_from_env() -> List[str]:
    """Return ordered ffmpeg binary candidates with env override support."""
    candidates: List[str] = []
    for env_key in ("STT_FFMPEG_BIN", "FFMPEG_BIN", "FFMPEG_BINARY"):
        value = (os.getenv(env_key) or "").strip()
        if value:
            candidates.append(value)
    # Prefer Synology ffmpeg7 package when present; fallback to default ffmpeg.
    candidates.extend(["ffmpeg7", "ffmpeg"])
    # Preserve order while de-duplicating.
    seen = set()
    ordered: List[str] = []
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        ordered.append(c)
    return ordered


def ffmpeg_supports_aac_decoder(ffmpeg_bin: str) -> bool:
    """Check whether ffmpeg binary exposes AAC decoder(s)."""
    try:
        proc = subprocess.run(
            [ffmpeg_bin, "-hide_banner", "-decoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=5,
            check=False,
        )
        if proc.returncode != 0:
            return False
        out = proc.stdout or ""
        return bool(re.search(r"\baac(?:_fixed|_latm)?\b", out))
    except Exception:
        return False
