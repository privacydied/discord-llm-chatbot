"""ffmpeg runtime helper utilities for STT pipeline."""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import subprocess  # nosec B404
from pathlib import Path
from typing import Any

_FFMPEG_BIN_CACHE: str | None = None
_FFMPEG_BIN_HAS_AAC: bool | None = None


def ffmpeg_candidates_from_env() -> list[str]:
    """Return ordered ffmpeg binary candidates with env override support."""
    candidates: list[str] = []
    for env_key in ("STT_FFMPEG_BIN", "FFMPEG_BIN", "FFMPEG_BINARY"):
        value = (os.getenv(env_key) or "").strip()
        if value:
            candidates.append(value)
    # Prefer Synology ffmpeg7 package when present; fallback to default ffmpeg.
    candidates.extend(["ffmpeg7", "ffmpeg"])
    # Preserve order while de-duplicating.
    seen = set()
    ordered: list[str] = []
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        ordered.append(c)
    return ordered


def ffmpeg_supports_aac_decoder(ffmpeg_bin: str) -> bool:
    """Check whether ffmpeg binary exposes AAC decoder(s)."""
    try:
        proc = subprocess.run(  # nosec B603
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


def ffmpeg_bin_has_aac() -> bool | None:
    """Return cached AAC decoder availability for selected ffmpeg binary."""
    return _FFMPEG_BIN_HAS_AAC


def reset_ffmpeg_runtime_cache() -> None:
    """Reset cached ffmpeg binary selection (used by tests)."""
    global _FFMPEG_BIN_CACHE, _FFMPEG_BIN_HAS_AAC
    _FFMPEG_BIN_CACHE = None
    _FFMPEG_BIN_HAS_AAC = None


def resolve_ffmpeg_bin(*, logger: Any | None = None) -> str:
    """Resolve and cache ffmpeg binary path with AAC capability probe."""
    global _FFMPEG_BIN_CACHE, _FFMPEG_BIN_HAS_AAC
    if _FFMPEG_BIN_CACHE:
        return _FFMPEG_BIN_CACHE

    for candidate in ffmpeg_candidates_from_env():
        ffmpeg_bin = None
        if os.path.sep in candidate:
            path_obj = Path(candidate)
            if path_obj.exists():
                ffmpeg_bin = str(path_obj)
        else:
            ffmpeg_bin = shutil.which(candidate)
        if not ffmpeg_bin:
            continue

        has_aac = ffmpeg_supports_aac_decoder(ffmpeg_bin)
        _FFMPEG_BIN_CACHE = ffmpeg_bin
        _FFMPEG_BIN_HAS_AAC = has_aac
        if logger is not None:
            with contextlib.suppress(Exception):
                logger.info(
                    "stt.ffmpeg.selected path=%s aac_decoder=%s",
                    ffmpeg_bin,
                    str(has_aac).lower(),
                )
        return ffmpeg_bin

    msg = "ffmpeg executable not found; set STT_FFMPEG_BIN to an installed ffmpeg binary"
    raise RuntimeError(msg)
