"""Minimal atomic JSON write helper for file-backed memory/context/profile writes.

Usage:
    from bot.atomic_json import write_json_atomic, read_json_safe

    await write_json_atomic(pathlib.Path("/data/profiles/user123.json"), data)
    data = read_json_safe(pathlib.Path("/data/profiles/user123.json"))
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from bot.utils.logging import get_logger

logger = get_logger(__name__)

# Per-path lock for concurrent write safety
_locks: dict[str, asyncio.Lock] = {}
_locks_lock: asyncio.Lock = asyncio.Lock()


async def _get_lock(path: str) -> asyncio.Lock:
    """Get or create a per-path async lock."""
    async with _locks_lock:
        if path not in _locks:
            _locks[path] = asyncio.Lock()
        return _locks[path]


async def write_json_atomic(path: Path, data: Any) -> None:
    """Atomically write *data* as JSON to *path*.

    - Acquires a per-path asyncio.Lock
    - Writes to a sibling temp file (OS atomic)
    - Runs the actual disk write via loop.run_in_executor (off event loop)
    - os.replace() into final path
    - Creates parent directories if needed

    Safe against concurrent writes to the same path and power-loss mid-write
    (temp file may be left behind but final path is never corrupt).
    """
    path = Path(path)
    lock = await _get_lock(str(path.resolve()))

    async with lock:
        path = Path(path)  # re-resolve in case it changed
        path.parent.mkdir(parents=True, exist_ok=True)

        loop = asyncio.get_running_loop()

        def _do_write():
            # Use tempfile on same filesystem so os.replace() is atomic
            fd, tmp_path = tempfile.mkstemp(
                dir=str(path.parent), prefix=f".{path.name}.tmp"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, str(path))
            except Exception:
                # Clean up temp file on failure
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

        await loop.run_in_executor(None, _do_write)


def read_json_safe(path: Path, default: Any = None) -> Any:
    """Read JSON from *path*, returning *default* on any error.

    Tolerates corrupt JSON (logged as warning) and missing files.
    This is synchronous — call from async code normally (it's just
    a file read, not a write, so blocking is acceptable for small files).
    """
    path = Path(path)
    if not path.exists():
        return default

    try:
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            logger.warning("Empty JSON file at %s", path)
            return default
        return json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("Corrupt JSON at %s: %s", path, exc)
        return default
    except Exception as exc:
        logger.error("Failed to read %s: %s", path, exc)
        return default
