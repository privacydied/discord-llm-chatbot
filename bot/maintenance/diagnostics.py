"""Maintenance diagnostics and SQLite WAL checkpoint utilities."""

from __future__ import annotations

import asyncio
import os
import sqlite3
from pathlib import Path

from bot.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Size helpers
# ---------------------------------------------------------------------------


async def _dir_size(path: str) -> int:
    """Return total size of *path* (dir or file) using os.walk on a thread."""
    path = os.path.expandvars(os.path.expanduser(path))
    if not os.path.exists(path):
        return 0
    try:
        return await asyncio.to_thread(_dir_size_sync, path)
    except Exception:
        return 0


def _dir_size_sync(path: str) -> int:
    if os.path.isfile(path):
        try:
            return os.path.getsize(path)
        except OSError:
            return 0
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                if not os.path.islink(fp):
                    total += os.path.getsize(fp)
            except OSError:
                continue
    return total


def _fmt_bytes(n: int) -> str:
    """Human-readable byte count."""
    if n < 1024:
        return f"{n} B"
    for unit in ("KB", "MB", "GB", "TB"):
        n /= 1024.0
        if n < 1024:
            return f"{n:.1f} {unit}"
    return f"{n:.1f} PB"


# ---------------------------------------------------------------------------
# Storage status
# ---------------------------------------------------------------------------


async def get_storage_status() -> str:
    """Return a concise multi-line storage report suitable for Discord.

    Each field is capped under 200 characters.  All filesystem operations run
    on a background thread via asyncio.to_thread to avoid blocking the event
    loop.
    """
    # Project root — walk up from this file
    proj_root = Path(__file__).resolve().parent.parent.parent
    root_str = str(proj_root)

    # Paths relative to project root
    memory_db = Path(root_str) / "data" / "memory.sqlite3"
    server_db = Path(root_str) / "data" / "server_archive.sqlite3"
    chroma_dir = Path(root_str) / "chroma_db"
    tts_cache = Path(root_str) / "tts_cache"
    screenshot_cache = Path(root_str) / "screenshot_cache"
    vision_ledger = Path(root_str) / "vision_data" / "ledger.jsonl"
    logs_dir = Path(root_str) / "logs"

    lines: list[str] = []

    # --- memory DB -----------------------------------------------
    if memory_db.exists():
        wal_path = str(memory_db) + "-wal"
        db_bytes = await _dir_size(str(memory_db))
        wal_bytes = await _dir_size(wal_path) if os.path.exists(wal_path) else 0
        lines.append(f"memory DB: {_fmt_bytes(db_bytes)}" + (f", WAL: {_fmt_bytes(wal_bytes)}" if wal_bytes else ""))
    else:
        lines.append("memory DB: not found")

    # --- server archive DB ---------------------------------------
    if server_db.exists():
        lines.append(f"server archive DB: {_fmt_bytes(await _dir_size(str(server_db)))}")
    else:
        lines.append("server archive DB: not found")

    # --- ChromaDB ------------------------------------------------
    if chroma_dir.exists():
        lines.append(f"ChromaDB: {_fmt_bytes(await _dir_size(str(chroma_dir)))}")
    else:
        lines.append("ChromaDB: not found")

    # --- TTS cache -----------------------------------------------
    if tts_cache.exists():
        lines.append(f"TTS cache: {_fmt_bytes(await _dir_size(str(tts_cache)))}")
    else:
        lines.append("TTS cache: not found")

    # --- screenshot cache ----------------------------------------
    if screenshot_cache.exists():
        lines.append(f"screenshot cache: {_fmt_bytes(await _dir_size(str(screenshot_cache)))}")
    else:
        lines.append("screenshot cache: not found")

    # --- vision ledger -------------------------------------------
    if vision_ledger.exists():
        lines.append(f"vision ledger: {_fmt_bytes(await _dir_size(str(vision_ledger)))}")
    else:
        lines.append("vision ledger: not found")

    # --- logs dir ------------------------------------------------
    if logs_dir.exists():
        lines.append(f"logs: {_fmt_bytes(await _dir_size(str(logs_dir)))}")
    else:
        lines.append("logs: not found")

    # --- disk usage warning --------------------------------------
    try:
        usage = await asyncio.to_thread(os.statvfs, root_str)
        total = usage.f_blocks * usage.f_frsize
        free = usage.f_bavail * usage.f_frsize
        if total > 0:
            used_pct = ((total - free) / total) * 100
            if used_pct >= 90:
                lines.append(f"WARNING: volume {used_pct:.0f}% full (critical)")
            elif used_pct >= 80:
                lines.append(f"CAUTION: volume {used_pct:.0f}% full")
    except Exception:
        pass

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# WAL checkpoint
# ---------------------------------------------------------------------------


async def checkpoint_wal(db_path: str) -> bool:
    """Safely checkpoint and truncate the SQLite WAL at *db_path*.

    Returns ``True`` on success, ``False`` on any failure.
    All blocking I/O is shunted to a thread executor.
    """
    db_path = os.path.expandvars(os.path.expanduser(db_path))

    if not os.path.exists(db_path):
        logger.warning("checkpoint_wal: db not found: %s", db_path)
        return False

    try:

        def _do_checkpoint() -> bool:
            conn = sqlite3.connect(db_path, timeout=10)
            try:
                cursor = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                row = cursor.fetchone()
                if row is not None:
                    code, total, checkpointed = row
                    logger.info(
                        "checkpoint_wal: code=%s total=%s checkpointed=%s",
                        code,
                        total,
                        checkpointed,
                    )
                return True
            finally:
                conn.close()

        return await asyncio.to_thread(_do_checkpoint)
    except Exception as exc:
        logger.error("checkpoint_wal failed for %s: %s", db_path, exc)
        return False
