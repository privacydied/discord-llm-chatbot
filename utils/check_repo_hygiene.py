#!/usr/bin/env python3
"""
Repo hygiene checker — fails CI if banned/unnatural artifact paths,
generated media, or committed cache/database files are present.

Usage: uv run python utils/check_repo_hygiene.py

Exits 0 on clean, 1 if violations found (prints each offending path).
Does NOT delete anything — it is a readonly audit tool.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

# Top-level directories that must not exist (or must be empty of tracked files)
BANNED_DIRS: list[str] = [
    "build",
    "MagicMock",
]

# Patterns that should only appear under .gitignore entries
COMMITTED_CACHE_PATTERNS: tuple[str, ...] = (
    "__pycache__",
    ".pytest_cache",
    "*.egg-info",
)

# Media extensions that should not be committed as artifacts
MEDIA_EXTS: frozenset[str] = frozenset(
    {".wav", ".ogg", ".mp3", ".mp4", ".webm", ".gif"}
)

# Size threshold for flagging individual files (10 MiB)
SIZE_THRESHOLD = 10 * 1024 * 1024

# Known exceptions — large/essential files that are intentionally tracked
ALLOWED_LARGE: frozenset[str] = frozenset(
    {
        # Add filenames here if a tracked file legitimately exceeds SIZE_THRESHOLD
    }
)

# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def _iter_repo_files() -> Iterator[Path]:
    """Walk the repo but skip .git, venv, and .nox."""
    skip = {".git", "venv", ".venv", ".nox", "node_modules"}
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        dirnames[:] = [d for d in dirnames if d not in skip]
        for fn in filenames:
            yield Path(dirpath) / fn


def _is_top_level_banned(p: Path) -> str | None:
    """Return a violation message if *p* lives inside a banned top-level dir."""
    try:
        rel = p.relative_to(REPO_ROOT)
    except ValueError:
        return None
    parts = rel.parts
    if parts and parts[0] in BANNED_DIRS:
        return f"BANNED_DIR  {p}"
    return None


def _is_committed_cache(p: Path) -> str | None:
    """Return a violation if the path looks like committed cache/__pycache__."""
    for part in p.parts:
        if part in COMMITTED_CACHE_PATTERNS:
            return f"CACHE_COMMIT  {p}"
    return None


def _is_untracked_media(p: Path) -> str | None:
    """Return a violation if a media file is committed."""
    if p.suffix.lower() in MEDIA_EXTS:
        return f"MEDIA_ARTIFACT  {p}"
    return None


def _is_oversized(p: Path) -> str | None:
    """Return a violation if a tracked file exceeds SIZE_THRESHOLD."""
    name = p.name
    if name in ALLOWED_LARGE:
        return None
    try:
        size = p.stat().st_size
    except OSError:
        return None
    if size > SIZE_THRESHOLD:
        return f"OVERSIZED ({_fmt_size(size)})  {p}"
    return None


def _fmt_size(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TiB"


def check_repo(root: Path | None = None) -> list[str]:
    """Run all hygiene checks; return list of violation strings."""
    global REPO_ROOT
    if root is not None:
        REPO_ROOT = Path(root).resolve()

    violations: list[str] = []

    # Check banned top-level directories
    for d in BANNED_DIRS:
        if (REPO_ROOT / d).exists():
            violations.append(f"BANNED_DIR  {REPO_ROOT / d}/ exists — run: rm -rf {d}/")

    # Walk files
    for p in _iter_repo_files():
        rel = p.relative_to(REPO_ROOT)
        # Skip .gitignore itself and the hygiene script
        if rel in (
            Path(".gitignore"),
            Path("utils/check_repo_hygiene.py"),
        ):
            continue
        for check in (_is_top_level_banned, _is_committed_cache, _is_untracked_media, _is_oversized):
            v = check(p)
            if v:
                violations.append(v)

    return violations


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    violations = check_repo()
    if violations:
        print(f"\n{len(violations)} hygiene violation(s) found:\n")
        for v in violations:
            print(f"  ✗ {v}")
        print("\nRun `git rm --cached <path>` and add to .gitignore if these are local artifacts.")
        return 1
    else:
        print("Repo hygiene: OK")
        return 0


if __name__ == "__main__":
    sys.exit(main())
