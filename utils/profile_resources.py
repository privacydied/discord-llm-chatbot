"""Baseline resource profiling utility.

Reports memory footprint, thread count, open FDs, loaded subsystem status,
and cache/directory sizes WITHOUT initializing heavy ML systems.

Usage:
    uv run python -m utils.profile_resources
"""

from __future__ import annotations

import os
import sys
import threading


def _fmt_bytes(n: int) -> str:
    """Format byte count into human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _dir_size(path: str) -> int:
    """Walk a directory tree and return total bytes. Skips broken symlinks."""
    total = 0
    if not os.path.isdir(path):
        return 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.islink(fp):
                continue
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def _sqlite_sizes(db_path: str) -> dict[str, int]:
    """Return main DB size and WAL size if present."""
    result: dict[str, int] = {"main": 0, "wal": 0, "shm": 0}
    if os.path.exists(db_path):
        try:
            result["main"] = os.path.getsize(db_path)
        except OSError:
            pass
    wal = db_path + "-wal"
    if os.path.exists(wal):
        try:
            result["wal"] = os.path.getsize(wal)
        except OSError:
            pass
    shm = db_path + "-shm"
    if os.path.exists(shm):
        try:
            result["shm"] = os.path.getsize(shm)
        except OSError:
            pass
    return result


def profile_memory(report: list[str]) -> None:
    """Report process memory metrics via psutil if available."""
    try:
        import psutil  # optional dependency
    except ImportError:
        report.append("  [!] psutil not installed — skipping memory metrics")
        return

    proc = psutil.Process(os.getpid())
    mem = proc.memory_info()

    report.append(f"  RSS : {_fmt_bytes(mem.rss)}")
    report.append(f"  VMS : {_fmt_bytes(mem.vms)}")

    # USS requires psutil >= 5.6 and Linux
    uss = getattr(proc.memory_full_info(), "uss", None)
    if uss is not None:
        report.append(f"  USS : {_fmt_bytes(uss)}")
    else:
        report.append("  USS : n/a (platform or psutil version)")


def profile_threads(report: list[str]) -> None:
    threads = threading.enumerate()
    report.append(f"  Thread count: {len(threads)}")
    report.append(f"  Main thread : {threading.main_thread().name}")
    for t in threads:
        report.append(f"    - {t.name} (alive={t.is_alive()})")


def profile_fds(report: list[str]) -> None:
    fd_path = "/proc/self/fd"
    if os.path.isdir(fd_path):
        fds = os.listdir(fd_path)
        report.append(f"  Open FDs: {len(fds)}")
    else:
        report.append("  Open FDs: n/a (/proc/self/fd not available)")


def profile_subsystems(report: list[str]) -> None:
    """Check which heavy modules are in sys.modules without importing them."""
    heavy = [
        "torch",
        "onnxruntime",
        "faster_whisper",
        "chromadb",
        "sentence_transformers",
        "PIL",
        "numpy",
        "aiohttp",
        "discord",
    ]
    report.append("  Loaded subsystems (sys.modules):")
    for name in heavy:
        status = "✅ loaded" if name in sys.modules else "⬜ not loaded"
        report.append(f"    {name:>25s} : {status}")


def profile_directories(report: list[str]) -> None:
    """Report sizes of common cache / data directories."""
    dirs = [
        "chroma_db",
        "cache",
        "data",
        "logs",
        "vision_data",
        "temp",
    ]
    report.append("  Directory sizes:")
    for d in dirs:
        sz = _dir_size(d)
        exists = os.path.isdir(d)
        flag = "✅" if exists else "⬜"
        report.append(f"    {flag} {d:>20s} : {_fmt_bytes(sz)}")


def profile_sqlite(report: list[str]) -> None:
    """Report sizes of known SQLite databases."""
    db_paths = [
        "data/memory.db",
        "data/server_archive.db",
    ]
    report.append("  SQLite databases:")
    for db in db_paths:
        sizes = _sqlite_sizes(db)
        exists = os.path.exists(db)
        flag = "✅" if exists else "⬜"
        parts = [f"{flag} {db:>30s}"]
        for kind in ("main", "wal", "shm"):
            if sizes[kind]:
                parts.append(f"{kind}={_fmt_bytes(sizes[kind])}")
        if not any(sizes[kind] for kind in sizes):
            parts.append("n/a")
        report.append(f"    {', '.join(parts)}")


def profile_asyncio_tasks(report: list[str]) -> None:
    """Report active asyncio tasks if a running event loop is available."""
    try:
        asyncio = __import__("asyncio")
        loop = asyncio.get_event_loop()
    except RuntimeError:
        report.append("  AsyncIO tasks: n/a (no running event loop)")
        return
    except Exception:
        report.append("  AsyncIO tasks: n/a (asyncio unavailable)")
        return

    tasks = asyncio.all_tasks(loop)
    report.append(f"  AsyncIO tasks: {len(tasks)}")
    for t in list(tasks)[:20]:  # cap display
        report.append(f"    - {t.get_name()} (done={t.done()})")
    if len(tasks) > 20:
        report.append(f"    ... and {len(tasks) - 20} more")


def run_profile() -> str:
    """Run all checks and return a formatted report string."""
    report: list[str] = [
        "=" * 60,
        " RESOURCE PROFILE",
        f" PID     : {os.getpid()}",
        f" Python  : {sys.version.split()[0]}",
        f" Platform: {sys.platform}",
        "=" * 60,
    ]

    section: list[str] = []
    profile_memory(section)
    report.append("\n--- Memory ---")
    report.extend(section)

    section = []
    profile_threads(section)
    report.append("\n--- Threads ---")
    report.extend(section)

    section = []
    profile_fds(section)
    report.append("\n--- File Descriptors ---")
    report.extend(section)

    section = []
    profile_subsystems(section)
    report.append("\n--- Subsystem Status ---")
    report.extend(section)

    section = []
    profile_directories(section)
    report.append("\n--- Directory Sizes ---")
    report.extend(section)

    section = []
    profile_sqlite(section)
    report.append("\n--- SQLite Databases ---")
    report.extend(section)

    section = []
    profile_asyncio_tasks(section)
    report.append("\n--- AsyncIO Tasks ---")
    report.extend(section)

    report.append("\n" + "=" * 60)
    return "\n".join(report)


def main() -> None:
    print(run_profile())


if __name__ == "__main__":
    main()
