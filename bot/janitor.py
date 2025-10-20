"""
Periodic cache and log janitor for long-running bot instances. [CA][REH][PA]

Runs every 60 minutes to:
- Rotate and compress logs
- Prune caches by age and size
- Keep disk usage under control without affecting routing/gating/pipelines

Guardrails:
- Never touches in-flight files (30-minute hold-off)
- Never deletes active log files
- Conservative batch limits (500 files max per run per directory)
- Cross-platform safe (basic ops only)
"""

from __future__ import annotations

import asyncio
import gzip
import logging
import random
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ===== POLICY CONSTANTS (inline, tweakable, no ENV sprawl) =====

@dataclass
class DirectoryPolicy:
    """Policy for a single directory."""
    path: Path
    age_ttl_hours: Optional[float] = None  # Delete files older than this
    size_cap_mb: Optional[int] = None  # Keep total under this size
    max_files_per_run: int = 500  # Batch limit to keep runs short


# Default policies - read from existing config paths
LOG_ROTATION_SIZE_MB = 50  # Rotate active log at this size
LOG_ROTATION_BACKUPS = 5  # Keep this many backups before compression
LOG_COMPRESS_AGE_HOURS = 1.0  # Compress rotated logs older than this
LOG_RETENTION_DAYS = 7  # Keep compressed logs this long
LOG_TOTAL_CAP_MB = 256  # Total logs directory cap

# Hold-off window: skip files modified < 30 minutes ago
HOLD_OFF_MINUTES = 30

# Scheduler: run every 60 min with ±5 min jitter
JANITOR_INTERVAL_MINUTES = 60
JANITOR_JITTER_MINUTES = 5


# ===== DIRECTORY SCANNER & SIZE CALCULATOR =====

def get_directory_size_bytes(path: Path) -> int:
    """Calculate total size of all files in directory (recursive). [PA]"""
    total = 0
    try:
        for item in path.rglob("*"):
            if item.is_file():
                try:
                    total += item.stat().st_size
                except (OSError, PermissionError):
                    pass
    except (OSError, PermissionError):
        pass
    return total


def get_files_by_mtime(path: Path, pattern: str = "*") -> List[Tuple[Path, float]]:
    """Get all files matching pattern with their mtime, sorted oldest first. [CA]"""
    files = []
    try:
        for item in path.rglob(pattern):
            if item.is_file():
                try:
                    mtime = item.stat().st_mtime
                    files.append((item, mtime))
                except (OSError, PermissionError):
                    pass
    except (OSError, PermissionError):
        pass
    
    files.sort(key=lambda x: x[1])  # Sort by mtime ascending (oldest first)
    return files


def is_recent_file(file_path: Path, hold_off_minutes: float) -> bool:
    """Check if file was modified recently (within hold-off window). [REH]"""
    try:
        mtime = file_path.stat().st_mtime
        age_minutes = (time.time() - mtime) / 60.0
        return age_minutes < hold_off_minutes
    except (OSError, PermissionError):
        # If we can't stat it, assume it's recent to be safe
        return True


# ===== LOG COMPRESSION =====

def compress_file_to_gz(file_path: Path) -> bool:
    """Compress file to .gz and delete original. Returns True on success. [CA][REH]"""
    try:
        gz_path = Path(str(file_path) + ".gz")
        if gz_path.exists():
            # Already compressed
            return False
        
        with open(file_path, "rb") as f_in:
            with gzip.open(gz_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Verify the compressed file exists before deleting original
        if gz_path.exists():
            file_path.unlink()
            return True
        else:
            return False
    except Exception as e:
        logger.debug(f"Failed to compress {file_path}: {e}")
        return False


# ===== CACHE PRUNING =====

def prune_by_age(
    policy: DirectoryPolicy,
    hold_off_minutes: float,
    active_file: Optional[Path] = None
) -> Tuple[int, int]:
    """
    Delete files older than TTL. Returns (files_deleted, bytes_freed). [CA][REH]
    
    Args:
        policy: Directory policy with age_ttl_hours
        hold_off_minutes: Skip files modified within this window
        active_file: Never delete this file (e.g., current log)
    """
    if not policy.path.exists() or policy.age_ttl_hours is None:
        return 0, 0
    
    cutoff_time = time.time() - (policy.age_ttl_hours * 3600)
    files_deleted = 0
    bytes_freed = 0
    skipped_recent = 0
    
    try:
        files = get_files_by_mtime(policy.path)
        
        for file_path, mtime in files:
            # Safety checks
            if files_deleted >= policy.max_files_per_run:
                break
            
            if active_file and file_path.resolve() == active_file.resolve():
                continue
            
            if is_recent_file(file_path, hold_off_minutes):
                skipped_recent += 1
                continue
            
            # Check age
            if mtime < cutoff_time:
                try:
                    size = file_path.stat().st_size
                    file_path.unlink()
                    files_deleted += 1
                    bytes_freed += size
                except FileNotFoundError:
                    # File disappeared between stat and unlink - that's fine
                    pass
                except (OSError, PermissionError) as e:
                    logger.debug(f"Could not delete {file_path}: {e}")
        
        if skipped_recent > 0:
            logger.debug(f"Skipped {skipped_recent} recent files in {policy.path}")
    
    except (OSError, PermissionError) as e:
        logger.warning(f"Could not access directory {policy.path}: {e}")
    
    return files_deleted, bytes_freed


def prune_by_size(
    policy: DirectoryPolicy,
    hold_off_minutes: float,
    active_file: Optional[Path] = None
) -> Tuple[int, int]:
    """
    Delete oldest files until under size cap. Returns (files_deleted, bytes_freed). [CA][REH]
    
    Args:
        policy: Directory policy with size_cap_mb
        hold_off_minutes: Skip files modified within this window
        active_file: Never delete this file
    """
    if not policy.path.exists() or policy.size_cap_mb is None:
        return 0, 0
    
    size_cap_bytes = policy.size_cap_mb * 1024 * 1024
    current_size = get_directory_size_bytes(policy.path)
    
    if current_size <= size_cap_bytes:
        return 0, 0
    
    files_deleted = 0
    bytes_freed = 0
    target_to_free = current_size - size_cap_bytes
    
    try:
        files = get_files_by_mtime(policy.path)
        
        for file_path, mtime in files:
            # Safety checks
            if files_deleted >= policy.max_files_per_run:
                break
            
            if bytes_freed >= target_to_free:
                break
            
            if active_file and file_path.resolve() == active_file.resolve():
                continue
            
            if is_recent_file(file_path, hold_off_minutes):
                continue
            
            try:
                size = file_path.stat().st_size
                file_path.unlink()
                files_deleted += 1
                bytes_freed += size
            except FileNotFoundError:
                pass
            except (OSError, PermissionError) as e:
                logger.debug(f"Could not delete {file_path}: {e}")
    
    except (OSError, PermissionError) as e:
        logger.warning(f"Could not access directory {policy.path}: {e}")
    
    return files_deleted, bytes_freed


# ===== LOG ROTATION & COMPRESSION =====

def compress_old_logs(log_dir: Path, compress_age_hours: float) -> int:
    """
    Compress log files older than compress_age_hours. Returns count compressed. [CA]
    
    Only compresses files matching *.log or *.jsonl that aren't already .gz
    """
    if not log_dir.exists():
        return 0
    
    compressed_count = 0
    cutoff_time = time.time() - (compress_age_hours * 3600)
    
    try:
        # Look for uncompressed log files
        for pattern in ["*.log", "*.jsonl"]:
            for log_file in log_dir.glob(pattern):
                try:
                    mtime = log_file.stat().st_mtime
                    
                    # Skip if too recent
                    if mtime >= cutoff_time:
                        continue
                    
                    # Skip if it looks like the active log (contains "bot.log" or "bot.jsonl")
                    if "bot.log" in log_file.name or "bot.jsonl" in log_file.name:
                        # Check if it's actually the current one
                        # Current logs are typically named exactly "bot.log" or "bot.jsonl"
                        # Rotated ones have suffixes like ".1", ".2", etc.
                        if not any(c.isdigit() for c in log_file.suffix):
                            # Likely the active log
                            continue
                    
                    if compress_file_to_gz(log_file):
                        compressed_count += 1
                
                except (OSError, PermissionError):
                    pass
    
    except (OSError, PermissionError) as e:
        logger.warning(f"Could not access log directory {log_dir}: {e}")
    
    return compressed_count


def prune_old_compressed_logs(log_dir: Path, retention_days: int, total_cap_mb: int) -> Tuple[int, int]:
    """
    Prune compressed logs by age and total size. Returns (files_deleted, bytes_freed). [CA][REH]
    """
    if not log_dir.exists():
        return 0, 0
    
    files_deleted = 0
    bytes_freed = 0
    cutoff_time = time.time() - (retention_days * 86400)
    
    try:
        # First pass: delete by age
        for gz_file in log_dir.glob("*.gz"):
            try:
                mtime = gz_file.stat().st_mtime
                if mtime < cutoff_time:
                    size = gz_file.stat().st_size
                    gz_file.unlink()
                    files_deleted += 1
                    bytes_freed += size
            except FileNotFoundError:
                pass
            except (OSError, PermissionError):
                pass
        
        # Second pass: enforce total cap
        current_size = get_directory_size_bytes(log_dir)
        cap_bytes = total_cap_mb * 1024 * 1024
        
        if current_size > cap_bytes:
            # Delete oldest .gz files first
            gz_files = get_files_by_mtime(log_dir, "*.gz")
            target_to_free = current_size - cap_bytes
            
            for file_path, mtime in gz_files:
                if bytes_freed >= target_to_free:
                    break
                
                try:
                    size = file_path.stat().st_size
                    file_path.unlink()
                    files_deleted += 1
                    bytes_freed += size
                except FileNotFoundError:
                    pass
                except (OSError, PermissionError):
                    pass
    
    except (OSError, PermissionError) as e:
        logger.warning(f"Could not access log directory {log_dir}: {e}")
    
    return files_deleted, bytes_freed


# ===== JANITOR ORCHESTRATOR =====

class Janitor:
    """Main janitor orchestrator. [CA][REH][PA]"""
    
    def __init__(self):
        self._task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._running = False
        self._policies: Dict[str, DirectoryPolicy] = {}
        
    def _load_policies_from_config(self) -> None:
        """Load directory policies from existing config paths. [CA]"""
        from .config import load_config
        
        config = load_config()
        
        # Build policies from config
        self._policies = {
            "logs": DirectoryPolicy(
                path=config.get("LOGS_DIR", Path("logs")),
                age_ttl_hours=LOG_RETENTION_DAYS * 24,
                size_cap_mb=LOG_TOTAL_CAP_MB,
            ),
            "video_audio": DirectoryPolicy(
                path=Path(config.get("VIDEO_CACHE_DIR", "cache/video_audio")),
                age_ttl_hours=3 * 24,  # 3 days
                size_cap_mb=2048,  # 2 GB
            ),
            "stt": DirectoryPolicy(
                path=Path(config.get("STT_CACHE_DIR", "stt/cache")),
                age_ttl_hours=24,  # 24 hours
                size_cap_mb=1024,  # 1 GB
            ),
            "stt_pcm": DirectoryPolicy(
                path=Path(config.get("STT_PCM_CACHE_DIR", "cache/stt_pcm")),
                age_ttl_hours=12,
                size_cap_mb=768,
            ),
            "stt_transcripts": DirectoryPolicy(
                path=Path(config.get("STT_TRANSCRIPT_CACHE_DIR", "cache/stt_transcripts")),
                age_ttl_hours=48,
                size_cap_mb=256,
            ),
            "tts": DirectoryPolicy(
                path=Path("cache/tts"),
                age_ttl_hours=24,
                size_cap_mb=512,
            ),
            "http": DirectoryPolicy(
                path=Path("cache/screenshots"),  # HTTP/download cache
                age_ttl_hours=24,
                size_cap_mb=512,
            ),
            "temp": DirectoryPolicy(
                path=config.get("TEMP_DIR", Path("temp")),
                age_ttl_hours=6,  # 6 hours
                size_cap_mb=None,  # No cap, just TTL
            ),
        }
        
        # Ensure directories exist
        for name, policy in self._policies.items():
            try:
                policy.path.mkdir(parents=True, exist_ok=True)
            except (OSError, PermissionError):
                logger.debug(f"Could not create directory {policy.path}")
    
    async def _run_once(self) -> None:
        """Run a single janitor cycle. [CA][REH]"""
        start_time = time.monotonic()
        
        try:
            logger.info(
                "janitor.run started",
                extra={
                    "subsys": "janitor",
                    "event": "janitor.run",
                    "detail": {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "interval_min": JANITOR_INTERVAL_MINUTES,
                    },
                },
            )
            
            # Reload policies from config (in case of hot-reload)
            self._load_policies_from_config()
            
            # Process logs first (compression + pruning)
            await self._process_logs()
            
            # Process each cache directory
            for name, policy in self._policies.items():
                if name == "logs":
                    continue  # Already handled
                
                await self._process_directory(name, policy)
            
            # Calculate duration
            duration_ms = (time.monotonic() - start_time) * 1000
            next_run_min = JANITOR_INTERVAL_MINUTES + random.uniform(
                -JANITOR_JITTER_MINUTES, JANITOR_JITTER_MINUTES
            )
            
            logger.info(
                "janitor.run done",
                extra={
                    "subsys": "janitor",
                    "event": "janitor.run.done",
                    "detail": {
                        "duration_ms": round(duration_ms, 1),
                        "next_in_min": round(next_run_min, 1),
                    },
                },
            )
        
        except Exception as e:
            logger.error(
                f"janitor.run failed: {e}",
                extra={
                    "subsys": "janitor",
                    "event": "janitor.run.error",
                    "detail": {"error": str(e)[:200]},
                },
                exc_info=True,
            )
    
    async def _process_logs(self) -> None:
        """Process log directory: compress and prune. [CA]"""
        policy = self._policies.get("logs")
        if not policy or not policy.path.exists():
            return
        
        try:
            before_bytes = get_directory_size_bytes(policy.path)
            
            # Compress old log files
            compressed_count = compress_old_logs(policy.path, LOG_COMPRESS_AGE_HOURS)
            
            # Prune old compressed logs
            deleted_count, bytes_freed = prune_old_compressed_logs(
                policy.path, LOG_RETENTION_DAYS, LOG_TOTAL_CAP_MB
            )
            
            after_bytes = get_directory_size_bytes(policy.path)
            
            logger.info(
                f"janitor.dir name=logs before_bytes={before_bytes} after_bytes={after_bytes} "
                f"deleted_files={deleted_count} deleted_bytes={bytes_freed} compressed_logs={compressed_count}",
                extra={
                    "subsys": "janitor",
                    "event": "janitor.dir",
                    "detail": {
                        "name": "logs",
                        "before_bytes": before_bytes,
                        "after_bytes": after_bytes,
                        "deleted_files": deleted_count,
                        "deleted_bytes": bytes_freed,
                        "compressed_logs": compressed_count,
                    },
                },
            )
        
        except Exception as e:
            logger.warning(
                f"janitor.warn dir=logs reason={str(e)[:100]}",
                extra={
                    "subsys": "janitor",
                    "event": "janitor.warn",
                    "detail": {"dir": "logs", "reason": str(e)[:200]},
                },
            )
    
    async def _process_directory(self, name: str, policy: DirectoryPolicy) -> None:
        """Process a single cache directory. [CA][REH]"""
        if not policy.path.exists():
            return
        
        try:
            before_bytes = get_directory_size_bytes(policy.path)
            
            # Prune by age first
            age_deleted = 0
            age_bytes_freed = 0
            if policy.age_ttl_hours is not None:
                age_deleted, age_bytes_freed = prune_by_age(policy, HOLD_OFF_MINUTES)
            
            # Then prune by size if needed
            size_deleted = 0
            size_bytes_freed = 0
            if policy.size_cap_mb is not None:
                size_deleted, size_bytes_freed = prune_by_size(policy, HOLD_OFF_MINUTES)
            
            after_bytes = get_directory_size_bytes(policy.path)
            total_deleted = age_deleted + size_deleted
            total_freed = age_bytes_freed + size_bytes_freed
            
            if total_deleted > 0 or total_freed > 0:
                logger.info(
                    f"janitor.dir name={name} before_bytes={before_bytes} after_bytes={after_bytes} "
                    f"deleted_files={total_deleted} deleted_bytes={total_freed}",
                    extra={
                        "subsys": "janitor",
                        "event": "janitor.dir",
                        "detail": {
                            "name": name,
                            "before_bytes": before_bytes,
                            "after_bytes": after_bytes,
                            "deleted_files": total_deleted,
                            "deleted_bytes": total_freed,
                        },
                    },
                )
        
        except Exception as e:
            logger.warning(
                f"janitor.warn dir={name} reason={str(e)[:100]}",
                extra={
                    "subsys": "janitor",
                    "event": "janitor.warn",
                    "detail": {"dir": name, "reason": str(e)[:200]},
                },
            )
    
    async def _run_loop(self) -> None:
        """Main janitor loop with jittered intervals. [REH][PA]"""
        try:
            while self._running:
                # Run janitor cycle
                await self._run_once()
                
                # Calculate next run time with jitter
                jitter = random.uniform(-JANITOR_JITTER_MINUTES, JANITOR_JITTER_MINUTES)
                sleep_minutes = JANITOR_INTERVAL_MINUTES + jitter
                sleep_seconds = sleep_minutes * 60
                
                # Sleep until next run
                await asyncio.sleep(sleep_seconds)
        
        except asyncio.CancelledError:
            logger.info(
                "Janitor task cancelled",
                extra={"subsys": "janitor", "event": "janitor.cancelled"},
            )
            raise
        except Exception as e:
            logger.error(
                f"Janitor loop crashed: {e}",
                extra={
                    "subsys": "janitor",
                    "event": "janitor.crash",
                    "detail": {"error": str(e)[:200]},
                },
                exc_info=True,
            )
    
    async def start(self) -> None:
        """Start the janitor task. [CA]"""
        async with self._lock:
            if self._running:
                logger.warning("Janitor already running")
                return
            
            self._running = True
            self._load_policies_from_config()
            self._task = asyncio.create_task(self._run_loop())
            
            logger.info(
                "Janitor started",
                extra={
                    "subsys": "janitor",
                    "event": "janitor.start",
                    "detail": {
                        "interval_min": JANITOR_INTERVAL_MINUTES,
                        "jitter_min": JANITOR_JITTER_MINUTES,
                        "hold_off_min": HOLD_OFF_MINUTES,
                    },
                },
            )
    
    async def stop(self) -> None:
        """Stop the janitor task. [CA]"""
        async with self._lock:
            if not self._running:
                return
            
            self._running = False
            
            if self._task and not self._task.done():
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
            
            logger.info(
                "Janitor stopped",
                extra={"subsys": "janitor", "event": "janitor.stop"},
            )
    
    async def restart(self) -> None:
        """Restart the janitor (for hot-reload). [CA]"""
        logger.info(
            "Janitor restarting",
            extra={"subsys": "janitor", "event": "janitor.restart"},
        )
        await self.stop()
        await self.start()


# Global janitor instance
_janitor: Optional[Janitor] = None


async def start_janitor() -> None:
    """Start the global janitor instance. [CA]"""
    global _janitor
    
    if _janitor is None:
        _janitor = Janitor()
    
    await _janitor.start()


async def stop_janitor() -> None:
    """Stop the global janitor instance. [CA]"""
    global _janitor
    
    if _janitor is not None:
        await _janitor.stop()


async def restart_janitor() -> None:
    """Restart the global janitor instance (for hot-reload). [CA]"""
    global _janitor
    
    if _janitor is not None:
        await _janitor.restart()


async def manual_clean() -> Dict[str, Any]:
    """
    Manually trigger janitor cleaning (for admin commands). [CA]
    
    Returns a summary dict with stats for user feedback.
    """
    global _janitor
    
    if _janitor is None:
        return {
            "success": False,
            "error": "Janitor not initialized",
        }
    
    # Create a temporary results tracker
    results = {
        "success": True,
        "directories_processed": 0,
        "total_files_deleted": 0,
        "total_bytes_freed": 0,
        "logs_compressed": 0,
    }
    
    try:
        # Reload policies
        _janitor._load_policies_from_config()
        
        # Process logs
        log_policy = _janitor._policies.get("logs")
        if log_policy and log_policy.path.exists():
            compressed_count = compress_old_logs(log_policy.path, LOG_COMPRESS_AGE_HOURS)
            deleted_count, bytes_freed = prune_old_compressed_logs(
                log_policy.path, LOG_RETENTION_DAYS, LOG_TOTAL_CAP_MB
            )
            results["logs_compressed"] = compressed_count
            results["total_files_deleted"] += deleted_count
            results["total_bytes_freed"] += bytes_freed
            results["directories_processed"] += 1
        
        # Process cache directories
        for name, policy in _janitor._policies.items():
            if name == "logs" or not policy.path.exists():
                continue
            
            # Prune by age
            age_deleted = 0
            age_bytes = 0
            if policy.age_ttl_hours is not None:
                age_deleted, age_bytes = prune_by_age(policy, HOLD_OFF_MINUTES)
            
            # Prune by size
            size_deleted = 0
            size_bytes = 0
            if policy.size_cap_mb is not None:
                size_deleted, size_bytes = prune_by_size(policy, HOLD_OFF_MINUTES)
            
            if age_deleted > 0 or size_deleted > 0:
                results["total_files_deleted"] += age_deleted + size_deleted
                results["total_bytes_freed"] += age_bytes + size_bytes
                results["directories_processed"] += 1
        
        logger.info(
            "Manual clean completed",
            extra={
                "subsys": "janitor",
                "event": "janitor.manual_clean",
                "detail": results,
            },
        )
        
        return results
    
    except Exception as e:
        logger.error(f"Manual clean failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
        }
