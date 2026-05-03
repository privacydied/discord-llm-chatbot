"""
Atomic persistence layer for JSON profile storage.

This module provides atomic write operations, corruption recovery,
and optional file-level locking for safe concurrent access.

[REH][SFT][CA] Atomic writes via temp file + rename
[REH] Corruption detection and backup restoration
[PA] fsync for durability guarantees
"""

import fcntl
import json
import logging
import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Union
from contextlib import contextmanager

# Thread-local storage for file handles (used during locking)
_thread_local = threading.local()

logger = logging.getLogger(__name__)


class PersistenceError(Exception):
    """Raised when persistence operations fail."""
    pass


class CorruptionError(PersistenceError):
    """Raised when data corruption is detected."""
    pass


class AtomicWriteError(PersistenceError):
    """Raised when atomic write fails."""
    pass


def _validate_json(data: Any) -> bool:
    """
    Validate that data can be serialized to JSON.

    [REH] Pre-serialization validation to catch errors before writing
    [SFT] Prevents writing corrupt/incomplete data
    """
    try:
        json.dumps(data)
        return True
    except (TypeError, ValueError) as e:
        logger.error(f"JSON validation failed: {e}")
        return False


def _atomic_write_file(
    target_path: Path,
    data: Dict[str, Any],
    indent: int = 2,
    ensure_ascii: bool = False,
    fsync: bool = True,
) -> bool:
    """
    Atomically write JSON data to a file.

    Strategy:
    1. Write to a temp file in the same directory as target
    2. fsync the temp file for durability
    3. Rename temp file to target (atomic on POSIX)

    Args:
        target_path: Final destination path
        data: Dictionary to serialize
        indent: JSON indentation
        ensure_ascii: Whether to escape non-ASCII characters
        fsync: Whether to call fsync for durability

    Returns:
        True if successful, False otherwise

    [REH] Atomic rename guarantees readers see either old or new version
    [PA] fsync ensures data reaches physical storage
    [SFT] Temp file in same filesystem guarantees atomic rename
    """
    if not _validate_json(data):
        logger.error(f"Cannot write invalid JSON to {target_path}")
        return False

    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Create temp file in same directory for atomic rename guarantee
    temp_fd = None
    temp_path = None
    try:
        # Use tempfile with same directory for atomic rename guarantee
        temp_fd, temp_str = tempfile.mkstemp(
            dir=target_path.parent,
            prefix=f".tmp_{target_path.stem}_",
            suffix=".json",
        )
        temp_path = Path(temp_str)

        # Serialize JSON
        json_bytes = json.dumps(data, indent=indent, ensure_ascii=ensure_ascii).encode("utf-8")

        # Write and fsync for durability
        os.write(temp_fd, json_bytes)

        if fsync:
            os.fsync(temp_fd)

        os.close(temp_fd)
        temp_fd = None

        # Atomic rename - readers see either old or new version
        os.replace(temp_path, target_path)

        # Sync parent directory to ensure rename is durable
        if fsync:
            try:
                dir_fd = os.open(target_path.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except OSError as e:
                logger.warning(f"Could not sync directory {target_path.parent}: {e}")

        logger.debug(f"Atomic write successful: {target_path}")
        return True

    except OSError as e:
        logger.error(f"Atomic write failed for {target_path}: {e}")
        return False

    except Exception as e:
        logger.error(f"Unexpected error during atomic write to {target_path}: {e}")
        return False

    finally:
        # Cleanup
        if temp_fd is not None:
            try:
                os.close(temp_fd)
            except OSError:
                pass
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def _create_backup(src: Path, dst: Path, max_retries: int = 3) -> bool:
    """
    Create a backup copy with retry logic.

    [REH] Handles permission errors by attempting chmod removal
    [PA] Limited retries with error logging
    """
    src = Path(src)
    dst = Path(dst)

    for attempt in range(max_retries):
        try:
            if dst.exists():
                try:
                    dst.unlink()
                except PermissionError:
                    try:
                        os.chmod(dst, 0o600)
                        dst.unlink()
                    except Exception as e:
                        logger.error(f"Cannot remove existing backup {dst}: {e}")
                        return False
            shutil.copy2(src, dst)
            return True
        except Exception as e:
            logger.warning(f"Backup attempt {attempt + 1} failed for {src}: {e}")

    logger.error(f"Failed to create backup for {src} after {max_retries} attempts")
    return False


def _restore_from_backup(src: Path, dst: Path) -> bool:
    """
    Restore a file from its backup.

    Returns:
        True if restoration was successful, False otherwise.

    [REH] Corruption recovery entry point
    """
    src = Path(src)
    dst = Path(dst)

    if not src.exists():
        logger.error(f"Cannot restore from backup: {src} does not exist")
        return False

    try:
        # Validate backup before restoring
        with open(src, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Write atomically
        if _atomic_write_file(dst, data, fsync=True):
            logger.info(f"Successfully restored {dst} from backup {src}")
            return True
        else:
            logger.error(f"Failed to atomically restore {dst} from backup")
            return False

    except json.JSONDecodeError as e:
        logger.error(f"Backup file {src} is corrupted: {e}")
        return False

    except Exception as e:
        logger.error(f"Failed to restore {dst} from backup {src}: {e}")
        return False


@contextmanager
def _file_lock(path: Path, exclusive: bool = True, timeout: Optional[float] = None):
    """
    Context manager for advisory file locking using fcntl.

    Args:
        path: Path to lock (uses .lock file)
        exclusive: If True, use exclusive lock; otherwise shared
        timeout: Maximum time to wait for lock (None = blocking)

    Yields:
        Path to the lock file

    [REH] Advisory locking for concurrent access safety
    [PA] Non-blocking option available with timeout
    [SFT] Lock file is separate from data file
    """
    path = Path(path)
    lock_path = path.with_suffix(path.suffix + ".lock")

    lock_file = None
    try:
        lock_file = open(lock_path, "w")

        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        if timeout is not None:
            operation |= fcntl.LOCK_NB

        try:
            if timeout is not None and timeout > 0:
                # Use alarm-based timeout for blocking lock
                import signal

                def _timeout_handler(signum, frame):
                    raise TimeoutError(f"Lock acquisition timed out for {path}")

                old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                signal.setitimer(signal.ITIMER_REAL, timeout)
                try:
                    fcntl.flock(lock_file.fileno(), operation)
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    signal.signal(signal.SIGALRM, old_handler)
            else:
                fcntl.flock(lock_file.fileno(), operation)

        except IOError as e:
            if e.errno == 11:  # EWOULDBLOCK / EAGAIN
                raise TimeoutError(f"Lock acquisition would block for {path}")
            raise

        yield lock_path

    finally:
        if lock_file is not None:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                lock_file.close()
            except Exception:
                pass


def atomic_save_json(
    target_path: Path,
    data: Dict[str, Any],
    create_backup: bool = True,
    validate_before_write: bool = True,
    use_lock: bool = True,
    indent: int = 2,
) -> bool:
    """
    Save JSON data atomically with optional backup and locking.

    This is the main entry point for profile persistence.

    Args:
        target_path: Destination path for the JSON file
        data: Dictionary to serialize
        create_backup: Whether to create a .bak file before overwriting
        validate_before_write: Whether to validate JSON before writing
        use_lock: Whether to use advisory file locking
        indent: JSON indentation level

    Returns:
        True if successful, False otherwise

    [REH] Full atomic write with backup and validation
    [SFT] Locking prevents race conditions
    """
    target_path = Path(target_path)

    if validate_before_write and not _validate_json(data):
        logger.error(f"Invalid JSON data, not writing to {target_path}")
        return False

    backup_path = target_path.with_suffix(target_path.suffix + ".bak")

    try:
        context = _file_lock(target_path) if use_lock else _null_context()
        with context:
            # Create backup of existing file
            if create_backup and target_path.exists():
                if not _create_backup(target_path, backup_path):
                    logger.warning(f"Could not create backup for {target_path}")
                    # Continue anyway - atomic write may still succeed

            # Atomic write
            if not _atomic_write_file(target_path, data, indent=indent, fsync=True):
                logger.error(f"Atomic write failed for {target_path}")

                # Attempt restoration from backup if it exists
                if backup_path.exists():
                    logger.info(f"Attempting to restore {target_path} from backup")
                    if _restore_from_backup(backup_path, target_path):
                        # Retry the write once more
                        if _atomic_write_file(target_path, data, indent=indent, fsync=True):
                            return True

                return False

            return True

    except TimeoutError as e:
        logger.error(f"Could not acquire lock for {target_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error saving {target_path}: {e}")
        return False


@contextmanager
def _null_context():
    """Null context manager for when locking is disabled."""
    yield None


def load_json_with_recovery(
    target_path: Path,
    default_data: Optional[Dict] = None,
    attempt_recovery: bool = True,
) -> Optional[Dict]:
    """
    Load JSON data with corruption recovery.

    Args:
        target_path: Path to JSON file
        default_data: Data to return if file doesn't exist/can't be loaded
        attempt_recovery: Whether to try restoring from backup on corruption

    Returns:
        Loaded data, or default_data if loading/recovery failed

    [REH] Automatic corruption detection and recovery
    """
    target_path = Path(target_path)

    if not target_path.exists():
        return default_data

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error for {target_path}: {e}")

        if attempt_recovery:
            backup_path = target_path.with_suffix(target_path.suffix + ".bak")

            if backup_path.exists():
                logger.info(f"Attempting recovery from backup {backup_path}")
                try:
                    with open(backup_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    # Validate recovered data is a dict before restoring [BUGFIX]
                    if not isinstance(data, dict):
                        logger.error(f"Backup data is not a dict, skipping recovery for {target_path}")
                        return default_data
                    # Attempt to restore the corrupted file
                    if _atomic_write_file(target_path, data, fsync=True):
                        logger.info(f"Successfully recovered and restored {target_path}")
                    else:
                        # Remove corrupted file so next save creates a fresh one [BUGFIX]
                        try:
                            target_path.unlink()
                            logger.warning(f"Removed corrupted file {target_path} after failed atomic restore; backup data returned")
                        except OSError:
                            pass
                    return data
                except Exception as recovery_error:
                    logger.error(f"Recovery from backup failed: {recovery_error}")

        return default_data

    except Exception as e:
        logger.error(f"Error loading {target_path}: {e}")
        return default_data


def validate_profile_integrity(data: Dict) -> tuple[bool, Optional[str]]:
    """
    Validate that a profile dictionary has basic structural integrity.

    Returns:
        Tuple of (is_valid, error_message)

    [SFT] Detects malformed profiles before they cause issues
    """
    if not isinstance(data, dict):
        return False, "Profile data is not a dictionary"

    # Check for common required fields (at least one identifier)
    has_user_id = bool(data.get("discord_id") or data.get("user_id"))
    has_guild_id = bool(data.get("guild_id") or data.get("server_id"))

    if not (has_user_id or has_guild_id):
        return False, "Profile missing identifier (discord_id/user_id/guild_id/server_id)"

    # Check required list fields exist and are lists
    for field in ["memories", "history"]:
        if field in data and not isinstance(data[field], list):
            return False, f"Profile field '{field}' is not a list"

    # Check preferences is a dict if present
    if "preferences" in data and not isinstance(data["preferences"], dict):
        return False, "Profile field 'preferences' is not a dictionary"

    return True, None
