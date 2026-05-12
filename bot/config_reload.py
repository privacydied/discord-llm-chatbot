"""
Dynamic configuration reloading system with hot-reload support.
Supports SIGHUP signal handling, file watching, and manual reload commands.
"""

import signal
import hashlib
import asyncio
import os
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, Set, Callable, List
from datetime import datetime
from dotenv import dotenv_values, load_dotenv

from .config import load_config
from .utils.logging import get_logger

logger = get_logger(__name__)

# Import janitor for hot-reload restart
_janitor_imported = False
try:
    from .janitor import restart_janitor

    _janitor_imported = True
except ImportError:
    restart_janitor = None

# Global state for configuration management
_current_config: Dict[str, Any] = {}
_config_version: str = ""
_config_lock = threading.RLock()
_reload_callbacks: Set[Callable[[Dict[str, Any], Dict[str, Any]], None]] = set()
_file_watcher_task: Optional[asyncio.Task] = None
_last_reload_time: float = 0
_watcher_lock: Optional[asyncio.Lock] = None


def _candidate_env_paths() -> List[Path]:
    """Return list of candidate env files to watch (resolved absolute paths).
    Order matters; the first existing is used as the preferred .env for default reloads.
    Includes repo-root variants and yoroi.env for developer workflows. [CMV]
    """
    root = Path(__file__).parent.parent.resolve()
    cwd = Path.cwd().resolve()
    candidates = [
        cwd / ".env",
        root / ".env",
        cwd / "yoroi.env",
        root / "yoroi.env",
    ]
    # Deduplicate while preserving order
    seen = set()
    out: List[Path] = []
    for p in candidates:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(rp)
    return out


def _preferred_env_path() -> Path:
    for p in _candidate_env_paths():
        if p.exists():
            return p
    # fallback to cwd/.env
    return (Path.cwd() / ".env").resolve()


_env_file_path = _preferred_env_path()
_last_env_digest: Optional[str] = None
_env_loaded_values_by_path: Dict[Path, Dict[str, str]] = {}


def _read_dotenv_values(path: Path) -> Dict[str, str]:
    """Read dotenv key/value pairs without mutating process env."""
    try:
        return {
            str(key): str(value)
            for key, value in dotenv_values(path).items()
            if key and value is not None
        }
    except Exception:
        return {}


def _snapshot_known_env_files() -> None:
    """Track values loaded from existing env files so deletion hot-reloads work."""
    for candidate in _candidate_env_paths():
        if candidate.exists():
            _env_loaded_values_by_path[candidate.resolve()] = _read_dotenv_values(
                candidate
            )


def _sync_dotenv_file(path: Path) -> None:
    """Apply one dotenv file, including removals from previous versions.

    python-dotenv updates/adds keys but intentionally does not remove variables that
    disappear from the file. Hot-reload semantics for this bot should mirror the file,
    so keys previously loaded from the same file are removed when deleted, unless the
    live process env was changed to a different value by some external source.
    """
    resolved = path.resolve()
    old_values = _env_loaded_values_by_path.get(resolved, {})
    new_values = _read_dotenv_values(resolved) if resolved.exists() else {}

    for key, old_value in old_values.items():
        if key not in new_values and os.environ.get(key) == old_value:
            os.environ.pop(key, None)

    if resolved.exists():
        load_dotenv(dotenv_path=resolved, override=True)

    _env_loaded_values_by_path[resolved] = new_values


_snapshot_known_env_files()


def _file_digest(p: Path) -> Optional[str]:
    try:
        data = p.read_bytes()
        return hashlib.sha256(data).hexdigest()
    except Exception:
        return None


# Sensitive keys that should be redacted in logs
SENSITIVE_KEYS = {
    "DISCORD_TOKEN",
    "OPENAI_API_KEY",
    "WHISPER_API_KEY",
    "API_KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASS",
}


def _generate_config_version(config: Dict[str, Any]) -> str:
    """Generate a hash-based version identifier for the configuration."""
    # Create a deterministic string representation of non-sensitive config
    config_str = ""
    for key in sorted(config.keys()):
        if key not in SENSITIVE_KEYS:
            config_str += f"{key}={config[key]}\n"

    return hashlib.sha256(config_str.encode()).hexdigest()[:12]


def _redact_sensitive_values(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create a copy of config with sensitive values redacted for logging."""
    redacted = {}
    for key, value in config.items():
        if any(sensitive in key.upper() for sensitive in SENSITIVE_KEYS):
            if value:
                redacted[key] = "[REDACTED]"
            else:
                redacted[key] = value
        else:
            redacted[key] = value
    return redacted


def _compare_configs(
    old_config: Dict[str, Any], new_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Compare two configurations and return the differences."""
    changes = {"added": {}, "removed": {}, "modified": {}, "unchanged_count": 0}

    old_keys = set(old_config.keys())
    new_keys = set(new_config.keys())

    # Added keys
    for key in new_keys - old_keys:
        changes["added"][key] = new_config[key]

    # Removed keys
    for key in old_keys - new_keys:
        changes["removed"][key] = old_config[key]

    # Modified keys
    for key in old_keys & new_keys:
        if old_config[key] != new_config[key]:
            changes["modified"][key] = {"old": old_config[key], "new": new_config[key]}
        else:
            changes["unchanged_count"] += 1

    return changes


def _infer_subsystems(changes: Dict[str, Any]) -> Set[str]:
    """Infer which subsystems are impacted based on changed keys. [CMV]"""
    impacted: Set[str] = set()
    keys: Set[str] = set()
    keys.update((changes.get("added") or {}).keys())
    keys.update((changes.get("removed") or {}).keys())
    keys.update((changes.get("modified") or {}).keys())

    for k in keys:
        ku = k.upper()
        if (
            ku.startswith("OPENAI_")
            or ku.startswith("OLLAMA_")
            or ku.startswith("TEXT_")
        ):
            impacted.add("text")
        if ku.startswith("VL_") or ku.startswith("VISION_"):
            impacted.add("vision")
        if ku.startswith("STT_") or ku.startswith("WHISPER_"):
            impacted.add("stt")
        if ku.startswith("TTS_"):
            impacted.add("tts")
        if ku.startswith("LOG_"):
            impacted.add("logging")
        if "PROXY" in ku or ku.startswith("HTTP_"):
            impacted.add("http")
    return impacted


def reload_env(env_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Reload environment variables from .env file and update configuration.

    Returns:
        Dictionary with reload results including success status, changes, and version
    """
    global _current_config, _config_version, _last_reload_time

    with _config_lock:
        try:
            # Compute current env digest (preload) to support idempotency
            # Choose target path
            target_path = (env_path or _preferred_env_path()).resolve()
            env_before = _file_digest(target_path) if target_path.exists() else None
            logger.info(
                "config.reload.started",
                extra={
                    "event": "config.reload.started",
                    "detail": {
                        "prev_digest": env_before or "",
                        "path": str(target_path),
                    },
                },
            )

            # Store old config for comparison
            old_config = _current_config.copy()
            old_version = _config_version

            # Reload .env file. python-dotenv does not remove deleted keys, so
            # sync through our tracker instead of calling load_dotenv directly.
            if target_path.exists():
                _sync_dotenv_file(target_path)
                logger.debug(f"📁 Reloaded .env from {target_path}")
            else:
                _sync_dotenv_file(target_path)
                logger.warning(f"⚠️ .env file not found at {target_path}")

            # Load new configuration
            try:
                # Invalidate cached snapshot so load_config reflects new env immediately
                from .config import invalidate_config_cache

                invalidate_config_cache()
            except Exception:
                pass

            new_config = load_config()

            env_after = _file_digest(target_path) if target_path.exists() else None

            # Validate critical required variables
            required_vars = ["DISCORD_TOKEN", "PROMPT_FILE", "VL_PROMPT_FILE"]
            missing_vars = [var for var in required_vars if not new_config.get(var)]
            if missing_vars:
                logger.error(
                    f"❌ Critical variables missing after reload: {missing_vars}"
                )
                return {
                    "success": False,
                    "error": f"Missing required variables: {missing_vars}",
                    "version": old_version,
                }

            # Generate new version
            new_version = _generate_config_version(new_config)

            # Compare configurations
            changes = _compare_configs(old_config, new_config)

            # Update global state
            _current_config = new_config
            _config_version = new_version
            _last_reload_time = time.time()

            # Diff breadcrumb
            try:
                redacted_added = _redact_sensitive_values(changes.get("added", {}))
                redacted_removed = _redact_sensitive_values(changes.get("removed", {}))
                redacted_modified = {
                    k: (
                        "[REDACTED]"
                        if any(s in k.upper() for s in SENSITIVE_KEYS)
                        else v
                    )
                    for k, v in changes.get("modified", {}).items()
                }
                logger.info(
                    "config.reload.diff",
                    extra={
                        "event": "config.reload.diff",
                        "detail": {
                            "added": list(redacted_added.keys()),
                            "removed": list(redacted_removed.keys()),
                            "modified": list(redacted_modified.keys()),
                        },
                    },
                )
            except Exception:
                pass

            # Log changes with sensitive values redacted
            if changes["added"] or changes["removed"] or changes["modified"]:
                logger.info("📊 Configuration changes detected:")

                if changes["added"]:
                    redacted_added = _redact_sensitive_values(changes["added"])
                    logger.info(f"  ➕ Added: {redacted_added}")

                if changes["removed"]:
                    redacted_removed = _redact_sensitive_values(changes["removed"])
                    logger.info(f"  ➖ Removed: {redacted_removed}")

                if changes["modified"]:
                    for key, change in changes["modified"].items():
                        if any(
                            sensitive in key.upper() for sensitive in SENSITIVE_KEYS
                        ):
                            logger.info(f"  🔄 Modified: {key} = [REDACTED]")
                        else:
                            logger.info(
                                f"  🔄 Modified: {key} = {change['old']} → {change['new']}"
                            )

                logger.info(
                    f"  📈 Total: +{len(changes['added'])} -{len(changes['removed'])} ~{len(changes['modified'])} ={changes['unchanged_count']}"
                )
            else:
                logger.info("📊 No configuration changes detected")

            # Applied breadcrumb with subsystem inference
            try:
                subsystems: Set[str] = _infer_subsystems(changes)
                logger.info(
                    "config.reload.applied",
                    extra={
                        "event": "config.reload.applied",
                        "detail": {
                            "subsystems": sorted(list(subsystems)),
                            "old_version": old_version,
                            "new_version": new_version,
                            "prev_digest": env_before or "",
                            "new_digest": env_after or "",
                        },
                    },
                )
            except Exception:
                pass

            # Notify callbacks
            for callback in _reload_callbacks:
                try:
                    callback(old_config, new_config)
                except Exception as e:
                    logger.error(f"❌ Config reload callback failed: {e}")

            # Refresh retry manager ladders (text/vision/media) from updated env [REH]
            try:
                from .enhanced_retry import get_retry_manager

                retry_mgr = get_retry_manager()
                ladder_summary = retry_mgr.refresh_from_env()
                logger.info(
                    "config.reload.ladders",
                    extra={
                        "event": "config.reload.ladders",
                        "detail": {
                            "text": ladder_summary.get("text", [])[:3],
                            "vision": ladder_summary.get("vision", [])[:3],
                            "media": ladder_summary.get("media", [])[:1],
                        },
                    },
                )
            except Exception as e:
                logger.warning(
                    f"⚠️ Failed to refresh retry ladders on config reload: {e}"
                )

            # Restart janitor to pick up new config
            if _janitor_imported and restart_janitor is not None:
                try:
                    import asyncio

                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        _task = asyncio.create_task(restart_janitor())
                        _task.add_done_callback(
                            lambda t: (
                                logger.error(
                                    f"Janitor restart task failed: {t.exception()}"
                                )
                                if t.exception()
                                else None
                            )
                        )
                    else:
                        loop.run_until_complete(restart_janitor())
                except Exception as e:
                    logger.warning(f"⚠️ Failed to restart janitor on config reload: {e}")

            return {
                "success": True,
                "changes": changes,
                "old_version": old_version,
                "new_version": new_version,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            # Rollback breadcrumb — we keep prior config active
            try:
                logger.info(
                    "config.reload.rollback",
                    extra={
                        "event": "config.reload.rollback",
                        "detail": {"reason": str(e)[:200]},
                    },
                )
            except Exception:
                pass
            logger.error(f"❌ Configuration reload failed: {e}", exc_info=True)
            return {"success": False, "error": str(e), "version": _config_version}


def get_current_config() -> Dict[str, Any]:
    """Get the current configuration (thread-safe)."""
    with _config_lock:
        return _current_config.copy()


def get_config_version() -> str:
    """Get the current configuration version."""
    return _config_version


def get_config_for_debug() -> Dict[str, Any]:
    """Get configuration with sensitive values redacted for debugging."""
    with _config_lock:
        return _redact_sensitive_values(_current_config)


def add_reload_callback(
    callback: Callable[[Dict[str, Any], Dict[str, Any]], None],
) -> None:
    """Add a callback to be called when configuration is reloaded."""
    with _config_lock:
        _reload_callbacks.add(callback)


def remove_reload_callback(
    callback: Callable[[Dict[str, Any], Dict[str, Any]], None],
) -> None:
    """Remove a reload callback."""
    with _config_lock:
        _reload_callbacks.discard(callback)


def _sighup_handler(signum: int, frame) -> None:
    """Signal handler for SIGHUP to trigger configuration reload."""
    logger.info("📡 Received SIGHUP signal, triggering configuration reload...")
    try:
        result = reload_env()
        if result["success"]:
            logger.info("✅ SIGHUP configuration reload completed successfully")
        else:
            logger.error(
                f"❌ SIGHUP configuration reload failed: {result.get('error')}"
            )
    except Exception as e:
        logger.error(f"❌ SIGHUP handler error: {e}", exc_info=True)


async def _file_watcher_loop() -> None:
    """Main file watcher loop with proper debouncing."""
    env_paths = _candidate_env_paths()
    last_digests: Dict[Path, Optional[str]] = {p: _file_digest(p) for p in env_paths}
    last_reload_time = 0.0
    debounce_delay = 0.5  # debounce to collapse editor burst events

    status = "completed"
    try:
        while True:
            await asyncio.sleep(0.5)  # Responsive but still light

            try:
                current_time = time.time()
                for p in _candidate_env_paths():
                    try:
                        prev = last_digests.get(p)
                        dig = _file_digest(p) if p.exists() else None
                        if (
                            dig != prev
                            and current_time - last_reload_time >= debounce_delay
                        ):
                            reload_env(p)
                            last_reload_time = current_time
                        last_digests[p] = dig
                    except Exception as _e:
                        # Continue checking other paths
                        last_digests[p] = last_digests.get(p, None)

            except Exception as e:
                logger.error(f"❌ Error in file watcher: {e}")
                await asyncio.sleep(5)  # Wait longer on error

    except asyncio.CancelledError:
        status = "cancelled"
        logger.info("🛑 File watcher cancelled")
        raise
    except Exception as e:
        status = "errored"
        logger.error(f"💥 File watcher crashed: {e}", exc_info=True)
    finally:
        try:
            logger.info(
                "config.watcher.exit",
                extra={
                    "event": "config.watcher.exit",
                    "detail": {"status": status},
                },
            )
        except Exception:
            pass


def setup_config_reload() -> None:
    """Set up configuration reload system with signal handlers and file watcher."""
    global _current_config, _config_version

    # Initialize current config
    _current_config = load_config()
    _config_version = _generate_config_version(_current_config)

    # Install SIGHUP handler (Unix only)
    try:
        signal.signal(signal.SIGHUP, _sighup_handler)
        logger.info("📡 SIGHUP signal handler installed")
    except (AttributeError, OSError) as e:
        logger.warning(f"⚠️ Could not install SIGHUP handler (likely Windows): {e}")

    logger.info(
        f"🔧 Configuration reload system initialized [version: {_config_version}]"
    )


async def start_file_watcher() -> None:
    """Start file watcher task to monitor .env changes."""
    global _file_watcher_task, _watcher_lock

    if _watcher_lock is None:
        _watcher_lock = asyncio.Lock()

    async with _watcher_lock:
        if _file_watcher_task and not _file_watcher_task.done():
            logger.info(
                "config.watcher.reused",
                extra={
                    "event": "config.watcher.reused",
                    "detail": {"task_id": id(_file_watcher_task)},
                },
            )
            return

        if _file_watcher_task and _file_watcher_task.done():
            try:
                _file_watcher_task.result()
            except asyncio.CancelledError:
                logger.info("Previous file watcher task had been cancelled")
            except Exception:
                logger.info(
                    "Previous file watcher task ended with error",
                    exc_info=True,
                )

        try:
            candidates = [str(p) for p in _candidate_env_paths()]
            preferred = str(_preferred_env_path())
            logger.info(
                "config.watcher.paths",
                extra={
                    "event": "config.watcher.paths",
                    "detail": {"candidates": candidates, "preferred": preferred},
                },
            )
        except Exception:
            pass

        _file_watcher_task = asyncio.create_task(
            _file_watcher_loop(), name="config_file_watcher"
        )
        logger.info(
            "config.watcher.started",
            extra={
                "event": "config.watcher.started",
                "detail": {"task_id": id(_file_watcher_task)},
            },
        )


async def stop_file_watcher() -> None:
    """Stop the file watcher background task."""
    global _file_watcher_task, _watcher_lock

    if _watcher_lock is None:
        _watcher_lock = asyncio.Lock()

    async with _watcher_lock:
        if _file_watcher_task and not _file_watcher_task.done():
            task_id = id(_file_watcher_task)
            _file_watcher_task.cancel()
            try:
                await _file_watcher_task
            except asyncio.CancelledError:
                pass
            finally:
                _file_watcher_task = None

            logger.info(
                "config.watcher.stopped",
                extra={
                    "event": "config.watcher.stopped",
                    "detail": {"task_id": task_id},
                },
            )
        else:
            logger.debug("config.watcher.stop.noop")


def manual_reload_command() -> str:
    """
    Manually trigger a configuration reload (for Discord commands).

    Returns:
        Human-readable status message
    """
    try:
        result = reload_env()

        if result["success"]:
            changes = result["changes"]
            change_summary = []

            if changes["added"]:
                change_summary.append(f"+{len(changes['added'])} added")
            if changes["removed"]:
                change_summary.append(f"-{len(changes['removed'])} removed")
            if changes["modified"]:
                change_summary.append(f"~{len(changes['modified'])} modified")

            if change_summary:
                summary = ", ".join(change_summary)
                return f"✅ Configuration reloaded successfully!\n📊 Changes: {summary}\n🔖 Version: {result['old_version']} → {result['new_version']}"
            else:
                return f"✅ Configuration reloaded (no changes detected)\n🔖 Version: {result['new_version']}"
        else:
            return f"❌ Configuration reload failed: {result.get('error', 'Unknown error')}"

    except Exception as e:
        logger.error(f"❌ Manual reload command failed: {e}", exc_info=True)
        return f"❌ Configuration reload failed: {str(e)}"
