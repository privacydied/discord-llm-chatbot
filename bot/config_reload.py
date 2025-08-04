"""
Dynamic configuration reloading system with hot-reload support.
Supports SIGHUP signal handling, file watching, and manual reload commands.
"""
import os
import signal
import hashlib
import asyncio
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, Set, Callable
from datetime import datetime
from dotenv import load_dotenv

from .config import load_config
from .util.logging import get_logger

logger = get_logger(__name__)

# Global state for configuration management
_current_config: Dict[str, Any] = {}
_config_version: str = ""
_config_lock = threading.RLock()
_reload_callbacks: Set[Callable[[Dict[str, Any], Dict[str, Any]], None]] = set()
_file_watcher_task: Optional[asyncio.Task] = None
_last_reload_time: float = 0
_env_file_path = Path.cwd() / '.env'

# Sensitive keys that should be redacted in logs
SENSITIVE_KEYS = {
    'DISCORD_TOKEN', 'OPENAI_API_KEY', 'WHISPER_API_KEY', 
    'API_KEY', 'TOKEN', 'SECRET', 'PASSWORD', 'PASS'
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
                redacted[key] = f"***{str(value)[-4:]}" if len(str(value)) > 4 else "***"
            else:
                redacted[key] = value
        else:
            redacted[key] = value
    return redacted


def _compare_configs(old_config: Dict[str, Any], new_config: Dict[str, Any]) -> Dict[str, Any]:
    """Compare two configurations and return the differences."""
    changes = {
        'added': {},
        'removed': {},
        'modified': {},
        'unchanged_count': 0
    }
    
    old_keys = set(old_config.keys())
    new_keys = set(new_config.keys())
    
    # Added keys
    for key in new_keys - old_keys:
        changes['added'][key] = new_config[key]
    
    # Removed keys
    for key in old_keys - new_keys:
        changes['removed'][key] = old_config[key]
    
    # Modified keys
    for key in old_keys & new_keys:
        if old_config[key] != new_config[key]:
            changes['modified'][key] = {
                'old': old_config[key],
                'new': new_config[key]
            }
        else:
            changes['unchanged_count'] += 1
    
    return changes


def reload_env() -> Dict[str, Any]:
    """
    Reload environment variables from .env file and update configuration.
    
    Returns:
        Dictionary with reload results including success status, changes, and version
    """
    global _current_config, _config_version, _last_reload_time
    
    with _config_lock:
        try:
            logger.info("🔄 Starting configuration reload...")
            
            # Store old config for comparison
            old_config = _current_config.copy()
            old_version = _config_version
            
            # Reload .env file
            if _env_file_path.exists():
                load_dotenv(dotenv_path=_env_file_path, override=True)
                logger.debug(f"📁 Reloaded .env from {_env_file_path}")
            else:
                logger.warning(f"⚠️ .env file not found at {_env_file_path}")
            
            # Load new configuration
            new_config = load_config()
            
            # Validate critical required variables
            required_vars = ['DISCORD_TOKEN']
            missing_vars = [var for var in required_vars if not new_config.get(var)]
            if missing_vars:
                logger.error(f"❌ Critical variables missing after reload: {missing_vars}")
                return {
                    'success': False,
                    'error': f"Missing required variables: {missing_vars}",
                    'version': old_version
                }
            
            # Generate new version
            new_version = _generate_config_version(new_config)
            
            # Compare configurations
            changes = _compare_configs(old_config, new_config)
            
            # Update global state
            _current_config = new_config
            _config_version = new_version
            _last_reload_time = time.time()
            
            # Log changes with sensitive values redacted
            if changes['added'] or changes['removed'] or changes['modified']:
                logger.info(f"📊 Configuration changes detected:")
                
                if changes['added']:
                    redacted_added = _redact_sensitive_values(changes['added'])
                    logger.info(f"  ➕ Added: {redacted_added}")
                
                if changes['removed']:
                    redacted_removed = _redact_sensitive_values(changes['removed'])
                    logger.info(f"  ➖ Removed: {redacted_removed}")
                
                if changes['modified']:
                    for key, change in changes['modified'].items():
                        if any(sensitive in key.upper() for sensitive in SENSITIVE_KEYS):
                            logger.info(f"  🔄 Modified: {key} = [REDACTED]")
                        else:
                            logger.info(f"  🔄 Modified: {key} = {change['old']} → {change['new']}")
                
                logger.info(f"  📈 Total: +{len(changes['added'])} -{len(changes['removed'])} ~{len(changes['modified'])} ={changes['unchanged_count']}")
            else:
                logger.info("📊 No configuration changes detected")
            
            logger.info(f"✅ Configuration reload complete [version: {old_version} → {new_version}]")
            
            # Notify callbacks
            for callback in _reload_callbacks:
                try:
                    callback(old_config, new_config)
                except Exception as e:
                    logger.error(f"❌ Config reload callback failed: {e}")
            
            return {
                'success': True,
                'changes': changes,
                'old_version': old_version,
                'new_version': new_version,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Configuration reload failed: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'version': _config_version
            }


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


def add_reload_callback(callback: Callable[[Dict[str, Any], Dict[str, Any]], None]) -> None:
    """Add a callback to be called when configuration is reloaded."""
    _reload_callbacks.add(callback)


def remove_reload_callback(callback: Callable[[Dict[str, Any], Dict[str, Any]], None]) -> None:
    """Remove a reload callback."""
    _reload_callbacks.discard(callback)


def _sighup_handler(signum: int, frame) -> None:
    """Signal handler for SIGHUP to trigger configuration reload."""
    logger.info(f"📡 Received SIGHUP signal, triggering configuration reload...")
    try:
        result = reload_env()
        if result['success']:
            logger.info("✅ SIGHUP configuration reload completed successfully")
        else:
            logger.error(f"❌ SIGHUP configuration reload failed: {result.get('error')}")
    except Exception as e:
        logger.error(f"❌ SIGHUP handler error: {e}", exc_info=True)


async def _file_watcher_loop() -> None:
    """Main file watcher loop with proper debouncing."""
    env_file = Path(".env")
    last_mtime = None
    last_reload_time = 0
    debounce_delay = 2.0  # Minimum seconds between reloads
    
    try:
        while True:
            await asyncio.sleep(1)  # Check every second
            
            try:
                if env_file.exists():
                    current_mtime = env_file.stat().st_mtime
                    current_time = time.time()
                    
                    # Check if file changed and enough time passed since last reload
                    if (last_mtime is not None and 
                        current_mtime != last_mtime and 
                        current_time - last_reload_time >= debounce_delay):
                        
                        logger.info("📁 .env file changed, reloading configuration...")
                        reload_env()
                        last_reload_time = current_time
                    
                    last_mtime = current_mtime
                    
            except Exception as e:
                logger.error(f"❌ Error in file watcher: {e}")
                await asyncio.sleep(5)  # Wait longer on error
                
    except asyncio.CancelledError:
        logger.info("🛑 File watcher cancelled")
        raise
    except Exception as e:
        logger.error(f"💥 File watcher crashed: {e}", exc_info=True)


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
    
    logger.info(f"🔧 Configuration reload system initialized [version: {_config_version}]")


async def start_file_watcher() -> None:
    """Start file watcher task to monitor .env changes."""
    global _file_watcher_task
    
    if _file_watcher_task and not _file_watcher_task.done():
        logger.warning("⚠️ File watcher already running")
        return
    
    _file_watcher_task = asyncio.create_task(_file_watcher_loop())
    logger.info("👁️ Configuration file watcher started")


async def stop_file_watcher() -> None:
    """Stop the file watcher background task."""
    global _file_watcher_task
    
    if _file_watcher_task and not _file_watcher_task.done():
        _file_watcher_task.cancel()
        try:
            await _file_watcher_task
        except asyncio.CancelledError:
            pass
        logger.info("👁️ File watcher task stopped")


def manual_reload_command() -> str:
    """
    Manually trigger a configuration reload (for Discord commands).
    
    Returns:
        Human-readable status message
    """
    try:
        result = reload_env()
        
        if result['success']:
            changes = result['changes']
            change_summary = []
            
            if changes['added']:
                change_summary.append(f"+{len(changes['added'])} added")
            if changes['removed']:
                change_summary.append(f"-{len(changes['removed'])} removed")
            if changes['modified']:
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
