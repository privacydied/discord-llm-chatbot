"""
User and server profile management with persistence.

Uses atomic file writes and corruption recovery for safe data storage.
[REH][SFT] Atomic persistence with backup/recovery
"""

import json
import os
import logging
import shutil
from datetime import datetime
from typing import Dict, Optional
import threading
from pathlib import Path

from bot.memory.persistence import (
    atomic_save_json,
    load_json_with_recovery,
    validate_profile_integrity,
)

# Initialize locks for thread safety
user_cache_lock = threading.Lock()
server_lock = threading.Lock()

# Initialize caches
user_cache: Dict[str, Dict] = {}
server_cache: Dict[str, Dict] = {}
# Legacy/test aliases
user_profiles = user_cache
server_profiles = server_cache
modified_servers = set()

# Track last save times
user_profiles_last_saved: Dict[str, float] = {}
server_profiles_last_saved: Dict[str, float] = {}


def _safe_backup_file(src: Path, dst: Path) -> None:
    """Create a safe backup of src at dst.

    Handles the case where an existing dst file is read-only by removing it
    first. Falls back to chmod before unlink on permission errors. Errors are
    logged but not raised to avoid blocking primary save operation.
    """
    try:
        if dst.exists():
            try:
                # Prefer unlink to avoid overwriting a read-only file
                dst.unlink()
            except PermissionError:
                try:
                    # Make writable then remove
                    os.chmod(dst, 0o600)
                    dst.unlink()
                except Exception as e:
                    logging.error(f"Failed to remove existing backup {dst}: {e}")
                    # If we cannot remove it, attempting to overwrite would likely fail
                    return
        shutil.copy2(src, dst)
    except Exception as e:
        logging.error(f"Failed to create backup for {src}: {e}")


def ensure_dirs():
    """Ensure all required directories exist."""
    from bot.config import load_config

    config = load_config()

    for dir_path in [
        config["USER_PROFILE_DIR"],
        config["SERVER_PROFILE_DIR"],
        config["USER_LOGS_DIR"],
        config["TEMP_DIR"],
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)


def default_profile(user_id=None, username=None):
    """Create a new user profile with default values."""
    return {
        "discord_id": user_id if user_id else "",
        "user_id": user_id if user_id else "",  # alias for legacy tests
        "username": username if username else "",
        "memories": [],
        "history": [],
        "preferences": {},
        "context_notes": "",
        "total_messages": 0,
        "last_updated": datetime.now().isoformat(),
        "created_at": datetime.now().isoformat(),
        "first_seen": datetime.now().isoformat(),
        "is_bot": False,
        "tone": "neutral",
        "last_seen": None,
        "custom_data": {},
    }


def ensure_profile_schema(
    profile: dict, user_id: Optional[str] = None, username: Optional[str] = None
) -> dict:
    """Ensure a user profile has all required fields."""
    if not isinstance(profile, dict):
        profile = {}

    # Get user ID from profile or parameters
    profile_user_id = profile.get("discord_id") or profile.get("user_id") or user_id
    if not profile_user_id and user_id:
        profile_user_id = user_id

    # Create default profile and update with existing values
    default = default_profile(profile_user_id, username)

    # Update default values with any existing values
    for key in list(default.keys()):
        if key in profile and profile[key] is not None:
            default[key] = profile[key]

    # Ensure username is updated if provided
    if username and username != default.get("username"):
        default["username"] = username

    # Ensure discord_id is set correctly
    if profile_user_id and default.get("discord_id") != profile_user_id:
        default["discord_id"] = profile_user_id
    # Keep alias in sync for legacy tests
    if profile_user_id and default.get("user_id") != profile_user_id:
        default["user_id"] = profile_user_id

    # Ensure required lists exist
    for field in ["memories", "history"]:
        if not isinstance(default.get(field), list):
            default[field] = []

    # Ensure preferences is a dict
    if not isinstance(default.get("preferences"), dict):
        default["preferences"] = {}

    # Ensure timestamps are set
    now = datetime.now().isoformat()
    if not default.get("created_at"):
        default["created_at"] = now
    if not default.get("first_seen"):
        default["first_seen"] = default.get("created_at", now)
    default["last_updated"] = now

    return default


def get_profile(user_id: str, username: Optional[str] = None) -> dict:
    """Get or create a user profile, ensuring it has all required fields.

    [REH] Corruption recovery via load_json_with_recovery
    [SFT] Validation before loading into cache
    """
    with user_cache_lock:
        # Check cache first
        if user_id in user_cache:
            return user_cache[user_id].copy()

        # Try to load from disk with corruption recovery
        from bot.config import load_config

        config = load_config()
        profile_path = config["USER_PROFILE_DIR"] / f"{user_id}.json"

        # Use persistence layer with recovery
        default = default_profile(user_id, username)
        profile = load_json_with_recovery(
            profile_path,
            default_data=None,  # We handle default separately
            attempt_recovery=True,
        )

        if profile is not None:
            # Validate loaded profile
            is_valid, error_msg = validate_profile_integrity(profile)
            if is_valid:
                # Ensure the profile has all required fields
                profile = ensure_profile_schema(profile, user_id, username)
                user_cache[user_id] = profile
                return profile.copy()
            else:
                logging.warning(f"Corrupted profile for user {user_id}: {error_msg}")
                # Fall through to create new profile

        # Create new profile if it doesn't exist or couldn't be loaded
        profile = default_profile(user_id, username)
        user_cache[user_id] = profile
        return profile.copy()


def save_profile(profile: dict, force: bool = False, caller_id: Optional[str] = None) -> bool:
    """Save a user profile to disk using atomic writes with corruption recovery.

    [REH] Atomic writes ensure readers see only complete writes
    [SFT] Automatic backup and recovery on failure

    Args:
        profile: The profile dict to save.
        force: If True, skip some safety checks (still validates integrity).
        caller_id: Optional Discord user ID of the caller. If provided,
                   validates that the profile belongs to the caller to
                   prevent cross-user profile overwrites.
    """
    try:
        user_id = str(profile.get("discord_id") or profile.get("user_id") or "")
        if not user_id:
            logging.error("Cannot save profile: missing user_id")
            return False

        # Cross-user overwrite prevention [SFT]
        if caller_id is not None and str(caller_id) != user_id:
            logging.error(
                f"Profile overwrite blocked: caller {caller_id} attempted to save "
                f"profile for user {user_id}"
            )
            return False

        with user_cache_lock:
            # Update cache
            user_cache[user_id] = profile

            # Update last updated timestamp
            profile["last_updated"] = datetime.now().isoformat()

            # Enforce user memory limit (prefer fresh env to avoid cached config)
            try:
                env_val = os.getenv("MAX_USER_MEMORY")
                if env_val is not None:
                    max_memories = int(str(env_val).split("#")[0].strip() or "20")
                else:
                    from bot.config import load_config

                    max_memories = int(load_config().get("MAX_USER_MEMORY", 20))
            except Exception:
                max_memories = 20

            if (
                isinstance(profile.get("memories"), list)
                and len(profile["memories"]) > max_memories
            ):
                profile["memories"] = profile["memories"][-max_memories:]

            # Save to disk using atomic persistence
            from bot.config import load_config

            config = load_config()
            profile_path = config["USER_PROFILE_DIR"] / f"{user_id}.json"

            # Validate profile before saving
            is_valid, error_msg = validate_profile_integrity(profile)
            if not is_valid:
                logging.error(f"Profile validation failed for user {user_id}: {error_msg}")
                return False

            # Atomic save with backup
            if not atomic_save_json(
                profile_path,
                profile,
                create_backup=True,
                validate_before_write=True,
                use_lock=True,
                indent=2,
            ):
                logging.error(f"Failed to save profile for user {user_id}")
                return False

            return True

    except Exception as e:
        logging.error(f"Unexpected error in save_profile: {e}", exc_info=True)
        return False


def default_server_profile(guild_id: Optional[str] = None) -> dict:
    """Create a new server profile with default values."""
    return {
        "guild_id": guild_id if guild_id else "",
        "server_id": guild_id if guild_id else "",  # alias for legacy tests
        "memories": [],
        "history": [],
        "preferences": {},
        "context_notes": "",
        "total_messages": 0,
        "last_updated": datetime.now().isoformat(),
        "created_at": datetime.now().isoformat(),
        "custom_data": {},
    }


def ensure_server_profile_schema(profile: dict, guild_id: Optional[str] = None) -> dict:
    """Ensure a server profile has all required fields."""
    if not isinstance(profile, dict):
        profile = {}

    # Create default profile and update with existing values
    default = default_server_profile(guild_id)

    # Update default values with any existing values
    for key in list(default.keys()):
        if key in profile and profile[key] is not None:
            default[key] = profile[key]

    # Ensure guild_id is set correctly
    if guild_id and default.get("guild_id") != guild_id:
        default["guild_id"] = guild_id
    # Keep alias in sync for legacy tests
    if guild_id and default.get("server_id") != guild_id:
        default["server_id"] = guild_id

    # Ensure required lists exist
    for field in ["memories", "history"]:
        if not isinstance(default.get(field), list):
            default[field] = []

    # Ensure preferences is a dict
    if not isinstance(default.get("preferences"), dict):
        default["preferences"] = {}

    # Ensure timestamps are set
    now = datetime.now().isoformat()
    if not default.get("created_at"):
        default["created_at"] = now
    default["last_updated"] = now

    return default


def get_server_profile(guild_id: str, force_reload: bool = False) -> dict:
    """Get or create a server profile.

    [REH] Corruption recovery via load_json_with_recovery
    [SFT] Validation before loading into cache
    """
    with server_lock:
        # Check cache first
        if guild_id in server_cache and not force_reload:
            return server_cache[guild_id].copy()  # [BUGFIX] return copy like get_profile

        # Try to load from disk with corruption recovery
        from bot.config import load_config

        config = load_config()
        profile_path = config["SERVER_PROFILE_DIR"] / f"{guild_id}.json"

        # Use persistence layer with recovery
        profile = load_json_with_recovery(
            profile_path,
            default_data=None,  # We handle default separately
            attempt_recovery=True,
        )

        if profile is not None:
            # Validate loaded profile
            is_valid, error_msg = validate_profile_integrity(profile)
            if is_valid:
                # Ensure the profile has all required fields
                profile = ensure_server_profile_schema(profile, guild_id)
                server_cache[guild_id] = profile
                return profile.copy()  # [BUGFIX] return copy
            else:
                logging.warning(f"Corrupted server profile for guild {guild_id}: {error_msg}")
                # Fall through to create new profile

        # Create new profile if it doesn't exist or couldn't be loaded
        profile = default_server_profile(guild_id)
        server_cache[guild_id] = profile
        return profile.copy()  # [BUGFIX] return copy


def save_server_profile(guild_id, force: bool = False) -> bool:
    """Save a server profile to disk using atomic writes with corruption recovery.

    Accepts either a guild_id/server_id string, or a full profile dict.

    [REH] Atomic writes ensure readers see only complete writes
    [SFT] Automatic backup and recovery on failure
    """
    try:
        # Support being passed a full profile dict (legacy tests)
        if isinstance(guild_id, dict):
            profile = guild_id
            gid = str(profile.get("guild_id") or profile.get("server_id") or "")
            if not gid:
                logging.error(
                    "Cannot save server profile: missing guild_id/server_id in profile"
                )
                return False
            # Ensure schema and cache are updated
            profile = ensure_server_profile_schema(profile, gid)
            with server_lock:
                server_cache[gid] = profile
        else:
            gid = str(guild_id)
            with server_lock:
                profile = server_cache.get(gid)

            if profile is None:
                # Instead of failing, lazily initialize from disk or defaults.
                # This avoids recurring "not in cache" errors during autosave
                # when a profile was evicted or not loaded on startup.
                logging.warning(
                    f"Server profile for guild {gid} not in cache; loading/creating before save"
                )
                # get_server_profile will either load from disk or create default
                profile = get_server_profile(gid, force_reload=True)
                if profile is None:
                    logging.error(f"Failed to load server profile for guild {gid}")
                    return False
                with server_lock:
                    server_cache[gid] = profile

        profile["last_updated"] = datetime.now().isoformat()

        from bot.config import load_config

        config = load_config()
        profile_path = config["SERVER_PROFILE_DIR"] / f"{gid}.json"

        # Enforce server memory limit (prefer fresh env to avoid cached config)
        try:
            env_val = os.getenv("MAX_SERVER_MEMORY")
            if env_val is not None:
                max_memories = int(str(env_val).split("#")[0].strip() or "100")
            else:
                from bot.config import load_config

                max_memories = int(load_config().get("MAX_SERVER_MEMORY", 100))
        except Exception:
            max_memories = 100

        if (
            isinstance(profile.get("memories"), list)
            and len(profile["memories"]) > max_memories
        ):
            profile["memories"] = profile["memories"][-max_memories:]

        # Create directory if it doesn't exist
        profile_path.parent.mkdir(parents=True, exist_ok=True)

        # Validate profile before saving
        is_valid, error_msg = validate_profile_integrity(profile)
        if not is_valid:
            logging.error(f"Server profile validation failed for guild {gid}: {error_msg}")
            return False

        # Atomic save with backup
        if not atomic_save_json(
            profile_path,
            profile,
            create_backup=True,
            validate_before_write=True,
            use_lock=True,
            indent=2,
        ):
            logging.error(f"Failed to save server profile for guild {gid}")
            return False

        return True

    except Exception as e:
        logging.error(f"Unexpected error in save_server_profile: {e}", exc_info=True)
        return False


def flush_all_profiles() -> bool:
    """Save all modified profiles to disk."""
    success = True

    # Save all modified user profiles
    with user_cache_lock:
        user_ids = list(user_cache.keys())

    for user_id in user_ids:
        try:
            profile = get_profile(user_id)
            if not save_profile(profile):
                success = False
                logging.error(f"Failed to save profile for user {user_id}")
        except Exception as e:
            success = False
            logging.error(f"Error saving profile for user {user_id}: {e}")

    # Save all modified server profiles
    with server_lock:
        server_ids = list(server_cache.keys())

    for guild_id in server_ids:
        try:
            if not save_server_profile(guild_id):
                success = False
                logging.error(f"Failed to save server profile for guild {guild_id}")
        except Exception as e:
            success = False
            logging.error(f"Error saving server profile for guild {guild_id}: {e}")

    return success


# Global profile stores that main.py expects to import
user_profiles = user_cache
server_profiles = server_cache


def load_all_profiles():
    """Load all user profiles from disk into memory cache using corruption recovery.

    Returns:
        tuple: A tuple containing (user_profiles, server_profiles)

    [REH] Corruption recovery on per-profile basis
    """
    from bot.config import load_config

    config = load_config()

    # Load user profiles
    profile_dir = config["USER_PROFILE_DIR"]
    if not profile_dir.exists():
        logging.info("User profile directory does not exist, creating it")
        profile_dir.mkdir(parents=True, exist_ok=True)
    else:
        loaded_count = 0
        corrupted_count = 0
        for profile_file in profile_dir.glob("*.json"):
            user_id = profile_file.stem

            # Skip backup and temp files
            if profile_file.suffix == ".bak" or profile_file.name.startswith("."):
                continue

            profile = load_json_with_recovery(
                profile_file,
                default_data=None,
                attempt_recovery=True,
            )

            if profile is not None:
                # Validate loaded profile
                is_valid, error_msg = validate_profile_integrity(profile)
                if is_valid:
                    profile = ensure_profile_schema(profile, user_id)
                    with user_cache_lock:
                        user_cache[user_id] = profile
                    loaded_count += 1
                else:
                    logging.warning(f"Skipping corrupted profile {profile_file}: {error_msg}")
                    corrupted_count += 1
            else:
                logging.warning(f"Could not load or recover profile {profile_file}")
                corrupted_count += 1

        if corrupted_count > 0:
            logging.warning(f"Encountered {corrupted_count} corrupted profiles during load")
        logging.info(f"Loaded {loaded_count} user profiles from disk")

    # Load server profiles
    load_all_server_profiles()

    # Return both caches
    return user_cache, server_cache


def save_all_profiles() -> bool:
    """Save all user profiles from memory cache to disk."""
    success = True

    with user_cache_lock:
        user_ids = list(user_cache.keys())

    for user_id in user_ids:
        try:
            profile = user_cache.get(user_id)
            if profile and not save_profile(profile, force=True):
                success = False
                logging.error(f"Failed to save profile for user {user_id}")
        except Exception as e:
            success = False
            logging.error(f"Error saving profile for user {user_id}: {e}")

    return success


def load_all_server_profiles() -> None:
    """Load all server profiles from disk into memory cache."""
    from bot.config import load_config

    config = load_config()

    profile_dir = config["SERVER_PROFILE_DIR"]
    if not profile_dir.exists():
        logging.info("Server profile directory does not exist, creating it")
        profile_dir.mkdir(parents=True, exist_ok=True)
        return

    loaded_count = 0
    for profile_file in profile_dir.glob("*.json"):
        try:
            guild_id = profile_file.stem
            with open(profile_file, "r", encoding="utf-8") as f:
                profile = json.load(f)

            # Ensure profile has all required fields
            profile = ensure_server_profile_schema(profile, guild_id)

            with server_lock:
                server_cache[guild_id] = profile

            loaded_count += 1

        except Exception as e:
            logging.error(f"Error loading server profile {profile_file}: {e}")

    logging.info(f"Loaded {loaded_count} server profiles from disk")


def save_all_server_profiles() -> bool:
    """Save all server profiles from memory cache to disk."""
    success = True

    with server_lock:
        guild_ids = list(server_cache.keys())

    for guild_id in guild_ids:
        try:
            if not save_server_profile(guild_id, force=True):
                success = False
                logging.error(f"Failed to save server profile for guild {guild_id}")
        except Exception as e:
            success = False
            logging.error(f"Error saving server profile for guild {guild_id}: {e}")

    return success


def add_memory(
    user_id: str,
    memory_text: str,
    guild_id: Optional[str] = None,
    username: Optional[str] = None,
) -> bool:
    """
    Add a memory for a user and optionally to a server.

    Args:
        user_id: Discord user ID
        memory_text: The memory text to add
        guild_id: Optional server ID to also add the memory to
        username: Optional username for server memory context

    Returns:
        bool: True if the memory was added successfully, False otherwise
    """
    if (
        not user_id
        or not memory_text
        or not isinstance(memory_text, str)
        or not memory_text.strip()
    ):
        logging.warning(
            f"Invalid memory data - user_id: {user_id}, memory_text: {type(memory_text)}"
        )
        return False

    memory_text = memory_text.strip()
    memory_lower = memory_text.lower()
    success = True

    # Add to user memories
    try:
        profile = get_profile(user_id)
        if not profile:
            logging.error(f"Failed to load profile for user {user_id}")
            return False

        # Initialize memories list if it doesn't exist
        if "memories" not in profile:
            profile["memories"] = []

        # Check for duplicates (case-insensitive)
        if not any(mem.lower() == memory_lower for mem in profile["memories"]):
            profile["memories"].append(memory_text)

            # Enforce memory limit
            from bot.config import load_config

            config = load_config()
            max_memories = config.get("MAX_MEMORIES", 100)
            if len(profile["memories"]) > max_memories:
                profile["memories"] = profile["memories"][-max_memories:]

            profile["last_updated"] = datetime.now().isoformat()

            # Save the updated profile
            if not save_profile(profile):
                logging.error(
                    f"Failed to save profile after adding memory for user {user_id}"
                )
                success = False

    except Exception as e:
        logging.error(f"Error adding user memory: {e}", exc_info=True)
        success = False

    # Optionally add to server memories if guild_id is provided
    if (
        guild_id
        and username
        and isinstance(guild_id, str)
        and isinstance(username, str)
    ):
        try:
            server_profile = get_server_profile(guild_id)
            if not server_profile:
                logging.error(f"Failed to load server profile for guild {guild_id}")
                return success and False

            # Format the memory with timestamp and username
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            server_memory = f"[{timestamp}] {username}: {memory_text}"

            # Check for duplicates in server memories (exact match) [SFT]
            if server_memory not in server_profile["memories"]:
                server_profile["memories"].append(server_memory)

                # Enforce server memory limit
                from bot.config import load_config

                config = load_config()
                max_memories = config.get("MAX_SERVER_MEMORY", 100)
                if len(server_profile["memories"]) > max_memories:
                    server_profile["memories"] = server_profile["memories"][
                        -max_memories:
                    ]

                server_profile["last_updated"] = datetime.now().isoformat()

                # Save the updated server profile
                if not save_server_profile(guild_id):
                    logging.error(f"Failed to save server profile for guild {guild_id}")
                    success = False

        except Exception as e:
            logging.error(f"Error adding server memory: {e}", exc_info=True)
            success = False

    return success


# Directories are ensured by callers/tests; avoid early config caching here.
