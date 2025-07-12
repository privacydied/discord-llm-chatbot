"""
User and server profile management with persistence.
"""
import os
import json
import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any
import threading

# Initialize locks for thread safety
user_cache_lock = threading.Lock()
server_lock = threading.Lock()

# Initialize caches
user_cache: Dict[str, Dict] = {}
server_cache: Dict[str, Dict] = {}
modified_servers = set()

# Track last save times
user_profiles_last_saved: Dict[str, float] = {}
server_profiles_last_saved: Dict[str, float] = {}

def ensure_dirs():
    """Ensure all required directories exist."""
    from .config import load_config
    config = load_config()
    
    for dir_path in [
        config["USER_PROFILE_DIR"],
        config["SERVER_PROFILE_DIR"],
        config["USER_LOGS_DIR"],
        config["TEMP_DIR"]
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)

def default_profile(user_id=None, username=None):
    """Create a new user profile with default values."""
    return {
        "discord_id": user_id if user_id else "",
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
        "custom_data": {}
    }

def ensure_profile_schema(profile: dict, user_id: Optional[str] = None, username: Optional[str] = None) -> dict:
    """Ensure a user profile has all required fields."""
    if not isinstance(profile, dict):
        profile = {}
    
    # Get user ID from profile or parameters
    profile_user_id = profile.get('discord_id', user_id)
    if not profile_user_id and user_id:
        profile_user_id = user_id
    
    # Create default profile and update with existing values
    default = default_profile(profile_user_id, username)
    
    # Update default values with any existing values
    for key in list(default.keys()):
        if key in profile and profile[key] is not None:
            default[key] = profile[key]
    
    # Ensure username is updated if provided
    if username and username != default.get('username'):
        default['username'] = username
    
    # Ensure discord_id is set correctly
    if profile_user_id and default['discord_id'] != profile_user_id:
        default['discord_id'] = profile_user_id
    
    # Ensure required lists exist
    for field in ['memories', 'history']:
        if not isinstance(default.get(field), list):
            default[field] = []
    
    # Ensure preferences is a dict
    if not isinstance(default.get('preferences'), dict):
        default['preferences'] = {}
    
    # Ensure timestamps are set
    now = datetime.now().isoformat()
    if not default.get('created_at'):
        default['created_at'] = now
    if not default.get('first_seen'):
        default['first_seen'] = default.get('created_at', now)
    default['last_updated'] = now
    
    return default

def get_profile(user_id: str, username: Optional[str] = None) -> dict:
    """Get or create a user profile, ensuring it has all required fields."""
    with user_cache_lock:
        # Check cache first
        if user_id in user_cache:
            return user_cache[user_id].copy()
        
        # Try to load from disk
        from .config import load_config
        config = load_config()
        profile_path = config["USER_PROFILE_DIR"] / f"{user_id}.json"
        
        if profile_path.exists():
            try:
                with open(profile_path, 'r', encoding='utf-8') as f:
                    profile = json.load(f)
                # Ensure the profile has all required fields
                profile = ensure_profile_schema(profile, user_id, username)
                user_cache[user_id] = profile
                return profile.copy()
            except (json.JSONDecodeError, IOError) as e:
                logging.error(f"Error loading profile for user {user_id}: {e}")
                # Fall through to create new profile
        
        # Create new profile if it doesn't exist or couldn't be loaded
        profile = default_profile(user_id, username)
        user_cache[user_id] = profile
        return profile.copy()

def save_profile(profile: dict, force: bool = False) -> bool:
    """Save a user profile to disk."""
    try:
        user_id = str(profile.get('discord_id'))
        if not user_id:
            logging.error("Cannot save profile: missing user_id")
            return False
            
        with user_cache_lock:
            # Update cache
            user_cache[user_id] = profile
            
            # Update last updated timestamp
            profile['last_updated'] = datetime.now().isoformat()
            
            # Save to disk
            from .config import load_config
            config = load_config()
            profile_path = config["USER_PROFILE_DIR"] / f"{user_id}.json"
            
            # Create backup of existing file if it exists
            if profile_path.exists():
                backup_path = profile_path.with_suffix('.json.bak')
                try:
                    shutil.copy2(profile_path, backup_path)
                except IOError as e:
                    logging.error(f"Failed to create backup for {profile_path}: {e}"
                    
            # Save the profile
            try:
                with open(profile_path, 'w', encoding='utf-8') as f:
                    json.dump(profile, f, indent=2, ensure_ascii=False)
                return True
            except IOError as e:
                logging.error(f"Failed to save profile for user {user_id}: {e}")
                # Try to restore from backup if save failed
                if 'backup_path' in locals() and backup_path.exists():
                    try:
                        shutil.copy2(backup_path, profile_path)
                        logging.info(f"Restored profile from backup for user {user_id}")
                    except IOError as restore_error:
                        logging.error(f"Failed to restore profile from backup: {restore_error}")
                return False
    except Exception as e:
        logging.error(f"Unexpected error in save_profile: {e}", exc_info=True)
        return False

def default_server_profile(guild_id: Optional[str] = None) -> dict:
    """Create a new server profile with default values."""
    return {
        "guild_id": guild_id if guild_id else "",
        "memories": [],
        "history": [],
        "preferences": {},
        "context_notes": "",
        "total_messages": 0,
        "last_updated": datetime.now().isoformat(),
        "created_at": datetime.now().isoformat(),
        "custom_data": {}
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
    if guild_id and default['guild_id'] != guild_id:
        default['guild_id'] = guild_id
    
    # Ensure required lists exist
    for field in ['memories', 'history']:
        if not isinstance(default.get(field), list):
            default[field] = []
    
    # Ensure preferences is a dict
    if not isinstance(default.get('preferences'), dict):
        default['preferences'] = {}
    
    # Ensure timestamps are set
    now = datetime.now().isoformat()
    if not default.get('created_at'):
        default['created_at'] = now
    default['last_updated'] = now
    
    return default

def get_server_profile(guild_id: str, force_reload: bool = False) -> dict:
    """Get or create a server profile."""
    with server_lock:
        # Check cache first
        if guild_id in server_cache and not force_reload:
            return server_cache[guild_id].copy()
        
        # Try to load from disk
        from .config import load_config
        config = load_config()
        profile_path = config["SERVER_PROFILE_DIR"] / f"{guild_id}.json"
        
        if profile_path.exists():
            try:
                with open(profile_path, 'r', encoding='utf-8') as f:
                    profile = json.load(f)
                # Ensure the profile has all required fields
                profile = ensure_server_profile_schema(profile, guild_id)
                server_cache[guild_id] = profile
                return profile.copy()
            except (json.JSONDecodeError, IOError) as e:
                logging.error(f"Error loading server profile for guild {guild_id}: {e}")
                # Fall through to create new profile
        
        # Create new profile if it doesn't exist or couldn't be loaded
        profile = default_server_profile(guild_id)
        server_cache[guild_id] = profile
        return profile.copy()

def save_server_profile(guild_id: str, force: bool = False) -> bool:
    """Save a server profile to disk."""
    try:
        with server_lock:
            if guild_id not in server_cache:
                logging.error(f"Cannot save server profile: guild {guild_id} not in cache")
                return False
                
            profile = server_cache[guild_id]
            profile['last_updated'] = datetime.now().isoformat()
            
            from .config import load_config
            config = load_config()
            profile_path = config["SERVER_PROFILE_DIR"] / f"{guild_id}.json"
            
            # Create directory if it doesn't exist
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create backup of existing file if it exists
            if profile_path.exists():
                backup_path = profile_path.with_suffix('.json.bak')
                try:
                    shutil.copy2(profile_path, backup_path)
                except IOError as e:
                    logging.error(f"Failed to create backup for {profile_path}: {e}")
            
            # Save the profile
            try:
                with open(profile_path, 'w', encoding='utf-8') as f:
                    json.dump(profile, f, indent=2, ensure_ascii=False)
                return True
            except IOError as e:
                logging.error(f"Failed to save server profile for guild {guild_id}: {e}")
                # Try to restore from backup if save failed
                if 'backup_path' in locals() and backup_path.exists():
                    try:
                        shutil.copy2(backup_path, profile_path)
                        logging.info(f"Restored server profile from backup for guild {guild_id}")
                    except IOError as restore_error:
                        logging.error(f"Failed to restore server profile from backup: {restore_error}")
                return False
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

# Initialize required directories when module is imported
ensure_dirs()
