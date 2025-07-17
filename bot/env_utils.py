"""Environment variable resolution utilities."""

import logging
import os
from pathlib import Path
from typing import Optional, Union, Dict, Any

logger = logging.getLogger(__name__)

# Global config singleton to store resolved paths
_resolved_paths: Dict[str, Path] = {}

def resolve_env(name_new: str, name_legacy: str, default: Optional[str] = None) -> Optional[str]:
    """
    Resolve environment variables with preference for new names.
    
    Args:
        name_new: New environment variable name
        name_legacy: Legacy environment variable name
        default: Default value if neither variable is set
        
    Returns:
        Resolved value or default if neither is set
    """
    new_value = os.environ.get(name_new)
    legacy_value = os.environ.get(name_legacy)
    
    # Case 1: Both variables set with different values
    if new_value is not None and legacy_value is not None and new_value != legacy_value:
        logger.info(f"Using {name_new}='{new_value}', ignoring legacy {name_legacy}='{legacy_value}'",
                  extra={'subsys': 'config', 'event': 'env.resolve.both_set', 
                         'new_var': name_new, 'legacy_var': name_legacy})
        return new_value
    
    # Case 2: Only new variable set
    if new_value is not None:
        return new_value
    
    # Case 3: Only legacy variable set
    if legacy_value is not None:
        logger.warning(f"Legacy environment variable {name_legacy} is deprecated. Please use {name_new} instead.",
                     extra={'subsys': 'config', 'event': 'env.resolve.legacy_only', 
                            'legacy_var': name_legacy, 'new_var': name_new})
        return legacy_value
    
    # Case 4: Neither variable set
    return default

def resolve_path(name_new: str, name_legacy: str, default: Optional[str] = None) -> Optional[Path]:
    """
    Resolve environment variables to a Path object with preference for new names.
    Stores the result in a singleton to ensure consistent paths across the application.
    
    Args:
        name_new: New environment variable name
        name_legacy: Legacy environment variable name
        default: Default path if neither variable is set
        
    Returns:
        Resolved Path or None if neither is set and no default provided
    """
    # Check if we've already resolved this path
    cache_key = f"{name_new}|{name_legacy}"
    if cache_key in _resolved_paths:
        return _resolved_paths[cache_key]
    
    # Resolve the environment variable
    value = resolve_env(name_new, name_legacy, default)
    
    # Convert to Path if we have a value
    if value is not None:
        path = Path(value)
        _resolved_paths[cache_key] = path
        return path
    
    return None

def get_config_singleton() -> Dict[str, Any]:
    """
    Get a singleton dictionary of all resolved configuration values.
    This ensures all modules see the same configuration.
    
    Returns:
        Dictionary of resolved configuration values
    """
    # This could be expanded to include other configuration sources
    return {
        "paths": {k: str(v) for k, v in _resolved_paths.items()}
    }
