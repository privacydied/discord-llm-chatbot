"""
Tokenizer registry for TTS tokenizer discovery and management.

This module provides a singleton registry for tokenizer availability
to ensure consistent state across imports and prevent reset issues.
"""

import logging
import os
from typing import Dict, Set, Optional, Any
import importlib.util
import shutil
import subprocess

from .tts.errors import MissingTokeniserError

logger = logging.getLogger(__name__)

# Tokenizer types
TOKENIZER_TYPES = {
    "espeak": "espeak",
    "espeak-ng": "espeak-ng",
    "phonemizer": "phonemizer",
    "g2p_en": "g2p_en",
    "misaki": "misaki",
    "grapheme": "grapheme"
}

# Language-to-tokenizer mapping
TOKENISER_MAP = {
    "en": ["espeak", "espeak-ng", "phonemizer", "g2p_en"],
    "ja": ["misaki"],
    "zh": ["misaki"],
    # Default for other languages
    "*": ["phonemizer", "espeak", "espeak-ng"]
}

# Default to grapheme for all languages
DEFAULT_TOKENIZER = "grapheme"

# Singleton registry pattern
class TokenizerRegistry:
    """Singleton registry for tokenizer availability."""
    
    _instance = None
    
    @classmethod
    def get_instance(cls) -> 'TokenizerRegistry':
        """Get the singleton instance of TokenizerRegistry."""
        if cls._instance is None:
            cls._instance = TokenizerRegistry()
        return cls._instance
    
    def __init__(self):
        """Initialize the registry. Should only be called once."""
        if TokenizerRegistry._instance is not None:
            logger.warning("TokenizerRegistry initialized more than once", 
                          extra={'subsys': 'tts', 'event': 'registry.duplicate_init'})
        
        # Initialize with empty set
        self._available_tokenizers: Set[str] = set()
        self._initialized = False
        self._size_at_init = 0
    
    def discover_tokenizers(self, force: bool = False) -> Dict[str, bool]:
        """
        Discover available tokenizers in the current environment.
        
        Args:
            force: Force rediscovery even if already initialized
            
        Returns:
            Dictionary mapping tokenizer names to availability status
        """
        # Skip if already initialized unless forced
        if self._initialized and not force:
            logger.debug("Tokenizer discovery already performed, skipping",
                       extra={'subsys': 'tts', 'event': 'registry.already_initialized'})
            return {t: t in self._available_tokenizers for t in TOKENIZER_TYPES}
        
        # Clear existing tokenizers
        self._available_tokenizers.clear()
        
        # Always add grapheme as fallback
        self._available_tokenizers.add(DEFAULT_TOKENIZER)
        
        # Get environment diagnostics
        diagnostics = self._dump_environment_diagnostics()
        
        # Check for espeak binary
        if diagnostics['espeak_binary']:
            try:
                # Verify espeak works by running a simple command
                result = subprocess.run(
                    ["espeak", "--version"], 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    timeout=2
                )
                if result.returncode == 0:
                    self._available_tokenizers.add("espeak")
                    logger.info("espeak tokenizer is available", 
                              extra={'subsys': 'tts', 'event': 'registry.available.espeak'})
            except (subprocess.SubprocessError, FileNotFoundError) as e:
                logger.info(f"espeak binary found but not working: {e}", 
                          extra={'subsys': 'tts', 'event': 'registry.unavailable.espeak'})
        
        # Check for espeak-ng binary
        if diagnostics['espeak_ng_binary']:
            try:
                # Verify espeak-ng works by running a simple command
                result = subprocess.run(
                    ["espeak-ng", "--version"], 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    timeout=2
                )
                if result.returncode == 0:
                    self._available_tokenizers.add("espeak-ng")
                    logger.info("espeak-ng tokenizer is available", 
                              extra={'subsys': 'tts', 'event': 'registry.available.espeak_ng'})
            except (subprocess.SubprocessError, FileNotFoundError) as e:
                logger.info(f"espeak-ng binary found but not working: {e}", 
                          extra={'subsys': 'tts', 'event': 'registry.unavailable.espeak_ng'})
        
        # Check for phonemizer module
        if diagnostics['phonemizer_module']:
            try:
                # Try to import phonemizer
                import phonemizer
                self._available_tokenizers.add("phonemizer")
                logger.info("phonemizer tokenizer is available", 
                          extra={'subsys': 'tts', 'event': 'registry.available.phonemizer'})
            except ImportError as e:
                logger.info(f"phonemizer module found but import failed: {e}", 
                          extra={'subsys': 'tts', 'event': 'registry.unavailable.phonemizer'})
        
        # Check for g2p_en module
        if diagnostics['g2p_en_module']:
            try:
                # Try to import g2p_en
                import g2p_en
                self._available_tokenizers.add("g2p_en")
                logger.info("g2p_en tokenizer is available", 
                          extra={'subsys': 'tts', 'event': 'registry.available.g2p_en'})
            except ImportError as e:
                logger.info(f"g2p_en module found but import failed: {e}", 
                          extra={'subsys': 'tts', 'event': 'registry.unavailable.g2p_en'})
        
        # Check for misaki module
        if diagnostics['misaki_module']:
            try:
                # Try to import misaki
                import misaki
                self._available_tokenizers.add("misaki")
                logger.info("misaki tokenizer is available", 
                          extra={'subsys': 'tts', 'event': 'registry.available.misaki'})
            except ImportError as e:
                logger.info(f"misaki module found but import failed: {e}", 
                          extra={'subsys': 'tts', 'event': 'registry.unavailable.misaki'})
        
        # Log the discovered tokenizers
        logger.info(f"Tokenizer discovery completed: {', '.join(sorted(self._available_tokenizers))}", 
                  extra={'subsys': 'tts', 'event': 'registry.discovery_complete'})
        
        # Mark as initialized
        self._initialized = True
        self._size_at_init = len(self._available_tokenizers)
        
        # Return dictionary of available tokenizers
        return {t: t in self._available_tokenizers for t in TOKENIZER_TYPES}
    
    def _dump_environment_diagnostics(self) -> Dict[str, Any]:
        """Dump environment diagnostics for tokenizer discovery."""
        diagnostics = {}
        
        # Check for binary availability
        diagnostics['espeak_binary'] = shutil.which('espeak')
        diagnostics['espeak_ng_binary'] = shutil.which('espeak-ng')
        
        # Check for module availability
        diagnostics['phonemizer_module'] = importlib.util.find_spec('phonemizer') is not None
        diagnostics['g2p_en_module'] = importlib.util.find_spec('g2p_en') is not None
        diagnostics['misaki_module'] = importlib.util.find_spec('misaki') is not None
        
        return diagnostics
    
    def select_tokenizer_for_language(self, language: str) -> str:
        """
        Select the best tokenizer for the given language.
        
        Args:
            language: Language code (e.g., 'en', 'ja')
            
        Returns:
            Selected tokenizer name
            
        Raises:
            MissingTokeniserError: If no suitable tokenizer is found for the language
        """
        # Ensure discovery has been performed
        if not self._initialized:
            self.discover_tokenizers()
        
        # Check for registry corruption
        if len(self._available_tokenizers) < self._size_at_init:
            logger.error("Tokenizer registry has been corrupted (size decreased)",
                       extra={'subsys': 'tts', 'event': 'registry.corrupted'})
            # Re-discover to fix corruption
            self.discover_tokenizers(force=True)
        
        # Canonicalize language
        language = self._canonicalize_language(language)
        
        # Check for environment override
        env_tokenizer = os.environ.get('TTS_TOKENISER', '').strip().lower()
        if env_tokenizer:
            if env_tokenizer in self._available_tokenizers:
                logger.info(f"Using environment-specified tokenizer: {env_tokenizer}",
                          extra={'subsys': 'tts', 'event': 'registry.env_override'})
                return env_tokenizer
            else:
                logger.warning(f"TTS_TOKENISER environment variable '{env_tokenizer}' is not available",
                             extra={'subsys': 'tts', 'event': 'registry.env_override_invalid'})
        
        # Get tokenizer preferences for the language
        preferences = TOKENISER_MAP.get(language, TOKENISER_MAP.get('*', []))
        
        # Find the first available tokenizer in the preference list
        for tokenizer in preferences:
            if tokenizer in self._available_tokenizers:
                return tokenizer
        
        # If no preferred tokenizer is available, use grapheme for non-English
        if language != 'en' and DEFAULT_TOKENIZER in self._available_tokenizers:
            logger.warning(f"No preferred tokenizer available for language '{language}', using grapheme fallback",
                         extra={'subsys': 'tts', 'event': 'registry.fallback_grapheme'})
            return DEFAULT_TOKENIZER
        
        # For English, we need a phonetic tokenizer
        if language == 'en':
            logger.error(f"No English phonetic tokenizer found. Required: {preferences}, Available: {sorted(self._available_tokenizers)}",
                       extra={'subsys': 'tts', 'event': 'registry.missing_english_tokenizer'})
            raise MissingTokeniserError(f"No English phonetic tokenizer found")
        
        # For other languages, if even grapheme is not available, raise error
        logger.error(f"No tokenizer available for language '{language}'",
                   extra={'subsys': 'tts', 'event': 'registry.no_tokenizer'})
        raise MissingTokeniserError(f"No tokenizer available for language '{language}'")
    
    def _canonicalize_language(self, language: str) -> str:
        """
        Canonicalize language code.
        
        Args:
            language: Language code (e.g., 'en', 'ja', 'en-US')
            
        Returns:
            Canonicalized language code
        """
        if not language:
            return 'en'  # Default to English
        
        # Strip whitespace and convert to lowercase
        language = language.strip().lower()
        
        # Remove regional tag (e.g., 'en-US' -> 'en')
        if '-' in language:
            language = language.split('-')[0]
        
        # Map aliases
        if language == 'eng':
            language = 'en'
        elif language == 'jpn':
            language = 'ja'
        elif language == 'zho':
            language = 'zh'
            
        return language
        
    def is_tokenizer_warning_needed(self, language: str = None) -> bool:
        """
        Check if a tokenizer warning needs to be shown to the user.
        
        Args:
            language: Optional language code to check. If None, uses TTS_LANGUAGE env var.
        
        Returns:
            True if warning should be shown, False otherwise
        """
        # Ensure discovery has been performed
        if not self._initialized:
            self.discover_tokenizers()
        
        # Get the current language from parameter or environment
        if language is None:
            language = os.environ.get('TTS_LANGUAGE', 'en')
        
        # Canonicalize the language
        language = self._canonicalize_language(language)
        
        # For English, we need a phonetic tokenizer
        if language == 'en':
            english_tokenizers = set(TOKENISER_MAP['en']).intersection(self._available_tokenizers)
            if not english_tokenizers:
                logger.warning("No English phonetic tokenizer found. Speech quality will be poor.",
                             extra={'subsys': 'tts', 'event': 'registry.warning.english'})
                return True
        
        # For Japanese/Chinese, we need misaki
        if language in ('ja', 'zh'):
            if 'misaki' not in self._available_tokenizers:
                logger.warning(f"No {language} tokenizer found. Speech quality will be poor.",
                             extra={'subsys': 'tts', 'event': f'registry.warning.{language}'})
                return True
        
        return False
    
    def get_tokenizer_warning_message(self, language: str = "en") -> str:
        """
        Get a user-friendly warning message about missing tokenizers.
        
        Args:
            language: Language code for the warning message
            
        Returns:
            A message suitable for displaying to users
        """
        language = self._canonicalize_language(language)
        
        if language == 'en':
            return (
                "⚠️ **English phonetic tokeniser missing**\n\n"
                "For better English speech quality, please install one of:\n"
                "- `espeak` (Linux/macOS): `apt install espeak` or `brew install espeak`\n"
                "- `phonemizer`: `pip install phonemizer`\n"
                "- `g2p_en`: `pip install g2p_en`\n\n"
                "Without these, English speech will sound robotic and unnatural."
            )
        elif language in ('ja', 'zh'):
            return (
                f"⚠️ **Asian language tokenizer missing**\n\n"
                f"For better {language} speech quality, please install:\n"
                f"- `misaki`: `pip install misaki`\n\n"
                f"Without this, {language} speech will sound incorrect."
            )
        else:
            return (
                f"⚠️ **Phonetic tokeniser missing for {language}**\n\n"
                f"For better speech quality, please install one of:\n"
                f"- `phonemizer`: `pip install phonemizer`\n"
                f"- `espeak`: `apt install espeak` or `brew install espeak`\n\n"
                f"Without these, speech will sound robotic and unnatural."
            )
    
    def get_available_tokenizers(self) -> Set[str]:
        """Get the set of available tokenizers."""
        # Ensure discovery has been performed
        if not self._initialized:
            self.discover_tokenizers()
        
        return self._available_tokenizers.copy()


# Convenience functions that use the registry singleton

def discover_tokenizers(force: bool = False) -> Dict[str, bool]:
    """
    Discover available tokenizers using the registry singleton.
    
    Args:
        force: Force rediscovery even if already initialized
        
    Returns:
        Dictionary mapping tokenizer names to availability status
    """
    registry = TokenizerRegistry.get_instance()
    return registry.discover_tokenizers(force)


def select_tokenizer_for_language(language: str) -> str:
    """
    Select the best tokenizer for the given language using the registry singleton.
    
    Args:
        language: Language code (e.g., 'en', 'ja')
        
    Returns:
        Selected tokenizer name
        
    Raises:
        MissingTokeniserError: If no suitable tokenizer is found for the language
    """
    registry = TokenizerRegistry.get_instance()
    return registry.select_tokenizer_for_language(language)


def is_tokenizer_warning_needed(language: str = "en") -> bool:
    """
    Check if a tokenizer warning needs to be shown to the user.
    
    Args:
        language: Language code to check for tokenizer warnings
        
    Returns:
        True if warning should be shown, False otherwise
    """
    registry = TokenizerRegistry.get_instance()
    return registry.is_tokenizer_warning_needed(language)


def get_tokenizer_warning_message(language: str = "en") -> str:
    """
    Get a user-friendly warning message about missing tokenizers.
    
    Args:
        language: Language code for the warning message
        
    Returns:
        A message suitable for displaying to users
    """
    registry = TokenizerRegistry.get_instance()
    return registry.get_tokenizer_warning_message(language)


def get_available_tokenizers() -> Set[str]:
    """Get the set of available tokenizers using the registry singleton."""
    registry = TokenizerRegistry.get_instance()
    return registry.get_available_tokenizers()
