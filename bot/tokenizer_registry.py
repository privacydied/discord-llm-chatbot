"""
Tokenizer registry for TTS tokenizer discovery and management.

This module provides a singleton registry for tokenizer availability
to ensure consistent state across imports and prevent reset issues.
"""

import logging
import os
from typing import Dict, Set, Optional, Any, Tuple, Literal
from dataclasses import dataclass
import importlib.util
import shutil
import subprocess
import json
import re
from pathlib import Path

from .tts.errors import MissingTokeniserError

logger = logging.getLogger(__name__)

@dataclass
class Decision:
    """Typed tokenizer decision for language processing."""
    mode: Literal["phonemes", "grapheme"]
    payload: str  # phoneme string for mode=="phonemes", normalized text for "grapheme"
    alphabet: Literal["IPA", "ARPABET", "GRAPHEME"]  # hint; use "GRAPHEME" for grapheme

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
        # Per-language lexicon cache (lower-cased keys)
        self._lexicons: Dict[str, Dict[str, str]] = {}
    
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
            # Removed warning here - only warn in autodiscovery branch when actually used
        
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
    
    def _load_lexicon(self, language: str) -> Dict[str, str]:
        """Load lexicon for a language from env or default path. Cached per language.
        Looks for env TTS_LEXICON or TTS_LEXICON_PATH. Defaults to `bot/tts/lexicon_<lang>.json`.
        Returns a dict mapping lowercased words to phoneme strings.
        """
        lang = self._canonicalize_language(language or os.environ.get('TTS_LANGUAGE', 'en'))
        if lang in self._lexicons:
            return self._lexicons[lang]
        lex: Dict[str, str] = {}
        try:
            env_path = os.environ.get('TTS_LEXICON') or os.environ.get('TTS_LEXICON_PATH')
            if env_path:
                p = Path(env_path)
                if p.exists():
                    with p.open('r', encoding='utf-8') as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        lex = {str(k).lower(): str(v) for k, v in data.items()}
                else:
                    logger.warning(
                        f"TTS lexicon path not found: {p}",
                        extra={'subsys': 'tts', 'event': 'registry.lexicon.missing_path'}
                    )
            else:
                default = Path(__file__).resolve().parent / 'tts' / f'lexicon_{lang}.json'
                if default.exists():
                    with default.open('r', encoding='utf-8') as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        lex = {str(k).lower(): str(v) for k, v in data.items()}
        except Exception as e:
            logger.error(
                f"Failed to load lexicon: {e}",
                extra={'subsys': 'tts', 'event': 'registry.lexicon.error'},
                exc_info=True,
            )
        self._lexicons[lang] = lex
        if lex:
            logger.info(
                f"Loaded lexicon for {lang} with {len(lex)} entries",
                extra={'subsys': 'tts', 'event': 'registry.lexicon.loaded'}
            )
        return lex
    
    def apply_lexicon(self, text: str, language: str) -> Tuple[str, bool]:
        """Apply lexicon replacements to text for a given language.
        Returns (new_text, changed).
        """
        if not text:
            return text, False
        lang = self._canonicalize_language(language or os.environ.get('TTS_LANGUAGE', 'en'))
        lex = self._load_lexicon(lang)
        if not lex:
            return text, False
        # Replace word tokens (simple Latin word pattern) case-insensitively
        pattern = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
        def repl(match: re.Match) -> str:
            w = match.group(0)
            return lex.get(w.lower(), w)
        new_text = pattern.sub(repl, text)
        changed = new_text != text
        if changed:
            logger.debug(
                f"Applied lexicon replacements for {lang}",
                extra={'subsys': 'tts', 'event': 'registry.lexicon_applied'}
            )
        return new_text, changed
    
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
    
    def select_for_language(self, language: str, text: str) -> Decision:
        """
        Select the best tokenizer for the given language and return a typed Decision.
        
        This method handles the complete tokenization decision process:
        1. Applies lexicon overrides if available
        2. Selects appropriate tokenizer based on language and availability
        3. Handles misaki fallback for English with proper logging
        4. Tokenizes the text and returns a Decision object
        
        Args:
            language: Language code (e.g., 'en', 'ja', 'en-US')
            text: Input text to tokenize
            
        Returns:
            Decision object with mode, payload, and alphabet information
            
        Raises:
            MissingTokeniserError: If no suitable tokenizer is found for the language
        """
        # Ensure discovery has been performed
        if not self._initialized:
            self.discover_tokenizers()
            
        # Canonicalize language
        language = self._canonicalize_language(language)
        
        # Apply lexicon first
        text, lex_changed = self.apply_lexicon(text, language)
        
        # Get tokenizer selection
        tokenizer = self._select_tokenizer_with_fallback(language)
        
        # Handle tokenization based on tokenizer type
        if tokenizer in ("phonemizer", "espeak", "espeak-ng", "g2p_en"):
            # These produce phonemes - try to tokenize
            try:
                phonemes = self._tokenize_to_phonemes(text, tokenizer, language)
                if phonemes:
                    return Decision(
                        mode="phonemes",
                        payload=phonemes,
                        alphabet="IPA"  # Most of these produce IPA-like output
                    )
            except Exception as e:
                logger.debug(f"Phoneme tokenization failed with {tokenizer}: {e}", 
                           extra={'subsys': 'tts', 'event': 'registry.phoneme_failed'})
                
        elif tokenizer == "misaki":
            # Misaki produces phonemes for Japanese/Chinese
            try:
                phonemes = self._tokenize_to_phonemes(text, tokenizer, language)
                if phonemes:
                    return Decision(
                        mode="phonemes", 
                        payload=phonemes,
                        alphabet="IPA"
                    )
            except Exception as e:
                logger.debug(f"Misaki tokenization failed: {e}",
                           extra={'subsys': 'tts', 'event': 'registry.misaki_failed'})
                
        # Fallback to grapheme for any failures or grapheme tokenizer
        logger.debug(f"Using grapheme tokenization for {language}",
                   extra={'subsys': 'tts', 'event': 'registry.grapheme_fallback'})
        return Decision(
            mode="grapheme",
            payload=text,
            alphabet="GRAPHEME"
        )
    
    def _select_tokenizer_with_fallback(self, language: str) -> str:
        """
        Select tokenizer for language with proper English misaki handling.
        
        For English, if misaki is specified via env but not ideal, 
        fall back to phonemizer with a debug log.
        """
        # Check for environment override
        env_tokenizer = os.environ.get('TTS_TOKENISER', '').strip().lower()
        if env_tokenizer:
            if env_tokenizer in self._available_tokenizers:
                # Special handling for English + misaki
                if language == 'en' and env_tokenizer == 'misaki':
                    # Misaki is JP/CN only, fall back to phonemizer for English
                    if 'phonemizer' in self._available_tokenizers:
                        logger.debug("misaki is JP-only; falling back to phonemizer for en",
                                   extra={'subsys': 'tts', 'event': 'registry.misaki_fallback'})
                        return 'phonemizer'
                    elif 'g2p_en' in self._available_tokenizers:
                        logger.debug("misaki is JP-only; falling back to g2p_en for en",
                                   extra={'subsys': 'tts', 'event': 'registry.misaki_fallback'})
                        return 'g2p_en'
                    elif 'espeak' in self._available_tokenizers:
                        logger.debug("misaki is JP-only; falling back to espeak for en",
                                   extra={'subsys': 'tts', 'event': 'registry.misaki_fallback'})
                        return 'espeak'
                
                logger.debug(f"Using environment-specified tokenizer: {env_tokenizer}",
                           extra={'subsys': 'tts', 'event': 'registry.env_override'})
                return env_tokenizer
        
        # Get tokenizer preferences for the language
        preferences = TOKENISER_MAP.get(language, TOKENISER_MAP.get('*', []))
        
        # Find the first available tokenizer in the preference list
        for tokenizer in preferences:
            if tokenizer in self._available_tokenizers:
                return tokenizer
        
        # If no preferred tokenizer is available, use grapheme for non-English
        if language != 'en' and DEFAULT_TOKENIZER in self._available_tokenizers:
            return DEFAULT_TOKENIZER
        
        # For English, we need a phonetic tokenizer
        if language == 'en':
            logger.error(f"No English phonetic tokenizer found. Required: {preferences}, Available: {sorted(self._available_tokenizers)}",
                       extra={'subsys': 'tts', 'event': 'registry.missing_english_tokenizer'})
            raise MissingTokeniserError("No English phonetic tokenizer found")
        
        # For other languages, if even grapheme is not available, raise error
        logger.error(f"No tokenizer available for language '{language}'",
                   extra={'subsys': 'tts', 'event': 'registry.no_tokenizer'})
        raise MissingTokeniserError(f"No tokenizer available for language '{language}'")
    
    def _tokenize_to_phonemes(self, text: str, tokenizer: str, language: str) -> str:
        """Tokenize text to phonemes using the specified tokenizer."""
        if tokenizer == "phonemizer":
            try:
                from phonemizer import phonemize
                return phonemize(text, language='en-us', backend='espeak', strip=True)
            except ImportError:
                raise Exception("phonemizer not available")
                
        elif tokenizer in ("espeak", "espeak-ng"):
            try:
                # Use subprocess to get phonemes from espeak
                cmd = [tokenizer, "--ipa", "-q", "--stdin"]
                result = subprocess.run(
                    cmd, 
                    input=text, 
                    text=True, 
                    capture_output=True, 
                    timeout=10
                )
                if result.returncode == 0:
                    return result.stdout.strip()
                else:
                    raise Exception(f"espeak failed: {result.stderr}")
            except (subprocess.SubprocessError, FileNotFoundError):
                raise Exception(f"{tokenizer} not available")
                
        elif tokenizer == "g2p_en":
            try:
                from g2p_en import G2p
                g2p = G2p()
                phonemes = g2p(text)
                return ' '.join(phonemes)
            except ImportError:
                raise Exception("g2p_en not available")
                
        elif tokenizer == "misaki":
            try:
                from misaki import en as misaki_en
                g2p = misaki_en.G2P()
                result = g2p(text)
                # Handle tuple return
                if isinstance(result, tuple):
                    return result[0]
                return result
            except ImportError:
                raise Exception("misaki not available")
                
        else:
            raise Exception(f"Unsupported tokenizer: {tokenizer}")

# Minimal ARPAbet -> IPA (covers core English; extend as needed)
ARPABET_TO_IPA = {
    "AA":"ɑ", "AE":"æ", "AH":"ʌ", "AO":"ɔ", "AW":"aʊ", "AY":"aɪ",
    "B":"b", "CH":"t͡ʃ", "D":"d", "DH":"ð",
    "EH":"ɛ", "ER":"ɝ", "EY":"eɪ", "F":"f", "G":"ɡ", "HH":"h",
    "IH":"ɪ", "IY":"i", "JH":"d͡ʒ", "K":"k", "L":"l", "M":"m",
    "N":"n", "NG":"ŋ", "OW":"oʊ", "OY":"ɔɪ", "P":"p", "R":"ɹ",
    "S":"s", "SH":"ʃ", "T":"t", "TH":"θ", "UH":"ʊ", "UW":"u",
    "V":"v", "W":"w", "Y":"j", "Z":"z", "ZH":"ʒ"
}

def arpabet_to_ipa(seq):
    """
    Convert a sequence of ARPAbet tokens (possibly with stress digits, e.g. 'IH1')
    into an IPA string separated by spaces.
    Example input: ['K', 'AW1', 'N', 'T', 'IH1', 'NG']
    Output: 'k aʊ n t ɪ ŋ'
    """
    out = []
    for s in seq:
        if not s or s.isspace():
            continue
        base = ''.join(ch for ch in s if not ch.isdigit()).upper()
        out.append(ARPABET_TO_IPA.get(base, base.lower()))
    return " ".join(out)

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


def apply_lexicon(text: str, language: str) -> Tuple[str, bool]:
    """Apply lexicon replacements using the registry singleton."""
    registry = TokenizerRegistry.get_instance()
    return registry.apply_lexicon(text, language)


def select_for_language(language: str, text: str) -> Decision:
    """Select tokenizer and tokenize text using the registry singleton."""
    registry = TokenizerRegistry.get_instance()
    return registry.select_for_language(language, text)
