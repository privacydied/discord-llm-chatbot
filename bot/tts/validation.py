"""TTS validation utilities for detecting issues with TTS output."""

import logging
import os
import subprocess
import shutil
import importlib
import sys
import site
import numpy as np
from enum import Enum
from pathlib import Path
from typing import Dict, Set, Optional, List, Tuple, Any, Union

from .errors import TTSGibberishError, MissingTokeniserError

logger = logging.getLogger(__name__)

# Global tokenizer availability cache
AVAILABLE_TOKENIZERS: Set[str] = set()

# Language-to-tokenizer mapping as specified
TOKENISER_MAP = {
    "en": ["espeak", "espeak-ng", "phonemizer", "g2p_en"],
    "ja": ["misaki"],
    "zh": ["misaki"],
    # Default for other languages
    "*": ["phonemizer", "espeak", "espeak-ng"]
}

# Default to grapheme for all languages
DEFAULT_TOKENIZER = "grapheme"

class TokenizerType(Enum):
    """Types of tokenizers available for TTS."""
    ESPEAK = "espeak"
    ESPEAK_NG = "espeak-ng"
    PHONEMIZER = "phonemizer"
    G2P_EN = "g2p_en"
    MISAKI = "misaki"
    GRAPHEME = "grapheme"
    UNKNOWN = "unknown"

_ALIAS_WARNING_SHOWN = False

def _get_env_tokenizer() -> Optional[str]:
    """Read tokenizer env with alias support for TTS_TOKENIZER.

    Prefer British spelling TTS_TOKENISER. If absent, accept TTS_TOKENIZER
    and log a one-time warning.
    """
    global _ALIAS_WARNING_SHOWN
    val = os.environ.get('TTS_TOKENISER')
    if val:
        return val
    alt = os.environ.get('TTS_TOKENIZER')
    if alt:
        if not _ALIAS_WARNING_SHOWN:
            logger.warning(
                "Using TTS_TOKENIZER (alias of TTS_TOKENISER). Prefer TTS_TOKENISER for consistency.",
                extra={'subsys': 'tts', 'event': 'tokenizer.env_alias'}
            )
            _ALIAS_WARNING_SHOWN = True
        return alt
    return None

def get_site_packages_dirs() -> List[str]:
    """Get a list of site-packages directories in the current Python environment."""
    try:
        # Get all site-packages directories
        site_packages = []
        for path in site.getsitepackages():
            if 'site-packages' in path:
                site_packages.append(path)
        
        # Add user site-packages if enabled
        if site.ENABLE_USER_SITE and site.USER_SITE:
            site_packages.append(site.USER_SITE)
            
        return site_packages
    except Exception as e:
        logger.error(f"Failed to get site-packages directories: {e}", 
                   extra={'subsys': 'tts', 'event': 'site_packages.error'})
        return []

def dump_environment_diagnostics() -> Dict[str, Any]:
    """Dump detailed environment diagnostics for tokenizer discovery."""
    diagnostics = {}
    
    # Get PATH environment variable
    path_env = os.environ.get('PATH', '')
    path_dirs = path_env.split(os.pathsep)
    diagnostics['PATH'] = path_dirs
    
    # Get site-packages directories
    site_packages = get_site_packages_dirs()
    diagnostics['site_packages'] = site_packages
    
    # Check for binary availability
    diagnostics['espeak_binary'] = shutil.which('espeak')
    diagnostics['espeak_ng_binary'] = shutil.which('espeak-ng')
    
    # Check for module availability
    diagnostics['phonemizer_module'] = importlib.util.find_spec('phonemizer') is not None
    diagnostics['g2p_en_module'] = importlib.util.find_spec('g2p_en') is not None
    diagnostics['misaki_module'] = importlib.util.find_spec('misaki') is not None
    
    # Log the diagnostics
    logger.info("Tokeniser discovery environment:", 
              extra={'subsys': 'tts', 'event': 'tokenizer.env.start'})
    
    # Log PATH
    logger.info(f"PATH: {os.pathsep.join(path_dirs[:3])}{'...' if len(path_dirs) > 3 else ''}", 
              extra={'subsys': 'tts', 'event': 'tokenizer.env.path'})
    
    # Log site-packages
    site_pkg_display = [os.path.basename(p) for p in site_packages[:5]]
    logger.info(f"python.site: {site_pkg_display}{'...' if len(site_packages) > 5 else ''}", 
              extra={'subsys': 'tts', 'event': 'tokenizer.env.site_packages'})
    
    # Log binary availability
    if diagnostics['espeak_binary']:
        logger.info(f"espeak binary: FOUND ({diagnostics['espeak_binary']})", 
                  extra={'subsys': 'tts', 'event': 'tokenizer.env.binary.espeak'})
    else:
        logger.info("espeak binary: NOT FOUND", 
                  extra={'subsys': 'tts', 'event': 'tokenizer.env.binary.espeak'})
    
    if diagnostics['espeak_ng_binary']:
        logger.info(f"espeak-ng binary: FOUND ({diagnostics['espeak_ng_binary']})", 
                  extra={'subsys': 'tts', 'event': 'tokenizer.env.binary.espeak_ng'})
    else:
        logger.info("espeak-ng binary: NOT FOUND", 
                  extra={'subsys': 'tts', 'event': 'tokenizer.env.binary.espeak_ng'})
    
    # Log module availability
    logger.info(f"phonemizer module: {'FOUND' if diagnostics['phonemizer_module'] else 'NOT FOUND'}", 
              extra={'subsys': 'tts', 'event': 'tokenizer.env.module.phonemizer'})
    
    logger.info(f"g2p_en module: {'FOUND' if diagnostics['g2p_en_module'] else 'NOT FOUND'}", 
              extra={'subsys': 'tts', 'event': 'tokenizer.env.module.g2p_en'})
    
    logger.info(f"misaki module: {'FOUND' if diagnostics['misaki_module'] else 'NOT APPLICABLE'}", 
              extra={'subsys': 'tts', 'event': 'tokenizer.env.module.misaki'})
    
    return diagnostics

def detect_available_tokenizers() -> Dict[str, bool]:
    """
    Detect which tokenizers are available in the current environment.
    Populates the global AVAILABLE_TOKENIZERS set.
    
    Returns:
        Dictionary mapping tokenizer names to availability status
        
    Raises:
        MissingTokeniserError: If no suitable tokenizer is found for English
    """
    global AVAILABLE_TOKENIZERS
    
    # Dump environment diagnostics
    diagnostics = dump_environment_diagnostics()
    
    # Initialize with grapheme (always available as fallback)
    available = {
        TokenizerType.ESPEAK.value: False,
        TokenizerType.ESPEAK_NG.value: False,
        TokenizerType.PHONEMIZER.value: False,
        TokenizerType.G2P_EN.value: False,
        TokenizerType.MISAKI.value: False,
        TokenizerType.GRAPHEME.value: True,  # Grapheme is always available as fallback
    }
    
    # Check for espeak binary
    espeak_path = diagnostics['espeak_binary']
    if espeak_path:
        try:
            result = subprocess.run(['espeak', '--version'], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE,
                                   text=True,
                                   check=False)
            available[TokenizerType.ESPEAK.value] = result.returncode == 0
        except (FileNotFoundError, subprocess.SubprocessError):
            pass
    
    # Check for espeak-ng binary (alternative name on some distros)
    espeak_ng_path = diagnostics['espeak_ng_binary']
    if espeak_ng_path:
        try:
            result = subprocess.run(['espeak-ng', '--version'], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE,
                                   text=True,
                                   check=False)
            available[TokenizerType.ESPEAK_NG.value] = result.returncode == 0
        except (FileNotFoundError, subprocess.SubprocessError):
            pass
    
    # Check for phonemizer Python package
    if diagnostics['phonemizer_module']:
        try:
            import phonemizer
            available[TokenizerType.PHONEMIZER.value] = True
        except ImportError:
            pass
    
    # Check for g2p_en Python package (specific for English)
    if diagnostics['g2p_en_module']:
        try:
            import g2p_en
            available[TokenizerType.G2P_EN.value] = True
        except ImportError:
            pass
    
    # Check for misaki (Japanese/Chinese tokenizer)
    if diagnostics['misaki_module']:
        try:
            import misaki
            available[TokenizerType.MISAKI.value] = True
        except ImportError:
            pass
    
    # Update global set of available tokenizers
    AVAILABLE_TOKENIZERS = {k for k, v in available.items() if v}
    
    # Log available tokenizers
    if AVAILABLE_TOKENIZERS - {TokenizerType.GRAPHEME.value}:  # Exclude grapheme from consideration
        logger.info(f"Tokenisers found: {sorted(list(AVAILABLE_TOKENIZERS))}",
                  extra={'subsys': 'tts', 'event': 'tokenizer.available', 
                         'tokenizers': sorted(list(AVAILABLE_TOKENIZERS))})
    else:
        logger.warning("No known tokenization methods found",
                     extra={'subsys': 'tts', 'event': 'tokenizer.none_available'})
    
    # Check if TTS_TOKENISER environment variable is set
    env_tokenizer = _get_env_tokenizer()
    if env_tokenizer and env_tokenizer not in AVAILABLE_TOKENIZERS:
        logger.error(f"TTS_TOKENISER environment variable '{env_tokenizer}' is not available. "
                   f"Available tokenizers: {sorted(list(AVAILABLE_TOKENIZERS))}",
                  extra={'subsys': 'tts', 'event': 'tokenizer.env_override.invalid',
                         'requested': env_tokenizer, 
                         'available': sorted(list(AVAILABLE_TOKENIZERS))})
    
    return available

def select_tokenizer_for_language(language: str, available_tokenizers: Optional[Dict[str, bool]] = None) -> str:
    """
    Select the best tokenizer for the given language based on TOKENISER_MAP.
    
    Args:
        language: Language code (e.g., 'en', 'ja')
        available_tokenizers: Dictionary of available tokenizers (optional)
        
    Returns:
        Selected tokenizer name
        
    Raises:
        MissingTokeniserError: If no suitable tokenizer is found for the language
    """
    global AVAILABLE_TOKENIZERS
    
    # Normalize language code (lowercase, strip whitespace)
    language = language.lower().strip()
    
    # Get available tokenizers
    if available_tokenizers is None:
        # Use cached available tokenizers
        available_set = AVAILABLE_TOKENIZERS
    else:
        # Use provided available tokenizers
        available_set = {k for k, v in available_tokenizers.items() if v}
    
    # Check for environment variable override
    env_tokenizer = _get_env_tokenizer()
    if env_tokenizer:
        if env_tokenizer in available_set:
            logger.info(f"Using tokenizer from TTS_TOKENISER environment variable: {env_tokenizer}",
                      extra={'subsys': 'tts', 'event': 'tokenizer.env_override',
                             'tokenizer': env_tokenizer})
            return env_tokenizer
        else:
            logger.error(f"TTS_TOKENISER environment variable '{env_tokenizer}' is not available",
                       extra={'subsys': 'tts', 'event': 'tokenizer.env_override.invalid',
                              'requested': env_tokenizer})
            # Continue with auto-selection
    
    # Get preferred tokenizers for this language
    lang_key = language[:2]  # Use first two chars (e.g., 'en' from 'en-us')
    preferred_tokenizers = TOKENISER_MAP.get(lang_key, TOKENISER_MAP['*'])
    
    # Find first available preferred tokenizer
    for tokenizer in preferred_tokenizers:
        if tokenizer in available_set:
            logger.info(f"Selected tokenizer '{tokenizer}' for language '{language}'",
                      extra={'subsys': 'tts', 'event': 'tokenizer.selected',
                             'tokenizer': tokenizer, 'language': language,
                             'fallback': False})
            return tokenizer
    
    # For English, we require a proper phonetic tokenizer
    if language.startswith('en') and not any(t in available_set for t in ['espeak', 'espeak-ng', 'phonemizer', 'g2p_en']):
        required = ['espeak-ng', 'phonemizer', 'g2p_en']
        available = list(available_set)
        logger.error(f"No English phonetic tokenizer found. Required: {required}, Available: {available}",
                   extra={'subsys': 'tts', 'event': 'tokenizer.missing.critical',
                          'language': language, 'required': required, 'available': available})
        raise MissingTokeniserError(language, available, required)
    
    # If no preferred tokenizer is available, fall back to grapheme
    logger.warning(f"No preferred tokenizer available for language '{language}', using grapheme fallback",
                 extra={'subsys': 'tts', 'event': 'tokenizer.fallback',
                        'language': language})
    return DEFAULT_TOKENIZER

def is_tokenizer_warning_needed() -> bool:
    """
    Check if a tokenizer warning needs to be shown to the user.
    
    Returns:
        True if warning should be shown, False otherwise
    """
    global AVAILABLE_TOKENIZERS
    
    # Get language from environment with fallback
    language = os.environ.get('TTS_LANGUAGE', 'en').lower().strip()
    
    # For English, we need a proper phonetic tokenizer
    if language.startswith('en'):
        english_tokenizers = {'espeak', 'espeak-ng', 'phonemizer', 'g2p_en'}
        has_english_tokenizer = any(t in AVAILABLE_TOKENIZERS for t in english_tokenizers)
        if not has_english_tokenizer:
            logger.warning("No English phonetic tokenizer found. Speech quality will be poor.",
                         extra={'subsys': 'tts', 'event': 'tokenizer.warning.english'})
        return not has_english_tokenizer
    
    # For Japanese/Chinese, we need misaki
    if language.startswith('ja') or language.startswith('zh'):
        has_asian_tokenizer = 'misaki' in AVAILABLE_TOKENIZERS
        if not has_asian_tokenizer:
            logger.warning(f"No {language} tokenizer found. Speech quality will be poor.",
                         extra={'subsys': 'tts', 'event': 'tokenizer.warning.asian'})
        return not has_asian_tokenizer
    
    return False

def get_tokenizer_warning_message(language: str = "en") -> str:
    """
    Get a user-friendly warning message about missing tokenizers.
    
    Args:
        language: Language code for the warning message
        
    Returns:
        A message suitable for displaying to users
    """
    global TOKENIZER_WARNING_SHOWN
    
    # Mark warning as shown
    TOKENIZER_WARNING_SHOWN = True
    
    language_prefix = language.split('-')[0].lower()
    
    if language_prefix == "en":
        return ("⚠ No English phonetic tokeniser installed; speech quality degraded. "
                "Install espeak‑ng or phonemizer for clearer output.")
    elif language_prefix == "ja" or language_prefix == "zh":
        return ("⚠ No Asian language tokenizer found. Speech quality may be reduced. "
                "Install misaki for better results.")
    else:
        return ("⚠ No phonetic tokenizer found for your language. Speech quality may be reduced. "
                "Install phonemizer or espeak-ng for better results.")

def validate_voice_vector(voice_vector: np.ndarray, voice_id: str = None) -> bool:
    """Validate that a voice vector has the expected properties.
    
    Args:
        voice_vector: The voice vector to validate
        voice_id: Optional voice ID for logging
        
    Returns:
        True if the voice vector is valid, False otherwise
    """
    # Check if voice vector is None
    if voice_vector is None:
        logger.error("Voice vector is None", 
                    extra={'subsys': 'tts', 'event': 'voice_validation', 'voice_id': voice_id})
        return False
    
    # Check vector shape - should be 1D with 256 elements for kokoro-onnx
    expected_length = 256
    if len(voice_vector.shape) != 1 or voice_vector.shape[0] != expected_length:
        logger.error(f"Voice vector has unexpected shape: {voice_vector.shape}, expected (256,)", 
                    extra={'subsys': 'tts', 'event': 'voice_validation', 'voice_id': voice_id})
        return False
    
    # Check if vector has non-zero norm (all zeros would be invalid)
    vector_norm = np.linalg.norm(voice_vector)
    if vector_norm < 1e-6:
        logger.error(f"Voice vector has near-zero norm: {vector_norm}", 
                    extra={'subsys': 'tts', 'event': 'voice_validation', 'voice_id': voice_id})
        return False
    
    # Check if vector contains NaN or Inf values
    if np.isnan(voice_vector).any() or np.isinf(voice_vector).any():
        logger.error("Voice vector contains NaN or Inf values", 
                    extra={'subsys': 'tts', 'event': 'voice_validation', 'voice_id': voice_id})
        return False
    
    # Check if vector is normalized (optional, depends on model requirements)
    # Some models expect normalized vectors, others don't
    normalized_norm = abs(vector_norm - 1.0)
    if normalized_norm > 0.1:  # If norm is not close to 1.0
        logger.warning(f"Voice vector is not normalized (norm: {vector_norm:.4f})", 
                      extra={'subsys': 'tts', 'event': 'voice_validation', 'voice_id': voice_id})
        # We don't return false here as this is just a warning
    
    # Check language tag in voice ID if provided
    if voice_id:
        # Extract language code from voice ID (typically format like 'en-US-GuyNeural')
        parts = voice_id.split('-')
        if len(parts) >= 2:
            lang_code = f"{parts[0]}-{parts[1]}"
            logger.debug(f"Voice ID {voice_id} has language code {lang_code}",
                        extra={'subsys': 'tts', 'event': 'voice_validation', 'voice_id': voice_id})
    
    logger.debug(f"Voice vector validated successfully (norm: {vector_norm:.4f}, length: {voice_vector.shape[0]})", 
                extra={'subsys': 'tts', 'event': 'voice_validation', 'voice_id': voice_id})
    return True
    """
    Validate a voice embedding vector.
    
    Args:
        voice_vector: Voice embedding vector
        voice_id: Voice ID for logging
        
    Returns:
        True if valid, False otherwise
        
    Raises:
        ValueError: If voice vector is invalid
    """
    if voice_vector is None:
        logger.error(f"Voice vector is None for voice '{voice_id}'",
                   extra={'subsys': 'tts', 'event': 'voice.validate.none', 'voice_id': voice_id})
        raise ValueError(f"Voice vector is None for voice '{voice_id}'")
    
    # Check shape
    if voice_vector.shape != (256,):
        logger.error(f"Voice vector has incorrect shape: {voice_vector.shape}, expected (256,) for voice '{voice_id}'",
                   extra={'subsys': 'tts', 'event': 'voice.validate.shape', 
                          'voice_id': voice_id, 'shape': voice_vector.shape})
        raise ValueError(f"Voice vector has incorrect shape: {voice_vector.shape}, expected (256,)")
    
    # Check for non-zero norm
    norm = np.linalg.norm(voice_vector)
    if norm < 1e-6:
        logger.error(f"Voice vector has near-zero norm: {norm:.6f} for voice '{voice_id}'",
                   extra={'subsys': 'tts', 'event': 'voice.validate.norm', 
                          'voice_id': voice_id, 'norm': norm})
        raise ValueError(f"Voice vector has near-zero norm: {norm:.6f}")
    
    logger.debug(f"Voice vector validated: shape={voice_vector.shape}, norm={norm:.6f} for voice '{voice_id}'",
               extra={'subsys': 'tts', 'event': 'voice.validate.success', 
                      'voice_id': voice_id, 'norm': norm})
    
    return True

def check_sample_rate_consistency(expected_rate: int, actual_rate: int) -> bool:
    """Check if the sample rates are consistent.
    
    Args:
        expected_rate: The expected sample rate
        actual_rate: The actual sample rate
        
    Returns:
        True if the sample rates are consistent, False otherwise
    """
    # Check if rates match exactly
    if expected_rate == actual_rate:
        return True
    
    # Calculate percentage difference
    diff_percent = abs(expected_rate - actual_rate) / expected_rate * 100
    
    # If difference is less than 1%, consider it consistent but log a warning
    if diff_percent < 1.0:
        logger.warning(f"Sample rates slightly different: expected {expected_rate}Hz, got {actual_rate}Hz",
                     extra={'subsys': 'tts', 'event': 'sample_rate'})
        return True
    
    # If difference is significant, log an error
    logger.error(f"Sample rate mismatch: expected {expected_rate}Hz, got {actual_rate}Hz",
                extra={'subsys': 'tts', 'event': 'sample_rate'})
    return False

def detect_gibberish_audio(audio_data: np.ndarray, sample_rate: int) -> bool:
    """Detect if the generated audio is likely gibberish or wrong language.
    
    Args:
        audio_data: The audio data as a numpy array
        sample_rate: The sample rate of the audio data
        
    Returns:
        True if the audio is likely gibberish, False otherwise
    """
    # Check if audio is mostly silence or very low amplitude
    mean_abs = np.mean(np.abs(audio_data))
    if mean_abs < 1e-4:
        logger.warning(f"Audio has very low amplitude (mean abs: {mean_abs}), likely gibberish",
                      extra={'subsys': 'tts', 'event': 'gibberish_detection'})
        return True
    
    # Calculate zero-crossing rate (high ZCR often indicates noise or gibberish)
    # First, ensure audio is mono
    if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
        audio_mono = np.mean(audio_data, axis=1)
    else:
        audio_mono = audio_data.flatten()
    
    # Calculate zero crossing rate
    zero_crossings = np.sum(np.abs(np.diff(np.signbit(audio_mono).astype(int))))
    zcr = zero_crossings / len(audio_mono)
    
    # Extremely high ZCR is often indicative of noise/gibberish
    # Normal speech typically has ZCR between 0.01-0.1
    if zcr > 0.2:  # Threshold determined empirically
        logger.warning(f"Audio has high zero-crossing rate ({zcr:.4f}), likely gibberish",
                      extra={'subsys': 'tts', 'event': 'gibberish_detection'})
        return True
    
    # Check for extreme values in the audio
    if np.max(np.abs(audio_data)) > 0.99:
        # Check if clipping is extensive (more than 1% of samples)
        clipping_ratio = np.sum(np.abs(audio_data) > 0.99) / audio_data.size
        if clipping_ratio > 0.01:
            logger.warning(f"Audio has extensive clipping ({clipping_ratio:.4f}), likely distorted",
                          extra={'subsys': 'tts', 'event': 'gibberish_detection'})
            return True
    
    # Check for constant segments (stuck voice)
    # Split audio into segments and check if any segment has very low variance
    segment_length = int(sample_rate * 0.5)  # 500ms segments
    if len(audio_mono) > segment_length * 2:  # Only check if audio is long enough
        for i in range(0, len(audio_mono) - segment_length, segment_length):
            segment = audio_mono[i:i+segment_length]
            if np.var(segment) < 1e-6 and np.mean(np.abs(segment)) > 1e-3:
                logger.warning(f"Audio has constant segment at {i/sample_rate:.2f}s, likely stuck voice",
                              extra={'subsys': 'tts', 'event': 'gibberish_detection'})
                return True
    
    return False

def detect_gibberish_audio_with_metrics(audio_data: np.ndarray, sample_rate: int) -> Tuple[bool, Dict[str, float]]:
    """Detect if the generated audio is likely gibberish or wrong language.
    
    Args:
        audio_data: The audio data as a numpy array
        sample_rate: The sample rate of the audio data
        
    Returns:
        Tuple of (is_gibberish, metrics)
        
    Raises:
        TTSGibberishError: If audio is detected as gibberish
    """
    metrics = {}
    
    # Check for all zeros (completely silent audio)
    if np.all(audio_data == 0):
        metrics['all_zeros'] = True
        metrics['avg_abs'] = 0.0
        metrics['rms'] = 0.0
        metrics['zcr'] = 0.0
        logger.error("Audio output is all zeros (silent)",
                   extra={'subsys': 'tts', 'event': 'gibberish.silent'})
        raise TTSGibberishError("Audio output is completely silent (all zeros)", metrics)
    
    # Calculate metrics
    avg_abs = np.mean(np.abs(audio))
    metrics['avg_abs'] = float(avg_abs)
    
    # Root mean square (volume)
    rms = np.sqrt(np.mean(np.square(audio)))
    metrics['rms'] = float(rms)
    
    # Zero crossing rate (rough measure of noise vs. speech)
    zero_crossings = np.sum(np.abs(np.diff(np.signbit(audio).astype(int))))
    zcr = zero_crossings / (len(audio) - 1)
    metrics['zcr'] = float(zcr)
    
    # Log metrics for debugging
    logger.debug(f"Audio metrics: avg_abs={avg_abs:.6f}, rms={rms:.6f}, zcr={zcr:.6f}",
               extra={'subsys': 'tts', 'event': 'gibberish.metrics', 
                      'avg_abs': avg_abs, 'rms': rms, 'zcr': zcr})
    
    # Detect gibberish using heuristics
    is_gibberish = False
    
    # Very low volume is suspicious
    if avg_abs < 1e-4:
        is_gibberish = True
        logger.error(f"Audio output has very low volume: avg_abs={avg_abs:.6f}",
                   extra={'subsys': 'tts', 'event': 'gibberish.low_volume'})
    
    # Extremely high zero-crossing rate often indicates noise
    if zcr > 0.4:  # This threshold may need tuning
        is_gibberish = True
        logger.error(f"Audio output has abnormally high zero-crossing rate: zcr={zcr:.6f}",
                   extra={'subsys': 'tts', 'event': 'gibberish.high_zcr'})
    
    if is_gibberish:
        raise TTSGibberishError("Audio output detected as gibberish", metrics)
    
    return False, metrics
