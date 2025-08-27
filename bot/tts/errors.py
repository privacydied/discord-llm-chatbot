"""Custom exceptions for TTS system."""
from typing import Optional

class TTSError(Exception):
    """Base class for TTS errors"""
    pass

class EngineLoadError(TTSError):
    """Error loading TTS engine"""
    pass

class SynthesisError(TTSError):
    """Error during audio synthesis"""
    pass

class ConfigurationError(TTSError):
    """Invalid TTS configuration"""
    pass

class TTSWriteError(Exception):
    """Exception raised when TTS fails to write output file."""
    pass

class TTSGibberishError(Exception):
    """Raised when synthesized audio is detected as gibberish or invalid.

    Optionally carries a metrics payload with diagnostic information
    (e.g., average amplitude, RMS, ZCR) collected during detection.
    """

    def __init__(self, message: str, metrics: Optional[dict] = None):  # type: ignore[name-defined]
        super().__init__(message)
        # Avoid strict typing import here to keep this a lean errors module
        self.metrics = metrics or {}

class TTSSynthesisError(Exception):
    """Exception raised when TTS synthesis fails (e.g., silent audio, model error)."""
    pass

class MissingTokeniserError(Exception):
    """Exception raised when no suitable tokeniser is found for a language.
    
    This is a critical error that should prevent TTS initialization, as using
    an incorrect tokeniser will result in gibberish output.
    """
    def __init__(self, language="en", available=None, required=None):
        self.language = language
        self.available = available or []
        self.required = required or []
        message = f"No suitable tokeniser found for language '{language}'. Required: {required}, Available: {available}"
        super().__init__(message)
    
    @property
    def user_message(self):
        """Get a user-friendly error message with installation instructions."""
        if self.language.startswith("en"):
            return (
                "⚠ No English phonetic tokeniser (espeak-ng / phonemizer / g2p_en) detected on the server.\n"
                "Install one of them and restart the bot.\n\n"
                "# Arch Linux\n"
                "sudo pacman -Sy espeak-ng\n\n"
                "# Python virtual-env\n"
                "uv pip install phonemizer g2p_en"
            )
        elif self.language.startswith("ja") or self.language.startswith("zh"):
            return (
                "⚠ No Asian language tokeniser (misaki) detected on the server.\n"
                "Install it and restart the bot.\n\n"
                "# Python virtual-env\n"
                "uv pip install misaki"
            )
        else:
            return (
                f"⚠ No suitable tokeniser found for language '{self.language}'.\n"
                "Please install the appropriate tokeniser for your language."
            )