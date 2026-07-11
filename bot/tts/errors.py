"""Custom exceptions for TTS system."""


# Re-export from ipa_vocab_loader for modules that already import from here


class TTSError(Exception):
    """Base class for TTS errors."""


class EngineLoadError(TTSError):
    """Error loading TTS engine."""


class SynthesisError(TTSError):
    """Error during audio synthesis."""


class ConfigurationError(TTSError):
    """Invalid TTS configuration."""


class TTSWriteError(Exception):
    """Exception raised when TTS fails to write output file."""


class TTSGibberishError(Exception):
    """Raised when synthesized audio is detected as gibberish or invalid.

    Optionally carries a metrics payload with diagnostic information
    (e.g., average amplitude, RMS, ZCR) collected during detection.
    """

    def __init__(self, message: str, metrics: dict | None = None) -> None:  # type: ignore[name-defined]
        super().__init__(message)
        # Avoid strict typing import here to keep this a lean errors module
        self.metrics = metrics or {}


class TTSSynthesisError(Exception):
    """Exception raised when TTS synthesis fails (e.g., silent audio, model error)."""


class MissingTokeniserError(Exception):
    """Exception raised when no suitable tokeniser is found for a language.

    This is a critical error that should prevent TTS initialization, as using
    an incorrect tokeniser will result in gibberish output.
    """

    def __init__(self, language="en", available=None, required=None) -> None:
        self.language = language
        self.available = available or []
        self.required = required or []
        message = f"No suitable tokeniser found for language '{language}'. Required: {required}, Available: {available}"
        super().__init__(message)

    @property
    def user_message(self) -> str:
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
        if self.language.startswith("ja") or self.language.startswith("zh"):
            return "⚠ No Asian language tokeniser (misaki) detected on the server.\nInstall it and restart the bot.\n\n# Python virtual-env\nuv pip install misaki"
        return f"⚠ No suitable tokeniser found for language '{self.language}'.\nPlease install the appropriate tokeniser for your language."
