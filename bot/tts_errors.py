"""Custom exceptions for TTS system."""

class TTSWriteError(Exception):
    """Exception raised when TTS fails to write output file."""
    pass


class MissingTokeniserError(Exception):
    """Exception raised when no suitable tokeniser is found for a language."""
    pass

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
                "Install one of the following and restart the bot:\n\n"
                "# Arch Linux\n"
                "sudo pacman -Sy espeak-ng\n\n"
                "# Python virtual-env\n"
                "uv pip install phonemizer"
            )

class TTSGibberishError(TTSSynthesisError):
    """Exception raised when TTS output is detected as gibberish or wrong language.
    
    This is a specialized TTSSynthesisError that indicates the audio was generated
    but appears to be gibberish or in the wrong language based on heuristics.
    """
    def __init__(self, message="TTS output detected as gibberish or wrong language", metrics=None):
        self.metrics = metrics or {}
        super().__init__(message)
    
    @property
    def user_message(self):
        """Get a user-friendly error message with suggestions."""
        return (
            "⚠️ The generated voice output appears to be gibberish or in the wrong language. \n\n"
            "Suggestions:\n"
            "• Check that the TTS_LANGUAGE setting matches your voice model\n"
            "• Try a different phonemiser (espeak for English, misaki for Japanese/Chinese)\n"
            "• Choose a different voice\n"
            "• Check that your TTS model and voice files are compatible"
        )
