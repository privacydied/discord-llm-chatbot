"""Custom exceptions for TTS system."""

class TTSWriteError(Exception):
    """Exception raised when TTS audio file writing fails."""
    pass

class TTSSynthesisError(Exception):
    """Exception raised when TTS synthesis fails (e.g., silent audio, model error)."""
    pass
