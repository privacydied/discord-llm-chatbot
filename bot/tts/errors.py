"""Custom exceptions for the TTS system."""


class TTSError(Exception):
    """Base class for TTS errors."""


class SynthesisError(TTSError):
    """Error during audio synthesis."""
