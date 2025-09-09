"""
Custom exceptions for the Discord bot, providing a structured error hierarchy.
"""


class BotBaseException(Exception):
    """Base exception for all custom exceptions in this bot."""

    pass


class ConfigurationError(BotBaseException):
    """Raised for errors in bot configuration, like missing keys or invalid values."""

    pass


class APIError(BotBaseException):
    """Raised for errors related to external API interactions (Discord, Ollama, etc.)."""

    pass


class InferenceError(BotBaseException):
    """Raised for errors during model inference (text, vision, etc.)."""

    pass


class TTSAudioError(BotBaseException):
    """Raised for errors during Text-to-Speech audio synthesis or processing."""

    pass


class FileProcessingError(BotBaseException):
    """Raised for errors when processing user-uploaded files."""

    pass


class DispatchEmptyError(BotBaseException):
    """Raised when a dispatch returns no result (violates 1 IN > 1 OUT)."""

    pass


class DispatchTypeError(BotBaseException):
    """Raised when a dispatch returns an invalid type."""

    pass
