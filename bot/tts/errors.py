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