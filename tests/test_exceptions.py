from bot.exceptions import (
    APIError,
    BackendError,
    BotBaseException,
    BotError,
    PersistenceError,
    VisionError,
)
from bot.memory.persistence import PersistenceError as MemoryPersistenceError
from bot.vision.types import VisionError as VisionTypesError
from bot.vision.types import VisionErrorType


def test_exception_base_compatibility_aliases():
    assert issubclass(BotError, BotBaseException)
    assert issubclass(APIError, BackendError)
    assert issubclass(APIError, BotError)


def test_persistence_error_is_canonical_import():
    assert MemoryPersistenceError is PersistenceError
    assert issubclass(MemoryPersistenceError, BotError)


def test_vision_error_is_reexported_from_vision_types():
    assert VisionTypesError is VisionError
    error = VisionTypesError(
        error_type=VisionErrorType.PROVIDER_ERROR,
        message="provider failed",
        user_message="try again later",
    )
    assert isinstance(error, BotError)
    assert error.error_type is VisionErrorType.PROVIDER_ERROR
    assert str(error) == "provider_error: provider failed"