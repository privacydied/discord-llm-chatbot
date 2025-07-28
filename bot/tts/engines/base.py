import abc
from typing import Any

class BaseEngine(abc.ABC):
    @abc.abstractmethod
    async def synthesize(self, text: str) -> bytes:
        raise NotImplementedError()