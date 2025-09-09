import abc


class BaseEngine(abc.ABC):
    @abc.abstractmethod
    async def synthesize(self, text: str) -> bytes:
        raise NotImplementedError()
