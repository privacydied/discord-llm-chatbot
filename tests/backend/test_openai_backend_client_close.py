import pytest

from bot.openai_backend import _safe_aclose_openai_client


class BrokenHttpxWrapperLikeClient:
    def __init__(self) -> None:
        self.closed_attempted = False

    async def aclose(self) -> None:
        self.closed_attempted = True
        msg = "'AsyncHttpxClientWrapper' object has no attribute '_transport'"
        raise AttributeError(msg)


class CloseOnlyClient:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_safe_aclose_suppresses_httpx_wrapper_missing_transport() -> None:
    client = BrokenHttpxWrapperLikeClient()

    await _safe_aclose_openai_client(client)

    assert client.closed_attempted is True


@pytest.mark.asyncio
async def test_safe_aclose_supports_sync_close_fallback() -> None:
    client = CloseOnlyClient()

    await _safe_aclose_openai_client(client)

    assert client.closed is True
