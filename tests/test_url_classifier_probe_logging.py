import pytest

from bot import url_classifier


@pytest.mark.asyncio
async def test_detect_url_content_type_returns_none_on_403_and_does_not_raise(monkeypatch) -> None:
    class _Resp:
        status_code = 403
        headers = {"content-type": "text/html", "content-length": "123"}

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def head(self, url: str):
            return _Resp()

    def _fake_client(*args, **kwargs):
        return _Client()

    monkeypatch.setattr(url_classifier.httpx, "AsyncClient", _fake_client)

    ct, cl = await url_classifier.detect_url_content_type("https://example.com")

    assert ct is None
    assert cl is None
