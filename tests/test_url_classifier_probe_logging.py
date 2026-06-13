import pytest

from bot import url_classifier, web


@pytest.mark.asyncio
async def test_detect_url_content_type_returns_none_on_403_and_does_not_raise(
    monkeypatch,
) -> None:
    captured: dict = {}

    class _Resp:
        status_code = 403
        headers = {"content-type": "text/html", "content-length": "123"}

    class _Client:
        def __init__(self, *args, **kwargs) -> None:
            captured["headers"] = kwargs.get("headers")

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def head(self, url: str):
            return _Resp()

    def _fake_client(*args, **kwargs):
        return _Client(*args, **kwargs)

    monkeypatch.setattr(url_classifier.httpx, "AsyncClient", _fake_client)

    ct, cl = await url_classifier.detect_url_content_type("https://example.com")

    assert ct is None
    assert cl is None
    assert captured.get("headers") == url_classifier.URL_HTTP_HEADERS


@pytest.mark.asyncio
async def test_download_url_to_temp_applies_url_http_headers(monkeypatch) -> None:
    captured: dict = {}

    class _StreamResp:
        headers = {"content-length": "3"}

        def raise_for_status(self) -> None:
            return None

        async def aiter_bytes(self, chunk_size: int = 65536):
            yield b"abc"

    class _StreamCtx:
        async def __aenter__(self):
            return _StreamResp()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _Client:
        def __init__(self, *args, **kwargs) -> None:
            captured["headers"] = kwargs.get("headers")

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, method: str, url: str):
            return _StreamCtx()

    monkeypatch.setattr(url_classifier.httpx, "AsyncClient", _Client)

    path, err = await url_classifier.download_url_to_temp("https://example.com/file.pdf")

    assert err is None
    assert path is not None
    assert captured.get("headers") == url_classifier.URL_HTTP_HEADERS
    path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_web_fetch_url_content_applies_browser_like_headers(monkeypatch) -> None:
    captured: dict = {}

    class _Resp:
        headers = {"Content-Type": "text/html"}

        def raise_for_status(self) -> None:
            return None

        async def aread(self) -> bytes:
            return b"hello"

    class _Client:
        def __init__(self, *args, **kwargs) -> None:
            captured["headers"] = kwargs.get("headers")

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url: str):
            return _Resp()

    # Mock httpx.AsyncClient
    monkeypatch.setattr(web.httpx, "AsyncClient", _Client)

    # Mock URL safety validation to allow example.com
    async def _mock_validate_url_with_dns(url: str) -> None:
        return None

    async def _mock_validate_redirect_response(response) -> None:
        return None

    def _mock_is_private_hostname(hostname: str) -> bool:
        return False

    monkeypatch.setattr("bot.url_safety.validate_url_with_dns", _mock_validate_url_with_dns)
    monkeypatch.setattr("bot.url_safety.validate_redirect_response", _mock_validate_redirect_response)
    monkeypatch.setattr("bot.utils.external_api._is_private_hostname", _mock_is_private_hostname)

    payload = await web.fetch_url_content("https://example.com")

    assert payload is not None
    body, content_type = payload
    assert body == b"hello"
    assert content_type == "text/html"
    assert captured.get("headers") is not None
    assert captured["headers"].get("User-Agent")
    assert captured["headers"].get("Accept")
