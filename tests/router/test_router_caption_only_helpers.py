import pytest

from bot.router import Router


class DummyBot:
    def __init__(self):
        self.config = {}
        self.tts_manager = None
        self.loop = None


class CaptureLogger:
    def __init__(self):
        self.info_lines = []
        self.calls = []

    def info(self, message, *args, **kwargs):
        if args:
            message = message % args
        text = str(message)
        self.info_lines.append(text)
        self.calls.append({"message": text, "extra": kwargs.get("extra")})

    def debug(self, *args, **kwargs):
        return None


def test_extract_x_api_primary_text_handles_dict_and_list_payloads() -> None:
    router = Router(DummyBot())

    assert router._extract_x_api_primary_text({"data": {"text": "dict text"}}) == "dict text"
    assert (
        router._extract_x_api_primary_text({"data": [{"text": "list text"}]})
        == "list text"
    )
    assert router._extract_x_api_primary_text({"data": []}) == ""
    assert router._extract_x_api_primary_text(None) == ""


def test_format_x_caption_only_transcription_prefers_api_text(monkeypatch) -> None:
    router = Router(DummyBot())
    captured = {}

    def _fmt(self, *, base_text=None, url="", stt_res=None, **kwargs):
        captured["base_text"] = base_text
        captured["url"] = url
        captured["stt_res"] = stt_res
        return "ok"

    monkeypatch.setattr(Router, "_format_x_tweet_with_transcription", _fmt)

    out = router._format_x_caption_only_transcription(
        url="https://x.com/u/status/1",
        base_text="base text",
        tweet_text="tweet text",
        api_data={"data": {"text": "api text"}},
    )

    assert out == "ok"
    assert captured["base_text"] == "api text"
    assert captured["url"] == "https://x.com/u/status/1"
    assert captured["stt_res"] == {}


def test_format_x_caption_only_transcription_falls_back_to_tweet_then_base(
    monkeypatch,
) -> None:
    router = Router(DummyBot())
    captured = {}

    def _fmt(self, *, base_text=None, url="", stt_res=None, **kwargs):
        captured["base_text"] = base_text
        captured["stt_res"] = stt_res
        return "ok"

    monkeypatch.setattr(Router, "_format_x_tweet_with_transcription", _fmt)

    router._format_x_caption_only_transcription(
        url="https://x.com/u/status/1",
        base_text="base text",
        tweet_text="tweet text",
        api_data={"data": {"text": ""}},
    )
    assert captured["base_text"] == "tweet text"
    assert captured["stt_res"] == {}

    router._format_x_caption_only_transcription(
        url="https://x.com/u/status/1",
        base_text="base text",
        tweet_text="",
        api_data={},
    )
    assert captured["base_text"] == "base text"
    assert captured["stt_res"] == {}


def test_emit_caption_only_fallback_event_logs_fallback_only() -> None:
    router = Router(DummyBot())
    router.logger = CaptureLogger()

    router._emit_caption_only_fallback_event()

    assert router.logger.info_lines == ["fallback"]


def test_emit_stt_fail_event_populates_extra_fields() -> None:
    router = Router(DummyBot())
    router.logger = CaptureLogger()

    router._emit_stt_fail_event(
        "no_speech",
        media_kind="video",
        msg_id=123,
    )

    assert router.logger.info_lines == ["stt.fail"]
    extra = router.logger.calls[0]["extra"]
    assert isinstance(extra, dict)
    assert extra["event"] == "stt.fail"
    assert extra["detail"]["reason"] == "no_speech"
    assert extra["detail"]["media_kind"] == "video"
    assert extra["msg_id"] == 123


def test_emit_caption_only_fallback_breadcrumbs_emits_stt_fail_and_fallback() -> None:
    router = Router(DummyBot())
    router.logger = CaptureLogger()

    router._emit_caption_only_fallback_breadcrumbs("error")

    assert router.logger.info_lines == ["stt.fail", "fallback"]
    stt_extra = router.logger.calls[0]["extra"]
    fallback_extra = router.logger.calls[1]["extra"]
    assert stt_extra["detail"]["reason"] == "error"
    assert fallback_extra["detail"]["kind"] == "caption_only"


def test_format_x_video_stt_error_result_preserves_video_context(monkeypatch) -> None:
    router = Router(DummyBot())
    captured = {}

    def _fmt(self, *, base_text=None, url="", stt_res=None, **kwargs):
        captured["base_text"] = base_text
        captured["url"] = url
        captured["stt_res"] = stt_res
        return "ok"

    monkeypatch.setattr(Router, "_format_x_tweet_with_transcription", _fmt)

    out = router._format_x_video_stt_error_result(
        url="https://x.com/u/status/1",
        stt_error="error",
        base_text="base text",
        tweet_text="tweet text",
    )

    assert out == "ok"
    assert captured["base_text"] == "tweet text"
    assert captured["url"] == "https://x.com/u/status/1"
    assert captured["stt_res"]["transcription"] is None
    assert captured["stt_res"]["error"] == "error"
    assert captured["stt_res"]["media_kind"] == "video"
    assert captured["stt_res"]["url"] == "https://x.com/u/status/1"


def test_format_x_video_stt_error_result_defaults_error(monkeypatch) -> None:
    router = Router(DummyBot())
    captured = {}

    def _fmt(self, *, base_text=None, url="", stt_res=None, **kwargs):
        captured["base_text"] = base_text
        captured["stt_res"] = stt_res
        return "ok"

    monkeypatch.setattr(Router, "_format_x_tweet_with_transcription", _fmt)

    router._format_x_video_stt_error_result(
        url="https://x.com/u/status/1",
        stt_error="",
        base_text="base text",
        tweet_text="",
    )

    assert captured["base_text"] == "base text"
    assert captured["stt_res"]["error"] == "transcription_failed"


def test_classify_stt_error_reason_matches_existing_semantics() -> None:
    router = Router(DummyBot())

    assert router._classify_stt_error_reason("error") == "error"
    assert router._classify_stt_error_reason(None) == "no_speech"
    assert router._classify_stt_error_reason("timeout") == "no_speech"
    # Preserve legacy exact-match behavior (case-sensitive)
    assert router._classify_stt_error_reason("ERROR") == "no_speech"


def test_extract_x_api_primary_tweet_handles_payload_variants() -> None:
    router = Router(DummyBot())

    assert router._extract_x_api_primary_tweet({"data": {"id": "1"}}) == {"id": "1"}
    assert router._extract_x_api_primary_tweet({"data": [{"id": "2"}]}) == {"id": "2"}
    assert router._extract_x_api_primary_tweet({"data": []}) == {}
    assert router._extract_x_api_primary_tweet({"data": ["bad"]}) == {}
    assert router._extract_x_api_primary_tweet(None) == {}


@pytest.mark.asyncio
async def test_route_twitter_syndication_to_vl_delegates_to_handler(monkeypatch) -> None:
    router = Router(DummyBot())
    captured = {}

    async def _handler(payload, url, pipeline, prompt, reply_style):
        captured["payload"] = payload
        captured["url"] = url
        captured["pipeline"] = pipeline
        captured["prompt"] = prompt
        captured["reply_style"] = reply_style
        return "handled"

    import bot.syndication.handler as syndication_handler

    monkeypatch.setattr(
        syndication_handler,
        "handle_twitter_syndication_to_vl",
        _handler,
    )

    payload = {"text": "caption", "photos": [{"url": "https://pbs.twimg.com/a.jpg"}]}
    out = await router._route_twitter_syndication_to_vl(
        payload,
        "https://x.com/u/status/1",
    )

    assert out == "handled"
    assert captured["payload"] == payload
    assert captured["url"] == "https://x.com/u/status/1"
    assert captured["pipeline"] == router._unified_vl_to_text_pipeline
    assert captured["prompt"] == router._get_system_prompt("vl_prompt")
    assert captured["reply_style"] == "ack+thoughts"


def test_log_twitter_syndication_images_with_and_without_msg_id() -> None:
    router = Router(DummyBot())
    router.logger = CaptureLogger()

    router._log_twitter_syndication_images(
        ["https://pbs.twimg.com/media/abc.jpg"],
        msg_id=123,
    )
    router._log_twitter_syndication_images(["not a url"])

    assert (
        router.logger.info_lines[0]
        == "route.twitter.syndication | images=1 | pbs.twimg.com | msg_id=123"
    )
    assert router.logger.info_lines[1] == "route.twitter.syndication | images=1 | n/a"


def test_build_syndication_photo_payload_shape() -> None:
    router = Router(DummyBot())

    payload = router._build_syndication_photo_payload(
        "caption text",
        ["u1", "u2"],
    )
    assert payload == {
        "text": "caption text",
        "photos": [{"url": "u1"}, {"url": "u2"}],
    }

    payload_none = router._build_syndication_photo_payload(None, [])
    assert payload_none == {"text": None, "photos": []}


def test_build_x_syn_quick_request_config_caps_timeouts() -> None:
    router = Router(DummyBot())
    router._x_syn_timeout_s = 9.0

    cfg = router._build_x_syn_quick_request_config()

    assert cfg.connect_timeout == 3.0
    assert cfg.read_timeout == 3.0
    assert cfg.total_timeout == 3.5
    assert cfg.max_retries == 0


def test_build_x_syn_quick_request_config_uses_lower_timeout() -> None:
    router = Router(DummyBot())
    router._x_syn_timeout_s = 1.2

    cfg = router._build_x_syn_quick_request_config()

    assert cfg.connect_timeout == 1.2
    assert cfg.read_timeout == 1.2
    assert cfg.total_timeout == 1.7
    assert cfg.max_retries == 0


def test_extract_fxtwitter_tweet_node_handles_variants() -> None:
    router = Router(DummyBot())

    assert router._extract_fxtwitter_tweet_node({"tweet": {"id": "1"}}) == {"id": "1"}
    assert router._extract_fxtwitter_tweet_node({"status": {"id": "2"}}) == {"id": "2"}
    assert router._extract_fxtwitter_tweet_node({"tweet": "bad"}) == {}
    assert router._extract_fxtwitter_tweet_node({"status": []}) == {}
    assert router._extract_fxtwitter_tweet_node(None) == {}


@pytest.mark.asyncio
async def test_resolve_twitter_caption_text_prefers_syndication(monkeypatch) -> None:
    router = Router(DummyBot())
    calls = {"hydrated": False}

    async def _get_syn(self, status_id):
        assert status_id == "123"
        return {"text": "from syndication"}

    async def _hydrate(self, status_id, syn, allow_tco_pointer=True):
        calls["hydrated"] = True
        assert status_id == "123"
        assert allow_tco_pointer is True
        return syn

    monkeypatch.setattr(Router, "_get_tweet_via_syndication", _get_syn)
    monkeypatch.setattr(Router, "_hydrate_syndication_article_if_needed", _hydrate)
    monkeypatch.setattr(Router, "_extract_syndication_text", lambda _s, n: n.get("text", ""))
    monkeypatch.setattr(
        "bot.router.get_http_client",
        lambda: (_ for _ in ()).throw(AssertionError("http fallback should not be used")),
    )

    out = await router._resolve_twitter_caption_text("123")

    assert out == "from syndication"
    assert calls["hydrated"] is True


@pytest.mark.asyncio
async def test_resolve_twitter_caption_text_falls_back_to_fx(monkeypatch) -> None:
    router = Router(DummyBot())

    async def _get_syn(self, _status_id):
        return {}

    async def _hydrate(self, _status_id, syn, allow_tco_pointer=True):
        return syn

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {"tweet": {"text": "from fx"}}

    class _Http:
        async def get(self, url, config=None):
            assert "api.fxtwitter.com/status/123" in url
            assert config is not None
            return _Resp()

    async def _get_http():
        return _Http()

    monkeypatch.setattr(Router, "_get_tweet_via_syndication", _get_syn)
    monkeypatch.setattr(Router, "_hydrate_syndication_article_if_needed", _hydrate)
    monkeypatch.setattr(Router, "_extract_syndication_text", lambda _s, n: n.get("text", ""))
    monkeypatch.setattr("bot.router.get_http_client", _get_http)

    out = await router._resolve_twitter_caption_text("123")

    assert out == "from fx"


@pytest.mark.asyncio
async def test_resolve_twitter_caption_text_empty_status_id() -> None:
    router = Router(DummyBot())

    assert await router._resolve_twitter_caption_text("") == ""
    assert await router._resolve_twitter_caption_text(None) == ""
