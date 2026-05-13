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
    assert router._extract_x_api_primary_text({"data": [{"text": "list text"}]}) == "list text"
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


def test_format_x_caption_only_fallback_result_emits_event_and_delegates(
    monkeypatch,
) -> None:
    router = Router(DummyBot())
    router.logger = CaptureLogger()
    captured = {}

    def _fmt(self, *, url="", base_text=None, tweet_text=None, api_data=None):
        captured["url"] = url
        captured["base_text"] = base_text
        captured["tweet_text"] = tweet_text
        captured["api_data"] = api_data
        return "ok"

    monkeypatch.setattr(Router, "_format_x_caption_only_transcription", _fmt)

    out = router._format_x_caption_only_fallback_result(
        url="https://x.com/u/status/1",
        base_text="base",
        tweet_text="tweet",
        api_data={"data": {"text": "api"}},
    )

    assert out == "ok"
    assert router.logger.info_lines == ["fallback"]
    assert captured == {
        "url": "https://x.com/u/status/1",
        "base_text": "base",
        "tweet_text": "tweet",
        "api_data": {"data": {"text": "api"}},
    }


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


def test_format_x_video_stt_probe_result_returns_formatted_when_transcription_present(
    monkeypatch,
) -> None:
    router = Router(DummyBot())

    monkeypatch.setattr(
        Router,
        "_format_x_transcription_if_present",
        lambda _self, **_kwargs: "formatted",
    )
    monkeypatch.setattr(
        Router,
        "_emit_stt_fail_event",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("emit should not be called on success")),
    )
    monkeypatch.setattr(
        Router,
        "_format_x_video_stt_error_result",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("error formatter should not be called on success")),
    )

    out = router._format_x_video_stt_probe_result(
        url="https://x.com/u/status/1",
        base_text="base",
        tweet_text="tweet",
        stt_res={"transcription": "hello"},
        stt_err=None,
        emit_fail_event=True,
        msg_id=123,
    )
    assert out == "formatted"


def test_format_x_video_stt_probe_result_emits_and_formats_error_when_missing_transcription(
    monkeypatch,
) -> None:
    router = Router(DummyBot())
    captured = {}

    monkeypatch.setattr(
        Router,
        "_format_x_transcription_if_present",
        lambda _self, **_kwargs: None,
    )
    monkeypatch.setattr(
        Router,
        "_classify_stt_error_reason",
        lambda _self, _err: "classified",
    )

    def _emit(self, reason, media_kind=None, msg_id=None):
        captured["emit"] = (reason, media_kind, msg_id)

    def _fmt_error(self, **kwargs):
        captured["error_kwargs"] = kwargs
        return "error-formatted"

    monkeypatch.setattr(Router, "_emit_stt_fail_event", _emit)
    monkeypatch.setattr(Router, "_format_x_video_stt_error_result", _fmt_error)

    out = router._format_x_video_stt_probe_result(
        url="https://x.com/u/status/1",
        base_text="base",
        tweet_text="tweet",
        stt_res={},
        stt_err="error",
        emit_fail_event=True,
        fail_media_kind="video",
        msg_id=123,
    )

    assert out == "error-formatted"
    assert captured["emit"] == ("classified", "video", 123)
    assert captured["error_kwargs"] == {
        "url": "https://x.com/u/status/1",
        "stt_error": "error",
        "base_text": "base",
        "tweet_text": "tweet",
    }


@pytest.mark.asyncio
async def test_format_x_with_resolved_base_text_delegates(monkeypatch) -> None:
    router = Router(DummyBot())
    captured = {}

    async def _resolve_base(self, url):
        captured["resolved_url"] = url
        return "resolved base"

    def _fmt(self, *, base_text=None, url="", stt_res=None, **kwargs):
        captured["base_text"] = base_text
        captured["format_url"] = url
        captured["stt_res"] = stt_res
        return "formatted"

    monkeypatch.setattr(Router, "_resolve_x_base_text_for_url", _resolve_base)
    monkeypatch.setattr(Router, "_format_x_tweet_with_transcription", _fmt)

    out = await router._format_x_with_resolved_base_text(
        url="https://x.com/u/status/1",
        stt_res={"transcription": "hello"},
    )

    assert out == "formatted"
    assert captured["resolved_url"] == "https://x.com/u/status/1"
    assert captured["base_text"] == "resolved base"
    assert captured["format_url"] == "https://x.com/u/status/1"
    assert captured["stt_res"] == {"transcription": "hello"}


@pytest.mark.asyncio
async def test_format_x_with_resolved_base_text_if_available_formats_when_present(
    monkeypatch,
) -> None:
    router = Router(DummyBot())
    captured = {}

    async def _resolve_base(self, url):
        captured["resolved_url"] = url
        return "resolved base"

    def _fmt(self, *, base_text=None, url="", stt_res=None, **kwargs):
        captured["base_text"] = base_text
        captured["format_url"] = url
        captured["stt_res"] = stt_res
        return "formatted"

    monkeypatch.setattr(Router, "_resolve_x_base_text_for_url", _resolve_base)
    monkeypatch.setattr(Router, "_format_x_tweet_with_transcription", _fmt)

    out = await router._format_x_with_resolved_base_text_if_available(
        url="https://x.com/u/status/1",
        stt_res={"transcription": ""},
    )

    assert out == "formatted"
    assert captured["resolved_url"] == "https://x.com/u/status/1"
    assert captured["base_text"] == "resolved base"
    assert captured["format_url"] == "https://x.com/u/status/1"
    assert captured["stt_res"] == {"transcription": ""}


@pytest.mark.asyncio
async def test_format_x_with_resolved_base_text_if_available_returns_none_when_missing(
    monkeypatch,
) -> None:
    router = Router(DummyBot())

    async def _resolve_base(self, _url):
        return ""

    monkeypatch.setattr(Router, "_resolve_x_base_text_for_url", _resolve_base)
    monkeypatch.setattr(
        Router,
        "_format_x_tweet_with_transcription",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("formatter should not be called when base text is empty")),
    )

    out = await router._format_x_with_resolved_base_text_if_available(
        url="https://x.com/u/status/1",
        stt_res={"transcription": ""},
    )
    assert out is None


@pytest.mark.asyncio
async def test_format_x_no_speech_fallback_resolves_base_and_emits_breadcrumbs(
    monkeypatch,
) -> None:
    router = Router(DummyBot())
    router.logger = CaptureLogger()
    captured = {}

    async def _fmt_resolved(self, *, url, stt_res):
        captured["url"] = url
        captured["stt_res"] = stt_res
        return "formatted"

    monkeypatch.setattr(Router, "_format_x_with_resolved_base_text", _fmt_resolved)

    out = await router._format_x_no_speech_fallback(
        url="https://x.com/u/status/1",
        stt_res=None,
    )

    assert out == "formatted"
    assert captured == {"url": "https://x.com/u/status/1", "stt_res": {}}
    assert router.logger.info_lines == ["stt.fail", "fallback"]
    assert router.logger.calls[0]["extra"]["detail"]["reason"] == "no_speech"


@pytest.mark.asyncio
async def test_format_x_no_speech_fallback_uses_explicit_base_without_resolve(
    monkeypatch,
) -> None:
    router = Router(DummyBot())
    router.logger = CaptureLogger()
    captured = {}

    async def _fmt_resolved(self, *, url, stt_res):
        raise AssertionError("resolved-base formatter should not be called")

    def _fmt(self, *, base_text=None, url="", stt_res=None, **kwargs):
        captured["base_text"] = base_text
        captured["url"] = url
        captured["stt_res"] = stt_res
        return "formatted"

    monkeypatch.setattr(Router, "_format_x_with_resolved_base_text", _fmt_resolved)
    monkeypatch.setattr(Router, "_format_x_tweet_with_transcription", _fmt)

    out = await router._format_x_no_speech_fallback(
        url="https://x.com/u/status/1",
        stt_res={"transcription": ""},
        base_text="base",
    )

    assert out == "formatted"
    assert captured == {
        "base_text": "base",
        "url": "https://x.com/u/status/1",
        "stt_res": {"transcription": ""},
    }
    assert router.logger.info_lines == ["stt.fail", "fallback"]
    assert router.logger.calls[0]["extra"]["detail"]["reason"] == "no_speech"


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
async def test_route_twitter_syndication_to_vl_delegates_to_handler(
    monkeypatch,
) -> None:
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


@pytest.mark.asyncio
async def test_route_twitter_images_with_caption_builds_payload_and_routes(
    monkeypatch,
) -> None:
    router = Router(DummyBot())
    calls = {}

    def _build_payload(self, text, image_urls):
        calls["build"] = (text, image_urls)
        return {"text": text, "photos": [{"url": u} for u in image_urls]}

    async def _route_payload(self, syn_payload, url):
        calls["route"] = (syn_payload, url)
        return "ok"

    monkeypatch.setattr(Router, "_build_syndication_photo_payload", _build_payload)
    monkeypatch.setattr(Router, "_route_twitter_syndication_to_vl", _route_payload)

    out = await router._route_twitter_images_with_caption(
        url="https://x.com/u/status/1",
        caption_text="caption",
        image_urls=["u1", "u2"],
    )

    assert out == "ok"
    assert calls["build"] == ("caption", ["u1", "u2"])
    assert calls["route"] == (
        {"text": "caption", "photos": [{"url": "u1"}, {"url": "u2"}]},
        "https://x.com/u/status/1",
    )


@pytest.mark.asyncio
async def test_route_probed_twitter_images_with_caption_logs_resolves_and_routes(
    monkeypatch,
) -> None:
    router = Router(DummyBot())
    calls = {}

    def _log_images(self, image_urls, msg_id=None):
        calls["log"] = (image_urls, msg_id)

    async def _resolve_text(self, status_id):
        calls["resolve"] = status_id
        return "caption"

    async def _route_images(self, *, url, caption_text, image_urls):
        calls["route"] = (url, caption_text, image_urls)
        return "ok"

    monkeypatch.setattr(Router, "_log_twitter_syndication_images", _log_images)
    monkeypatch.setattr(Router, "_resolve_twitter_caption_text", _resolve_text)
    monkeypatch.setattr(Router, "_route_twitter_images_with_caption", _route_images)

    out = await router._route_probed_twitter_images_with_caption(
        url="https://x.com/u/status/1",
        status_id="123",
        image_urls=["u1", "u2"],
    )

    assert out == "ok"
    assert calls["log"] == (["u1", "u2"], None)
    assert calls["resolve"] == "123"
    assert calls["route"] == ("https://x.com/u/status/1", "caption", ["u1", "u2"])


@pytest.mark.asyncio
async def test_resolve_and_probe_twitter_images_delegates_to_resolver_and_probe(
    monkeypatch,
) -> None:
    router = Router(DummyBot())
    calls = {}

    def _resolve_status(self, url, tweet_id=None):
        calls["resolve"] = (url, tweet_id)
        return "123"

    async def _probe_images(self, url, status_id):
        calls["probe"] = (url, status_id)
        return ["u1", "u2"]

    monkeypatch.setattr(Router, "_resolve_twitter_status_id", _resolve_status)
    monkeypatch.setattr(Router, "_probe_twitter_syndication_images", _probe_images)

    status_id, image_urls = await router._resolve_and_probe_twitter_images(
        url="https://x.com/u/status/1",
        tweet_id="hint",
    )

    assert status_id == "123"
    assert image_urls == ["u1", "u2"]
    assert calls["resolve"] == ("https://x.com/u/status/1", "hint")
    assert calls["probe"] == ("https://x.com/u/status/1", "123")


def test_log_twitter_syndication_images_with_and_without_msg_id() -> None:
    router = Router(DummyBot())
    router.logger = CaptureLogger()

    router._log_twitter_syndication_images(
        ["https://pbs.twimg.com/media/abc.jpg"],
        msg_id=123,
    )
    router._log_twitter_syndication_images(["not a url"])

    assert router.logger.info_lines[0] == "route.twitter.syndication | images=1 | pbs.twimg.com | msg_id=123"
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


def test_resolve_twitter_status_id_prefers_explicit_hint(monkeypatch) -> None:
    router = Router(DummyBot())

    monkeypatch.setattr(
        Router,
        "_parse_twitter_status_id",
        lambda _self, _url: (_ for _ in ()).throw(AssertionError("parser should not be called when tweet_id is provided")),
    )

    assert (
        router._resolve_twitter_status_id(
            "https://x.com/user/status/111",
            tweet_id="123",
        )
        == "123"
    )


def test_resolve_twitter_status_id_falls_back_to_parser(monkeypatch) -> None:
    router = Router(DummyBot())

    monkeypatch.setattr(
        Router,
        "_parse_twitter_status_id",
        lambda _self, _url: "456",
    )
    assert router._resolve_twitter_status_id("https://x.com/user/status/111") == "456"

    monkeypatch.setattr(
        Router,
        "_parse_twitter_status_id",
        lambda _self, _url: None,
    )
    assert router._resolve_twitter_status_id("https://x.com/user/status/111") == ""


def test_x_syn_probe_budget_timeout_caps_at_four_point_five() -> None:
    router = Router(DummyBot())
    router._x_syn_timeout_s = 9.0

    assert router._x_syn_probe_budget_timeout_s() == 4.5


def test_x_syn_probe_budget_timeout_uses_syn_timeout_plus_one() -> None:
    router = Router(DummyBot())
    router._x_syn_timeout_s = 2.2

    assert router._x_syn_probe_budget_timeout_s() == 3.2


def test_stt_result_has_transcription_matches_existing_truthiness() -> None:
    router = Router(DummyBot())

    assert router._stt_result_has_transcription({"transcription": "hello"}) is True
    # Preserve existing bool() semantics used in call sites.
    assert router._stt_result_has_transcription({"transcription": "   "}) is True
    assert router._stt_result_has_transcription({"transcription": ""}) is False
    assert router._stt_result_has_transcription({"text": "fallback only"}) is False
    assert router._stt_result_has_transcription(None) is False


def test_extract_sparse_media_resolution_defaults_and_sanitizes() -> None:
    router = Router(DummyBot())

    assert router._extract_sparse_media_resolution(None, default_url="https://x.com/a") == (
        "unknown",
        [],
        "https://x.com/a",
    )

    assert router._extract_sparse_media_resolution(
        {"kind": "video", "images": "bad", "url": ""},
        default_url="https://x.com/b",
    ) == ("video", [], "https://x.com/b")

    assert router._extract_sparse_media_resolution(
        {"kind": "", "images": ["i1"], "url": "https://x.com/c"},
        default_url="https://x.com/d",
    ) == ("unknown", ["i1"], "https://x.com/c")


def test_format_x_transcription_if_present_returns_formatted_output(
    monkeypatch,
) -> None:
    router = Router(DummyBot())
    captured = {}

    def _fmt(self, *, base_text=None, url="", stt_res=None, **kwargs):
        captured["base_text"] = base_text
        captured["url"] = url
        captured["stt_res"] = stt_res
        return "formatted"

    monkeypatch.setattr(Router, "_format_x_tweet_with_transcription", _fmt)

    out = router._format_x_transcription_if_present(
        base_text="base",
        url="https://x.com/u/status/1",
        stt_res={"transcription": "hello"},
    )

    assert out == "formatted"
    assert captured["base_text"] == "base"
    assert captured["url"] == "https://x.com/u/status/1"
    assert captured["stt_res"] == {"transcription": "hello"}


def test_format_x_transcription_if_present_returns_none_without_transcription(
    monkeypatch,
) -> None:
    router = Router(DummyBot())

    monkeypatch.setattr(
        Router,
        "_format_x_tweet_with_transcription",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("formatter should not be called")),
    )

    out = router._format_x_transcription_if_present(
        base_text="base",
        url="https://x.com/u/status/1",
        stt_res={},
    )
    assert out is None


@pytest.mark.asyncio
async def test_maybe_hydrate_syndication_payload_no_tweet_id_returns_input() -> None:
    router = Router(DummyBot())
    payload = {"text": "x"}

    out = await router._maybe_hydrate_syndication_payload("", payload)

    assert out is payload


@pytest.mark.asyncio
async def test_maybe_hydrate_syndication_payload_non_dict_returns_input() -> None:
    router = Router(DummyBot())
    payload = "raw"

    out = await router._maybe_hydrate_syndication_payload("123", payload)

    assert out == "raw"


@pytest.mark.asyncio
async def test_maybe_hydrate_syndication_payload_calls_hydrator(monkeypatch) -> None:
    router = Router(DummyBot())
    payload = {"text": "x"}
    called = {}

    async def _hydrate(self, tweet_id, syn, allow_tco_pointer=False):
        called["tweet_id"] = tweet_id
        called["syn"] = syn
        called["allow_tco_pointer"] = allow_tco_pointer
        return {"text": "hydrated"}

    monkeypatch.setattr(Router, "_hydrate_syndication_article_if_needed", _hydrate)

    out = await router._maybe_hydrate_syndication_payload(
        "123",
        payload,
        allow_tco_pointer=True,
    )

    assert out == {"text": "hydrated"}
    assert called["tweet_id"] == "123"
    assert called["syn"] is payload
    assert called["allow_tco_pointer"] is True


@pytest.mark.asyncio
async def test_resolve_syndication_caption_from_payload_hydrates_and_extracts(
    monkeypatch,
) -> None:
    router = Router(DummyBot())
    called = {}

    async def _maybe_hydrate(self, tweet_id, payload, allow_tco_pointer=False):
        called["tweet_id"] = tweet_id
        called["payload"] = payload
        called["allow_tco_pointer"] = allow_tco_pointer
        return {"text": "hydrated"}

    monkeypatch.setattr(Router, "_maybe_hydrate_syndication_payload", _maybe_hydrate)
    monkeypatch.setattr(Router, "_extract_syndication_text", lambda _s, n: n.get("text", ""))

    out = await router._resolve_syndication_caption_from_payload(
        "123",
        {"text": "raw"},
        fallback_text="fallback",
    )

    assert out == "hydrated"
    assert called["tweet_id"] == "123"
    assert called["payload"] == {"text": "raw"}
    assert called["allow_tco_pointer"] is True


@pytest.mark.asyncio
async def test_resolve_syndication_caption_from_payload_returns_fallback_on_miss_or_error(
    monkeypatch,
) -> None:
    router = Router(DummyBot())

    out_non_dict = await router._resolve_syndication_caption_from_payload(
        "123",
        "bad",
        fallback_text="fallback",
    )
    assert out_non_dict == "fallback"

    async def _maybe_hydrate_raises(self, _tweet_id, _payload, allow_tco_pointer=False):
        raise RuntimeError("boom")

    monkeypatch.setattr(Router, "_maybe_hydrate_syndication_payload", _maybe_hydrate_raises)

    out_error = await router._resolve_syndication_caption_from_payload(
        "123",
        {"text": "raw"},
        fallback_text="fallback",
    )
    assert out_error == "fallback"


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


@pytest.mark.asyncio
async def test_resolve_twitter_caption_from_syndication_prefers_hydrated_text(
    monkeypatch,
) -> None:
    router = Router(DummyBot())

    async def _get_syn(self, status_id):
        assert status_id == "123"
        return {"text": "raw"}

    async def _maybe_hydrate(self, tweet_id, payload, allow_tco_pointer=False):
        assert tweet_id == "123"
        assert payload == {"text": "raw"}
        assert allow_tco_pointer is True
        return {"text": "hydrated"}

    monkeypatch.setattr(Router, "_get_tweet_via_syndication", _get_syn)
    monkeypatch.setattr(Router, "_maybe_hydrate_syndication_payload", _maybe_hydrate)
    monkeypatch.setattr(Router, "_extract_syndication_text", lambda _s, n: n.get("text", ""))

    out = await router._resolve_twitter_caption_from_syndication(
        "123",
        fallback_text="fallback",
    )

    assert out == "hydrated"


@pytest.mark.asyncio
async def test_resolve_twitter_caption_from_syndication_returns_fallback_on_miss(
    monkeypatch,
) -> None:
    router = Router(DummyBot())

    async def _get_syn(self, _status_id):
        return {}

    async def _maybe_hydrate(self, _tweet_id, payload, allow_tco_pointer=False):
        return payload

    monkeypatch.setattr(Router, "_get_tweet_via_syndication", _get_syn)
    monkeypatch.setattr(Router, "_maybe_hydrate_syndication_payload", _maybe_hydrate)
    monkeypatch.setattr(Router, "_extract_syndication_text", lambda _s, _n: "")

    out = await router._resolve_twitter_caption_from_syndication(
        "123",
        fallback_text="fallback",
    )
    assert out == "fallback"

    out_empty = await router._resolve_twitter_caption_from_syndication(
        "",
        fallback_text="fallback",
    )
    assert out_empty == "fallback"


@pytest.mark.asyncio
async def test_resolve_twitter_caption_from_syndication_default_fallback_empty(
    monkeypatch,
) -> None:
    router = Router(DummyBot())

    async def _get_syn(self, _status_id):
        return {}

    monkeypatch.setattr(Router, "_get_tweet_via_syndication", _get_syn)
    monkeypatch.setattr(Router, "_extract_syndication_text", lambda _s, _n: "")

    out = await router._resolve_twitter_caption_from_syndication("123")
    assert out == ""
