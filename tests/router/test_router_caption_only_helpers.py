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
