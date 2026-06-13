from bot.stt_pipeline.logging import log_stt_job_complete, transcript_preview


class _Logger:
    def __init__(self) -> None:
        self.calls = []

    def info(self, msg, *args) -> None:
        self.calls.append((msg, args))


def test_transcript_preview_truncates_with_ellipsis() -> None:
    text = "a" * 65
    assert transcript_preview(text, limit=60) == ("a" * 60 + "...")


def test_transcript_preview_returns_short_text() -> None:
    assert transcript_preview("hello", limit=60) == "hello"


def test_log_stt_job_complete_uses_canonical_format() -> None:
    logger = _Logger()
    url = "https://x.com/user/status/1234567890"
    transcript_text = "hello world"

    log_stt_job_complete(logger=logger, url=url, transcript_text=transcript_text)

    assert len(logger.calls) == 1
    msg, args = logger.calls[0]
    assert msg == "stt.job.complete url=%s chars=%d preview=%s"
    assert args[0] == url[:80]
    assert args[1] == len(transcript_text)
    assert args[2] == repr("hello world")
