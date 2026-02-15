from bot.router import Router


class DummyBot:
    def __init__(self):
        self.config = {"X_API_ENABLED": False}
        self.tts_manager = None
        self.loop = None
        self.system_prompts = {"vl_prompt": None}


def test_x_tweet_caption_and_transcript_are_concatenated():
    router = Router(DummyBot())
    result = router._format_x_tweet_with_transcription(
        base_text="ignored",
        url="https://x.com/user/status/123",
        stt_res={"transcription": "This is a sufficiently long audio transcript."},
        tweet_data={"text": "This is the tweet caption."},
    )

    assert "[Tweet Caption + Audio Transcript]" in result
    assert "This is the tweet caption." in result
    assert "This is a sufficiently long audio transcript." in result
    assert result.index("This is the tweet caption.") < result.index(
        "This is a sufficiently long audio transcript."
    )
    assert "[Tweet Caption]" not in result
    assert "[Audio Transcript]" not in result


def test_x_tweet_transcript_only_keeps_audio_section():
    router = Router(DummyBot())
    result = router._format_x_tweet_with_transcription(
        base_text=None,
        url="https://x.com/user/status/123",
        stt_res={"transcription": "This is a sufficiently long audio transcript."},
        tweet_data=None,
    )

    assert "[Audio Transcript]" in result
    assert "This is a sufficiently long audio transcript." in result


def test_x_tweet_base_text_caption_and_transcript_are_concatenated():
    router = Router(DummyBot())
    base_text = "[Tweet Caption]\nThis caption came from syndication.\n"
    result = router._format_x_tweet_with_transcription(
        base_text=base_text,
        url="https://x.com/user/status/123",
        stt_res={"transcription": "This is a sufficiently long audio transcript."},
        tweet_data=None,
    )

    assert "[Tweet Caption + Audio Transcript]" in result
    assert "This caption came from syndication." in result
    assert "This is a sufficiently long audio transcript." in result
    assert "[Tweet Caption]" not in result
