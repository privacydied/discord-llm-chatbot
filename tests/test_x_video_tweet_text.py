"""Video tweets: the tweet text leads; a short low-confidence transcript is dropped. [REH]"""

from bot.router_components.compose import format_x_tweet_with_transcription

TWEET = {"text": "My rape victim went to the police but took a shower first so there's no evidence"}  # live 2026-10-25


def _fmt(stt_res, tweet_data=TWEET):
    return format_x_tweet_with_transcription(base_text=None, url="https://x.com/a/status/2114041972293783621", stt_res=stt_res, tweet_data=tweet_data)


def test_silent_clip_hallucination_dropped_tweet_text_kept():
    out = _fmt({"transcription": "You're scared to see this thing!", "confidence_status": "low"})
    assert "took a shower first" in out
    assert "scared to see this thing" not in out and "STT GROUNDING" not in out


def test_confident_speech_kept_with_caption():
    out = _fmt({"transcription": "and that's why the knicks will win it all", "confidence_status": "high"})
    assert "took a shower first" in out and "knicks will win" in out


def test_long_low_confidence_transcript_kept():
    speech = "this is a long noisy recording of someone actually talking about the game " * 3
    assert "noisy recording" in _fmt({"transcription": speech, "confidence_status": "low"})


def test_no_tweet_data_still_works():
    assert "knicks" in _fmt({"transcription": "knicks in six", "confidence_status": "high"}, tweet_data=None)
