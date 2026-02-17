from bot.router_components.compose import (
    compose_x_tweet_with_visual_facts,
    format_x_tweet_with_transcription,
)


def test_format_x_tweet_with_transcription_combines_caption_and_transcript() -> None:
    result = format_x_tweet_with_transcription(
        base_text=None,
        url="https://x.com/alx/status/123",
        stt_res={"transcription": "audio words"},
        tweet_data={"text": "tweet caption"},
        extract_primary_tweet_id=lambda _url: "123",
    )

    assert "[Tweet Caption + Audio Transcript]" in result
    assert "tweet caption" in result
    assert "audio words" in result


def test_compose_x_tweet_with_visual_facts_returns_user_text_when_empty() -> None:
    result = compose_x_tweet_with_visual_facts(
        user_text="what do you think?",
        tweet_caption="",
        vl_notes="",
    )
    assert result == "what do you think?"


def test_compose_x_tweet_with_visual_facts_includes_caption_and_vl() -> None:
    result = compose_x_tweet_with_visual_facts(
        user_text="analyze this",
        tweet_caption="tweet body",
        vl_notes="image notes",
    )

    assert "analyze this" in result
    assert "VISUAL_FACTS:" in result
    assert "tweet caption:" in result
    assert "tweet body" in result
    assert "vl prompt output:" in result
    assert "image notes" in result
