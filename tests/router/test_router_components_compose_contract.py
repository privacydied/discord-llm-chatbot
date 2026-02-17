from bot.router_components.compose import (
    compose_x_tweet_with_visual_facts,
    format_x_tweet_with_transcription,
)


def test_contract_caption_plus_transcript_collapses_to_combined_section() -> None:
    out = format_x_tweet_with_transcription(
        base_text=None,
        url="https://x.com/u/status/1",
        stt_res={"transcription": "spoken words"},
        tweet_data={"text": "tweet caption"},
        extract_primary_tweet_id=lambda _u: "1",
    )
    assert "[Tweet Caption + Audio Transcript]" in out
    assert "tweet caption" in out
    assert "spoken words" in out
    assert "[Tweet Caption]" not in out
    assert "[Audio Transcript]" not in out


def test_contract_caption_only_preserved_when_no_transcript() -> None:
    out = format_x_tweet_with_transcription(
        base_text=None,
        url="https://x.com/u/status/1",
        stt_res={"transcription": ""},
        tweet_data={"text": "tweet caption"},
        extract_primary_tweet_id=lambda _u: "1",
    )
    assert "[Tweet Caption]" in out
    assert "tweet caption" in out


def test_contract_transcript_only_when_no_caption() -> None:
    out = format_x_tweet_with_transcription(
        base_text=None,
        url="https://x.com/u/status/1",
        stt_res={"transcription": "spoken words"},
        tweet_data=None,
        extract_primary_tweet_id=lambda _u: "1",
    )
    assert "[Audio Transcript]" in out
    assert "spoken words" in out


def test_contract_visual_facts_includes_caption_and_vl_blocks() -> None:
    out = compose_x_tweet_with_visual_facts(
        user_text="analyze this",
        tweet_caption="caption body",
        vl_notes="visual notes",
    )
    assert "VISUAL_FACTS:" in out
    assert "tweet caption:" in out
    assert "caption body" in out
    assert "vl prompt output:" in out
    assert "visual notes" in out
