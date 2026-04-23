from bot.router_components.compose import (
    build_visual_analysis_anchor_prompt,
    compose_x_tweet_with_visual_facts,
    format_x_tweet_result,
    format_x_tweet_with_transcription,
    has_visual_facts_section,
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


def test_format_x_tweet_result_handles_wrapped_payload_and_photos() -> None:
    result = format_x_tweet_result(
        api_data={
            "data": {"text": "photo post", "author_id": "u1"},
            "includes": {
                "users": [{"id": "u1", "username": "user"}],
                "media": [{"type": "photo"}, {"type": "photo"}],
            },
        },
        url="https://twitter.com/u/status/1",
        canonicalize_status_url=lambda _u: "https://x.com/i/status/1",
    )

    assert "photo post" in result
    assert "Photos: 2" in result
    assert "— user" in result
    assert "https://x.com/i/status/1" in result


def test_format_x_tweet_result_falls_back_to_canonical_url() -> None:
    result = format_x_tweet_result(
        api_data={"data": {"text": ""}, "includes": {"users": [], "media": []}},
        url="https://twitter.com/u/status/1",
        canonicalize_status_url=lambda _u: "https://x.com/i/status/1",
    )

    assert result == "https://x.com/i/status/1"


def test_has_visual_facts_section_detects_expected_markers() -> None:
    assert has_visual_facts_section("VISUAL_FACTS:\nfoo")
    assert has_visual_facts_section("vl prompt output:\nfoo")
    assert has_visual_facts_section("Image 2: the scene is dark")
    assert has_visual_facts_section("tweet caption:\nfoo")
    assert has_visual_facts_section("🖼️ **Image Analysis (photo.jpg)**\ncat")
    assert has_visual_facts_section("[IMAGE: image.jpg]\nred car")
    assert not has_visual_facts_section("plain text without markers")


def test_build_visual_analysis_anchor_prompt_includes_anchor_block() -> None:
    base = "SYSTEM BASE"
    out = build_visual_analysis_anchor_prompt(base)

    assert out.startswith(base)
    assert "[VISUAL-ANALYSIS-ANCHOR]" in out
    assert "vl prompt output:" in out
    assert "Do not claim there is no image" in out
