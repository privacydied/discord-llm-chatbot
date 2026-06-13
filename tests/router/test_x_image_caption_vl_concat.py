from bot.router import Router


class DummyBot:
    def __init__(self) -> None:
        self.config = {"X_API_ENABLED": False}
        self.tts_manager = None
        self.loop = None
        self.system_prompts = {"vl_prompt": None}


def test_x_image_route_concatenates_caption_and_vl_notes() -> None:
    router = Router(DummyBot())

    result = router._compose_x_tweet_with_visual_facts(
        user_text="what do you think",
        tweet_caption="this is the tweet caption",
        vl_notes="a person is speaking at a podium",
    )

    assert result.startswith("what do you think")
    assert "VISUAL_FACTS:" in result
    assert "tweet caption:\nthis is the tweet caption" in result
    assert "vl prompt output:\na person is speaking at a podium" in result
    assert result.index("tweet caption:") < result.index("vl prompt output:")


def test_x_image_route_uses_caption_placeholder_when_missing() -> None:
    router = Router(DummyBot())

    result = router._compose_x_tweet_with_visual_facts(
        user_text="",
        tweet_caption="",
        vl_notes="detected text on sign: hello",
    )

    assert result.startswith("VISUAL_FACTS:")
    assert "tweet caption:\n—" in result
    assert "vl prompt output:\ndetected text on sign: hello" in result


def test_x_image_route_passthrough_when_no_caption_or_vl_notes() -> None:
    router = Router(DummyBot())

    result = router._compose_x_tweet_with_visual_facts(user_text="  just user text  ", tweet_caption="", vl_notes="")

    assert result == "just user text"
