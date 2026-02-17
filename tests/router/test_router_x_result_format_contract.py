from bot.router import Router


class DummyBot:
    def __init__(self, *, system_prompts=None):
        self.config = {"X_API_ENABLED": False}
        self.tts_manager = None
        self.loop = None
        if system_prompts is not None:
            self.system_prompts = system_prompts


def test_format_x_tweet_result_handles_wrapped_api_payload() -> None:
    router = Router(DummyBot(system_prompts={"vl_prompt": None}))
    out = router._format_x_tweet_result(
        {
            "data": {"text": "photo post", "author_id": "u1"},
            "includes": {
                "users": [{"id": "u1", "username": "user"}],
                "media": [{"type": "photo"}, {"type": "photo"}],
            },
        },
        "https://twitter.com/user/status/1",
    )

    assert "photo post" in out
    assert "Photos: 2" in out
    assert "— user" in out
    assert "status/1" in out
    assert ("twitter.com" in out) or ("x.com" in out)


def test_format_x_tweet_result_falls_back_to_canonical_url() -> None:
    router = Router(DummyBot(system_prompts={"vl_prompt": None}))
    out = router._format_x_tweet_result(
        {"data": {"text": ""}, "includes": {"users": [], "media": []}},
        "https://twitter.com/user/status/1",
    )
    assert out.endswith("/status/1")
    assert ("twitter.com" in out) or ("x.com" in out)


def test_get_system_prompt_is_safe_when_missing_prompt_map() -> None:
    router = Router(DummyBot())
    assert router._get_system_prompt("vl_prompt", "default-vl") == "default-vl"


def test_get_system_prompt_reads_bot_system_prompts() -> None:
    router = Router(DummyBot(system_prompts={"text_prompt": "be helpful"}))
    assert router._get_system_prompt("text_prompt", "fallback") == "be helpful"
