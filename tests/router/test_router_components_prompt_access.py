from typing import Never

from bot.router_components.prompt_access import get_system_prompt


class _BotNoPrompts:
    pass


class _BrokenPrompts:
    def get(self, *_args, **_kwargs) -> Never:
        msg = "broken"
        raise RuntimeError(msg)


class _BotWithPrompts:
    def __init__(self, prompts) -> None:
        self.system_prompts = prompts


def test_get_system_prompt_returns_default_when_missing_prompts_attr() -> None:
    assert get_system_prompt(_BotNoPrompts(), "vl_prompt", "default") == "default"


def test_get_system_prompt_reads_dict_prompts() -> None:
    bot = _BotWithPrompts({"text_prompt": "be helpful"})
    assert get_system_prompt(bot, "text_prompt", "fallback") == "be helpful"


def test_get_system_prompt_returns_default_for_missing_key() -> None:
    bot = _BotWithPrompts({"vl_prompt": "analyze image"})
    assert get_system_prompt(bot, "text_prompt", "fallback") == "fallback"


def test_get_system_prompt_returns_default_when_getter_raises() -> None:
    bot = _BotWithPrompts(_BrokenPrompts())
    assert get_system_prompt(bot, "text_prompt", "fallback") == "fallback"
