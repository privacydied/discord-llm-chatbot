from types import SimpleNamespace

from bot.router_components.gating import (
    is_reply_to_bot,
    mentions_bot,
    strip_leading_bot_mention,
)


def test_mentions_bot_true_and_false() -> None:
    msg = SimpleNamespace(
        mentions=[SimpleNamespace(id=123), SimpleNamespace(id=456)],
    )
    assert mentions_bot(msg, 123) is True
    assert mentions_bot(msg, 999) is False


def test_is_reply_to_bot_from_resolved_message() -> None:
    ref = SimpleNamespace(
        message_id=1,
        resolved=SimpleNamespace(author=SimpleNamespace(id=123)),
        cached_message=None,
    )
    msg = SimpleNamespace(reference=ref)
    assert is_reply_to_bot(msg, 123) is True
    assert is_reply_to_bot(msg, 456) is False


def test_is_reply_to_bot_from_cached_message() -> None:
    ref = SimpleNamespace(
        message_id=1,
        resolved=None,
        cached_message=SimpleNamespace(author=SimpleNamespace(id=123)),
    )
    msg = SimpleNamespace(reference=ref)
    assert is_reply_to_bot(msg, 123) is True


def test_strip_leading_bot_mention() -> None:
    assert strip_leading_bot_mention("<@123> hello", 123) == "hello"
    assert strip_leading_bot_mention("<@!123> hello", 123) == "hello"
    assert strip_leading_bot_mention("no mention", 123) == "no mention"
