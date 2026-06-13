import sys
import types

# Provide lightweight stand-ins for optional third-party modules that are
# imported transitively by the logging helpers.
sys.modules.setdefault("httpx", types.ModuleType("httpx"))

from bot.decision_helpers import (
    choose_route,
    compose_context,
    detect_media_intent,
    extract_chat_text,
    harvest_in_scope_io,
    resolve_scope,
    select_reply_target,
)


class Dummy:
    pass


def test_resolve_scope_plain() -> None:
    msg = Dummy()
    msg.id = "1"
    res = resolve_scope(msg)
    assert res.case == "plain"
    assert res.scope_id == "1"


def test_resolve_scope_thread_and_reply() -> None:
    class Thread:
        def __init__(self, id: str) -> None:
            self.id = id

    thread = Thread("t1")
    msg_thread = Dummy()
    msg_thread.id = "2"
    msg_thread.channel = thread
    res_thread = resolve_scope(msg_thread)
    assert res_thread.case == "thread"
    assert res_thread.scope_id == "t1"

    ref = types.SimpleNamespace(message_id="p1")
    msg_reply = Dummy()
    msg_reply.id = "3"
    msg_reply.reference = ref
    res_reply = resolve_scope(msg_reply)
    assert res_reply.case == "reply"
    assert res_reply.scope_id == "p1"


def test_extract_and_intent_and_route() -> None:
    messages = ["<@123> hi", "there!"]
    result = extract_chat_text(messages)
    assert result["has_text_flag"] is True

    intent = detect_media_intent("please analyze this image")
    harvested = harvest_in_scope_io([{"urls": [], "attachments": []}])
    route = choose_route(result["has_text_flag"], intent, harvested)
    assert route == "nag"


def test_select_reply_and_compose_context() -> None:
    target = select_reply_target("thread", "5", now=0)
    assert target == "5"

    context = compose_context(["a", "b", "a"], {"max_items": 2, "max_chars": 10})
    assert context["items"] == ["a", "b"]
    assert context["joined_text"] == "a\nb"
