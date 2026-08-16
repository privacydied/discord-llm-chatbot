"""Tests for the tool-calling loop and its router hook.
[CA][REH][PA].
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from bot.tools import ToolContext, inference


# --------------------------------------------------------------------------
# Fakes mimicking the OpenAI SDK response shape
# --------------------------------------------------------------------------
def _call(name, arguments, call_id="c1"):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


def _response(content=None, tool_calls=None, reasoning=None, reasoning_details=None):
    message = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        reasoning=reasoning,
        reasoning_details=reasoning_details,
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("model called more times than the test scripted")
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


class _FakeClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=_FakeCompletions(responses))


def _install(monkeypatch, responses, model="test-model"):
    client = _FakeClient(responses)
    monkeypatch.setattr(inference, "_client_for", lambda cfg, timeout: (client, model))
    monkeypatch.setattr(inference, "_default_system_prompt", lambda cfg: "persona")
    return client


CFG = {"TOOLS_ENABLED": True, "TOOLS_MAX_ITERATIONS": 3, "TOOLS_TIMEOUT_S": 30.0}


# --------------------------------------------------------------------------
# Loop behaviour
# --------------------------------------------------------------------------


async def test_direct_answer_without_tools(monkeypatch):
    client = _install(monkeypatch, [_response(content="just an answer")])
    out = await inference.run_tool_conversation(prompt="hi", ctx=ToolContext(), cfg=CFG)
    assert out == "just an answer"
    assert len(client.chat.completions.calls) == 1


async def test_tools_are_offered_to_the_model(monkeypatch):
    client = _install(monkeypatch, [_response(content="ok")])
    await inference.run_tool_conversation(prompt="hi", ctx=ToolContext(), cfg=CFG)
    sent = client.chat.completions.calls[0]
    names = {t["function"]["name"] for t in sent["tools"]}
    from bot.tools.registry import ALLOWED_TOOL_NAMES

    assert names == set(ALLOWED_TOOL_NAMES)
    assert sent["tool_choice"] == "auto"


async def test_executes_tool_then_answers(monkeypatch):
    client = _install(
        monkeypatch,
        [
            _response(tool_calls=[_call("get_current_time", "{}")]),
            _response(content="it is now"),
        ],
    )
    out = await inference.run_tool_conversation(prompt="what time is it", ctx=ToolContext(), cfg=CFG)
    assert out == "it is now"

    second = client.chat.completions.calls[1]["messages"]
    assert second[-2]["role"] == "assistant"
    assert second[-2]["tool_calls"][0]["function"]["name"] == "get_current_time"
    assert second[-1]["role"] == "tool"
    assert "UTC" in second[-1]["content"]


async def test_multiple_tool_calls_in_one_turn(monkeypatch):
    _install(
        monkeypatch,
        [
            _response(tool_calls=[_call("get_current_time", "{}", "a"), _call("get_current_time", "{}", "b")]),
            _response(content="done"),
        ],
    )
    assert await inference.run_tool_conversation(prompt="x", ctx=ToolContext(), cfg=CFG) == "done"


async def test_failed_tool_is_reported_to_model_not_raised(monkeypatch):
    """A tool error becomes a tool message so the model can recover."""
    client = _install(
        monkeypatch,
        [
            _response(tool_calls=[_call("read_channel_history", '{"posts_ago": 5}')]),
            _response(content="sorry, could not read that"),
        ],
    )
    out = await inference.run_tool_conversation(prompt="x", ctx=ToolContext(), cfg=CFG)
    assert out == "sorry, could not read that"
    tool_turn = client.chat.completions.calls[1]["messages"][-1]
    assert tool_turn["role"] == "tool"
    assert tool_turn["content"].startswith("ERROR:")


async def test_unknown_tool_requested_by_model(monkeypatch):
    client = _install(
        monkeypatch,
        [
            _response(tool_calls=[_call("delete_all_files", '{"path": "/"}')]),
            _response(content="I cannot do that"),
        ],
    )
    out = await inference.run_tool_conversation(prompt="x", ctx=ToolContext(), cfg=CFG)
    assert out == "I cannot do that"
    assert "unknown tool" in client.chat.completions.calls[1]["messages"][-1]["content"]


async def test_malformed_tool_arguments_do_not_crash(monkeypatch):
    _install(
        monkeypatch,
        [
            _response(tool_calls=[_call("get_current_time", "{not json")]),
            _response(content="fine"),
        ],
    )
    assert await inference.run_tool_conversation(prompt="x", ctx=ToolContext(), cfg=CFG) == "fine"


async def test_iteration_cap_enforced(monkeypatch):
    """A model that only ever calls tools must be stopped, not looped forever."""
    responses = [_response(tool_calls=[_call("get_current_time", "{}")]) for _ in range(10)]
    client = _install(monkeypatch, responses)
    cfg = {**CFG, "TOOLS_MAX_ITERATIONS": 3}
    out = await inference.run_tool_conversation(prompt="x", ctx=ToolContext(), cfg=cfg)
    assert out is None
    assert len(client.chat.completions.calls) == 3


async def test_api_failure_returns_none_for_fallback(monkeypatch):
    _install(monkeypatch, [RuntimeError("model does not support tools")])
    assert await inference.run_tool_conversation(prompt="x", ctx=ToolContext(), cfg=CFG) is None


async def test_empty_answer_returns_none(monkeypatch):
    _install(monkeypatch, [_response(content="   ")])
    assert await inference.run_tool_conversation(prompt="x", ctx=ToolContext(), cfg=CFG) is None


async def test_disabled_by_default(monkeypatch):
    called = False

    def _boom(cfg, timeout):
        nonlocal called
        called = True
        raise AssertionError("must not build a client when disabled")

    monkeypatch.setattr(inference, "_client_for", _boom)
    assert await inference.run_tool_conversation(prompt="x", ctx=ToolContext(), cfg={}) is None
    assert not called


async def test_no_model_configured_returns_none(monkeypatch):
    monkeypatch.setattr(inference, "_client_for", lambda cfg, timeout: (_FakeClient([]), ""))
    assert await inference.run_tool_conversation(prompt="x", ctx=ToolContext(), cfg=CFG) is None


async def test_context_and_system_prompt_are_sent(monkeypatch):
    client = _install(monkeypatch, [_response(content="ok")])
    await inference.run_tool_conversation(
        prompt="question",
        ctx=ToolContext(),
        system_prompt="be terse",
        context="earlier chat",
        cfg=CFG,
    )
    messages = client.chat.completions.calls[0]["messages"]
    assert messages[0] == {"role": "system", "content": "be terse"}
    assert messages[-1] == {"role": "user", "content": "question"}

    system_text = " ".join(m["content"] for m in messages if m["role"] == "system")
    assert "earlier chat" in system_text
    assert "do not think out loud" in system_text.lower()


# --------------------------------------------------------------------------
# Reasoning leakage
#
# Verified live: when the model deliberates without concluding, the provider
# duplicates `reasoning` into `content` verbatim, and the user sees the model
# thinking out loud instead of an answer.
# --------------------------------------------------------------------------

CHAIN_OF_THOUGHT = (
    "The user asks: summarise the last few posts. I need to interpret what they mean. "
    "Typically that could mean posts 3, 4 and 5. Or maybe 4 and 5? Wait, let me check the "
    "tool description again. It said posts_ago is the smaller number. Hmm, but I called it "
    "with 5. Let me reconsider what the labels mean and whether they are off by one."
)


def test_leak_detected_when_content_equals_reasoning():
    assert inference._is_reasoning_leak(CHAIN_OF_THOUGHT, CHAIN_OF_THOUGHT)


def test_leak_detected_when_content_is_truncated_reasoning():
    assert inference._is_reasoning_leak(CHAIN_OF_THOUGHT[:300], CHAIN_OF_THOUGHT)


def test_clean_answer_is_not_a_leak():
    assert not inference._is_reasoning_leak("Frank asked about lunch.", CHAIN_OF_THOUGHT)


def test_no_reasoning_field_means_no_leak():
    assert not inference._is_reasoning_leak("some answer", "")


def test_short_answer_matching_its_reasoning_is_not_a_leak():
    """A terse reply that coincides with its reasoning must still be delivered."""
    assert not inference._is_reasoning_leak("Yes.", "Yes.")
    assert not inference._is_reasoning_leak("PINEAPPLE BELONGS ON PIZZA", "PINEAPPLE BELONGS ON PIZZA")


async def test_short_matching_answer_is_returned_not_retried(monkeypatch):
    client = _install(monkeypatch, [_response(content="Yes.", reasoning="Yes.")])
    assert await inference.run_tool_conversation(prompt="x", ctx=ToolContext(), cfg=CFG) == "Yes."
    assert len(client.chat.completions.calls) == 1


def test_extract_reads_reasoning_details_fallback():
    response = _response(
        content="answer",
        reasoning_details=[{"type": "reasoning.text", "text": "thinking hard"}],
    )
    content, reasoning = inference._extract(response)
    assert content == "answer"
    assert "thinking hard" in reasoning


async def test_leaked_reasoning_triggers_forced_answer(monkeypatch):
    """The user must get the answer, not the deliberation."""
    client = _install(
        monkeypatch,
        [
            _response(content=CHAIN_OF_THOUGHT, reasoning=CHAIN_OF_THOUGHT),
            _response(content="Frank asked about lunch and Dave reported a green deploy."),
        ],
    )
    out = await inference.run_tool_conversation(prompt="summarise", ctx=ToolContext(), cfg=CFG)
    assert out == "Frank asked about lunch and Dave reported a green deploy."
    assert CHAIN_OF_THOUGHT not in (out or "")

    retry = client.chat.completions.calls[1]
    assert "tools" not in retry, "tools must be withheld so it cannot stall again"
    assert "Stop deliberating" in retry["messages"][-1]["content"]


async def test_empty_content_also_triggers_forced_answer(monkeypatch):
    """reasoning.exclude-style empty content is the same failure."""
    _install(
        monkeypatch,
        [
            _response(content="", reasoning=CHAIN_OF_THOUGHT),
            _response(content="recovered answer"),
        ],
    )
    out = await inference.run_tool_conversation(prompt="x", ctx=ToolContext(), cfg=CFG)
    assert out == "recovered answer"


async def test_forced_answer_that_still_leaks_returns_none(monkeypatch):
    """Two strikes and we fall back rather than show chain of thought."""
    _install(
        monkeypatch,
        [
            _response(content=CHAIN_OF_THOUGHT, reasoning=CHAIN_OF_THOUGHT),
            _response(content=CHAIN_OF_THOUGHT, reasoning=CHAIN_OF_THOUGHT),
        ],
    )
    assert await inference.run_tool_conversation(prompt="x", ctx=ToolContext(), cfg=CFG) is None


async def test_forced_answer_api_failure_returns_none(monkeypatch):
    _install(
        monkeypatch,
        [
            _response(content="", reasoning=CHAIN_OF_THOUGHT),
            RuntimeError("upstream down"),
        ],
    )
    assert await inference.run_tool_conversation(prompt="x", ctx=ToolContext(), cfg=CFG) is None


async def test_forced_answer_runs_after_tool_use(monkeypatch):
    """Recovery must retain the tool results already gathered."""
    client = _install(
        monkeypatch,
        [
            _response(tool_calls=[_call("get_current_time", "{}")]),
            _response(content=CHAIN_OF_THOUGHT, reasoning=CHAIN_OF_THOUGHT),
            _response(content="It is currently midday."),
        ],
    )
    out = await inference.run_tool_conversation(prompt="time?", ctx=ToolContext(), cfg=CFG)
    assert out == "It is currently midday."

    retry_messages = client.chat.completions.calls[2]["messages"]
    assert any(m["role"] == "tool" for m in retry_messages), "tool results must survive into the retry"


async def test_clean_answer_never_triggers_a_retry(monkeypatch):
    client = _install(monkeypatch, [_response(content="clean", reasoning="some thinking")])
    assert await inference.run_tool_conversation(prompt="x", ctx=ToolContext(), cfg=CFG) == "clean"
    assert len(client.chat.completions.calls) == 1


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"a": 1}', {"a": 1}),
        ({"a": 1}, {"a": 1}),
        ("", {}),
        (None, {}),
        ("not json", {}),
        ("[1,2]", {}),
        ("null", {}),
    ],
)
def test_parse_arguments(raw, expected):
    assert inference._parse_arguments(raw) == expected


# --------------------------------------------------------------------------
# Router hook
# --------------------------------------------------------------------------


class _StubRouter:
    def __init__(self, config):
        self.config = config
        self.bot = None
        self.logger = logging.getLogger("test.router.tools")


def _hook(config):
    from bot.router import Router

    return lambda *a, **k: Router._maybe_answer_with_tools(_StubRouter(config), *a, **k)


async def test_router_hook_noop_when_disabled(monkeypatch):
    called = False

    async def _never(**kwargs):
        nonlocal called
        called = True
        return "x"

    monkeypatch.setattr("bot.tools.inference.run_tool_conversation", _never)
    assert await _hook({})("hi", None, "") is None
    assert not called


async def test_router_hook_returns_bot_action(monkeypatch):
    async def _answer(**kwargs):
        return "the answer"

    monkeypatch.setattr("bot.tools.inference.run_tool_conversation", _answer)
    action = await _hook({"TOOLS_ENABLED": True})("hi", None, "")
    assert action is not None
    assert action.content == "the answer"


async def test_router_hook_falls_back_on_none(monkeypatch):
    async def _nothing(**kwargs):
        return None

    monkeypatch.setattr("bot.tools.inference.run_tool_conversation", _nothing)
    assert await _hook({"TOOLS_ENABLED": True})("hi", None, "") is None


async def test_router_hook_survives_exception(monkeypatch):
    async def _boom(**kwargs):
        raise RuntimeError("upstream")

    monkeypatch.setattr("bot.tools.inference.run_tool_conversation", _boom)
    assert await _hook({"TOOLS_ENABLED": True})("hi", None, "") is None


# --------------------------------------------------------------------------
# Config registration
# --------------------------------------------------------------------------


def _getter(values):
    def get(key, default=None):
        return values.get(key, default)

    return get


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("TOOLS_ENABLED", False),
        ("TOOLS_MAX_ITERATIONS", 3),
        # Must exceed the slowest tool's own budget (view_image, 45s). [PA]
        ("TOOLS_TIMEOUT_S", 90.0),
    ],
)
def test_tools_config_defaults(key, expected):
    from bot.config._base import _build_config

    assert _build_config(_getter({}))[key] == expected


def test_tools_config_reads_env():
    from bot.config._base import _build_config

    cfg = _build_config(_getter({"TOOLS_ENABLED": "true", "TOOLS_MAX_ITERATIONS": "5"}))
    assert cfg["TOOLS_ENABLED"] is True
    assert cfg["TOOLS_MAX_ITERATIONS"] == 5


def test_tools_disabled_when_env_absent_so_rollout_is_opt_in():
    from bot.config._base import _build_config

    assert _build_config(_getter({}))["TOOLS_ENABLED"] is False
