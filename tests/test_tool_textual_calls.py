"""Tool calls written as text, provider fallback for the tool loop [REH][IV]."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from bot.action import BotAction
from bot.tools import ToolContext, inference
from bot.tools.textual_calls import find_textual_calls, has_textual_call_markup, strip_textual_calls

ALLOWED = frozenset({"web_search", "get_price"})
LIVE = "<function=web_search> <parameter=query> poland vs england crime rate statistics 2024 2025 comparison </parameter> </function>\n</invoke>"


def test_parses_the_live_leak():
    calls = find_textual_calls(LIVE, ALLOWED)
    assert [(c.name, c.arguments) for c in calls] == [("web_search", {"query": "poland vs england crime rate statistics 2024 2025 comparison"})]
    assert strip_textual_calls(LIVE) == ""


def test_parses_json_tool_call_tags_and_numbers():
    text = '<tool_call>{"name": "get_price", "arguments": {"symbol": "BTC"}}</tool_call>'
    assert find_textual_calls(text, ALLOWED)[0].arguments == {"symbol": "BTC"}
    text = "<function=web_search><parameter=query>x</parameter><parameter=max_results>3</parameter></function>"
    assert find_textual_calls(text, ALLOWED)[0].arguments == {"query": "x", "max_results": 3}


def test_unknown_tools_ignored_and_plain_text_untouched():
    assert find_textual_calls("<function=rm_rf><parameter=path>/</parameter></function>", ALLOWED) == []
    plain = "poland's crime rate is lower than england's, 1 < 2"
    assert not has_textual_call_markup(plain)
    assert strip_textual_calls(plain) == plain


def test_strip_keeps_surrounding_answer():
    assert strip_textual_calls("lemme check <function=web_search><parameter=query>q</parameter></function>") == "lemme check"


# Captured live from nemotron-3-ultra on the plain text path (2026-11-20).
LIVE_BARE_PARAMS = "<function=web_search>\n<query>Lucia GTA 6 height weight BMI</query>\n</function>\n<function=web_search>\n<query>UK women BMI distribution percentage statistics 2024 2025</query>\n</function>\n</parameter=invoke>\n</functions>"
LIVE_INVOKE_WRAP = '<function=web_search>\n<query>Lucia GTA 6 height weight BMI estimate</query>\n</function>\n</invoke>\n<invoke name="web_search">\n<function=web_search>\n<query>UK women BMI distribution percentage under 22</query>\n</function>\n</invoke>'
LIVE_JSON_LINES = '{"tool": "search", "query": "Lucia GTA 6 BMI height weight estimate"}\n{"tool": "search", "query": "UK women BMI distribution percentage underweight normal weight 2024 2025"}'


def test_bare_param_tags_parse_and_leave_no_closing_debris():
    calls = find_textual_calls(LIVE_BARE_PARAMS, ALLOWED)
    assert [c.arguments for c in calls] == [{"query": "Lucia GTA 6 height weight BMI"}, {"query": "UK women BMI distribution percentage statistics 2024 2025"}]
    assert strip_textual_calls(LIVE_BARE_PARAMS) == ""  # was "</functions>" in Discord


def test_invoke_wrapper_does_not_duplicate_calls():
    calls = find_textual_calls(LIVE_INVOKE_WRAP, ALLOWED)
    assert [c.arguments["query"] for c in calls] == ["Lucia GTA 6 height weight BMI estimate", "UK women BMI distribution percentage under 22"]
    assert strip_textual_calls(LIVE_INVOKE_WRAP) == ""


def test_anthropic_style_invoke_parses():
    text = '<function_calls><invoke name="web_search"><parameter name="query">uk bmi</parameter></invoke></function_calls>'
    assert [(c.name, c.arguments) for c in find_textual_calls(text, ALLOWED)] == [("web_search", {"query": "uk bmi"})]
    assert strip_textual_calls(text) == ""


def test_json_lines_with_invented_tool_name_map_to_web_search():
    calls = find_textual_calls(LIVE_JSON_LINES, ALLOWED)
    assert [(c.name, c.arguments["query"]) for c in calls] == [("web_search", "Lucia GTA 6 BMI height weight estimate"), ("web_search", "UK women BMI distribution percentage underweight normal weight 2024 2025")]
    assert has_textual_call_markup(LIVE_JSON_LINES)
    assert strip_textual_calls(LIVE_JSON_LINES) == ""


def test_json_answer_not_naming_a_tool_is_kept():
    answer = 'here you go:\n{"name": "Lucia", "height_cm": 163}'
    assert not has_textual_call_markup(answer)
    assert strip_textual_calls(answer) == answer


# -- tool loop -------------------------------------------------------------


def _response(content=None, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls, reasoning=None, reasoning_details=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _Completions:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        nxt = self.responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def _install(monkeypatch, responses):
    client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions(responses)))
    monkeypatch.setattr(inference, "_client_for", lambda cfg, timeout: (client, "primary"))
    monkeypatch.setattr(inference, "_default_system_prompt", lambda cfg: "persona")
    return client.chat.completions


CFG = {
    "TOOLS_ENABLED": True,
    "TOOLS_MAX_ITERATIONS": 3,
    "TOOLS_TIMEOUT_S": 30.0,
    "OPENAI_API_BASE": "https://openrouter.ai/api/v1",
    "TEXT_FALLBACK_MODELS": "openrouter|backup-a,nvidia|other-provider,openrouter|primary,openrouter|backup-b",
}


def test_model_candidates_same_provider_only():
    assert inference._model_candidates(CFG, "primary") == ["primary", "backup-a", "backup-b"]
    assert inference._model_candidates({**CFG, "TOOLS_FALLBACK_MODELS": "x, y"}, "primary") == ["primary", "x", "y"]


async def test_provider_error_falls_back_to_next_model(monkeypatch):
    completions = _install(monkeypatch, [RuntimeError("404 Function not found for account"), _response("poland's lower")])
    out = await inference.run_tool_conversation(prompt="q", ctx=ToolContext(), cfg=CFG)
    assert out == "poland's lower"
    assert [c["model"] for c in completions.calls] == ["primary", "backup-a"]


async def test_all_models_failing_returns_none(monkeypatch):
    _install(monkeypatch, [RuntimeError("a"), RuntimeError("b"), RuntimeError("c")])
    assert await inference.run_tool_conversation(prompt="q", ctx=ToolContext(), cfg=CFG) is None


async def test_textual_call_in_loop_is_executed(monkeypatch):
    completions = _install(monkeypatch, [_response(LIVE), _response("england's higher per capita")])
    run = AsyncMock(return_value=SimpleNamespace(to_message_content=lambda: "stats..."))
    monkeypatch.setattr(inference, "execute_tool", run)
    out = await inference.run_tool_conversation(prompt="q", ctx=ToolContext(), cfg=CFG)
    assert out == "england's higher per capita"
    assert run.await_args.args[:2] == ("web_search", {"query": "poland vs england crime rate statistics 2024 2025 comparison"})
    tool_turn = completions.calls[1]["messages"][-1]
    assert tool_turn["role"] == "tool" and tool_turn["content"] == "stats..."


async def test_force_answer_runs_a_call_written_after_max_iterations(monkeypatch):
    """Live failure: iterations exhausted, the tool-free forced answer was a
    written web_search call, so every gathered result was dropped. [REH]"""
    api_call = SimpleNamespace(id="c1", function=SimpleNamespace(name="web_search", arguments='{"query": "lucia height"}'))
    written = "<tool_call>\n<function=web_search>\n<parameter=query>\nUK women BMI distribution\n</parameter>\n</function>\n</tool_call>\n"
    responses = [_response(tool_calls=[api_call]) for _ in range(3)] + [_response(written), _response("about 15% of uk women")]
    completions = _install(monkeypatch, responses)
    run = AsyncMock(return_value=SimpleNamespace(to_message_content=lambda: "stats..."))
    monkeypatch.setattr(inference, "execute_tool", run)
    out = await inference.run_tool_conversation(prompt="q", ctx=ToolContext(), cfg=CFG)
    assert out == "about 15% of uk women"
    assert run.await_args.args[:2] == ("web_search", {"query": "UK women BMI distribution"})
    assert "tools" not in completions.calls[-1]
    assert "MORE TOOL RESULTS" in completions.calls[-1]["messages"][-3]["content"]
    assert completions.calls[-1]["messages"][-2]["content"] == inference.FINAL_ANSWER_DIRECTIVE


async def test_force_answer_runs_written_calls_only_once(monkeypatch):
    written = "<function=web_search><parameter=query>q</parameter></function>"
    run = AsyncMock(return_value=SimpleNamespace(to_message_content=lambda: "stats..."))
    monkeypatch.setattr(inference, "execute_tool", run)
    client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions([_response(written), _response(written)])))
    out = await inference._force_answer(client, "m", [], 100, {}, 10**12, ToolContext())
    assert out is None
    assert run.await_count == 1


# -- router plain-text path ---------------------------------------------------


def _router(cfg, retry_content):
    from bot.router import Router

    router = Router.__new__(Router)
    router.config = cfg
    router.bot = None
    router.logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)
    router._invoke_text_flow = AsyncMock(return_value=SimpleNamespace(content=retry_content))
    return router


def _leaky_action():
    action = BotAction(content="ok")
    action.content = LIVE  # as built by the text flow before the router sees it
    return action


async def test_router_runs_textual_call_and_regenerates(monkeypatch):
    monkeypatch.setattr("bot.tools.inference.run_textual_calls", AsyncMock(return_value="[web_search result]\nstats"))
    router = _router({"TOOLS_ENABLED": True}, "england's is higher")
    out = await router._resolve_textual_tool_calls(_leaky_action(), "crime rate?", None, "ctx")
    assert out.content == "england's is higher"
    assert "[web_search result]" in router._invoke_text_flow.await_args.args[2]


async def test_router_never_sends_bare_markup_when_tools_off():
    router = _router({"TOOLS_ENABLED": False}, "unused")
    assert await router._resolve_textual_tool_calls(_leaky_action(), "q", None, "ctx") is None
    router._invoke_text_flow.assert_not_awaited()


async def test_router_leaves_normal_replies_alone():
    router = _router({"TOOLS_ENABLED": True}, "unused")
    action = BotAction(content="poland is safer")
    assert await router._resolve_textual_tool_calls(action, "q", None, "ctx") is action


async def test_conclude_moves_to_next_model_when_one_keeps_deliberating(monkeypatch):
    """Live residual: one model kept asking for another search after the
    forced turn. The next model must answer from the same gathered results. [REH]"""
    written = "<function=web_search><parameter=query>q</parameter></function>"
    run = AsyncMock(return_value=SimpleNamespace(to_message_content=lambda: "stats..."))
    monkeypatch.setattr(inference, "execute_tool", run)
    completions = _Completions([_response(written), _response(written), _response("final answer")])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    out = await inference._conclude(client, ["a", "b"], [{"role": "user", "content": "q"}], 100, {}, 10**12, ToolContext())
    assert out == "final answer"
    assert [c["model"] for c in completions.calls] == ["a", "a", "b"]
    assert run.await_count == 1
