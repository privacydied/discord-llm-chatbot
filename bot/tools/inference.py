"""Bounded tool-calling conversation loop.
[CA][REH][PA][CMV][SFT].

Deliberately a *parallel* path rather than a change to
``generate_openai_response``: that function always rebuilds a fixed
system/user message array and treats an empty ``content`` as a hard error --
which is exactly what a tool call looks like. Wiring tools through it would
make the first tool call burn every rung of the fallback ladder. This module
therefore owns its own messages array and its own response handling, and the
ordinary text path keeps working byte-for-byte as before.

On any failure this returns None so the caller can fall back to the normal
flow; a tool loop must never be the reason a user gets no reply.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from bot.utils.logging import get_logger

from .registry import execute_tool, get_registry
from .types import ToolContext

logger = get_logger(__name__)

# Hard cap on model round-trips per user message. Each iteration is a full
# completion, so this bounds both latency and spend. [CMV][PA]
DEFAULT_MAX_ITERATIONS = 3

# Whole-loop wall clock. [CMV]
DEFAULT_TIMEOUT_S = 30.0

# Tool results are appended to the conversation; stop growing it past this. [PA]
MAX_TRANSCRIPT_CHARS = 24000


def _client_for(cfg: dict[str, Any], timeout_s: float) -> tuple[Any, str]:
    """Build (client, model) for the tool-capable completion. [PA]

    Reuses openai_backend's cached client factory so the connection pool stays
    warm and the Synology SSL workaround applies.
    """
    from bot.openai_backend import _make_openai_async_client, _resolve_openai_compatible_endpoint

    base = str(cfg.get("OPENAI_API_BASE") or "")
    provider = "openrouter" if "openrouter" in base.lower() else ("nvidia" if "nvidia.com" in base.lower() else "openai")
    api_key, base_url, _ = _resolve_openai_compatible_endpoint(provider, cfg)

    model = (cfg.get("TOOLS_MODEL") or "").strip() or cfg.get("OPENAI_TEXT_MODEL") or ""
    client = _make_openai_async_client(
        api_key=api_key,
        base_url=base_url,
        timeout=httpx.Timeout(timeout_s),
        max_retries=0,
    )
    return client, model


def _default_system_prompt(cfg: dict[str, Any]) -> str | None:
    """Fall back to the configured persona file so tool replies stay in voice."""
    path = cfg.get("PROMPT_FILE")
    if not path:
        return None
    try:
        from bot.openai_backend import _load_prompt_cached

        return _load_prompt_cached(path)
    except Exception as exc:  # [REH] a missing persona must not block the loop
        logger.debug("tools.prompt_file_unavailable error=%s", exc)
        return None


# First of three layers against chain-of-thought reaching the user. This one
# just reduces how often the model deliberates instead of answering; the real
# guarantees are _is_reasoning_leak (detection) and _force_answer (recovery)
# below, because a prompt directive alone cannot be relied upon. [REH][CMV]
TOOL_OUTPUT_DIRECTIVE = (
    "You have tools available. Call them when they help, then reply to the user directly.\n"
    "Give only the final answer. Do not narrate your reasoning, do not restate the question, "
    "do not describe which tools you called or quote their parameters, and do not think out loud."
)

# Sent when the model deliberated without producing an answer. Forcing a
# tool-free turn stops it re-entering the same loop of indecision. [REH]
FORCE_ANSWER_DIRECTIVE = (
    "Stop deliberating and answer now, using only the information already gathered above. Reply with the final answer only, in at most three sentences. If the information is insufficient, say so plainly in one sentence."
)


def _build_messages(prompt: str, system_prompt: str | None, context: str | None) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "system", "content": TOOL_OUTPUT_DIRECTIVE})
    if context:
        messages.append({"role": "system", "content": f"PREVIOUS_CONVERSATION_HISTORY:\n{context}"})
    messages.append({"role": "user", "content": prompt})
    return messages


def _parse_arguments(raw: Any) -> dict[str, Any]:
    """Tool arguments arrive as a JSON string. Malformed input is not fatal. [IV]"""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _tool_calls_of(response: Any) -> list[Any]:
    try:
        message = response.choices[0].message
    except (AttributeError, IndexError):
        return []
    return list(getattr(message, "tool_calls", None) or [])


def _reasoning_of(message: Any) -> str:
    """The provider's separate chain-of-thought field, if any."""
    reasoning = getattr(message, "reasoning", None) or ""
    if reasoning:
        return str(reasoning).strip()
    details = getattr(message, "reasoning_details", None) or []
    if isinstance(details, list):
        parts = [str(d.get("text", "")) for d in details if isinstance(d, dict)]
        return " ".join(p for p in parts if p).strip()
    return ""


def _extract(response: Any) -> tuple[str, str]:
    """Return (content, reasoning) for the assistant turn.

    Reasoning normally lands in its own field with clean content. But when the
    model deliberates without ever concluding, the provider duplicates the
    chain of thought into `content` verbatim -- that is the leak users see.
    Both values are returned so the caller can detect it exactly. [REH]
    """
    try:
        message = response.choices[0].message
    except (AttributeError, IndexError):
        return "", ""
    raw = getattr(message, "content", "") or ""
    try:
        from bot.vl.postprocess import sanitize_model_output

        content = sanitize_model_output(raw) or ""
    except Exception:  # [REH] sanitiser must never cost us the answer
        content = raw
    return content.strip(), _reasoning_of(message)


# How much of the reasoning must prefix the content before we call it a leak.
# Verified live: the duplicate is byte-identical, so this is a safety margin
# for providers that append a partial trailer rather than an exact copy. [CMV]
_LEAK_PREFIX_CHARS = 200

# A dumped transcript is long; a real answer that merely coincides with a terse
# reasoning field is short. Requiring length removes the false positive where a
# brief reply like "Yes." equals its own reasoning. Observed leaks ran to
# ~1400 characters. [CMV]
_MIN_LEAK_CHARS = 200


def _is_reasoning_leak(content: str, reasoning: str) -> bool:
    """True when `content` is the model's deliberation rather than its answer.

    Exact, not heuristic: the provider emits the same text in both fields.
    """
    if not content or not reasoning:
        return False
    if len(content) < _MIN_LEAK_CHARS:
        return False
    if content == reasoning:
        return True
    head = reasoning[:_LEAK_PREFIX_CHARS]
    return len(head) >= _LEAK_PREFIX_CHARS and content.startswith(head)


def _reasoning_extra_body(cfg: dict[str, Any]) -> dict[str, Any]:
    """Reuse the backend's OpenRouter reasoning-exclusion request body."""
    try:
        from bot.openai_backend import _reasoning_exclude_extra_body

        base = str(cfg.get("OPENAI_API_BASE") or "").lower()
        provider = "openrouter" if "openrouter" in base else ("nvidia" if "nvidia.com" in base else "openai")
        return _reasoning_exclude_extra_body(provider, cfg) or {}
    except Exception as exc:  # [REH]
        logger.debug("tools.reasoning_extra_body_unavailable error=%s", exc)
        return {}


def _assistant_turn(tool_calls: list[Any]) -> dict[str, Any]:
    """Echo the assistant's tool-call turn back into the transcript."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in tool_calls
        ],
    }


def _transcript_size(messages: list[dict[str, Any]]) -> int:
    return sum(len(str(m.get("content") or "")) for m in messages)


async def _run_tool_calls(tool_calls: list[Any], ctx: ToolContext) -> list[dict[str, Any]]:
    """Execute each requested tool, returning the `tool` role turns."""
    turns: list[dict[str, Any]] = []
    for call in tool_calls:
        name = getattr(getattr(call, "function", None), "name", "") or ""
        arguments = _parse_arguments(getattr(getattr(call, "function", None), "arguments", None))
        logger.info("tool.call name=%s args=%s", name, str(arguments)[:200])
        result = await execute_tool(name, arguments, ctx)
        turns.append(
            {
                "role": "tool",
                "tool_call_id": getattr(call, "id", ""),
                "content": result.to_message_content(),
            }
        )
    return turns


async def _force_answer(
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    extra_body: dict[str, Any],
) -> str | None:
    """One tool-free retry to make an indecisive model commit. [REH]

    Tools are withheld entirely (not merely tool_choice="none") so the model
    cannot answer with another call, and the nudge caps the length so the reply
    cannot become another wall of deliberation.
    """
    attempt = [*messages, {"role": "system", "content": FORCE_ANSWER_DIRECTIVE}]
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": attempt,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if extra_body:
        kwargs["extra_body"] = extra_body
    try:
        response = await client.chat.completions.create(**kwargs)
    except Exception as exc:  # [REH]
        logger.warning("tools.force_answer_failed error=%s", exc)
        return None

    content, reasoning = _extract(response)
    if content and not _is_reasoning_leak(content, reasoning):
        logger.info("tools.force_answer_ok chars=%d", len(content))
        return content
    logger.warning("tools.force_answer_still_deliberating")
    return None


async def run_tool_conversation(
    *,
    prompt: str,
    ctx: ToolContext,
    system_prompt: str | None = None,
    context: str | None = None,
    cfg: dict[str, Any] | None = None,
) -> str | None:
    """Run the model with tools available, resolving calls until it answers.

    Returns the final assistant text, or None if tools could not be used and
    the caller should fall back to the ordinary text flow. [REH]
    """
    from bot.config import load_config

    config = cfg if cfg is not None else load_config()
    if not config.get("TOOLS_ENABLED", False):
        return None

    max_iterations = int(config.get("TOOLS_MAX_ITERATIONS", DEFAULT_MAX_ITERATIONS))
    timeout_s = float(config.get("TOOLS_TIMEOUT_S", DEFAULT_TIMEOUT_S))
    deadline = time.monotonic() + timeout_s

    try:
        client, model = _client_for(config, timeout_s)
    except Exception as exc:  # [REH]
        logger.warning("tools.client_failed error=%s", exc)
        return None
    if not model:
        logger.warning("tools.no_model_configured")
        return None

    messages = _build_messages(prompt, system_prompt or _default_system_prompt(config), context)
    schemas = get_registry().schemas()
    max_tokens = config.get("MAX_RESPONSE_TOKENS", 1000)
    extra_body = _reasoning_extra_body(config)

    for iteration in range(max_iterations):
        if time.monotonic() >= deadline:
            logger.warning("tools.deadline_exceeded iteration=%d", iteration)
            return None
        try:
            create_kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "tools": schemas,
                "tool_choice": "auto",
                "max_tokens": max_tokens,
                "stream": False,
            }
            if extra_body:
                create_kwargs["extra_body"] = extra_body
            response = await client.chat.completions.create(**create_kwargs)
        except Exception as exc:  # [REH] model may not support tools at all
            logger.warning("tools.completion_failed iteration=%d error=%s", iteration, exc)
            return None

        tool_calls = _tool_calls_of(response)
        if not tool_calls:
            content, reasoning = _extract(response)
            if content and not _is_reasoning_leak(content, reasoning):
                logger.info("tools.answered iterations=%d", iteration)
                return content
            # Either empty, or the provider echoed the chain of thought as the
            # answer. Both mean "deliberated, never concluded" -- ask once more
            # with tools withheld so it must commit. [REH]
            logger.info(
                "tools.no_conclusion iteration=%d leak=%s",
                iteration,
                bool(content),
            )
            return await _force_answer(client, model, messages, max_tokens, extra_body)

        messages.append(_assistant_turn(tool_calls))
        messages.extend(await _run_tool_calls(tool_calls, ctx))

        if _transcript_size(messages) > MAX_TRANSCRIPT_CHARS:
            logger.warning("tools.transcript_too_large iteration=%d", iteration)
            return None

    logger.warning("tools.max_iterations_reached max=%d", max_iterations)
    return None
