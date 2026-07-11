"""Focused tests for memory ingestion gate and relevance gating."""

from __future__ import annotations

import pytest

from bot.memory.gates import (
    MemoryIngestionContext,
    select_memories_for_prompt,
    should_auto_store_memory,
)


def make_context(**kwargs):
    payload = {
        "source_message_id": "m1",
        "source_user_id": "u1",
        "source_guild_id": "g1",
        "source_channel_id": "c1",
        "is_explicit_command": False,
        "guild_only": False,
        "raw_text": None,
    }
    payload.update(kwargs)
    return MemoryIngestionContext(**payload)


@pytest.mark.parametrize(
    "text, expected_category",
    [
        ("I prefer short replies", "user_preference"),
        ("The canonical repo path is /opt/repo", "project_fact"),
        ("From now on always use uv run", "recurring_instruction"),
        ("We decided to use postgres", "conversation_decision"),
        ("Keep this config only for today", "temporary_context"),
        ("This server rule requires mod approval", "server_fact"),
    ],
)
def test_auto_store_category_routing(text, expected_category):
    context = make_context(guild_only="server rule" in text or "This server" in text)
    decision = should_auto_store_memory(
        {"content": text, "importance": 0.95, "confidence": 0.95},
        context,
    )
    assert decision.allowed is True
    assert decision.category == expected_category


def test_ordinary_chat_is_not_stored():
    decision = should_auto_store_memory(
        {"content": "man I'm tired today, anyone want ramen?"},
        make_context(),
    )
    assert decision.allowed is False


def test_noisy_or_offensive_is_rejected():
    decision = should_auto_store_memory(
        {"content": "you are worthless lol"},
        make_context(),
    )
    assert decision.allowed is False


def test_quoted_external_content_is_rejected():
    decision = should_auto_store_memory(
        {"content": 'According to https://example.com "the sky is blue"'},
        make_context(),
    )
    assert decision.allowed is False


def test_diagnostic_artifact_is_rejected():
    decision = should_auto_store_memory(
        {"content": "A/B test result: mode=diagnostic"},
        make_context(),
    )
    assert decision.allowed is False


def test_recurring_instruction_requires_future_marker():
    candidate = {"content": "always reply shorter", "importance": 1.0, "confidence": 1.0}
    context = make_context()
    decision = should_auto_store_memory(candidate, context)
    assert decision.allowed is False
    # classifies as recurring_instruction but rejected because no explicit future marker
    assert decision.category == "recurring_instruction"


def test_explicit_future_marker_allows_recurring_instruction():
    context = make_context(raw_text="from now on always reply shorter")
    decision = should_auto_store_memory(
        {"content": "from now on always reply shorter", "importance": 0.9, "confidence": 0.9},
        context,
    )
    assert decision.allowed is True
    assert decision.category == "recurring_instruction"


def test_select_memories_filters_irrelevant():
    candidates = [
        {
            "memory_id": "related",
            "document": "Use uv run for python commands in this project.",
            "metadata": {"importance": 0.9, "confidence": 0.9, "source": "inferred_curated"},
            "semantic_score": 0.4,
        },
        {
            "memory_id": "noise",
            "document": "I prefer coffee every morning before standup.",
            "metadata": {"importance": 0.5, "confidence": 0.5, "source": "inferred_curated"},
            "semantic_score": 0.2,
        },
        {
            "memory_id": "explicit",
            "document": "remember the canonical deploy script path",
            "metadata": {"importance": 0.95, "confidence": 0.9, "source": "explicit_memory_command"},
            "semantic_score": 0.3,
        },
    ]
    selected = select_memories_for_prompt("How do I deploy this repo?", candidates, max_items=3)
    ids = [item["memory_id"] for item in selected]
    assert "related" in ids
    assert "explicit" in ids
    assert "noise" not in ids


def test_select_memories_prefers_relevant_even_if_lower_rank():
    candidates = [
        {
            "memory_id": "garbage",
            "document": "unrelated personal preference about pizza toppings",
            "metadata": {"importance": 0.9, "confidence": 0.9, "source": "inferred_curated"},
            "semantic_score": 0.6,
        },
        {
            "memory_id": "project",
            "document": "project fact: pytest is the test runner",
            "metadata": {"importance": 0.6, "confidence": 0.8, "source": "inferred_curated"},
            "semantic_score": 0.5,
        },
    ]
    selected = select_memories_for_prompt("run the tests for this repo", candidates, max_items=2)
    ids = [item["memory_id"] for item in selected]
    assert "project" in ids
    assert "garbage" not in ids


def test_low_confidence_memory_is_filtered():
    candidates = [
        {
            "memory_id": "uncertain",
            "document": "maybe the path is /opt/app",
            "metadata": {"importance": 0.95, "confidence": 0.2, "source": "inferred_curated"},
            "semantic_score": 0.9,
        }
    ]
    selected = select_memories_for_prompt("where is the app path?", candidates, max_items=3)
    assert selected == []
