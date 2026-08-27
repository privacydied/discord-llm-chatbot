"""Tests for the shared parameter-size ranking heuristic. [CMV]"""

import pytest

from bot.vision.model_ranking import SMALLEST_TIER, UNKNOWN_TIER, extract_param_billions, param_tier


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("meta-llama/llama-3.1-405b-instruct:free", 405.0),
        ("qwen/qwen3-235b-a22b:free", 235.0),
        ("meta-llama/llama-3.3-70b-instruct:free", 70.0),
        ("google/gemma-2-9b-it:free", 9.0),
        ("meta-llama/llama-3.2-1b-instruct:free", 1.0),
        ("meta-llama/llama-3.2-0.5b-instruct:free", 0.5),
        ("mistralai/mixtral-8x7b-instruct:free", 56.0),  # MoE: 8 experts * 7B
        ("mistralai/mixtral-8x22b-instruct:free", 176.0),
        ("openai/gpt-oss-20b:free", 20.0),
    ],
)
def test_extract_param_billions_parses_known_notations(model_id, expected):
    assert extract_param_billions(model_id) == expected


@pytest.mark.parametrize(
    "model_id",
    [
        "deepseek/deepseek-r1:free",
        "moonshotai/kimi-k2:free",
        "z-ai/glm-4.5-air:free",
        "google/gemini-2.0-flash-exp:free",
        "",
        "vendor/no-size-hint",
    ],
)
def test_extract_param_billions_returns_none_when_no_hint(model_id):
    assert extract_param_billions(model_id) is None


def test_param_tier_orders_larger_models_first():
    huge = param_tier("meta-llama/llama-3.1-405b-instruct:free")
    large = param_tier("meta-llama/llama-3.3-70b-instruct:free")
    mid = param_tier("openai/gpt-oss-20b:free")
    small = param_tier("meta-llama/llama-3.2-1b-instruct:free")
    assert huge < large < mid < small


def test_param_tier_unknown_sits_between_midsize_and_small():
    """Undisclosed-size frontier models must not be ranked as if they were tiny."""
    mid = param_tier("openai/gpt-oss-20b:free")  # 20B, confirmed >=15B tier
    unknown = param_tier("deepseek/deepseek-r1:free")
    small = param_tier("google/gemma-2-9b-it:free")  # 9B, confirmed <15B tier
    assert mid < unknown < small
    assert unknown == UNKNOWN_TIER
    assert small == SMALLEST_TIER
