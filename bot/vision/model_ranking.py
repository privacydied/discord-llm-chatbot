"""Parameter-size heuristics for ranking free OpenRouter models. [CMV][PA]

OpenRouter's catalogue does not expose a parameter count field, but model
slugs usually encode one (``llama-3.1-70b-instruct``, ``mixtral-8x7b``,
``qwen3-235b-a22b``). This module extracts that count on a best-effort basis
and buckets it into coarse tiers so larger models sort ahead of smaller ones
in the discovery ladder -- the "prefer larger parameter models" rule shared by
both the vision (``free_model_discovery.py``) and text (``free_text_discovery.py``)
discovery ladders.

Not every strong free model publishes a parameter count in its slug (DeepSeek
R1/V3, Kimi K2, GLM-4.5, Grok, Gemini are common examples). Those land in
``UNKNOWN_TIER`` -- deliberately placed *between* the 15B+ and sub-15B tiers so
an undisclosed-size frontier model is never punished as if it were tiny, while
a model that confirms it is small still sorts behind it.
"""

from __future__ import annotations

import re

# Matched against the leaf segment of the model slug (everything after the
# last "/", lower-cased, ":free"/tag suffix stripped).
_MOE_RE = re.compile(r"(\d+)x(\d+(?:\.\d+)?)b\b")
_PLAIN_RE = re.compile(r"(\d+(?:\.\d+)?)b(?=[-_]|$)")

# Tier boundaries in billions of parameters, largest tier first. [CMV]
_TIER_THRESHOLDS_B: tuple[float, ...] = (100.0, 40.0, 15.0)
UNKNOWN_TIER = len(_TIER_THRESHOLDS_B)  # sits between the 15B+ and <15B tiers
SMALLEST_TIER = UNKNOWN_TIER + 1


def extract_param_billions(model_id: str) -> float | None:
    """Best-effort parse of a model's parameter count, in billions.

    Handles plain notation (``70b``, ``0.5b``) and MoE "AxB" notation
    (``8x7b`` -> total 56B). Returns ``None`` when the slug carries no size
    hint at all -- callers must not treat that as "small".
    """
    if not model_id:
        return None
    leaf = model_id.strip().lower().split("/")[-1]
    leaf = leaf.split(":", 1)[0]  # drop ":free" and similar variant suffixes

    moe = _MOE_RE.search(leaf)
    if moe:
        try:
            return float(moe.group(1)) * float(moe.group(2))
        except ValueError:
            pass  # fall through to plain-notation parsing

    sizes = [float(m) for m in _PLAIN_RE.findall(leaf)]
    return max(sizes) if sizes else None


def param_tier(model_id: str) -> int:
    """Bucket a model into a coarse size tier. 0 sorts first (largest).

    Unknown-size models sit at ``UNKNOWN_TIER``, between the >=15B and <15B
    tiers -- see module docstring for why.
    """
    billions = extract_param_billions(model_id)
    if billions is None:
        return UNKNOWN_TIER
    for tier, threshold in enumerate(_TIER_THRESHOLDS_B):
        if billions >= threshold:
            return tier
    return SMALLEST_TIER
