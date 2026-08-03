"""Every counter incremented via Router._metric_inc must be pre-registered. [REH][CMV]

`PrometheusMetrics.inc()` only recognizes a counter name if `define_counter()`
was called for it first (see bot/metrics/prometheus_metrics.py); otherwise it
silently falls through to a "Counter '...' not defined" warning and the metric
is dropped. Registration lives in a hand-maintained list inside
`LLMBot.setup_hook()` (bot/core/bot.py), so a new `self._metric_inc("x.foo", ...)`
call site in router.py that forgets the matching `define_counter("x.foo", ...)`
call fails silently at runtime with no test failure.

This mirrors the tests/test_config_budget_keys.py precedent (b719ad1), which
caught the same "call site reads it, but the allowlist never learned about it"
bug class for config knobs. Here the allowlist is `self._counters` populated
by define_counter(); this test statically diffs the two literal-string sets
instead of instantiating the whole bot (which requires live Discord/DB config).

Regression: x.tweet_image_only.{syndication,api} and vision.image_only_tweet.*
(8 counters total) were incremented in router.py but never registered, so the
image-only-tweet pipeline emitted "not defined" warnings on every OCR run.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ROUTER_PATH = REPO_ROOT / "bot" / "router.py"
BOT_PATH = REPO_ROOT / "bot" / "core" / "bot.py"

# Matches: self._metric_inc(\n    "name", ...)  or  self._metric_inc("name", ...)
_METRIC_INC_RE = re.compile(r'_metric_inc\(\s*\n?\s*"([^"]+)"')
# Matches: self.metrics.define_counter(\n    "name", ...)
_DEFINE_COUNTER_RE = re.compile(r'define_counter\(\s*\n?\s*"([^"]+)"')


def _incremented_counter_names() -> set[str]:
    text = ROUTER_PATH.read_text(encoding="utf-8")
    return set(_METRIC_INC_RE.findall(text))


def _registered_counter_names() -> set[str]:
    text = BOT_PATH.read_text(encoding="utf-8")
    return set(_DEFINE_COUNTER_RE.findall(text))


def test_router_finds_metric_inc_call_sites() -> None:
    """Sanity check the regex still matches the real call-site style."""
    assert len(_incremented_counter_names()) >= 20


def test_bot_finds_define_counter_registrations() -> None:
    """Sanity check the regex still matches the real registration style."""
    assert len(_registered_counter_names()) >= 20


def test_every_router_counter_is_registered() -> None:
    incremented = _incremented_counter_names()
    registered = _registered_counter_names()
    missing = sorted(incremented - registered)
    assert not missing, (
        f"Counter(s) incremented via _metric_inc() in router.py but never "
        f"registered via define_counter() in bot.py's setup_hook(): {missing}. "
        "They will silently no-op with a 'Counter ... not defined' warning at "
        "runtime. Add a matching self.metrics.define_counter(...) call in "
        "bot/core/bot.py."
    )
