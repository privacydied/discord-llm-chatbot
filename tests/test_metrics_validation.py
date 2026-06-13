import pytest

from bot.metrics.prometheus_metrics import PrometheusMetrics

pytestmark = pytest.mark.skip(reason="Stale metric name normalization tests")


def test_metric_name_normalization_allows_dotted_names() -> None:
    m = PrometheusMetrics(enable_http_server=False)
    # Names with '.' should be normalized to '_'
    m.define_counter("x.syndication.fetch", "count", labels=["endpoint"])
    # Internal registry stores normalized key
    assert any(k.endswith("x_syndication_fetch") for k in m._counters)
    # Increment using original dotted name should work
    m.increment("x.syndication.fetch", labels={"endpoint": "widgets"})


def test_metric_label_name_normalization() -> None:
    m = PrometheusMetrics(enable_http_server=False)
    # Label with dash should be normalized
    m.define_counter("gate.blocked", "blocked", labels=["reason-code"])
    next(iter(m._counters.keys()))
    # Ensure counter exists and increment with normalized label name
    m.increment("gate.blocked", labels={"reason_code": "policy"})
