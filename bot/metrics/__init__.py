"""Metrics interface and provider selection.

Provides a safe no-op fallback (NullMetrics) and optional Prometheus-based
implementation. Exposes a simple interface used across the codebase:

  - increment(name, labels: dict | None = None, value: int = 1)
  - inc(name, value: int = 1, labels: dict | None = None)
  - observe(name, value: float, labels: dict | None = None)
  - gauge(name, value: float, labels: dict | None = None)
  - timer(name, labels: dict | None = None) -> context manager

Backwards compatible export: `NullMetrics` is available and implements all methods.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from .null_metrics import NoopMetrics as NullMetrics  # Back-compat alias


@runtime_checkable
class Metrics(Protocol):
    def inc(self, name: str, value: int = 1, labels: Optional[dict] = None) -> None: ...
    def increment(self, name: str, labels: Optional[dict] = None, value: int = 1) -> None: ...
    def observe(self, name: str, value: float, labels: Optional[dict] = None) -> None: ...
    def gauge(self, name: str, value: float, labels: Optional[dict] = None) -> None: ...
    def timer(self, name: str, labels: Optional[dict] = None): ...


def get_metrics() -> Metrics:
    """Return a metrics provider instance.

    Attempts to initialize a Prometheus metrics provider if available; otherwise
    falls back to the safe no-op provider (NullMetrics).
    """
    try:
        # Delayed import to avoid hard dependency when not installed
        from .prometheus_metrics import PrometheusMetrics  # type: ignore
        # If the module exists and class is importable, return a Prometheus provider
        return PrometheusMetrics()  # type: ignore
    except Exception:
        # Any failure (module missing or runtime issue) falls back to no-op
        return NullMetrics()


# Optional module-level singleton that callers may use directly if convenient
metrics: Metrics = get_metrics()

