"""No-op metrics implementation that provides safe no-op methods."""

import logging

logger = logging.getLogger(__name__)


class NoopMetrics:
    """Metrics provider that does nothing but implements full interface safely."""

    def __init__(self) -> None:
        logger.info("📊 Prometheus disabled: using NoopMetrics")

    def define_counter(self, name: str, description: str, labels: list | None = None) -> None:
        """Define a counter metric (no-op)."""

    def define_histogram(
        self,
        name: str,
        description: str,
        labels: list | None = None,
        buckets: tuple | None = None,
    ) -> None:
        """Define a histogram metric (no-op)."""

    def inc(self, name: str, value: int = 1, labels: dict | None = None) -> None:
        """Increment a counter (no-op)."""

    def increment(self, name: str, labels: dict | None = None, value: int = 1) -> None:
        """Increment a counter (no-op) - alternative interface."""

    def observe(self, name: str, value: float, labels: dict | None = None) -> None:
        """Observe a histogram value (no-op)."""

    def gauge(self, name: str, value: float, labels: dict | None = None) -> None:
        """Set a gauge value (no-op)."""

    def timer(self, name: str, labels: dict | None = None):
        """Context manager for timing (no-op)."""
        return NoopTimer()


class NoopTimer:
    """No-op timer context manager."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# Backward compatibility
NullMetrics = NoopMetrics
