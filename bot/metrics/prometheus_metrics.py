"""Prometheus-based metrics implementation for bot monitoring and observability."""

from typing import Dict, Optional
import re
import time
from contextlib import contextmanager
import logging

try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server, REGISTRY
except ImportError as e:
    raise ImportError(
        "prometheus_client is required for PrometheusMetrics. "
        "Install it with: pip install prometheus-client"
    ) from e

logger = logging.getLogger(__name__)


class PrometheusMetrics:
    """Prometheus-based metrics provider for comprehensive bot monitoring."""

    def __init__(self, port: int = 8000, enable_http_server: bool = True):
        """Initialize Prometheus metrics with optional HTTP server for scraping.

        Args:
            port: Port for Prometheus metrics HTTP server (0 for auto-select)
            enable_http_server: Whether to start HTTP server for metrics scraping
        """
        self.port = port
        self.enable_http_server = enable_http_server
        self._counters: Dict[str, Counter] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._http_server_started = False

        logger.info(
            f"📊 Initializing Prometheus metrics (HTTP server: {enable_http_server}, port: {port})"
        )

        # Start HTTP server for metrics scraping if enabled
        if enable_http_server:
            try:
                actual_port = start_http_server(port)
                self._http_server_started = True
                self.port = actual_port if port == 0 else port
                logger.info(f"✅ Prometheus HTTP server started on port {self.port}")
            except Exception as e:
                logger.warning(f"⚠️  Failed to start Prometheus HTTP server: {e}")
        else:
            logger.info("📊 Prometheus metrics initialized without HTTP server")

    def define_counter(
        self, name: str, description: str, labels: Optional[list] = None
    ) -> None:
        """Define a counter metric.

        Args:
            name: Metric name
            description: Metric description
            labels: List of label names
        """
        norm_name = self._normalize_metric_name(name)
        norm_labels = self._normalize_label_names(labels or [])
        if name != norm_name:
            logger.debug(f"Normalizing counter name: '{name}' -> '{norm_name}'")
        if labels and norm_labels != labels:
            logger.debug(f"Normalizing counter labels: {labels} -> {norm_labels}")
        if norm_name not in self._counters:
            self._counters[norm_name] = Counter(
                name=norm_name, documentation=description, labelnames=norm_labels
            )
            logger.debug(f"📈 Defined counter: {norm_name}")

    def define_histogram(
        self,
        name: str,
        description: str,
        labels: Optional[list] = None,
        buckets: Optional[tuple] = None,
    ) -> None:
        """Define a histogram metric.

        Args:
            name: Metric name
            description: Metric description
            labels: List of label names
            buckets: Histogram buckets
        """
        norm_name = self._normalize_metric_name(name)
        norm_labels = self._normalize_label_names(labels or [])
        if name != norm_name:
            logger.debug(f"Normalizing histogram name: '{name}' -> '{norm_name}'")
        if labels and norm_labels != labels:
            logger.debug(f"Normalizing histogram labels: {labels} -> {norm_labels}")
        if norm_name not in self._histograms:
            kwargs = {
                "name": norm_name,
                "documentation": description,
                "labelnames": norm_labels,
            }
            if buckets:
                kwargs["buckets"] = buckets

            self._histograms[norm_name] = Histogram(**kwargs)
            logger.debug(f"📊 Defined histogram: {norm_name}")

    def define_gauge(
        self, name: str, description: str, labels: Optional[list] = None
    ) -> None:
        """Define a gauge metric.

        Args:
            name: Metric name
            description: Metric description
            labels: List of label names
        """
        norm_name = self._normalize_metric_name(name)
        norm_labels = self._normalize_label_names(labels or [])
        if name != norm_name:
            logger.debug(f"Normalizing gauge name: '{name}' -> '{norm_name}'")
        if labels and norm_labels != labels:
            logger.debug(f"Normalizing gauge labels: {labels} -> {norm_labels}")
        if norm_name not in self._gauges:
            self._gauges[norm_name] = Gauge(
                name=norm_name, documentation=description, labelnames=norm_labels
            )
            logger.debug(f"📏 Defined gauge: {norm_name}")

    def inc(
        self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Increment a counter.

        Args:
            name: Counter name
            value: Increment value
            labels: Label values
        """
        norm_name = self._normalize_metric_name(name)
        norm_labels = labels or None
        if norm_name in self._counters:
            if norm_labels:
                self._counters[norm_name].labels(**norm_labels).inc(value)
            else:
                self._counters[norm_name].inc(value)
        else:
            logger.warning(f"⚠️  Counter '{name}' not defined")

    def increment(
        self, name: str, labels: Optional[Dict[str, str]] = None, value: int = 1
    ) -> None:
        """Increment a counter (alternative interface).

        Args:
            name: Counter name
            labels: Label values
            value: Increment value
        """
        self.inc(name, value, labels)

    def observe(
        self, name: str, value: float, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Observe a histogram value.

        Args:
            name: Histogram name
            value: Value to observe
            labels: Label values
        """
        norm_name = self._normalize_metric_name(name)
        norm_labels = labels or None
        if norm_name in self._histograms:
            if norm_labels:
                self._histograms[norm_name].labels(**norm_labels).observe(value)
            else:
                self._histograms[norm_name].observe(value)
        else:
            logger.warning(f"⚠️  Histogram '{name}' not defined")

    def gauge(
        self, name: str, value: float, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Set a gauge value.

        Args:
            name: Gauge name
            value: Gauge value
            labels: Label values
        """
        norm_name = self._normalize_metric_name(name)
        norm_labels = labels or None
        if norm_name in self._gauges:
            if norm_labels:
                self._gauges[norm_name].labels(**norm_labels).set(value)
            else:
                self._gauges[norm_name].set(value)
        else:
            logger.warning(f"⚠️  Gauge '{name}' not defined")

    @contextmanager
    def timer(self, name: str, labels: Optional[Dict[str, str]] = None):
        """Context manager for timing operations.

        Args:
            name: Histogram name for timing
            labels: Label values
        """
        if name not in self._histograms:
            logger.warning(f"⚠️  Timer histogram '{name}' not defined")
            yield
            return

        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.observe(name, duration, labels)

    def get_registry(self):
        """Get the Prometheus registry for advanced usage."""
        return REGISTRY

    def _normalize_metric_name(self, name: str) -> str:
        """Normalize metric name to Prometheus spec.

        - Replace invalid characters with '_'
        - Ensure leading char is [a-zA-Z_:] (prefix with 'm_' if not)
        """
        # Replace anything not allowed with underscore
        norm = re.sub(r"[^a-zA-Z0-9_:]", "_", name)
        # If starts with invalid, prefix
        if not re.match(r"^[a-zA-Z_:]", norm):
            norm = f"m_{norm}"
        return norm

    def _normalize_label_names(self, labels: list) -> list:
        """Normalize label names to Prometheus spec."""
        normed = []
        for label in labels:
            if not isinstance(label, str):
                continue
            n = re.sub(r"[^a-zA-Z0-9_]", "_", label)
            if not re.match(r"^[a-zA-Z_]", n):
                n = f"l_{n}"
            normed.append(n)
        return normed
