"""Enhanced metrics interface and provider selection with degraded mode support.

Provides a thin metrics interface with safe no-op fallback (NoopMetrics) and 
optional Prometheus-based implementation controlled by environment variables.

Interface methods:
  - increment(name, labels: dict | None = None, value: int = 1)
  - inc(name, value: int = 1, labels: dict | None = None) 
  - observe(name, value: float, labels: dict | None = None)
  - gauge(name, value: float, labels: dict | None = None)
  - timer(name, labels: dict | None = None) -> context manager

Environment variables:
  - OBS_ENABLE_PROMETHEUS=false (default, uses NoopMetrics)
  - OBS_ENABLE_RESOURCE_METRICS=true (enables RSS/event loop tracking)
  - OBS_ENABLE_HEALTHCHECKS=true (enables health monitoring)

[RAT: PA, REH, CMV] - Performance Awareness, Robust Error Handling, Constants over Magic Values
"""

from __future__ import annotations

import os
import logging
from typing import Optional, Protocol, runtime_checkable

from .null_metrics import NoopMetrics as NullMetrics  # Back-compat alias

# Global state for degraded mode tracking
_degraded_mode = False
_degraded_reasons = []


@runtime_checkable
class Metrics(Protocol):
    def inc(self, name: str, value: int = 1, labels: Optional[dict] = None) -> None: ...
    def increment(self, name: str, labels: Optional[dict] = None, value: int = 1) -> None: ...
    def observe(self, name: str, value: float, labels: Optional[dict] = None) -> None: ...
    def gauge(self, name: str, value: float, labels: Optional[dict] = None) -> None: ...
    def timer(self, name: str, labels: Optional[dict] = None): ...


def get_metrics() -> Metrics:
    """Return a metrics provider instance with environment-controlled selection.

    Uses environment variables to determine metrics provider:
    - OBS_ENABLE_PROMETHEUS=true: Attempt Prometheus, fallback to Noop on failure
    - OBS_ENABLE_PROMETHEUS=false (default): Use NoopMetrics for zero overhead
    
    On Prometheus initialization failure: warns, sets degraded_mode, falls back to Noop.
    Thread/async-safe, no blocking on hot paths.
    
    [RAT: PA, REH, CMV] - Performance Awareness, Robust Error Handling, Constants over Magic Values
    """
    global _degraded_mode, _degraded_reasons
    
    logger = logging.getLogger(__name__)
    
    # Check environment flag for Prometheus enablement
    prometheus_enabled = os.getenv("OBS_ENABLE_PROMETHEUS", "false").lower() == "true"
    
    if not prometheus_enabled:
        # Default path: zero-overhead NoopMetrics
        logger.info("📊 Prometheus disabled by config: using NoopMetrics", extra={"subsys": "metrics"})
        return NullMetrics()
    
    # Prometheus enabled: attempt initialization with graceful fallback
    try:
        # Delayed import to avoid hard dependency when not installed
        from .prometheus_metrics import PrometheusMetrics  # type: ignore
        
        # Get optional configuration
        prometheus_port = int(os.getenv("PROMETHEUS_PORT", "8001"))
        http_server_enabled = os.getenv("PROMETHEUS_HTTP_SERVER", "true").lower() == "true"
        
        metrics_instance = PrometheusMetrics(
            port=prometheus_port,
            enable_http_server=http_server_enabled
        )
        
        logger.info("📊 Prometheus metrics initialized successfully", extra={"subsys": "metrics"})
        return metrics_instance
        
    except ImportError as e:
        # Missing prometheus-client dependency
        reason = f"prometheus-client not installed: {e}"
        _degraded_mode = True
        _degraded_reasons.append(reason)
        logger.warning(f"📊 Prometheus unavailable, falling back to NoopMetrics: {reason}", extra={"subsys": "metrics"})
        return NullMetrics()
        
    except Exception as e:
        # Runtime initialization failure (port bind, etc.)
        reason = f"Prometheus init failed: {e}"
        _degraded_mode = True
        _degraded_reasons.append(reason)
        logger.warning(f"📊 Prometheus failed to initialize, falling back to NoopMetrics: {reason}", extra={"subsys": "metrics"})
        return NullMetrics()


def is_degraded_mode() -> bool:
    """Return True if metrics system is in degraded mode.
    
    Degraded mode indicates that Prometheus was requested but failed to initialize,
    causing fallback to NoopMetrics.
    """
    return _degraded_mode


def get_degraded_reasons() -> list[str]:
    """Return list of reasons why metrics system is in degraded mode."""
    return _degraded_reasons.copy()


def reset_degraded_mode() -> None:
    """Reset degraded mode state (for testing/recovery scenarios)."""
    global _degraded_mode, _degraded_reasons
    _degraded_mode = False
    _degraded_reasons.clear()


# Standard metric name constants [CMV]
METRIC_STARTUP_TOTAL_DURATION = "bot_startup_total_duration_seconds"
METRIC_STARTUP_COMPONENT_DURATION = "bot_startup_component_duration_seconds"
METRIC_STARTUP_PARALLEL_GROUPS = "bot_startup_parallel_groups_total"
METRIC_COMPONENT_INIT_SUCCESS = "bot_component_init_success_total"
METRIC_COMPONENT_INIT_FAILURE = "bot_component_init_failure_total"
METRIC_COMPONENT_LAST_INIT_TIMESTAMP = "bot_component_last_init_timestamp"
METRIC_DEGRADED_MODE = "bot_degraded_mode"
METRIC_BACKGROUND_HEARTBEAT = "bot_background_heartbeat_total"
METRIC_BACKGROUND_LAST_HEARTBEAT = "bot_background_last_heartbeat_timestamp"
METRIC_BACKGROUND_CONSECUTIVE_ERRORS = "bot_background_consecutive_errors"
METRIC_BACKGROUND_STALENESS_SECONDS = "bot_background_staleness_seconds"
METRIC_PROCESS_RSS_BYTES = "bot_process_rss_bytes"
METRIC_EVENT_LOOP_LAG_SECONDS = "bot_event_loop_lag_seconds"
METRIC_ERRORS_BY_MODULE = "bot_errors_by_module_total"

# Optional module-level singleton that callers may use directly if convenient
metrics: Metrics = get_metrics()

