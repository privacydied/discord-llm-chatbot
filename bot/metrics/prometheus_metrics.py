"""Prometheus-based metrics implementation for bot monitoring and observability."""

from typing import Dict, Optional, Any
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
    
    def __init__(self, port: int = 8000):
        """Initialize Prometheus metrics with HTTP server for scraping.
        
        Args:
            port: Port for Prometheus metrics HTTP server
        """
        self.port = port
        self._counters: Dict[str, Counter] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._http_server_started = False
        
        logger.info(f"📊 Initializing Prometheus metrics on port {port}")
        
        # Start HTTP server for metrics scraping
        try:
            start_http_server(port)
            self._http_server_started = True
            logger.info(f"✅ Prometheus HTTP server started on port {port}")
        except Exception as e:
            logger.warning(f"⚠️  Failed to start Prometheus HTTP server: {e}")
    
    def define_counter(self, name: str, description: str, labels: Optional[list] = None) -> None:
        """Define a counter metric.
        
        Args:
            name: Metric name
            description: Metric description
            labels: List of label names
        """
        if name not in self._counters:
            self._counters[name] = Counter(
                name=name,
                documentation=description,
                labelnames=labels or []
            )
            logger.debug(f"📈 Defined counter: {name}")
    
    def define_histogram(self, name: str, description: str, labels: Optional[list] = None, 
                        buckets: Optional[tuple] = None) -> None:
        """Define a histogram metric.
        
        Args:
            name: Metric name
            description: Metric description
            labels: List of label names
            buckets: Histogram buckets
        """
        if name not in self._histograms:
            kwargs = {
                'name': name,
                'documentation': description,
                'labelnames': labels or []
            }
            if buckets:
                kwargs['buckets'] = buckets
                
            self._histograms[name] = Histogram(**kwargs)
            logger.debug(f"📊 Defined histogram: {name}")
    
    def define_gauge(self, name: str, description: str, labels: Optional[list] = None) -> None:
        """Define a gauge metric.
        
        Args:
            name: Metric name  
            description: Metric description
            labels: List of label names
        """
        if name not in self._gauges:
            self._gauges[name] = Gauge(
                name=name,
                documentation=description,
                labelnames=labels or []
            )
            logger.debug(f"📏 Defined gauge: {name}")
    
    def inc(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter.
        
        Args:
            name: Counter name
            value: Increment value
            labels: Label values
        """
        if name in self._counters:
            if labels:
                self._counters[name].labels(**labels).inc(value)
            else:
                self._counters[name].inc(value)
        else:
            logger.warning(f"⚠️  Counter '{name}' not defined")
    
    def increment(self, name: str, labels: Optional[Dict[str, str]] = None, value: int = 1) -> None:
        """Increment a counter (alternative interface).
        
        Args:
            name: Counter name
            labels: Label values
            value: Increment value
        """
        self.inc(name, value, labels)
    
    def observe(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Observe a histogram value.
        
        Args:
            name: Histogram name
            value: Value to observe
            labels: Label values
        """
        if name in self._histograms:
            if labels:
                self._histograms[name].labels(**labels).observe(value)
            else:
                self._histograms[name].observe(value)
        else:
            logger.warning(f"⚠️  Histogram '{name}' not defined")
    
    def gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge value.
        
        Args:
            name: Gauge name
            value: Gauge value
            labels: Label values
        """
        if name in self._gauges:
            if labels:
                self._gauges[name].labels(**labels).set(value)
            else:
                self._gauges[name].set(value)
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
