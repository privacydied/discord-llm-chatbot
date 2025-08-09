"""No-op metrics implementation that provides safe no-op methods."""

from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)

class NoopMetrics:
    """Metrics provider that does nothing but implements full interface safely."""
    
    def __init__(self):
        logger.info("📊 Prometheus disabled: using NoopMetrics")
    
    def define_counter(self, name: str, description: str, labels: Optional[list] = None) -> None:
        """Define a counter metric (no-op)."""
        pass
        
    def define_histogram(self, name: str, description: str, labels: Optional[list] = None, buckets: Optional[tuple] = None) -> None:
        """Define a histogram metric (no-op)."""
        pass
        
    def inc(self, name: str, value: int = 1, labels: Optional[dict] = None) -> None:
        """Increment a counter (no-op)."""
        pass
    
    def increment(self, name: str, labels: Optional[dict] = None, value: int = 1) -> None:
        """Increment a counter (no-op) - alternative interface."""
        pass
        
    def observe(self, name: str, value: float, labels: Optional[dict] = None) -> None:
        """Observe a histogram value (no-op)."""
        pass
    
    def gauge(self, name: str, value: float, labels: Optional[dict] = None) -> None:
        """Set a gauge value (no-op)."""
        pass
    
    def timer(self, name: str, labels: Optional[dict] = None):
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