"""Null metrics implementation that provides no-op methods."""

from typing import Any, Optional

class NullMetrics:
    """Metrics provider that does nothing."""
    
    def define_counter(self, name: str, description: str) -> None:
        """Define a counter metric (no-op)."""
        pass
        
    def define_histogram(self, name: str, description: str) -> None:
        """Define a histogram metric (no-op)."""
        pass
        
    def inc(self, name: str, value: int = 1, labels: Optional[dict] = None) -> None:
        """Increment a counter (no-op)."""
        pass
        
    def observe(self, name: str, value: float, labels: Optional[dict] = None) -> None:
        """Observe a histogram value (no-op)."""
        pass