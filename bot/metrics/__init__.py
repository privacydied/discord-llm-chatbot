class NullMetrics:
    def inc(self, *a, **k): pass
    def observe(self, *a, **k): pass
    def gauge(self, *a, **k): pass

