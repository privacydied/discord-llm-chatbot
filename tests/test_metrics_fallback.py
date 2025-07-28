import unittest
from unittest.mock import patch

class TestMetricsFallback(unittest.TestCase):

    def test_null_metrics_import(self):
        from bot.metrics import NullMetrics

    @patch.dict('sys.modules', {'prometheus_client': None})
    def test_prometheus_metrics_import_fails_without_prometheus_client(self):
        with self.assertRaises(ImportError):
            from bot.metrics.prometheus_metrics import PrometheusMetrics

if __name__ == '__main__':
    unittest.main()
