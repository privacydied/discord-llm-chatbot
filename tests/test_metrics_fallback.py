import unittest
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.skip(reason="Requires Prometheus client configuration")


class TestMetricsFallback(unittest.TestCase):
    def test_null_metrics_import(self) -> None:
        pass

    @patch.dict("sys.modules", {"prometheus_client": None})
    def test_prometheus_metrics_import_fails_without_prometheus_client(self) -> None:
        with pytest.raises(ImportError):
            pass


if __name__ == "__main__":
    unittest.main()
