"""
Pytest configuration for Kokoro-related tests.

Ensures the vendored IPA vocabulary is allowed so KokoroDirect can load
phoneme-to-id mapping in environments without the official assets.
"""

import os
import pytest
from unittest import mock

# Allow vendored IPA vocabulary for KokoroDirect tests (safe in CI)
os.environ.setdefault("KOKORO_ALLOW_VENDORED_VOCAB", "true")


@pytest.fixture
def mocker(request):
    """
    Lightweight replacement for pytest-mock's `mocker` fixture.
    Provides:
    - mocker.patch(target, ...) -> started mock (auto-teardown on test end)
    - mocker.mock_open(...) -> unittest.mock.mock_open
    """

    class _SimpleMocker:
        def patch(self, target, *args, **kwargs):
            patcher = mock.patch(target, *args, **kwargs)
            started = patcher.start()
            request.addfinalizer(patcher.stop)
            return started

        @staticmethod
        def mock_open(*args, **kwargs):
            return mock.mock_open(*args, **kwargs)

    return _SimpleMocker()
