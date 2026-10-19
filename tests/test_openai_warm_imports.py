"""openai's lazy resource modules are importable ahead of the first request [PA]."""

import sys

from bot import openai_backend


def test_warm_openai_imports_loads_resources():
    openai_backend.warm_openai_imports()
    assert "openai.resources" in sys.modules
    assert "openai.resources.chat" in sys.modules


def test_warm_openai_imports_noop_without_openai(monkeypatch):
    monkeypatch.setattr(openai_backend, "_openai", None)
    openai_backend.warm_openai_imports()  # must not raise
