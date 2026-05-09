from __future__ import annotations

import builtins

from bot.utils.logging import init_logging


def test_init_logging_disables_rich_tracebacks_when_pygments_missing(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("pygments"):
            raise ModuleNotFoundError("No module named 'pygments'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    init_logging()
