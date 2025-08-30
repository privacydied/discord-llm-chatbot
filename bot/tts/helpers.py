# -*- coding: utf-8 -*-
"""
Helper utilities for stub-safe access patterns in TTS engines.

Currently provides:
- maybe_onnx_session(obj): Safely extract an ONNX session-like object from
  kokoro drivers or stubs by checking common attribute names ("onnx_session",
  "sess"). Returns None if not present.
"""
from __future__ import annotations
from typing import Any, Optional

__all__ = ["maybe_onnx_session"]


def _maybe_attr(obj: Any, *names: str) -> Optional[Any]:
    for name in names:
        try:
            val = getattr(obj, name, None)
        except Exception:
            val = None
        if val is not None:
            return val
    return None


def maybe_onnx_session(obj: Any) -> Optional[Any]:
    """Return an ONNX session-like object from obj if available.

    Checks common attributes used by our engines and tests ("onnx_session",
    then "sess"). Returns None when not present or inaccessible.
    """
    return _maybe_attr(obj, "onnx_session", "sess")
