"""Compatibility helpers for PyTorch usage across the codebase.

These helpers intentionally avoid importing heavy subsystems at module import time.
"""

from __future__ import annotations

import logging
import sys
import warnings
from typing import Any

logger = logging.getLogger(__name__)


def ensure_reduce_op_alias() -> None:
    """Rebind torch.distributed.reduce_op to ReduceOp to silence deprecation warnings.

    PyTorch 2.3+ emits a warning whenever the legacy attribute is accessed.
    Some third-party libraries still rely on the legacy name, so we replace it
    with the modern enum to avoid the warning while preserving behaviour.

    Torch-gated: this shim is only meaningful when torch is actually in use,
    and importing torch.distributed here pulled the full ~300-400 MB torch
    runtime into RSS on every startup — even now that STT (ctranslate2) and
    RAG (fastembed/onnxruntime) no longer need torch at all. Only act when
    some other subsystem has already paid the import cost. [PA]
    """
    warnings.filterwarnings(
        "ignore",
        message="torch.distributed.reduce_op is deprecated",
        category=UserWarning,
    )

    if "torch" not in sys.modules:
        logger.debug("torch not loaded; reduce_op shim skipped")
        return

    try:
        import torch.distributed as dist  # type: ignore
    except Exception:
        return

    try:
        reduce_enum: Any = dist.ReduceOp  # type: ignore[attr-defined]
    except AttributeError:
        return
    except Exception:
        return

    try:
        # Rebind without reading dist.reduce_op first; getattr would re-trigger the warning.
        dist.reduce_op = reduce_enum
        logger.debug("torch.compat.reduce_op_rebound")
    except Exception as exc:
        # Safety-first: avoid failing import-time logic because of torch internals.
        logger.debug(f"reduce_op rebind failed: {exc}")
