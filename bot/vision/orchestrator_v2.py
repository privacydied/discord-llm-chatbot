"""Vision Job Orchestrator V2 - compatibility shim [CA].

The full ``VisionOrchestratorV2`` implementation that once lived here was dead
code: the module always overwrote it with the canonical
``bot.vision.orchestrator.VisionOrchestrator`` at import time, so the ~500-line
class body below the shim was never instantiated. Maintaining a second, subtly
different, unreachable orchestrator is a hazard (someone "fixing a bug" here
would be editing code nothing calls), so the body has been removed.

This module now only re-exports the canonical orchestrator so existing import
paths keep working unchanged::

    from bot.vision.orchestrator_v2 import VisionOrchestratorV2

``VisionJobStore`` is re-exported as well because it is referenced through this
module namespace (e.g. ``patch("bot.vision.orchestrator_v2.VisionJobStore")``).
"""

from __future__ import annotations

from bot.utils.logging import get_logger

from .job_store import VisionJobStore
from .orchestrator import VisionOrchestrator

logger = get_logger(__name__)

# Canonical orchestrator; VisionOrchestratorV2 is retained purely as an alias so
# that any `from .orchestrator_v2 import VisionOrchestratorV2` keeps resolving.
VisionOrchestratorV2 = VisionOrchestrator

__all__ = ["VisionJobStore", "VisionOrchestrator", "VisionOrchestratorV2"]
