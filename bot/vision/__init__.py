"""Vision Generation System - Provider-agnostic image and video generation.

Core components for the vision generation system:
- VisionGateway: Provider-agnostic facade for Together.ai and Novita.ai
- VisionOrchestrator: Job management and async execution
- VisionJobStore: JSON-based job persistence
- VisionIntentRouter: Routing between Vision and OpenRouter
- VisionSafetyFilter: Content safety validation
- VisionBudgetManager: Cost tracking and quota enforcement
- VisionArtifactCache: Generated content caching
- Vision types: Request/response models and enums

Usage:
    from bot.vision import VisionGateway, VisionOrchestrator

    gateway = VisionGateway(config)
    orchestrator = VisionOrchestrator(config)

    # Submit vision generation job
    job = await orchestrator.submit_job(request)
"""

from .artifact_cache import ArtifactMetadata, CacheStats, VisionArtifactCache
from .budget_manager import BudgetResult, VisionBudgetManager
from .gateway import VisionGateway
from .intent_router import VisionIntentRouter
from .job_store import VisionJobStore
from .orchestrator import VisionOrchestrator
from .safety_filter import SafetyResult, VisionSafetyFilter
from .types import (
    RoutingDecision,
    VisionError,
    VisionErrorType,
    VisionJob,
    VisionJobState,
    VisionProvider,
    VisionRequest,
    VisionResponse,
    VisionTask,
)

__all__ = [
    "ArtifactMetadata",
    "BudgetResult",
    "CacheStats",
    "RoutingDecision",
    "SafetyResult",
    "VisionArtifactCache",
    "VisionBudgetManager",
    "VisionError",
    "VisionErrorType",
    "VisionGateway",
    "VisionIntentRouter",
    "VisionJob",
    "VisionJobState",
    "VisionJobStore",
    "VisionOrchestrator",
    "VisionProvider",
    "VisionRequest",
    "VisionResponse",
    "VisionSafetyFilter",
    "VisionTask",
]
