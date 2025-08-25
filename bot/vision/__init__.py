"""
Vision Generation System - Provider-agnostic image and video generation

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

from .gateway import VisionGateway
from .orchestrator import VisionOrchestrator  
from .job_store import VisionJobStore
from .intent_router import VisionIntentRouter
from .safety_filter import VisionSafetyFilter, SafetyResult
from .budget_manager import VisionBudgetManager, BudgetResult
from .artifact_cache import VisionArtifactCache, ArtifactMetadata, CacheStats
from .types import (
    VisionRequest, VisionResponse, VisionJob,
    VisionTask, VisionProvider, VisionJobState,
    VisionError, VisionErrorType, IntentDecision, IntentResult
)

__all__ = [
    'VisionGateway',
    'VisionOrchestrator', 
    'VisionJobStore',
    'VisionIntentRouter',
    'VisionSafetyFilter',
    'VisionBudgetManager',
    'VisionArtifactCache',
    'VisionRequest',
    'VisionResponse', 
    'VisionJob',
    'VisionTask',
    'VisionProvider',
    'VisionJobState',
    'VisionError',
    'VisionErrorType',
    'IntentDecision',
    'IntentResult',
    'SafetyResult',
    'BudgetResult',
    'ArtifactMetadata',
    'CacheStats'
]
