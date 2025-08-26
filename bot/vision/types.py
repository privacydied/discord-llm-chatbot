"""
Core types and data models for Vision Generation System

Defines standardized interfaces following Clean Architecture (CA) and
type-safe patterns for provider-agnostic vision generation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Literal
import uuid
from datetime import datetime, timezone
import json


class VisionTask(Enum):
    """Supported vision generation tasks"""
    TEXT_TO_IMAGE = "text_to_image"
    IMAGE_TO_IMAGE = "image_to_image"
    TEXT_TO_VIDEO = "text_to_video" 
    IMAGE_TO_VIDEO = "image_to_video"
    VIDEO_GENERATION = "video_generation"


class VisionProvider(Enum):
    """Supported vision providers"""
    TOGETHER = "together"
    NOVITA = "novita"


class VisionJobState(Enum):
    """Job state machine for orchestration"""
    CREATED = "created"
    QUEUED = "queued"
    VALIDATING = "validating"
    RUNNING = "running"
    POLLING = "polling"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class VisionErrorType(Enum):
    """Categorized error types for proper handling"""
    VALIDATION_ERROR = "validation_error"
    QUOTA_EXCEEDED = "quota_exceeded"
    CONTENT_FILTERED = "content_filtered"
    PROVIDER_ERROR = "provider_error"
    CONNECTION_ERROR = "connection_error"  # For HTTP/network connection failures [REH]
    RATE_LIMITED = "rate_limited"  # For rate limiting responses [REH]
    TIMEOUT_ERROR = "timeout_error"
    NETWORK_ERROR = "network_error"
    SYSTEM_ERROR = "system_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class VisionError(Exception):
    """Structured error with categorization and user-friendly messaging"""
    error_type: VisionErrorType
    message: str
    user_message: str
    provider: Optional[VisionProvider] = None
    retry_after_seconds: Optional[int] = None
    details: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        return f"{self.error_type.value}: {self.message}"


@dataclass
class VisionRequest:
    """Normalized request for any vision generation task"""
    task: VisionTask
    prompt: str
    user_id: str
    guild_id: Optional[str] = None
    channel_id: str = ""
    
    # Image generation parameters
    width: int = 1024
    height: int = 1024
    steps: int = 30
    guidance_scale: float = 7.0
    negative_prompt: str = ""
    seed: Optional[int] = None
    batch_size: int = 1
    
    # Image editing parameters  
    input_image: Optional[Path] = None
    input_image_data: Optional[bytes] = None  # Raw image bytes [CA]
    input_image_url: Optional[str] = None     # Image URL [CA]
    mask_image: Optional[Path] = None
    strength: float = 0.8
    
    # Video generation parameters
    duration_seconds: int = 3
    fps: int = 24
    style: str = "natural"
    
    # Image-to-video parameters
    mode: Literal["image2video", "start_end"] = "image2video"
    end_image: Optional[Path] = None
    
    # Provider preferences
    preferred_provider: Optional[VisionProvider] = None
    preferred_model: Optional[str] = None
    
    # System metadata
    safety_check: bool = True
    estimated_cost: float = 0.0
    timeout_seconds: int = 300
    idempotency_key: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize request for JSON storage [CMV]"""
        data = {
            "task": self.task.value,
            "prompt": self.prompt,
            "user_id": self.user_id,
            "guild_id": self.guild_id,
            "channel_id": self.channel_id,
            "width": self.width,
            "height": self.height,
            "steps": self.steps,
            "guidance_scale": self.guidance_scale,
            "negative_prompt": self.negative_prompt,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "strength": self.strength,
            "duration_seconds": self.duration_seconds,
            "fps": self.fps,
            "style": self.style,
            "mode": self.mode,
            "preferred_provider": self.preferred_provider.value if self.preferred_provider else None,
            "preferred_model": self.preferred_model,
            "safety_check": self.safety_check,
            "estimated_cost": self.estimated_cost,
            "timeout_seconds": self.timeout_seconds,
            "idempotency_key": self.idempotency_key
        }
        
        # Convert Path objects to strings for JSON serialization
        if self.input_image:
            data["input_image"] = str(self.input_image)
        if self.mask_image:
            data["mask_image"] = str(self.mask_image)
        if self.end_image:
            data["end_image"] = str(self.end_image)
            
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VisionRequest:
        """Deserialize request from JSON storage [CMV]"""
        # Convert string paths back to Path objects
        if "input_image" in data and data["input_image"]:
            data["input_image"] = Path(data["input_image"])
        if "mask_image" in data and data["mask_image"]:
            data["mask_image"] = Path(data["mask_image"])
        if "end_image" in data and data["end_image"]:
            data["end_image"] = Path(data["end_image"])
        
        # Convert provider string back to enum
        if "preferred_provider" in data and data["preferred_provider"]:
            data["preferred_provider"] = VisionProvider(data["preferred_provider"])
        
        # Convert task string back to enum    
        data["task"] = VisionTask(data["task"])
        
        return cls(**data)


@dataclass 
class VisionResponse:
    """Standardized response from vision providers"""
    success: bool
    job_id: str
    provider: VisionProvider
    model_used: str
    
    # Generated content
    artifacts: List[Path] = field(default_factory=list)
    thumbnails: List[Path] = field(default_factory=list)
    
    # Execution metadata
    processing_time_seconds: float = 0.0
    actual_cost: float = 0.0
    provider_job_id: Optional[str] = None
    
    # Quality metrics
    dimensions: Optional[tuple[int, int]] = None
    duration_seconds: Optional[float] = None
    file_size_bytes: int = 0
    
    # Error information
    error: Optional[VisionError] = None
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize response for JSON storage [CMV]"""
        return {
            "success": self.success,
            "job_id": self.job_id,
            "provider": self.provider.value,
            "model_used": self.model_used,
            "artifacts": [str(p) for p in self.artifacts],
            "thumbnails": [str(p) for p in self.thumbnails],
            "processing_time_seconds": self.processing_time_seconds,
            "actual_cost": self.actual_cost,
            "provider_job_id": self.provider_job_id,
            "dimensions": self.dimensions,
            "duration_seconds": self.duration_seconds,
            "file_size_bytes": self.file_size_bytes,
            "error": self.error.__dict__ if self.error else None,
            "warnings": self.warnings
        }


@dataclass
class VisionJob:
    """Complete job state for orchestration and persistence"""
    job_id: str
    request: VisionRequest
    state: VisionJobState
    
    # Timestamps [CMV]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Execution tracking
    provider_assigned: Optional[VisionProvider] = None
    provider_job_id: Optional[str] = None  # Job ID returned by provider (may differ from orchestrator job_id)
    model_assigned: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    # Progress and results
    progress_percentage: int = 0
    response: Optional[VisionResponse] = None
    error: Optional[VisionError] = None
    
    # Discord integration
    discord_interaction_id: Optional[str] = None
    discord_message_id: Optional[str] = None
    progress_message_id: Optional[str] = None
    
    # Audit trail
    log_entries: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_log_entry(self, level: str, message: str, **kwargs) -> None:
        """Add structured log entry with timestamp [REH]"""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "message": message,
            "state": self.state.value,
            **kwargs
        }
        self.log_entries.append(entry)
        self.last_updated = datetime.now(timezone.utc)
    
    def update_progress(self, percentage: int, message: str = "") -> None:
        """Update job progress with optional status message [PA]"""
        self.progress_percentage = max(0, min(100, percentage))
        self.last_updated = datetime.now(timezone.utc)
        if message:
            self.add_log_entry("info", f"Progress {percentage}%: {message}")
    
    def transition_to(self, new_state: VisionJobState, message: str = "") -> None:
        """Transition job state with audit logging [CA]"""
        old_state = self.state
        self.state = new_state
        self.last_updated = datetime.now(timezone.utc)
        
        # Update state-specific timestamps
        if new_state == VisionJobState.RUNNING and not self.started_at:
            self.started_at = datetime.now(timezone.utc)
        elif new_state in [VisionJobState.COMPLETED, VisionJobState.FAILED, VisionJobState.CANCELLED]:
            self.completed_at = datetime.now(timezone.utc)
        
        log_message = f"State transition: {old_state.value} → {new_state.value}"
        if message:
            log_message += f" ({message})"
        
        self.add_log_entry("info", log_message)
    
    def is_terminal_state(self) -> bool:
        """Check if job has reached a terminal state [CMV]"""
        return self.state in [
            VisionJobState.COMPLETED,
            VisionJobState.FAILED, 
            VisionJobState.CANCELLED,
            VisionJobState.EXPIRED
        ]
    
    def is_expired(self, timeout_seconds: int) -> bool:
        """Check if job has exceeded timeout [CMV]"""
        if not self.started_at:
            return False
        elapsed = (datetime.now(timezone.utc) - self.started_at).total_seconds()
        return elapsed > timeout_seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize job for JSON storage [CMV]"""
        return {
            "job_id": self.job_id,
            "request": self.request.to_dict(),
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "last_updated": self.last_updated.isoformat(),
            "provider_assigned": self.provider_assigned.value if self.provider_assigned else None,
            "model_assigned": self.model_assigned,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "progress_percentage": self.progress_percentage,
            "response": self.response.to_dict() if self.response else None,
            "error": self.error.__dict__ if self.error else None,
            "discord_interaction_id": self.discord_interaction_id,
            "discord_message_id": self.discord_message_id,
            "progress_message_id": self.progress_message_id,
            "log_entries": self.log_entries
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VisionJob:
        """Deserialize job from JSON storage [CMV]"""
        # Parse timestamps
        created_at = datetime.fromisoformat(data["created_at"])
        started_at = datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None
        completed_at = datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None
        last_updated = datetime.fromisoformat(data["last_updated"])
        
        # Parse enums
        state = VisionJobState(data["state"])
        provider_assigned = VisionProvider(data["provider_assigned"]) if data.get("provider_assigned") else None
        
        # Parse nested objects
        request = VisionRequest.from_dict(data["request"])
        response = None
        if data.get("response"):
            response_data = data["response"]
            response_data["provider"] = VisionProvider(response_data["provider"])
            response_data["artifacts"] = [Path(p) for p in response_data["artifacts"]]
            response_data["thumbnails"] = [Path(p) for p in response_data["thumbnails"]]
            if response_data.get("error"):
                error_data = response_data["error"]
                error_data["error_type"] = VisionErrorType(error_data["error_type"])
                response_data["error"] = VisionError(**error_data)
            response = VisionResponse(**response_data)
        
        # Parse error
        error = None
        if data.get("error"):
            error_data = data["error"]
            # Handle enum deserialization safely [REH]
            error_type_value = error_data["error_type"]
            if isinstance(error_type_value, VisionErrorType):
                # Already an enum
                pass
            elif isinstance(error_type_value, str):
                # Handle both "SYSTEM_ERROR" and "VisionErrorType.SYSTEM_ERROR" formats
                if "." in error_type_value:
                    error_type_value = error_type_value.split(".")[-1]  # Get just the enum name
                error_data["error_type"] = VisionErrorType[error_type_value]
            else:
                error_data["error_type"] = VisionErrorType(error_type_value)
            if "provider" in error_data and error_data["provider"]:
                error_data["provider"] = VisionProvider(error_data["provider"])
            error = VisionError(**error_data)
        
        return cls(
            job_id=data["job_id"],
            request=request,
            state=state,
            created_at=created_at,
            started_at=started_at,
            completed_at=completed_at,
            last_updated=last_updated,
            provider_assigned=provider_assigned,
            model_assigned=data.get("model_assigned"),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
            progress_percentage=data.get("progress_percentage", 0),
            response=response,
            error=error,
            discord_interaction_id=data.get("discord_interaction_id"),
            discord_message_id=data.get("discord_message_id"),
            progress_message_id=data.get("progress_message_id"),
            log_entries=data.get("log_entries", [])
        )


# Provider-specific request/response types for adapter implementations

@dataclass
class ProviderRequest:
    """Base class for provider-specific requests"""
    pass


@dataclass
class ProviderResponse:
    """Base class for provider-specific responses"""
    success: bool
    provider_job_id: Optional[str] = None
    error_message: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None


# Intent routing types

@dataclass
class IntentScore:
    """Intent classification result"""
    task: Optional[VisionTask]
    confidence: float
    extracted_parameters: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""


@dataclass  
class RoutingDecision:
    """Complete routing decision with justification"""
    route_to_vision: bool
    task: Optional[VisionTask] = None
    confidence: float = 0.0
    provider: Optional[VisionProvider] = None
    model: Optional[str] = None
    estimated_cost: float = 0.0
    reasoning: str = ""
    fallback_reason: Optional[str] = None
