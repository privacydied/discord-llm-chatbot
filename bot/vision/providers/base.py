"""
Base Vision Provider Interface

Abstract base class defining the contract for vision generation providers.
Ensures consistent behavior across Together.ai, Novita.ai, and future providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path

from bot.util.logging import get_logger
from ..types import VisionRequest, VisionResponse, VisionProvider, VisionError, VisionErrorType


class BaseVisionProvider(ABC):
    """
    Abstract base class for vision generation providers
    
    Standardizes provider interfaces and ensures consistent error handling,
    logging, and response formatting across all adapters [CA].
    """
    
    def __init__(self, config: Dict[str, Any], policy: Dict[str, Any]):
        self.config = config
        self.policy = policy
        self.logger = get_logger(__name__)
        
        # Validate required configuration
        self._validate_config()
        
        # Initialize provider-specific settings
        self._initialize()
    
    @abstractmethod
    def get_provider_name(self) -> VisionProvider:
        """Return the provider enum value"""
        pass
    
    @abstractmethod
    def _validate_config(self) -> None:
        """Validate provider-specific configuration [IV]"""
        pass
    
    @abstractmethod  
    def _initialize(self) -> None:
        """Initialize provider-specific resources"""
        pass
    
    @abstractmethod
    async def generate(self, request: VisionRequest, model: str) -> VisionResponse:
        """
        Generate vision content using provider's API
        
        Args:
            request: Normalized vision generation request
            model: Provider-specific model identifier
            
        Returns:
            VisionResponse with generated artifacts or error details
            
        Raises:
            VisionError: On validation failure or provider error
        """
        pass
    
    @abstractmethod
    async def get_job_status(self, provider_job_id: str) -> Dict[str, Any]:
        """
        Poll provider for job status (for async providers)
        
        Args:
            provider_job_id: Provider's job identifier
            
        Returns:
            Dictionary with status, progress, and results
        """
        pass
    
    @abstractmethod
    async def cancel_job(self, provider_job_id: str) -> bool:
        """
        Cancel running job if supported by provider
        
        Args:
            provider_job_id: Provider's job identifier
            
        Returns:
            True if cancellation was successful or job already complete
        """
        pass
    
    def _get_provider_config(self) -> Dict[str, Any]:
        """Get provider-specific configuration from policy"""
        provider_name = self.get_provider_name().value
        return self.policy["providers"].get(provider_name, {})
    
    def _get_model_config(self, task_type: str, model: str) -> Optional[Dict[str, Any]]:
        """Get model-specific configuration from policy [CMV]"""
        provider_config = self._get_provider_config()
        capabilities = provider_config.get("capabilities", {})
        task_config = capabilities.get(task_type, {})
        
        for model_config in task_config.get("models", []):
            if model_config["name"] == model:
                return model_config
        
        return None
    
    def _log_request_start(self, request: VisionRequest, model: str) -> None:
        """Log request initiation with structured data [REH]"""
        self.logger.info(
            "Starting generation request",
            extra={
                "event": "vision.provider.request.start",
                "detail": {
                    "provider": self.get_provider_name().value,
                    "task": request.task.value,
                    "model": model,
                    "user_id": request.user_id,
                    "guild_id": request.guild_id,
                    "dimensions": f"{request.width}x{request.height}" if request.task.name.endswith("IMAGE") else None,
                    "duration": f"{request.duration_seconds}s" if "VIDEO" in request.task.name else None,
                    "batch_size": request.batch_size,
                    "estimated_cost": request.estimated_cost,
                },
            },
        )
    
    def _log_request_complete(
        self,
        request: VisionRequest,
        response: VisionResponse,
        processing_time: float,
    ) -> None:
        """Log successful request completion [REH]"""
        self.logger.info(
            "Generation request completed successfully",
            extra={
                "event": "vision.provider.request.complete",
                "detail": {
                    "provider": self.get_provider_name().value,
                    "task": request.task.value,
                    "model": response.model_used,
                    "processing_time_seconds": processing_time,
                    "actual_cost": response.actual_cost,
                    "artifacts_count": len(response.artifacts),
                    "file_size_mb": round(response.file_size_bytes / (1024 * 1024), 2) if response.file_size_bytes else 0,
                },
            },
        )
    
    def _log_request_error(
        self,
        request: VisionRequest,
        error: VisionError,
        processing_time: float,
    ) -> None:
        """Log request failure with error details [REH]"""
        self.logger.error(
            "Generation request failed",
            extra={
                "event": "vision.provider.request.error",
                "detail": {
                    "provider": self.get_provider_name().value,
                    "task": request.task.value,
                    "error_type": error.error_type.value,
                    "error_message": error.message,
                    "user_message": error.user_message,
                    "processing_time_seconds": processing_time,
                    "retry_after": error.retry_after_seconds,
                },
            },
        )
    
    def _validate_file_size(self, file_path: Path, max_size_mb: int) -> None:
        """Validate uploaded file size against limits [IV]"""
        if not file_path.exists():
            raise VisionError(
                error_type=VisionErrorType.VALIDATION_ERROR,
                message=f"Input file not found: {file_path}",
                user_message="Input file could not be found. Please try uploading again."
            )
        
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > max_size_mb:
            raise VisionError(
                error_type=VisionErrorType.VALIDATION_ERROR,
                message=f"File size {file_size_mb:.1f}MB exceeds limit {max_size_mb}MB",
                user_message=f"File too large. Maximum size: {max_size_mb}MB."
            )
    
    def _validate_image_format(self, file_path: Path) -> None:
        """Validate image file format [IV]"""
        allowed_formats = self.policy["artifact_management"]["supported_formats"]["image"]
        file_extension = file_path.suffix.lower().lstrip('.')
        
        if file_extension not in allowed_formats:
            raise VisionError(
                error_type=VisionErrorType.VALIDATION_ERROR,
                message=f"Unsupported image format: {file_extension}",
                user_message=f"Unsupported image format. Allowed: {', '.join(allowed_formats)}"
            )
    
    def _create_error_response(
        self, 
        job_id: str, 
        error: VisionError, 
        processing_time: float = 0.0
    ) -> VisionResponse:
        """Create standardized error response [CA]"""
        return VisionResponse(
            success=False,
            job_id=job_id,
            provider=self.get_provider_name(),
            model_used="",
            processing_time_seconds=processing_time,
            error=error
        )
    
    def _extract_dimensions_from_path(self, file_path: Path) -> Optional[tuple[int, int]]:
        """Extract image/video dimensions using PIL/ffprobe if available"""
        try:
            if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                from PIL import Image
                with Image.open(file_path) as img:
                    return img.size  # (width, height)
        except ImportError:
            self.logger.debug("PIL not available for dimension extraction")
        except Exception as e:
            self.logger.debug(f"Could not extract dimensions: {e}")
        
        return None
    
    def _calculate_file_size(self, file_paths: list[Path]) -> int:
        """Calculate total file size in bytes [CMV]"""
        total_size = 0
        for path in file_paths:
            if path.exists():
                total_size += path.stat().st_size
        return total_size
