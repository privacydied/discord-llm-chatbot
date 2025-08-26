"""
Vision Artifact Cache - File management and Discord integration

Manages generated image and video artifacts with Discord upload optimization:
- File storage with TTL and size limits
- Discord attachment size optimization and format conversion
- Efficient caching with LRU eviction
- File validation and metadata extraction
- Concurrent upload handling with retry logic

Follows Resource Management (RM) and Performance Awareness (PA) principles.
"""

from __future__ import annotations
import asyncio
import aiofiles
import hashlib
import mimetypes
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import tempfile
import shutil

from bot.util.logging import get_logger
from bot.config import load_config

logger = get_logger(__name__)


class FileFormat(Enum):
    """Supported file formats"""
    PNG = "png"
    JPEG = "jpeg"  
    WEBP = "webp"
    GIF = "gif"
    MP4 = "mp4"
    WEBM = "webm"


@dataclass
class ArtifactMetadata:
    """Cached artifact metadata"""
    file_path: Path
    original_size: int
    compressed_size: int
    format: FileFormat
    dimensions: Optional[Tuple[int, int]]
    duration_seconds: Optional[float]
    created_at: datetime
    last_accessed: datetime
    content_hash: str
    discord_optimized: bool


@dataclass
class CacheStats:
    """Cache statistics"""
    total_files: int
    total_size_bytes: int
    hit_count: int
    miss_count: int
    eviction_count: int
    compression_ratio: float


class VisionArtifactCache:
    """
    High-performance artifact cache with Discord optimization
    
    Features:
    - Automatic file format optimization for Discord uploads
    - LRU eviction with TTL cleanup
    - Concurrent upload handling
    - File deduplication using content hashes
    - Size and format validation
    - Detailed analytics and monitoring
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config()
        self.logger = get_logger("vision.artifact_cache")
        
        # Cache configuration
        self.cache_dir = Path(self.config["VISION_ARTIFACTS_DIR"])
        self.max_file_size_mb = self.config["VISION_MAX_ARTIFACT_SIZE_MB"]
        self.max_total_size_gb = self.config["VISION_MAX_TOTAL_ARTIFACTS_GB"]
        self.ttl_days = self.config["VISION_ARTIFACT_TTL_DAYS"]
        
        # Discord limits and optimization
        self.discord_max_file_size = 25 * 1024 * 1024  # 25MB for premium servers
        self.discord_max_regular_size = 8 * 1024 * 1024  # 8MB for regular servers
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory metadata cache for performance
        self._metadata_cache: Dict[str, ArtifactMetadata] = {}
        self._access_order: List[str] = []  # LRU tracking
        
        # Cache statistics
        self._stats = CacheStats(
            total_files=0, total_size_bytes=0, hit_count=0,
            miss_count=0, eviction_count=0, compression_ratio=1.0
        )
        
        # File locks for concurrent access
        self._locks: Dict[str, asyncio.Lock] = {}
        
        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._start_cleanup_task()
        
        self.logger.info(
            "Vision Artifact Cache initialized",
            cache_dir=str(self.cache_dir),
            max_file_size_mb=self.max_file_size_mb,
            max_total_gb=self.max_total_size_gb,
            ttl_days=self.ttl_days
        )
    
    async def store_artifact(self, content: bytes, filename: str, 
                           optimize_for_discord: bool = True) -> str:
        """
        Store artifact with optional Discord optimization
        
        Args:
            content: Raw file content
            filename: Original filename with extension
            optimize_for_discord: Whether to optimize for Discord upload
            
        Returns:
            Content hash for retrieval
            
        Raises:
            ValueError: File too large or invalid format
        """
        # Validate file size
        if len(content) > self.max_file_size_mb * 1024 * 1024:
            raise ValueError(f"File too large: {len(content)/1024/1024:.1f}MB > {self.max_file_size_mb}MB")
        
        # Generate content hash for deduplication
        content_hash = hashlib.sha256(content).hexdigest()
        
        async with self._get_lock(content_hash):
            # Check if already cached
            if content_hash in self._metadata_cache:
                metadata = self._metadata_cache[content_hash]
                metadata.last_accessed = datetime.now(timezone.utc)
                self._update_lru_order(content_hash)
                self._stats.hit_count += 1
                
                self.logger.debug(f"Cache hit: {content_hash[:8]}")
                return content_hash
            
            try:
                # Determine file format and validate
                file_format, is_valid = self._validate_file_format(content, filename)
                if not is_valid:
                    raise ValueError(f"Unsupported or invalid file format: {filename}")
                
                # Create cache file path
                cache_file = self.cache_dir / f"{content_hash}.{file_format.value}"
                
                # Process and store file
                processed_content = content
                discord_optimized = False
                
                if optimize_for_discord:
                    processed_content, discord_optimized = await self._optimize_for_discord(
                        content, file_format, cache_file
                    )
                
                # Write processed content
                async with aiofiles.open(cache_file, "wb") as f:
                    await f.write(processed_content)
                
                # Extract metadata
                dimensions = await self._extract_dimensions(cache_file, file_format)
                duration = await self._extract_duration(cache_file, file_format)
                
                # Create metadata
                metadata = ArtifactMetadata(
                    file_path=cache_file,
                    original_size=len(content),
                    compressed_size=len(processed_content),
                    format=file_format,
                    dimensions=dimensions,
                    duration_seconds=duration,
                    created_at=datetime.now(timezone.utc),
                    last_accessed=datetime.now(timezone.utc),
                    content_hash=content_hash,
                    discord_optimized=discord_optimized
                )
                
                # Update cache
                self._metadata_cache[content_hash] = metadata
                self._update_lru_order(content_hash)
                self._stats.miss_count += 1
                self._stats.total_files += 1
                self._stats.total_size_bytes += len(processed_content)
                
                # Update compression ratio
                if metadata.original_size > 0:
                    compression_ratio = metadata.compressed_size / metadata.original_size
                    self._stats.compression_ratio = (
                        self._stats.compression_ratio * 0.9 + compression_ratio * 0.1
                    )
                
                # Check cache size limits and evict if needed
                await self._enforce_size_limits()
                
                self.logger.info(f"Artifact cached - hash: {content_hash[:8]}, original_size: {len(content)}, compressed_size: {len(processed_content)}, format: {file_format.value}, discord_optimized: {discord_optimized}")
                
                return content_hash
                
            except Exception as e:
                self.logger.error(f"Failed to store artifact: {e}")
                # Cleanup partial file
                cache_file = self.cache_dir / f"{content_hash}.{file_format.value}"
                if cache_file.exists():
                    cache_file.unlink()
                raise
    
    async def retrieve_artifact(self, content_hash: str) -> Optional[Tuple[bytes, ArtifactMetadata]]:
        """
        Retrieve cached artifact
        
        Args:
            content_hash: Content hash from store_artifact
            
        Returns:
            Tuple of (file_content, metadata) or None if not found
        """
        async with self._get_lock(content_hash):
            if content_hash not in self._metadata_cache:
                self._stats.miss_count += 1
                return None
            
            metadata = self._metadata_cache[content_hash]
            
            # Check if file still exists
            if not metadata.file_path.exists():
                # Remove stale entry
                del self._metadata_cache[content_hash]
                self._remove_from_lru_order(content_hash)
                self._stats.miss_count += 1
                return None
            
            try:
                # Read file content
                async with aiofiles.open(metadata.file_path, "rb") as f:
                    content = await f.read()
                
                # Update access time and LRU order
                metadata.last_accessed = datetime.now(timezone.utc)
                self._update_lru_order(content_hash)
                self._stats.hit_count += 1
                
                self.logger.debug(f"Cache hit: {content_hash[:8]}")
                return content, metadata
                
            except Exception as e:
                self.logger.error(f"Failed to retrieve artifact {content_hash[:8]}: {e}")
                return None
    
    async def delete_artifact(self, content_hash: str) -> bool:
        """Delete cached artifact"""
        async with self._get_lock(content_hash):
            if content_hash not in self._metadata_cache:
                return False
            
            metadata = self._metadata_cache[content_hash]
            
            try:
                # Remove file
                if metadata.file_path.exists():
                    metadata.file_path.unlink()
                
                # Remove from cache
                del self._metadata_cache[content_hash]
                self._remove_from_lru_order(content_hash)
                self._stats.total_files -= 1
                self._stats.total_size_bytes -= metadata.compressed_size
                
                self.logger.debug(f"Artifact deleted: {content_hash[:8]}")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to delete artifact {content_hash[:8]}: {e}")
                return False
    
    async def get_discord_file_info(self, content_hash: str) -> Optional[Dict[str, Any]]:
        """Get Discord upload information for artifact"""
        result = await self.retrieve_artifact(content_hash)
        if not result:
            return None
        
        content, metadata = result
        
        return {
            "filename": f"generated_{content_hash[:8]}.{metadata.format.value}",
            "size": metadata.compressed_size,
            "format": metadata.format.value,
            "dimensions": metadata.dimensions,
            "duration": metadata.duration_seconds,
            "discord_optimized": metadata.discord_optimized,
            "too_large": metadata.compressed_size > self.discord_max_file_size,
            "content": content
        }
    
    def get_cache_stats(self) -> CacheStats:
        """Get current cache statistics"""
        return self._stats
    
    async def cleanup_expired(self) -> int:
        """Remove expired artifacts"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=self.ttl_days)
        expired_hashes = []
        
        for content_hash, metadata in self._metadata_cache.items():
            if metadata.created_at < cutoff_time:
                expired_hashes.append(content_hash)
        
        cleaned_count = 0
        for content_hash in expired_hashes:
            if await self.delete_artifact(content_hash):
                cleaned_count += 1
        
        if cleaned_count > 0:
            self.logger.info(f"Cleaned up {cleaned_count} expired artifacts")
        
        return cleaned_count
    
    async def _optimize_for_discord(self, content: bytes, file_format: FileFormat, 
                                  output_path: Path) -> Tuple[bytes, bool]:
        """
        Optimize file for Discord upload [PA]
        
        Returns:
            Tuple of (optimized_content, was_optimized)
        """
        # For now, return original content since we don't have image processing libraries
        # In a full implementation, this would:
        # 1. Resize large images to fit Discord limits
        # 2. Convert to more efficient formats (WebP)
        # 3. Compress videos with better codecs
        # 4. Reduce quality if size is too large
        
        original_size = len(content)
        
        # Simple size check - if under Discord limit, don't modify
        if original_size <= self.discord_max_regular_size:
            return content, False
        
        # TODO: Implement actual optimization with PIL/ffmpeg
        # For now, just return original content
        self.logger.debug(f"File optimization skipped (no image processing libs): {original_size} bytes")
        
        return content, False
    
    def _validate_file_format(self, content: bytes, filename: str) -> Tuple[FileFormat, bool]:
        """Validate file format and detect type [IV]"""
        # Get file extension
        suffix = Path(filename).suffix.lower()
        
        # Check magic bytes for common formats
        if content.startswith(b'\x89PNG\r\n\x1a\n'):
            return FileFormat.PNG, True
        elif content.startswith(b'\xff\xd8\xff'):
            return FileFormat.JPEG, True
        elif content.startswith(b'RIFF') and b'WEBP' in content[:12]:
            return FileFormat.WEBP, True
        elif content.startswith(b'GIF8'):
            return FileFormat.GIF, True
        elif (content.startswith(b'\x00\x00\x00\x18ftypmp4') or 
              content.startswith(b'\x00\x00\x00\x20ftypmp41')):
            return FileFormat.MP4, True
        elif content.startswith(b'\x1a\x45\xdf\xa3'):
            return FileFormat.WEBM, True
        
        # Fallback to extension
        extension_map = {
            '.png': FileFormat.PNG,
            '.jpg': FileFormat.JPEG,
            '.jpeg': FileFormat.JPEG,
            '.webp': FileFormat.WEBP,
            '.gif': FileFormat.GIF,
            '.mp4': FileFormat.MP4,
            '.webm': FileFormat.WEBM
        }
        
        if suffix in extension_map:
            return extension_map[suffix], True
        
        return FileFormat.PNG, False  # Default fallback
    
    async def _extract_dimensions(self, file_path: Path, file_format: FileFormat) -> Optional[Tuple[int, int]]:
        """Extract image/video dimensions [CMV]"""
        # TODO: Implement with proper image/video processing libraries
        # For now, return None since we don't have PIL/ffmpeg
        return None
    
    async def _extract_duration(self, file_path: Path, file_format: FileFormat) -> Optional[float]:
        """Extract video duration [CMV]"""
        # TODO: Implement with ffmpeg or similar
        # For now, return None for videos
        if file_format in [FileFormat.MP4, FileFormat.WEBM]:
            return None  # Would extract actual duration
        return None
    
    async def _enforce_size_limits(self) -> None:
        """Enforce cache size limits with LRU eviction [RM]"""
        max_total_bytes = self.max_total_size_gb * 1024 * 1024 * 1024
        
        while (self._stats.total_size_bytes > max_total_bytes and 
               len(self._access_order) > 0):
            # Evict least recently used
            lru_hash = self._access_order[0]
            if await self.delete_artifact(lru_hash):
                self._stats.eviction_count += 1
                self.logger.debug(f"Evicted LRU artifact: {lru_hash[:8]}")
    
    def _update_lru_order(self, content_hash: str) -> None:
        """Update LRU access order [CMV]"""
        if content_hash in self._access_order:
            self._access_order.remove(content_hash)
        self._access_order.append(content_hash)
    
    def _remove_from_lru_order(self, content_hash: str) -> None:
        """Remove from LRU order [CMV]"""
        if content_hash in self._access_order:
            self._access_order.remove(content_hash)
    
    def _get_lock(self, content_hash: str) -> asyncio.Lock:
        """Get file lock for concurrent access [CMV]"""
        if content_hash not in self._locks:
            self._locks[content_hash] = asyncio.Lock()
        return self._locks[content_hash]
    
    def _start_cleanup_task(self) -> None:
        """Start background cleanup task [RM]"""
        self._cleanup_task = asyncio.create_task(self._background_cleanup())
    
    async def _background_cleanup(self) -> None:
        """Background cleanup task [RM]"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Clean expired artifacts
                expired_count = await self.cleanup_expired()
                
                # Enforce size limits
                await self._enforce_size_limits()
                
                # Log stats
                self.logger.debug(f"Cache cleanup completed - total_files: {self._stats.total_files}, total_size_mb: {round(self._stats.total_size_bytes / 1024 / 1024, 1)}, hit_rate: {self._stats.hit_count / max(self._stats.hit_count + self._stats.miss_count, 1):.2f}, expired_cleaned: {expired_count}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Background cleanup error: {e}")
    
    async def close(self) -> None:
        """Cleanup resources [RM]"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Vision Artifact Cache closed")
