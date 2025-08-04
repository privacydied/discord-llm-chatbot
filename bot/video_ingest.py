"""
URL-based video ingestion module for YouTube/TikTok audio extraction and STT processing.
Integrates with existing hear_infer() pipeline for consistent audio processing.
"""
import os
import re
import asyncio
import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Optional, NamedTuple, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
import subprocess

from .util.logging import get_logger
from .exceptions import InferenceError

logger = get_logger(__name__)

# Configuration from environment
MAX_DURATION_SECONDS = int(os.getenv("VIDEO_MAX_DURATION", "600"))  # 10 minutes default
MAX_CONCURRENT_DOWNLOADS = int(os.getenv("VIDEO_MAX_CONCURRENT", "3"))
CACHE_DIR = Path(os.getenv("VIDEO_CACHE_DIR", "cache/video_audio"))
DEFAULT_SPEEDUP = float(os.getenv("VIDEO_SPEEDUP", "1.5"))
CACHE_EXPIRY_DAYS = int(os.getenv("VIDEO_CACHE_EXPIRY_DAYS", "7"))

# Supported URL patterns - must match MEDIA_CAPABLE_DOMAINS from media_capability.py
SUPPORTED_PATTERNS = [
    # YouTube patterns
    r'https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+',
    r'https?://(?:www\.)?youtube\.com/shorts/[\w-]+',
    r'https?://youtu\.be/[\w-]+',
    
    # TikTok patterns
    r'https?://(?:www\.)?tiktok\.com/@[\w.-]+/video/\d+',
    r'https?://(?:vm\.)?tiktok\.com/[\w-]+',
    
    # Twitter/X patterns
    r'https?://(?:www\.)?twitter\.com/[\w]+/status/\d+',
    r'https?://(?:www\.)?x\.com/[\w]+/status/\d+',
    
    # Other video platforms
    r'https?://(?:www\.)?vimeo\.com/\d+',
    r'https?://(?:www\.)?dailymotion\.com/video/[\w-]+',
    r'https?://(?:www\.)?twitch\.tv/videos/\d+',
    r'https?://(?:www\.)?twitch\.tv/[\w]+/clip/[\w-]+',
    r'https?://(?:www\.)?bilibili\.com/video/[\w-]+',
    r'https?://(?:www\.)?rumble\.com/[\w-]+',
    r'https?://(?:www\.)?odysee\.com/@[\w-]+:[\w-]+/[\w-]+:[\w-]+',
    r'https?://(?:www\.)?lbry\.tv/@[\w-]+:[\w-]+/[\w-]+:[\w-]+',
    r'https?://(?:www\.)?veoh\.com/watch/[\w-]+',
    r'https?://(?:www\.)?metacafe\.com/watch/\d+/[\w-]+',
    
    # Audio platforms
    r'https?://(?:www\.)?soundcloud\.com/[\w-]+/[\w-]+',
    r'https?://[\w-]+\.bandcamp\.com/track/[\w-]+',
    r'https?://(?:www\.)?mixcloud\.com/[\w-]+/[\w-]+',
    r'https?://(?:www\.)?audiomack\.com/song/[\w-]+/[\w-]+',
    
    # Educational/conference
    r'https?://(?:www\.)?ted\.com/talks/[\w-]+',
    r'https?://(?:www\.)?archive\.org/details/[\w-]+',
    
    # Social media (flexible patterns for various post formats)
    r'https?://(?:www\.)?reddit\.com/r/[\w-]+/comments/[\w-]+',
    r'https?://v\.redd\.it/[\w-]+',  # Reddit video direct links
    r'https?://(?:www\.)?reddit\.com/video/[\w-]+',  # Reddit video pages
    r'https?://(?:www\.)?facebook\.com/[\w.-]+/videos/\d+',
    r'https?://(?:www\.)?fb\.com/[\w.-]+/videos/\d+',
    r'https?://(?:www\.)?instagram\.com/p/[\w-]+',
    r'https?://(?:www\.)?instagram\.com/reel/[\w-]+',
    r'https?://(?:www\.)?linkedin\.com/posts/[\w-]+'
]

# Global semaphore for download concurrency
_download_semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)


@dataclass
class VideoMetadata:
    """Metadata extracted from video source."""
    url: str
    title: str
    duration_seconds: float
    uploader: str
    upload_date: str
    source_type: str  # 'youtube' or 'tiktok'


@dataclass 
class ProcessedAudio:
    """Result of video audio processing."""
    audio_path: Path
    metadata: VideoMetadata
    processed_duration_seconds: float
    speedup_factor: float
    cache_hit: bool
    timestamp: datetime


class VideoIngestError(InferenceError):
    """Specific error for video ingestion failures."""
    pass


class VideoIngestionManager:
    """Manages video URL ingestion, caching, and audio processing."""
    
    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._setup_cache_index()
        logger.info(f"🎥 VideoIngestionManager initialized with cache: {self.cache_dir}")
    
    def _setup_cache_index(self):
        """Initialize cache index file."""
        self.cache_index_path = self.cache_dir / "index.json"
        if not self.cache_index_path.exists():
            with open(self.cache_index_path, 'w') as f:
                json.dump({}, f)
    
    def _get_cache_key(self, url: str) -> str:
        """Generate deterministic cache key for URL."""
        return hashlib.sha256(url.encode()).hexdigest()[:16]
    
    def _is_supported_url(self, url: str) -> bool:
        """Check if URL matches supported patterns."""
        return any(re.match(pattern, url) for pattern in SUPPORTED_PATTERNS)
    
    def _get_source_type(self, url: str) -> str:
        """Determine source type from URL."""
        if 'youtube.com' in url or 'youtu.be' in url:
            return 'youtube'
        elif 'tiktok.com' in url:
            return 'tiktok'
        else:
            return 'unknown'
    
    async def _download_with_ytdlp(self, url: str, output_path: Path) -> Tuple[VideoMetadata, Path]:
        """Download video audio using yt-dlp."""
        logger.info(f"📥 Downloading audio from: {url}")
        
        # First get metadata with JSON output for reliable parsing
        metadata_cmd = [
            'yt-dlp',
            '--dump-json',
            '--no-playlist',
            '--quiet',
            url
        ]
        
        try:
            # Get metadata first
            proc = await asyncio.create_subprocess_exec(
                *metadata_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            
            if proc.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown yt-dlp error"
                raise VideoIngestError(f"yt-dlp metadata extraction failed: {error_msg}")
            
            # Parse JSON metadata
            metadata_json = json.loads(stdout.decode())
            
            # Extract metadata with safe defaults
            title = metadata_json.get('title', 'Unknown Title')
            duration = float(metadata_json.get('duration', 0.0) or 0.0)
            uploader = metadata_json.get('uploader', 'Unknown')
            upload_date = metadata_json.get('upload_date', '')
            
            # Now download the audio
            download_cmd = [
                'yt-dlp',
                '--extract-audio',
                '--audio-format', 'wav',
                '--audio-quality', '0',  # Best quality
                '--no-playlist',
                '--output', str(output_path / '%(title)s.%(ext)s'),
                '--print', 'after_move:filepath',
                '--quiet',
                url
            ]
            
            proc = await asyncio.create_subprocess_exec(
                *download_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            
            if proc.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown yt-dlp error"
                raise VideoIngestError(f"yt-dlp download failed: {error_msg}")
            
            # Get the filepath from output
            filepath = stdout.decode().strip()
            if not filepath:
                raise VideoIngestError("No filepath returned from yt-dlp")
            
            return VideoMetadata(
                url=url,
                title=title,
                duration_seconds=duration,
                uploader=uploader,
                upload_date=upload_date,
                source_type=self._get_source_type(url)
            ), Path(filepath)
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse yt-dlp JSON metadata: {e}")
            raise VideoIngestError(f"Failed to parse video metadata: {str(e)}")
        except Exception as e:
            logger.error(f"❌ yt-dlp download failed: {e}")
            raise VideoIngestError(f"Failed to download video: {str(e)}")
    
    async def _process_audio(self, raw_audio_path: Path, speedup: float = DEFAULT_SPEEDUP) -> Path:
        """Process raw audio with same normalization as hear_infer()."""
        logger.info(f"🔄 Processing audio with {speedup}x speedup")
        
        # Create processed audio path
        processed_path = raw_audio_path.with_suffix('.processed.wav')
        
        # FFmpeg command matching hear_infer() logic
        cmd = [
            'ffmpeg', '-y', '-i', str(raw_audio_path),
            '-ar', '16000',  # 16kHz sample rate
            '-ac', '1',      # Mono
            '-acodec', 'pcm_s16le',  # 16-bit PCM
            '-af', f'atempo={speedup},aresample=async=1:first_pts=0',  # Speed + resample
            str(processed_path)
        ]
        
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            
            if proc.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown ffmpeg error"
                raise VideoIngestError(f"Audio processing failed: {error_msg}")
            
            logger.info(f"✅ Audio processed: {processed_path}")
            return processed_path
            
        except Exception as e:
            logger.error(f"❌ Audio processing failed: {e}")
            raise VideoIngestError(f"Failed to process audio: {str(e)}")
    
    def _update_cache_index(self, cache_key: str, metadata: VideoMetadata, processed_path: Path):
        """Update cache index with new entry."""
        try:
            with open(self.cache_index_path, 'r') as f:
                index = json.load(f)
            
            index[cache_key] = {
                'url': metadata.url,
                'title': metadata.title,
                'duration_seconds': metadata.duration_seconds,
                'uploader': metadata.uploader,
                'upload_date': metadata.upload_date,
                'source_type': metadata.source_type,
                'processed_path': str(processed_path),
                'cached_at': datetime.now(timezone.utc).isoformat(),
                'speedup_factor': DEFAULT_SPEEDUP
            }
            
            with open(self.cache_index_path, 'w') as f:
                json.dump(index, f, indent=2)
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to update cache index: {e}")
    
    def _get_cached_entry(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached entry if it exists and is valid."""
        try:
            with open(self.cache_index_path, 'r') as f:
                index = json.load(f)
            
            if cache_key not in index:
                return None
            
            entry = index[cache_key]
            processed_path = Path(entry['processed_path'])
            
            # Check if cached file exists
            if not processed_path.exists():
                logger.warning(f"⚠️ Cached file missing: {processed_path}")
                return None
            
            # Check cache expiry
            cached_at = datetime.fromisoformat(entry['cached_at'])
            age_days = (datetime.now(timezone.utc) - cached_at).days
            
            if age_days > CACHE_EXPIRY_DAYS:
                logger.info(f"🗑️ Cache entry expired ({age_days} days old)")
                return None
            
            return entry
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to read cache index: {e}")
            return None
    
    async def fetch_and_prepare_url_audio(
        self, 
        url: str, 
        speedup: float = DEFAULT_SPEEDUP,
        force_refresh: bool = False
    ) -> ProcessedAudio:
        """
        Main entry point: fetch video URL and prepare audio for STT pipeline.
        
        Args:
            url: YouTube or TikTok URL
            speedup: Audio speedup factor (default 1.5x)
            force_refresh: Force re-download even if cached
            
        Returns:
            ProcessedAudio object ready for hear_infer()
        """
        if not self._is_supported_url(url):
            raise VideoIngestError(f"Unsupported URL format: {url}")
        
        cache_key = self._get_cache_key(url)
        
        # Check cache first (unless force refresh)
        if not force_refresh:
            cached_entry = self._get_cached_entry(cache_key)
            if cached_entry:
                logger.info(f"💾 Cache hit for: {url}")
                
                metadata = VideoMetadata(
                    url=cached_entry['url'],
                    title=cached_entry['title'],
                    duration_seconds=cached_entry['duration_seconds'],
                    uploader=cached_entry['uploader'],
                    upload_date=cached_entry['upload_date'],
                    source_type=cached_entry['source_type']
                )
                
                return ProcessedAudio(
                    audio_path=Path(cached_entry['processed_path']),
                    metadata=metadata,
                    processed_duration_seconds=cached_entry['duration_seconds'] / speedup,
                    speedup_factor=speedup,
                    cache_hit=True,
                    timestamp=datetime.now(timezone.utc)
                )
        
        # Download and process new content
        async with _download_semaphore:
            logger.info(f"🔄 Processing new URL: {url}")
            
            # Create temporary directory for this download
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                try:
                    # Download with yt-dlp
                    metadata, raw_audio_path = await self._download_with_ytdlp(url, temp_path)
                    
                    # Validate duration
                    if metadata.duration_seconds > MAX_DURATION_SECONDS:
                        raise VideoIngestError(
                            f"Video too long: {metadata.duration_seconds:.1f}s "
                            f"(max: {MAX_DURATION_SECONDS}s)"
                        )
                    
                    # Process audio (normalize + speedup)
                    processed_path = await self._process_audio(raw_audio_path, speedup)
                    
                    # Move to cache directory (handle cross-filesystem moves)
                    cache_audio_path = self.cache_dir / f"{cache_key}.wav"
                    shutil.move(str(processed_path), str(cache_audio_path))
                    
                    # Update cache index
                    self._update_cache_index(cache_key, metadata, cache_audio_path)
                    
                    logger.info(f"✅ Successfully processed: {metadata.title}")
                    
                    return ProcessedAudio(
                        audio_path=cache_audio_path,
                        metadata=metadata,
                        processed_duration_seconds=metadata.duration_seconds / speedup,
                        speedup_factor=speedup,
                        cache_hit=False,
                        timestamp=datetime.now(timezone.utc)
                    )
                    
                except Exception as e:
                    logger.error(f"❌ Failed to process URL {url}: {e}")
                    raise


# Global instance
video_manager = VideoIngestionManager()


async def fetch_and_prepare_url_audio(
    url: str, 
    speedup: float = DEFAULT_SPEEDUP,
    force_refresh: bool = False
) -> ProcessedAudio:
    """
    Convenience function to fetch and prepare URL audio.
    
    Args:
        url: YouTube or TikTok URL
        speedup: Audio speedup factor (default 1.5x)  
        force_refresh: Force re-download even if cached
        
    Returns:
        ProcessedAudio object ready for STT pipeline
    """
    return await video_manager.fetch_and_prepare_url_audio(url, speedup, force_refresh)
