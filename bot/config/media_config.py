"""
Configuration settings for smart media ingestion system.
All settings can be overridden via environment variables.
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Set


@dataclass
class MediaIngestionConfig:
    """Configuration for media ingestion system."""
    
    # Capability detection settings
    probe_cache_ttl_seconds: int = int(os.getenv("MEDIA_PROBE_CACHE_TTL", "300"))  # 5 minutes
    probe_timeout_seconds: int = int(os.getenv("MEDIA_PROBE_TIMEOUT", "10"))  # 10 seconds
    probe_cache_dir: Path = Path(os.getenv("MEDIA_PROBE_CACHE_DIR", "cache/media_probes"))
    
    # Download and processing settings
    max_concurrent_downloads: int = int(os.getenv("MEDIA_MAX_CONCURRENT", "2"))
    download_timeout_seconds: int = int(os.getenv("MEDIA_DOWNLOAD_TIMEOUT", "60"))
    speedup_factor: float = float(os.getenv("MEDIA_SPEEDUP_FACTOR", "1.5"))
    
    # Retry and backoff settings
    retry_max_attempts: int = int(os.getenv("MEDIA_RETRY_MAX_ATTEMPTS", "3"))
    retry_base_delay_seconds: float = float(os.getenv("MEDIA_RETRY_BASE_DELAY", "2.0"))
    
    # Cache settings
    audio_cache_dir: Path = Path(os.getenv("VIDEO_CACHE_DIR", "cache/video_audio"))
    cache_expiry_days: int = int(os.getenv("VIDEO_CACHE_EXPIRY_DAYS", "7"))
    
    # Domain whitelist for media ingestion
    whitelisted_domains: Set[str] = {
        "youtube.com",
        "youtu.be",
        "tiktok.com", 
        "vm.tiktok.com",
        "twitter.com",
        "x.com"
    }
    
    # Content safety settings
    max_title_length: int = int(os.getenv("MEDIA_MAX_TITLE_LENGTH", "200"))
    max_uploader_length: int = int(os.getenv("MEDIA_MAX_UPLOADER_LENGTH", "100"))
    max_url_length: int = int(os.getenv("MEDIA_MAX_URL_LENGTH", "500"))
    
    # Feature flags
    enable_media_ingestion: bool = os.getenv("ENABLE_MEDIA_INGESTION", "true").lower() == "true"
    enable_twitter_video_detection: bool = os.getenv("ENABLE_TWITTER_VIDEO_DETECTION", "true").lower() == "true"
    enable_contextual_brain: bool = os.getenv("USE_ENHANCED_CONTEXT", "true").lower() == "true"
    
    def __post_init__(self):
        """Ensure cache directories exist."""
        self.probe_cache_dir.mkdir(parents=True, exist_ok=True)
        self.audio_cache_dir.mkdir(parents=True, exist_ok=True)
    
    def is_domain_whitelisted(self, domain: str) -> bool:
        """Check if domain is whitelisted for media ingestion."""
        # Remove www. prefix if present
        clean_domain = domain.lower()
        if clean_domain.startswith('www.'):
            clean_domain = clean_domain[4:]
        
        return clean_domain in self.whitelisted_domains
    
    def add_whitelisted_domain(self, domain: str):
        """Add a domain to the whitelist (for runtime configuration)."""
        clean_domain = domain.lower()
        if clean_domain.startswith('www.'):
            clean_domain = clean_domain[4:]
        
        self.whitelisted_domains.add(clean_domain)
    
    def remove_whitelisted_domain(self, domain: str):
        """Remove a domain from the whitelist (for runtime configuration)."""
        clean_domain = domain.lower()
        if clean_domain.startswith('www.'):
            clean_domain = clean_domain[4:]
        
        self.whitelisted_domains.discard(clean_domain)
    
    def get_config_summary(self) -> dict:
        """Get configuration summary for logging/debugging."""
        return {
            "probe_cache_ttl_seconds": self.probe_cache_ttl_seconds,
            "probe_timeout_seconds": self.probe_timeout_seconds,
            "max_concurrent_downloads": self.max_concurrent_downloads,
            "download_timeout_seconds": self.download_timeout_seconds,
            "speedup_factor": self.speedup_factor,
            "retry_max_attempts": self.retry_max_attempts,
            "retry_base_delay_seconds": self.retry_base_delay_seconds,
            "cache_expiry_days": self.cache_expiry_days,
            "whitelisted_domains": sorted(list(self.whitelisted_domains)),
            "enable_media_ingestion": self.enable_media_ingestion,
            "enable_twitter_video_detection": self.enable_twitter_video_detection,
            "enable_contextual_brain": self.enable_contextual_brain
        }


# Global configuration instance
media_config = MediaIngestionConfig()


def get_media_config() -> MediaIngestionConfig:
    """Get the global media ingestion configuration."""
    return media_config


def reload_media_config() -> MediaIngestionConfig:
    """Reload configuration from environment variables."""
    global media_config
    media_config = MediaIngestionConfig()
    return media_config
