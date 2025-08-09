"""
Media capability detection module for smart URL routing.
Determines whether URLs contain downloadable media that yt-dlp can handle.
"""
import asyncio
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Set, Tuple
from urllib.parse import urlparse

from .util.logging import get_logger

logger = get_logger(__name__)

# Configuration
PROBE_CACHE_TTL_SECONDS = int(os.getenv("MEDIA_PROBE_CACHE_TTL", "300"))  # 5 minutes default
PROBE_TIMEOUT_SECONDS = int(os.getenv("MEDIA_PROBE_TIMEOUT", "10"))  # 10 seconds default
CACHE_DIR = Path("cache/media_probes")

# Domains that should be probed for media content
# These domains may or may not have video/audio - we probe with yt-dlp and fallback to scraping
MEDIA_CAPABLE_DOMAINS = {
    # Video platforms (high confidence)
    "youtube.com",
    "youtu.be", 
    "tiktok.com",
    "m.tiktok.com",
    "vm.tiktok.com",
    "vimeo.com",
    "dailymotion.com",
    "twitch.tv",
    "bilibili.com",
    "rumble.com",
    "odysee.com",
    "lbry.tv",
    "veoh.com",
    "metacafe.com",
    
    # Audio/music platforms
    "soundcloud.com",
    "bandcamp.com",
    "mixcloud.com",
    "audiomack.com",
    
    # Educational/conference
    "ted.com",
    "archive.org",
    
    # Social media with mixed content (probe first, fallback to scraping)
    "twitter.com",
    "x.com",
    "reddit.com",
    "v.redd.it",  # Reddit video direct links
    "facebook.com",
    "fb.com",
    "instagram.com",
    "linkedin.com"
}

@dataclass
class ProbeResult:
    """Result of media capability probe."""
    is_media_capable: bool
    reason: str
    cached: bool = False
    probe_duration_ms: Optional[float] = None


class MediaCapabilityDetector:
    """Detects whether URLs contain media that can be processed by yt-dlp."""
    
    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "probe_cache.json"
        self._cache: Dict[str, Dict] = {}
        self._load_cache()
        logger.info(f"✔ MediaCapabilityDetector initialized with cache: {self.cache_dir}")
    
    def _load_cache(self):
        """Load probe cache from disk."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    self._cache = json.load(f)
                logger.debug(f"Loaded {len(self._cache)} cached probe results")
        except Exception as e:
            logger.warning(f"Failed to load probe cache: {e}")
            self._cache = {}
    
    def _save_cache(self):
        """Save probe cache to disk."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self._cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save probe cache: {e}")
    
    def _get_cache_key(self, url: str) -> str:
        """Generate cache key for URL."""
        return hashlib.sha256(url.encode()).hexdigest()[:16]
    
    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        """Check if cache entry is still valid based on TTL."""
        if 'timestamp' not in cache_entry:
            return False
        
        age_seconds = time.time() - cache_entry['timestamp']
        return age_seconds < PROBE_CACHE_TTL_SECONDS
    
    def _is_whitelisted_domain(self, url: str) -> bool:
        """Check if URL domain is in the whitelist."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Remove www. prefix if present
            if domain.startswith('www.'):
                domain = domain[4:]
            
            return domain in MEDIA_CAPABLE_DOMAINS
        except Exception:
            return False
    
    async def _probe_url_lightweight(self, url: str) -> Tuple[bool, str]:
        """
        Lightweight probe to check if URL has downloadable media.
        Uses yt-dlp's --simulate flag for fast, non-destructive checking.
        """
        logger.debug(f"🔍 Probing URL for media capability: {url}")
        
        try:
            # Use yt-dlp to simulate extraction without downloading
            cmd = [
                'yt-dlp',
                '--simulate',
                '--no-playlist',
                '--quiet',
                '--no-warnings',
                '--print', 'title',
                '--print', 'duration',
                url
            ]
            
            proc = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                ),
                timeout=PROBE_TIMEOUT_SECONDS
            )
            
            stdout, stderr = await proc.communicate()
            
            if proc.returncode == 0:
                # yt-dlp successfully extracted metadata - check if it's actually video content
                output_lines = stdout.decode().strip().split('\n')
                if len(output_lines) >= 2 and output_lines[0] and output_lines[1]:
                    title = output_lines[0].strip()
                    duration = output_lines[1].strip()
                    
                    # More rigorous check - ensure we have meaningful video metadata
                    if title and title != "NA" and duration and duration != "NA":
                        try:
                            # Try to parse duration as a number (seconds)
                            float(duration)
                            return True, "video content detected"
                        except ValueError:
                            # Duration is not a number, might be text-only content
                            return False, "no valid video duration found"
                    else:
                        return False, "incomplete video metadata"
                else:
                    return False, "metadata extraction incomplete"
            else:
                # yt-dlp failed to extract - likely no media
                error_output = stderr.decode().lower()
                if "unsupported url" in error_output or "no video" in error_output:
                    return False, "no video content found"
                elif "private" in error_output or "unavailable" in error_output:
                    return False, "content unavailable"
                elif "not a video" in error_output or "no formats" in error_output:
                    return False, "no video formats available"
                else:
                    return False, f"probe failed: {error_output[:100]}"
                    
        except asyncio.TimeoutError:
            return False, "probe timeout"
        except FileNotFoundError:
            logger.error("yt-dlp not found - media capability detection disabled")
            return False, "yt-dlp not available"
        except Exception as e:
            logger.debug(f"Probe exception for {url}: {e}")
            return False, f"probe error: {str(e)[:50]}"
    
    async def is_media_capable(self, url: str) -> ProbeResult:
        """
        Determine if URL contains media that can be processed by yt-dlp.
        
        Args:
            url: URL to check
            
        Returns:
            ProbeResult with capability decision and reasoning
        """
        start_time = time.time()
        
        # Quick domain whitelist check
        if not self._is_whitelisted_domain(url):
            return ProbeResult(
                is_media_capable=False,
                reason="domain not whitelisted",
                probe_duration_ms=0
            )
        
        # Check cache first
        cache_key = self._get_cache_key(url)
        if cache_key in self._cache:
            cache_entry = self._cache[cache_key]
            if self._is_cache_valid(cache_entry):
                logger.debug(f"💾 Cache hit for media probe: {url}")
                return ProbeResult(
                    is_media_capable=cache_entry['is_media_capable'],
                    reason=cache_entry['reason'],
                    cached=True,
                    probe_duration_ms=cache_entry.get('probe_duration_ms', 0)
                )
            else:
                # Remove expired entry
                del self._cache[cache_key]
        
        # Perform actual probe
        try:
            is_capable, reason = await self._probe_url_lightweight(url)
            probe_duration_ms = (time.time() - start_time) * 1000
            
            # Cache the result
            self._cache[cache_key] = {
                'url': url,
                'is_media_capable': is_capable,
                'reason': reason,
                'probe_duration_ms': probe_duration_ms,
                'timestamp': time.time()
            }
            
            # Periodically save cache (every 10 entries)
            if len(self._cache) % 10 == 0:
                self._save_cache()
            
            logger.debug(f"🔍 Probe result for {url}: {is_capable} ({reason}) in {probe_duration_ms:.1f}ms")
            
            return ProbeResult(
                is_media_capable=is_capable,
                reason=reason,
                cached=False,
                probe_duration_ms=probe_duration_ms
            )
            
        except Exception as e:
            probe_duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Media capability probe failed for {url}: {e}")
            
            return ProbeResult(
                is_media_capable=False,
                reason=f"probe exception: {str(e)[:50]}",
                cached=False,
                probe_duration_ms=probe_duration_ms
            )
    
    async def is_twitter_video_present(self, url: str) -> ProbeResult:
        """
        Specialized check for Twitter/X URLs to detect video presence.
        This is more thorough than the general probe for Twitter-specific cases.
        """
        if not any(domain in url.lower() for domain in ['twitter.com', 'x.com']):
            return ProbeResult(
                is_media_capable=False,
                reason="not a twitter/x url"
            )
        
        # Use the general probe first
        general_result = await self.is_media_capable(url)
        
        if general_result.is_media_capable:
            return ProbeResult(
                is_media_capable=True,
                reason="twitter video detected via probe",
                cached=general_result.cached,
                probe_duration_ms=general_result.probe_duration_ms
            )
        
        # For Twitter, if general probe fails, we can try a secondary metadata check
        # This is a placeholder for potential future enhancement
        logger.debug(f"Twitter URL {url} did not pass general media probe: {general_result.reason}")
        
        return ProbeResult(
            is_media_capable=False,
            reason=f"no twitter video found: {general_result.reason}",
            cached=general_result.cached,
            probe_duration_ms=general_result.probe_duration_ms
        )
    
    def cleanup_expired_cache(self):
        """Remove expired entries from cache."""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self._cache.items():
            if 'timestamp' not in entry:
                expired_keys.append(key)
            elif current_time - entry['timestamp'] > PROBE_CACHE_TTL_SECONDS:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
            self._save_cache()
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics for monitoring."""
        valid_entries = 0
        expired_entries = 0
        current_time = time.time()
        
        for entry in self._cache.values():
            if 'timestamp' in entry and current_time - entry['timestamp'] < PROBE_CACHE_TTL_SECONDS:
                valid_entries += 1
            else:
                expired_entries += 1
        
        return {
            'total_entries': len(self._cache),
            'valid_entries': valid_entries,
            'expired_entries': expired_entries,
            'cache_hit_potential': valid_entries
        }


# Global instance
media_detector = MediaCapabilityDetector()


async def is_media_capable_url(url: str) -> ProbeResult:
    """
    Convenience function to check if URL is media-capable.
    
    Args:
        url: URL to check
        
    Returns:
        ProbeResult with capability decision
    """
    return await media_detector.is_media_capable(url)


async def is_twitter_video_url(url: str) -> ProbeResult:
    """
    Convenience function to check for Twitter/X video presence.
    
    Args:
        url: Twitter/X URL to check
        
    Returns:
        ProbeResult with video presence decision
    """
    return await media_detector.is_twitter_video_present(url)
