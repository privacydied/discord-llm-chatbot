# Smart Media Ingestion System

## Overview

The Smart Media Ingestion System enables intelligent processing of URLs containing downloadable media (audio/video) through yt-dlp, with graceful fallback to existing web scraping when media is not available. This system prioritizes actual media content over scraped HTML, providing richer context for AI responses.

## Architecture

### Core Components

1. **Media Capability Detector** (`bot/media_capability.py`)
   - Determines if URLs contain yt-dlp-compatible media
   - Implements caching with configurable TTL
   - Supports domain whitelisting for security

2. **Media Ingestion Manager** (`bot/media_ingestion.py`)
   - Orchestrates smart routing between media and scraping flows
   - Handles concurrency control and retry logic
   - Provides seamless fallback mechanisms

3. **Configuration System** (`bot/config/media_config.py`)
   - Centralized configuration with environment variable overrides
   - Runtime configuration management
   - Feature flags for gradual rollout

### Flow Diagram

```
URL Input
    ↓
Domain Whitelist Check
    ↓
Media Capability Probe (cached)
    ↓
┌─────────────────┬─────────────────┐
│   Media Path    │  Fallback Path  │
│                 │                 │
│ yt-dlp Extract  │  Web Scraping   │
│      ↓          │       ↓         │
│ Audio Process   │  Screenshot/    │
│      ↓          │  Text Extract   │
│ STT Transcribe  │       ↓         │
│      ↓          │  Vision/Text    │
│ Context Build   │  Processing     │
└─────────────────┴─────────────────┘
    ↓
LLM Processing
    ↓
Response Generation
```

## Supported Platforms

### Whitelisted Domains
- `youtube.com` / `youtu.be` - YouTube videos and shorts
- `tiktok.com` / `vm.tiktok.com` - TikTok videos
- `twitter.com` / `x.com` - Twitter/X posts with video content

### Media Types
- Video files with extractable audio
- Audio-only content
- Live streams (where supported by yt-dlp)

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_MEDIA_INGESTION` | `true` | Enable/disable media ingestion system |
| `MEDIA_PROBE_CACHE_TTL` | `300` | Probe cache TTL in seconds (5 minutes) |
| `MEDIA_PROBE_TIMEOUT` | `10` | Probe timeout in seconds |
| `MEDIA_MAX_CONCURRENT` | `2` | Max concurrent media downloads |
| `MEDIA_DOWNLOAD_TIMEOUT` | `60` | Download timeout in seconds |
| `MEDIA_SPEEDUP_FACTOR` | `1.5` | Audio speedup factor for processing |
| `MEDIA_RETRY_MAX_ATTEMPTS` | `3` | Max retry attempts for failed downloads |
| `MEDIA_RETRY_BASE_DELAY` | `2.0` | Base delay for exponential backoff |
| `VIDEO_CACHE_DIR` | `cache/video_audio` | Audio cache directory |
| `VIDEO_CACHE_EXPIRY_DAYS` | `7` | Cache expiry in days |
| `ENABLE_TWITTER_VIDEO_DETECTION` | `true` | Enable specialized Twitter video detection |
| `USE_ENHANCED_CONTEXT` | `true` | Use contextual brain inference when available |

### Runtime Configuration

```python
from bot.config.media_config import get_media_config

config = get_media_config()

# Add custom domain
config.add_whitelisted_domain("custom-video-site.com")

# Modify settings
config.max_concurrent_downloads = 1
config.speedup_factor = 2.0
```

## Usage

### Basic Integration

The system is automatically integrated into the router and requires no manual intervention:

```python
# URLs are automatically processed through smart media ingestion
# No code changes needed - existing URL handling is enhanced
```

### Manual Usage

```python
from bot.media_ingestion import create_media_ingestion_manager
from bot.media_capability import is_media_capable_url

# Check if URL is media-capable
probe_result = await is_media_capable_url("https://youtube.com/watch?v=example")
print(f"Media capable: {probe_result.is_media_capable}")
print(f"Reason: {probe_result.reason}")

# Process URL through smart ingestion
manager = create_media_ingestion_manager(bot)
result = await manager.process_url_smart(url, message)
```

## Decision Logic

### Media Capability Detection

1. **Domain Whitelist Check**: Fast rejection of non-whitelisted domains
2. **Cache Lookup**: Check for recent probe results (TTL-based)
3. **Lightweight Probe**: Use `yt-dlp --simulate` for fast detection
4. **Result Caching**: Store results with timestamp for future use

### Routing Decision Tree

```
URL → Domain Whitelisted?
  ├─ No → Fallback to Web Scraping
  └─ Yes → Probe for Media
      ├─ Media Available → Try Media Path
      │   ├─ Success → Return Media Response
      │   └─ Failure → Fallback to Web Scraping
      └─ No Media → Fallback to Web Scraping
```

### Twitter/X Special Handling

Twitter/X URLs receive enhanced video detection:
1. General media probe first
2. If probe fails, secondary metadata check
3. Only fallback to scraping if no video detected

## Error Handling

### Graceful Degradation

- **Probe Failures**: Automatic fallback to web scraping
- **Download Failures**: Retry with exponential backoff, then fallback
- **Processing Failures**: Preserve error context, fallback gracefully
- **Timeout Handling**: Configurable timeouts with clean cancellation

### Error Categories

1. **Transient Errors**: Network issues, temporary unavailability
   - **Handling**: Retry with backoff
   - **Fallback**: Web scraping after max attempts

2. **Permanent Errors**: Unsupported format, private content
   - **Handling**: Immediate fallback
   - **Logging**: Detailed error context

3. **System Errors**: yt-dlp not available, disk space issues
   - **Handling**: Disable media ingestion temporarily
   - **Alerting**: Log critical errors for monitoring

## Performance Considerations

### Caching Strategy

- **Probe Results**: 5-minute TTL to balance freshness and performance
- **Audio Files**: 7-day expiry with LRU cleanup
- **Metadata**: Embedded in audio cache entries

### Concurrency Control

- **Download Semaphore**: Limits concurrent yt-dlp processes
- **Timeout Management**: Prevents resource exhaustion
- **Memory Usage**: Streaming processing for large files

### Optimization Tips

1. **Probe Cache Tuning**: Increase TTL for stable content
2. **Concurrent Downloads**: Adjust based on system resources
3. **Audio Cache**: Monitor disk usage and adjust expiry
4. **Speedup Factor**: Balance processing speed vs. quality

## Monitoring and Observability

### Metrics

The system exposes metrics for monitoring (when metrics are enabled):

- `media_ingestion_success_total`: Successful media processing count
- `media_ingestion_fallback_total`: Fallback to scraping count
- `media_ingestion_duration_ms`: Processing time distribution
- `media_probe_cache_hits_total`: Cache hit rate

### Logging

All decision points are logged with structured context:

```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "level": "INFO",
  "message": "Media capability probe",
  "url": "https://youtube.com/watch?v=example",
  "result": true,
  "reason": "media available",
  "cache_status": "fresh probe",
  "duration_ms": 150.5,
  "msg_id": 12345
}
```

### Health Checks

Monitor these indicators for system health:

1. **Probe Success Rate**: Should be >95% for whitelisted domains
2. **Fallback Rate**: Monitor for unexpected increases
3. **Cache Hit Rate**: Should be >60% for typical usage
4. **Processing Latency**: P95 should be <30 seconds

## Security Considerations

### Input Validation

- **Domain Whitelist**: Only approved domains are processed
- **URL Sanitization**: Malformed URLs are rejected
- **Metadata Sanitization**: All extracted content is cleaned

### Content Safety

- **Length Limits**: Metadata fields have maximum lengths
- **Control Character Removal**: Prevents injection attacks
- **Safe Character Sets**: Only printable characters allowed

### Resource Protection

- **Concurrent Limits**: Prevents resource exhaustion
- **Timeout Enforcement**: Prevents hanging processes
- **Disk Space Monitoring**: Cache cleanup prevents disk filling

## Troubleshooting

### Common Issues

#### Media Not Detected
```
Symptoms: URLs that should have media fall back to scraping
Causes: 
- yt-dlp not installed or outdated
- Network connectivity issues
- Content geo-blocked or private

Solutions:
- Verify yt-dlp installation: `yt-dlp --version`
- Check network connectivity
- Test with known working URLs
- Review probe cache for stale entries
```

#### High Fallback Rate
```
Symptoms: Most URLs fall back to scraping instead of media processing
Causes:
- Overly restrictive domain whitelist
- yt-dlp compatibility issues
- Network or service issues

Solutions:
- Review domain whitelist configuration
- Update yt-dlp to latest version
- Check service status of media platforms
- Monitor probe error logs
```

#### Performance Issues
```
Symptoms: Slow response times for media URLs
Causes:
- Too many concurrent downloads
- Large media files
- Slow network connection

Solutions:
- Reduce MEDIA_MAX_CONCURRENT setting
- Increase timeout values
- Monitor system resources
- Consider audio quality settings
```

### Debug Mode

Enable detailed logging for troubleshooting:

```bash
export LOG_LEVEL=DEBUG
export MEDIA_PROBE_TIMEOUT=30  # Increase for debugging
```

### Cache Management

Clear caches to resolve persistent issues:

```python
from bot.media_capability import media_detector

# Clear probe cache
media_detector._cache.clear()
media_detector._save_cache()

# Clear audio cache
import shutil
shutil.rmtree("cache/video_audio", ignore_errors=True)
```

## Testing

### Unit Tests

Run the comprehensive test suite:

```bash
# Run all media ingestion tests
pytest tests/test_media_capability.py tests/test_media_ingestion.py -v

# Run with coverage
pytest tests/test_media_*.py --cov=bot.media_capability --cov=bot.media_ingestion
```

### Integration Tests

Test with real URLs (requires yt-dlp):

```bash
# Run integration tests
pytest tests/test_media_*.py -m integration

# Test specific platforms
pytest tests/test_media_capability.py::TestMediaCapabilityIntegration::test_real_youtube_url
```

### Manual Testing

Test the system manually with various URL types:

```python
# Test YouTube video
url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Test TikTok video  
url = "https://www.tiktok.com/@username/video/1234567890"

# Test Twitter video
url = "https://twitter.com/user/status/1234567890"

# Test non-media URL (should fallback)
url = "https://example.com/article"
```

## Migration Guide

### From Existing URL Processing

The smart media ingestion system is designed to be a drop-in replacement:

1. **No Code Changes**: Existing URL handling automatically uses the new system
2. **Backward Compatibility**: All existing functionality is preserved
3. **Gradual Rollout**: Can be disabled via `ENABLE_MEDIA_INGESTION=false`

### Configuration Migration

Update your environment configuration:

```bash
# Add new media ingestion settings
ENABLE_MEDIA_INGESTION=true
MEDIA_PROBE_CACHE_TTL=300
MEDIA_MAX_CONCURRENT=2
MEDIA_SPEEDUP_FACTOR=1.5

# Existing video settings are reused
VIDEO_CACHE_DIR=cache/video_audio
VIDEO_CACHE_EXPIRY_DAYS=7
```

## Future Enhancements

### Planned Features

1. **Additional Platforms**: Support for more video platforms
2. **Content Analysis**: Enhanced metadata extraction and analysis
3. **Quality Selection**: Configurable audio quality preferences
4. **Batch Processing**: Efficient handling of multiple URLs
5. **Advanced Caching**: Distributed cache support for scaling

### Extension Points

The system is designed for easy extension:

```python
# Add custom domain support
config.add_whitelisted_domain("new-platform.com")

# Custom probe logic
class CustomMediaDetector(MediaCapabilityDetector):
    async def _probe_url_lightweight(self, url: str):
        # Custom probing logic
        pass

# Custom processing pipeline
class CustomMediaManager(MediaIngestionManager):
    async def _process_media_path(self, url: str, message):
        # Custom media processing
        pass
```

## Support

For issues, questions, or contributions:

1. **Documentation**: Check this guide and inline code documentation
2. **Logging**: Enable debug logging for detailed troubleshooting
3. **Testing**: Use the comprehensive test suite to validate changes
4. **Monitoring**: Use metrics and health checks to monitor system health

The Smart Media Ingestion System provides a robust, scalable solution for processing media URLs while maintaining backward compatibility and providing graceful fallback mechanisms.
