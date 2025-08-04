# 🎥 Video URL Ingestion & STT Pipeline

This document describes the video URL ingestion feature that allows the Discord bot to download, process, and transcribe audio from YouTube and TikTok videos through the existing STT pipeline.

## 🌟 Features

### **Supported Platforms**
- **YouTube**: `youtube.com`, `youtu.be` URLs
- **TikTok**: `tiktok.com`, `vm.tiktok.com` URLs

### **Core Capabilities**
- **Automatic URL Detection**: Bot automatically detects and processes video URLs in messages
- **Manual Commands**: `!watch`, `!transcribe`, `!listen` commands for explicit processing
- **Audio Normalization**: Same 16kHz mono WAV processing as voice messages
- **Speed Enhancement**: Configurable speedup (default 1.5x) for faster processing
- **Intelligent Caching**: Avoids re-downloading the same content
- **Rich Metadata**: Preserves video title, uploader, duration, and source information
- **Context Integration**: Video transcriptions are integrated into conversation context

## 🚀 Usage

### **Automatic Processing**
Simply paste a YouTube or TikTok URL in any channel where the bot is active:

```
User: Check out this video: https://youtu.be/dQw4w9WgXcQ
Bot: 🎥 Processing YouTube video...
     📊 Speed: 1.5x | Cache: Enabled
     ⏳ This may take a moment...
     
     [Rich embed with transcription and metadata]
```

### **Manual Commands**

#### Basic Usage
```bash
!watch <url>                    # Transcribe video
!transcribe <url>               # Same as watch
!listen <url>                   # Same as watch
```

#### Advanced Options
```bash
!watch <url> --speed 2.0        # Custom speed (0.5x to 3.0x)
!watch <url> --force            # Force re-download (ignore cache)
!transcribe <url> --speed 1.2 --force  # Combined options
```

#### Help and Information
```bash
!video-help                     # Show help information
!video-cache                    # Show cache statistics (Admin only)
```

### **Examples**

```bash
# Basic transcription
!watch https://youtu.be/dQw4w9WgXcQ

# Fast processing
!transcribe https://tiktok.com/@user/video/123 --speed 2.5

# Force refresh cached content
!listen https://youtu.be/example --force

# Get help
!video-help
```

## 🏗️ Architecture

### **Pipeline Flow**

```
1. URL Detection
   ├── Router detects VIDEO_URL modality
   └── Extracts URL from message content

2. Video Processing
   ├── fetch_and_prepare_url_audio()
   ├── yt-dlp downloads audio-only stream
   ├── ffmpeg normalizes to 16kHz mono WAV
   ├── ffmpeg applies speedup (atempo filter)
   └── Cache processed audio

3. STT Processing
   ├── hear_infer_from_url()
   ├── STTManager.transcribe_async()
   └── faster-whisper transcription

4. Context Integration
   ├── Combine transcription with metadata
   ├── Merge with conversation history
   └── Enhanced contextual brain inference

5. Response Generation
   ├── LLM processes enriched context
   └── Bot replies with insights/summary
```

### **Key Components**

#### **VideoIngestionManager** (`bot/video_ingest.py`)
- Handles video URL downloading and processing
- Manages caching and deduplication
- Integrates with yt-dlp and ffmpeg
- Provides rate limiting and error handling

#### **Enhanced STT Pipeline** (`bot/hear.py`)
- `hear_infer_from_url()` function for URL processing
- Preserves existing voice message functionality
- Returns rich metadata alongside transcription

#### **Router Integration** (`bot/router.py`)
- `VIDEO_URL` input modality detection
- `_flow_process_video_url()` processing flow
- Automatic URL pattern matching
- Context enrichment and LLM integration

#### **Discord Commands** (`bot/commands/video_commands.py`)
- `!watch`, `!transcribe`, `!listen` commands
- Rich embed responses with metadata
- Admin cache management commands
- Comprehensive help system

## ⚙️ Configuration

### **Environment Variables**

```bash
# Video Processing
VIDEO_MAX_DURATION=600          # Max video length in seconds (default: 10 minutes)
VIDEO_MAX_CONCURRENT=3          # Max concurrent downloads (default: 3)
VIDEO_CACHE_DIR=cache/video_audio  # Cache directory (default: cache/video_audio)
VIDEO_SPEEDUP=1.5               # Default speedup factor (default: 1.5x)
VIDEO_CACHE_EXPIRY_DAYS=7       # Cache expiry in days (default: 7)

# STT Configuration (existing)
STT_ENGINE=faster-whisper       # STT engine
WHISPER_MODEL_SIZE=base-int8    # Whisper model size
```

### **Dependencies**

The feature requires these additional dependencies:

```bash
# Core video processing
yt-dlp>=2024.1.0               # Video downloading
ffmpeg                         # Audio processing (system dependency)

# Existing STT dependencies
faster-whisper==1.1.1          # Speech-to-text
torch                          # ML framework
```

## 🛡️ Security & Rate Limiting

### **URL Validation**
- Whitelist-based URL pattern matching
- Only YouTube and TikTok domains allowed
- Prevents arbitrary URL fetching

### **Rate Limiting**
- Maximum 3 concurrent downloads
- Per-user rate limiting via Discord command cooldowns
- Exponential backoff on network failures

### **Content Validation**
- Maximum video duration enforcement (10 minutes default)
- File size and format validation
- Metadata sanitization before LLM processing

### **Caching Security**
- Deterministic cache keys (SHA256-based)
- Cache expiry to prevent indefinite storage
- Integrity checking of cached files

## 🎯 Performance Optimizations

### **Caching Strategy**
- **Raw Download Cache**: Stores original yt-dlp downloads
- **Processed Audio Cache**: Stores normalized 16kHz mono WAV files
- **Metadata Cache**: JSON index with video information
- **Cache Deduplication**: Same URL reuses existing processed audio

### **Audio Processing**
- **Audio-Only Downloads**: yt-dlp extracts audio streams only
- **Speed Enhancement**: 1.5x default speedup reduces processing time by ~33%
- **Format Optimization**: Direct WAV output from yt-dlp when possible
- **Parallel Processing**: Multiple videos can be processed concurrently

### **Memory Management**
- **Temporary File Cleanup**: Automatic cleanup of processing artifacts
- **Streaming Processing**: Large files processed in chunks
- **Resource Limits**: Configurable memory and duration limits

## 🔧 Troubleshooting

### **Common Issues**

#### Video Download Failures
```
❌ Could not download the video. It may be private, unavailable, or region-locked.
```
**Solutions:**
- Check if video is public and available
- Verify URL format is correct
- Try a different video from the same platform

#### Audio Processing Errors
```
❌ Could not process the audio from this video. The audio format may be unsupported.
```
**Solutions:**
- Ensure ffmpeg is installed and accessible
- Check video has audio track
- Try with a different video

#### STT Engine Unavailable
```
❌ STT engine not available
```
**Solutions:**
- Verify faster-whisper is installed
- Check CUDA/CPU availability
- Restart bot to reinitialize STT engine

### **Debug Commands**

```bash
# Check cache status (Admin only)
!video-cache

# Check configuration
!config-status

# Force refresh problematic video
!watch <url> --force
```

### **Logging**

The feature provides comprehensive structured logging:

```json
{
  "subsys": "video_ingest",
  "event": "transcription_complete",
  "user_id": 12345,
  "guild_id": 67890,
  "url": "https://youtu.be/example",
  "source": "youtube",
  "duration": 120.5,
  "cache_hit": false
}
```

## 🧪 Testing

### **Unit Tests**
```bash
# Run video ingestion tests
pytest tests/test_video_ingest.py -v

# Run with coverage
pytest tests/test_video_ingest.py --cov=bot.video_ingest
```

### **Integration Tests**
```bash
# Test with real videos (requires test_videos directory)
pytest tests/test_video_ingest.py::TestVideoIngestionIntegration -v
```

### **Manual Testing**

1. **Basic Functionality**
   ```bash
   !watch https://youtu.be/dQw4w9WgXcQ
   ```

2. **Cache Testing**
   ```bash
   !watch <url>          # First time (cache miss)
   !watch <url>          # Second time (cache hit)
   !watch <url> --force  # Force refresh
   ```

3. **Error Handling**
   ```bash
   !watch https://youtu.be/invalid_id
   !watch https://unsupported-site.com/video
   ```

## 🔮 Future Enhancements

### **Planned Features**
- **Additional Platforms**: Vimeo, Instagram, Twitter video support
- **Playlist Support**: Process entire YouTube playlists
- **Live Stream Support**: Real-time transcription of live streams
- **Multi-language Support**: Automatic language detection and transcription
- **Video Summarization**: AI-powered video content summarization
- **Timestamp Extraction**: Chapter/segment detection and timestamping

### **Performance Improvements**
- **Parallel Audio Processing**: Process multiple audio streams simultaneously
- **Adaptive Quality**: Dynamic quality selection based on content length
- **Predictive Caching**: Pre-cache popular or trending videos
- **CDN Integration**: Distribute cached content across multiple servers

### **Advanced Features**
- **Speaker Diarization**: Identify and separate multiple speakers
- **Sentiment Analysis**: Analyze emotional content of transcriptions
- **Topic Extraction**: Automatic topic and keyword extraction
- **Content Filtering**: Configurable content moderation and filtering

## 📚 API Reference

### **Core Functions**

#### `fetch_and_prepare_url_audio(url, speedup=1.5, force_refresh=False)`
Downloads and processes video URL audio for STT pipeline.

**Parameters:**
- `url` (str): YouTube or TikTok URL
- `speedup` (float): Audio speedup factor (0.5-3.0)
- `force_refresh` (bool): Force re-download even if cached

**Returns:**
- `ProcessedAudio`: Object with audio path, metadata, and processing info

#### `hear_infer_from_url(url, speedup=1.5, force_refresh=False)`
Transcribes audio from video URL with metadata preservation.

**Parameters:**
- `url` (str): YouTube or TikTok URL
- `speedup` (float): Audio speedup factor
- `force_refresh` (bool): Force re-download

**Returns:**
- `Dict[str, Any]`: Transcription and rich metadata

### **Data Structures**

#### `VideoMetadata`
```python
@dataclass
class VideoMetadata:
    url: str                    # Original video URL
    title: str                  # Video title
    duration_seconds: float     # Original duration
    uploader: str              # Channel/user name
    upload_date: str           # Upload date (YYYYMMDD)
    source_type: str           # 'youtube' or 'tiktok'
```

#### `ProcessedAudio`
```python
@dataclass
class ProcessedAudio:
    audio_path: Path           # Path to processed audio file
    metadata: VideoMetadata    # Video metadata
    processed_duration_seconds: float  # Duration after speedup
    speedup_factor: float      # Applied speedup factor
    cache_hit: bool           # Whether result was cached
    timestamp: datetime       # Processing timestamp
```

---

This video ingestion feature seamlessly extends the existing STT pipeline to support YouTube and TikTok content, providing users with powerful video analysis capabilities while maintaining the bot's performance and reliability standards.
