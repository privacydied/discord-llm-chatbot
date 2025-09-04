"""
Fast classification system for router speed optimization. [PA][CA]

This module provides zero-I/O classification and planning using pre-compiled regex tables
and deterministic routing logic. All classification happens without network calls or heavy CPU.

Key optimizations:
- Pre-compiled regex patterns cached at module load
- O(1) host-to-modality mapping
- Zero network I/O during classification
- Streaming eligibility computed before first network call
- Tweet URLs always route to Tweet flow, never GENERAL_URL
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Any
from urllib.parse import urlparse

from .modality import InputModality, InputItem
from .util.logging import get_logger

logger = get_logger(__name__)

# Pre-compiled regex patterns for fast classification [PA]
_COMPILED_PATTERNS: Dict[str, re.Pattern] = {}

def _compile_patterns() -> None:
    """Compile all regex patterns at module load time. [PA]"""
    global _COMPILED_PATTERNS
    
    if _COMPILED_PATTERNS:
        return  # Already compiled
    
    patterns = {
        # Twitter/X patterns (multiple domains and formats)
        'twitter_status': re.compile(
            r'https?://(?:www\.)?(twitter\.com|x\.com)/[^/]+/status/(\d+)(?:\?.*)?$',
            re.IGNORECASE
        ),
        'twitter_short': re.compile(
            r'https?://(?:www\.)?(t\.co)/(\w+)$',
            re.IGNORECASE
        ),
        
        # Video URL patterns
        'youtube': re.compile(
            r'https?://(?:www\.)?(youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]+)',
            re.IGNORECASE
        ),
        'tiktok': re.compile(
            r'https?://(?:www\.|m\.|vm\.)?(tiktok\.com)/([@\w]+/video/\d+|t/\w+|@[\w.]+/video/\d+)',
            re.IGNORECASE
        ),
        'instagram_video': re.compile(
            r'https?://(?:www\.)?(instagram\.com)/(reel|p)/([a-zA-Z0-9_-]+)',
            re.IGNORECASE
        ),
        
        # Direct image URL patterns
        'direct_image': re.compile(
            r'https?://[^\s<>"\'[\]{}|\\^`]+\.(jpg|jpeg|png|gif|webp|bmp|svg)(?:\?[^\s]*)?$',
            re.IGNORECASE
        ),
        
        # Direct PDF URL patterns
        'direct_pdf': re.compile(
            r'https?://[^\s<>"\'[\]{}|\\^`]+\.pdf(?:\?[^\s]*)?$',
            re.IGNORECASE
        ),
        
        # Audio/Video file patterns
        'audio_video_file': re.compile(
            r'\.(mp3|wav|flac|aac|ogg|m4a|mp4|avi|mkv|mov|wmv|flv|webm)$',
            re.IGNORECASE
        ),
        
        # General URL pattern
        'general_url': re.compile(
            r'https?://[^\s<>"\'[\]{}|\\^`]+',
            re.IGNORECASE
        ),
        
        # Command patterns
        'command_prefix': re.compile(r'^!(\w+)', re.IGNORECASE),
        
        # Inline search directives: capture parenthesized content, allow whitespace, multiline
        'inline_search': re.compile(r'\[search\s*\((.*?)\)\]', re.IGNORECASE | re.DOTALL),
        
        # Bot mention pattern (will be dynamically compiled)
        'bot_mention': None,  # Set dynamically in Router.__init__
    }
    
    _COMPILED_PATTERNS.update(patterns)
    logger.info(f"✅ Compiled {len(patterns)} regex patterns for fast classification")

# Host-to-modality mapping for O(1) lookups [PA]
_HOST_MODALITY_MAP: Dict[str, InputModality] = {
    # Twitter/X domains
    'twitter.com': InputModality.GENERAL_URL,  # Will be upgraded to Tweet flow
    'x.com': InputModality.GENERAL_URL,        # Will be upgraded to Tweet flow
    't.co': InputModality.GENERAL_URL,         # Will be upgraded to Tweet flow
    
    # Video platforms
    'youtube.com': InputModality.VIDEO_URL,
    'youtu.be': InputModality.VIDEO_URL,
    'tiktok.com': InputModality.VIDEO_URL,
    'm.tiktok.com': InputModality.VIDEO_URL,
    'vm.tiktok.com': InputModality.VIDEO_URL,
    'instagram.com': InputModality.VIDEO_URL,
    'vimeo.com': InputModality.VIDEO_URL,
    'twitch.tv': InputModality.VIDEO_URL,
    
    # Known SPA/WAF hosts (skip tier A, go to Playwright)
    'medium.com': InputModality.GENERAL_URL,
    'heavy.com': InputModality.GENERAL_URL,
    'reddit.com': InputModality.GENERAL_URL,
    'stackoverflow.com': InputModality.GENERAL_URL,
}

@dataclass
class ClassificationResult:
    """Result of fast classification without I/O. [CA]"""
    modality: InputModality
    host: Optional[str] = None
    is_twitter: bool = False
    is_direct_image: bool = False
    is_direct_pdf: bool = False
    is_video_capable: bool = False
    is_spa_host: bool = False
    extracted_id: Optional[str] = None  # Tweet ID, video ID, etc.
    confidence: float = 1.0

@dataclass
class PlanResult:
    """Result of zero-I/O planning phase. [CA]"""
    items: List[Tuple[InputItem, ClassificationResult]]
    streaming_eligible: bool
    streaming_reason: str
    text_content: str
    has_commands: bool
    has_inline_searches: bool
    estimated_heavy_work: bool
    plan_duration_ms: float

class FastClassifier:
    """Fast, zero-I/O classifier for router optimization. [PA][CA]"""
    
    def __init__(self, bot_user_id: Optional[int] = None):
        """Initialize fast classifier with bot user ID for mention detection."""
        _compile_patterns()
        
        # Compile bot mention pattern if bot user ID provided
        if bot_user_id:
            _COMPILED_PATTERNS['bot_mention'] = re.compile(
                fr'<@!?{bot_user_id}>',
                re.IGNORECASE
            )
    
    def classify_url(self, url: str) -> ClassificationResult:
        """Classify a single URL without any I/O. [PA][IV]"""
        try:
            parsed = urlparse(url)
            host = parsed.netloc.lower()
            
            # Remove www. prefix for consistent mapping
            if host.startswith('www.'):
                host = host[4:]
            
            # Check for Twitter/X URLs first (highest priority)
            if _COMPILED_PATTERNS['twitter_status'].match(url):
                match = _COMPILED_PATTERNS['twitter_status'].match(url)
                tweet_id = match.group(2) if match else None
                return ClassificationResult(
                    modality=InputModality.GENERAL_URL,  # Will be routed to Tweet flow
                    host=host,
                    is_twitter=True,
                    extracted_id=tweet_id,
                    confidence=1.0
                )
            
            if _COMPILED_PATTERNS['twitter_short'].match(url):
                return ClassificationResult(
                    modality=InputModality.GENERAL_URL,  # Will be routed to Tweet flow
                    host=host,
                    is_twitter=True,
                    confidence=0.9  # Short URLs might redirect elsewhere
                )
            
            # Check for direct image URLs
            if _COMPILED_PATTERNS['direct_image'].match(url):
                return ClassificationResult(
                    modality=InputModality.SINGLE_IMAGE,
                    host=host,
                    is_direct_image=True,
                    confidence=1.0
                )
            
            # Check for direct PDF URLs
            if _COMPILED_PATTERNS['direct_pdf'].match(url):
                return ClassificationResult(
                    modality=InputModality.PDF_DOCUMENT,
                    host=host,
                    is_direct_pdf=True,
                    confidence=1.0
                )
            
            # Check for video platforms
            if _COMPILED_PATTERNS['youtube'].match(url):
                match = _COMPILED_PATTERNS['youtube'].match(url)
                video_id = match.group(2) if match else None
                return ClassificationResult(
                    modality=InputModality.VIDEO_URL,
                    host=host,
                    is_video_capable=True,
                    extracted_id=video_id,
                    confidence=1.0
                )
            
            if _COMPILED_PATTERNS['tiktok'].match(url):
                return ClassificationResult(
                    modality=InputModality.VIDEO_URL,
                    host=host,
                    is_video_capable=True,
                    confidence=1.0
                )
            
            if _COMPILED_PATTERNS['instagram_video'].match(url):
                return ClassificationResult(
                    modality=InputModality.VIDEO_URL,
                    host=host,
                    is_video_capable=True,
                    confidence=1.0
                )
            
            # Use host-based mapping for known platforms
            if host in _HOST_MODALITY_MAP:
                modality = _HOST_MODALITY_MAP[host]
                is_spa = host in ['medium.com', 'heavy.com', 'reddit.com', 'stackoverflow.com']
                return ClassificationResult(
                    modality=modality,
                    host=host,
                    is_video_capable=(modality == InputModality.VIDEO_URL),
                    is_spa_host=is_spa,
                    confidence=0.8
                )
            
            # Default to general URL
            return ClassificationResult(
                modality=InputModality.GENERAL_URL,
                host=host,
                confidence=0.5
            )
            
        except Exception as e:
            logger.warning(f"URL classification failed for {url}: {e}")
            return ClassificationResult(
                modality=InputModality.GENERAL_URL,
                confidence=0.1
            )
    
    def classify_attachment(self, attachment) -> ClassificationResult:
        """Classify a Discord attachment without I/O. [PA][IV]"""
        try:
            filename = attachment.filename.lower()
            
            # Check for audio/video files
            if _COMPILED_PATTERNS['audio_video_file'].search(filename):
                return ClassificationResult(
                    modality=InputModality.AUDIO_VIDEO_FILE,
                    confidence=1.0
                )
            
            # Check for PDF files
            if filename.endswith('.pdf'):
                return ClassificationResult(
                    modality=InputModality.PDF_DOCUMENT,
                    confidence=1.0
                )
            
            # Check for Word documents
            if filename.endswith(('.doc', '.docx')):
                return ClassificationResult(
                    modality=InputModality.PDF_DOCUMENT,  # Same handler
                    confidence=1.0
                )
            
            # Check for images (most common case)  
            if filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp')):
                return ClassificationResult(
                    modality=InputModality.SINGLE_IMAGE,
                    confidence=1.0
                )
            
            # Default to single image for unknown attachments
            return ClassificationResult(
                modality=InputModality.SINGLE_IMAGE,
                confidence=0.3
            )
            
        except Exception as e:
            logger.warning(f"Attachment classification failed: {e}")
            return ClassificationResult(
                modality=InputModality.SINGLE_IMAGE,
                confidence=0.1
            )
    
    def extract_inline_searches(self, text: str) -> List[str]:
        """Extract inline search directives from text. [PA][IV]"""
        try:
            matches = _COMPILED_PATTERNS['inline_search'].findall(text)
            return [match.strip() for match in matches]
        except Exception as e:
            logger.warning(f"Inline search extraction failed: {e}")
            return []
    
    def has_command_prefix(self, text: str) -> bool:
        """Check if text starts with a command prefix. [PA][IV]"""
        try:
            # Remove bot mention first
            if _COMPILED_PATTERNS.get('bot_mention'):
                text = _COMPILED_PATTERNS['bot_mention'].sub('', text).strip()
            
            return bool(_COMPILED_PATTERNS['command_prefix'].match(text))
        except Exception as e:
            logger.warning(f"Command prefix check failed: {e}")
            return False
    
    def plan_message(self, message, items: List[InputItem]) -> PlanResult:
        """Create execution plan for message without any I/O. [PA][CA]"""
        start_time = time.time()
        
        try:
            # Classify all items
            classified_items = []
            has_heavy_work = False
            has_video = False
            has_ocr = False
            
            for item in items:
                if item.source_type == "url":
                    classification = self.classify_url(str(item.payload))
                elif item.source_type == "attachment":
                    classification = self.classify_attachment(item.payload)
                else:
                    # Embed or other
                    classification = ClassificationResult(
                        modality=InputModality.GENERAL_URL,
                        confidence=0.5
                    )
                
                classified_items.append((item, classification))
                
                # Track heavy work indicators
                if classification.modality in [InputModality.VIDEO_URL, InputModality.AUDIO_VIDEO_FILE]:
                    has_heavy_work = True
                    has_video = True
                elif classification.modality in [InputModality.PDF_DOCUMENT, InputModality.PDF_OCR]:
                    has_heavy_work = True
                    has_ocr = True
            
            # Process text content
            content = message.content or ""
            
            # Remove bot mention
            if _COMPILED_PATTERNS.get('bot_mention'):
                content = _COMPILED_PATTERNS['bot_mention'].sub('', content).strip()
            
            # Remove URLs that will be processed separately
            content = _COMPILED_PATTERNS['general_url'].sub('', content).strip()
            
            # Check for commands and inline searches
            has_commands = self.has_command_prefix(message.content or "")
            inline_searches = self.extract_inline_searches(content)
            has_inline_searches = len(inline_searches) > 0
            
            # Determine streaming eligibility
            streaming_eligible = False
            streaming_reason = "TEXT_ONLY"
            
            if has_heavy_work:
                streaming_eligible = True
                if has_video:
                    streaming_reason = "VIDEO_STT"
                elif has_ocr:
                    streaming_reason = "PDF_OCR"
                else:
                    streaming_reason = "HEAVY_PROCESSING"
            elif len(classified_items) > 1:
                streaming_eligible = True
                streaming_reason = "MULTI_ITEM"
            elif any(c.modality == InputModality.SINGLE_IMAGE for _, c in classified_items):
                # Images can stream for vision processing
                streaming_eligible = True
                streaming_reason = "VISION_PROCESSING"
            
            # Text-only flows should never stream
            if not classified_items and not has_inline_searches:
                streaming_eligible = False
                streaming_reason = "TEXT_ONLY"
            
            plan_duration_ms = (time.time() - start_time) * 1000
            
            return PlanResult(
                items=classified_items,
                streaming_eligible=streaming_eligible,
                streaming_reason=streaming_reason,
                text_content=content,
                has_commands=has_commands,
                has_inline_searches=has_inline_searches,
                estimated_heavy_work=has_heavy_work,
                plan_duration_ms=plan_duration_ms
            )
            
        except Exception as e:
            logger.error(f"Message planning failed: {e}", exc_info=True)
            plan_duration_ms = (time.time() - start_time) * 1000
            
            # Return safe fallback plan
            return PlanResult(
                items=[(item, ClassificationResult(InputModality.GENERAL_URL)) for item in items],
                streaming_eligible=False,
                streaming_reason="PLANNING_ERROR",
                text_content=message.content or "",
                has_commands=False,
                has_inline_searches=False,
                estimated_heavy_work=False,
                plan_duration_ms=plan_duration_ms
            )

# Module-level classifier instance
_classifier_instance: Optional[FastClassifier] = None

def get_classifier(bot_user_id: Optional[int] = None) -> FastClassifier:
    """Get or create the module-level classifier instance. [CA]"""
    global _classifier_instance
    
    if _classifier_instance is None:
        _classifier_instance = FastClassifier(bot_user_id)
    
    return _classifier_instance

def warm_classifier() -> None:
    """Warm up the classifier by compiling patterns. [PA]"""
    _compile_patterns()
    logger.info("🔥 Fast classifier warmed up")
