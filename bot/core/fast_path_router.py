"""
Fast-Path Router with Decision Budget - Optimized routing for simple DM messages.
Implements PA (Performance Awareness) and REH (Robust Error Handling) rules.
"""
import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, Callable, List, Union
import re

from discord import Message, DMChannel, Embed, File

from .phase_constants import PhaseConstants as PC
from .phase_timing import get_timing_manager, PipelineTracker
from ..util.logging import get_logger

logger = get_logger(__name__)

class MessageComplexity(Enum):
    """Classification of message complexity for routing decisions."""
    SIMPLE_TEXT = "simple_text"           # Plain text under limits
    COMPLEX_TEXT = "complex_text"         # Long text or special patterns  
    MULTIMODAL = "multimodal"            # Has attachments/embeds/URLs
    COMMAND = "command"                   # Bot commands
    UNKNOWN = "unknown"                   # Couldn't classify

class RouteDecision(Enum):
    """Router decision outcomes."""
    FAST_PATH_TEXT = "fast_path_text"     # Direct to text processing
    STANDARD_PIPELINE = "standard_pipeline"  # Full multimodal pipeline
    COMMAND_HANDLER = "command_handler"    # Command processing
    REJECT = "reject"                     # Don't process

@dataclass
class RouteAnalysis:
    """Analysis result from router decision process."""
    complexity: MessageComplexity
    decision: RouteDecision
    decision_time_ms: int
    confidence: float = 1.0
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Fast-path optimizations detected
    skip_context_heavy: bool = False
    skip_rag_search: bool = False
    skip_modality_detection: bool = False
    use_simple_template: bool = False

@dataclass
class MessageFeatures:
    """Extracted features for routing decisions."""
    char_count: int
    word_count: int
    line_count: int
    has_attachments: bool
    has_embeds: bool
    has_urls: bool
    has_mentions: bool
    is_reply: bool
    is_dm: bool
    has_command_prefix: bool
    
    # Content patterns
    has_questions: bool = False
    has_code_blocks: bool = False
    has_special_chars: bool = False
    language_complexity: float = 0.0  # 0-1 score

class FastPathClassifier:
    """Lightweight message classifier for routing decisions."""
    
    def __init__(self):
        # Pre-compiled regex patterns for performance [PA]
        self.url_pattern = re.compile(r'https?://[^\s]+')
        self.question_pattern = re.compile(r'\?|\bhow\b|\bwhat\b|\bwhy\b|\bwhen\b|\bwhere\b', re.IGNORECASE)
        self.code_pattern = re.compile(r'```|`[^`]+`')
        self.special_char_pattern = re.compile(r'[^\w\s\.\?\!,;:\'"()-]')
        self.command_pattern = re.compile(r'^[!\./]')
        
        # Simple patterns that indicate complexity
        self.complexity_indicators = [
            r'\b(analyze|explain|summarize|compare|contrast)\b',  # Analysis keywords
            r'\b(step by step|detailed|comprehensive)\b',        # Detail requests
            r'\b(code|programming|debug|error)\b',               # Technical content
            r'\b(help me with|assistance with|can you)\b'       # Help requests
        ]
        self.complexity_pattern = re.compile('|'.join(self.complexity_indicators), re.IGNORECASE)
        
        logger.debug("✅ FastPathClassifier initialized")
    
    def extract_features(self, message: Message) -> MessageFeatures:
        """Extract features from message for classification [PA]."""
        content = message.content or ""
        
        return MessageFeatures(
            char_count=len(content),
            word_count=len(content.split()),
            line_count=len(content.split('\n')),
            has_attachments=len(message.attachments) > 0,
            has_embeds=len(message.embeds) > 0,
            has_urls=bool(self.url_pattern.search(content)),
            has_mentions=len(message.mentions) > 0 or len(message.role_mentions) > 0,
            is_reply=message.reference is not None,
            is_dm=isinstance(message.channel, DMChannel),
            has_command_prefix=bool(self.command_pattern.match(content)),
            has_questions=bool(self.question_pattern.search(content)),
            has_code_blocks=bool(self.code_pattern.search(content)),
            has_special_chars=bool(self.special_char_pattern.search(content)),
            language_complexity=self._calculate_language_complexity(content)
        )
    
    def _calculate_language_complexity(self, content: str) -> float:
        """Calculate language complexity score 0-1 [PA]."""
        if not content:
            return 0.0
        
        complexity_score = 0.0
        
        # Length complexity (longer = more complex)
        length_score = min(len(content) / 500, 1.0)  # Cap at 500 chars
        complexity_score += length_score * 0.3
        
        # Pattern complexity
        if self.complexity_pattern.search(content):
            complexity_score += 0.4
        
        # Structural complexity
        if content.count('\n') > 2:  # Multi-line
            complexity_score += 0.2
        
        if self.code_pattern.search(content):  # Has code
            complexity_score += 0.3
        
        return min(complexity_score, 1.0)
    
    def classify_complexity(self, features: MessageFeatures) -> MessageComplexity:
        """Classify message complexity based on features [PA]."""
        
        # Command detection
        if features.has_command_prefix:
            return MessageComplexity.COMMAND
        
        # Multimodal detection (has non-text content)
        if (features.has_attachments or features.has_embeds or 
            features.has_urls):
            return MessageComplexity.MULTIMODAL
        
        # Text complexity analysis
        complexity_factors = 0
        
        # Length factors
        if features.char_count > 200:  # Long text
            complexity_factors += 1
        if features.word_count > 50:   # Many words
            complexity_factors += 1
        if features.line_count > 3:    # Multi-line
            complexity_factors += 1
        
        # Content factors
        if features.has_questions:
            complexity_factors += 1
        if features.has_code_blocks:
            complexity_factors += 2  # Code is complex
        if features.language_complexity > 0.6:
            complexity_factors += 1
        if features.has_mentions or features.is_reply:
            complexity_factors += 1  # Context-dependent
        
        # Classify based on total complexity
        if complexity_factors >= 3:
            return MessageComplexity.COMPLEX_TEXT
        else:
            return MessageComplexity.SIMPLE_TEXT

class FastPathRouter:
    """High-performance router with fast-path optimizations and decision budget."""
    
    def __init__(self, decision_budget_ms: int = None):
        self.decision_budget_ms = decision_budget_ms or PC.ROUTER_DECISION_BUDGET_MS
        self.classifier = FastPathClassifier()
        
        # Performance stats
        self.stats = {
            "total_decisions": 0,
            "fast_path_decisions": 0,
            "budget_exceeded": 0,
            "avg_decision_time_ms": 0,
            "complexity_distribution": {complexity.value: 0 for complexity in MessageComplexity}
        }
        
        logger.info(f"🚀 FastPathRouter initialized (budget: {self.decision_budget_ms}ms)")
    
    async def analyze_message_route(
        self,
        message: Message,
        tracker: Optional[PipelineTracker] = None
    ) -> RouteAnalysis:
        """Analyze message and determine optimal routing path [PA]."""
        start_time = time.time()
        
        try:
            # Use phase timing if available [PA]
            timing_manager = get_timing_manager()
            tracker = timing_manager.create_pipeline_tracker(
                msg_id=str(message.id),
                user_id=str(message.author.id),
                guild_id=str(message.guild.id) if message.guild else None
            )
            
            phase_context = timing_manager.track_phase(
                tracker, PC.PHASE_ROUTER_DISPATCH,
                message_length=len(message.content),
                has_attachments=len(message.attachments) > 0
            )
            
            # Python 3.12 compatibility: use contextlib.nullcontext instead of asyncio.nullcontext
            from contextlib import nullcontext
            async with phase_context if phase_context else nullcontext():
                # Extract message features quickly [PA]
                features = self.classifier.extract_features(message)
                
                # Classify complexity
                complexity = self.classifier.classify_complexity(features)
                
                # Make routing decision with budget enforcement [REH]
                decision_start = time.time()
                decision, optimizations = await self._make_routing_decision(
                    features, 
                    complexity,
                    self.decision_budget_ms
                )
                decision_time_ms = int((time.time() - decision_start) * 1000)
                
                # Check budget exceeded [REH]
                total_time_ms = int((time.time() - start_time) * 1000)
                if total_time_ms > self.decision_budget_ms:
                    self.stats["budget_exceeded"] += 1
                    logger.warning(f"⚠️ Router decision budget exceeded: {total_time_ms}ms > {self.decision_budget_ms}ms")
                    
                    # Force fast-path if budget exceeded
                    if complexity == MessageComplexity.SIMPLE_TEXT:
                        decision = RouteDecision.FAST_PATH_TEXT
                        logger.info("🚀 Forcing fast-path due to budget exceeded")
                
                # Create analysis result
                analysis = RouteAnalysis(
                    complexity=complexity,
                    decision=decision,
                    decision_time_ms=total_time_ms,
                    confidence=self._calculate_confidence(features, complexity, decision),
                    reasoning=self._generate_reasoning(features, complexity, decision),
                    metadata={
                        "features": {
                            "char_count": features.char_count,
                            "word_count": features.word_count,
                            "is_dm": features.is_dm,
                            "has_attachments": features.has_attachments,
                            "language_complexity": features.language_complexity
                        },
                        "decision_time_ms": decision_time_ms
                    },
                    **optimizations
                )
                
                # Update statistics
                self._update_stats(analysis)
                
                # Log decision with Rich formatting [CA]
                self._log_routing_decision(message, analysis)
                
                return analysis
                
        except asyncio.TimeoutError:
            # Emergency fallback on timeout [REH]
            logger.error(f"❌ Router decision timeout after {self.decision_budget_ms}ms")
            self.stats["budget_exceeded"] += 1
            
            # Default to safe fast-path for DMs, standard for guild
            emergency_decision = (RouteDecision.FAST_PATH_TEXT if isinstance(message.channel, DMChannel) 
                                else RouteDecision.STANDARD_PIPELINE)
            
            return RouteAnalysis(
                complexity=MessageComplexity.UNKNOWN,
                decision=emergency_decision,
                decision_time_ms=self.decision_budget_ms,
                confidence=0.5,
                reasoning="Emergency timeout fallback",
                metadata={"timeout": True}
            )
        
        except Exception as e:
            logger.error(f"❌ Router decision error: {e}", exc_info=True)
            
            # Safe fallback [REH]
            return RouteAnalysis(
                complexity=MessageComplexity.UNKNOWN,
                decision=RouteDecision.STANDARD_PIPELINE,
                decision_time_ms=int((time.time() - start_time) * 1000),
                confidence=0.5,
                reasoning=f"Error fallback: {str(e)}",
                metadata={"error": str(e)}
            )
    
    async def _make_routing_decision(
        self,
        features: MessageFeatures,
        complexity: MessageComplexity,
        remaining_budget_ms: int
    ) -> tuple[RouteDecision, Dict[str, bool]]:
        """Make routing decision with optimizations [PA]."""
        optimizations = {
            "skip_context_heavy": False,
            "skip_rag_search": False,
            "skip_modality_detection": False,
            "use_simple_template": False
        }
        
        # Command routing
        if complexity == MessageComplexity.COMMAND:
            return RouteDecision.COMMAND_HANDLER, optimizations
        
        # Multimodal routing
        if complexity == MessageComplexity.MULTIMODAL:
            return RouteDecision.STANDARD_PIPELINE, optimizations
        
        # Text routing with fast-path detection
        if complexity == MessageComplexity.SIMPLE_TEXT:
            # DM fast-path conditions [PA]
            if features.is_dm and features.char_count < 100 and not features.has_mentions:
                optimizations.update({
                    "skip_context_heavy": True,
                    "skip_rag_search": True,
                    "skip_modality_detection": True,
                    "use_simple_template": True
                })
                return RouteDecision.FAST_PATH_TEXT, optimizations
        
        # Complex text or default
        if features.is_dm and not features.is_reply:
            # Light DM processing
            optimizations.update({
                "skip_context_heavy": True,
                "use_simple_template": features.char_count < 150
            })
        
        return RouteDecision.STANDARD_PIPELINE, optimizations
    
    def _calculate_confidence(
        self,
        features: MessageFeatures,
        complexity: MessageComplexity,
        decision: RouteDecision
    ) -> float:
        """Calculate confidence score for routing decision [PA]."""
        confidence = 1.0
        
        # Lower confidence for edge cases
        if features.char_count > 300:  # Long messages harder to classify
            confidence -= 0.1
        
        if features.language_complexity > 0.8:  # Very complex language
            confidence -= 0.2
        
        if complexity == MessageComplexity.UNKNOWN:
            confidence = 0.5
        
        # High confidence for clear cases
        if decision == RouteDecision.FAST_PATH_TEXT and complexity == MessageComplexity.SIMPLE_TEXT:
            confidence = 0.95
        
        if features.has_command_prefix and decision == RouteDecision.COMMAND_HANDLER:
            confidence = 0.95
        
        return max(0.1, confidence)
    
    def _generate_reasoning(
        self,
        features: MessageFeatures,
        complexity: MessageComplexity,
        decision: RouteDecision
    ) -> str:
        """Generate human-readable reasoning for decision [CA]."""
        reasons = []
        
        if decision == RouteDecision.FAST_PATH_TEXT:
            reasons.append(f"Simple DM text ({features.char_count} chars)")
            if features.char_count < 50:
                reasons.append("very short")
            if not features.has_questions:
                reasons.append("no questions")
        
        elif decision == RouteDecision.STANDARD_PIPELINE and complexity == MessageComplexity.MULTIMODAL:
            if features.has_attachments:
                reasons.append("has attachments")
            if features.has_urls:
                reasons.append("contains URLs")
            if features.has_embeds:
                reasons.append("has embeds")
        
        elif decision == RouteDecision.COMMAND_HANDLER:
            reasons.append("command prefix detected")
        
        else:
            reasons.append(f"{complexity.value} message")
            if features.word_count > 30:
                reasons.append("lengthy content")
            if features.language_complexity > 0.6:
                reasons.append("complex language")
        
        return ", ".join(reasons) if reasons else "default routing"
    
    def _log_routing_decision(self, message: Message, analysis: RouteAnalysis):
        """Log routing decision with Rich formatting [CA]."""
        # Choose icon based on decision
        decision_icons = {
            RouteDecision.FAST_PATH_TEXT: "🚀",
            RouteDecision.STANDARD_PIPELINE: "🔄",
            RouteDecision.COMMAND_HANDLER: "⚡",
            RouteDecision.REJECT: "🚫"
        }
        
        icon = decision_icons.get(analysis.decision, "❓")
        decision_name = analysis.decision.value.upper()
        
        # Create log message
        msg = f"{icon} Routing: {decision_name} ({analysis.decision_time_ms}ms, {analysis.confidence:.1%} confidence)"
        
        # Detailed info for JSONL
        log_detail = {
            "msg_id": message.id,
            "user_id": message.author.id,
            "guild_id": getattr(message.guild, 'id', None),
            "is_dm": isinstance(message.channel, DMChannel),
            "complexity": analysis.complexity.value,
            "decision": analysis.decision.value,
            "decision_time_ms": analysis.decision_time_ms,
            "confidence": analysis.confidence,
            "reasoning": analysis.reasoning,
            "optimizations": {
                "skip_context_heavy": analysis.skip_context_heavy,
                "skip_rag_search": analysis.skip_rag_search,
                "use_simple_template": analysis.use_simple_template
            },
            **analysis.metadata
        }
        
        # Log with appropriate level
        if analysis.decision_time_ms > self.decision_budget_ms:
            logger.warning(msg, extra={"detail": log_detail})
        else:
            logger.info(msg, extra={"detail": log_detail})
    
    def _update_stats(self, analysis: RouteAnalysis):
        """Update router performance statistics [PA]."""
        self.stats["total_decisions"] += 1
        
        if analysis.decision == RouteDecision.FAST_PATH_TEXT:
            self.stats["fast_path_decisions"] += 1
        
        # Update complexity distribution
        self.stats["complexity_distribution"][analysis.complexity.value] += 1
        
        # Update average decision time
        old_avg = self.stats["avg_decision_time_ms"]
        self.stats["avg_decision_time_ms"] = (
            (old_avg * (self.stats["total_decisions"] - 1) + analysis.decision_time_ms) /
            self.stats["total_decisions"]
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get router performance statistics."""
        total_decisions = self.stats["total_decisions"]
        fast_path_rate = (self.stats["fast_path_decisions"] / total_decisions 
                         if total_decisions > 0 else 0)
        
        return {
            "total_decisions": total_decisions,
            "fast_path_rate": fast_path_rate,
            "budget_exceeded_rate": self.stats["budget_exceeded"] / total_decisions if total_decisions > 0 else 0,
            "avg_decision_time_ms": self.stats["avg_decision_time_ms"],
            "complexity_distribution": self.stats["complexity_distribution"].copy(),
            "decision_budget_ms": self.decision_budget_ms
        }

# Global fast-path router instance [PA]
_router_instance: Optional[FastPathRouter] = None

def get_fast_path_router() -> FastPathRouter:
    """Get global fast-path router instance."""
    global _router_instance
    
    if _router_instance is None:
        _router_instance = FastPathRouter()
        logger.info("🚀 Global FastPathRouter created")
    
    return _router_instance
