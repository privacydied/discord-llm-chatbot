"""
Vision Intent Router - Smart routing between Vision and OpenRouter

Analyzes Discord messages to determine whether they should be processed by:
- Vision Generation System (image/video generation tasks)
- OpenRouter (traditional text/VL processing)

Uses deterministic rules, natural language patterns, and ML-based intent scoring.
"""

from __future__ import annotations
import re
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from types import SimpleNamespace
from dataclasses import dataclass
from pathlib import Path

import discord
from bot.util.logging import get_logger
from bot.config import load_config
from .types import (
    VisionTask, VisionProvider, IntentScore, RoutingDecision,
    VisionRequest, VisionError, VisionErrorType, IntentDecision, IntentResult
)

logger = get_logger(__name__)


@dataclass
class MessageContext:
    """Context information for intent classification"""
    content: str
    has_attachments: bool
    attachment_types: List[str]
    is_slash_command: bool
    command_name: Optional[str] = None
    user_id: str = ""
    guild_id: Optional[str] = None
    channel_id: str = ""
    
    # Parsed command context
    force_vision: bool = False
    force_openrouter: bool = False


class VisionIntentRouter:
    """
    Smart router for Vision vs OpenRouter decision making
    
    Implements multi-layered routing strategy:
    1. Deterministic rules (slash commands, force prefixes)
    2. Attachment + verb analysis 
    3. Trigger phrase detection
    4. Parameter extraction and normalization
    5. Confidence scoring and threshold-based routing [CA]
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, policy: Optional[Dict[str, Any]] = None):
        self.config = config or load_config()
        self.logger = get_logger("vision.intent_router")
        
        # Load vision policy if not provided
        if policy is None:
            import json
            policy_path = Path(self.config["VISION_POLICY_PATH"])
            if policy_path.exists():
                with open(policy_path) as f:
                    policy = json.load(f)
            else:
                policy = {"intent_patterns": {}}
        
        self.policy = policy
        self.intent_patterns = policy.get("intent_patterns", {})
        
        # Confidence thresholds from policy
        thresholds = policy.get("routing", {}).get("confidence_thresholds", {})
        self.high_confidence_threshold = thresholds.get("high_confidence", 0.8)
        self.medium_confidence_threshold = thresholds.get("medium_confidence", 0.6) 
        self.low_confidence_threshold = thresholds.get("low_confidence", 0.4)
        self.fallback_threshold = thresholds.get("fallback_to_openrouter", 0.3)
        
        self.logger.info("Vision Intent Router initialized")
    
    async def route_message(
        self, 
        message: discord.Message,
        parsed_command: Optional[Any] = None
    ) -> RoutingDecision:
        """
        Main routing decision for Discord messages
        
        Args:
            message: Discord message to analyze
            parsed_command: Optional pre-parsed command context
            
        Returns:
            RoutingDecision with routing choice and extracted parameters
        """
        # Extract message context
        context = self._extract_message_context(message, parsed_command)
        
        self.logger.debug(
            "Analyzing message for routing",
            user_id=context.user_id,
            content_length=len(context.content),
            has_attachments=context.has_attachments,
            is_slash_command=context.is_slash_command
        )
        
        # Check if Vision is globally disabled
        if not self.config.get("VISION_ENABLED", False):
            return RoutingDecision(
                route_to_vision=False,
                confidence=1.0,
                reasoning="Vision generation is disabled globally",
                fallback_reason="VISION_DISABLED"
            )
        
        # Apply deterministic routing rules (highest priority)
        deterministic_decision = self._apply_deterministic_rules(context)
        if deterministic_decision:
            return deterministic_decision
        
        # Analyze intent using pattern matching and ML scoring
        intent_scores = await self._analyze_intent(context)
        
        # Select best intent and make routing decision
        return self._make_routing_decision(context, intent_scores)
    
    async def determine_intent(
        self,
        user_message: str,
        context: str = "",
        user_id: str = "",
        guild_id: Optional[str] = None,
    ) -> IntentResult:
        """Determine whether to use Vision and extract parameters.
        
        Returns an IntentResult compatible with router expectations:
        - decision.use_vision (bool)
        - extracted_params.prompt, .task, optional width/height/batch_size/negative_prompt/etc.
        - confidence (float)
        """
        content = (user_message or "").strip()
        
        # Detect force prefixes similar to _extract_message_context
        force_vision = content.startswith("!!image") or content.startswith("!!video") or content.startswith("!!vision")
        force_openrouter = content.startswith("!!text") or content.startswith("!!chat")
        if force_vision or force_openrouter:
            content = re.sub(r'^!!(image|video|vision|text|chat)\s*', '', content).strip()
        
        # Build lightweight MessageContext
        msg_ctx = MessageContext(
            content=content,
            has_attachments=False,
            attachment_types=[],
            is_slash_command=False,
            command_name=None,
            user_id=str(user_id) if user_id is not None else "",
            guild_id=str(guild_id) if guild_id is not None else None,
            channel_id="",
            force_vision=force_vision,
            force_openrouter=force_openrouter,
        )
        
        # Vision globally disabled
        if not self.config.get("VISION_ENABLED", False):
            decision = IntentDecision(
                use_vision=False,
                confidence=1.0,
                reasoning="Vision generation is disabled globally",
                fallback_reason="VISION_DISABLED",
            )
            extracted = SimpleNamespace(prompt=content, task=None)
            return IntentResult(decision=decision, extracted_params=extracted, confidence=decision.confidence)
        
        # Deterministic rules first
        det = self._apply_deterministic_rules(msg_ctx)
        if det:
            decision = IntentDecision(
                use_vision=det.route_to_vision,
                task=det.task,
                confidence=det.confidence,
                provider=det.provider,
                model=det.model,
                estimated_cost=det.estimated_cost,
                reasoning=det.reasoning,
                fallback_reason=det.fallback_reason,
            )
            extracted_map: Dict[str, Any] = {"prompt": content}
            if det.task is not None:
                extracted_map["task"] = det.task
            extracted = SimpleNamespace(**extracted_map)
            return IntentResult(decision=decision, extracted_params=extracted, confidence=decision.confidence)
        
        # Pattern analysis
        intent_scores = await self._analyze_intent(msg_ctx)
        decision_rd = self._make_routing_decision(msg_ctx, intent_scores)
        
        # Best intent parameters (if any)
        best = intent_scores[0] if intent_scores else None
        params: Dict[str, Any] = dict(best.extracted_parameters) if best and best.extracted_parameters else {}
        
        # Normalize common parameters to what router expects
        # Size → width/height
        size = params.get("size")
        if isinstance(size, dict):
            if "width" in size:
                try:
                    params["width"] = int(size["width"])  # type: ignore[arg-type]
                except Exception:
                    pass
            if "height" in size:
                try:
                    params["height"] = int(size["height"])  # type: ignore[arg-type]
                except Exception:
                    pass
        elif isinstance(size, str):
            m = re.search(r"(\d+)\s*[x×]\s*(\d+)", size)
            if m:
                try:
                    params["width"] = int(m.group(1))
                    params["height"] = int(m.group(2))
                except Exception:
                    pass
        
        # batch → batch_size
        if "batch" in params and "batch_size" not in params:
            try:
                params["batch_size"] = int(params["batch"])  # type: ignore[arg-type]
            except Exception:
                pass
        
        # negative → negative_prompt
        if "negative" in params and "negative_prompt" not in params:
            params["negative_prompt"] = str(params["negative"])  # type: ignore[arg-type]
        
        # provider → preferred_provider (enum if valid)
        provider_val = params.get("provider")
        if provider_val is not None and "preferred_provider" not in params:
            try:
                params["preferred_provider"] = VisionProvider(provider_val)
            except Exception:
                # Leave unset if not a valid provider
                pass
        
        # Always include prompt and inferred task
        params["prompt"] = content
        if decision_rd.task is not None:
            params["task"] = decision_rd.task
        elif best and best.task is not None:
            params["task"] = best.task
        
        decision = IntentDecision(
            use_vision=decision_rd.route_to_vision,
            task=decision_rd.task,
            confidence=decision_rd.confidence,
            provider=decision_rd.provider,
            model=decision_rd.model,
            estimated_cost=decision_rd.estimated_cost,
            reasoning=decision_rd.reasoning,
            fallback_reason=decision_rd.fallback_reason,
        )
        extracted = SimpleNamespace(**params)
        return IntentResult(decision=decision, extracted_params=extracted, confidence=decision.confidence)
    
    def _extract_message_context(
        self, 
        message: discord.Message, 
        parsed_command: Optional[Any]
    ) -> MessageContext:
        """Extract context information from Discord message [IV]"""
        content = message.content.strip()
        
        # Analyze attachments
        attachment_types = []
        for attachment in message.attachments:
            if attachment.filename:
                ext = Path(attachment.filename).suffix.lower().lstrip('.')
                if ext in ['jpg', 'jpeg', 'png', 'webp', 'gif']:
                    attachment_types.append('image')
                elif ext in ['mp4', 'mov', 'avi', 'webm']:
                    attachment_types.append('video')
                else:
                    attachment_types.append('file')
        
        # Check for force prefixes
        force_vision = content.startswith("!!image") or content.startswith("!!video") or content.startswith("!!vision")
        force_openrouter = content.startswith("!!text") or content.startswith("!!chat")
        
        # Remove force prefixes from content for analysis
        if force_vision or force_openrouter:
            content = re.sub(r'^!!(image|video|vision|text|chat)\s*', '', content).strip()
        
        return MessageContext(
            content=content,
            has_attachments=bool(message.attachments),
            attachment_types=attachment_types,
            is_slash_command=parsed_command is not None,
            command_name=getattr(parsed_command, 'command_name', None) if parsed_command else None,
            user_id=str(message.author.id),
            guild_id=str(message.guild.id) if message.guild else None,
            channel_id=str(message.channel.id),
            force_vision=force_vision,
            force_openrouter=force_openrouter
        )
    
    def _apply_deterministic_rules(self, context: MessageContext) -> Optional[RoutingDecision]:
        """Apply high-priority deterministic routing rules [CA]"""
        
        # Rule 1: Force prefixes (highest priority)
        if context.force_vision:
            return RoutingDecision(
                route_to_vision=True,
                confidence=1.0,
                reasoning="Force vision prefix detected (!!vision, !!image, !!video)",
                task=self._infer_task_from_content(context)
            )
        
        if context.force_openrouter:
            return RoutingDecision(
                route_to_vision=False,
                confidence=1.0,
                reasoning="Force OpenRouter prefix detected (!!text, !!chat)",
                fallback_reason="USER_OVERRIDE"
            )
        
        # Rule 2: Vision slash commands
        if context.is_slash_command and context.command_name in ['image', 'imgedit', 'video', 'vidref']:
            task_mapping = {
                'image': VisionTask.TEXT_TO_IMAGE,
                'imgedit': VisionTask.IMAGE_TO_IMAGE,
                'video': VisionTask.TEXT_TO_VIDEO,
                'vidref': VisionTask.IMAGE_TO_VIDEO
            }
            
            return RoutingDecision(
                route_to_vision=True,
                confidence=1.0,
                task=task_mapping[context.command_name],
                reasoning=f"Vision slash command: /{context.command_name}"
            )
        
        # Rule 3: Non-vision slash commands
        if context.is_slash_command:
            return RoutingDecision(
                route_to_vision=False,
                confidence=1.0,
                reasoning=f"Non-vision slash command: /{context.command_name}",
                fallback_reason="SLASH_COMMAND"
            )
        
        # Rule 4: Attachment + edit verbs → image editing
        if context.has_attachments and 'image' in context.attachment_types:
            edit_verbs = self.intent_patterns.get("image_editing", {}).get("trigger_phrases", [])
            content_lower = context.content.lower()
            
            for verb in edit_verbs:
                if verb.lower() in content_lower:
                    return RoutingDecision(
                        route_to_vision=True,
                        confidence=0.9,
                        task=VisionTask.IMAGE_TO_IMAGE,
                        reasoning=f"Image attachment + edit verb '{verb}' detected"
                    )
        
        # Rule 5: Image attachment + motion verbs → image-to-video
        if context.has_attachments and 'image' in context.attachment_types:
            i2v_patterns = self.intent_patterns.get("image_to_video", {}).get("trigger_phrases", [])
            content_lower = context.content.lower()
            
            for phrase in i2v_patterns:
                if phrase.lower() in content_lower:
                    return RoutingDecision(
                        route_to_vision=True,
                        confidence=0.9,
                        task=VisionTask.IMAGE_TO_VIDEO,
                        reasoning=f"Image attachment + motion phrase '{phrase}' detected"
                    )
        
        # No deterministic rule matched
        return None
    
    async def _analyze_intent(self, context: MessageContext) -> List[IntentScore]:
        """Analyze intent using pattern matching and scoring [CA]"""
        intent_scores = []
        
        # Analyze each potential task
        for task_name, task_config in self.intent_patterns.items():
            task = self._task_name_to_enum(task_name)
            if not task:
                continue
                
            score = await self._score_task_intent(context, task, task_config)
            if score.confidence > 0.0:
                intent_scores.append(score)
        
        # Sort by confidence (highest first)
        intent_scores.sort(key=lambda s: s.confidence, reverse=True)
        
        return intent_scores
    
    async def _score_task_intent(
        self, 
        context: MessageContext, 
        task: VisionTask, 
        task_config: Dict[str, Any]
    ) -> IntentScore:
        """Score intent for specific vision task [PA]"""
        confidence = 0.0
        extracted_params = {}
        reasoning_parts = []
        
        content_lower = context.content.lower()
        
        # Check trigger phrases (primary signal)
        trigger_phrases = task_config.get("trigger_phrases", [])
        phrase_matches = 0
        best_phrase = None
        
        for phrase in trigger_phrases:
            if phrase.lower() in content_lower:
                phrase_matches += 1
                if not best_phrase or len(phrase) > len(best_phrase):
                    best_phrase = phrase
        
        if phrase_matches > 0:
            # Base confidence from phrase matching
            confidence = min(0.8, 0.4 + (phrase_matches * 0.2))
            reasoning_parts.append(f"trigger phrase '{best_phrase}'")
        
        # Check attachment requirements
        if task_config.get("attachment_required", False):
            if context.has_attachments and 'image' in context.attachment_types:
                confidence += 0.2
                reasoning_parts.append("required image attachment present")
            else:
                confidence = max(0.0, confidence - 0.5)  # Strong penalty
                reasoning_parts.append("required attachment missing")
        
        # Extract parameters and boost confidence
        parameter_patterns = task_config.get("parameter_patterns", {})
        
        for param_name, param_config in parameter_patterns.items():
            param_value = self._extract_parameter(context.content, param_config)
            if param_value is not None:
                extracted_params[param_name] = param_value
                confidence += 0.1  # Small boost per parameter
                reasoning_parts.append(f"extracted {param_name}")
        
        # Apply task-specific logic
        if task == VisionTask.TEXT_TO_VIDEO:
            # Look for video-specific indicators
            video_keywords = ["video", "clip", "movie", "animate", "motion", "cinematic"]
            for keyword in video_keywords:
                if keyword in content_lower:
                    confidence += 0.15
                    reasoning_parts.append(f"video keyword '{keyword}'")
                    break
        
        # Penalty for very short or generic prompts
        if len(context.content.strip()) < 10:
            confidence *= 0.7
            reasoning_parts.append("short prompt penalty")
        
        # Cap confidence and create score
        confidence = min(1.0, max(0.0, confidence))
        reasoning = " + ".join(reasoning_parts) if reasoning_parts else "no strong indicators"
        
        return IntentScore(
            task=task,
            confidence=confidence,
            extracted_parameters=extracted_params,
            reasoning=reasoning
        )
    
    def _extract_parameter(self, content: str, param_config: Any) -> Optional[Any]:
        """Extract parameter value using configured pattern [IV]"""
        if isinstance(param_config, dict):
            if "pattern" in param_config:
                # Regex pattern extraction
                pattern = param_config["pattern"]
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    return match.group(1) if match.groups() else match.group(0)
            
            elif "keywords" in param_config:
                # Keyword-based extraction
                content_lower = content.lower()
                for keyword in param_config["keywords"]:
                    if keyword.lower() in content_lower:
                        # Extract text after keyword
                        idx = content_lower.find(keyword.lower())
                        remaining = content[idx + len(keyword):].strip()
                        # Simple heuristic: take next 50 chars or until punctuation
                        match = re.search(r'^[^.!?;]*', remaining)
                        if match:
                            return match.group(0).strip()
            
            else:
                # Dictionary mapping (e.g., size keywords)
                content_lower = content.lower()
                for key, value in param_config.items():
                    if key.lower() in content_lower:
                        return value
        
        return None
    
    def _make_routing_decision(self, context: MessageContext, intent_scores: List[IntentScore]) -> RoutingDecision:
        """Make final routing decision based on intent analysis [CA]"""
        
        # No vision intents detected
        if not intent_scores:
            return RoutingDecision(
                route_to_vision=False,
                confidence=0.8,
                reasoning="No vision generation patterns detected",
                fallback_reason="NO_VISION_INTENT"
            )
        
        best_intent = intent_scores[0]
        
        # High confidence → route to Vision
        if best_intent.confidence >= self.high_confidence_threshold:
            return RoutingDecision(
                route_to_vision=True,
                confidence=best_intent.confidence,
                task=best_intent.task,
                reasoning=f"High confidence vision intent: {best_intent.reasoning}",
                estimated_cost=self._estimate_task_cost(best_intent.task, best_intent.extracted_parameters)
            )
        
        # Medium confidence → route to Vision with note
        elif best_intent.confidence >= self.medium_confidence_threshold:
            return RoutingDecision(
                route_to_vision=True,
                confidence=best_intent.confidence,
                task=best_intent.task,
                reasoning=f"Medium confidence vision intent: {best_intent.reasoning}",
                estimated_cost=self._estimate_task_cost(best_intent.task, best_intent.extracted_parameters)
            )
        
        # Low confidence → check for ambiguity, potentially ask for clarification
        elif best_intent.confidence >= self.low_confidence_threshold:
            # Check if there are competing high-confidence non-vision signals
            if self._has_strong_text_signals(context):
                return RoutingDecision(
                    route_to_vision=False,
                    confidence=0.7,
                    reasoning=f"Ambiguous intent: vision patterns present ({best_intent.reasoning}) but strong text signals detected",
                    fallback_reason="AMBIGUOUS_INTENT"
                )
            else:
                return RoutingDecision(
                    route_to_vision=True,
                    confidence=best_intent.confidence,
                    task=best_intent.task,
                    reasoning=f"Low confidence vision intent: {best_intent.reasoning}",
                    estimated_cost=self._estimate_task_cost(best_intent.task, best_intent.extracted_parameters)
                )
        
        # Very low confidence → fallback to OpenRouter
        else:
            return RoutingDecision(
                route_to_vision=False,
                confidence=0.8,
                reasoning=f"Vision patterns too weak ({best_intent.confidence:.2f}): {best_intent.reasoning}",
                fallback_reason="LOW_CONFIDENCE"
            )
    
    def _has_strong_text_signals(self, context: MessageContext) -> bool:
        """Check for strong non-vision conversational signals [CMV]"""
        content_lower = context.content.lower()
        
        # Question patterns
        question_words = ["what", "how", "why", "when", "where", "who", "explain", "tell me"]
        if any(word in content_lower for word in question_words):
            return True
        
        # Conversational patterns
        conversational = ["i think", "in my opinion", "what do you think", "do you know", "can you help"]
        if any(phrase in content_lower for phrase in conversational):
            return True
        
        # Technical/factual queries
        technical = ["calculate", "compute", "analyze", "define", "compare", "list"]
        if any(word in content_lower for word in technical):
            return True
        
        return False
    
    def _infer_task_from_content(self, context: MessageContext) -> Optional[VisionTask]:
        """Infer likely task from content when force prefix is used [CMV]"""
        content_lower = context.content.lower()
        
        # Video indicators
        video_words = ["video", "clip", "movie", "animate", "motion", "fps", "seconds"]
        if any(word in content_lower for word in video_words):
            if context.has_attachments and 'image' in context.attachment_types:
                return VisionTask.IMAGE_TO_VIDEO
            else:
                return VisionTask.TEXT_TO_VIDEO
        
        # Image editing indicators  
        if context.has_attachments and 'image' in context.attachment_types:
            edit_words = ["edit", "modify", "change", "fix", "remove", "add"]
            if any(word in content_lower for word in edit_words):
                return VisionTask.IMAGE_TO_IMAGE
        
        # Default to text-to-image
        return VisionTask.TEXT_TO_IMAGE
    
    def _task_name_to_enum(self, task_name: str) -> Optional[VisionTask]:
        """Convert task name from policy to VisionTask enum [CMV]"""
        mapping = {
            "image_generation": VisionTask.TEXT_TO_IMAGE,
            "text_to_image": VisionTask.TEXT_TO_IMAGE,
            "image_editing": VisionTask.IMAGE_TO_IMAGE,
            "image_to_image": VisionTask.IMAGE_TO_IMAGE,
            "video_generation": VisionTask.TEXT_TO_VIDEO,
            "text_to_video": VisionTask.TEXT_TO_VIDEO,
            "image_to_video": VisionTask.IMAGE_TO_VIDEO
        }
        return mapping.get(task_name)
    
    def _estimate_task_cost(self, task: VisionTask, parameters: Dict[str, Any]) -> float:
        """Rough cost estimation for routing decision [CMV]"""
        # Very rough estimates for display purposes
        base_costs = {
            VisionTask.TEXT_TO_IMAGE: 0.04,
            VisionTask.IMAGE_TO_IMAGE: 0.06,
            VisionTask.TEXT_TO_VIDEO: 1.50,
            VisionTask.IMAGE_TO_VIDEO: 2.00
        }
        
        base_cost = base_costs.get(task, 0.10)
        
        # Scale by duration for video
        if task in [VisionTask.TEXT_TO_VIDEO, VisionTask.IMAGE_TO_VIDEO]:
            duration = parameters.get("duration", {})
            if isinstance(duration, dict):
                duration_seconds = duration.get("seconds", 3)
            elif isinstance(duration, (int, float)):
                duration_seconds = duration
            else:
                duration_seconds = 3
            
            base_cost *= duration_seconds
        
        # Scale by batch size
        batch_info = parameters.get("batch", {})
        if isinstance(batch_info, dict):
            batch_size = batch_info.get("count", 1)
        elif isinstance(batch_info, (int, float)):
            batch_size = batch_info
        else:
            batch_size = 1
        
        base_cost *= batch_size
        
        return round(base_cost, 2)
    
    def create_vision_request_from_context(
        self, 
        context: MessageContext, 
        decision: RoutingDecision,
        attachments: List[discord.Attachment] = None
    ) -> VisionRequest:
        """Create VisionRequest from routing decision and message context [CA]"""
        if not decision.route_to_vision or not decision.task:
            raise VisionError(
                error_type=VisionErrorType.SYSTEM_ERROR,
                message="Cannot create VisionRequest for non-vision routing decision",
                user_message="Internal routing error"
            )
        
        # Base request
        request = VisionRequest(
            task=decision.task,
            prompt=context.content,
            user_id=context.user_id,
            guild_id=context.guild_id,
            channel_id=context.channel_id,
            estimated_cost=decision.estimated_cost or 0.0
        )
        
        # Apply extracted parameters
        params = decision.task and decision.estimated_cost  # Use decision params if available
        if hasattr(decision, 'extracted_parameters'):
            self._apply_extracted_parameters(request, decision.extracted_parameters)
        
        # Handle attachments for editing/i2v tasks
        if attachments and decision.task in [VisionTask.IMAGE_TO_IMAGE, VisionTask.IMAGE_TO_VIDEO]:
            # Note: Actual file download and path assignment would happen in orchestrator
            request.input_image = Path("placeholder")  # Will be replaced with actual path
        
        return request
    
    def _apply_extracted_parameters(self, request: VisionRequest, parameters: Dict[str, Any]) -> None:
        """Apply extracted parameters to VisionRequest [CMV]"""
        
        # Size parameters
        if "size" in parameters:
            size = parameters["size"]
            if isinstance(size, dict):
                request.width = size.get("width", request.width)
                request.height = size.get("height", request.height)
        
        # Negative prompt
        if "negative" in parameters:
            request.negative_prompt = str(parameters["negative"])
        
        # Seed
        if "seed" in parameters:
            try:
                request.seed = int(parameters["seed"])
            except (ValueError, TypeError):
                pass
        
        # Batch size
        if "batch" in parameters:
            try:
                batch_size = int(parameters["batch"])
                request.batch_size = min(batch_size, 4)  # Cap at policy limit
            except (ValueError, TypeError):
                pass
        
        # Duration for video tasks
        if "duration" in parameters and request.task in [VisionTask.TEXT_TO_VIDEO, VisionTask.IMAGE_TO_VIDEO]:
            try:
                duration = float(parameters["duration"])
                request.duration_seconds = min(duration, 10)  # Cap at policy limit
            except (ValueError, TypeError):
                pass
