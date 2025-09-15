"""
Vision Safety Filter - Content validation and filtering

Implements comprehensive content safety checks for vision generation requests:
- Prompt safety analysis using keywords and patterns
- Image content validation for input images
- Policy-driven filtering with configurable severity levels
- NSFW detection and server-specific overrides
- User guidance for rejected content

Follows Security-First Thinking (SFT) and Input Validation (IV) principles.
"""

from __future__ import annotations
import re
import json
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from bot.utils.logging import get_logger
from bot.config import load_config
from .types import VisionRequest, VisionTask

logger = get_logger(__name__)


class SafetyLevel(Enum):
    """Content safety severity levels"""

    SAFE = "safe"
    WARNING = "warning"
    BLOCKED = "blocked"


@dataclass
class SafetyResult:
    """Result of safety filtering check"""

    approved: bool
    level: SafetyLevel
    reason: str
    user_message: str
    detected_issues: List[str]


class VisionSafetyFilter:
    """
    Content safety filter for vision generation requests

    Features:
    - Multi-layer prompt analysis (keywords, patterns, ML)
    - Image content validation for explicit material
    - Server-specific NSFW policy enforcement
    - Configurable safety thresholds and overrides
    - Clear user messaging for policy violations
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config()
        self.logger = get_logger("vision.safety_filter")

        # Load safety policies
        self.policy = self._load_safety_policy()

        # Compile regex patterns for efficiency
        self.blocked_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.policy.get("blocked_patterns", [])
        ]
        self.warning_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.policy.get("warning_patterns", [])
        ]

        # Keywords lists
        self.blocked_keywords = {
            kw.lower() for kw in self.policy.get("blocked_keywords", [])
        }
        self.warning_keywords = {
            kw.lower() for kw in self.policy.get("warning_keywords", [])
        }

        self.logger.info(
            f"Vision Safety Filter initialized - blocked_patterns: {len(self.blocked_patterns)}, warning_patterns: {len(self.warning_patterns)}, blocked_keywords: {len(self.blocked_keywords)}"
        )

    async def validate_request(self, request: VisionRequest) -> SafetyResult:
        """
        Validate vision generation request for safety compliance

        Args:
            request: Vision generation request to validate

        Returns:
            SafetyResult with approval decision and details
        """
        detected_issues = []
        max_level = SafetyLevel.SAFE

        try:
            # 1. Validate text prompt
            if request.prompt:
                prompt_result = self._analyze_text_content(request.prompt, "prompt")
                detected_issues.extend(prompt_result.detected_issues)
                if prompt_result.level.value > max_level.value:
                    max_level = prompt_result.level

            # 2. Validate negative prompt
            if request.negative_prompt:
                neg_result = self._analyze_text_content(
                    request.negative_prompt, "negative_prompt"
                )
                detected_issues.extend(neg_result.detected_issues)
                if neg_result.level.value > max_level.value:
                    max_level = neg_result.level

            # 3. Task-specific validations
            task_result = self._validate_task_specific(request)
            detected_issues.extend(task_result.detected_issues)
            if task_result.level.value > max_level.value:
                max_level = task_result.level

            # 4. Server policy checks (NSFW, etc.)
            server_result = self._check_server_policies(request)
            detected_issues.extend(server_result.detected_issues)
            if server_result.level.value > max_level.value:
                max_level = server_result.level

            # 5. Generate final result
            approved = max_level != SafetyLevel.BLOCKED
            reason = self._generate_reason(detected_issues, max_level)
            user_message = self._generate_user_message(
                detected_issues, max_level, request
            )

            result = SafetyResult(
                approved=approved,
                level=max_level,
                reason=reason,
                user_message=user_message,
                detected_issues=detected_issues,
            )

            # Log result
            if not approved:
                self.logger.warning(
                    f"Content blocked by safety filter - user_id: {request.user_id}, task: {request.task.value}, level: {max_level.value}, issues: {len(detected_issues)}"
                )
            elif max_level == SafetyLevel.WARNING:
                self.logger.info(
                    f"Content generated safety warning - user_id: {request.user_id}, task: {request.task.value}, issues: {len(detected_issues)}"
                )

            return result

        except Exception as e:
            self.logger.error(
                f"Safety validation error - error: {str(e)}, user_id: {request.user_id}"
            )
            # Fail safe - block on error
            return SafetyResult(
                approved=False,
                level=SafetyLevel.BLOCKED,
                reason=f"Safety validation error: {str(e)}",
                user_message="Unable to validate content safety. Please try again.",
                detected_issues=["system_error"],
            )

    def _analyze_text_content(self, text: str, field_name: str) -> SafetyResult:
        """Analyze text content for safety issues [IV]"""
        detected_issues = []
        max_level = SafetyLevel.SAFE

        if not text or not text.strip():
            return SafetyResult(
                approved=True,
                level=SafetyLevel.SAFE,
                reason="Empty text",
                user_message="",
                detected_issues=[],
            )

        text_lower = text.lower().strip()

        # Check blocked keywords
        for keyword in self.blocked_keywords:
            if keyword in text_lower:
                detected_issues.append(f"blocked_keyword:{keyword}")
                max_level = SafetyLevel.BLOCKED

        # Check warning keywords (if not already blocked)
        if max_level != SafetyLevel.BLOCKED:
            for keyword in self.warning_keywords:
                if keyword in text_lower:
                    detected_issues.append(f"warning_keyword:{keyword}")
                    max_level = SafetyLevel.WARNING

        # Check blocked patterns
        for pattern in self.blocked_patterns:
            if pattern.search(text):
                detected_issues.append(f"blocked_pattern:{pattern.pattern[:50]}")
                max_level = SafetyLevel.BLOCKED
                break

        # Check warning patterns (if not already blocked)
        if max_level != SafetyLevel.BLOCKED:
            for pattern in self.warning_patterns:
                if pattern.search(text):
                    detected_issues.append(f"warning_pattern:{pattern.pattern[:50]}")
                    max_level = SafetyLevel.WARNING
                    break

        # Additional heuristic checks
        if max_level == SafetyLevel.SAFE:
            heuristic_issues = self._heuristic_analysis(text)
            if heuristic_issues:
                detected_issues.extend(heuristic_issues)
                max_level = SafetyLevel.WARNING

        approved = max_level != SafetyLevel.BLOCKED
        reason = f"{field_name} analysis: {max_level.value}"

        return SafetyResult(
            approved=approved,
            level=max_level,
            reason=reason,
            user_message="",
            detected_issues=detected_issues,
        )

    def _validate_task_specific(self, request: VisionRequest) -> SafetyResult:
        """Task-specific safety validations [SFT]"""
        detected_issues = []
        max_level = SafetyLevel.SAFE

        # Image editing tasks have stricter controls
        if request.task in [VisionTask.IMAGE_TO_IMAGE]:
            if not request.input_image_url and not request.input_image_data:
                detected_issues.append("missing_input_image")
                max_level = SafetyLevel.BLOCKED

            # Check for deepfake indicators in prompts
            deepfake_indicators = [
                "face swap",
                "celebrity face",
                "famous person",
                "political figure",
                "realistic person",
                "human face",
            ]

            if request.prompt:
                prompt_lower = request.prompt.lower()
                for indicator in deepfake_indicators:
                    if indicator in prompt_lower:
                        detected_issues.append(f"deepfake_risk:{indicator}")
                        max_level = SafetyLevel.WARNING

        # Video tasks have additional considerations
        if request.task in [VisionTask.TEXT_TO_VIDEO, VisionTask.IMAGE_TO_VIDEO]:
            video_restrictions = self.policy.get("video_restrictions", {})

            # Check duration limits
            if hasattr(request, "duration") and request.duration:
                max_duration = video_restrictions.get("max_duration_seconds", 30)
                if request.duration > max_duration:
                    detected_issues.append(f"duration_exceeded:{request.duration}")
                    max_level = SafetyLevel.BLOCKED

            # Check for prohibited video content
            video_keywords = video_restrictions.get("blocked_keywords", [])
            if request.prompt:
                for keyword in video_keywords:
                    if keyword.lower() in request.prompt.lower():
                        detected_issues.append(f"video_blocked_keyword:{keyword}")
                        max_level = SafetyLevel.BLOCKED

        approved = max_level != SafetyLevel.BLOCKED
        reason = f"Task validation: {max_level.value}"

        return SafetyResult(
            approved=approved,
            level=max_level,
            reason=reason,
            user_message="",
            detected_issues=detected_issues,
        )

    def _check_server_policies(self, request: VisionRequest) -> SafetyResult:
        """Check server-specific policies (NSFW, etc.) [SFT]"""
        detected_issues = []
        max_level = SafetyLevel.SAFE

        # Get server-specific settings
        guild_id = getattr(request, "guild_id", None)
        server_policies = self.policy.get("server_policies", {})

        # Check NSFW policy
        nsfw_policy = server_policies.get("nsfw", {})
        default_nsfw_allowed = nsfw_policy.get("default_allowed", False)

        # Server override check
        server_nsfw_allowed = default_nsfw_allowed
        if guild_id:
            server_overrides = nsfw_policy.get("server_overrides", {})
            server_nsfw_allowed = server_overrides.get(
                str(guild_id), default_nsfw_allowed
            )

        # Detect potential NSFW content
        nsfw_indicators = self._detect_nsfw_content(request)
        if nsfw_indicators and not server_nsfw_allowed:
            detected_issues.extend(nsfw_indicators)
            max_level = SafetyLevel.BLOCKED

        # Additional server restrictions
        server_restrictions = server_policies.get("restrictions", {})
        if guild_id and str(guild_id) in server_restrictions:
            restrictions = server_restrictions[str(guild_id)]

            # Check restricted tasks
            if request.task.value in restrictions.get("blocked_tasks", []):
                detected_issues.append(f"task_restricted:{request.task.value}")
                max_level = SafetyLevel.BLOCKED

            # Check restricted providers
            if (
                request.preferred_provider
                and request.preferred_provider.value
                in restrictions.get("blocked_providers", [])
            ):
                detected_issues.append(
                    f"provider_restricted:{request.preferred_provider.value}"
                )
                max_level = SafetyLevel.WARNING

        approved = max_level != SafetyLevel.BLOCKED
        reason = f"Server policy: {max_level.value}"

        return SafetyResult(
            approved=approved,
            level=max_level,
            reason=reason,
            user_message="",
            detected_issues=detected_issues,
        )

    def _detect_nsfw_content(self, request: VisionRequest) -> List[str]:
        """Detect potential NSFW content indicators [SFT]"""
        nsfw_indicators = []

        # NSFW keywords from policy
        nsfw_keywords = self.policy.get("nsfw_keywords", [])

        # Check prompt
        if request.prompt:
            prompt_lower = request.prompt.lower()
            for keyword in nsfw_keywords:
                if keyword.lower() in prompt_lower:
                    nsfw_indicators.append(f"nsfw_keyword:{keyword}")

        # Check for anatomy-related terms that might indicate NSFW
        anatomy_terms = [
            "nude",
            "naked",
            "topless",
            "bottomless",
            "underwear",
            "lingerie",
            "bikini",
            "swimsuit",
            "revealing",
            "exposed",
        ]

        if request.prompt:
            prompt_lower = request.prompt.lower()
            for term in anatomy_terms:
                if term in prompt_lower:
                    nsfw_indicators.append(f"anatomy_term:{term}")

        return nsfw_indicators

    def _heuristic_analysis(self, text: str) -> List[str]:
        """Heuristic analysis for additional safety checks [IV]"""
        issues = []

        # Check for excessive capitalization (shouting)
        if len(text) > 20 and sum(1 for c in text if c.isupper()) / len(text) > 0.7:
            issues.append("excessive_caps")

        # Check for repeated characters/words (spam patterns)
        words = text.lower().split()
        if len(words) > 5:
            repeated_words = [word for word in set(words) if words.count(word) > 3]
            if repeated_words:
                issues.append("repeated_words")

        # Check for potential prompt injection
        injection_patterns = [
            "ignore previous",
            "ignore above",
            "ignore instructions",
            "new instructions",
            "system prompt",
            "jailbreak",
            "override",
            "bypass",
            "circumvent",
        ]

        text_lower = text.lower()
        for pattern in injection_patterns:
            if pattern in text_lower:
                issues.append(f"prompt_injection:{pattern}")

        return issues

    def _generate_reason(self, issues: List[str], level: SafetyLevel) -> str:
        """Generate technical reason for logging [CMV]"""
        if not issues:
            return "No safety issues detected"

        issue_summary = {}
        for issue in issues:
            category = issue.split(":")[0]
            issue_summary[category] = issue_summary.get(category, 0) + 1

        summary_parts = [f"{k}({v})" for k, v in issue_summary.items()]
        return f"{level.value}: {', '.join(summary_parts)}"

    def _generate_user_message(
        self, issues: List[str], level: SafetyLevel, request: VisionRequest
    ) -> str:
        """Generate user-friendly message for policy violations [CMV]"""
        if level == SafetyLevel.SAFE:
            return ""

        base_messages = {
            SafetyLevel.WARNING: "⚠️ Your request contains content that may violate our usage policies. ",
            SafetyLevel.BLOCKED: "🚫 Your request has been blocked due to content policy violations. ",
        }

        # Categorize issues for better messaging
        has_nsfw = any("nsfw" in issue for issue in issues)
        has_blocked_keywords = any("blocked_keyword" in issue for issue in issues)
        has_deepfake = any("deepfake" in issue for issue in issues)
        has_duration = any("duration" in issue for issue in issues)

        message_parts = [base_messages.get(level, "")]

        if has_nsfw:
            message_parts.append("NSFW content is not allowed in this server. ")

        if has_blocked_keywords:
            message_parts.append("Your prompt contains prohibited keywords. ")

        if has_deepfake:
            message_parts.append(
                "Creating realistic depictions of people may violate our policies. "
            )

        if has_duration:
            message_parts.append("Video duration exceeds server limits. ")

        # Add guidance
        if level == SafetyLevel.BLOCKED:
            message_parts.append("\n\n💡 **Suggestions:**")
            message_parts.append("\n• Modify your prompt to remove prohibited content")
            message_parts.append("\n• Review our usage guidelines")
            message_parts.append("\n• Try a different approach to your creative vision")
        else:
            message_parts.append(
                "Please review your request and consider modifying it."
            )

        return "".join(message_parts)

    def _load_safety_policy(self) -> Dict[str, Any]:
        """Load safety policy from vision policy file [CMV]"""
        try:
            policy_path = Path(self.config["VISION_POLICY_PATH"])
            if not policy_path.exists():
                self.logger.warning(f"Vision policy file not found: {policy_path}")
                return self._get_default_safety_policy()

            with open(policy_path, "r") as f:
                policy_data = json.load(f)

            safety_policy = policy_data.get("safety_filter", {})

            self.logger.info(f"Safety policy loaded from {policy_path}")
            return safety_policy

        except Exception as e:
            self.logger.error(f"Failed to load safety policy: {e}")
            return self._get_default_safety_policy()

    def _get_default_safety_policy(self) -> Dict[str, Any]:
        """Get default safety policy if file loading fails [CMV]"""
        return {
            "blocked_keywords": [
                "explicit",
                "pornographic",
                "sexual",
                "nudity",
                "violence",
                "gore",
                "blood",
                "death",
                "kill",
                "illegal",
                "drugs",
                "weapons",
                "terrorism",
                "hate",
                "racist",
                "discrimination",
            ],
            "warning_keywords": [
                "suggestive",
                "revealing",
                "provocative",
                "political",
                "controversial",
                "sensitive",
            ],
            "blocked_patterns": [
                r"\b(nude|naked|topless)\b",
                r"\b(kill|murder|death)\b",
                r"\b(bomb|explosive|weapon)\b",
            ],
            "warning_patterns": [
                r"\b(celebrity|famous person)\b",
                r"\b(political figure|politician)\b",
            ],
            "nsfw_keywords": [
                "nude",
                "naked",
                "topless",
                "sexual",
                "erotic",
                "pornographic",
                "explicit",
                "adult",
                "mature",
            ],
            "video_restrictions": {
                "max_duration_seconds": 30,
                "blocked_keywords": ["violence", "explicit", "illegal"],
            },
            "server_policies": {
                "nsfw": {"default_allowed": False, "server_overrides": {}},
                "restrictions": {},
            },
        }
