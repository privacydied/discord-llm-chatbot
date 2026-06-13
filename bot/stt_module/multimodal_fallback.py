"""Multimodal STT Fallback Provider.

Provides fallback transcription using multimodal models when faster-whisper fails.
Uses OpenRouter-compatible models that are different from vision models.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import aiohttp

from bot.config import load_config
from bot.exceptions import InferenceError
from bot.utils.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from .failure_classifier import FailureClassification

logger = get_logger(__name__)


@dataclass
class FallbackTranscriptResult:
    """Result from multimodal fallback transcription."""

    text: str
    provider: str
    confidence: float
    is_fallback: bool = True
    failure_context: str | None = None
    processing_time_ms: float = 0.0
    error: str | None = None
    metadata: dict[str, Any] | None = None


class MultimodalSTTFallbackProvider:
    """Handles multimodal fallback when faster-whisper fails."""

    def __init__(self) -> None:
        self.config = load_config()
        self._session: aiohttp.ClientSession | None = None
        self._timeout = float(self.config.get("STT_MULTIMODAL_FALLBACK_TIMEOUT_S", 30.0))
        self._min_confidence = float(self.config.get("STT_MULTIMODAL_FALLBACK_MIN_CONFIDENCE", 0.5))
        self._max_retries = int(self.config.get("STT_MULTIMODAL_FALLBACK_MAX_RETRIES", 1))
        self._max_audio_duration_s = float(self.config.get("STT_MAX_AUDIO_DURATION_S", 300.0))
        self._models = self._load_fallback_models()

    def _load_fallback_models(self) -> list[dict[str, Any]]:
        """Load fallback model configuration from environment."""
        models_config = self.config.get("STT_MULTIMODAL_FALLBACK_MODELS", "")
        if not models_config:
            # Default models if not configured
            models_config = "openrouter/openai-whisper-large-v3,openrouter/meta-llama-3-8b-instruct:free"

        models = []
        for model_str in models_config.split(","):
            model_str = model_str.strip()
            if not model_str:
                continue

            # Parse model string (format: "model" or "model:provider")
            if ":" in model_str:
                model_name, provider = model_str.split(":", 1)
            else:
                model_name = model_str
                provider = "openrouter"

            models.append(
                {
                    "name": model_name,
                    "provider": provider,
                    "timeout": self._timeout,
                    "max_retries": self._max_retries,
                },
            )

        logger.info(f"Loaded {len(models)} multimodal fallback models: {[m['name'] for m in models]}")
        return models

    async def transcribe_with_fallback(
        self,
        audio_path: Path,
        pre_result: Any | None = None,
        failure_reason: FailureClassification | None = None,
        model_config: dict[str, Any] | None = None,
    ) -> FallbackTranscriptResult:
        """Attempt multimodal transcription when primary STT fails.

        Args:
            audio_path: Path to the audio file
            pre_result: Optional preprocessing result (for audio analysis)
            failure_reason: Why the primary STT failed
            model_config: Optional model configuration override

        Returns:
            FallbackTranscriptResult with the transcription

        """
        start_time = time.time()

        try:
            # Cap audio duration before remote fallback [Phase 12-16]
            self._check_audio_duration(audio_path)

            # Check if fallback is enabled
            if not self.config.get("STT_MULTIMODAL_FALLBACK_ENABLED", False):
                msg = "Multimodal fallback is disabled"
                raise InferenceError(msg)

            # Try each model in sequence
            last_error = None

            for model_info in self._models:
                try:
                    logger.info(f"[MultimodalSTT] Attempting fallback with model: {model_info['name']}")

                    result = await self._try_single_model(
                        audio_path=audio_path,
                        pre_result=pre_result,
                        model_info=model_info,
                        failure_reason=failure_reason,
                    )

                    # Validate result
                    if result and result.text and len(result.text.strip()) > 0:
                        result.processing_time_ms = (time.time() - start_time) * 1000
                        result.provider = model_info["name"]
                        return result

                except Exception as e:
                    last_error = e
                    logger.warning(f"[MultimodalSTT] Model {model_info['name']} failed: {e}")
                    continue

            # All models failed
            msg = f"All multimodal fallback models failed. Last error: {last_error}"
            raise InferenceError(msg)

        except Exception as e:
            # Return error result
            return FallbackTranscriptResult(
                text="",
                provider="multimodal_fallback",
                confidence=0.0,
                error=str(e),
                processing_time_ms=(time.time() - start_time) * 1000,
            )

    async def _try_single_model(
        self,
        audio_path: Path,
        pre_result: Any | None,
        model_info: dict[str, Any],
        failure_reason: FailureClassification | None,
    ) -> FallbackTranscriptResult | None:
        """Try transcription with a single model."""
        for attempt in range(model_info.get("max_retries", self._max_retries) + 1):
            try:
                if attempt > 0:
                    await asyncio.sleep(1.0)  # Delay between retries

                # Select the appropriate transcription strategy based on model
                if "whisper" in model_info["name"].lower():
                    result = await self._transcribe_with_whisper_model(audio_path, model_info)
                else:
                    result = await self._transcribe_with_multimodal_model(audio_path, pre_result, model_info, failure_reason)

                if result and result.confidence >= self._min_confidence:
                    return result

            except Exception as e:
                logger.warning(f"[MultimodalSTT] Model attempt {attempt + 1} failed: {e}")
                if attempt == model_info.get("max_retries", self._max_retries):
                    raise
                continue

        return None

    async def _transcribe_with_whisper_model(self, audio_path: Path, model_info: dict[str, Any]) -> FallbackTranscriptResult:
        """Transcribe using a Whisper model via API."""
        # For Whisper models, we can use the audio file directly
        audio_data = await self._read_audio_bytes(audio_path)

        payload = {
            "model": model_info["name"],
            "file": ("audio.wav", audio_data, "audio/wav"),
            "language": "auto",
            "response_format": "json",
        }

        if model_info["provider"] == "openrouter":
            response = await self._call_openrouter_api(payload, "audio")
            text = self._extract_text_from_response(response)

            return FallbackTranscriptResult(
                text=text,
                provider=model_info["name"],
                confidence=0.8,  # Default confidence for Whisper
                metadata={"response_length": len(text) if text else 0},
            )
        msg = f"Unsupported provider for Whisper model: {model_info['provider']}"
        raise InferenceError(msg)

    async def _transcribe_with_multimodal_model(
        self,
        audio_path: Path,
        pre_result: Any | None,
        model_info: dict[str, Any],
        failure_reason: FailureClassification | None,
    ) -> FallbackTranscriptResult:
        """Transcribe using a multimodal model that can process audio."""
        # Read audio data
        audio_data = await self._read_audio_bytes(audio_path)
        audio_base64 = self._encode_audio_base64(audio_data)

        # Prepare the multimodal prompt
        prompt = self._build_multimodal_prompt(failure_reason)

        if model_info["provider"] == "openrouter":
            payload = {
                "model": model_info["name"],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "audio_url",
                                "audio_url": {"url": f"data:audio/wav;base64,{audio_base64}"},
                            },
                        ],
                    },
                ],
                "temperature": 0.0,  # For more deterministic output
                "max_tokens": 1000,
            }

            response = await self._call_openrouter_api(payload, "chat")
            text = self._extract_text_from_response(response)

            # Calculate confidence based on response quality
            confidence = self._calculate_confidence(text, failure_reason)

            return FallbackTranscriptResult(
                text=text,
                provider=model_info["name"],
                confidence=confidence,
                metadata={"response_length": len(text) if text else 0},
            )
        msg = f"Unsupported provider for multimodal model: {model_info['provider']}"
        raise InferenceError(msg)

    async def _call_openrouter_api(
        self,
        payload: dict[str, Any],
        endpoint_type: str,  # "audio" or "chat"
    ) -> dict[str, Any]:
        """Call OpenRouter API with the given payload."""
        if not self._session:
            timeout = aiohttp.ClientTimeout(total=self._timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)

        api_key = self.config.get("OPENAI_API_KEY")
        if not api_key:
            msg = "OpenRouter API key not configured"
            raise InferenceError(msg)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://discord-llm-chatbot",
            "X-Title": "Discord LLM Chatbot STT Fallback",
        }

        url = "https://openrouter.ai/api/v1/chat/completions" if endpoint_type == "chat" else "https://openrouter.ai/api/v1/audio/transcriptions"

        try:
            if endpoint_type == "chat":
                async with self._session.post(url, json=payload, headers=headers) as response:
                    response.raise_for_status()
                    return await response.json()
            else:
                # For audio transcription, use multipart/form-data
                form_data = aiohttp.FormData()
                for key, value in payload.items():
                    if key == "file":
                        form_data.add_field(key, value[1], content_type=value[2])
                    else:
                        form_data.add_field(key, str(value))

                async with self._session.post(url, data=form_data, headers=headers) as response:
                    response.raise_for_status()
                    return await response.json()

        except aiohttp.ClientError as e:
            msg = f"OpenRouter API error: {e}"
            raise InferenceError(msg)
        except TimeoutError:
            msg = f"OpenRouter API timeout after {self._timeout}s"
            raise InferenceError(msg)

    def _build_multimodal_prompt(self, failure_reason: FailureClassification | None) -> str:
        """Build a prompt for multimodal models that includes context about the failure."""
        base_prompt = "You are a transcription assistant. The user has sent an audio message. Please transcribe the spoken content as accurately as possible. If the audio is unclear, provide your best interpretation."

        if failure_reason:
            context = f"Note: The primary transcription failed due to {failure_reason.category}. "
            if failure_reason.category in ["decode", "corrupted"]:
                context += "The audio may be noisy or corrupted. "
            elif failure_reason.category == "whisper_runtime":
                context += "The system had processing issues. "

            base_prompt += f" {context}"

        return base_prompt

    def _extract_text_from_response(self, response: dict[str, Any]) -> str:
        """Extract text from various API response formats."""
        if "choices" in response and len(response["choices"]) > 0:
            choice = response["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                return choice["message"]["content"].strip()

        # Fallback for audio transcription responses
        if "text" in response:
            return response["text"].strip()

        return ""

    def _calculate_confidence(self, text: str, failure_reason: FailureClassification | None) -> float:
        """Calculate confidence score for the fallback result."""
        base_confidence = 0.6  # Default confidence for fallback

        # Adjust based on text length
        if len(text) < 10:
            base_confidence -= 0.2
        elif len(text) > 100:
            base_confidence += 0.1

        # Adjust based on failure reason
        if failure_reason:
            if failure_reason.category in ["decode", "corrupted"]:
                base_confidence -= 0.1  # Lower confidence for corrupted audio
            elif failure_reason.category == "whisper_runtime":
                base_confidence += 0.05  # Slightly higher for runtime errors

        # Ensure confidence is within bounds
        return max(0.0, min(1.0, base_confidence))

    async def _read_audio_bytes(self, audio_path: Path) -> bytes:
        """Read audio file as bytes."""
        try:
            with open(audio_path, "rb") as f:
                return f.read()
        except Exception as e:
            msg = f"Failed to read audio file: {e}"
            raise InferenceError(msg)

    def _encode_audio_base64(self, audio_bytes: bytes) -> str:
        """Encode audio data as base64."""
        import base64

        return base64.b64encode(audio_bytes).decode("utf-8")

    def _check_audio_duration(self, audio_path: Path) -> None:
        """Enforce max audio duration cap before remote fallback [Phase 12-16].

        Falls back to a size-based heuristic when no duration reader is available.
        """
        import struct

        duration_s: float | None = None
        try:
            # Try WAV header sample rate + data chunk size
            with open(audio_path, "rb") as f:
                header = f.read(44)
                if header[:4] == b"RIFF" and len(header) >= 44:
                    sample_rate = struct.unpack_from("<I", header, 24)[0]
                    total_size = struct.unpack_from("<I", header, 4)[0]
                    # WAV data is typically ~44 bytes header + raw PCM
                    raw_bytes = max(total_size + 8 - 44, 0)  # +8 for RIFF header
                    if sample_rate > 0:
                        # rough estimate assuming 1 channel, 16-bit
                        duration_s = raw_bytes / (sample_rate * 2)
        except Exception:
            pass

        if duration_s is not None and duration_s > self._max_audio_duration_s:
            logger.warning(
                "stt:audio_duration_exceeded duration=%.1fs max=%.1fs",
                duration_s,
                self._max_audio_duration_s,
            )
            msg = f"Audio duration {duration_s:.1f}s exceeds cap of {self._max_audio_duration_s}s"
            raise InferenceError(msg)

    async def close(self) -> None:
        """Clean up resources."""
        if self._session:
            await self._session.close()
            self._session = None


# Global fallback provider instance
multimodal_fallback_provider = MultimodalSTTFallbackProvider()
