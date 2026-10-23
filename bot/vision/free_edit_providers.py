"""Free image-to-image providers: Cloudflare Workers AI and a Hugging Face Space. [CA][REH]

Both run the whole edit inside ``submit`` (like OpenRouterPlugin), so a quota
or outage error raises there and the adapter falls through to the next
provider -- an async job that failed later in ``poll`` could not fall through.

- ``cloudflare``: ``@cf/black-forest-labs/flux-2-klein-4b`` (FLUX.2 klein). Free
  Cloudflare account: 10,000 neurons/day (~80 1024px edits), reset 00:00 UTC; on
  the free plan an empty allocation fails instead of charging. Needs
  CLOUDFLARE_ACCOUNT_ID + CLOUDFLARE_API_TOKEN (Workers AI permission).
- ``hfspace``: a public gradio Space (default FLUX.1-Kontext-dev) on Hugging
  Face's shared ZeroGPU. No key; HF_TOKEN raises the small anonymous GPU quota.
  Queues and outages happen, so it is the fallback, not the primary.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import re
import time
from typing import Any

from bot.utils.logging import get_logger

from .types import VisionError, VisionErrorType, VisionTask
from .unified_adapter import NormalizedRequest, ProviderPlugin, UnifiedJobStatus, UnifiedResult, UnifiedStatus

logger = get_logger(__name__)

CLOUDFLARE_API_BASE = "https://api.cloudflare.com/client/v4"
CLOUDFLARE_DEFAULT_MODEL = "@cf/black-forest-labs/flux-2-klein-4b"
CLOUDFLARE_INPUT_MAX_SIDE = 511  # docs: "All input images must be smaller than 512x512"
CLOUDFLARE_MIN_SIDE, CLOUDFLARE_MAX_SIDE = 256, 1920
OUTPUT_LONG_SIDE = 1024
HF_DEFAULT_SPACE = "black-forest-labs/FLUX.1-Kontext-dev"
HF_DEFAULT_TIMEOUT_S = 120.0
HF_INPUT_MAX_SIDE = 1024
HF_KONTEXT_GUIDANCE = 2.5
HF_KONTEXT_STEPS = 28
_ERROR_SNIPPET_CHARS = 300
_CF_QUOTA_MARKERS = ("4006", "daily free allocation", "neurons")
_CF_FILTER_MARKERS = ("nsfw", "flagged", "safety", "moderat")
_HF_QUOTA_MARKERS = ("quota", "gpu task aborted", "exceeded your")


def _snippet(text: str) -> str:
    return " ".join(str(text or "").split())[:_ERROR_SNIPPET_CHARS]


def _round16(value: float) -> int:
    return max(16, int(round(value / 16.0)) * 16)


def output_dims(width: int, height: int, long_side: int = OUTPUT_LONG_SIDE) -> tuple[int, int]:
    """Keep the source aspect ratio; long side ``long_side``; multiples of 16 within Cloudflare's 256-1920."""
    width, height = max(1, int(width)), max(1, int(height))
    scale = long_side / max(width, height)
    w = min(CLOUDFLARE_MAX_SIDE, max(CLOUDFLARE_MIN_SIDE, _round16(width * scale)))
    h = min(CLOUDFLARE_MAX_SIDE, max(CLOUDFLARE_MIN_SIDE, _round16(height * scale)))
    return w, h


def _shrink_to_png(data: bytes, max_side: int) -> tuple[bytes, int, int]:
    """Decode any Pillow-readable image, fit it in ``max_side``, return (png, orig_w, orig_h)."""
    from PIL import Image  # lazy: heavy import

    with Image.open(io.BytesIO(data)) as src:
        orig_w, orig_h = src.size
        img = src.convert("RGB")
    img.thumbnail((max_side, max_side))
    out = io.BytesIO()
    img.save(out, "PNG")
    return out.getvalue(), orig_w, orig_h


async def prepare_input_image(request: NormalizedRequest, max_side: int) -> tuple[bytes, int, int]:
    """Source image as PNG within ``max_side`` (off the event loop). [PA][IV]"""
    data = request.input_image_data
    if not data:
        raise VisionError(
            message="image edit needs input_image_data",
            error_type=VisionErrorType.VALIDATION_ERROR,
            user_message="I need an image to edit.",
        )
    try:
        return await asyncio.to_thread(_shrink_to_png, data, max_side)
    except Exception as exc:  # noqa: BLE001 - any decode failure is a bad input
        raise VisionError(
            message=f"could not decode input image: {exc}",
            error_type=VisionErrorType.VALIDATION_ERROR,
            user_message="I couldn't read that image. Try a PNG, JPG or WebP.",
        ) from exc


def _data_url(png_or_b64: bytes | str, mime: str = "image/png") -> str:
    b64 = png_or_b64 if isinstance(png_or_b64, str) else base64.b64encode(png_or_b64).decode()
    return f"data:{mime};base64,{b64}"


class _InlineResultPlugin(ProviderPlugin):
    """Shared poll/fetch for providers that finish inside ``submit``."""

    def __init__(self, name: str, config: dict[str, Any], api_key: str) -> None:
        super().__init__(name, config, api_key)
        self._results: dict[str, dict[str, Any]] = {}

    def _store(self, assets: list[str], model: str) -> str:
        job_id = f"{self.name}_{int(time.time() * 1000)}"
        self._results[job_id] = {"assets": assets, "model": model}
        return job_id

    async def poll(self, job_id: str) -> UnifiedJobStatus:
        if job_id not in self._results:
            return UnifiedJobStatus(status=UnifiedStatus.FAILED, progress_percentage=0, phase="Job not found")
        return UnifiedJobStatus(status=UnifiedStatus.COMPLETED, progress_percentage=100, phase="Edit complete", actual_cost=0.0)

    async def fetch_result(self, job_id: str) -> UnifiedResult:
        job = self._results.pop(job_id, {})
        return UnifiedResult(assets=list(job.get("assets") or []), final_cost=0.0, provider_used=self.name, metadata={"model": job.get("model", "unknown")})

    def _unsupported(self, request: NormalizedRequest) -> VisionError:
        return VisionError(
            message=f"{self.name} does not support task {request.task}",
            error_type=VisionErrorType.UNSUPPORTED_TASK,
            user_message=f"{self.name} only edits images.",
        )


class CloudflareWorkersAIPlugin(_InlineResultPlugin):
    """Cloudflare Workers AI FLUX.2 klein image edit (multipart ``input_image_0``)."""

    def __init__(self, name: str, config: dict[str, Any], api_key: str) -> None:
        super().__init__(name, config, api_key)
        self.base_url = str(config.get("base_url") or CLOUDFLARE_API_BASE).rstrip("/")
        self.model_map = {VisionTask.IMAGE_TO_IMAGE: self.model}

    @property
    def model(self) -> str:
        return str(self.config.get("model") or CLOUDFLARE_DEFAULT_MODEL)

    @property
    def account_id(self) -> str:
        return str(self.config.get("account_id") or "").strip()

    def capabilities(self) -> dict[str, Any]:
        return {
            "modes": [VisionTask.IMAGE_TO_IMAGE],
            "max_size": (CLOUDFLARE_MAX_SIDE, CLOUDFLARE_MAX_SIDE),
            "max_steps": 4,
            "supports_negative_prompt": False,
            "supports_batch": False,
            "nsfw_policy": "filtered",
        }

    def _form(self, request: NormalizedRequest, png: bytes, orig_w: int, orig_h: int):
        import aiohttp

        width, height = output_dims(orig_w, orig_h)
        form = aiohttp.FormData()
        form.add_field("prompt", request.prompt)
        form.add_field("input_image_0", png, filename="input.png", content_type="image/png")
        form.add_field("width", str(width))
        form.add_field("height", str(height))
        if request.seed is not None:
            form.add_field("seed", str(int(request.seed)))
        return form

    async def submit(self, request: NormalizedRequest) -> str:
        if request.task != VisionTask.IMAGE_TO_IMAGE:
            raise self._unsupported(request)
        if not self.account_id or not self.api_key:
            raise VisionError(
                message="cloudflare needs CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN",
                error_type=VisionErrorType.AUTHENTICATION_ERROR,
                user_message="Cloudflare image editing is not configured.",
            )
        png, orig_w, orig_h = await prepare_input_image(request, CLOUDFLARE_INPUT_MAX_SIDE)
        await self.startup()
        url = f"{self.base_url}/accounts/{self.account_id}/ai/run/{self.model}"
        try:
            async with self.session.post(url, data=self._form(request, png, orig_w, orig_h), headers={"Authorization": f"Bearer {self.api_key}"}) as response:
                text = await response.text()
                self._raise_for_status(response.status, text)
        except VisionError:
            raise
        except Exception as exc:  # noqa: BLE001 - network/timeout -> retryable connection error
            raise VisionError(
                message=f"cloudflare connection error: {exc!s}",
                error_type=VisionErrorType.CONNECTION_ERROR,
                user_message="Couldn't reach Cloudflare. Please try again.",
            ) from exc
        return self._store([_data_url(self._extract_image(text))], self.model)

    @staticmethod
    def _extract_image(text: str) -> str:
        try:
            data = json.loads(text)
        except ValueError:
            data = {}
        result = data.get("result") if isinstance(data, dict) else None
        image = result.get("image") if isinstance(result, dict) else None
        if not isinstance(image, str) or not image:
            raise VisionError(
                message=f"cloudflare returned no image: {_snippet(text)}",
                error_type=VisionErrorType.PROVIDER_ERROR,
                user_message="Cloudflare returned no image.",
            )
        return image

    def _raise_for_status(self, status: int, text: str) -> None:
        if status == 200:
            return
        lowered = (text or "").lower()
        detail = f"cloudflare {status}: {_snippet(text)}"
        if any(marker in lowered for marker in _CF_QUOTA_MARKERS) or status == 402:
            raise VisionError(message=detail, error_type=VisionErrorType.QUOTA_EXCEEDED, user_message="Today's free Cloudflare image allowance is used up (resets 00:00 UTC).")
        if status in (401, 403):
            raise VisionError(message=detail, error_type=VisionErrorType.AUTHENTICATION_ERROR, user_message="Cloudflare rejected the API token. Check CLOUDFLARE_API_TOKEN (needs Workers AI).")
        if status == 429:
            raise VisionError(message=detail, error_type=VisionErrorType.RATE_LIMITED, user_message="Cloudflare is rate limiting edits. Wait a moment.")
        if any(marker in lowered for marker in _CF_FILTER_MARKERS):
            raise VisionError(message=detail, error_type=VisionErrorType.CONTENT_FILTERED, user_message="Cloudflare's safety filter blocked that edit.")
        if status >= 500:
            raise VisionError(message=detail, error_type=VisionErrorType.SERVER_ERROR, user_message="Cloudflare is having trouble. Please try again.")
        raise VisionError(message=detail, error_type=VisionErrorType.VALIDATION_ERROR, user_message="Cloudflare rejected that edit request.")


def space_base_url(space: str) -> str:
    """``owner/Name.x`` -> ``https://owner-name-x.hf.space`` (Hugging Face's subdomain rule)."""
    slug = re.sub(r"[^a-z0-9]+", "-", space.strip().lower()).strip("-")
    return f"https://{slug}.hf.space"


class HuggingFaceSpacePlugin(_InlineResultPlugin):
    """FLUX.1 Kontext edit through a public gradio Space's REST API (upload -> call -> SSE)."""

    def __init__(self, name: str, config: dict[str, Any], api_key: str) -> None:
        super().__init__(name, config, api_key)
        self.model_map = {VisionTask.IMAGE_TO_IMAGE: self.space}

    @property
    def space(self) -> str:
        return str(self.config.get("space") or HF_DEFAULT_SPACE)

    @property
    def base_url(self) -> str:
        return str(self.config.get("base_url") or space_base_url(self.space)).rstrip("/")

    @base_url.setter
    def base_url(self, value: str) -> None:  # the adapter's hot-reload assigns base_url
        if value:
            self.config["base_url"] = value

    def session_timeout_s(self) -> float:
        """Shared GPU queues: allow the whole run, not VISION_PROVIDER_TIMEOUT_MS. [REH]"""
        try:
            return max(30.0, float(self.config.get("timeout_s") or HF_DEFAULT_TIMEOUT_S))
        except (TypeError, ValueError):
            return HF_DEFAULT_TIMEOUT_S

    def capabilities(self) -> dict[str, Any]:
        return {
            "modes": [VisionTask.IMAGE_TO_IMAGE],
            "max_size": (2048, 2048),
            "max_steps": HF_KONTEXT_STEPS,
            "supports_negative_prompt": False,
            "supports_batch": False,
            "nsfw_policy": "provider",
        }

    def _headers(self) -> dict[str, str]:
        token = str(self.api_key or "").strip()
        return {"Authorization": f"Bearer {token}"} if token.startswith("hf_") else {}

    async def submit(self, request: NormalizedRequest) -> str:
        if request.task != VisionTask.IMAGE_TO_IMAGE:
            raise self._unsupported(request)
        png, _, _ = await prepare_input_image(request, HF_INPUT_MAX_SIDE)
        await self.startup()
        try:
            path = await self._upload(png)
            event_id = await self._call(path, request)
            result_url = await self._await_result(event_id)
            mime, data = await self._download(result_url)
        except VisionError:
            raise
        except Exception as exc:  # noqa: BLE001 - network/timeout/bad JSON
            raise VisionError(
                message=f"hfspace {self.space} error: {exc!s}",
                error_type=VisionErrorType.CONNECTION_ERROR,
                user_message="The free Hugging Face editor didn't answer. Please try again.",
            ) from exc
        return self._store([_data_url(data, mime)], self.space)

    async def _upload(self, png: bytes) -> str:
        import aiohttp

        form = aiohttp.FormData()
        form.add_field("files", png, filename="input.png", content_type="image/png")
        async with self.session.post(f"{self.base_url}/gradio_api/upload", data=form, headers=self._headers()) as response:
            text = await response.text()
            self._check_http(response.status, text, "upload")
        return json.loads(text)[0]

    async def _call(self, path: str, request: NormalizedRequest) -> str:
        image = {"path": path, "meta": {"_type": "gradio.FileData"}}
        seed = int(request.seed) if request.seed is not None else 0
        payload = {"data": [image, request.prompt, seed, request.seed is None, HF_KONTEXT_GUIDANCE, HF_KONTEXT_STEPS]}
        async with self.session.post(f"{self.base_url}/gradio_api/call/infer", json=payload, headers=self._headers()) as response:
            text = await response.text()
            self._check_http(response.status, text, "call")
        return json.loads(text)["event_id"]

    async def _await_result(self, event_id: str) -> str:
        event = None
        async with self.session.get(f"{self.base_url}/gradio_api/call/infer/{event_id}", headers=self._headers()) as response:
            self._check_http(response.status, "", "result")
            async for raw in response.content:
                line = raw.decode("utf-8", "replace").strip()
                if line.startswith("event:"):
                    event = line[len("event:") :].strip()
                elif line.startswith("data:") and event == "complete":
                    return self._result_url(line[len("data:") :])
                elif line.startswith("data:") and event == "error":
                    self._raise_space_error(line[len("data:") :])
        raise VisionError(message=f"hfspace {self.space} stream ended without a result", error_type=VisionErrorType.PROVIDER_ERROR, user_message="The free Hugging Face editor returned nothing.")

    def _result_url(self, payload: str) -> str:
        try:
            first = json.loads(payload)[0]
            url = first.get("url") if isinstance(first, dict) else None
        except (ValueError, IndexError, TypeError):
            url = None
        if not url or not str(url).startswith("https://"):
            raise VisionError(message=f"hfspace {self.space} bad result: {_snippet(payload)}", error_type=VisionErrorType.PROVIDER_ERROR, user_message="The free Hugging Face editor returned nothing.")
        return str(url)

    async def _download(self, url: str) -> tuple[str, bytes]:
        async with self.session.get(url, headers=self._headers()) as response:
            self._check_http(response.status, "", "download")
            mime = (response.headers.get("Content-Type") or "image/webp").split(";")[0].strip()
            return (mime if mime.startswith("image/") else "image/webp"), await response.read()

    def _raise_space_error(self, payload: str) -> None:
        lowered = payload.lower()
        detail = f"hfspace {self.space} error: {_snippet(payload)}"
        if any(marker in lowered for marker in _HF_QUOTA_MARKERS):
            raise VisionError(message=detail, error_type=VisionErrorType.QUOTA_EXCEEDED, user_message="The free Hugging Face GPU quota is used up for now.")
        raise VisionError(message=detail, error_type=VisionErrorType.PROVIDER_ERROR, user_message="The free Hugging Face editor failed on that one.")

    def _check_http(self, status: int, text: str, stage: str) -> None:
        if status == 200:
            return
        detail = f"hfspace {self.space} {stage} {status}: {_snippet(text)}"
        if status == 429:
            raise VisionError(message=detail, error_type=VisionErrorType.RATE_LIMITED, user_message="The free Hugging Face editor is busy. Wait a moment.")
        if status in (401, 403):
            raise VisionError(message=detail, error_type=VisionErrorType.AUTHENTICATION_ERROR, user_message="Hugging Face rejected the request. Check HF_TOKEN.")
        error_type = VisionErrorType.SERVER_ERROR if status >= 500 else VisionErrorType.PROVIDER_ERROR
        raise VisionError(message=detail, error_type=error_type, user_message="The free Hugging Face editor is unavailable right now.")
